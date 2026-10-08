"""
DisTorch Safetensor Memory Management Module
Contains all safetensor related code for distributed memory management
"""

import contextlib
import dataclasses
import functools
import itertools
import math
import torch
import torch.utils._pytree as pytree
import logging
import re
from collections import defaultdict, namedtuple
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes

logger = logging.getLogger("MultiGPU")
import comfy.model_management as mm
import comfy.model_patcher
import comfy.ops
from .device_utils import get_device_list
from .model_management_mgpu import multigpu_memory_log

_HIP_SOFTWARE_GEMM_ARCHITECTURES = frozenset(
    {
        "gfx900",
        "gfx906",
        "gfx90c",
        "gfx1010",
        "gfx1011",
        "gfx1012",
        "gfx1030",
        "gfx1031",
        "gfx1032",
        "gfx1033",
        "gfx1034",
        "gfx1035",
        "gfx1036",
    }
)
# Headroom left free on the compute GPU for GEMM workspaces PyTorch cannot see.
_COMPUTE_GPU_RESERVE_BYTES = 512 * 1024 * 1024
# Attention shrinks its chunks when it runs out of memory, so it needs less headroom.
_ATTENTION_RESERVE_BYTES = 128 * 1024 * 1024
# Chunks per mixed GEMM or attention call: with several, each chunk's copies to
# and from the compute GPU overlap the neighbouring chunks' compute.
_PIPELINE_CHUNKS = 8
# Comfy Kitchen's int8 GEMM computes whole 128-row tiles, so a chunk of 520 rows
# costs as much as 640. Chunks are cut to a multiple of this.
_GEMM_ROW_TILE = 128
# Conv tiles stop getting faster well below this, and larger ones let the
# backend's im2col workspace exhaust the compute GPU. A fixed cap also keeps
# tile shapes stable so MIOpen reuses its tuned kernels between runs.
_CONV_TILE_BUDGET_BYTES = 128 * 1024 * 1024


def unpack_load_item(item):
    """Handle ComfyUI 0.6.0+ 5-tuple vs legacy 4-tuple"""
    if len(item) == 5:
        # (module_offload_mem, module_mem, module_name, module_object, params)
        return item[1], item[2], item[3], item[4]
    # (module_mem, module_name, module_object, params)
    return item[0], item[1], item[2], item[3]


def _move_tensors(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, tuple):
        return tuple(_move_tensors(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tensors(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_tensors(item, device) for key, item in value.items()}
    return value


def _current_device(device):
    """Make a CUDA device current, as Kitchen HIP kernels require; a no-op for others."""
    device = torch.device(device)
    return torch.cuda.device(device.index if device.type == "cuda" else -1)


# Ops that only change tensor metadata or allocate, so they never launch a kernel.
_METADATA_OPS = frozenset(
    {
        "aten::empty",
        "aten::empty_like",
        "aten::empty_strided",
        "aten::new_empty",
        "aten::new_empty_strided",
        "aten::_unsafe_view",
        "aten::resize_",
        "aten::set_",
        "aten::squeeze_",
        "aten::unsqueeze_",
        "aten::transpose_",
        "aten::t_",
        "aten::as_strided_",
        "aten::lift_fresh",
    }
)
# In-place ops that overwrite their target without reading it.
_OVERWRITE_OPS = frozenset(
    {"aten::zero_", "aten::fill_", "aten::random_", "aten::normal_", "aten::uniform_", "aten::bernoulli_"}
)

# Ops that copy elementwise without the pointwise tag.
_COPY_OPS = frozenset({"aten::_to_copy", "aten::copy_", "aten::clone"})
# An op on at most this many bytes of donor tensors hops whole without checking free VRAM.
_STREAM_DIRECT_BYTES = 32 * 1024 * 1024
# A tiled op moves slices of about this many bytes, enough to keep the copies busy.
_STREAM_TILE_BYTES = 128 * 1024 * 1024


def _int_list(value):
    return [value] if isinstance(value, int) else list(value or ())


def _tile_axes(func, args, kwargs, leaves):
    """Axes an op can be split along, outermost first, as (extent, per-leaf dim or None, output dim).

    Splitting an axis must give the same result as slicing every leaf along its
    dim, running the op on each slice and joining the results along the output
    dim. Returns an empty list for ops without a rule.
    """
    schema = func._schema
    name = schema.name
    ndim = leaves[0].ndim
    shape = leaves[0].shape
    rest = [None] * (len(leaves) - 1)

    if name in ("aten::_softmax", "aten::_log_softmax"):
        return [(shape[d], [d], d) for d in range(ndim) if d != args[1] % ndim and shape[d] > 1]
    if name == "aten::native_layer_norm":
        return [(shape[d], [d] + rest, d) for d in range(ndim - len(args[1])) if shape[d] > 1]
    if name.startswith(("aten::upsample_", "aten::_upsample_")):
        return [(shape[d], [d], d) for d in range(2) if shape[d] > 1]
    if name == "aten::constant_pad_nd":
        return [(shape[d], [d], d) for d in range(ndim - len(args[1]) // 2) if shape[d] > 1]
    if name == "aten::addmm":
        bias_dim = 0 if leaves[0].ndim == 2 and leaves[0].shape[0] == leaves[1].shape[0] else None
        return [(leaves[1].shape[0], [bias_dim, 0, None], 0)]
    if name == "aten::mm":
        return [(shape[0], [0, None], 0)]
    if name == "aten::bmm":
        return [(shape[0], [0, 0], 0), (shape[1], [1, None], 1)]
    if name in ("aten::cat", "aten::stack"):
        stack = name == "aten::stack"
        dim = kwargs.get("dim", args[1] if len(args) > 1 else 0) % (ndim + stack)
        return [
            (shape[d], [d] * len(leaves), d + (stack and d >= dim))
            for d in range(ndim)
            if shape[d] > 1 and (stack or d != dim)
        ]
    if name == "comfy_kitchen::rms_rope_split_half_":
        q, k, freqs_cis = leaves[:3]
        return [
            (shape[d], [d, d, d if freqs_cis.shape[d] == shape[d] else None] + [None] * (len(leaves) - 3), d)
            for d in range(3)
            if shape[d] > 1 and k.shape[d] == shape[d]
        ]
    if torch.Tag.reduction in func.tags:
        values = {
            argument.name: args[i] if i < len(args) else kwargs.get(argument.name, argument.default_value)
            for i, argument in enumerate(schema.arguments)
            if argument.name in ("dim", "keepdim")
        }
        reduced = {d % ndim for d in _int_list(values.get("dim"))}
        keepdim = values.get("keepdim", False)
        return [
            (shape[d], [d] + rest, d if keepdim else d - sum(r < d for r in reduced))
            for d in range(ndim)
            if reduced and d not in reduced and shape[d] > 1
        ]
    if torch.Tag.pointwise in func.tags or name in _COPY_OPS or name in _OVERWRITE_OPS:
        broadcast = torch.broadcast_shapes(*[leaf.shape for leaf in leaves])
        axes = []
        for d, extent in enumerate(broadcast):
            if extent > 1:
                dims = [d - len(broadcast) + leaf.ndim for leaf in leaves]
                axes.append((extent, [i if i >= 0 and leaf.shape[i] == extent else None for i, leaf in zip(dims, leaves)], d))
        return axes
    return []


class _StreamedActivations(TorchDispatchMode):
    """Keep activations on the donor GPU but run every op on the compute GPU.

    An op with a donor tensor among its inputs copies them to the compute GPU,
    runs there and copies its results back, so the donor only stores tensors
    and does no math. Views, allocations and plain copies launch no kernel and
    run in place. Code that is already on the compute GPU is unaffected.
    """

    def __init__(self, donor_device, compute_device):
        super().__init__()
        self.donor = torch.device(donor_device)
        self.compute = torch.device(compute_device)
        self.kinds = {}

    def _kind(self, func):
        kind = self.kinds.get(func)
        if kind is None:
            schema = func._schema
            if schema.name in _METADATA_OPS:
                kind = "metadata"
            elif schema.is_mutable:
                kind = "overwrite" if schema.name in _OVERWRITE_OPS else "mutating"
            elif not any("Tensor" in str(r.type) for r in schema.returns) or any(
                r.alias_info is not None for r in schema.returns
            ):
                kind = "metadata"
            else:
                kind = "plain"
            self.kinds[func] = kind
        return kind

    def _to_compute(self, tensor, dtype=None):
        if tensor.device != self.compute:
            tensor = tensor.to(device=self.compute, non_blocking=True)
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        return tensor

    def _resolve(self, device):
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        return device

    def _tiled(self, func, kind, args, kwargs, budget):
        """Run an op on the compute GPU one slice at a time so that only a few slices are ever there.

        The slices go through _pipeline_on_compute, and the results are joined on the donor.
        """
        name = func._schema.name
        flat, spec = pytree.tree_flatten((args, kwargs))
        positions = [i for i, value in enumerate(flat) if isinstance(value, torch.Tensor)]
        leaves = [flat[i] for i in positions]
        mutating = kind != "plain"
        axes = [axis for axis in _tile_axes(func, args, kwargs, leaves) if not mutating or axis[1][0] is not None]
        if not axes:
            raise torch.OutOfMemoryError(
                f"{name} on {sum(leaf.nbytes for leaf in leaves) >> 20} MiB of activations does not fit on "
                f"{self.compute} and has no tiling rule"
            )

        def rows_that_fit(axis):
            extent, dims, _ = axis
            sliced = max(leaf.nbytes for leaf, dim in zip(leaves, dims) if dim is not None)
            whole = sum(leaf.nbytes for leaf, dim in zip(leaves, dims) if dim is None)
            # Slices of the inputs and results, three of which are on the compute GPU at once.
            fit = (budget - whole) * extent // (3 * 2 * sliced)
            return min(fit, _STREAM_TILE_BYTES * extent // sliced)

        fitting = [axis for axis in axes if rows_that_fit(axis) >= 1]
        extent, dims, out_dim = fitting[0] if fitting else max(axes)
        rows = max(1, min(rows_that_fit((extent, dims, out_dim)), extent))
        whole = {
            index: leaf.to(device=self.compute, non_blocking=True) if leaf.device == self.donor else leaf
            for index, (leaf, dim) in enumerate(zip(leaves, dims))
            if dim is None
        }
        written = sum(
            argument.alias_info is not None and argument.alias_info.is_write and "Tensor" in str(argument.type)
            for argument in func._schema.arguments
        )
        outputs = leaves[:written] if mutating else []
        progress = [0]

        def load(chunk):
            start, stop = chunk
            tiles = []
            for index, (leaf, dim) in enumerate(zip(leaves, dims)):
                if dim is None:
                    tiles.append(whole[index])
                    continue
                tile = leaf.narrow(dim, start, stop - start)
                if tile.device == self.donor:
                    if kind == "overwrite" and index == 0:
                        tile = torch.empty_like(tile, device=self.compute)
                    else:
                        tile = tile.to(device=self.compute, non_blocking=True)
                tiles.append(tile)
            return tuple(tiles)

        def run(tiles):
            values = list(flat)
            for position, tile in zip(positions, tiles):
                values[position] = tile
            tile_args, tile_kwargs = pytree.tree_unflatten(values, spec)
            result = func(*tile_args, **tile_kwargs)
            return tiles[:written] if result is None else result

        def store(chunk, result):
            start, stop = chunk
            results = result if isinstance(result, tuple) else (result,)
            if not outputs:
                for tile in results:
                    shape = list(tile.shape)
                    shape[out_dim] = extent
                    outputs.append(torch.empty(shape, dtype=tile.dtype, device=self.donor))
            for index, (output, tile) in enumerate(zip(outputs, results)):
                output.narrow(dims[index] if mutating else out_dim, start, stop - start).copy_(tile, non_blocking=True)
            progress[0] = stop

        while progress[0] < extent:
            chunks = [(start, min(start + rows, extent)) for start in range(progress[0], extent, rows)]
            try:
                _pipeline_on_compute(chunks, self.compute, load, run, store)
            except torch.OutOfMemoryError:
                if rows == 1:
                    raise
                rows //= 2
                if self.compute.type == "cuda":
                    torch.cuda.synchronize(self.compute)
                torch.cuda.empty_cache()
        if self.compute.type == "cuda":
            torch.cuda.synchronize(self.compute)
        if mutating:
            return args[0] if func._schema.returns else None
        return tuple(outputs) if len(outputs) > 1 else outputs[0]

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        kind = self._kind(func)
        if kind == "metadata":
            # Under a dispatch mode aten::alias keeps the input's subclass, which breaks
            # torch.Tensor(subclass) for subclasses whose __init__ takes required kwargs (GGMLTensor).
            if func is torch.ops.aten.alias.default:
                return func(*args, **kwargs).as_subclass(torch.Tensor)
            return func(*args, **kwargs)
        target = kwargs.get("device")
        target = None if target is None else self._resolve(target)
        tensors = [leaf for leaf in pytree.tree_leaves((args, kwargs)) if isinstance(leaf, torch.Tensor)]
        if target != self.donor and not any(tensor.device == self.donor for tensor in tensors):
            return func(*args, **kwargs)

        name = func._schema.name
        donor_bytes = sum(tensor.nbytes for tensor in tensors if tensor.device == self.donor)
        if (
            donor_bytes > _STREAM_DIRECT_BYTES
            and target in (None, self.donor)
            and (name != "aten::copy_" or all(tensor.device == self.donor for tensor in args[:2]))
            and (name != "aten::_to_copy" or args[0].device == self.donor)
        ):
            budget = mm.get_free_memory(self.compute) - _COMPUTE_GPU_RESERVE_BYTES
            if 3 * donor_bytes > budget:
                return self._tiled(func, kind, args, kwargs, budget)

        if name == "aten::copy_":
            dst, src = args[0], args[1]
            if dst.device == self.donor or src.device == self.donor:
                if src.device == self.donor or src.dtype != dst.dtype:
                    src = self._to_compute(src, dst.dtype)
                return func(dst, src, *args[2:], **kwargs)
        elif name == "aten::_to_copy":
            src = args[0]
            dtype = kwargs.get("dtype")
            if (dtype is None or dtype == src.dtype) and target is not None and target != src.device:
                return func(*args, **kwargs)
            result = self._to_compute(src, dtype)
            destination = src.device if target is None else target
            return result if destination == self.compute else result.to(device=destination, non_blocking=True)

        moved = {}

        def to_compute(tensor):
            if tensor.device != self.donor:
                return tensor
            if id(tensor) not in moved:
                if kind == "overwrite" and tensor is args[0]:
                    moved[id(tensor)] = torch.empty_like(tensor, device=self.compute)
                else:
                    moved[id(tensor)] = self._to_compute(tensor)
            return moved[id(tensor)]

        compute_args, compute_kwargs = pytree.tree_map_only(torch.Tensor, to_compute, (args, kwargs))
        if target == self.donor:
            compute_kwargs["device"] = self.compute
        result = func(*compute_args, **compute_kwargs)

        originals = {}
        if kind != "plain":
            for index, argument in enumerate(func._schema.arguments):
                if argument.alias_info is None or not argument.alias_info.is_write:
                    continue
                value = args[index] if index < len(args) else kwargs.get(argument.name)
                for tensor in pytree.tree_leaves(value):
                    if isinstance(tensor, torch.Tensor) and tensor.device == self.donor:
                        tensor.copy_(moved[id(tensor)], non_blocking=True)
                        originals[id(moved[id(tensor)])] = tensor
        if target is not None and target != self.donor:
            return result

        def to_donor(tensor):
            if id(tensor) in originals:
                return originals[id(tensor)]
            return tensor.to(device=self.donor, non_blocking=True) if tensor.device == self.compute else tensor

        return pytree.tree_map_only(torch.Tensor, to_donor, result)


def _is_hip_software_gemm_device(device):
    device = torch.device(device)
    if (
        not getattr(torch.version, "hip", None)
        or device.type != "cuda"
        or device.index is None
    ):
        return False

    try:
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
    except (AttributeError, RuntimeError):
        return False
    return arch in _HIP_SOFTWARE_GEMM_ARCHITECTURES


def _is_comfy_kitchen_attention_enabled():
    enabled = getattr(mm, "comfy_kitchen_attention_enabled", None)
    return callable(enabled) and enabled()


def _is_eligible_donor_gemm(module):
    weight = getattr(module, "weight", None)
    return isinstance(weight, torch.Tensor) and weight.ndim == 2


def _can_tile_linear_on_compute(module):
    weight = module.weight
    # ComfyUI's mixed-precision Linear (int8, fp8, ...) is not a torch.nn.Linear.
    is_linear = isinstance(module, torch.nn.Linear) or hasattr(module, "quant_format")
    if not is_linear or weight.ndim != 2:
        return False
    if callable(getattr(module, "cast_bias_weight", None)):
        return True
    return not hasattr(weight, "tensor_type") and weight.shape == (
        module.out_features,
        module.in_features,
    )


def _can_tile_conv_on_compute(module):
    return (
        isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d))
        and module.groups == 1
        and not isinstance(module.padding, str)
    )


def _donor_gemm_skip_reasons(compute_device, donor_device):
    reasons = []
    if donor_device == compute_device:
        reasons.append("donor and compute devices are the same")
    if not _is_hip_software_gemm_device(compute_device):
        reasons.append(f"compute device {compute_device} lacks HIP software GEMM")
    if not _is_hip_software_gemm_device(donor_device):
        reasons.append(f"donor device {donor_device} lacks HIP software GEMM")
    return reasons


def _compute_budget(compute_device, reserve=_COMPUTE_GPU_RESERVE_BYTES):
    """Bytes of compute-GPU VRAM one GEMM or attention call may use for tiles."""
    # Release cached blocks first: the cudaMallocAsync pool (--cuda-malloc) keeps
    # freed tiles and cannot serve a larger request from them, so they are only
    # free once returned to the driver. Blocks still read by pending copies to
    # or from the donor are only released once every device has synchronized.
    if torch.device(compute_device).type == "cuda":
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)
    torch.cuda.empty_cache()
    return max(0, mm.get_free_memory(compute_device) - reserve)


_PIPELINE_STREAMS = {}


def _without_streaming(function):
    """Run function with _StreamedActivations paused: its tensors are already where it computes."""

    @functools.wraps(function)
    def run(*args, **kwargs):
        with _disable_current_modes():
            return function(*args, **kwargs)

    return run


@_without_streaming
def _pipeline_on_compute(chunks, compute_device, load, run, store, during=None):
    """Run chunks on the compute GPU with their copies overlapping the compute.

    load(chunk) copies a chunk's inputs to the compute GPU, run(inputs) computes
    its result there and store(chunk, result) copies the result back. Copies go
    on a side stream while the compute stream works on the neighbouring chunks.
    At most two chunks are in flight, so the compute GPU holds a few chunk
    buffers rather than the whole activation. during() runs once the first
    chunk's compute is queued, so work it starts overlaps the rest.
    """
    compute_device = torch.device(compute_device)
    if compute_device.type != "cuda":
        for index, chunk in enumerate(chunks):
            store(chunk, run(load(chunk)))
            if index == 0 and during is not None:
                during()
        return
    compute_stream = torch.cuda.current_stream(compute_device)
    # One stream per device: the allocator cannot reuse a stream's cached blocks for another stream.
    if compute_device not in _PIPELINE_STREAMS:
        _PIPELINE_STREAMS[compute_device] = torch.cuda.Stream(compute_device)
    copy_stream = _PIPELINE_STREAMS[compute_device]

    def load_on_copy_stream(chunk):
        with torch.cuda.stream(copy_stream):
            inputs = load(chunk)
        return inputs, copy_stream.record_event()

    in_flight = []
    pending = load_on_copy_stream(chunks[0])
    for index, chunk in enumerate(chunks):
        inputs, loaded = pending
        if index + 1 < len(chunks):
            pending = load_on_copy_stream(chunks[index + 1])
        compute_stream.wait_event(loaded)
        for tensor in inputs if isinstance(inputs, tuple) else (inputs,):
            tensor.record_stream(compute_stream)
        result = run(inputs)
        if index == 0 and during is not None:
            during()
        copy_stream.wait_event(compute_stream.record_event())
        with torch.cuda.stream(copy_stream):
            store(chunk, result)
        for tensor in result if isinstance(result, tuple) else (result,):
            tensor.record_stream(copy_stream)
        in_flight.append(copy_stream.record_event())
        if len(in_flight) > 2:
            in_flight.pop(0).synchronize()


def _token_chunks(tokens, token_bytes, budget):
    """Token slices for _pipeline_on_compute: at least _PIPELINE_CHUNKS, three of which fit the budget.

    Three chunks sit on the compute GPU at once: one loading, one computing and
    one being stored.
    """
    per_chunk = max(1, min(-(-tokens // _PIPELINE_CHUNKS), budget // (3 * token_bytes)))
    if per_chunk > _GEMM_ROW_TILE:
        per_chunk -= per_chunk % _GEMM_ROW_TILE
    return [slice(start, min(start + per_chunk, tokens)) for start in range(0, tokens, per_chunk)]


def _linear_on_compute(input_chunk, weight, bias, lora_down, lora_up):
    output_chunk = comfy.ops.dense_linear(input_chunk, weight, bias)
    if lora_down is not None:
        output_chunk.addmm_(input_chunk @ lora_down.t(), lora_up.t())
    return output_chunk


def _run_linear_chunks(
    flat_input, output, weight, bias, lora_down, lora_up, chunks, compute_device, during=None
):
    """Stream token chunks through a linear whose weight is already on the compute GPU."""
    with _current_device(compute_device):
        _pipeline_on_compute(
            chunks,
            compute_device,
            lambda chunk: flat_input[chunk].to(device=compute_device, non_blocking=True),
            lambda input_chunk: _linear_on_compute(input_chunk, weight, bias, lora_down, lora_up),
            lambda chunk, result: output[chunk].copy_(result, non_blocking=True),
            during,
        )


def _lora_on_compute(lora, compute_device, dtype):
    """Stack (down, up, scale) LoRA patches into one low-rank pair on the compute GPU."""
    if not lora:
        return None, None
    down = torch.cat([down.to(device=compute_device, dtype=dtype) for down, _, _ in lora])
    up = torch.cat(
        [up.to(device=compute_device, dtype=dtype) * scale for _, up, scale in lora], dim=1
    )
    return down, up


def _run_tiled_linear_on_compute(input_tensor, weight, bias, compute_device, lora=()):
    """Run a linear on the compute GPU in tiles sized to its free VRAM.

    The input and output stay on the input's device; only token chunks, weight
    tiles and their GEMM results visit the compute GPU. lora holds (down, up,
    scale) patches that are added as low-rank GEMMs instead of being merged
    into the weight.
    """
    flat_input = input_tensor.reshape(-1, input_tensor.shape[-1])
    tokens, in_features = flat_input.shape
    out_features = weight.shape[0]
    output = torch.empty(
        (tokens, out_features), device=input_tensor.device, dtype=input_tensor.dtype
    )
    lora_down, lora_up = _lora_on_compute(lora, compute_device, output.dtype)
    lora_rank = 0 if lora_down is None else lora_down.shape[0]
    row_bytes = in_features * weight.element_size()
    budget = _compute_budget(compute_device)

    if isinstance(bias, torch.Tensor):
        bias = bias.to(device=compute_device, non_blocking=True)

    # Keep the whole weight on the compute GPU when it takes at most half the
    # budget; otherwise stream output-channel tiles for every token chunk.
    rows_per_tile = min(out_features, max(1, budget // 2 // row_bytes))
    if rows_per_tile == out_features:
        token_bytes = (in_features + out_features + lora_rank) * output.element_size()
        chunks = _token_chunks(tokens, token_bytes, budget - out_features * row_bytes)
        _run_linear_chunks(
            flat_input,
            output,
            weight.to(device=compute_device, non_blocking=True),
            bias,
            lora_down,
            lora_up,
            chunks,
            compute_device,
        )
        return output.reshape(*input_tensor.shape[:-1], out_features)

    tokens_per_chunk = min(
        tokens,
        max(
            1,
            (budget - rows_per_tile * row_bytes)
            // ((in_features + rows_per_tile + lora_rank) * output.element_size()),
        ),
    )
    # Allocate the tile buffers once and reuse them: a cudaMallocAsync pool
    # (--cuda-malloc) cannot serve a differently sized chunk from freed ones.
    weight_buffer = torch.empty(
        (rows_per_tile, in_features), device=compute_device, dtype=weight.dtype
    )
    input_buffer = torch.empty(
        (tokens_per_chunk, in_features), device=compute_device, dtype=output.dtype
    )
    output_buffer = torch.empty(
        tokens_per_chunk * rows_per_tile, device=compute_device, dtype=output.dtype
    )
    lora_buffer = torch.empty(
        (tokens_per_chunk, lora_rank), device=compute_device, dtype=output.dtype
    )
    with torch.cuda.device_of(input_buffer):
        for token_start in range(0, tokens, tokens_per_chunk):
            token_end = min(token_start + tokens_per_chunk, tokens)
            input_chunk = input_buffer[: token_end - token_start]
            input_chunk.copy_(flat_input[token_start:token_end], non_blocking=True)
            if lora_down is not None:
                lora_chunk = lora_buffer[: input_chunk.shape[0]]
                torch.mm(input_chunk, lora_down.t(), out=lora_chunk)
            for start in range(0, out_features, rows_per_tile):
                end = min(start + rows_per_tile, out_features)
                weight_tile = weight_buffer[: end - start]
                weight_tile.copy_(weight[start:end], non_blocking=True)
                output_tile = output_buffer[: input_chunk.shape[0] * (end - start)].view(
                    input_chunk.shape[0], end - start
                )
                if bias is None:
                    torch.mm(input_chunk, weight_tile.t(), out=output_tile)
                else:
                    torch.addmm(bias[start:end], input_chunk, weight_tile.t(), out=output_tile)
                if lora_down is not None:
                    output_tile.addmm_(lora_chunk, lora_up[start:end].t())
                output[token_start:token_end, start:end].copy_(output_tile, non_blocking=True)

    return output.reshape(*input_tensor.shape[:-1], out_features)


def _run_tiled_conv_on_compute(conv, input_tensor, weight, bias, padding, compute_device):
    """Run a zero-padded conv2d or conv3d on the compute GPU in tiles.

    The input and output stay on the input's device. Every spatial dim but the
    width is tiled; each tile copies only the input it reads, including the
    kernel halo, and padding is applied on the compute GPU. Tiles run on
    Kitchen's fp16 implicit-GEMM conv where ComfyUI would use it.
    """
    stride, dilation = conv.stride, conv.dilation
    batch, channels, *size = input_tensor.shape
    out_channels, _, *kernel = weight.shape
    dims = len(size)
    span = [d * (k - 1) + 1 for d, k in zip(dilation, kernel)]
    out_size = [
        (length + 2 * pad - extent) // step + 1
        for length, pad, extent, step in zip(size, padding, span, stride)
    ]
    output = torch.empty(
        (batch, out_channels, *out_size),
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )
    element_size = output.element_size()
    channel_bytes = weight[0].numel() * weight.element_size()
    budget = _CONV_TILE_BUDGET_BYTES
    # Only pay for synchronizing every device when VRAM is actually short.
    if mm.get_free_memory(compute_device) - _COMPUTE_GPU_RESERVE_BYTES < budget:
        budget = min(budget, _compute_budget(compute_device))

    channels_per_tile = min(out_channels, max(1, budget // 2 // channel_bytes))
    weight_buffer = torch.empty(
        (channels_per_tile, *weight.shape[1:]), device=compute_device, dtype=weight.dtype
    )
    packed = comfy.ops._packed_conv_wanted(conv, input_tensor, weight_buffer)
    # MIOpen may lower the conv to a GEMM over an im2col workspace; Kitchen's conv has none.
    workspace = 0 if packed else weight[0].numel()

    def input_extent(chunk, dim):
        return (chunk[dim] - 1) * stride[dim] + span[dim]

    def tile_bytes(chunk):
        input_elements = channels
        for dim in range(dims):
            input_elements *= input_extent(chunk, dim)
        output_elements = 1
        for length in chunk:
            output_elements *= length
        return (input_elements + (channels_per_tile + workspace) * output_elements) * element_size

    # Halve the largest tiled dim until a tile fits; halving keeps the number
    # of distinct tile shapes, and so backend kernel searches, small.
    chunk = list(out_size)
    tile_budget = budget - channels_per_tile * channel_bytes
    while tile_bytes(chunk) > tile_budget and max(chunk[:-1]) > 1:
        dim = max(range(dims - 1), key=lambda d: chunk[d])
        chunk[dim] = -(-chunk[dim] // 2)

    input_numel = channels
    for dim in range(dims):
        input_numel *= input_extent(chunk, dim)
    input_buffer = torch.empty(input_numel, device=compute_device, dtype=output.dtype)
    if isinstance(bias, torch.Tensor):
        bias = bias.to(device=compute_device, non_blocking=True)
    resident = channels_per_tile == out_channels
    if resident:
        weight_buffer.copy_(weight, non_blocking=True)
    conv_op = torch.nn.functional.conv3d if dims == 3 else torch.nn.functional.conv2d
    no_padding = (0,) * dims
    tile_starts = [range(0, out_size[dim], chunk[dim]) for dim in range(dims)]

    with torch.cuda.device_of(input_buffer):
        for sample in range(batch):
            for starts in itertools.product(*tile_starts):
                ends = [min(start + length, total) for start, length, total in zip(starts, chunk, out_size)]
                tile_shape = [channels]
                source = [slice(sample, sample + 1), slice(None)]
                interior = [slice(None), slice(None)]
                pads = []
                for dim in range(dims):
                    in_start = starts[dim] * stride[dim] - padding[dim]
                    extent = (ends[dim] - starts[dim] - 1) * stride[dim] + span[dim]
                    pad_before = max(0, -in_start)
                    filled = extent - max(0, in_start + extent - size[dim])
                    tile_shape.append(extent)
                    source.append(slice(in_start + pad_before, in_start + filled))
                    interior.append(slice(pad_before, filled))
                    pads.append((pad_before, filled))
                input_tile = input_buffer[: math.prod(tile_shape)].view(1, *tile_shape)
                for dim, (pad_before, filled) in enumerate(pads):
                    edge = [slice(None)] * (dims + 2)
                    edge[dim + 2] = slice(None, pad_before)
                    input_tile[tuple(edge)].zero_()
                    edge[dim + 2] = slice(filled, None)
                    input_tile[tuple(edge)].zero_()
                input_tile[tuple(interior)].copy_(input_tensor[tuple(source)], non_blocking=True)
                target = [slice(sample, sample + 1), None] + [
                    slice(start, end) for start, end in zip(starts, ends)
                ]
                for start in range(0, out_channels, channels_per_tile):
                    end = min(start + channels_per_tile, out_channels)
                    weight_tile = weight_buffer[: end - start]
                    if not resident:
                        weight_tile.copy_(weight[start:end], non_blocking=True)
                    target[1] = slice(start, end)
                    bias_tile = None if bias is None else bias[start:end]
                    if packed:
                        result = comfy.ops.dense_conv(input_tile, weight_tile, bias_tile, stride, no_padding)
                    else:
                        result = conv_op(input_tile, weight_tile, bias_tile, stride, 0, dilation)
                    output[tuple(target)].copy_(result, non_blocking=True)

    return output


def _run_attention_on_compute(func, q, k, v, heads, mask=None, *args, compute_device, plan, **kwargs):
    """Run attention on the compute GPU in head chunks over the whole sequence.

    Heads are independent, so a chunk copies only its own heads' Q, K and V.
    Every chunk takes all queries: backends prepare K and V on each call (Kitchen
    int8 rotates and quantizes them), so query chunks would repeat that work.
    There are at least _PIPELINE_CHUNKS head chunks, so their copies overlap
    each other's attention, and three fit half the free VRAM. On OOM the call
    restarts with half the heads, then half the queries, and plan keeps the
    result for later calls. When one head group and one query still do not
    fit, attention fails with an out-of-memory error.
    """
    key = (q.shape, k.shape)
    if mask is not None:
        return func(q, k, v, heads, mask, *args, **kwargs)

    skip_reshape = kwargs.get("skip_reshape", False)
    skip_output_reshape = kwargs.get("skip_output_reshape", False)
    seq_dim = 2 if skip_reshape else 1
    out_seq_dim = 2 if skip_output_reshape else 1
    tokens = q.shape[seq_dim]
    head_dim = q.shape[-1] if skip_reshape else q.shape[-1] // heads
    kv_heads = k.shape[1] if skip_reshape else k.shape[-1] // head_dim
    group = heads // kv_heads

    def head_slice(tensor, start, count, total, heads_last):
        if not heads_last:
            return tensor.narrow(1, start, count)
        width = tensor.shape[-1] // total
        return tensor.narrow(-1, start * width, count * width)

    chunk = plan.get(key)
    if chunk is None:
        budget = _compute_budget(compute_device, _ATTENTION_RESERVE_BYTES)
        group_bytes = (
            (k.numel() + v.numel()) // kv_heads + 3 * group * q.numel() // heads
        ) * q.element_size()
        groups = max(
            1,
            min(-(-kv_heads // _PIPELINE_CHUNKS), budget // 2 // (3 * group_bytes)),
        )
        # [heads per chunk, queries per chunk]
        chunk = plan[key] = [groups * group, tokens]

    output = None

    def load(part):
        head, count, start, length = part
        return (
            head_slice(q, head, count, heads, not skip_reshape).narrow(seq_dim, start, length).to(device=compute_device, non_blocking=True),
            head_slice(k, head // group, count // group, kv_heads, not skip_reshape).to(device=compute_device, non_blocking=True),
            head_slice(v, head // group, count // group, kv_heads, not skip_reshape).to(device=compute_device, non_blocking=True),
        )

    def run(inputs):
        q_chunk, k_chunk, v_chunk = inputs
        count = q_chunk.shape[1] if skip_reshape else q_chunk.shape[-1] // head_dim
        return func(q_chunk, k_chunk, v_chunk, count, None, *args, **kwargs)

    def store(part, result):
        nonlocal output
        head, count, start, length = part
        if output is None:
            shape = list(result.shape)
            shape[out_seq_dim] = tokens
            head_axis = 1 if skip_output_reshape else -1
            shape[head_axis] = shape[head_axis] // count * heads
            output = torch.empty(shape, device=q.device, dtype=result.dtype)
        head_slice(output, head, count, heads, not skip_output_reshape).narrow(out_seq_dim, start, length).copy_(result, non_blocking=True)

    while True:
        parts = [
            (head, min(chunk[0], heads - head), start, min(chunk[1], tokens - start))
            for head in range(0, heads, chunk[0])
            for start in range(0, tokens, chunk[1])
        ]
        try:
            with _current_device(compute_device):
                _pipeline_on_compute(parts, compute_device, load, run, store)
            return output
        except torch.OutOfMemoryError:
            # Recover outside the except block, once the failed call's tensors are freed.
            pass
        if torch.device(compute_device).type == "cuda":
            torch.cuda.synchronize(compute_device)
        torch.cuda.empty_cache()
        if chunk[0] > group:
            chunk[0] = max(group, chunk[0] // 2 // group * group)
        elif chunk[1] > 1:
            chunk[1] = max(1, chunk[1] // 2)
        else:
            raise torch.OutOfMemoryError(
                f"Attention does not fit on {compute_device} for one head group and one query"
            )


def _materialize_linear(module, dtype, device):
    """Apply ComfyUI weight casts and patches on the given device."""
    module_cast_bias_weight = getattr(module, "cast_bias_weight", None)
    if callable(module_cast_bias_weight):
        weight, bias = module_cast_bias_weight(dtype=dtype, device=device)
        return weight, bias, None

    needs_materialization = (
        getattr(module, "comfy_cast_weights", False)
        or bool(getattr(module, "weight_function", ()))
        or bool(getattr(module, "bias_function", ()))
    )
    if not needs_materialization:
        weight = module.weight.to(device=device, dtype=dtype)
        bias = getattr(module, "bias", None)
        if isinstance(bias, torch.Tensor):
            bias = bias.to(device=device, dtype=dtype)
        return weight, bias, None

    from comfy.ops import cast_bias_weight

    weight, bias, offload_state = cast_bias_weight(
        module,
        dtype=dtype,
        device=device,
        bias_dtype=dtype,
        offloadable=True,
    )
    return weight, bias, offload_state


def _log_weight_preparation(module, device, weight, used_comfy_cast):
    """Confirm once where a weight was materialized."""
    if getattr(module, "_mgpu_donor_preparation_logged", False):
        return

    operations = ["dtype cast"]
    if used_comfy_cast:
        operations.append("ComfyUI weight functions")
    if hasattr(module.weight, "dequantize"):
        operations.append("dequantization")
    if getattr(module, "weight_function", ()):
        operations.append("weight patches")
    if getattr(module, "bias_function", ()):
        operations.append("bias patches")

    logger.info(
        "[MultiGPU DisTorch V2] Weight preparation confirmed on %s: %s; "
        "prepared weight is on %s (%s, %.2f MiB) for the compute-GPU GEMM",
        device,
        ", ".join(operations),
        weight.device,
        weight.dtype,
        weight.numel() * weight.element_size() / (1024**2),
    )
    module._mgpu_donor_preparation_logged = True


def _plain_lora_patches(patches):
    """ComfyUI weight patches as (down, up, scale), or () to merge them as usual.

    Merging a LoRA costs a full-size GEMM. Plain LoRAs are
    instead applied as two thin GEMMs on the compute GPU.
    """
    if not patches:
        return ()
    from comfy.weight_adapter.lora import LoRAAdapter

    lora = []
    for strength, adapter, strength_model, offset, function in patches:
        if (
            not isinstance(adapter, LoRAAdapter)
            or strength_model != 1.0
            or offset is not None
            or function is not None
        ):
            return ()
        up, down, alpha, mid, dora_scale, reshape = adapter.weights
        if mid is not None or dora_scale is not None or reshape is not None or up.ndim != 2:
            return ()
        rank = down.shape[0]
        lora.append((down, up, strength * (1.0 if alpha is None else float(alpha) / rank)))
    return lora


def _is_mixed_int8_linear(module):
    """Whether mixed_int8 stores a linear as int8 ConvRot, which rotates 256-wide input groups."""
    return (
        _is_eligible_donor_gemm(module)
        and _can_tile_linear_on_compute(module)
        and getattr(module, "layout_type", None) is None
        and module.in_features % 256 == 0
    )


def _prepare_mixed_int8_weight(module, name, patches, dtype, donor_device, key):
    """Quantize a mixed_int8 linear to int8 ConvRot once, at load, and keep it in system RAM.

    The donor quantizes one layer at a time and keeps only activations; the
    compute GPU reads the int8 weight from RAM as fast as from the donor, since
    PCIe limits both. The original weight is neither moved nor copied, so a
    memory-mapped GGUF stays on disk. Plain LoRAs stay low-rank GEMMs for the
    compute GPU; other patches are merged before quantizing. A reload with the
    same patches (key) reuses the int8 weight.
    """
    if getattr(module, "_mgpu_int8_key", None) == key:
        return

    from comfy.lora import calculate_weight
    from comfy.quant_ops import QuantizedTensor

    weight_patches = patches.get(f"{name}.weight", ())
    bias_patches = patches.get(f"{name}.bias", ())
    lora = _plain_lora_patches(weight_patches)
    with _current_device(donor_device):
        weight, bias, offload_state = _materialize_linear(module, dtype, donor_device)
        try:
            if weight_patches and not lora:
                weight = calculate_weight(weight_patches, weight.float(), f"{name}.weight")
            if bias is not None and bias_patches:
                bias = calculate_weight(bias_patches, bias.float(), f"{name}.bias")
            quantized = QuantizedTensor.from_float(
                weight.to(dtype),
                "TensorWiseINT8Layout",
                is_weight=True,
                per_channel=True,
                convrot=True,
            )
            quantized = quantized.to(device="cpu")
            bias = None if bias is None else bias.to(device="cpu", dtype=dtype)
        finally:
            if offload_state is not None:
                from comfy.ops import uncast_bias_weight

                uncast_bias_weight(module, weight, bias, offload_state)
    module.register_buffer("_mgpu_int8_weight", quantized, persistent=False)
    module.register_buffer("_mgpu_int8_bias", bias, persistent=False)
    module._mgpu_int8_lora = lora
    module._mgpu_int8_key = key


@_without_streaming
def _prepare_linear_on_donor(module, dtype, donor_device):
    """Materialize a linear's weight on the donor; returns (weight, bias, offload_state, lora).

    Plain GGUF LoRAs are left out of the weight and returned as low-rank
    patches for the compute GPU; release with _release_prepared_linear.
    """
    used_comfy_cast = (
        getattr(module, "comfy_cast_weights", False)
        or bool(getattr(module, "weight_function", ()))
        or bool(getattr(module, "bias_function", ()))
    )
    # The GGUF loader merges its patches into the dequantized weight on every call.
    lora = _plain_lora_patches(
        [patch for patch_list, _ in getattr(module.weight, "patches", ()) for patch in patch_list]
    )
    if lora:
        patches = module.weight.patches
        module.weight.patches = []
    try:
        with _current_device(donor_device):
            weight, bias, offload_state = _materialize_linear(module, dtype, donor_device)
    finally:
        if lora:
            module.weight.patches = patches
    _log_weight_preparation(module, donor_device, weight, used_comfy_cast)
    return weight, bias, offload_state, lora


def _release_prepared_linear(module, weight, bias, offload_state):
    if offload_state is not None:
        from comfy.ops import uncast_bias_weight

        uncast_bias_weight(module, weight, bias, offload_state)


def _run_donor_prepared_linear_on_compute(
    input_tensor, module, compute_device, donor_device
):
    """Prepare the weight on the donor and stream its tiles through the compute GPU."""
    weight, bias, offload_state, lora = _prepare_linear_on_donor(
        module, input_tensor.dtype, donor_device
    )
    try:
        return _run_tiled_linear_on_compute(
            input_tensor, weight, bias, compute_device, lora
        )
    finally:
        _release_prepared_linear(module, weight, bias, offload_state)


class _MixedSchedule:
    """The order a model's mixed units ran in, for staging each next unit's weights early.

    A unit is a mixed linear or a fused MLP (see _run_fused_mlp). It stages the
    next unit's weights on the compute GPU from a copy stream while its own last
    chunks run. Memory allocated on that stream is freed only once the compute
    stream has finished with it (see retire), since the allocator would
    otherwise hand it back to the copy stream too early.
    """

    def __init__(self, compute_device, donor_device):
        self.compute_device = compute_device
        self.copy_stream = torch.cuda.Stream(compute_device)
        # Preparing a weight on the donor's default stream would queue it ahead of
        # the chunk copies, which run there.
        self.donor_stream = torch.cuda.Stream(donor_device)
        # Kept here, not on the modules: a module stored on another would be
        # registered as its submodule.
        self.following = {}
        self.mlp_linears = {}
        self.previous = None
        self.pending = []
        self.retired = []
        self.free = None

    def start_forward(self):
        # Stages left from an interrupted forward would pin compute memory.
        for module in self.pending:
            module._mgpu_staged = None
        self.pending = []
        self.previous = None
        self.free = None

    def linears(self, unit):
        return self.mlp_linears.get(unit, (unit,))

    def weight_bytes(self, unit, dtype):
        return sum(_staged_weight_bytes(linear, dtype) for linear in self.linears(unit))

    def stage(self, unit, dtype):
        """Stage a unit's weights, or None when one is too large and streams in tiles instead."""
        staged = tuple(_stage_linear(linear, dtype, self) for linear in self.linears(unit))
        return None if any(linear is None for linear in staged) else staged

    def budget(self, staged_bytes):
        """Tile budget while staged_bytes of weights are on the compute GPU.

        Measured once per forward: _compute_budget synchronizes every device, so
        measuring at every unit would drain the pipeline between them.
        """
        if self.free is None:
            self.free = _compute_budget(self.compute_device) + staged_bytes
        return self.free - staged_bytes

    def retire(self, staged, compute_stream):
        for _, done in self.retired:
            done.synchronize()
        self.retired = [(staged, compute_stream.record_event())]


_StagedLinear = namedtuple("_StagedLinear", "weight bias lora_down lora_up dtype ready")


def _staged_weight_bytes(module, dtype):
    if module._mgpu_quantized:
        return mm.module_size(module)
    return module.out_features * module.in_features * (1 if module._mgpu_int8 else dtype.itemsize)


def _copy_to_compute(weight, compute_device):
    """Copy a weight to the compute GPU without blocking the host.

    QuantizedTensor.to() ignores non_blocking, so an int8 weight's parts are
    copied here; its small pageable scales go first so that they don't wait
    behind the pinned data.
    """
    from comfy.quant_ops import QuantizedTensor

    if not isinstance(weight, QuantizedTensor):
        return weight.to(device=compute_device, non_blocking=True)
    params = weight.params
    params = dataclasses.replace(
        params,
        **{
            name: getattr(params, name).to(device=compute_device, non_blocking=True)
            for name in params._tensor_fields()
        },
    )
    qdata = weight._qdata.to(device=compute_device, non_blocking=True)
    return QuantizedTensor(qdata, weight._layout_cls, params)


def _stage_linear(module, dtype, schedule=None):
    """Put a mixed linear's GEMM weight on the compute GPU, on the schedule's streams if given.

    A quantized linear and mixed_int8 send their stored weight; mixed prepares the
    weight on the donor. Returns None for a prepared weight larger than half
    the compute GPU's free memory, which streams in tiles instead.
    """
    compute_device = module._mgpu_compute_device
    if not (module._mgpu_int8 or module._mgpu_quantized) and compute_device.type == "cuda" and (
        2 * _staged_weight_bytes(module, dtype)
        > mm.get_free_memory(compute_device) - _COMPUTE_GPU_RESERVE_BYTES
    ):
        return None
    streams = contextlib.ExitStack()
    if schedule is not None:
        streams.enter_context(torch.cuda.stream(schedule.donor_stream))
        streams.enter_context(torch.cuda.stream(schedule.copy_stream))
    with streams:
        if module._mgpu_quantized:
            weight, bias, lora = module.weight, module.bias, ()
            offload_state = None
        elif module._mgpu_int8:
            weight, bias, lora = module._mgpu_int8_weight, module._mgpu_int8_bias, module._mgpu_int8_lora
            offload_state = None
        else:
            weight, bias, offload_state, lora = _prepare_linear_on_donor(
                module, dtype, module._mgpu_donor_execution_device
            )
        try:
            staged_weight = _copy_to_compute(weight, compute_device)
            staged_bias = None if bias is None else bias.to(device=compute_device, non_blocking=True)
            lora_down, lora_up = _lora_on_compute(lora, compute_device, dtype)
        finally:
            _release_prepared_linear(module, weight, bias, offload_state)
    ready = schedule.copy_stream.record_event() if schedule is not None else None
    return _StagedLinear(staged_weight, staged_bias, lora_down, lora_up, dtype, ready)


def _run_staged_linear(input_tensor, staged, compute_device, budget, during=None):
    """Stream token chunks through a staged weight, with copies overlapping the GEMMs."""
    flat_input = input_tensor.reshape(-1, input_tensor.shape[-1])
    tokens, in_features = flat_input.shape
    out_features = staged.weight.shape[0]
    output = torch.empty(
        (tokens, out_features), device=input_tensor.device, dtype=input_tensor.dtype
    )
    lora_rank = 0 if staged.lora_down is None else staged.lora_down.shape[0]
    # Input, output and LoRA chunks, doubled for the kernel's temporaries.
    token_bytes = 2 * (in_features + out_features + lora_rank) * output.element_size()
    _run_linear_chunks(
        flat_input,
        output,
        staged.weight,
        staged.bias,
        staged.lora_down,
        staged.lora_up,
        _token_chunks(tokens, token_bytes, budget),
        compute_device,
        during,
    )
    return output.reshape(*input_tensor.shape[:-1], out_features)


def _take_staged(unit, dtype, schedule):
    """A unit's weights, as the previous unit staged them or staged now."""
    staged = unit._mgpu_staged
    unit._mgpu_staged = None
    if staged is None or staged[0].dtype != dtype:
        staged = schedule.stage(unit, dtype)
    return staged


def _start_unit(unit, staged, dtype, schedule):
    """Record a unit as running now and wait for its weights.

    The schedule learns the order units run in during each forward, so no
    model-specific knowledge is needed. Returns the unit's tile budget and a
    callback that stages the following unit's weights once the unit's first
    chunk is queued, so they are prepared and copied while the rest runs.
    """
    if schedule.previous is not None:
        schedule.following[schedule.previous] = unit
    schedule.previous = unit
    budget = schedule.budget(schedule.weight_bytes(unit, dtype))
    compute_stream = torch.cuda.current_stream(schedule.compute_device)
    for linear in staged:
        compute_stream.wait_event(linear.ready)
    following = schedule.following.get(unit)
    if following is None or following is unit:
        return budget, None

    def during():
        following._mgpu_staged = schedule.stage(following, dtype)
        schedule.pending.append(following)

    return budget - schedule.weight_bytes(following, dtype), during


def _run_mixed_linear(input_tensor, module):
    """Run a mixed linear on the compute GPU from a staged weight, and stage the next unit."""
    compute_device = module._mgpu_compute_device
    schedule = module._mgpu_schedule
    if schedule is None:
        staged = _stage_linear(module, input_tensor.dtype)
        if staged is not None:
            return _run_staged_linear(
                input_tensor, staged, compute_device, _compute_budget(compute_device)
            )
    else:
        staged = _take_staged(module, input_tensor.dtype, schedule)
    if staged is None:
        return _run_donor_prepared_linear_on_compute(
            input_tensor, module, compute_device, module._mgpu_donor_execution_device
        )
    budget, during = _start_unit(module, staged, input_tensor.dtype, schedule)
    output = _run_staged_linear(input_tensor, staged[0], compute_device, budget, during)
    schedule.retire(staged, torch.cuda.current_stream(compute_device))
    return output


@contextlib.contextmanager
def _staged_parameters(module, staged):
    weight, bias = module.weight, module.bias
    module.weight = torch.nn.Parameter(staged.weight, requires_grad=False)
    if bias is not None:
        module.bias = torch.nn.Parameter(staged.bias, requires_grad=False)
    try:
        yield
    finally:
        module.weight, module.bias = weight, bias


def _run_fused_mlp(input_tensor, mlp):
    """Run a token-wise MLP whole on the compute GPU, one token chunk at a time.

    Each chunk is copied there and its result back once, so the wide hidden
    activation never visits the donor and the activation function runs on the
    compute GPU. Its linears run the weights staged here (see configure_mixed_gemm).
    """
    schedule = mlp._mgpu_schedule
    compute_device = schedule.compute_device
    staged = _take_staged(mlp, input_tensor.dtype, schedule)
    if staged is None:
        return mlp._mgpu_original_forward(input_tensor)
    linears = schedule.linears(mlp)
    budget, during = _start_unit(mlp, staged, input_tensor.dtype, schedule)
    flat_input = input_tensor.reshape(-1, input_tensor.shape[-1])
    tokens = flat_input.shape[0]
    output = torch.empty(
        (tokens, linears[-1].out_features), device=input_tensor.device, dtype=input_tensor.dtype
    )
    # The input and every linear's output, doubled for activations and kernel temporaries.
    token_bytes = (
        2 * (flat_input.shape[-1] + sum(linear.out_features for linear in linears))
        * output.element_size()
    )
    with contextlib.ExitStack() as stack:
        for linear, weights in zip(linears, staged):
            linear._mgpu_fused = weights
            stack.callback(setattr, linear, "_mgpu_fused", None)
            if linear._mgpu_quantized:
                stack.enter_context(_staged_parameters(linear, weights))
        with _current_device(compute_device):
            _pipeline_on_compute(
                _token_chunks(tokens, token_bytes, budget),
                compute_device,
                lambda chunk: flat_input[chunk].to(device=compute_device, non_blocking=True),
                mlp._mgpu_original_forward,
                lambda chunk, result: output[chunk].copy_(result, non_blocking=True),
                during,
            )
    schedule.retire(staged, torch.cuda.current_stream(compute_device))
    return output.reshape(*input_tensor.shape[:-1], output.shape[-1])


def _run_quantized_linear_on_compute(input_tensor, module, compute_device):
    """Run a ComfyUI quantized linear's own forward on the compute GPU in token chunks.

    The packed weight is staged once, and the next unit's while this one runs
    (see _start_unit). The module's forward runs against the staged weight, so it
    quantizes each input chunk and runs the quantized GEMM (such as Kitchen int8)
    itself. Token chunks stream through with their copies overlapping the GEMMs.
    """
    dtype = input_tensor.dtype
    schedule = module._mgpu_schedule
    if schedule is None:
        staged = (_stage_linear(module, dtype),)
        budget = _compute_budget(compute_device) - _staged_weight_bytes(module, dtype)
        during = None
    else:
        staged = _take_staged(module, dtype, schedule)
        budget, during = _start_unit(module, staged, dtype, schedule)
    flat_input = input_tensor.reshape(-1, input_tensor.shape[-1])
    tokens = flat_input.shape[0]
    # A quantized copy of each input chunk and its output.
    token_bytes = 2 * (module.in_features + module.out_features) * input_tensor.element_size()
    output = None

    def store(chunk, result):
        nonlocal output
        if output is None:
            output = torch.empty(
                (tokens, result.shape[-1]), device=input_tensor.device, dtype=result.dtype
            )
        output[chunk].copy_(result, non_blocking=True)

    with _staged_parameters(module, staged[0]), _current_device(compute_device):
        _pipeline_on_compute(
            _token_chunks(tokens, token_bytes, budget),
            compute_device,
            lambda chunk: flat_input[chunk].to(device=compute_device, non_blocking=True),
            module._mgpu_original_forward,
            store,
            during,
        )
    if schedule is not None:
        schedule.retire(staged, torch.cuda.current_stream(compute_device))
    return output.reshape(*input_tensor.shape[:-1], output.shape[-1])


def _configure_mixed_conv(module):
    """Tile a conv's math onto the compute GPU below its own forward.

    Hooking _conv_forward keeps layer-specific padding and temporal caches
    (such as Wan's CausalConv3d) and ComfyUI's weight casts on the donor.
    """
    module._mgpu_original_conv_forward = module._conv_forward
    module._mgpu_donor_gemm_calls = 0

    def mixed_conv_forward(input, weight, bias, autopad=None):
        padding = module.padding
        if module.padding_mode != "zeros":
            input = torch.nn.functional.pad(
                input, module._reversed_padding_repeated_twice, mode=module.padding_mode
            )
            padding = (0,) * len(padding)
        # Matches comfy.ops.Conv3d: the causal kernel is cut to the frames present.
        if autopad == "causal_zero":
            weight = weight[:, :, -input.shape[2] :]
        output = _run_tiled_conv_on_compute(
            module, input, weight, bias, padding, module._mgpu_compute_device
        )
        module._mgpu_donor_gemm_calls += 1
        if module._mgpu_donor_gemm_calls == 1:
            logger.info(
                "[MultiGPU DisTorch V2] Donor-prepared compute conv active: %s -> %s (%s)",
                input.device,
                module._mgpu_compute_device,
                type(module).__name__,
            )
        return output

    module._conv_forward = mixed_conv_forward


# comfy.ops Linear.forward's fused prologue/epilogue arguments (see linear_input_act_).
_LINEAR_FUSION_KWARGS = frozenset(("input_act", "act_weight", "residual", "residual_scale"))


def configure_mixed_gemm(module, compute_device, donor_device, int8=False, schedule=None):
    """Run an eligible linear's or conv's GEMM on the compute GPU; its activations stay on the donor.

    With int8, the linear runs its load-time int8 weight (see
    _prepare_mixed_int8_weight) as a Comfy Kitchen int8 GEMM. With a schedule,
    each linear stages the next one's weight while it runs (see _run_mixed_linear).
    """
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)):
        tileable = _can_tile_conv_on_compute(module)
    elif _is_eligible_donor_gemm(module):
        tileable = _can_tile_linear_on_compute(module)
    else:
        return False
    if not tileable:
        if not getattr(module, "_mgpu_donor_skip_logged", False):
            logger.info(
                "[MultiGPU DisTorch V2] Mixed compute GEMM skipped for %s: module is not a standard linear or ungrouped conv",
                type(module).__name__,
            )
            module._mgpu_donor_skip_logged = True
        return False

    module._mgpu_compute_device = torch.device(compute_device)
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)):
        if not hasattr(module, "_mgpu_original_conv_forward"):
            _configure_mixed_conv(module)
        return True

    module._mgpu_donor_execution_device = torch.device(donor_device)
    module._mgpu_int8 = int8
    # ComfyUI sets layout_type on layers whose weight is quantized (int8, fp8, ...).
    module._mgpu_quantized = getattr(module, "layout_type", None) is not None
    module._mgpu_schedule = schedule
    module._mgpu_staged = None
    # The weight a fused MLP staged for this linear while it runs (see _run_fused_mlp).
    module._mgpu_fused = None
    if not hasattr(module, "_mgpu_original_forward"):
        module._mgpu_original_forward = module.forward
        module._mgpu_donor_gemm_calls = 0

        def mixed_forward(*args, **kwargs):
            if len(args) != 1 or not isinstance(args[0], torch.Tensor) or kwargs.keys() - _LINEAR_FUSION_KWARGS:
                return module._mgpu_original_forward(*args, **kwargs)
            if kwargs:
                # ComfyUI's fused prologue/epilogue kernels would run on the donor
                # tensors; unfold them so the GEMM itself still goes to the compute GPU.
                x = comfy.ops._eager_input_act(args[0], kwargs.get("input_act"), kwargs.get("act_weight"))
                return comfy.ops._linear_residual(mixed_forward(x), kwargs.get("residual"), kwargs.get("residual_scale"))
            fused = module._mgpu_fused
            if fused is not None:
                if module._mgpu_quantized:
                    return module._mgpu_original_forward(args[0])
                return _linear_on_compute(
                    args[0], fused.weight, fused.bias, fused.lora_down, fused.lora_up
                )
            if module._mgpu_quantized:
                output = _run_quantized_linear_on_compute(
                    args[0], module, module._mgpu_compute_device
                )
            else:
                output = _run_mixed_linear(args[0], module)
            module._mgpu_donor_gemm_calls += 1
            if module._mgpu_donor_gemm_calls == 1:
                logger.info(
                    "[MultiGPU DisTorch V2] Donor-prepared compute GEMM active: %s -> %s (%s%s)",
                    module._mgpu_donor_execution_device,
                    module._mgpu_compute_device,
                    type(module).__name__,
                    ", int8" if module._mgpu_int8 else "",
                )
            return output

        module.forward = mixed_forward
    return True


# Elementwise modules that may sit between a Sequential MLP's linears.
_MLP_ACTIVATIONS = (
    torch.nn.GELU,
    torch.nn.SiLU,
    torch.nn.ReLU,
    torch.nn.Tanh,
    torch.nn.Sigmoid,
    torch.nn.Dropout,
    torch.nn.Identity,
)


def _fused_mlp_linears(module, schedule):
    """The linears of a token-wise MLP whose GEMMs all run staged on the compute GPU, or None.

    Token-wise MLPs are a Sequential of linears and activations, or a SwiGLU
    FeedForward's w1, w3 and w2 (as in Lumina and Z-Image).
    """
    children = dict(module.named_children())
    if isinstance(module, torch.nn.Sequential):
        linears = [child for child in children.values() if not isinstance(child, _MLP_ACTIVATIONS)]
    elif children.keys() == {"w1", "w2", "w3"}:
        linears = [children["w1"], children["w3"], children["w2"]]
    else:
        return None
    if len(linears) < 2 or any(
        getattr(linear, "_mgpu_schedule", None) is not schedule
        for linear in linears
    ):
        return None
    return linears


def configure_fused_mlp(mlp, linears, schedule):
    """Run a token-wise MLP whole on the compute GPU in token chunks (see _run_fused_mlp)."""
    schedule.mlp_linears[mlp] = linears
    mlp._mgpu_schedule = schedule
    mlp._mgpu_staged = None
    if not hasattr(mlp, "_mgpu_original_forward"):
        mlp._mgpu_original_forward = mlp.forward

        def fused_forward(*args, **kwargs):
            if len(args) != 1 or kwargs or not isinstance(args[0], torch.Tensor):
                return mlp._mgpu_original_forward(*args, **kwargs)
            return _run_fused_mlp(args[0], mlp)

        mlp.forward = fused_forward


def _first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        value = value.values()
    elif not isinstance(value, (tuple, list)):
        return None
    for item in value:
        tensor = _first_tensor(item)
        if tensor is not None:
            return tensor
    return None


def _mixed_weight_device(module, donor_device):
    """Where mixed mode stores a module's weights.

    Castable weights stay where they were loaded (a memory-mapped GGUF stays
    on disk) and are prepared on the donor one layer at a time, so the donor's
    memory is left to activations and the compute GPU only holds GEMM tiles.
    Modules that cannot be cast at runtime live on the donor.
    """
    return "cpu" if hasattr(module, "comfy_cast_weights") else donor_device


def select_mixed_donor_device(model, block_assignments, compute_device):
    """Pick the donor GPU that holds a mixed-mode model's activations, or None."""
    compute_device = torch.device(compute_device)
    donors = sorted(
        {str(device) for device in block_assignments.values()}
        - {"cpu", str(compute_device)}
    )
    reasons = []
    if not hasattr(model, "diffusion_model"):
        reasons.append("model has no diffusion model")
    if not _is_comfy_kitchen_attention_enabled():
        reasons.append("Comfy Kitchen attention is disabled")
    if not donors:
        reasons.append("allocation has no donor GPU")
    else:
        reasons.extend(_donor_gemm_skip_reasons(compute_device, torch.device(donors[0])))
    if reasons:
        logger.info(
            "[MultiGPU DisTorch V2] Mixed mode disabled: %s", "; ".join(reasons)
        )
        return None
    return torch.device(donors[0])


def configure_mixed_execution(diffusion_model, donor_device, compute_device, schedule):
    """Keep the diffusion model's activations on the donor and run all compute on the compute GPU.

    Every op that touches a donor tensor is streamed to the compute GPU (see
    _StreamedActivations); linears and convs (see configure_mixed_gemm), token-wise
    MLPs (see _run_fused_mlp) and attention run there in tiles sized to its free VRAM.
    """
    if not hasattr(diffusion_model, "_mgpu_original_forward"):
        diffusion_model._mgpu_original_forward = diffusion_model.forward

        def mixed_forward(*args, transformer_options={}, **kwargs):
            donor = diffusion_model._mgpu_donor_execution_device
            compute = diffusion_model._mgpu_compute_device
            diffusion_model._mgpu_schedule.start_forward()
            output_device = _first_tensor(args).device
            transformer_options = transformer_options.copy()
            previous_override = transformer_options.get("optimized_attention_override")

            def attention_on_compute(func, *attention_args, **attention_kwargs):
                if previous_override is not None:
                    func = functools.partial(previous_override, func)
                return _run_attention_on_compute(
                    func,
                    *attention_args,
                    compute_device=compute,
                    plan=diffusion_model._mgpu_attention_plan,
                    **attention_kwargs,
                )

            transformer_options["optimized_attention_override"] = attention_on_compute
            donor_args = _move_tensors(args, donor)
            with _StreamedActivations(donor, compute), _current_device(compute):
                output = diffusion_model._mgpu_original_forward(
                    *donor_args,
                    transformer_options=transformer_options,
                    **_move_tensors(kwargs, donor),
                )
            return _move_tensors(output, output_device)

        diffusion_model.forward = mixed_forward
        logger.info(
            "[MultiGPU DisTorch V2] Mixed execution: activations on %s, all compute on %s",
            donor_device,
            compute_device,
        )

    diffusion_model._mgpu_donor_execution_device = torch.device(donor_device)
    diffusion_model._mgpu_compute_device = torch.device(compute_device)
    diffusion_model._mgpu_schedule = schedule
    # Chunk sizes learned from OOMs, a few ints per attention shape, kept until the next load.
    diffusion_model._mgpu_attention_plan = {}
    for module in diffusion_model.modules():
        linears = _fused_mlp_linears(module, schedule)
        if linears is not None:
            configure_fused_mlp(module, linears, schedule)
    if schedule.mlp_linears:
        logger.info(
            "[MultiGPU DisTorch V2] Mixed execution: %d MLPs run whole on %s in token chunks",
            len(schedule.mlp_linears),
            compute_device,
        )


def configure_hip_donor_gemm_offload(
    module, compute_device, donor_device, execution_mode="disabled"
):
    """Run an eligible donor-assigned software-GEMM module on its donor GPU."""
    original_forward = getattr(module, "_mgpu_original_forward", None)
    compute_device = torch.device(compute_device)
    donor_device = torch.device(donor_device)
    enabled = (
        execution_mode == "all"
        and _is_eligible_donor_gemm(module)
        and _is_comfy_kitchen_attention_enabled()
        and not _donor_gemm_skip_reasons(compute_device, donor_device)
    )

    if not enabled:
        if original_forward is not None:
            module.forward = original_forward
            del module._mgpu_original_forward
            del module._mgpu_donor_execution_device
            if hasattr(module, "_mgpu_donor_preparation_logged"):
                del module._mgpu_donor_preparation_logged
        return False

    if original_forward is None:
        original_forward = module.forward
        module._mgpu_original_forward = original_forward
        module._mgpu_donor_gemm_calls = 0

        def donor_forward(*args, **kwargs):
            target_device = module._mgpu_donor_execution_device
            donor_args = _move_tensors(args, target_device)
            donor_kwargs = _move_tensors(kwargs, target_device)
            with _current_device(target_device):
                output = module._mgpu_original_forward(*donor_args, **donor_kwargs)
            output = _move_tensors(output, module._mgpu_compute_device)
            module._mgpu_donor_gemm_calls += 1
            if module._mgpu_donor_gemm_calls == 1:
                logger.info(
                    "[MultiGPU DisTorch V2] Donor GEMM active: %s -> %s (%s)",
                    module._mgpu_compute_device,
                    target_device,
                    type(module).__name__,
                )
            return output

        module.forward = donor_forward

    module._mgpu_compute_device = compute_device
    module._mgpu_donor_execution_device = donor_device
    return True


def pin_hip_offloaded_weight(module):
    """Convert a CPU module weight into the HIP backend's mapped-host format."""
    if not getattr(torch.version, "hip", None):
        return False

    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.device.type != "cpu":
        return False
    if weight.is_pinned():
        return True

    try:
        from comfy_kitchen.backends import hip
    except ImportError:
        return False

    offload_weight = getattr(hip, "offload_weight", None)
    if not callable(offload_weight):
        return False

    # Swap only the storage: replacing the weight drops tensor-subclass metadata
    # such as a GGUF weight's quant type and logical shape.
    weight.data = offload_weight(weight.data)
    return True


def register_patched_safetensor_modelpatcher():
    """Register and patch the ModelPatcher for distributed safetensor loading"""
    # Patch ComfyUI's ModelPatcher
    if not hasattr(comfy.model_patcher.ModelPatcher, "_distorch_patched"):

        # PATCH load_models_gpu with correct memory calculations per model flags
        def patched_load_models_gpu(
            models,
            memory_required=0,
            force_patch_weights=False,
            minimum_memory_required=None,
            force_full_load=False,
        ):
            from comfy.model_management import (
                cleanup_models_gc,
                get_free_memory,
                free_memory,
                current_loaded_models,
            )
            from comfy.model_management import (
                VRAMState,
                vram_state,
                lowvram_available,
                MIN_WEIGHT_MEMORY_RATIO,
            )
            from comfy.model_management import (
                minimum_inference_memory,
                extra_reserved_memory,
                is_device_cpu,
            )

            multigpu_memory_log("load_models_gpu_top_level", "start")

            cleanup_models_gc()

            inference_memory = minimum_inference_memory()
            extra_reserved_mem = extra_reserved_memory()
            memory_required_total = memory_required + extra_reserved_mem
            extra_mem = max(inference_memory, memory_required_total)
            if minimum_memory_required is None:
                minimum_memory_required = extra_mem
            else:
                minimum_memory_required = max(
                    inference_memory, minimum_memory_required + extra_reserved_mem
                )

            models_temp = set()
            for m in models:
                models_temp.add(m)
                model_type = type(m).__name__

                if (
                    ("GGUF" in model_type or "ModelPatcher" in model_type)
                    and hasattr(m, "model_patches_to")
                    and not hasattr(m, "model_patches_models")
                ):
                    logger.info(
                        f"[MultiGPU DisTorch V2] {type(m).__name__} missing 'model_patches_models' attribute, using 'model_patches_to' fallback."
                    )
                    target_device = m.load_device
                    logger.debug(
                        f"[MultiGPU DisTorch V2] Target device: {target_device}"
                    )
                    patches = m.model_patches_to(target_device)
                    if patches:
                        logger.debug(
                            f"[MultiGPU DisTorch V2] Found {len(patches)} mm_patch(es) for {type(m).__name__} on device {target_device}"
                        )
                        for mm_patch in patches:
                            logger.debug(
                                f"[MultiGPU DisTorch V2] Registering mm_patch: {type(mm_patch).__name__}"
                            )
                            models_temp.add(mm_patch)
                    continue

                for mm_patch in m.model_patches_models():
                    models_temp.add(mm_patch)
                patches = m.model_patches_to(m.load_device)
                if patches:
                    for mm_patch in patches:
                        models_temp.add(mm_patch)

            models = models_temp

            models_to_load = []

            for x in models:
                loaded_model = mm.LoadedModel(x)
                try:
                    loaded_model_index = current_loaded_models.index(loaded_model)
                except ValueError:
                    loaded_model_index = None

                if loaded_model_index is not None:
                    loaded = current_loaded_models[loaded_model_index]
                    loaded.currently_used = True
                    models_to_load.append(loaded)
                else:
                    if hasattr(x, "model"):
                        logging.info(f"Requested to load {x.model.__class__.__name__}")
                    models_to_load.append(loaded_model)

            for loaded_model in models_to_load:
                to_unload = []
                for i, current_loaded_model in enumerate(current_loaded_models):
                    if loaded_model.model.is_clone(current_loaded_model.model):
                        to_unload = [i] + to_unload
                for i in to_unload:
                    model_to_unload = current_loaded_models.pop(i)
                    model_to_unload.model.detach(unpatch_all=False)
                    model_to_unload.model_finalizer.detach()

            # DisTorch Processing
            total_memory_required = {}
            eject_device = None

            for loaded_model in models_to_load:
                device = loaded_model.device
                base_memory = loaded_model.model_memory_required(device)

                inner_model = loaded_model.model.model

                if hasattr(inner_model, "_distorch_v2_meta"):
                    meta = inner_model._distorch_v2_meta
                    allocation_str = meta["full_allocation"]

                    # Parse allocation string: "expert#compute_device;virtual_vram_gb;donors"
                    parts = allocation_str.split("#")
                    virtual_vram_gb = 0.0
                    has_eject = False

                    if len(parts) > 1:
                        virtual_vram_str = parts[1]
                        virtual_info = virtual_vram_str.split(";")
                        if len(virtual_info) > 1:
                            virtual_vram_gb = float(virtual_info[1])
                        if len(virtual_info) > 2 and virtual_info[2]:
                            has_eject = True

                    if has_eject:
                        eject_device = device
                        logger.mgpu_mm_log(
                            "DisTorch eject_models detected - MAX memory eviction"
                        )

                    virtual_vram_bytes = virtual_vram_gb * (1024**3)
                    adjusted_memory = max(0, base_memory - virtual_vram_bytes)
                    total_memory_required[device] = (
                        total_memory_required.get(device, 0) + adjusted_memory
                    )
                    logger.mgpu_mm_log(
                        f"DisTorch model adjusted {(base_memory - virtual_vram_bytes)/(1024**3):.2f}GB for device {device}"
                    )
                else:
                    # Standard model: use full model size
                    total_memory_required[device] = (
                        total_memory_required.get(device, 0) + base_memory
                    )
                    logger.mgpu_mm_log(
                        f"[LOAD_MODELS_GPU] Standard model {(base_memory)/(1024**3):.2f}GB for device {device}"
                    )

            for device, device_memory in total_memory_required.items():
                if device != torch.device("cpu"):
                    requested_mem = device_memory * 1.1 + extra_mem
                    logger.mgpu_mm_log(
                        f"[FREE_MEMORY_CALL] Device {device}: requesting {requested_mem/(1024**3):.2f}GB = {device_memory/(1024**3):.2f}GB * 1.1 + {extra_mem/(1024**3):.2f}GB inference"
                    )

            multigpu_memory_log("free_memory", "pre")

            for device, device_memory in total_memory_required.items():
                if device != torch.device("cpu"):
                    if device == eject_device:
                        total_device_memory = mm.get_total_memory(device)
                        logger.mgpu_mm_log(
                            f"[LOAD_MODELS_GPU] eject_models=1, is_distorch=1 → using MAX memory ({total_device_memory/(1024**3):.2f}GB) for eviction"
                        )
                        free_memory(total_device_memory, device)
                    else:
                        logger.mgpu_mm_log(
                            f"[LOAD_MODELS_GPU] eject_models=0, using Comfy Core Computed memory ({(device_memory * 1.1 + extra_mem)/(1024**3):.2f}GB) for eviction"
                        )
                        free_memory(device_memory * 1.1 + extra_mem, device)

            multigpu_memory_log("free_memory/minimum_memory_required", "post/pre")

            for device in total_memory_required:
                if device != torch.device("cpu"):
                    free_mem = get_free_memory(device)
                    free_mem_gb = free_mem / (1024**3)
                    min_required_gb = minimum_memory_required / (1024**3)
                    logger.mgpu_mm_log(
                        f"[MIN_MEMORY_CHECK] Device {device}: free={free_mem_gb:.2f}GB, required={min_required_gb:.2f}GB, will_evict={free_mem < minimum_memory_required}"
                    )

                    if free_mem < minimum_memory_required:
                        models_l = free_memory(minimum_memory_required, device)
                        logger.mgpu_mm_log(
                            f"[EVICTION] Device {device}: unloaded {len(models_l)} models due to insufficient memory"
                        )
                        logging.info(f"{len(models_l)} models unloaded.")

            multigpu_memory_log("minimum_memory_required", "post")

            for loaded_model in models_to_load:
                model = loaded_model.model
                torch_dev = model.load_device
                if is_device_cpu(torch_dev):
                    vram_set_state = VRAMState.DISABLED
                else:
                    vram_set_state = vram_state
                lowvram_model_memory = 0
                if (
                    lowvram_available
                    and vram_set_state in (VRAMState.LOW_VRAM, VRAMState.NORMAL_VRAM)
                    and not force_full_load
                ):
                    loaded_memory = loaded_model.model_loaded_memory()
                    current_free_mem = get_free_memory(torch_dev) + loaded_memory

                    lowvram_model_memory = max(
                        128 * 1024 * 1024,
                        (current_free_mem - minimum_memory_required),
                        min(
                            current_free_mem * MIN_WEIGHT_MEMORY_RATIO,
                            current_free_mem - minimum_inference_memory(),
                        ),
                    )
                    lowvram_model_memory = max(
                        0.1, lowvram_model_memory - loaded_memory
                    )

                if vram_set_state == VRAMState.NO_VRAM:
                    lowvram_model_memory = 0.1

                loaded_model.model_load(
                    lowvram_model_memory, force_patch_weights=force_patch_weights
                )
                current_loaded_models.insert(0, loaded_model)

        # Replace the module function
        mm.load_models_gpu = patched_load_models_gpu

        original_partially_load = comfy.model_patcher.ModelPatcher.partially_load

        def new_partially_load(
            self,
            device_to,
            extra_memory=0,
            full_load=False,
            force_patch_weights=False,
            **kwargs,
        ):
            """Override to use direct model annotation for allocation"""

            mp_id = id(self)
            inner_model = self.model
            inner_model_id = id(inner_model)

            if not hasattr(inner_model, "_distorch_v2_meta"):
                logger.debug(
                    f"[DISTORCH_SKIP] ModelPatcher=0x{mp_id:x} inner_model=0x{inner_model_id:x} type={type(inner_model).__name__} - no metadata, using standard loading"
                )
                result = original_partially_load(
                    self, device_to, extra_memory, force_patch_weights
                )
                if hasattr(self, "_distorch_block_assignments"):
                    del self._distorch_block_assignments
                return result

            allocations = inner_model._distorch_v2_meta["full_allocation"]
            donor_gemm_execution_mode = inner_model._distorch_v2_meta.get(
                "donor_gemm_execution_mode", "disabled"
            )

            if not hasattr(self.model, "_distorch_high_precision_loras"):
                self.model._distorch_high_precision_loras = True

            if not hasattr(self.model, "current_weight_patches_uuid"):
                self.model.current_weight_patches_uuid = None

            unpatch_weights = self.model.current_weight_patches_uuid is not None and (
                self.model.current_weight_patches_uuid != self.patches_uuid
                or force_patch_weights
            )

            if unpatch_weights:
                logger.debug(
                    "[MultiGPU DisTorch V2] Patches changed or forced. Unpatching model."
                )
                self.unpatch_model(self.offload_device, unpatch_weights=True)

            self.patch_model(load_weights=False)

            mem_counter = 0

            is_clip_model = getattr(self, "is_clip", False)
            ## TODO - I do not believe this code is needed and needs to be flagged for proof it is needed
            # Check for valid cache
            allocations_match = (
                hasattr(self, "_distorch_last_allocations")
                and self._distorch_last_allocations == allocations
            )
            compute_device_matches = hasattr(
                self, "_distorch_last_compute_device"
            ) and self._distorch_last_compute_device == str(torch.device(device_to))
            cache_exists = hasattr(self, "_distorch_cached_assignments")

            if (
                cache_exists
                and allocations_match
                and compute_device_matches
                and not unpatch_weights
                and not force_patch_weights
            ):
                device_assignments = self._distorch_cached_assignments
                logger.debug(
                    f"[MultiGPU DisTorch V2] Reusing cached analysis for {type(inner_model).__name__}"
                )
            else:
                device_assignments = analyze_safetensor_loading(
                    self,
                    allocations,
                    is_clip=is_clip_model,
                    compute_device=device_to,
                )  ## This should be the only required line - that is how it worked previous release so if it doesn't it is Comfy changes
                self._distorch_cached_assignments = device_assignments
                self._distorch_last_allocations = allocations
                self._distorch_last_compute_device = str(torch.device(device_to))

            model_original_dtype = comfy.utils.weight_dtype(self.model.state_dict())
            high_precision_loras = getattr(
                self.model, "_distorch_high_precision_loras", True
            )
            # DisTorch2 VAE loaders patch the VAE's first_stage_model directly. A VAE's
            # decode tiles already bound its activations, so it runs on the compute GPU
            # with its weights where the allocation put them instead of in mixed mode.
            is_vae = (
                not is_clip_model
                and not hasattr(self.model, "diffusion_model")
                and hasattr(self.model, "decode")
            )
            mixed_donor_device = None
            mixed_int8 = donor_gemm_execution_mode == "mixed_int8"
            int8_layers = 0
            schedule = None
            if donor_gemm_execution_mode in ("mixed", "mixed_int8") and not is_vae:
                mixed_donor_device = select_mixed_donor_device(
                    self.model,
                    device_assignments["block_assignments"],
                    device_to,
                )
            # Mixed mode keeps activations on the donor, so weights stored anywhere
            # else are cast to it at runtime.
            activation_device = mixed_donor_device or device_to
            if mixed_donor_device is not None:
                logger.info(
                    "[MultiGPU DisTorch V2] Mixed mode: %s holds activations; weights stay in system RAM or memory-mapped on disk",
                    mixed_donor_device,
                )
                schedule = _MixedSchedule(device_to, mixed_donor_device)
                if mixed_int8:
                    int8_dtype = self.model.get_dtype_inference()

            def prepare_int8(module_object, module_name):
                keys = [f"{module_name}._mgpu_int8_weight", f"{module_name}._mgpu_int8_bias"]
                if getattr(module_object, "_mgpu_int8_key", None) != self.patches_uuid:
                    # New patches replace the int8 weights; release the old pins first.
                    for key in keys:
                        self.unpin_weight(key)
                _prepare_mixed_int8_weight(
                    module_object,
                    module_name,
                    self.patches,
                    int8_dtype,
                    mixed_donor_device,
                    self.patches_uuid,
                )
                # Pinned, the weight copies to the compute GPU while the previous unit runs.
                # Also re-pins weights reused after unpatch_model unpinned them.
                for key in keys:
                    self.pin_weight_to_device(key)

            # Use standard ComfyUI load list - the device comparison fix ensures we don't crash
            loading = self._load_list()
            loading.sort(reverse=True)
            for item in loading:
                module_size, module_name, module_object, params = unpack_load_item(item)
                if (
                    not unpatch_weights
                    and hasattr(module_object, "comfy_patched_weights")
                    and module_object.comfy_patched_weights is True
                ):
                    block_target_device = device_assignments["block_assignments"].get(
                        module_name, device_to
                    )
                    int8_layer = (
                        mixed_donor_device is not None
                        and mixed_int8
                        and _is_mixed_int8_linear(module_object)
                    )
                    if int8_layer:
                        block_target_device = "cpu"
                    elif mixed_donor_device is not None:
                        block_target_device = _mixed_weight_device(
                            module_object, mixed_donor_device
                        )
                    current_module_device = None
                    try:
                        if any(
                            p.numel() > 0
                            for p in module_object.parameters(recurse=False)
                        ):
                            current_module_device = next(
                                module_object.parameters(recurse=False)
                            ).device
                    except StopIteration:
                        pass

                    if current_module_device is not None and str(
                        current_module_device
                    ) != str(block_target_device):
                        logger.debug(
                            f"[MultiGPU DisTorch V2] Moving already patched {module_name} to {block_target_device}"
                        )
                        module_object.to(block_target_device)

                    if mixed_donor_device is not None:
                        configure_mixed_gemm(
                            module_object, device_to, mixed_donor_device, int8_layer, schedule
                        )
                        if int8_layer:
                            prepare_int8(module_object, module_name)
                            int8_layers += 1
                    else:
                        configure_hip_donor_gemm_offload(
                            module_object,
                            device_to,
                            block_target_device,
                            donor_gemm_execution_mode,
                        )

                    mem_counter += module_size
                    continue

                block_target_device = device_assignments["block_assignments"].get(
                    module_name, device_to
                )
                int8_layer = (
                    mixed_donor_device is not None
                    and mixed_int8
                    and _is_mixed_int8_linear(module_object)
                )
                if int8_layer:
                    # The original stays where it was loaded (a memory-mapped GGUF
                    # stays on disk); the load-time int8 weight is kept in RAM.
                    block_target_device = "cpu"
                elif mixed_donor_device is not None:
                    block_target_device = _mixed_weight_device(
                        module_object, mixed_donor_device
                    )

                # Move directly to the assigned device. Staging donor weights on the
                # compute GPU defeats offload and can OOM before the forward wrapper.
                module_object.to(block_target_device)

                # Step 2: Apply LoRa patches on the assigned device.
                weight_key = f"{module_name}.weight"
                bias_key = f"{module_name}.bias"
                # int8 layers merge their patches into the int8 weight instead.
                with _current_device(block_target_device):
                    if weight_key in self.patches and not int8_layer:
                        self.patch_weight_to_device(
                            weight_key, device_to=block_target_device
                        )
                    if bias_key in self.patches and not int8_layer:
                        self.patch_weight_to_device(
                            bias_key, device_to=block_target_device
                        )
                if weight_key in self.weight_wrapper_patches:
                    module_object.weight_function.extend(
                        self.weight_wrapper_patches[weight_key]
                    )

                if bias_key in self.weight_wrapper_patches:
                    module_object.bias_function.extend(
                        self.weight_wrapper_patches[bias_key]
                    )

                # Step 3: FP8 casting for CPU storage (if enabled)
                has_patches = weight_key in self.patches or bias_key in self.patches

                if (
                    not high_precision_loras
                    and not int8_layer
                    and block_target_device == "cpu"
                    and has_patches
                    and model_original_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
                ):
                    for param_name, param in module_object.named_parameters():
                        if param.dtype.is_floating_point:
                            cast_data = comfy.float.stochastic_rounding(
                                param.data, torch.float8_e4m3fn
                            )
                            new_param = torch.nn.Parameter(
                                cast_data.to(torch.float8_e4m3fn)
                            )
                            new_param.requires_grad = param.requires_grad
                            setattr(module_object, param_name, new_param)
                            logger.debug(
                                f"[MultiGPU DisTorch V2] Cast {module_name}.{param_name} to FP8 for CPU storage"
                            )

                # Step 4: Enable runtime weight casting for offloaded modules.
                if str(block_target_device) != str(activation_device):
                    # Only donor GEMMs (all) read host weights directly; everywhere
                    # else pinning would just copy a memory-mapped GGUF into RAM.
                    if (
                        block_target_device == "cpu"
                        and donor_gemm_execution_mode == "all"
                        and pin_hip_offloaded_weight(module_object)
                    ):
                        logger.debug(
                            f"[MultiGPU DisTorch V2] Pinned {module_name} weight for direct HIP host access"
                        )
                    module_object.comfy_cast_weights = True

                if mixed_donor_device is not None:
                    configure_mixed_gemm(
                        module_object, device_to, mixed_donor_device, int8_layer, schedule
                    )
                    if int8_layer:
                        prepare_int8(module_object, module_name)
                        int8_layers += 1
                elif configure_hip_donor_gemm_offload(
                    module_object,
                    device_to,
                    block_target_device,
                    donor_gemm_execution_mode,
                ):
                    logger.debug(
                        f"[MultiGPU DisTorch V2] Running {module_name} GEMM on donor {block_target_device}"
                    )

                # Mark as patched and update memory counter
                module_object.comfy_patched_weights = True
                mem_counter += module_size

            if int8_layers:
                logger.info(
                    "[MultiGPU DisTorch V2] mixed_int8: %d linears stored as int8 in system RAM; %s holds activations",
                    int8_layers,
                    mixed_donor_device,
                )
            if mixed_donor_device is not None:
                configure_mixed_execution(
                    self.model.diffusion_model, mixed_donor_device, device_to, schedule
                )

            self.model.current_weight_patches_uuid = self.patches_uuid

            self.model.device = device_to

            logger.info("[MultiGPU DisTorch V2] DisTorch loading completed.")
            logger.info(
                f"[MultiGPU DisTorch V2] Total memory: {mem_counter / (1024 * 1024):.2f}MB"
            )

            return 0

        comfy.model_patcher.ModelPatcher.partially_load = new_partially_load
        comfy.model_patcher.ModelPatcher._distorch_patched = True
        logger.info(
            "[MultiGPU Core Patching] Successfully patched ModelPatcher.partially_load"
        )


def _extract_clip_head_blocks(raw_block_list, compute_device):
    """Identify and pre-assign CLIP head blocks to compute device returning head_blocks, distributable_blocks, block_assignments, and head_memory."""
    head_keywords = ["embed", "wte", "wpe", "token_embedding", "position_embedding"]
    head_blocks = []
    distributable_blocks = []
    head_memory = 0
    block_assignments = {}

    block_assignments = {}

    for item in raw_block_list:
        module_size, module_name, module_object, params = unpack_load_item(item)
        # A GGUF-quantized embedding is dequantized in full on every call, so keeping
        # it resident saves only its transfer while holding compute VRAM; it follows
        # the allocation like any other layer.
        quantized = hasattr(getattr(module_object, "weight", None), "tensor_type")
        if not quantized and any(kw in module_name.lower() for kw in head_keywords):
            head_blocks.append((module_size, module_name, module_object, params))
            block_assignments[module_name] = compute_device
            head_memory += module_size
        else:
            distributable_blocks.append(
                (module_size, module_name, module_object, params)
            )

    return head_blocks, distributable_blocks, block_assignments, head_memory


def analyze_safetensor_loading(
    model_patcher, allocations_string, is_clip=False, compute_device=None
):
    """
    Analyze and distribute safetensor model blocks across devices.
    Supports CLIP head preservation when is_clip=True.
    """
    DEVICE_RATIOS_DISTORCH = {}
    device_table = {}
    distorch_alloc = ""
    virtual_vram_str = ""
    if "#" in allocations_string:
        distorch_alloc, virtual_vram_str = allocations_string.split("#", 1)
    else:
        distorch_alloc = allocations_string

    distorch_alloc = distorch_alloc.strip()
    if distorch_alloc and not any(
        "," in allocation for allocation in distorch_alloc.split(";")
    ):
        logger.warning(
            "[MultiGPU DisTorch V2] Ignoring invalid expert allocation %r; "
            "using virtual-VRAM allocation instead.",
            distorch_alloc,
        )
        distorch_alloc = ""

    metadata_compute_device = (
        virtual_vram_str.split(";")[0] if virtual_vram_str else "cuda:0"
    )
    if compute_device is None:
        compute_device = metadata_compute_device
    else:
        compute_device = str(torch.device(compute_device))
        if compute_device != metadata_compute_device:
            logger.warning(
                "[MultiGPU DisTorch V2] Allocation metadata names %s as the compute "
                "device, but inference is running on %s; using the runtime device.",
                metadata_compute_device,
                compute_device,
            )
    logger.debug(f"[MultiGPU DisTorch V2] Compute Device: {compute_device}")

    if not distorch_alloc:
        if virtual_vram_str:
            virtual_vram_parts = virtual_vram_str.split(";", 1)
            virtual_vram_str = ";".join([compute_device, *virtual_vram_parts[1:]])
        distorch_alloc = calculate_safetensor_vvram_allocation(
            model_patcher, virtual_vram_str
        )

    elif any(c in distorch_alloc.lower() for c in ["g", "m", "k", "b"]):
        distorch_alloc = calculate_fraction_from_byte_expert_string(
            model_patcher, distorch_alloc
        )
    elif "%" in distorch_alloc:
        distorch_alloc = calculate_fraction_from_ratio_expert_string(
            model_patcher, distorch_alloc
        )

    all_devices = get_device_list()
    present_devices = {
        item.split(",")[0] for item in distorch_alloc.split(";") if "," in item
    }
    for device in all_devices:
        if device not in present_devices:
            distorch_alloc += f";{device},0.0"

    eq_line = "=" * 50
    dash_line = "-" * 50
    fmt_assign = "{:<18}{:>7}{:>14}{:>10}"

    logger.info(eq_line)
    logger.info(f"[MultiGPU DisTorch V2] Final Allocation String:\n{distorch_alloc}")

    for allocation in distorch_alloc.split(";"):
        if "," not in allocation:
            continue
        dev_name, fraction = allocation.split(",")
        fraction = float(fraction)
        total_mem_bytes = mm.get_total_memory(torch.device(dev_name))
        alloc_gb = (total_mem_bytes * fraction) / (1024**3)
        DEVICE_RATIOS_DISTORCH[dev_name] = alloc_gb
        device_table[dev_name] = {
            "fraction": fraction,
            "total_gb": total_mem_bytes / (1024**3),
            "alloc_gb": alloc_gb,
        }

    logger.info(eq_line)
    logger.info("    DisTorch2 Model Device Allocations")
    logger.info(eq_line)

    fmt_rosetta = "{:<8}{:>9}{:>9}{:>11}{:>10}"
    logger.info(fmt_rosetta.format("Device", "VRAM GB", "Dev %", "Model GB", "Dist %"))
    logger.info(dash_line)

    sorted_devices = sorted(device_table.keys(), key=lambda d: (d == "cpu", d))

    total_allocated_model_bytes = sum(
        d["alloc_gb"] * (1024**3) for d in device_table.values()
    )

    for dev in sorted_devices:
        total_dev_gb = device_table[dev]["total_gb"]
        alloc_fraction = device_table[dev]["fraction"]
        alloc_gb = device_table[dev]["alloc_gb"]

        dist_ratio_percent = (
            (alloc_gb * (1024**3) / total_allocated_model_bytes) * 100
            if total_allocated_model_bytes > 0
            else 0
        )

        logger.info(
            fmt_rosetta.format(
                dev,
                f"{total_dev_gb:.2f}",
                f"{alloc_fraction*100:.1f}%",
                f"{alloc_gb:.2f}",
                f"{dist_ratio_percent:.1f}%",
            )
        )

    logger.info(dash_line)

    block_summary = {}
    block_list = []
    memory_by_type = defaultdict(int)
    total_memory = 0

    raw_block_list = model_patcher._load_list()
    total_memory = sum(unpack_load_item(x)[0] for x in raw_block_list)

    MIN_BLOCK_THRESHOLD = total_memory * 0.0001
    logger.debug(f"[MultiGPU DisTorch V2] Total model memory: {total_memory} bytes")
    logger.debug(
        f"[MultiGPU DisTorch V2] Tiny block threshold (0.01%): {MIN_BLOCK_THRESHOLD} bytes"
    )

    # CLIP-specific: Extract head blocks and get pre-assignments
    head_memory = 0
    block_assignments = {}
    if is_clip:
        head_blocks, distributable_raw, block_assignments, head_memory = (
            _extract_clip_head_blocks(raw_block_list, compute_device)
        )
        logger.info(
            f"[MultiGPU DisTorch V2 CLIP] Preserving {len(head_blocks)} head layer(s) ({head_memory/(1024**2):.2f} MB) on compute device: {compute_device}"
        )
    else:
        distributable_raw = raw_block_list

    # Build all_blocks list for summary (using full raw_block_list)
    all_blocks = []
    for item in raw_block_list:
        module_size, module_name, module_object, params = unpack_load_item(item)
        block_type = type(module_object).__name__
        # Populate summary dictionaries
        block_summary[block_type] = block_summary.get(block_type, 0) + 1
        memory_by_type[block_type] += module_size
        all_blocks.append((module_name, module_object, block_type, module_size))

    # Use distributable blocks for actual allocation (for CLIP, this excludes heads)
    distributable_all_blocks = []
    for item in distributable_raw:
        module_size, module_name, module_object, params = unpack_load_item(item)
        distributable_all_blocks.append(
            (module_name, module_object, type(module_object).__name__, module_size)
        )

    block_list = [
        block
        for block in distributable_all_blocks
        if block[3] >= MIN_BLOCK_THRESHOLD
        and hasattr(block[1], "bias")
        and hasattr(block[1], "comfy_cast_weights")
    ]
    tiny_block_list = [b for b in distributable_all_blocks if b not in block_list]

    logger.debug(f"[MultiGPU DisTorch V2] Total blocks: {len(all_blocks)}")
    logger.debug(f"[MultiGPU DisTorch V2] Distributable blocks: {len(block_list)}")
    logger.debug(f"[MultiGPU DisTorch V2] Tiny blocks (<0.01%): {len(tiny_block_list)}")

    logger.info("    DisTorch2 Model Layer Distribution")
    logger.info(dash_line)
    fmt_layer = "{:<18}{:>7}{:>14}{:>10}"
    logger.info(fmt_layer.format("Layer Type", "Layers", "Memory (MB)", "% Total"))
    logger.info(dash_line)

    for layer_type, count in block_summary.items():
        mem_mb = memory_by_type[layer_type] / (1024 * 1024)
        mem_percent = (
            (memory_by_type[layer_type] / total_memory) * 100 if total_memory > 0 else 0
        )
        logger.info(
            fmt_layer.format(
                layer_type[:18], str(count), f"{mem_mb:.2f}", f"{mem_percent:.1f}%"
            )
        )

    logger.info(dash_line)

    # Distribute blocks sequentially from the tail of the model

    device_assignments = {device: [] for device in DEVICE_RATIOS_DISTORCH}
    # Create a memory quota for each donor device based on its calculated allocation.
    donor_devices = list(sorted_devices)
    donor_quotas = {
        dev: device_table[dev]["alloc_gb"] * (1024**3) for dev in donor_devices
    }

    # CLIP-specific: Adjust compute_device quota to account for locked head blocks
    if is_clip and compute_device in donor_quotas and head_memory > 0:
        donor_quotas[compute_device] = max(
            0, donor_quotas[compute_device] - head_memory
        )
        logger.debug(
            f"[MultiGPU DisTorch V2 CLIP] Adjusted {compute_device} quota by -{head_memory/(1024**2):.2f} MB for head preservation"
        )

    # Iterate from the TAIL of the model, assigning blocks to donors until their quotas are filled.
    for block_name, module, block_type, block_memory in reversed(block_list):
        assigned_to_donor = False
        for donor in donor_devices:
            if donor_quotas[donor] >= block_memory:
                block_assignments[block_name] = donor
                donor_quotas[donor] -= block_memory
                assigned_to_donor = True
                break  # Move to the next block

        if (
            not assigned_to_donor
        ):  # Note - small rounding errors and tensor-fitting on devices make a block occasionally an orphan. We treat orphans the same as tiny_block_list as they are generally small rounding errors
            block_assignments[block_name] = compute_device

    if tiny_block_list:
        for block_name, module, block_type, block_memory in tiny_block_list:
            block_assignments[block_name] = compute_device

    # Populate device_assignments from the final block_assignments
    for block_name, device in block_assignments.items():
        # Find the block in the original list to get all its info
        for b_name, b_module, b_type, b_mem in all_blocks:
            if b_name == block_name:
                device_assignments[device].append((b_name, b_module, b_type, b_mem))
                break

    logger.info("DisTorch2 Model Final Device/Layer Assignments")
    logger.info(dash_line)
    logger.info(fmt_assign.format("Device", "Layers", "Memory (MB)", "% Total"))
    logger.info(dash_line)

    if tiny_block_list:
        tiny_block_memory = sum(b[3] for b in tiny_block_list)
        tiny_mem_mb = tiny_block_memory / (1024 * 1024)
        tiny_mem_percent = (
            (tiny_block_memory / total_memory) * 100 if total_memory > 0 else 0
        )
        device_label = f"{compute_device} (<0.01%)"
        logger.info(
            fmt_assign.format(
                device_label,
                str(len(tiny_block_list)),
                f"{tiny_mem_mb:.2f}",
                f"{tiny_mem_percent:.1f}%",
            )
        )
        logger.debug(
            f"[MultiGPU DisTorch V2] Tiny block memory breakdown: {tiny_block_memory} bytes ({tiny_mem_mb:.2f} MB), which is {tiny_mem_percent:.4f}% of total model memory."
        )

    total_assigned_memory = 0
    device_memories = {}

    for device, blocks in device_assignments.items():
        dist_blocks = [b for b in blocks if b[3] >= MIN_BLOCK_THRESHOLD]
        if not dist_blocks:
            continue

        device_memory = sum(b[3] for b in dist_blocks)
        device_memories[device] = device_memory
        total_assigned_memory += device_memory

    sorted_assignments = sorted(device_memories.keys(), key=lambda d: (d == "cpu", d))

    for dev in sorted_assignments:
        # Get only the distributed blocks for the count
        dist_blocks = [
            b for b in device_assignments[dev] if b[3] >= MIN_BLOCK_THRESHOLD
        ]
        if not dist_blocks:
            continue

        mem_mb = device_memories[dev] / (1024 * 1024)
        mem_percent = (
            (device_memories[dev] / total_memory) * 100 if total_memory > 0 else 0
        )
        logger.info(
            fmt_assign.format(
                dev, str(len(dist_blocks)), f"{mem_mb:.2f}", f"{mem_percent:.1f}%"
            )
        )

    logger.info(dash_line)

    return {
        "device_assignments": device_assignments,
        "block_assignments": block_assignments,
    }


def parse_memory_string(mem_str):
    """Parses a memory string (e.g., '4.0g', '512M') and returns bytes."""
    mem_str = mem_str.strip().lower()
    match = re.match(r"(\d+\.?\d*)\s*([gmkb]?)", mem_str)
    if not match:
        raise ValueError(f"Invalid memory string format: {mem_str}")

    val, unit = match.groups()
    val = float(val)

    if unit == "g":
        return val * (1024**3)
    elif unit == "m":
        return val * (1024**2)
    elif unit == "k":
        return val * 1024
    else:  # b or no unit
        return val


def calculate_fraction_from_byte_expert_string(model_patcher, byte_str):
    """Convert byte allocation string (e.g. 'cuda:1,4gb;cpu,*') to fractional VRAM allocation string respecting device order and byte quotas."""
    raw_block_list = model_patcher._load_list()
    total_model_memory = sum(unpack_load_item(x)[0] for x in raw_block_list)
    remaining_model_bytes = total_model_memory

    # Use a list of tuples to preserve the user-defined order
    parsed_allocations = []
    wildcard_device = "cpu"  # Default wildcard device

    for allocation in byte_str.split(";"):
        if "," not in allocation:
            continue
        dev_name, val_str = allocation.split(",", 1)
        is_wildcard = "*" in val_str

        if is_wildcard:
            wildcard_device = dev_name
            # Don't add wildcard to the priority list yet
        else:
            byte_val = parse_memory_string(val_str)
            parsed_allocations.append({"device": dev_name, "bytes": byte_val})

    final_byte_allocations = defaultdict(int)

    # Process devices with specific byte allocations first, in order
    for alloc in parsed_allocations:
        dev = alloc["device"]
        requested_bytes = alloc["bytes"]

        # Determine the actual bytes to allocate to this device
        bytes_to_assign = min(requested_bytes, remaining_model_bytes)

        if bytes_to_assign > 0:
            final_byte_allocations[dev] = bytes_to_assign
            remaining_model_bytes -= bytes_to_assign
            logger.info(
                f"[MultiGPU DisTorch V2] Assigning {bytes_to_assign / (1024**2):.2f}MB of model to {dev} (requested {requested_bytes / (1024**2):.2f}MB)."
            )

        if remaining_model_bytes <= 0:
            logger.info(
                "[MultiGPU DisTorch V2] All model blocks have been allocated. Subsequent devices in the string will receive no assignment."
            )
            break

    # Assign any leftover model bytes to the wildcard device
    if remaining_model_bytes > 0:
        final_byte_allocations[wildcard_device] += remaining_model_bytes
        logger.info(
            f"[MultiGPU DisTorch V2] Assigning remaining {remaining_model_bytes / (1024**2):.2f}MB of model to wildcard device '{wildcard_device}'."
        )

    # Convert the final byte allocations to VRAM fractions
    allocation_parts = []
    for dev, bytes_alloc in final_byte_allocations.items():
        total_device_vram = mm.get_total_memory(torch.device(dev))
        if total_device_vram > 0:
            fraction = bytes_alloc / total_device_vram
            allocation_parts.append(f"{dev},{fraction:.4f}")

    allocations_string = ";".join(allocation_parts)

    return allocations_string


def calculate_fraction_from_ratio_expert_string(model_patcher, ratio_str):
    """Convert ratio allocation string (e.g. 'cuda:0,25%;cpu,75%') describing model split to fractional VRAM allocation string."""
    raw_block_list = model_patcher._load_list()
    total_model_memory = sum(unpack_load_item(x)[0] for x in raw_block_list)

    raw_ratios = {}
    for allocation in ratio_str.split(";"):
        if "," not in allocation:
            continue
        dev_name, val_str = allocation.split(",", 1)
        # Assumes the value is a unitless ratio number, ignores '%' for simplicity.
        value = float(val_str.replace("%", "").strip())
        raw_ratios[dev_name] = value

    total_ratio_parts = sum(raw_ratios.values())
    allocation_parts = []

    for dev, ratio_val in raw_ratios.items():
        bytes_of_model_for_device = (ratio_val / total_ratio_parts) * total_model_memory

        total_vram_of_device = mm.get_total_memory(torch.device(dev))

        if total_vram_of_device > 0:
            required_fraction = bytes_of_model_for_device / total_vram_of_device
            allocation_parts.append(f"{dev},{required_fraction:.4f}")

    ratio_values = [str(v) for v in raw_ratios.values()]
    ratio_string = ":".join(ratio_values)

    normalized_pcts = [(v / total_ratio_parts) * 100 for v in raw_ratios.values()]

    put_parts = []
    for i, dev_name in enumerate(raw_ratios.keys()):
        put_parts.append(f"{int(normalized_pcts[i])}% on {dev_name}")

    if len(put_parts) == 1:
        put_part = put_parts[0]
    elif len(put_parts) == 2:
        put_part = f"{put_parts[0]} and {put_parts[1]}"
    else:
        put_part = ", ".join(put_parts[:-1]) + f", and {put_parts[-1]}"

    logger.info(
        f"[MultiGPU DisTorch V2] Ratio(%) Mode - {ratio_str} -> {ratio_string} ratio, put {put_part}"
    )

    allocations_string = ";".join(allocation_parts)

    return allocations_string


def calculate_safetensor_vvram_allocation(model_patcher, virtual_vram_str):
    """Calculate virtual VRAM allocation string for distributed safetensor loading"""
    recipient_device, vram_amount, donors = virtual_vram_str.split(";")
    virtual_vram_gb = float(vram_amount)

    eq_line = "=" * 47
    dash_line = "-" * 47
    fmt_assign = "{:<8} {:<6} {:>11} {:>9} {:>9}"

    logger.info(eq_line)
    logger.info("    DisTorch2 Model Virtual VRAM Analysis")
    logger.info(eq_line)
    logger.info(
        fmt_assign.format("Object", "Role", "Original(GB)", "Total(GB)", "Virt(GB)")
    )
    logger.info(dash_line)

    # Calculate recipient VRAM
    recipient_vram = mm.get_total_memory(torch.device(recipient_device)) / (1024**3)
    recipient_virtual = recipient_vram + virtual_vram_gb

    logger.info(
        fmt_assign.format(
            recipient_device,
            "recip",
            f"{recipient_vram:.2f}GB",
            f"{recipient_virtual:.2f}GB",
            f"+{virtual_vram_gb:.2f}GB",
        )
    )

    # Handle donor devices
    ram_donors = list(donors.split(","))
    remaining_vram_needed = virtual_vram_gb

    donor_device_info = {}
    donor_allocations = {}

    for donor in ram_donors:
        donor_vram = mm.get_total_memory(torch.device(donor)) / (1024**3)
        max_donor_capacity = donor_vram

        donation = min(remaining_vram_needed, max_donor_capacity)
        donor_virtual = donor_vram - donation
        remaining_vram_needed -= donation
        donor_allocations[donor] = donation

        donor_device_info[donor] = (donor_vram, donor_virtual)
        logger.info(
            fmt_assign.format(
                donor,
                "donor",
                f"{donor_vram:.2f}GB",
                f"{donor_virtual:.2f}GB",
                f"-{donation:.2f}GB",
            )
        )

    logger.info(dash_line)

    # Calculate model size
    # Stored bytes, as the block assignment measures them: a quantized weight's
    # element_size() is its logical dtype, not its int8 or fp8 storage.
    total_memory = sum(
        unpack_load_item(item)[0] for item in model_patcher._load_list()
    )

    model_size_gb = total_memory / (1024**3)
    new_model_size_gb = max(0, model_size_gb - virtual_vram_gb)

    logger.info(
        fmt_assign.format(
            "model",
            "model",
            f"{model_size_gb:.2f}GB",
            f"{new_model_size_gb:.2f}GB",
            f"-{virtual_vram_gb:.2f}GB",
        )
    )

    # Warning if model too large
    if model_size_gb > (recipient_vram * 0.9):
        required_offload_gb = model_size_gb - (recipient_vram * 0.9)
        logger.warning(
            f"\n\n[MultiGPU DisTorch V2] Model size ({model_size_gb:.2f}GB) is larger than 90% of available VRAM on: {recipient_device} ({recipient_vram * 0.9:.2f}GB)."
        )
        logger.warning(
            f"[MultiGPU DisTorch V2] To prevent an OOM error, set 'virtual_vram_gb' to at least {required_offload_gb:.2f}.\n\n"
        )

    new_on_recipient = max(0, model_size_gb - virtual_vram_gb)

    # Build allocation string
    allocation_parts = []
    recipient_percent = new_on_recipient / recipient_vram
    allocation_parts.append(f"{recipient_device},{recipient_percent:.4f}")

    for donor in ram_donors:
        donor_vram = donor_device_info[donor][0]
        donor_percent = donor_allocations[donor] / donor_vram
        allocation_parts.append(f"{donor},{donor_percent:.4f}")

    allocations_string = ";".join(allocation_parts)
    return allocations_string
