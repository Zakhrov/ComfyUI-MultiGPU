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
from collections import defaultdict, namedtuple
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes

logger = logging.getLogger("MultiGPU")
import comfy.model_management as mm
import comfy.model_patcher
import comfy.ops
import comfy.sd1_clip
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
# Share of each storage device's free memory left to other models and spilled activations.
_STORAGE_RESERVE_FRACTION = 0.25
# Chunks per mixed GEMM or attention call: with several, each chunk's copies to
# and from the compute GPU overlap the neighbouring chunks' compute.
_PIPELINE_CHUNKS = 8
# Comfy Kitchen's int8 GEMM computes whole 128-row tiles, so a chunk of 520 rows
# costs as much as 640. Chunks are cut to a multiple of this.
_GEMM_ROW_TILE = 128
# Below this many tokens a chunk's GEMM costs little more than its launch, so no
# budget makes chunks smaller: a shortfall surfaces as an out-of-memory retry
# instead of tens of thousands of one-token GEMMs.
_MIN_CHUNK_TOKENS = 512
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
# An op on at most this many bytes of spilled tensors hops whole, and a smaller result is never spilled.
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
    if name in ("aten::native_layer_norm", "aten::rms_norm", "aten::_fused_rms_norm"):
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
    if name.startswith("comfy_kitchen::rms_rope"):
        # [B, S, H, D] q and k (x for the "1" ops), normed over D and rotated by a freqs_cis
        # that broadcasts over the first three dims; the scales are per D.
        count = 1 if name.removesuffix("_").endswith("1") else 2
        freqs_cis = leaves[count]
        return [
            (shape[d], [d] * count + [d if freqs_cis.ndim > d and freqs_cis.shape[d] == shape[d] else None]
             + [None] * (len(leaves) - count - 1), d)
            for d in range(3)
            if shape[d] > 1 and all(leaf.shape[d] == shape[d] for leaf in leaves[:count])
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


# Storage pointers of the pinned system-RAM buffers the running forward spilled activations to.
_spilled = set()


def _is_spilled(tensor):
    return tensor.device.type == "cpu" and tensor.untyped_storage().data_ptr() in _spilled


def _spill_buffer(shape, dtype):
    """A pinned system-RAM buffer for an activation the compute GPU has no room for.

    Pinned memory lets the compute GPU's copy engine move it asynchronously, so
    neither the host nor another GPU (an APU would blit it with its shaders) does.
    """
    buffer = torch.empty(shape, dtype=dtype, pin_memory=True)
    _spilled.add(buffer.untyped_storage().data_ptr())
    return buffer


def _activation_empty(shape, dtype, compute_device, reserve=_COMPUTE_GPU_RESERVE_BYTES):
    """An activation buffer on the compute GPU when it fits, else spilled to pinned system RAM."""
    nbytes = math.prod(shape) * dtype.itemsize
    if nbytes <= _compute_budget(compute_device, reserve):
        try:
            return torch.empty(shape, dtype=dtype, device=compute_device)
        except torch.OutOfMemoryError:
            pass
    return _spill_buffer(shape, dtype)


class _StreamedActivations(TorchDispatchMode):
    """Keep a model's activations on the compute GPU, spilling to pinned system RAM when it is full.

    Ops on compute-GPU tensors run there as usual. One whose result would not fit
    runs in slices whose results are copied to a pinned buffer as they complete
    (see _tiled), so the GPU never waits on the host. An op that reads a spilled
    tensor streams it back in slices and keeps its result on the compute GPU when
    that fits. Ops on the model's own CPU tensors run on the CPU as usual.
    """

    def __init__(self, compute_device):
        super().__init__()
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

    def _tiled(self, func, kind, args, kwargs, budget, spill):
        """Run an op on the compute GPU one slice at a time so that only a few slices are ever there.

        Spilled inputs are copied in slice by slice through _pipeline_on_compute.
        A plain op's result goes to a pinned buffer when spill is set and to the
        compute GPU otherwise; a mutating op writes its spilled tensors in place.
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
        # Every input moves, not just spilled ones: a weight cast to a spilled input's device
        # (cast_bias_weight) is on the CPU without being spilled.
        whole = {index: self._to_compute(leaf) for index, (leaf, dim) in enumerate(zip(leaves, dims)) if dim is None}
        written = sum(
            argument.alias_info is not None and argument.alias_info.is_write and "Tensor" in str(argument.type)
            for argument in func._schema.arguments
        )
        outputs = leaves[:written] if mutating else []
        progress = [0]
        compute_stream = torch.cuda.current_stream(self.compute)

        def load(chunk):
            start, stop = chunk
            tiles = []
            for index, (leaf, dim) in enumerate(zip(leaves, dims)):
                if dim is None:
                    tiles.append(whole[index])
                    continue
                tile = leaf.narrow(dim, start, stop - start)
                if tile.device != self.compute:
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
                    # A result the compute GPU has no room for spills rather than raising: an
                    # out-of-memory retry would find only the results allocated before it.
                    output = _spill_buffer(shape, tile.dtype) if spill else _activation_empty(shape, tile.dtype, self.compute)
                    if output.device == self.compute:
                        # Allocated on the copy stream; the compute stream reads it next.
                        output.record_stream(compute_stream)
                    outputs.append(output)
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
                torch.cuda.empty_cache()
        if mutating:
            return args[0] if func._schema.returns else None
        return tuple(outputs) if len(outputs) > 1 else outputs[0]

    def _result_bytes(self, func, leaves):
        """An upper estimate of an op's result size, from its inputs."""
        if torch.Tag.pointwise in func.tags:
            shape = torch.broadcast_shapes(*[leaf.shape for leaf in leaves])
            return math.prod(shape) * max(leaf.element_size() for leaf in leaves)
        if func._schema.name in ("aten::cat", "aten::stack"):
            return sum(leaf.nbytes for leaf in leaves)
        return max(leaf.nbytes for leaf in leaves)

    def _on_compute(self, func, kind, args, kwargs, tensors):
        """Run an op on compute-GPU tensors, spilling its result when the GPU has no room for it."""
        large = kind == "plain" and tensors and max(tensor.nbytes for tensor in tensors) > _STREAM_DIRECT_BYTES
        if large:
            budget = _compute_budget(self.compute)
            if self._result_bytes(func, tensors) > budget and _tile_axes(func, args, kwargs, tensors):
                return self._tiled(func, kind, args, kwargs, budget, spill=True)
        try:
            return func(*args, **kwargs)
        except torch.OutOfMemoryError:
            if not large or not _tile_axes(func, args, kwargs, tensors):
                raise
        # cudaMallocAsync keeps freed blocks that a larger request cannot use until they are released.
        torch.cuda.empty_cache()
        return self._tiled(func, kind, args, kwargs, _compute_budget(self.compute), spill=True)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        kind = self._kind(func)
        tensors = [leaf for leaf in pytree.tree_leaves((args, kwargs)) if isinstance(leaf, torch.Tensor)]
        spilled = [tensor for tensor in tensors if _is_spilled(tensor)]
        if kind == "metadata":
            # Under a dispatch mode aten::alias keeps the input's subclass, which breaks
            # torch.Tensor(subclass) for subclasses whose __init__ takes required kwargs (GGMLTensor).
            if func is torch.ops.aten.alias.default:
                return func(*args, **kwargs).as_subclass(torch.Tensor)
            if spilled and func is torch.ops.aten._local_scalar_dense.default:
                # The host reads a spilled value, whose copy may still be in flight.
                torch.cuda.current_stream(self.compute).synchronize()
            return func(*args, **kwargs)
        target = kwargs.get("device")
        target = None if target is None else self._resolve(target)
        if not spilled:
            if target not in (None, self.compute) or not all(tensor.device == self.compute for tensor in tensors):
                return func(*args, **kwargs)
            return self._on_compute(func, kind, args, kwargs, tensors)

        name = func._schema.name
        if target is not None and target.type == "cpu":
            torch.cuda.current_stream(self.compute).synchronize()
            return func(*args, **kwargs)
        spilled_bytes = sum(tensor.nbytes for tensor in spilled)
        if spilled_bytes > _STREAM_DIRECT_BYTES and name not in ("aten::copy_", "aten::_to_copy"):
            budget = _compute_budget(self.compute)
            if 3 * spilled_bytes > budget:
                # The result stays on the compute GPU when it fits beside the slices.
                result_bytes = self._result_bytes(func, tensors) if kind == "plain" else 0
                spill = result_bytes > budget // 2
                return self._tiled(func, kind, args, kwargs, budget - (0 if spill else result_bytes), spill)

        if name == "aten::copy_":
            dst, src = args[0], args[1]
            if _is_spilled(src) or src.dtype != dst.dtype:
                src = self._to_compute(src, dst.dtype)
            return func(dst, src, *args[2:], **kwargs)
        if name == "aten::_to_copy":
            src = args[0]
            dtype = kwargs.get("dtype")
            if (dtype is None or dtype == src.dtype) and target is not None and target != src.device:
                return func(*args, **kwargs)
            return self._to_compute(src, dtype)

        moved = {}

        def to_compute(tensor):
            if tensor.device == self.compute:
                return tensor
            if id(tensor) not in moved:
                if kind == "overwrite" and tensor is args[0]:
                    moved[id(tensor)] = torch.empty_like(tensor, device=self.compute)
                else:
                    moved[id(tensor)] = self._to_compute(tensor)
            return moved[id(tensor)]

        compute_args, compute_kwargs = pytree.tree_map_only(torch.Tensor, to_compute, (args, kwargs))
        result = func(*compute_args, **compute_kwargs)
        if kind == "plain":
            return result
        originals = {}
        for index, argument in enumerate(func._schema.arguments):
            if argument.alias_info is None or not argument.alias_info.is_write:
                continue
            value = args[index] if index < len(args) else kwargs.get(argument.name)
            for tensor in pytree.tree_leaves(value):
                if isinstance(tensor, torch.Tensor) and id(tensor) in moved:
                    tensor.copy_(moved[id(tensor)], non_blocking=True)
                    originals[id(moved[id(tensor)])] = tensor
        return pytree.tree_map_only(torch.Tensor, lambda tensor: originals.get(id(tensor), tensor), result)


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


def _is_eligible_streamed_gemm(module):
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


def _compute_budget(compute_device, reserve=_COMPUTE_GPU_RESERVE_BYTES):
    """Bytes of compute-GPU VRAM one GEMM or attention call may use for tiles.

    Measured without synchronizing, so the GPU keeps its queue. Memory the
    allocator caches counts as free; the cudaMallocAsync pool (--cuda-malloc)
    cannot serve a larger request from it, so callers that run out of memory
    release it with torch.cuda.empty_cache() and retry smaller.
    """
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
    its result there and store(chunk, result) puts the result in place. Copies go
    on a side stream while the compute stream works on the neighbouring chunks.
    At most two chunks are in flight, so the compute GPU holds a few chunk
    buffers rather than the whole activation. during() runs once the first
    chunk's compute is queued, so work it starts overlaps the rest. The streams
    wait for each other on the GPU, never on the host: loads see what the compute
    stream wrote before, and later work on it sees every stored chunk.
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
    copy_stream.wait_stream(compute_stream)

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
    compute_stream.wait_stream(copy_stream)


def _token_chunks(tokens, token_bytes, budget, streamed=True):
    """Token slices for _pipeline_on_compute, three of which fit the budget.

    Three chunks sit on the compute GPU at once: one loading, one computing and
    one being stored. Chunks are at least _MIN_CHUNK_TOKENS long, and chunks
    that are copied (streamed) number at least _PIPELINE_CHUNKS, so that their
    copies overlap the compute.
    """
    per_chunk = budget // (3 * token_bytes)
    if streamed:
        per_chunk = min(per_chunk, -(-tokens // _PIPELINE_CHUNKS))
    per_chunk = max(_MIN_CHUNK_TOKENS, per_chunk)
    if per_chunk > _GEMM_ROW_TILE:
        per_chunk -= per_chunk % _GEMM_ROW_TILE
    return [slice(start, min(start + per_chunk, tokens)) for start in range(0, tokens, per_chunk)]


def _gemm_weight_dtype(dtype, compute_device):
    """The dtype a float GEMM weight is prepared in for activations of dtype.

    float16 where dense_linear runs Comfy Kitchen's packed-fp16 GEMM, which would
    otherwise cast the weight to float16 on every call; the staged weight then
    also takes half the compute GPU's memory of float32. dtype elsewhere.
    """
    if (
        getattr(comfy.ops, "_CK_PACKED", False)
        and dtype in (torch.float32, torch.bfloat16)
        and comfy.ops.quant_ops.ck.fp16_packed_linear_is_accelerated(torch.device(compute_device))
    ):
        return torch.float16
    return dtype


def _linear_on_compute(input_chunk, weight, bias, lora_down, lora_up):
    if weight.dtype == torch.uint8:
        # a GGUF weight staged as its raw blocks (see _gguf_native), Q8_0's 34 bytes or
        # Q4_K's 18 bytes per 32 elements
        op = comfy.ops.quant_ops.ck.q8_0_linear if weight.shape[1] * 32 == input_chunk.shape[-1] * 34 \
            else comfy.ops.quant_ops.ck.q4_k_linear
        output_chunk = op(input_chunk, weight, weight.shape[0], bias)
    elif weight.dtype != input_chunk.dtype and not comfy.ops._packed_linear_wanted(input_chunk, weight):
        # A float16 weight (see _gemm_weight_dtype) on a GEMM too small for the packed kernel.
        output_chunk = comfy.ops.dense_linear(input_chunk, weight.to(input_chunk.dtype), bias)
    else:
        output_chunk = comfy.ops.dense_linear(input_chunk, weight, bias)
    if lora_down is not None:
        output_chunk.addmm_(input_chunk @ lora_down.t(), lora_up.t())
    return output_chunk


def _run_token_chunks(flat_input, out_features, run, token_bytes, budget, compute_device, during=None):
    """Run a token-wise run(input_chunk) over flat_input on the compute GPU; returns its result.

    token_bytes bounds what one token needs there: its input, result and
    temporaries. When the whole input does, it runs in one call. Otherwise the
    result is an activation on the compute GPU, or spilled to pinned system RAM
    when it does not fit (see _activation_empty), and is filled in token chunks;
    only a spilled input is copied in, chunk by chunk. On running out of memory
    the chunks are halved and the run repeated, which is safe because every
    chunk writes only its own rows. during() runs once the first chunk is queued.
    """
    tokens = flat_input.shape[0]
    compute_device = torch.device(compute_device)
    with _current_device(compute_device):
        if flat_input.device == compute_device and tokens * token_bytes <= budget:
            try:
                output = run(flat_input)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
            else:
                if during is not None:
                    during()
                return output
        # The output stays on the compute GPU only if the chunks still fit beside it.
        output_bytes = tokens * out_features * flat_input.element_size()
        if output_bytes + 3 * _MIN_CHUNK_TOKENS * token_bytes <= budget:
            output = _activation_empty((tokens, out_features), flat_input.dtype, compute_device)
        else:
            output = _spill_buffer((tokens, out_features), flat_input.dtype)
        if output.device == compute_device:
            budget -= output.nbytes
        streamed = flat_input.device != compute_device or output.device != compute_device
        while True:
            chunks = _token_chunks(tokens, token_bytes, budget, streamed)
            try:
                _pipeline_on_compute(
                    chunks,
                    compute_device,
                    lambda chunk: flat_input[chunk].to(device=compute_device, non_blocking=True),
                    run,
                    lambda chunk, result: output[chunk].copy_(result, non_blocking=True),
                    during,
                )
                return output
            except torch.OutOfMemoryError:
                # Chunks at the floor already ran after a cache release.
                if chunks[0].stop <= _MIN_CHUNK_TOKENS and budget <= 0:
                    raise
            during = None
            torch.cuda.empty_cache()
            budget = budget // 2 if budget > 3 * _MIN_CHUNK_TOKENS * token_bytes else 0


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

    Only token chunks, weight tiles and their GEMM results visit the compute
    GPU when the input is spilled; the output is an activation (see
    _activation_empty). lora holds (down, up, scale) patches that are added as
    low-rank GEMMs instead of being merged into the weight.
    """
    flat_input = input_tensor.reshape(-1, input_tensor.shape[-1])
    tokens, in_features = flat_input.shape
    out_features = weight.shape[0]
    dtype = input_tensor.dtype
    lora_down, lora_up = _lora_on_compute(lora, compute_device, dtype)
    lora_rank = 0 if lora_down is None else lora_down.shape[0]
    row_bytes = in_features * weight.element_size()
    budget = _compute_budget(compute_device)

    if isinstance(bias, torch.Tensor):
        bias = bias.to(device=compute_device, non_blocking=True)

    # Keep the whole weight on the compute GPU when it takes at most half the
    # budget; otherwise stream output-channel tiles for every token chunk.
    rows_per_tile = min(out_features, max(1, budget // 2 // row_bytes))
    if rows_per_tile == out_features:
        weight = weight.to(device=compute_device, non_blocking=True)
        output = _run_token_chunks(
            flat_input,
            out_features,
            lambda chunk: _linear_on_compute(chunk, weight, bias, lora_down, lora_up),
            2 * (in_features + out_features + lora_rank) * dtype.itemsize,
            budget - out_features * row_bytes,
            compute_device,
        )
        return output.reshape(*input_tensor.shape[:-1], out_features)

    output = _activation_empty((tokens, out_features), dtype, compute_device)
    budget = _compute_budget(compute_device)

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

    The output is an activation (see _activation_empty). Every spatial dim but
    the width is tiled; each tile copies only the input it reads, including the
    kernel halo, and padding is applied on the compute GPU. Tiles run on
    comfy.ops.conv_nd, which takes Kitchen's rdna1 conv kernels where ComfyUI would.
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
    output = _activation_empty((batch, out_channels, *out_size), input_tensor.dtype, compute_device)
    element_size = output.element_size()
    channel_bytes = weight[0].numel() * weight.element_size()
    budget = min(_CONV_TILE_BUDGET_BYTES, _compute_budget(compute_device))

    channels_per_tile = min(out_channels, max(1, budget // 2 // channel_bytes))
    weight_buffer = torch.empty(
        (channels_per_tile, *weight.shape[1:]), device=compute_device, dtype=weight.dtype
    )
    # MIOpen may lower the conv to a GEMM over an im2col workspace; Kitchen's convs have none.
    workspace = 0 if comfy.ops._rdna1_conv_wanted(input_tensor, weight_buffer) else weight[0].numel()

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
                    result = comfy.ops.conv_nd(input_tile, weight_tile, bias_tile, stride, 0, dilation)
                    output[tuple(target)].copy_(result, non_blocking=True)

    return output


def _run_attention_on_compute(func, q, k, v, heads, mask=None, *args, compute_device, plan, **kwargs):
    """Run attention on the compute GPU, in head chunks over the whole sequence when it must.

    Q, K and V on the compute GPU run in one call, retried once after releasing
    the allocator's cached memory, until that runs out of memory; then, and for
    spilled ones, heads are split. Heads are independent, so a chunk copies only
    its own heads' Q, K and V. Every chunk takes all queries: backends prepare K
    and V on each call (Kitchen int8 rotates and quantizes them), so query chunks
    would repeat that work. Spilled heads come in at least _PIPELINE_CHUNKS
    chunks, so their copies overlap each other's attention, and three chunks
    fit half the free VRAM; heads already on the compute GPU run in as few
    chunks as half the free VRAM allows. On OOM the call restarts with half the
    heads, then half the queries, and plan keeps the result for later calls.
    When one head group and one query still do not fit, attention fails with an
    out-of-memory error.
    """
    key = (q.shape, k.shape)
    compute_device = torch.device(compute_device)
    if mask is not None:
        return func(q, k, v, heads, mask, *args, **kwargs)
    on_compute = q.device == k.device == v.device == compute_device
    if key not in plan and on_compute:
        for _ in range(2):
            try:
                return func(q, k, v, heads, None, *args, **kwargs)
            except torch.OutOfMemoryError:
                pass
            # Recover outside the except block, once the failed call's tensors are freed.
            torch.cuda.empty_cache()

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
        if on_compute:
            # Chunks of Q, K and V already there are views: only the running chunk's
            # backend temporaries, about group_bytes a head, take memory.
            groups = budget // 2 // group_bytes
        else:
            groups = min(budget // 2 // (3 * group_bytes), -(-kv_heads // _PIPELINE_CHUNKS))
        groups = max(1, min(groups, kv_heads))
        # [heads per chunk, queries per chunk]
        chunk = plan[key] = [groups * group, tokens]

    output = None
    compute_stream = torch.cuda.current_stream(compute_device) if compute_device.type == "cuda" else None

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
            output = _activation_empty(shape, result.dtype, compute_device)
            if output.device.type == "cuda":
                # Allocated on the copy stream; the compute stream reads it next.
                output.record_stream(compute_stream)
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
    if getattr(module, "_mgpu_preparation_logged", False):
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
    module._mgpu_preparation_logged = True


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


@_without_streaming
def _prepare_linear(module, dtype, compute_device):
    """Materialize a linear's weight on the compute GPU; returns (weight, bias, offload_state, lora).

    The weight is copied there as stored (quantized) and dequantized and patched
    there, wherever it is stored: the compute GPU is the fastest at that, and a
    storage GPU such as an APU keeps its shaders free.

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
        with _current_device(compute_device):
            weight, bias, offload_state = _materialize_linear(module, dtype, compute_device)
    finally:
        if lora:
            module.weight.patches = patches
    _log_weight_preparation(module, compute_device, weight, used_comfy_cast)
    return weight, bias, offload_state, lora


def _release_prepared_linear(module, weight, bias, offload_state):
    if offload_state is not None:
        from comfy.ops import uncast_bias_weight

        uncast_bias_weight(module, weight, bias, offload_state)


def _run_prepared_linear_on_compute(input_tensor, module, compute_device):
    """Prepare a weight too large to stage and stream its tiles through the compute GPU."""
    weight, bias, offload_state, lora = _prepare_linear(module, input_tensor.dtype, compute_device)
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

    def __init__(self, compute_device):
        self.compute_device = compute_device
        self.copy_stream = torch.cuda.Stream(compute_device)
        # Kept here, not on the modules: a module stored on another would be
        # registered as its submodule.
        self.following = {}
        self.mlp_linears = {}
        self.previous = None
        self.pending = []
        self.retired = []

    def end_forward(self):
        """Drop every staged weight, so that none stays on the compute GPU between steps."""
        for module in self.pending:
            module._mgpu_staged = None
        self.pending = []
        self.retired = []
        self.previous = None

    def linears(self, unit):
        return self.mlp_linears.get(unit, (unit,))

    def weight_bytes(self, unit, dtype):
        return sum(_staged_weight_bytes(linear, dtype) for linear in self.linears(unit))

    def stage(self, unit, dtype):
        """Stage a unit's weights, or None when one is too large and streams in tiles instead."""
        staged = tuple(_stage_linear(linear, dtype, self) for linear in self.linears(unit))
        return None if any(linear is None for linear in staged) else staged

    def retire(self, staged, compute_stream):
        for _, done in self.retired:
            done.synchronize()
        self.retired = [(staged, compute_stream.record_event())]


_StagedLinear = namedtuple("_StagedLinear", "weight bias lora_down lora_up dtype ready")


# GGUF weight types Comfy Kitchen's packed GEMM reads as stored, and its op for each.
_GGUF_NATIVE_OPS = {"Q8_0": "q8_0_linear", "Q4_K": "q4_k_linear"}


def _gguf_native(module, compute_device):
    """Whether a GGUF linear's Q8_0 or Q4_K weight runs on Comfy Kitchen's q8_0_linear or
    q4_k_linear, which read its blocks in the packed GEMM: staged as stored (Q8_0 half, Q4_K
    under a third of the dequantized float16 weight's bytes), 1.15-1.2x faster than its GEMM
    on gfx1010. Its patches are checked as it is staged (_gguf_native_lora)."""
    weight = module.weight
    op = _GGUF_NATIVE_OPS.get(getattr(getattr(weight, "tensor_type", None), "name", None))
    return (
        op is not None
        and getattr(comfy.ops, "_CK_PACKED", False)
        and hasattr(comfy.ops.quant_ops.ck, op)
        and comfy.ops.quant_ops.ck.fp16_packed_linear_is_accelerated(torch.device(compute_device))
    )


def _gguf_native_lora(module):
    """A native GGUF weight's patches as low-rank (down, up, scale) LoRAs, which run beside
    the staged blocks as for a dequantized weight, or None when a patch is not a plain LoRA
    and the loader has to merge it into the dequantized weight."""
    patches = [patch for patch_list, _ in getattr(module.weight, "patches", ()) for patch in patch_list]
    lora = _plain_lora_patches(patches)
    return lora if lora or not patches else None


def _staged_weight_bytes(module, dtype):
    if module._mgpu_quantized:
        return mm.module_size(module)
    if module._mgpu_gguf_native:
        return module.weight.nbytes
    return module.out_features * module.in_features * _gemm_weight_dtype(dtype, module._mgpu_compute_device).itemsize


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

    A quantized linear sends its stored weight; others are prepared there (see
    _prepare_linear). Returns None for a prepared weight larger than half the
    compute GPU's free memory, or one that still runs out of memory after the
    allocator's cache is released; it streams in tiles instead.
    """
    compute_device = module._mgpu_compute_device
    if not module._mgpu_quantized and compute_device.type == "cuda" and (
        2 * _staged_weight_bytes(module, dtype)
        > mm.get_free_memory(compute_device) - _COMPUTE_GPU_RESERVE_BYTES
    ):
        return None
    for attempt in range(2):
        streams = contextlib.ExitStack()
        if schedule is not None:
            streams.enter_context(torch.cuda.stream(schedule.copy_stream))
        try:
            with streams:
                if module._mgpu_quantized:
                    weight, bias, lora = module.weight, module.bias, ()
                    offload_state = None
                elif module._mgpu_gguf_native and (native_lora := _gguf_native_lora(module)) is not None:
                    weight = module.weight.as_subclass(torch.Tensor).reshape(module.out_features, -1)
                    bias = None if module.bias is None else module.bias.as_subclass(torch.Tensor)
                    lora, offload_state = native_lora, None
                else:
                    weight, bias, offload_state, lora = _prepare_linear(
                        module, _gemm_weight_dtype(dtype, compute_device), compute_device
                    )
                try:
                    staged_weight = _copy_to_compute(weight, compute_device)
                    # A float linear's bias stays in the activation dtype, whatever its weight is prepared in.
                    bias_dtype = None if module._mgpu_quantized else dtype
                    staged_bias = None if bias is None else bias.to(device=compute_device, dtype=bias_dtype, non_blocking=True)
                    lora_down, lora_up = _lora_on_compute(lora, compute_device, dtype)
                finally:
                    _release_prepared_linear(module, weight, bias, offload_state)
            break
        except torch.OutOfMemoryError:
            if attempt and module._mgpu_quantized:
                raise
        # cudaMallocAsync (--cuda-malloc) keeps freed blocks that a request of another
        # size cannot use until they are released.
        torch.cuda.empty_cache()
    else:
        return None
    ready = schedule.copy_stream.record_event() if schedule is not None else None
    return _StagedLinear(staged_weight, staged_bias, lora_down, lora_up, dtype, ready)


def _run_staged_linear(input_tensor, staged, compute_device, budget, during=None):
    """Run token chunks through a staged weight, with any copies overlapping the GEMMs."""
    flat_input = input_tensor.reshape(-1, input_tensor.shape[-1])
    in_features = flat_input.shape[1]
    out_features = staged.weight.shape[0]
    lora_rank = 0 if staged.lora_down is None else staged.lora_down.shape[0]
    # Input, output and LoRA chunks, doubled for the kernel's temporaries.
    token_bytes = 2 * (in_features + out_features + lora_rank) * input_tensor.element_size()
    output = _run_token_chunks(
        flat_input,
        out_features,
        lambda chunk: _linear_on_compute(
            chunk, staged.weight, staged.bias, staged.lora_down, staged.lora_up
        ),
        token_bytes,
        budget,
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
    # The unit's own weights are already on the compute GPU.
    budget = _compute_budget(schedule.compute_device)
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
        return _run_prepared_linear_on_compute(input_tensor, module, compute_device)
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

    A spilled input is copied there and its result back once per chunk, and the
    wide hidden activation never leaves the compute GPU. Its linears run the
    weights staged here (see configure_mixed_gemm).
    """
    schedule = mlp._mgpu_schedule
    compute_device = schedule.compute_device
    staged = _take_staged(mlp, input_tensor.dtype, schedule)
    if staged is None:
        return mlp._mgpu_original_forward(input_tensor)
    linears = schedule.linears(mlp)
    budget, during = _start_unit(mlp, staged, input_tensor.dtype, schedule)
    flat_input = input_tensor.reshape(-1, input_tensor.shape[-1])
    # The input and every linear's output, doubled for activations and kernel temporaries.
    token_bytes = (
        2 * (flat_input.shape[-1] + sum(linear.out_features for linear in linears))
        * input_tensor.element_size()
    )
    with contextlib.ExitStack() as stack:
        for linear, weights in zip(linears, staged):
            linear._mgpu_fused = weights
            stack.callback(setattr, linear, "_mgpu_fused", None)
            if linear._mgpu_quantized:
                stack.enter_context(_staged_parameters(linear, weights))
        output = _run_token_chunks(
            flat_input,
            linears[-1].out_features,
            mlp._mgpu_original_forward,
            token_bytes,
            budget,
            compute_device,
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
    # A quantized copy of each input chunk and its output.
    token_bytes = 2 * (module.in_features + module.out_features) * input_tensor.element_size()
    with _staged_parameters(module, staged[0]):
        output = _run_token_chunks(
            flat_input,
            module.out_features,
            module._mgpu_original_forward,
            token_bytes,
            budget,
            compute_device,
            during,
        )
    if schedule is not None:
        schedule.retire(staged, torch.cuda.current_stream(compute_device))
    return output.reshape(*input_tensor.shape[:-1], output.shape[-1])


def _configure_mixed_conv(module):
    """Tile a conv's math onto the compute GPU below its own forward.

    Hooking _conv_forward keeps layer-specific padding and temporal caches
    (such as Wan's CausalConv3d) and ComfyUI's weight casts.
    """
    module._mgpu_original_conv_forward = module._conv_forward
    module._mgpu_gemm_calls = 0

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
        if comfy.ops._linear_conv_wanted(module, input, weight):
            # A kernel spanning its whole input (a patch embed) is a linear over the flat patches.
            output = _run_tiled_linear_on_compute(
                input.flatten(1), weight.flatten(1), bias, module._mgpu_compute_device
            ).reshape(input.shape[0], -1, *(1,) * (input.dim() - 2))
        else:
            output = _run_tiled_conv_on_compute(
                module, input, weight, bias, padding, module._mgpu_compute_device
            )
        module._mgpu_gemm_calls += 1
        if module._mgpu_gemm_calls == 1:
            logger.info(
                "[MultiGPU DisTorch V2] Streamed conv active on %s (%s)",
                module._mgpu_compute_device,
                type(module).__name__,
            )
        return output

    module._conv_forward = mixed_conv_forward


# comfy.ops Linear.forward's fused prologue/epilogue arguments (see linear_input_act_).
_LINEAR_FUSION_KWARGS = frozenset(("input_act", "act_weight", "residual", "residual_scale"))


def configure_mixed_gemm(module, compute_device, schedule=None):
    """Run an eligible linear's or conv's GEMM on the compute GPU from weights stored elsewhere.

    With a schedule, each linear stages the next one's weight while it runs (see _run_mixed_linear).
    """
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)):
        tileable = _can_tile_conv_on_compute(module)
    elif _is_eligible_streamed_gemm(module):
        tileable = _can_tile_linear_on_compute(module)
    else:
        return False
    if not tileable:
        if not getattr(module, "_mgpu_skip_logged", False):
            logger.info(
                "[MultiGPU DisTorch V2] Mixed compute GEMM skipped for %s: module is not a standard linear or ungrouped conv",
                type(module).__name__,
            )
            module._mgpu_skip_logged = True
        return False

    module._mgpu_compute_device = torch.device(compute_device)
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)):
        if not hasattr(module, "_mgpu_original_conv_forward"):
            _configure_mixed_conv(module)
        return True

    # ComfyUI sets layout_type on layers whose weight is quantized (int8, fp8, ...).
    module._mgpu_quantized = getattr(module, "layout_type", None) is not None
    module._mgpu_gguf_native = not module._mgpu_quantized and _gguf_native(module, compute_device)
    module._mgpu_schedule = schedule
    module._mgpu_staged = None
    # The weight a fused MLP staged for this linear while it runs (see _run_fused_mlp).
    module._mgpu_fused = None
    if not hasattr(module, "_mgpu_original_forward"):
        module._mgpu_original_forward = module.forward
        module._mgpu_gemm_calls = 0

        def mixed_forward(*args, **kwargs):
            if len(args) != 1 or not isinstance(args[0], torch.Tensor) or kwargs.keys() - _LINEAR_FUSION_KWARGS:
                return module._mgpu_original_forward(*args, **kwargs)
            if kwargs:
                # ComfyUI's fused prologue/epilogue kernels expect the weight beside the
                # input; unfold them so the GEMM itself runs from the staged weight.
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
            module._mgpu_gemm_calls += 1
            if module._mgpu_gemm_calls == 1:
                logger.info(
                    "[MultiGPU DisTorch V2] Streamed GEMM active on %s (%s)",
                    module._mgpu_compute_device,
                    type(module).__name__,
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

    Token-wise MLPs are a Sequential of linears and activations, a SwiGLU
    FeedForward's w1, w3 and w2 (as in Lumina and Z-Image), or an fc1 and fc2
    pair (as in MiniMax H3).
    """
    children = dict(module.named_children())
    if isinstance(module, torch.nn.Sequential):
        linears = [child for child in children.values() if not isinstance(child, _MLP_ACTIVATIONS)]
    elif children.keys() == {"w1", "w2", "w3"}:
        linears = [children["w1"], children["w3"], children["w2"]]
    elif children.keys() == {"fc1", "fc2"}:
        linears = [children["fc1"], children["fc2"]]
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


def _streams_activations(model, model_size, compute_device, is_clip=False):
    """Whether a model runs streamed: weights stored off the compute GPU, activations on it.

    Only diffusion models and text encoders that would crowd the compute GPU
    stream: one taking at most half its free memory loads whole as usual.
    """
    if not (is_clip or hasattr(model, "diffusion_model")) or 2 * model_size <= mm.get_free_memory(compute_device):
        return False
    reasons = []
    if not _is_comfy_kitchen_attention_enabled():
        reasons.append("Comfy Kitchen attention is disabled")
    if not _is_hip_software_gemm_device(compute_device):
        reasons.append(f"compute device {compute_device} lacks HIP software GEMM")
    if reasons:
        logger.info(
            "[MultiGPU DisTorch V2] Streamed execution disabled: %s", "; ".join(reasons)
        )
        return False
    return True


def _evict_step(schedule, compute_device):
    """Leave nothing of a finished (or failed) forward on the compute GPU.

    Its activations are already unreferenced; this drops the staged weights and
    returns the allocator's cached blocks, which the cudaMallocAsync pool
    (--cuda-malloc) would otherwise keep out of a later, larger request. The next
    step, and whatever runs between steps (such as a preview decode), starts
    with the compute GPU's memory free. It synchronizes once per step.
    """
    if compute_device.type == "cuda":
        # The staged weights came from the copy stream; the compute stream must be done with them.
        torch.cuda.synchronize(compute_device)
    schedule.end_forward()
    _spilled.clear()
    if compute_device.type == "cuda":
        torch.cuda.empty_cache()


def configure_mixed_execution(diffusion_model, compute_device, schedule):
    """Run the diffusion model with its activations on the compute GPU and its weights streamed in.

    Activations spill to pinned system RAM only when the compute GPU has no room
    for them (see _StreamedActivations); linears and convs (see
    configure_mixed_gemm), token-wise MLPs (see _run_fused_mlp) and attention run
    in tiles sized to its free VRAM. Every step ends by evicting what it left on
    the compute GPU (see _evict_step).
    """
    if not hasattr(diffusion_model, "_mgpu_original_forward"):
        diffusion_model._mgpu_original_forward = diffusion_model.forward

        def mixed_forward(*args, transformer_options={}, **kwargs):
            compute = diffusion_model._mgpu_compute_device
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
            try:
                with _StreamedActivations(compute), _current_device(compute):
                    output = diffusion_model._mgpu_original_forward(
                        *_move_tensors(args, compute),
                        transformer_options=transformer_options,
                        **_move_tensors(kwargs, compute),
                    )
                return _move_tensors(output, output_device)
            finally:
                _evict_step(diffusion_model._mgpu_schedule, compute)

        diffusion_model.forward = mixed_forward
        logger.info(
            "[MultiGPU DisTorch V2] Streamed execution: activations on %s, evicted when each step completes",
            compute_device,
        )

    diffusion_model._mgpu_compute_device = torch.device(compute_device)
    diffusion_model._mgpu_schedule = schedule
    # Chunk sizes learned from OOMs, a few ints per attention shape, kept until the next load.
    diffusion_model._mgpu_attention_plan = {}
    _configure_fused_mlps(diffusion_model, compute_device, schedule)


def _configure_fused_mlps(model, compute_device, schedule):
    for module in model.modules():
        linears = _fused_mlp_linears(module, schedule)
        if linears is not None:
            configure_fused_mlp(module, linears, schedule)
    if schedule.mlp_linears:
        logger.info(
            "[MultiGPU DisTorch V2] Streamed execution: %d MLPs run whole on %s in token chunks",
            len(schedule.mlp_linears),
            compute_device,
        )


def configure_streamed_text_encoder(model, compute_device, schedule):
    """Run a text encoder's encodes with activations on the compute GPU and weights streamed in.

    Each SDClipModel forward (its vision tower included) runs as a diffusion
    model's does (see configure_mixed_execution) and ends by evicting what it
    left on the compute GPU. Its attention takes no override, so an op the
    compute GPU has no room for is tiled by _StreamedActivations.
    """
    for encoder in model.modules():
        if not isinstance(encoder, comfy.sd1_clip.SDClipModel):
            continue
        if not hasattr(encoder, "_mgpu_original_forward"):
            encoder._mgpu_original_forward = encoder.forward

            def streamed_forward(*args, encoder=encoder, **kwargs):
                compute = encoder._mgpu_compute_device
                try:
                    with _StreamedActivations(compute), _current_device(compute):
                        return encoder._mgpu_original_forward(*args, **kwargs)
                finally:
                    _evict_step(encoder._mgpu_schedule, compute)

            encoder.forward = streamed_forward
            logger.info(
                "[MultiGPU DisTorch V2] Streamed execution: %s activations on %s, evicted when each encode completes",
                type(encoder).__name__,
                compute_device,
            )
        encoder._mgpu_compute_device = torch.device(compute_device)
        encoder._mgpu_schedule = schedule
    _configure_fused_mlps(model, compute_device, schedule)


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
                    if inner_model._distorch_v2_meta["eject_models"]:
                        eject_device = device
                        logger.mgpu_mm_log(
                            "DisTorch eject_models detected - MAX memory eviction"
                        )
                    # Weights the compute GPU cannot hold are placed on other devices
                    # when the model loads (see place_weights), so the request is capped
                    # at what the device has.
                    adjusted_memory = min(base_memory, mm.get_total_memory(device))
                    total_memory_required[device] = (
                        total_memory_required.get(device, 0) + adjusted_memory
                    )
                    logger.mgpu_mm_log(
                        f"DisTorch model requests {adjusted_memory/(1024**3):.2f}GB for device {device}"
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
                return original_partially_load(
                    self, device_to, extra_memory, force_patch_weights
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
            device_to = torch.device(device_to)
            is_clip_model = getattr(self, "is_clip", False)
            # DisTorch2 VAE loaders patch the VAE's first_stage_model directly. A VAE's
            # decode tiles already bound its activations, so it is never streamed.
            is_vae = (
                not is_clip_model
                and not hasattr(self.model, "diffusion_model")
                and hasattr(self.model, "decode")
            )
            model_size = sum(unpack_load_item(item)[0] for item in self._load_list())
            streamed = not is_vae and _streams_activations(
                self.model, model_size, device_to, is_clip_model
            )
            placement_key = (str(device_to), streamed)
            if (
                getattr(self, "_distorch_placement_key", None) == placement_key
                and not unpatch_weights
                and not force_patch_weights
            ):
                block_assignments = self._distorch_cached_assignments
            else:
                block_assignments = place_weights(
                    self, device_to, streamed, is_clip=is_clip_model
                )
                self._distorch_cached_assignments = block_assignments
                self._distorch_placement_key = placement_key

            model_original_dtype = comfy.utils.weight_dtype(self.model.state_dict())
            high_precision_loras = getattr(
                self.model, "_distorch_high_precision_loras", True
            )
            schedule = _MixedSchedule(device_to) if streamed else None

            # Use standard ComfyUI load list - the device comparison fix ensures we don't crash
            loading = self._load_list()
            loading.sort(reverse=True)
            for item in loading:
                module_size, module_name, module_object, params = unpack_load_item(item)
                block_target_device = torch.device(
                    block_assignments.get(module_name, device_to)
                )
                if (
                    not unpatch_weights
                    and hasattr(module_object, "comfy_patched_weights")
                    and module_object.comfy_patched_weights is True
                ):
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
                        _pin_cpu_weights(module_object)

                    if streamed:
                        configure_mixed_gemm(module_object, device_to, schedule)

                    mem_counter += module_size
                    continue

                # Move directly to the assigned device: staging other devices' weights
                # on the compute GPU would defeat the placement and can OOM.
                module_object.to(block_target_device)

                # Step 2: Apply LoRa patches on the assigned device.
                weight_key = f"{module_name}.weight"
                bias_key = f"{module_name}.bias"
                with _current_device(block_target_device):
                    if weight_key in self.patches:
                        self.patch_weight_to_device(
                            weight_key, device_to=block_target_device
                        )
                    if bias_key in self.patches:
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
                    and block_target_device.type == "cpu"
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
                _pin_cpu_weights(module_object)

                # Step 4: Enable runtime weight casting for modules stored off the compute GPU.
                if block_target_device != device_to:
                    module_object.comfy_cast_weights = True

                if streamed:
                    configure_mixed_gemm(module_object, device_to, schedule)

                # Mark as patched and update memory counter
                module_object.comfy_patched_weights = True
                mem_counter += module_size

            # _load_list skips modules that also own child parameters (a weight_norm conv
            # keeps its bias while the weight lives in its parametrization), so their own
            # parameters are moved here as ComfyUI's full load does with model.to().
            listed = {unpack_load_item(item)[1] for item in loading}
            for name, module in self.model.named_modules():
                if name not in listed:
                    for param in module.parameters(recurse=False):
                        param.data = param.data.to(device_to)

            if streamed and is_clip_model:
                configure_streamed_text_encoder(self.model, device_to, schedule)
            elif streamed:
                configure_mixed_execution(self.model.diffusion_model, device_to, schedule)

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


def place_weights(model_patcher, compute_device, streamed, is_clip=False):
    """Choose a device for every module's weights; returns {module name: device}.

    The compute GPU keeps what is computed there directly: modules ComfyUI
    cannot cast at runtime, tiny ones and CLIP's embeddings. A streamed model
    leaves the rest of it to activations; any other model fills it from the
    head, leaving ComfyUI's inference reserve and room to cast its largest
    weight there in float32 (a GGUF weight is dequantized whole). The remaining
    weights are spread over every other device, CPU included, in proportion to
    their free memory, so that none is paged in from disk while the model runs
    (see _pin_cpu_weights).
    """
    compute_device = str(torch.device(compute_device))
    raw_block_list = model_patcher._load_list()
    if is_clip:
        _, distributable, assignments, _ = _extract_clip_head_blocks(raw_block_list, compute_device)
    else:
        distributable, assignments = raw_block_list, {}
    total_memory = sum(unpack_load_item(item)[0] for item in raw_block_list)
    blocks = []
    for item in distributable:
        size, name, module, _ = unpack_load_item(item)
        if size >= total_memory * 0.0001 and hasattr(module, "bias") and hasattr(module, "comfy_cast_weights"):
            blocks.append((name, size))
        else:
            assignments[name] = compute_device

    room = 0
    if not streamed:
        largest_cast = max(
            (
                math.prod(module.weight.shape) * 4
                for _, _, module, _ in map(unpack_load_item, raw_block_list)
                if isinstance(getattr(module, "weight", None), torch.Tensor)
            ),
            default=0,
        )
        room = (
            mm.get_free_memory(torch.device(compute_device))
            - mm.minimum_inference_memory()
            - largest_cast
        )
    kept = sum(size for _, size in blocks)
    offload = []
    for name, size in reversed(blocks):
        if kept > room:
            offload.append((name, size))
            kept -= size
        else:
            assignments[name] = compute_device

    capacity = {
        device: mm.get_free_memory(torch.device(device)) * (1 - _STORAGE_RESERVE_FRACTION)
        for device in get_device_list()
        if device != compute_device
    }
    total_capacity = sum(capacity.values())
    offload_bytes = sum(size for _, size in offload)
    share = {
        device: offload_bytes * free / total_capacity for device, free in capacity.items()
    } if total_capacity > 0 else {}
    placed = defaultdict(int)
    for name, size in reversed(offload):
        candidates = sorted(share, key=lambda device: placed[device] - share[device])
        # A weight no device has room for runs from the compute GPU.
        assignments[name] = next(
            (device for device in candidates if placed[device] + size <= capacity[device]),
            compute_device,
        )
        placed[assignments[name]] += size

    sizes = {name: size for name, size in blocks}
    per_device = defaultdict(lambda: [0, 0])
    for name, device in assignments.items():
        per_device[device][0] += 1
        per_device[device][1] += sizes.get(name, 0)
    logger.info(
        "[MultiGPU DisTorch V2] %s weights%s: %s",
        type(model_patcher.model).__name__,
        " (streamed)" if streamed else "",
        ", ".join(
            f"{device} {count} layers {nbytes / 1024**2:.0f} MB"
            for device, (count, nbytes) in sorted(per_device.items(), key=lambda kv: (kv[0] == "cpu", kv[0]))
        ),
    )
    return assignments


def _pin_cpu_weights(module):
    """Copy a module's CPU weights into pinned RAM.

    They then stay resident rather than being paged in from a memory-mapped
    file, and the compute GPU's copy engine reads them asynchronously.
    """
    module._apply(
        lambda tensor: tensor.pin_memory() if tensor.device.type == "cpu" and not tensor.is_pinned() else tensor,
        recurse=False,
    )
