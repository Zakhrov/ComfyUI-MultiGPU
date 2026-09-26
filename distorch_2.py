"""
DisTorch Safetensor Memory Management Module
Contains all safetensor related code for distributed memory management
"""

import functools
import itertools
import math
import torch
import logging
import re
from collections import defaultdict

logger = logging.getLogger("MultiGPU")
import comfy.model_management as mm
import comfy.model_patcher
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
# Attention sizes chunks from its measured peak, which includes the backend's
# workspace, so it needs less headroom.
_ATTENTION_RESERVE_BYTES = 128 * 1024 * 1024
# Conv tiles stop getting faster well below this, and larger ones let the
# backend's im2col workspace exhaust the compute GPU. A fixed cap also keeps
# tile shapes stable so MIOpen reuses its tuned kernels between runs.
_CONV_TILE_BUDGET_BYTES = 128 * 1024 * 1024
# Below this many queries per chunk, per-call overhead makes the donor faster.
_MIN_ATTENTION_CHUNK_TOKENS = 256


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
    if not isinstance(module, torch.nn.Linear) or weight.ndim != 2:
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


def _run_tiled_linear_on_compute(input_tensor, weight, bias, compute_device):
    """Run a linear on the compute GPU in tiles sized to its free VRAM.

    The input and output stay on the input's device; only token chunks, weight
    tiles and their GEMM results visit the compute GPU.
    """
    flat_input = input_tensor.reshape(-1, input_tensor.shape[-1])
    tokens, in_features = flat_input.shape
    out_features = weight.shape[0]
    output = torch.empty(
        (tokens, out_features), device=input_tensor.device, dtype=input_tensor.dtype
    )
    row_bytes = in_features * weight.element_size()
    budget = _compute_budget(compute_device)

    # Keep the whole weight on the compute GPU when it takes at most half the
    # budget; otherwise stream output-channel tiles for every token chunk.
    rows_per_tile = min(out_features, max(1, budget // 2 // row_bytes))
    tokens_per_chunk = min(
        tokens,
        max(
            1,
            (budget - rows_per_tile * row_bytes)
            // ((in_features + rows_per_tile) * output.element_size()),
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
    if isinstance(bias, torch.Tensor):
        bias = bias.to(device=compute_device)
    resident = rows_per_tile == out_features
    if resident:
        weight_buffer.copy_(weight)

    with torch.cuda.device_of(input_buffer):
        for token_start in range(0, tokens, tokens_per_chunk):
            token_end = min(token_start + tokens_per_chunk, tokens)
            input_chunk = input_buffer[: token_end - token_start]
            input_chunk.copy_(flat_input[token_start:token_end])
            for start in range(0, out_features, rows_per_tile):
                end = min(start + rows_per_tile, out_features)
                weight_tile = weight_buffer[: end - start]
                if not resident:
                    weight_tile.copy_(weight[start:end])
                output_tile = output_buffer[: input_chunk.shape[0] * (end - start)].view(
                    input_chunk.shape[0], end - start
                )
                if bias is None:
                    torch.mm(input_chunk, weight_tile.t(), out=output_tile)
                else:
                    torch.addmm(bias[start:end], input_chunk, weight_tile.t(), out=output_tile)
                output[token_start:token_end, start:end].copy_(output_tile)

    return output.reshape(*input_tensor.shape[:-1], out_features)


def _run_tiled_conv_on_compute(
    input_tensor, weight, bias, stride, padding, dilation, compute_device
):
    """Run a zero-padded conv2d or conv3d on the compute GPU in tiles.

    The input and output stay on the input's device. Every spatial dim but the
    width is tiled; each tile copies only the input it reads, including the
    kernel halo, and padding is applied on the compute GPU.
    """
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

    def input_extent(chunk, dim):
        return (chunk[dim] - 1) * stride[dim] + span[dim]

    def tile_bytes(chunk):
        input_elements = channels
        for dim in range(dims):
            input_elements *= input_extent(chunk, dim)
        output_elements = 1
        for length in chunk:
            output_elements *= length
        # Includes an im2col workspace in case the backend lowers the conv to a GEMM.
        return (
            input_elements + (channels_per_tile + weight[0].numel()) * output_elements
        ) * element_size

    # Halve the largest tiled dim until a tile fits; halving keeps the number
    # of distinct tile shapes, and so backend kernel searches, small.
    chunk = list(out_size)
    tile_budget = budget - channels_per_tile * channel_bytes
    while tile_bytes(chunk) > tile_budget and max(chunk[:-1]) > 1:
        dim = max(range(dims - 1), key=lambda d: chunk[d])
        chunk[dim] = -(-chunk[dim] // 2)

    weight_buffer = torch.empty(
        (channels_per_tile, *weight.shape[1:]), device=compute_device, dtype=weight.dtype
    )
    input_numel = channels
    for dim in range(dims):
        input_numel *= input_extent(chunk, dim)
    input_buffer = torch.empty(input_numel, device=compute_device, dtype=output.dtype)
    if isinstance(bias, torch.Tensor):
        bias = bias.to(device=compute_device)
    resident = channels_per_tile == out_channels
    if resident:
        weight_buffer.copy_(weight)
    conv = torch.nn.functional.conv3d if dims == 3 else torch.nn.functional.conv2d
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
                input_tile[tuple(interior)].copy_(input_tensor[tuple(source)])
                target = [slice(sample, sample + 1), None] + [
                    slice(start, end) for start, end in zip(starts, ends)
                ]
                for start in range(0, out_channels, channels_per_tile):
                    end = min(start + channels_per_tile, out_channels)
                    weight_tile = weight_buffer[: end - start]
                    if not resident:
                        weight_tile.copy_(weight[start:end])
                    target[1] = slice(start, end)
                    output[tuple(target)].copy_(
                        conv(
                            input_tile,
                            weight_tile,
                            None if bias is None else bias[start:end],
                            stride,
                            0,
                            dilation,
                        )
                    )

    return output


def _run_attention_on_compute(func, q, k, v, heads, *args, compute_device, mask=None, **kwargs):
    """Run attention on the compute GPU in query chunks; K and V are copied once.

    Chunks are sized from the measured memory of earlier chunks: a 1-token
    probe gives the backend's fixed per-call cost, and a second chunk gives
    the per-token cost. Queries that cannot be chunked efficiently in the
    remaining VRAM run on the donor instead.
    """
    kv_bytes = (k.numel() + v.numel()) * k.element_size()
    budget = _compute_budget(compute_device, _ATTENTION_RESERVE_BYTES) - kv_bytes
    if mask is not None or budget <= 0:
        return func(q, k, v, heads, *args, mask=mask, **kwargs)

    seq_dim = 2 if kwargs.get("skip_reshape", False) else 1
    out_dim = 2 if kwargs.get("skip_output_reshape", False) else 1
    tokens = q.shape[seq_dim]
    q_token_bytes = q.numel() // tokens * q.element_size()
    # Used for the second chunk only: assumes the backend materializes scores,
    # softmax and one temporary per query.
    conservative_token_bytes = (
        2 * q_token_bytes + 3 * q.shape[0] * heads * k.shape[seq_dim] * q.element_size()
    )
    measure_peak = compute_device.type == "cuda"
    donor_k, donor_v = k, v
    k = k.to(device=compute_device)
    v = v.to(device=compute_device)
    output = None
    probes = []
    use_donor = False
    tokens_per_chunk = 1 if measure_peak else max(1, budget // conservative_token_bytes)
    start = 0
    while start < tokens:
        length = min(tokens_per_chunk, tokens - start)
        q_chunk = q.narrow(seq_dim, start, length)
        on_compute = not use_donor
        if on_compute:
            q_chunk = q_chunk.to(device=compute_device)
            if measure_peak:
                torch.cuda.reset_peak_memory_stats(compute_device)
                base = torch.cuda.memory_allocated(compute_device)
            with torch.cuda.device_of(q_chunk):
                out_chunk = func(q_chunk, k, v, heads, *args, mask=None, **kwargs)
        else:
            # Too little compute VRAM to amortize the per-call cost: finish on the donor.
            del k, v
            length = tokens - start
            q_chunk = q.narrow(seq_dim, start, length)
            with torch.cuda.device_of(q_chunk):
                out_chunk = func(q_chunk, donor_k, donor_v, heads, *args, mask=None, **kwargs)
        if output is None:
            shape = list(out_chunk.shape)
            shape[out_dim] = tokens
            output = torch.empty(shape, device=q.device, dtype=out_chunk.dtype)
        output.narrow(out_dim, start, length).copy_(out_chunk)
        del q_chunk, out_chunk
        start += length
        if not (measure_peak and on_compute):
            continue
        used = torch.cuda.max_memory_allocated(compute_device) - base + length * q_token_bytes
        # Re-read free memory each chunk so allocations outside PyTorch, such
        # as a resident text encoder, shrink the next chunk.
        budget = _compute_budget(compute_device, _ATTENTION_RESERVE_BYTES)
        probes.append((length, used))
        if len(probes) == 1:
            tokens_per_chunk = max(2, (budget - used) // conservative_token_bytes)
        elif length > probes[0][0]:
            token_bytes = max(
                q_token_bytes, (used - probes[0][1]) // (length - probes[0][0])
            )
            fixed_bytes = max(0, probes[0][1] - probes[0][0] * token_bytes)
            fit = (budget - fixed_bytes) // token_bytes
            use_donor = fit < _MIN_ATTENTION_CHUNK_TOKENS
            # Small probes hide the per-token cost in allocator rounding, so grow
            # at most 4x per chunk and re-measure before trusting the estimate.
            tokens_per_chunk = max(1, min(fit, 4 * length))
    return output


def _materialize_linear_on_donor(module, dtype, donor_device):
    """Apply ComfyUI weight casts and patches on the donor before transfer."""
    module_cast_bias_weight = getattr(module, "cast_bias_weight", None)
    if callable(module_cast_bias_weight):
        weight, bias = module_cast_bias_weight(dtype=dtype, device=donor_device)
        return weight, bias, None

    needs_materialization = (
        getattr(module, "comfy_cast_weights", False)
        or bool(getattr(module, "weight_function", ()))
        or bool(getattr(module, "bias_function", ()))
    )
    if not needs_materialization:
        weight = module.weight.to(device=donor_device, dtype=dtype)
        bias = getattr(module, "bias", None)
        if isinstance(bias, torch.Tensor):
            bias = bias.to(device=donor_device, dtype=dtype)
        return weight, bias, None

    from comfy.ops import cast_bias_weight

    weight, bias, offload_state = cast_bias_weight(
        module,
        dtype=dtype,
        device=donor_device,
        bias_dtype=dtype,
        offloadable=True,
    )
    return weight, bias, offload_state


def _log_donor_preparation(module, donor_device, weight, used_comfy_cast):
    """Confirm once that the donor materialized the weight before compute transfer."""
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
        "[MultiGPU DisTorch V2] Donor preparation confirmed on %s: %s; "
        "prepared weight is on %s (%s, %.2f MiB) before compute-GPU GEMM",
        donor_device,
        ", ".join(operations),
        weight.device,
        weight.dtype,
        weight.numel() * weight.element_size() / (1024**2),
    )
    module._mgpu_donor_preparation_logged = True


def _run_donor_prepared_linear_on_compute(
    input_tensor, module, compute_device, donor_device
):
    """Prepare the weight on the donor and run its GEMM on the compute GPU."""
    used_comfy_cast = (
        getattr(module, "comfy_cast_weights", False)
        or bool(getattr(module, "weight_function", ()))
        or bool(getattr(module, "bias_function", ()))
    )
    weight, bias, offload_state = _materialize_linear_on_donor(
        module, input_tensor.dtype, donor_device
    )
    try:
        _log_donor_preparation(module, donor_device, weight, used_comfy_cast)
        return _run_tiled_linear_on_compute(input_tensor, weight, bias, compute_device)
    finally:
        if offload_state is not None:
            from comfy.ops import uncast_bias_weight

            uncast_bias_weight(module, weight, bias, offload_state)


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
            input,
            weight,
            bias,
            module.stride,
            padding,
            module.dilation,
            module._mgpu_compute_device,
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


def configure_mixed_gemm(module, compute_device, donor_device):
    """Run an eligible linear's or conv's GEMM on the compute GPU; its activations stay on the donor."""
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
    if not hasattr(module, "_mgpu_original_forward"):
        module._mgpu_original_forward = module.forward
        module._mgpu_donor_gemm_calls = 0

        def mixed_forward(*args, **kwargs):
            if len(args) != 1 or kwargs or not isinstance(args[0], torch.Tensor):
                return module._mgpu_original_forward(*args, **kwargs)
            output = _run_donor_prepared_linear_on_compute(
                args[0],
                module,
                module._mgpu_compute_device,
                module._mgpu_donor_execution_device,
            )
            module._mgpu_donor_gemm_calls += 1
            if module._mgpu_donor_gemm_calls == 1:
                logger.info(
                    "[MultiGPU DisTorch V2] Donor-prepared compute GEMM active: %s -> %s (%s)",
                    module._mgpu_donor_execution_device,
                    module._mgpu_compute_device,
                    type(module).__name__,
                )
            return output

        module.forward = mixed_forward
    return True


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


def select_mixed_donor_device(model, block_assignments, compute_device, is_vae=False):
    """Pick the donor GPU that holds a mixed-mode model's activations, or None."""
    compute_device = torch.device(compute_device)
    donors = sorted(
        {str(device) for device in block_assignments.values()}
        - {"cpu", str(compute_device)}
    )
    reasons = []
    if not is_vae and not hasattr(model, "diffusion_model"):
        reasons.append("model has no diffusion model or VAE")
    # VAE attention stays on the donor, so only diffusion models need Kitchen attention.
    if not is_vae and not _is_comfy_kitchen_attention_enabled():
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


def configure_mixed_execution(diffusion_model, donor_device, compute_device):
    """Run the diffusion model on the donor and send attention to the compute GPU.

    Activations, norms, modulation, rope and weight preparation stay on the
    donor. Linears and convs (see configure_mixed_gemm) and attention run their GEMMs on
    the compute GPU in tiles sized to its free VRAM.
    """
    if not hasattr(diffusion_model, "_mgpu_original_forward"):
        diffusion_model._mgpu_original_forward = diffusion_model.forward

        def mixed_forward(*args, transformer_options={}, **kwargs):
            donor = diffusion_model._mgpu_donor_execution_device
            compute = diffusion_model._mgpu_compute_device
            output_device = _first_tensor(args).device
            transformer_options = transformer_options.copy()
            previous_override = transformer_options.get("optimized_attention_override")

            def attention_on_compute(func, *attention_args, **attention_kwargs):
                if previous_override is not None:
                    func = functools.partial(previous_override, func)
                return _run_attention_on_compute(
                    func, *attention_args, compute_device=compute, **attention_kwargs
                )

            transformer_options["optimized_attention_override"] = attention_on_compute
            donor_args = _move_tensors(args, donor)
            with torch.cuda.device_of(_first_tensor(donor_args)):
                output = diffusion_model._mgpu_original_forward(
                    *donor_args,
                    transformer_options=transformer_options,
                    **_move_tensors(kwargs, donor),
                )
            return _move_tensors(output, output_device)

        diffusion_model.forward = mixed_forward
        logger.info(
            "[MultiGPU DisTorch V2] Mixed execution: activations on %s, GEMMs and attention on %s",
            donor_device,
            compute_device,
        )

    diffusion_model._mgpu_donor_execution_device = torch.device(donor_device)
    diffusion_model._mgpu_compute_device = torch.device(compute_device)


def configure_mixed_vae(vae_model, donor_device, compute_device):
    """Run VAE encode and decode on the donor; conv and linear GEMMs go to the compute GPU.

    Activations, norms, attention, upsampling and temporal caches stay on the
    donor, so the compute GPU only holds conv tiles (see configure_mixed_gemm).
    """
    if not hasattr(vae_model, "_mgpu_original_methods"):
        vae_model._mgpu_original_methods = {}

        def donor_call(method):
            def mixed_call(*args, **kwargs):
                donor = vae_model._mgpu_donor_execution_device
                output_device = _first_tensor(args).device
                # Chunked-IO VAEs write straight into output_buffer and move
                # their own chunks to the given device.
                donor_kwargs = {
                    key: value if key == "output_buffer" else _move_tensors(value, donor)
                    for key, value in kwargs.items()
                }
                if "device" in kwargs:
                    donor_kwargs["device"] = donor
                donor_args = _move_tensors(args, donor)
                with torch.cuda.device_of(_first_tensor(donor_args)):
                    output = method(*donor_args, **donor_kwargs)
                if output is kwargs.get("output_buffer"):
                    return output
                return _move_tensors(output, output_device)

            return mixed_call

        for name in ("encode", "decode", "encode_tiled", "decode_tiled"):
            method = getattr(vae_model, name, None)
            if callable(method):
                vae_model._mgpu_original_methods[name] = method
                setattr(vae_model, name, donor_call(method))
        logger.info(
            "[MultiGPU DisTorch V2] Mixed VAE execution: activations on %s, conv GEMMs on %s",
            donor_device,
            compute_device,
        )

    vae_model._mgpu_donor_execution_device = torch.device(donor_device)
    vae_model._mgpu_compute_device = torch.device(compute_device)


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

    pinned_weight = offload_weight(weight)
    if isinstance(weight, torch.nn.Parameter):
        pinned_weight = torch.nn.Parameter(
            pinned_weight, requires_grad=weight.requires_grad
        )
    module.weight = pinned_weight
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
            # DisTorch2 VAE loaders patch the VAE's first_stage_model directly.
            is_vae = (
                not is_clip_model
                and not hasattr(self.model, "diffusion_model")
                and hasattr(self.model, "decode")
            )
            mixed_donor_device = None
            if donor_gemm_execution_mode == "mixed":
                mixed_donor_device = select_mixed_donor_device(
                    self.model,
                    device_assignments["block_assignments"],
                    device_to,
                    is_vae,
                )
            # Mixed mode keeps activations on the donor, so weights stored anywhere
            # else are cast to it at runtime.
            activation_device = mixed_donor_device or device_to

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
                            module_object, device_to, mixed_donor_device
                        )
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

                # Move directly to the assigned device. Staging donor weights on the
                # compute GPU defeats offload and can OOM before the forward wrapper.
                module_object.to(block_target_device)

                # Step 2: Apply LoRa patches on the assigned device.
                weight_key = f"{module_name}.weight"
                bias_key = f"{module_name}.bias"

                if weight_key in self.patches:
                    self.patch_weight_to_device(
                        weight_key, device_to=block_target_device
                    )
                if weight_key in self.weight_wrapper_patches:
                    module_object.weight_function.extend(
                        self.weight_wrapper_patches[weight_key]
                    )

                if bias_key in self.patches:
                    self.patch_weight_to_device(bias_key, device_to=block_target_device)
                if bias_key in self.weight_wrapper_patches:
                    module_object.bias_function.extend(
                        self.weight_wrapper_patches[bias_key]
                    )

                # Step 3: FP8 casting for CPU storage (if enabled)
                has_patches = weight_key in self.patches or bias_key in self.patches

                if (
                    not high_precision_loras
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
                    if block_target_device == "cpu" and pin_hip_offloaded_weight(
                        module_object
                    ):
                        logger.debug(
                            f"[MultiGPU DisTorch V2] Pinned {module_name} weight for direct HIP host access"
                        )
                    module_object.comfy_cast_weights = True

                if mixed_donor_device is not None:
                    configure_mixed_gemm(
                        module_object, device_to, mixed_donor_device
                    )
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

            if mixed_donor_device is not None and is_vae:
                configure_mixed_vae(self.model, mixed_donor_device, device_to)
            elif mixed_donor_device is not None:
                configure_mixed_execution(
                    self.model.diffusion_model, mixed_donor_device, device_to
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
        if any(kw in module_name.lower() for kw in head_keywords):
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
    model = model_patcher.model if hasattr(model_patcher, "model") else model_patcher
    total_memory = 0

    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            if module.weight is not None:
                total_memory += module.weight.numel() * module.weight.element_size()
            if hasattr(module, "bias") and module.bias is not None:
                total_memory += module.bias.numel() * module.bias.element_size()

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
