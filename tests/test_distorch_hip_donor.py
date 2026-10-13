import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as functional

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_distorch_module():
    package_name = "multigpu_donor_test"
    for module_name in tuple(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            del sys.modules[module_name]

    package = types.ModuleType(package_name)
    package.__path__ = [str(REPOSITORY_ROOT)]
    sys.modules[package_name] = package

    comfy = types.ModuleType("comfy")
    model_management = types.ModuleType("comfy.model_management")
    model_management.get_free_memory = lambda *_: 0
    model_patcher = types.ModuleType("comfy.model_patcher")
    quant_ops = types.ModuleType("comfy.quant_ops")
    quant_ops.QuantizedTensor = type("QuantizedTensor", (torch.Tensor,), {})
    ops = types.ModuleType("comfy.ops")
    ops.dense_linear = functional.linear
    ops._packed_conv_wanted = lambda *_: False
    ops._direct_conv_wanted = lambda *_: False
    ops._linear_conv_wanted = lambda *_: False
    sd1_clip = types.ModuleType("comfy.sd1_clip")
    sd1_clip.SDClipModel = type("SDClipModel", (torch.nn.Module,), {})
    comfy.model_management = model_management
    comfy.model_patcher = model_patcher
    comfy.ops = ops
    comfy.sd1_clip = sd1_clip
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = model_management
    sys.modules["comfy.model_patcher"] = model_patcher
    sys.modules["comfy.quant_ops"] = quant_ops
    sys.modules["comfy.ops"] = ops
    sys.modules["comfy.sd1_clip"] = sd1_clip

    device_utils = types.ModuleType(f"{package_name}.device_utils")
    device_utils.get_device_list = lambda: ["cpu"]
    sys.modules[device_utils.__name__] = device_utils
    management = types.ModuleType(f"{package_name}.model_management_mgpu")
    management.multigpu_memory_log = lambda *_: None
    sys.modules[management.__name__] = management

    spec = importlib.util.spec_from_file_location(
        f"{package_name}.distorch_2", REPOSITORY_ROOT / "distorch_2.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DonorModule:
    def __init__(self):
        self.weight = torch.empty(2, 2)
        self.calls = []

    def forward(self, value, *, scale=1):
        self.calls.append((value, scale))
        return value * scale


class AllocationModule:
    def __init__(self):
        self.weight = torch.empty(2, 2)
        self.bias = torch.empty(2)
        self.comfy_cast_weights = True


class AllocationModel:
    def __init__(self, module):
        self.module = module

    def named_modules(self):
        return (("linear", self.module),)


class AllocationPatcher:
    def __init__(self, *modules):
        self.model = AllocationModel(modules[0])
        self.modules = modules

    def _load_list(self):
        return tuple(
            (module.weight.nbytes, f"layer{index}", module, {})
            for index, module in enumerate(self.modules)
        )


class TestHipDonorGemmOffload(unittest.TestCase):
    def setUp(self):
        self.distorch = load_distorch_module()

    def spill(self, tensor):
        """A copy of tensor in a pinned buffer, as an activation the compute GPU had no room for."""
        spilled = self.distorch._spill_buffer(tensor.shape, tensor.dtype)
        spilled.copy_(tensor)
        return spilled

    def test_mixed_linear_wraps_large_linears(self):
        module = torch.nn.Linear(1, 16 * 1024 * 1024 // 4 + 1, bias=False)

        self.assertTrue(self.distorch.configure_mixed_gemm(module, "cuda:0"))
        self.assertEqual(module._mgpu_compute_device, torch.device("cuda:0"))

    def test_streamed_execution_requires_comfy_kitchen_attention(self):
        model = types.SimpleNamespace(diffusion_model=object())

        with mock.patch.object(
            self.distorch, "_is_hip_software_gemm_device", return_value=True
        ), self.assertLogs("MultiGPU", level="INFO") as logs:
            self.assertFalse(self.distorch._streams_activations(model, 1 << 30, "cuda:0"))

        self.assertIn("Comfy Kitchen attention is disabled", logs.output[0])

    def test_streams_only_diffusion_models_that_crowd_the_compute_gpu(self):
        model = types.SimpleNamespace(diffusion_model=object())

        with mock.patch.object(
            self.distorch, "_is_hip_software_gemm_device", return_value=True
        ), mock.patch.object(
            self.distorch.mm, "comfy_kitchen_attention_enabled", return_value=True, create=True
        ), mock.patch.object(self.distorch.mm, "get_free_memory", lambda *_: 4 << 30):
            self.assertTrue(self.distorch._streams_activations(model, 3 << 30, "cuda:0"))
            self.assertFalse(self.distorch._streams_activations(model, 2 << 30, "cuda:0"))
            self.assertFalse(
                self.distorch._streams_activations(types.SimpleNamespace(), 3 << 30, "cuda:0")
            )

    def test_tiled_linear_streams_weight_tiles_when_budget_is_small(self):
        module = torch.nn.Linear(3, 5, bias=True).requires_grad_(False)
        input_tensor = torch.randn(2, 4, 3)

        with mock.patch.object(self.distorch, "_COMPUTE_GPU_RESERVE_BYTES", 0), mock.patch.object(
            self.distorch.mm, "get_free_memory", lambda *_: 48
        ):
            output = self.distorch._run_tiled_linear_on_compute(
                input_tensor, module.weight, module.bias, "cpu"
            )

        self.assertTrue(torch.allclose(output, module(input_tensor)))
        self.assertEqual(output.shape, (2, 4, 5))

    def test_tiled_linear_keeps_weight_resident_when_budget_allows(self):
        module = torch.nn.Linear(3, 5, bias=True).requires_grad_(False)
        input_tensor = torch.randn(7, 3)

        with mock.patch.object(self.distorch, "_COMPUTE_GPU_RESERVE_BYTES", 0), mock.patch.object(
            self.distorch.mm, "get_free_memory", lambda *_: 1 << 20
        ):
            output = self.distorch._run_tiled_linear_on_compute(
                input_tensor, module.weight, module.bias, "cpu"
            )

        self.assertTrue(torch.allclose(output, module(input_tensor)))

    def test_mixed_conv_matches_full_conv(self):
        cases = (
            (torch.nn.Conv2d, dict(kernel_size=3, padding=1), (2, 3, 9, 7), 1 << 20),
            (torch.nn.Conv2d, dict(kernel_size=3, padding=1), (2, 3, 9, 7), 400),
            (torch.nn.Conv2d, dict(kernel_size=3, stride=2), (2, 3, 9, 7), 400),
            (torch.nn.Conv2d, dict(kernel_size=3, padding=2, dilation=2), (2, 3, 9, 7), 400),
            (torch.nn.Conv2d, dict(kernel_size=1), (2, 3, 9, 7), 64),
            (torch.nn.Conv2d, dict(kernel_size=3, padding=1, padding_mode="reflect"), (1, 3, 9, 7), 400),
            (torch.nn.Conv3d, dict(kernel_size=3, padding=1), (1, 3, 5, 9, 7), 1 << 20),
            (torch.nn.Conv3d, dict(kernel_size=3, padding=(0, 1, 1)), (1, 3, 5, 9, 7), 2000),
            (torch.nn.Conv3d, dict(kernel_size=3, stride=(1, 2, 2), padding=1), (2, 3, 6, 9, 7), 2000),
        )
        for conv_type, kwargs, shape, free_memory in cases:
            with self.subTest(conv=conv_type.__name__, kwargs=kwargs, free_memory=free_memory):
                module = conv_type(3, 5, **kwargs).requires_grad_(False)
                input_tensor = torch.randn(shape)
                expected = module(input_tensor)
                self.assertTrue(self.distorch.configure_mixed_gemm(module, "cpu"))
                with mock.patch.object(
                    self.distorch, "_COMPUTE_GPU_RESERVE_BYTES", 0
                ), mock.patch.object(
                    self.distorch.mm, "get_free_memory", lambda *_: free_memory
                ):
                    output = module(input_tensor)

                self.assertTrue(torch.allclose(output, expected, atol=1e-5))

    def test_mixed_conv3d_honors_causal_zero_autopad(self):
        module = torch.nn.Conv3d(3, 5, 3, padding=(0, 1, 1)).requires_grad_(False)
        input_tensor = torch.randn(1, 3, 1, 6, 6)
        self.distorch.configure_mixed_gemm(module, "cpu")

        output = module._conv_forward(
            input_tensor, module.weight, module.bias, autopad="causal_zero"
        )

        expected = functional.conv3d(
            input_tensor, module.weight[:, :, -1:], module.bias, padding=(0, 1, 1)
        )
        self.assertTrue(torch.allclose(output, expected, atol=1e-5))

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "needs two GPUs")
    def test_mixed_conv3d_tiles_run_on_kitchen_conv(self):
        import comfy_kitchen

        storage, compute = torch.device("cuda:1"), torch.device("cuda:0")
        module = torch.nn.Conv3d(64, 40, 3, stride=(1, 2, 2), padding=1).half().requires_grad_(False)
        input_tensor = torch.randn(1, 64, 5, 18, 14, dtype=torch.float16)
        expected = functional.conv3d(input_tensor.to(compute), module.weight.to(compute), module.bias.to(compute),
                                     stride=(1, 2, 2), padding=1)
        packed_on, kitchen_padding = [], []

        def dense_conv(x, weight, bias, stride, padding, dilation):
            kitchen_padding.append(padding)
            return comfy_kitchen.fp16_packed_conv3d(x, weight, bias, stride, padding, dilation)

        module.to(storage)
        self.assertTrue(self.distorch.configure_mixed_gemm(module, compute))
        with (
            mock.patch.object(self.distorch.comfy.ops, "_packed_conv_wanted",
                              lambda conv, x, weight: packed_on.append(weight.device) or True),
            mock.patch.object(self.distorch.comfy.ops, "dense_conv", dense_conv, create=True),
            mock.patch.object(self.distorch, "_CONV_TILE_BUDGET_BYTES", 1 << 18),
            mock.patch.object(self.distorch.mm, "get_free_memory", lambda *_: 1 << 40),
        ):
            output = module(input_tensor.to(storage))

        self.assertEqual(packed_on, [compute])
        self.assertGreater(len(kitchen_padding), 1)
        self.assertEqual(set(kitchen_padding), {(0, 0, 0)})
        self.assertEqual(output.device, compute)
        error = (output.to(compute).float() - expected.float()).norm() / expected.float().norm()
        self.assertLess(error.item(), 2e-3)

    def test_mixed_gemm_rejects_grouped_conv(self):
        with self.assertLogs("MultiGPU", level="INFO"):
            self.assertFalse(
                self.distorch.configure_mixed_gemm(
                    torch.nn.Conv2d(2, 2, 3, padding=1, groups=2), "cuda:0"
                )
            )

    def test_attention_runs_whole_when_qkv_are_on_the_compute_device(self):
        q, k, v = (torch.randn(1, 2, 9, 4) for _ in range(3))
        calls = []

        def attention(q, k, v, heads, mask=None, skip_reshape=False, **kwargs):
            calls.append(heads)
            out = functional.scaled_dot_product_attention(q, k, v)
            return out.transpose(1, 2).reshape(q.shape[0], -1, heads * q.shape[-1])

        with mock.patch.object(self.distorch.mm, "get_free_memory", lambda *_: 700):
            output = self.distorch._run_attention_on_compute(
                attention, q, k, v, 2, compute_device=torch.device("cpu"), plan={}, skip_reshape=True
            )

        self.assertEqual(calls, [2])
        self.assertTrue(
            torch.allclose(output, attention(q, k, v, 2, skip_reshape=True), atol=1e-6)
        )

    @unittest.skipUnless(torch.cuda.is_available(), "needs a compute GPU")
    def test_attention_chunks_heads_of_spilled_qkv_over_the_whole_sequence(self):
        compute = torch.device("cuda:0")
        q, k, v = (torch.randn(1, 2, 9, 4) for _ in range(3))
        calls = []

        def attention(q, k, v, heads, mask=None, skip_reshape=False, **kwargs):
            calls.append((q.device, q.shape[2]))
            out = functional.scaled_dot_product_attention(q, k, v)
            return out.transpose(1, 2).reshape(q.shape[0], -1, heads * q.shape[-1])

        with mock.patch.object(self.distorch, "_ATTENTION_RESERVE_BYTES", 0), mock.patch.object(
            self.distorch.mm, "get_free_memory", lambda *_: 700
        ):
            output = self.distorch._run_attention_on_compute(
                attention, *(self.spill(t) for t in (q, k, v)), 2,
                compute_device=compute, plan={}, skip_reshape=True,
            )
            torch.cuda.synchronize()

        self.assertEqual(calls, [(compute, 9), (compute, 9)])
        self.assertTrue(
            torch.allclose(output.cpu(), attention(q, k, v, 2, skip_reshape=True), atol=1e-6)
        )

    @unittest.skipUnless(torch.cuda.is_available(), "needs a compute GPU")
    def test_attention_chunks_heads_with_grouped_kv(self):
        compute = torch.device("cuda:0")
        q = torch.randn(1, 5, 4 * 3)
        k, v = (torch.randn(1, 7, 2 * 3) for _ in range(2))
        head_counts = []

        def attention(q, k, v, heads, mask=None, skip_output_reshape=False, **kwargs):
            head_counts.append(heads)
            q, k, v = (t.unflatten(-1, (-1, 3)).transpose(1, 2) for t in (q, k, v))
            out = functional.scaled_dot_product_attention(q, k, v, enable_gqa=True)
            if skip_output_reshape:
                return out
            return out.transpose(1, 2).flatten(2)

        for skip_output_reshape in (False, True):
            with mock.patch.object(self.distorch, "_ATTENTION_RESERVE_BYTES", 0), mock.patch.object(
                self.distorch.mm, "get_free_memory", lambda *_: 2 * 7 * 3 * 4 * 2
            ):
                output = self.distorch._run_attention_on_compute(
                    attention, *(self.spill(t) for t in (q, k, v)), 4,
                    compute_device=compute, plan={}, skip_output_reshape=skip_output_reshape,
                )
                torch.cuda.synchronize()
            expected = attention(q, k, v, 4, skip_output_reshape=skip_output_reshape)
            self.assertTrue(torch.allclose(output.cpu(), expected, atol=1e-6))
        self.assertIn(2, head_counts)

    def test_attention_shrinks_chunks_after_oom(self):
        q, k, v = (torch.randn(1, 4, 8, 3) for _ in range(3))
        plan = {}

        def attention(q, k, v, heads, mask=None, **kwargs):
            if q.shape[1] * q.shape[2] > 4:
                raise torch.OutOfMemoryError("out of memory")
            return functional.scaled_dot_product_attention(q, k, v)

        with mock.patch.object(self.distorch.mm, "get_free_memory", lambda *_: 1 << 30):
            output = self.distorch._run_attention_on_compute(
                attention, q, k, v, 4, compute_device=torch.device("cpu"), plan=plan,
                skip_reshape=True, skip_output_reshape=True,
            )

        self.assertTrue(
            torch.allclose(output, functional.scaled_dot_product_attention(q, k, v), atol=1e-6)
        )
        chunk_heads, chunk_tokens = plan[(q.shape, k.shape)]
        self.assertEqual((chunk_heads, chunk_tokens), (1, 4))

    @unittest.skipUnless(torch.cuda.is_available(), "needs a compute GPU")
    def test_streamed_activations_bring_spilled_inputs_back_to_compute(self):
        compute = torch.device("cuda:0")
        x = torch.randn(4, 8)
        spilled = self.spill(x)
        with self.distorch._StreamedActivations(compute):
            y = (functional.layer_norm(spilled, (8,)) * 2 + 1).transpose(0, 1).contiguous()
            y.add_(1)
            y[0] = 5
            spilled.mul_(2)
            filled = torch.empty(3, device=compute).fill_(2)
        torch.cuda.synchronize()
        expected = (functional.layer_norm(x, (8,)) * 2 + 1).transpose(0, 1) + 1
        expected[0] = 5
        self.assertEqual((y.device, filled.device, spilled.device), (compute, compute, torch.device("cpu")))
        self.assertTrue(torch.allclose(y.cpu(), expected, atol=1e-5))
        self.assertTrue(torch.allclose(spilled, 2 * x))
        self.assertTrue(torch.equal(filled.cpu(), torch.full((3,), 2.0)))

    @unittest.skipUnless(torch.cuda.is_available(), "needs a compute GPU")
    def test_streamed_activations_spill_only_results_that_do_not_fit(self):
        compute = torch.device("cuda:0")
        x = torch.randn(64, 256, device=compute)
        with mock.patch.object(self.distorch, "_STREAM_DIRECT_BYTES", 1000):
            with mock.patch.object(self.distorch.mm, "get_free_memory", lambda *_: 1 << 40), \
                    self.distorch._StreamedActivations(compute):
                kept = functional.silu(x)
            with mock.patch.object(
                self.distorch.mm, "get_free_memory", lambda *_: self.distorch._COMPUTE_GPU_RESERVE_BYTES + 30000
            ), self.distorch._StreamedActivations(compute):
                spilled = functional.silu(x) + 1
            torch.cuda.synchronize()

        self.assertEqual(kept.device, compute)
        self.assertTrue(self.distorch._is_spilled(spilled))
        self.assertTrue(spilled.is_pinned())
        self.assertTrue(torch.allclose(spilled, (functional.silu(x) + 1).cpu(), atol=1e-6))

    def test_streamed_activations_allow_legacy_ctor_on_subclass(self):
        class Quantized(torch.Tensor):
            def __new__(cls, *args, tensor_type, **kwargs):
                return super().__new__(cls, *args, **kwargs)

            def __init__(self, *args, tensor_type, **kwargs):
                super().__init__()

        weight = Quantized(torch.ones(2), tensor_type=None)
        with self.distorch._StreamedActivations("cpu"):
            plain = torch.Tensor(weight)
        self.assertIs(type(plain), torch.Tensor)

    @unittest.skipUnless(torch.cuda.is_available(), "needs a compute GPU")
    def test_streamed_activations_tile_ops_that_do_not_fit(self):
        compute = torch.device("cuda:0")
        x, y, v = torch.randn(6, 40, 32), torch.randn(1, 40, 32), torch.randn(32)

        def run(x, y, v):
            shifted = x.clone()
            shifted.mul_(y).add_(v)
            return [
                functional.silu(x) * y + v,
                shifted,
                x.softmax(1),
                functional.layer_norm(x, (32,), v, v),
                functional.rms_norm(x, (32,), v, 1e-6),
                x.pow(2).mean(-1, keepdim=True),
                x.sum(dim=1),
                torch.cat([x, y.expand_as(x)], dim=1),
                torch.bmm(x, x.transpose(1, 2)),
                x.to(torch.float16),
                functional.pad(x, (1, 2, 0, 1)),
            ]

        expected = run(x, y, v)
        tiled = []
        original = self.distorch._StreamedActivations._tiled

        def spy(mode, func, *args):
            tiled.append(func._schema.name)
            return original(mode, func, *args)

        with (
            mock.patch.object(self.distorch, "_STREAM_DIRECT_BYTES", 1000),
            mock.patch.object(self.distorch.mm, "get_free_memory", lambda *_: self.distorch._COMPUTE_GPU_RESERVE_BYTES + 60000),
            mock.patch.object(self.distorch._StreamedActivations, "_tiled", spy),
            self.distorch._StreamedActivations(compute),
        ):
            actual = run(self.spill(x), self.spill(y), self.spill(v))
        torch.cuda.synchronize()

        for output, reference in zip(actual, expected):
            self.assertTrue(torch.allclose(output.cpu().float(), reference.float(), atol=1e-4, rtol=1e-4))
        self.assertTrue({"aten::_softmax", "aten::native_layer_norm", "aten::bmm", "aten::cat", "aten::mean", "aten::mul_"} <= set(tiled))

    @unittest.skipUnless(torch.cuda.is_available(), "needs a compute GPU")
    def test_streamed_activations_tile_fused_rope_on_packed_qkv(self):
        import comfy_kitchen

        compute = torch.device("cuda:0")
        qkv = torch.randn(2, 12, 3, 4, 16)
        freqs_cis = torch.randn(2, 12, 1, 8, 2, 2)
        scale = torch.rand(16) + 0.5

        def run(qkv, freqs_cis, scale):
            q, k = qkv[:, :, 0], qkv[:, :, 1]
            torch.ops.comfy_kitchen.rms_rope_split_half_(q, k, freqs_cis, scale, scale)
            return q, k

        expected = run(qkv.clone().to(compute), freqs_cis.to(compute), scale.to(compute))
        tiled = []
        original = self.distorch._StreamedActivations._tiled

        def spy(mode, func, *args):
            tiled.append(func._schema.name)
            return original(mode, func, *args)

        packed = self.spill(qkv)
        with (
            mock.patch.object(self.distorch, "_STREAM_DIRECT_BYTES", 100),
            mock.patch.object(self.distorch.mm, "get_free_memory", lambda *_: self.distorch._COMPUTE_GPU_RESERVE_BYTES + 6000),
            mock.patch.object(self.distorch._StreamedActivations, "_tiled", spy),
            self.distorch._StreamedActivations(compute),
        ):
            actual = run(packed, self.spill(freqs_cis), self.spill(scale))
        torch.cuda.synchronize()

        self.assertIn("comfy_kitchen::rms_rope_split_half_", tiled)
        for output, reference in zip(actual, expected):
            self.assertEqual(output.device.type, "cpu")
            self.assertTrue(torch.allclose(output.cpu(), reference.cpu(), atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.equal(packed[:, :, 2].cpu(), qkv[:, :, 2]))

    @unittest.skipUnless(torch.cuda.is_available(), "needs a compute GPU")
    def test_streamed_activations_tile_functional_rms_rope(self):
        import comfy_kitchen

        compute = torch.device("cuda:0")
        # Lumina's layout: [B, S, H, D] q and k, freqs_cis broadcast over heads; k has fewer heads (GQA)
        q, k = torch.randn(2, 12, 4, 16), torch.randn(2, 12, 2, 16)
        freqs_cis = torch.randn(2, 12, 1, 8, 2, 2)
        q_scale, k_scale = torch.rand(16) + 0.5, torch.rand(16) + 0.5

        def run(q, k, freqs_cis, q_scale, k_scale):
            return (*torch.ops.comfy_kitchen.rms_rope(q, q, freqs_cis, q_scale, k_scale),
                    torch.ops.comfy_kitchen.rms_rope1(k, freqs_cis, k_scale))

        expected = run(*(t.to(compute) for t in (q, k, freqs_cis, q_scale, k_scale)))
        tiled = []
        original = self.distorch._StreamedActivations._tiled

        def spy(mode, func, *args):
            tiled.append(func._schema.name)
            return original(mode, func, *args)

        with (
            mock.patch.object(self.distorch, "_STREAM_DIRECT_BYTES", 100),
            mock.patch.object(self.distorch.mm, "get_free_memory", lambda *_: self.distorch._COMPUTE_GPU_RESERVE_BYTES + 6000),
            mock.patch.object(self.distorch._StreamedActivations, "_tiled", spy),
            self.distorch._StreamedActivations(compute),
        ):
            # The norm scales follow the spilled activation to the CPU without being spilled
            # themselves, as cast_bias_weight leaves them.
            actual = run(*(self.spill(t) for t in (q, k, freqs_cis)), q_scale, k_scale)
        torch.cuda.synchronize()

        self.assertTrue({"comfy_kitchen::rms_rope", "comfy_kitchen::rms_rope1"} <= set(tiled))
        for output, reference in zip(actual, expected):
            self.assertTrue(torch.allclose(output.cpu(), reference.cpu(), atol=1e-5, rtol=1e-5))

    @unittest.skipUnless(torch.cuda.is_available(), "needs a compute GPU")
    def test_streamed_activations_keep_every_result_when_one_does_not_fit(self):
        import comfy_kitchen

        compute = torch.device("cuda:0")
        q, k = torch.randn(2, 64, 4, 16), torch.randn(2, 64, 4, 16)
        freqs_cis = torch.randn(2, 64, 1, 8, 2, 2)
        scale = torch.rand(16) + 0.5
        expected = torch.ops.comfy_kitchen.rms_rope(*(t.to(compute) for t in (q, k, freqs_cis, scale, scale)))
        empty = torch.empty
        calls = []

        def second_result_out_of_memory(*args, **kwargs):
            # The first result fits on the compute GPU and the second does not.
            if torch.device(kwargs.get("device", "cpu")) == compute and tuple(args[0]) == tuple(q.shape):
                calls.append(None)
                if len(calls) == 2:
                    raise torch.OutOfMemoryError("out of memory")
            return empty(*args, **kwargs)

        with (
            mock.patch.object(self.distorch, "_STREAM_DIRECT_BYTES", 100),
            mock.patch.object(self.distorch.mm, "get_free_memory", lambda *_: self.distorch._COMPUTE_GPU_RESERVE_BYTES + 100000),
            self.distorch._StreamedActivations(compute),
            mock.patch.object(torch, "empty", second_result_out_of_memory),
        ):
            actual = torch.ops.comfy_kitchen.rms_rope(*(self.spill(t) for t in (q, k, freqs_cis)), scale, scale)
        torch.cuda.synchronize()

        self.assertEqual(len(calls), 2)
        self.assertEqual(len(actual), 2)
        for output, reference in zip(actual, expected):
            self.assertTrue(torch.allclose(output.cpu(), reference.cpu(), atol=1e-5, rtol=1e-5))

    @unittest.skipUnless(torch.cuda.is_available(), "needs a compute GPU")
    def test_streamed_activations_report_ops_that_cannot_be_tiled(self):
        compute = torch.device("cuda:0")
        x = self.spill(torch.randn(4, 64))
        with (
            mock.patch.object(self.distorch, "_STREAM_DIRECT_BYTES", 100),
            mock.patch.object(self.distorch.mm, "get_free_memory", lambda *_: self.distorch._COMPUTE_GPU_RESERVE_BYTES),
            self.distorch._StreamedActivations(compute),
            self.assertRaisesRegex(torch.OutOfMemoryError, "no tiling rule"),
        ):
            x.sort(dim=-1)

    def test_attention_raises_out_of_memory_when_nothing_fits(self):
        q, k, v = (torch.randn(1, 2, 4, 3) for _ in range(3))

        def attention(*args, **kwargs):
            raise torch.OutOfMemoryError("out of memory")

        with mock.patch.object(self.distorch.mm, "get_free_memory", lambda *_: 1 << 30):
            with self.assertRaises(torch.OutOfMemoryError):
                self.distorch._run_attention_on_compute(
                    attention, q, k, v, 2, compute_device=torch.device("cpu"), plan={},
                    skip_reshape=True, skip_output_reshape=True,
                )


    def test_mixed_execution_moves_inputs_to_compute_and_routes_attention(self):
        calls = []

        class DiffusionModel(torch.nn.Module):
            def forward(self, x, timestep, transformer_options={}):
                calls.append(transformer_options)
                return [x[0] * 2]

        model = DiffusionModel()
        transformer_options = {"patches": {}}
        schedule = types.SimpleNamespace(
            end_forward=lambda: calls.append("end_forward"), mlp_linears={}
        )
        self.distorch.configure_mixed_execution(model, "cpu", schedule)
        output = model.forward(
            [torch.ones(2)], torch.zeros(1), transformer_options=transformer_options
        )

        self.assertTrue(torch.equal(output[0], torch.full((2,), 2.0)))
        self.assertIn("optimized_attention_override", calls[0])
        self.assertEqual(calls[1], "end_forward")
        self.assertNotIn("optimized_attention_override", transformer_options)

    @unittest.skipUnless(torch.cuda.is_available(), "needs a compute GPU")
    def test_streamed_forward_keeps_activations_on_compute_and_matches_cpu(self):
        class DiffusionModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(64, 128)
                self.norm = torch.nn.LayerNorm(128)
                self.out = torch.nn.Linear(128, 64)
                self.devices = []

            def forward(self, x, timestep, transformer_options={}):
                h = self.norm(self.proj(x)) * (1 + timestep)
                self.devices.append(h.device)
                return self.out(functional.gelu(h)) + x

        model = DiffusionModel().requires_grad_(False)
        x, timestep = torch.randn(2, 32, 64), torch.full((1,), 0.5)
        expected = model(x, timestep)
        compute = torch.device("cuda:0")
        # Placement keeps modules that cannot be cast at runtime on the compute GPU.
        model.norm.to(compute)
        schedule = self.distorch._MixedSchedule(compute)
        for module in (model.proj, model.out):
            self.assertTrue(self.distorch.configure_mixed_gemm(module, compute, schedule=schedule))
        self.distorch.configure_mixed_execution(model, compute, schedule)

        for free, hidden in ((1 << 30, compute), (self.distorch._COMPUTE_GPU_RESERVE_BYTES + 20000, torch.device("cpu"))):
            with self.subTest(free=free), mock.patch.object(self.distorch, "_STREAM_DIRECT_BYTES", 1000), \
                    mock.patch.object(self.distorch.mm, "get_free_memory", lambda *_: free, create=True):
                output = model(x, timestep)
            self.assertEqual(model.devices[-1], hidden)
            self.assertEqual(output.device, x.device)
            self.assertTrue(torch.allclose(output, expected, atol=1e-4))

    @unittest.skipUnless(torch.cuda.is_available(), "needs a compute GPU")
    def test_streamed_forward_synchronizes_only_to_evict_each_step(self):
        compute = torch.device("cuda:0")
        model = torch.nn.Sequential(*(torch.nn.Linear(256, 256) for _ in range(3))).requires_grad_(False)

        class DiffusionModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = model

            def forward(self, x, timestep, transformer_options={}):
                return self.layers(x) * 2

        diffusion = DiffusionModel()
        x = torch.randn(512, 256)
        expected = diffusion(x, None)
        x = x.to(compute)
        schedule = self.distorch._MixedSchedule(compute)
        for module in model:
            self.distorch.configure_mixed_gemm(module, compute, schedule=schedule)
        self.distorch.configure_mixed_execution(diffusion, compute, schedule)
        with mock.patch.object(self.distorch.mm, "get_free_memory", lambda *_: 1 << 30, create=True), \
                mock.patch.object(torch.cuda, "synchronize", wraps=torch.cuda.synchronize) as synchronize, \
                mock.patch.object(torch.cuda, "empty_cache", wraps=torch.cuda.empty_cache) as empty_cache:
            for _ in range(2):
                output = diffusion(x, None)
                # Nothing staged for this forward stays on the compute GPU.
                self.assertTrue(all(module._mgpu_staged is None for module in model))
        self.assertEqual((synchronize.call_count, empty_cache.call_count), (2, 2))
        self.assertEqual(output.device, compute)
        self.assertTrue(torch.allclose(output.cpu(), expected, atol=1e-4))

    def test_prepared_linear_matches_full_linear(self):
        module = torch.nn.Linear(3, 5, bias=True).requires_grad_(False)
        input_tensor = torch.randn(2, 3)

        with self.assertLogs("MultiGPU", level="INFO") as logs:
            output = self.distorch._run_prepared_linear_on_compute(input_tensor, module, "cpu")

        self.assertTrue(torch.allclose(output, module(input_tensor)))
        self.assertEqual(output.shape, (2, 5))
        self.assertIn("Weight preparation confirmed on cpu", logs.output[0])
        self.assertIn("prepared weight is on cpu", logs.output[0])

    def test_materializes_cast_weights_on_compute(self):
        module = torch.nn.Linear(3, 5, bias=True).requires_grad_(False)
        module.comfy_cast_weights = True
        module.weight_function = []
        module.bias_function = []
        input_tensor = torch.randn(2, 3)
        calls = []
        comfy_ops = types.ModuleType("comfy.ops")

        def cast_bias_weight(module, **kwargs):
            calls.append(kwargs)
            return module.weight + 1, module.bias, "offload-state"

        comfy_ops.cast_bias_weight = cast_bias_weight
        comfy_ops.uncast_bias_weight = lambda *args: calls.append("uncast")

        with mock.patch.dict(sys.modules, {"comfy.ops": comfy_ops}):
            output = self.distorch._run_prepared_linear_on_compute(input_tensor, module, "cpu")

        self.assertTrue(
            torch.allclose(
                output, functional.linear(input_tensor, module.weight + 1, module.bias)
            )
        )
        self.assertEqual(calls[0]["device"], "cpu")
        self.assertTrue(calls[0]["offloadable"])
        self.assertEqual(calls[1], "uncast")

    def lora_adapter_module(self):
        lora = types.ModuleType("comfy.weight_adapter.lora")

        class LoRAAdapter:
            def __init__(self, weights):
                self.weights = weights

        lora.LoRAAdapter = LoRAAdapter
        return lora

    def test_tiled_linear_applies_lora_as_low_rank_gemm(self):
        weight, bias = torch.randn(5, 3), torch.randn(5)
        up, down = torch.randn(5, 2), torch.randn(2, 3)
        input_tensor = torch.randn(4, 3)
        merged = weight + 0.25 * (up @ down)

        with mock.patch.object(self.distorch, "_COMPUTE_GPU_RESERVE_BYTES", 0), mock.patch.object(
            self.distorch.mm, "get_free_memory", lambda *_: 64
        ):
            output = self.distorch._run_tiled_linear_on_compute(
                input_tensor, weight, bias, "cpu", [(down, up, 0.25)]
            )

        self.assertTrue(torch.allclose(output, functional.linear(input_tensor, merged, bias), atol=1e-5))

    def test_prepared_linear_applies_plain_lora_as_low_rank_gemm(self):
        lora_module = self.lora_adapter_module()
        up, down = torch.randn(5, 2), torch.randn(2, 3)
        adapter = lora_module.LoRAAdapter((up, down, torch.tensor(4.0), None, None, None))
        module = torch.nn.Linear(3, 5, bias=False).requires_grad_(False)
        patches = [([(0.5, adapter, 1.0, None, None)], "linear.weight")]
        module.weight.patches = patches
        seen = []

        def cast_bias_weight(dtype, device):
            seen.append(list(module.weight.patches))
            return module.weight, None

        module.cast_bias_weight = cast_bias_weight
        input_tensor = torch.randn(2, 3)

        with mock.patch.dict(sys.modules, {"comfy.weight_adapter.lora": lora_module}):
            output = self.distorch._run_prepared_linear_on_compute(input_tensor, module, "cpu")

        # scale = strength * alpha / rank = 0.5 * 4 / 2
        expected = functional.linear(input_tensor, module.weight + 1.0 * (up @ down))
        self.assertTrue(torch.allclose(output, expected, atol=1e-5))
        self.assertEqual(seen, [[]])
        self.assertIs(module.weight.patches, patches)

    def test_non_plain_lora_patches_stay_merged(self):
        lora_module = self.lora_adapter_module()
        up, down = torch.randn(5, 2), torch.randn(2, 3)
        dora = lora_module.LoRAAdapter((up, down, None, None, torch.ones(5), None))

        with mock.patch.dict(sys.modules, {"comfy.weight_adapter.lora": lora_module}):
            self.assertEqual(self.distorch._plain_lora_patches([(1.0, dora, 1.0, None, None)]), ())
            self.assertEqual(
                self.distorch._plain_lora_patches([(1.0, object(), 1.0, None, None)]), ()
            )

    def placement(self, patcher, streamed, free):
        gibibyte = 1024**3
        with (
            mock.patch.object(self.distorch, "get_device_list", return_value=["cpu", "cuda:0", "cuda:1"]),
            mock.patch.object(
                self.distorch.mm, "get_free_memory", lambda device: free[str(device)] * gibibyte
            ),
            mock.patch.object(self.distorch.mm, "minimum_inference_memory", lambda: gibibyte, create=True),
        ):
            return self.distorch.place_weights(patcher, "cuda:0", streamed)

    def test_streamed_model_spreads_weights_over_other_devices_by_free_memory(self):
        modules = [AllocationModule() for _ in range(4)]
        modules.append(torch.nn.LayerNorm(2))  # not castable at runtime
        assignments = self.placement(AllocationPatcher(*modules), True, {"cuda:0": 6, "cuda:1": 30, "cpu": 10})

        placed = [assignments[f"layer{index}"] for index in range(4)]
        self.assertEqual((placed.count("cuda:1"), placed.count("cpu")), (3, 1))
        self.assertEqual(assignments["layer4"], "cuda:0")

    def test_unstreamed_model_fills_the_compute_gpu_from_the_head(self):
        modules = [AllocationModule() for _ in range(4)]
        for module in modules:
            # 1 GiB stored, as a quantized weight; 256 MiB once cast to float32.
            module.weight = torch.empty(1024, 65536, device="meta")
        patcher = AllocationPatcher(*modules)
        patcher._load_list = lambda: tuple(
            (1024**3, f"layer{index}", module, {}) for index, module in enumerate(modules)
        )
        # 1 GiB of inference reserve and the largest weight's cast leave room for two
        # layers of 4 GiB free, but only one of 3 GiB.
        for free, on_compute in ((4, 2), (3, 1)):
            assignments = self.placement(patcher, False, {"cuda:0": free, "cuda:1": 0, "cpu": 64})
            self.assertEqual(
                [assignments[f"layer{index}"] for index in range(4)],
                ["cuda:0"] * on_compute + ["cpu"] * (4 - on_compute),
            )

    def test_pins_cpu_weights_in_place(self):
        module = torch.nn.Linear(3, 5).requires_grad_(False)
        weight = module.weight
        self.distorch._pin_cpu_weights(module)
        self.assertIs(module.weight, weight)
        self.assertTrue(module.weight.is_pinned() and module.bias.is_pinned())

    def test_stages_float_weights_as_fp16_for_the_packed_gemm(self):
        module = torch.nn.Linear(16, 8).requires_grad_(False)
        self.assertTrue(self.distorch.configure_mixed_gemm(module, "cpu"))
        packed_inputs = []

        def dense_linear(x, weight, bias):
            packed_inputs.append((x.shape[0], weight.dtype))
            return functional.linear(x, weight.to(x.dtype), bias)

        with (
            mock.patch.object(self.distorch.comfy.ops, "_CK_PACKED", True, create=True),
            mock.patch.object(
                self.distorch.comfy.ops,
                "quant_ops",
                types.SimpleNamespace(ck=types.SimpleNamespace(fp16_packed_linear_is_accelerated=lambda _: True)),
                create=True,
            ),
            mock.patch.object(
                self.distorch.comfy.ops, "_packed_linear_wanted", lambda x, weight: x.shape[0] > 8, create=True
            ),
            mock.patch.object(self.distorch.comfy.ops, "dense_linear", dense_linear),
        ):
            staged = self.distorch._stage_linear(module, torch.float32)
            for tokens in (32, 2):
                x = torch.randn(tokens, 16)
                output = self.distorch._linear_on_compute(x, staged.weight, staged.bias, None, None)
                self.assertTrue(torch.allclose(output, functional.linear(x, module.weight, module.bias), atol=1e-2))

        self.assertEqual((staged.weight.dtype, staged.bias.dtype), (torch.float16, torch.float32))
        # Only the GEMM big enough for the packed kernel gets the float16 weight.
        self.assertEqual(packed_inputs, [(32, torch.float16), (2, torch.float32)])

    def test_staging_retries_after_releasing_cache_then_falls_back(self):
        module = torch.nn.Linear(16, 8).requires_grad_(False)
        self.assertTrue(self.distorch.configure_mixed_gemm(module, "cpu"))
        prepare = self.distorch._prepare_linear
        failures = []

        def flaky_prepare(*args):
            if len(failures) < failures_wanted:
                failures.append(None)
                raise torch.OutOfMemoryError("out of memory")
            return prepare(*args)

        for failures_wanted, staged in ((1, True), (2, False)):
            failures.clear()
            with mock.patch.object(self.distorch, "_prepare_linear", flaky_prepare), \
                    mock.patch.object(torch.cuda, "empty_cache") as empty_cache:
                result = self.distorch._stage_linear(module, torch.float32)
            self.assertEqual(result is not None, staged)
            self.assertGreaterEqual(empty_cache.call_count, 1)

    def test_mixed_mode_requires_standard_linear_tiling(self):
        self.assertFalse(self.distorch.configure_mixed_gemm(DonorModule(), "cuda:0"))

    def test_token_chunks_are_whole_gemm_tiles(self):
        chunks = self.distorch._token_chunks(4160, 1, 1 << 30)
        self.assertEqual([chunk.stop - chunk.start for chunk in chunks], [512] * 8 + [64])
        # No budget or chunk count makes a chunk shorter than _MIN_CHUNK_TOKENS.
        self.assertEqual(self.distorch._token_chunks(100, 1, 1 << 30), [slice(0, 100)])
        self.assertEqual(self.distorch._token_chunks(1000, 10, 3 * 10 * 100)[0], slice(0, 512))
        self.assertEqual(self.distorch._token_chunks(4160, 10, -(1 << 20))[0], slice(0, 512))
        # Chunks that are not copied are only as many as the budget needs.
        self.assertEqual(self.distorch._token_chunks(4160, 1, 1 << 30, streamed=False), [slice(0, 4160)])

    def test_mixed_quantized_linear_runs_its_own_forward_in_token_chunks(self):
        class QuantizedLinear(torch.nn.Linear):
            layout_type = "TensorWiseINT8Layout"

        module = QuantizedLinear(3, 5).requires_grad_(False)
        chunks = []
        original_forward = module.forward

        def forward(value):
            chunks.append(value.shape[0])
            return original_forward(value)

        module.forward = forward
        input_tensor = torch.randn(2, 550, 3)
        expected = module(input_tensor)
        chunks.clear()

        self.assertTrue(self.distorch.configure_mixed_gemm(module, "cpu"))
        # 1100 tokens need 2 * (3 + 5) * 4 bytes each; the weight takes 72. Short of
        # that, chunks shrink only to _MIN_CHUNK_TOKENS.
        for free, expected_chunks in ((1100 * 64, [1100]), (384, [512, 512, 76])):
            chunks.clear()
            with (
                mock.patch.object(self.distorch.mm, "get_free_memory", lambda *_: self.distorch._COMPUTE_GPU_RESERVE_BYTES + 72 + free, create=True),
                mock.patch.object(self.distorch.mm, "module_size", lambda _: 72, create=True),
            ):
                output = module.forward(input_tensor)

            self.assertEqual(chunks, expected_chunks)
            self.assertTrue(torch.allclose(output, expected))

    def test_token_chunks_give_up_when_floor_chunks_do_not_fit(self):
        calls = []

        def run(chunk):
            calls.append(chunk.shape[0])
            raise torch.OutOfMemoryError("out of memory")

        with self.assertRaises(torch.OutOfMemoryError):
            self.distorch._run_token_chunks(torch.randn(2000, 4), 4, run, 64, 1 << 20, "cpu")
        # Halved until the chunks reach _MIN_CHUNK_TOKENS, tried once more there, then raised.
        self.assertEqual(calls[-1], 512)
        self.assertLess(len(calls), 10)

    def test_mixed_linear_unfolds_fused_prologue_and_residual(self):
        """Linear(x, input_act=..., residual=...) keeps its GEMM on the compute path."""
        module = torch.nn.Linear(3, 5).requires_grad_(False)
        norm = torch.rand(3) + 0.5
        ops = types.ModuleType("comfy.ops")
        ops._eager_input_act = lambda x, input_act, act_weight: functional.rms_norm(x, (3,), act_weight)
        ops._linear_residual = lambda out, residual, scale: torch.addcmul(residual, out, scale)
        ops.dense_linear = functional.linear
        gemm_inputs = []
        run_mixed_linear = self.distorch._run_mixed_linear

        def spy(value, linear):
            gemm_inputs.append(value)
            return run_mixed_linear(value, linear)

        input_tensor, residual, scale = torch.randn(2, 4, 3), torch.randn(2, 4, 5), torch.rand(5)
        free_bytes = self.distorch._COMPUTE_GPU_RESERVE_BYTES + (1 << 20)
        with (
            mock.patch.dict(sys.modules, {"comfy.ops": ops}),
            mock.patch.object(sys.modules["comfy"], "ops", ops, create=True),
            mock.patch.object(self.distorch, "_run_mixed_linear", spy),
            mock.patch.object(self.distorch.mm, "get_free_memory", lambda *_: free_bytes, create=True),
        ):
            self.assertTrue(self.distorch.configure_mixed_gemm(module, "cpu"))
            output = module(input_tensor, input_act="rms_norm", act_weight=norm,
                            residual=residual, residual_scale=scale)

        normed = functional.rms_norm(input_tensor, (3,), norm)
        self.assertEqual(len(gemm_inputs), 1)
        self.assertTrue(torch.allclose(gemm_inputs[0], normed))
        expected = torch.addcmul(residual, functional.linear(normed, module.weight, module.bias), scale)
        self.assertTrue(torch.allclose(output, expected, atol=1e-5))

    @unittest.skipUnless(torch.cuda.is_available(), "weight prefetch uses CUDA streams")
    def test_mixed_linears_learn_their_order_and_stage_the_next_weight(self):
        device = torch.device("cuda:0")
        schedule = self.distorch._MixedSchedule(device)
        first, second = (torch.nn.Linear(256, 256).requires_grad_(False) for _ in range(2))
        for module in (first, second):
            self.assertTrue(self.distorch.configure_mixed_gemm(module, device, schedule=schedule))
        x = torch.randn(64, 256, device=device)
        expected = x
        for module in (first, second):
            expected = functional.linear(expected, module.weight.to(device), module.bias.to(device))

        with mock.patch.object(self.distorch.mm, "get_free_memory", lambda *_: 1 << 30, create=True):
            for _ in range(2):
                schedule.end_forward()
                output = second(first(x))
                self.assertEqual(output.device, device)
                self.assertTrue(torch.allclose(output, expected, atol=1e-4))
                self.assertIs(schedule.following[first], second)
            schedule.end_forward()
            first(x)
            self.assertIsNotNone(second._mgpu_staged)
            schedule.end_forward()

        self.assertIsNone(second._mgpu_staged)
        self.assertEqual(list(first.children()), [])

    @unittest.skipUnless(torch.cuda.is_available(), "weight prefetch uses CUDA streams")
    def test_mixed_quantized_linears_stage_each_weight_once_per_forward(self):
        class QuantizedLinear(torch.nn.Linear):
            layout_type = "TensorWiseINT8Layout"

        device = torch.device("cuda:0")
        schedule = self.distorch._MixedSchedule(device)
        linears = [QuantizedLinear(256, 256).requires_grad_(False) for _ in range(2)]
        for module in linears:
            self.distorch.configure_mixed_gemm(module, device, schedule=schedule)
        weights = [(module.weight, module.bias) for module in linears]
        x = torch.randn(64, 256, device=device)
        expected = x
        for module in linears:
            expected = functional.linear(expected, module.weight.to(device), module.bias.to(device))
        staged = []
        stage_linear = self.distorch._stage_linear

        def counting_stage(*args, **kwargs):
            staged.append(args[0])
            return stage_linear(*args, **kwargs)

        with mock.patch.object(self.distorch.mm, "get_free_memory", lambda *_: 1 << 30, create=True), \
                mock.patch.object(self.distorch.mm, "module_size", lambda _: 256 * 256 * 4, create=True), \
                mock.patch.object(self.distorch, "_stage_linear", counting_stage):
            for _ in range(2):
                schedule.end_forward()
                staged.clear()
                output = x
                for module in linears:
                    output = module(output)
                torch.cuda.synchronize(device)
                self.assertTrue(torch.allclose(output.to(device), expected, atol=1e-3))

        self.assertEqual(staged, linears)
        for module, (weight, bias) in zip(linears, weights):
            self.assertIs(module.weight, weight)
            self.assertIs(module.bias, bias)

    @unittest.skipUnless(torch.cuda.is_available(), "weight prefetch uses CUDA streams")
    def test_fused_mlp_runs_whole_on_compute(self):
        device = torch.device("cuda:0")

        class FeedForward(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = torch.nn.Linear(256, 512, bias=False)
                self.w2 = torch.nn.Linear(512, 256, bias=False)
                self.w3 = torch.nn.Linear(256, 512, bias=False)

            def forward(self, x):
                return self.w2(functional.silu(self.w1(x)) * self.w3(x))

        class SwiGLU(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(256, 1024, bias=False)
                self.fc2 = torch.nn.Linear(512, 256, bias=False)

            def forward(self, x):
                gate, value = self.fc1(x).chunk(2, dim=-1)
                return self.fc2(functional.silu(gate) * value)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(256, 256)
                self.feed_forward = FeedForward()
                self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(256, 512), torch.nn.GELU(), torch.nn.Linear(512, 256)
                )
                self.swiglu = SwiGLU()

            def forward(self, x, timestep, transformer_options={}):
                return self.swiglu(self.mlp(self.feed_forward(self.proj(x))))

        model = Model().requires_grad_(False)
        x = torch.randn(2, 64, 256)
        expected = model.forward(x, None)
        schedule = self.distorch._MixedSchedule(device)
        for module in model.modules():
            self.distorch.configure_mixed_gemm(module, device, schedule=schedule)
        self.distorch.configure_mixed_execution(model, device, schedule)
        self.assertEqual(
            schedule.mlp_linears,
            {
                model.feed_forward: [model.feed_forward.w1, model.feed_forward.w3, model.feed_forward.w2],
                model.mlp: [model.mlp[0], model.mlp[2]],
                model.swiglu: [model.swiglu.fc1, model.swiglu.fc2],
            },
        )
        hidden_devices = []
        model.mlp[1].register_forward_hook(
            lambda _, inputs, __: hidden_devices.append(inputs[0].device)
        )

        with mock.patch.object(self.distorch.mm, "get_free_memory", lambda *_: 1 << 30, create=True):
            for _ in range(2):
                output = model.forward(x, None)
                self.assertEqual(output.device, x.device)
                self.assertTrue(torch.allclose(output, expected, atol=1e-4))
            self.assertIs(schedule.following[model.feed_forward], model.mlp)
        self.assertTrue(hidden_devices)
        self.assertTrue(all(hidden == device for hidden in hidden_devices))
        self.assertTrue(all(linear._mgpu_fused is None for linear in model.mlp[::2]))

    def test_fused_mlps_hold_only_staged_linears_and_activations(self):
        schedule = object()
        normed = torch.nn.Sequential(torch.nn.LayerNorm(4), torch.nn.Linear(4, 4))
        single = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(4, 4))
        unstaged = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
        normed[1]._mgpu_schedule = single[1]._mgpu_schedule = schedule
        unstaged[0]._mgpu_schedule = schedule

        for module in (normed, single, unstaged):
            self.assertIsNone(self.distorch._fused_mlp_linears(module, schedule))

    def test_accepts_comfy_mixed_precision_linear_for_compute_tiling(self):
        class MixedPrecisionLinear(torch.nn.Module):
            quant_format = "int8_tensorwise"

            def __init__(self):
                super().__init__()
                self.in_features, self.out_features = 3, 5
                self.weight = torch.nn.Parameter(torch.empty(5, 3), requires_grad=False)

        self.assertTrue(self.distorch._can_tile_linear_on_compute(MixedPrecisionLinear()))

    def test_rejects_packed_linear_weight_for_compute_tiling(self):
        module = torch.nn.Linear(1, 1, bias=False)
        module.in_features = 5376
        module.out_features = 16128
        module.weight = torch.nn.Parameter(torch.empty(3024, 3120, device="meta"))
        module.weight.tensor_type = "Q4_K"

        self.assertFalse(self.distorch._can_tile_linear_on_compute(module))

    def test_accepts_packed_linear_with_donor_materializer(self):
        module = torch.nn.Linear(1, 1, bias=False)
        module.in_features = 5376
        module.out_features = 16128
        module.weight = torch.nn.Parameter(torch.empty(3024, 3120, device="meta"))
        module.weight.tensor_type = "Q4_K"
        module.cast_bias_weight = lambda **kwargs: (module.weight, None)

        self.assertTrue(self.distorch._can_tile_linear_on_compute(module))

    def test_logs_mixed_mode_eligibility_rejection(self):
        module = torch.nn.Linear(1, 1, bias=False)
        module.weight = torch.nn.Parameter(torch.empty(3, 3))

        with self.assertLogs("MultiGPU", level="INFO") as logs:
            self.assertFalse(self.distorch.configure_mixed_gemm(module, "cuda:0"))

        self.assertIn("Mixed compute GEMM skipped", logs.output[0])

    def test_clip_keeps_only_unquantized_embeddings_on_compute(self):
        embed = torch.nn.Embedding(4, 2)
        packed = torch.nn.Embedding(4, 2)
        packed.weight.tensor_type = "Q8_0"
        items = ((8, "embed_tokens", embed, {}), (8, "packed.embed_tokens", packed, {}))

        _, distributable, assignments, head_memory = self.distorch._extract_clip_head_blocks(
            items, "cuda:0"
        )

        self.assertEqual(assignments, {"embed_tokens": "cuda:0"})
        self.assertEqual([name for _, name, _, _ in distributable], ["packed.embed_tokens"])
        self.assertEqual(head_memory, 8)

    def test_recognizes_only_supported_software_gemm_architectures(self):
        properties = types.SimpleNamespace(gcnArchName="gfx1030:sramecc+:xnack-")
        with (
            mock.patch.object(torch.version, "hip", "6.4"),
            mock.patch.object(
                torch.cuda, "get_device_properties", return_value=properties
            ),
        ):
            self.assertTrue(self.distorch._is_hip_software_gemm_device("cuda:0"))
            properties.gcnArchName = "gfx1100"
            self.assertFalse(self.distorch._is_hip_software_gemm_device("cuda:0"))
