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
    comfy.model_management = model_management
    comfy.model_patcher = model_patcher
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = model_management
    sys.modules["comfy.model_patcher"] = model_patcher
    sys.modules["comfy.quant_ops"] = quant_ops

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


class DonorLinearModule:
    def __init__(self, device):
        self.weight = torch.eye(2, device=device)
        self.input_device = None

    def forward(self, value):
        self.input_device = value.device
        return functional.linear(value, self.weight)


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
    def __init__(self, module):
        self.model = AllocationModel(module)
        self.module = module

    def _load_list(self):
        return ((self.module.weight.nbytes, "linear", self.module, {}),)


class TestHipDonorGemmOffload(unittest.TestCase):
    def setUp(self):
        self.distorch = load_distorch_module()

    def test_executes_and_returns_through_donor_wrapper(self):
        module = DonorModule()
        moved_devices = []

        def record_move(value, device):
            moved_devices.append(torch.device(device))
            return value

        with (
            mock.patch.object(
                self.distorch, "_is_hip_software_gemm_device", return_value=True
            ),
            mock.patch.object(
                self.distorch.mm,
                "comfy_kitchen_attention_enabled",
                return_value=True,
                create=True,
            ),
            mock.patch.object(self.distorch, "_move_tensors", side_effect=record_move),
        ):
            self.assertTrue(
                self.distorch.configure_hip_donor_gemm_offload(
                    module, "cuda:0", "cuda:1", execution_mode="all"
                )
            )
            result = module.forward(torch.tensor(3), scale=2)

        self.assertEqual(result.item(), 6)
        self.assertEqual(module.calls, [(torch.tensor(3), 2)])
        self.assertEqual(module._mgpu_donor_gemm_calls, 1)
        self.assertEqual(
            moved_devices, [torch.device("cuda:1")] * 2 + [torch.device("cuda:0")]
        )

    def test_restores_original_forward_when_not_eligible(self):
        module = DonorModule()
        original_forward = module.forward

        with mock.patch.object(
            self.distorch, "_is_hip_software_gemm_device", return_value=True
        ), mock.patch.object(
            self.distorch.mm,
            "comfy_kitchen_attention_enabled",
            return_value=True,
            create=True,
        ):
            self.distorch.configure_hip_donor_gemm_offload(
                module, "cuda:0", "cuda:1", execution_mode="all"
            )
            self.assertFalse(
                self.distorch.configure_hip_donor_gemm_offload(
                    module, "cuda:0", "cuda:0"
                )
            )

        self.assertNotIn("_mgpu_original_forward", module.__dict__)
        self.assertEqual(module.forward.__func__, original_forward.__func__)

    def test_requires_two_dimensional_weight(self):
        module = DonorModule()
        module.weight = torch.empty(2)

        with mock.patch.object(
            self.distorch, "_is_hip_software_gemm_device", return_value=True
        ), mock.patch.object(
            self.distorch.mm,
            "comfy_kitchen_attention_enabled",
            return_value=True,
            create=True,
        ):
            self.assertFalse(
                self.distorch.configure_hip_donor_gemm_offload(
                    module, "cuda:0", "cuda:1", execution_mode="all"
                )
            )

    def test_mixed_linear_wraps_large_linears(self):
        module = torch.nn.Linear(1, 16 * 1024 * 1024 // 4 + 1, bias=False)

        self.assertTrue(self.distorch.configure_mixed_gemm(module, "cuda:0", "cuda:1"))
        self.assertEqual(module._mgpu_compute_device, torch.device("cuda:0"))

    def test_disabled_mode_does_not_enable_donor_gemm(self):
        module = DonorModule()

        with mock.patch.object(
            self.distorch, "_is_hip_software_gemm_device", return_value=True
        ):
            self.assertFalse(
                self.distorch.configure_hip_donor_gemm_offload(
                    module, "cuda:0", "cuda:1"
                )
            )

    def test_mixed_mode_requires_comfy_kitchen_attention(self):
        model = types.SimpleNamespace(diffusion_model=object())

        with mock.patch.object(
            self.distorch, "_is_hip_software_gemm_device", return_value=True
        ), self.assertLogs("MultiGPU", level="INFO") as logs:
            self.assertIsNone(
                self.distorch.select_mixed_donor_device(
                    model, {"a": "cuda:0", "b": "cuda:1"}, "cuda:0"
                )
            )

        self.assertIn("Comfy Kitchen attention is disabled", logs.output[0])

    def test_selects_donor_gpu_for_mixed_execution(self):
        model = types.SimpleNamespace(diffusion_model=object())

        with mock.patch.object(
            self.distorch, "_is_hip_software_gemm_device", return_value=True
        ), mock.patch.object(
            self.distorch.mm,
            "comfy_kitchen_attention_enabled",
            return_value=True,
            create=True,
        ):
            donor = self.distorch.select_mixed_donor_device(
                model, {"a": "cuda:0", "b": "cuda:1", "c": "cpu"}, "cuda:0"
            )

        self.assertEqual(donor, torch.device("cuda:1"))

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
                self.assertTrue(self.distorch.configure_mixed_gemm(module, "cpu", "cpu"))
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
        self.distorch.configure_mixed_gemm(module, "cpu", "cpu")

        output = module._conv_forward(
            input_tensor, module.weight, module.bias, autopad="causal_zero"
        )

        expected = functional.conv3d(
            input_tensor, module.weight[:, :, -1:], module.bias, padding=(0, 1, 1)
        )
        self.assertTrue(torch.allclose(output, expected, atol=1e-5))

    def test_mixed_gemm_rejects_grouped_conv(self):
        with self.assertLogs("MultiGPU", level="INFO"):
            self.assertFalse(
                self.distorch.configure_mixed_gemm(
                    torch.nn.Conv2d(2, 2, 3, padding=1, groups=2), "cuda:0", "cuda:1"
                )
            )

    def test_mixed_vae_runs_encode_and_decode_on_donor(self):
        calls = []

        class VAEModel:
            def encode(self, x, device=None):
                calls.append(device)
                return x * 2

            def decode(self, z, output_buffer=None):
                output_buffer.copy_(z + 1)
                return output_buffer

        model = VAEModel()
        output_buffer = torch.empty(2)
        self.distorch.configure_mixed_vae(model, "cpu", "cpu")

        self.assertTrue(torch.equal(model.encode(torch.ones(2), device="meta"), torch.full((2,), 2.0)))
        self.assertIs(model.decode(torch.ones(2), output_buffer=output_buffer), output_buffer)
        self.assertEqual(calls, [torch.device("cpu")])
        self.assertTrue(torch.equal(output_buffer, torch.full((2,), 2.0)))

    def test_mixed_vae_does_not_require_comfy_kitchen_attention(self):
        with mock.patch.object(
            self.distorch, "_is_hip_software_gemm_device", return_value=True
        ):
            donor = self.distorch.select_mixed_donor_device(
                object(), {"a": "cuda:0", "b": "cuda:1"}, "cuda:0", is_vae=True
            )

        self.assertEqual(donor, torch.device("cuda:1"))

    def test_mixed_int8_vae_runs_attention_on_compute_with_frames_as_heads(self):
        calls = []

        def attention(name):
            def run(q, k, v, heads, mask=None, skip_reshape=False, skip_output_reshape=False):
                calls.append((name, heads))
                return functional.scaled_dot_product_attention(q, k, v)

            return run

        attention_module = types.ModuleType("comfy.ldm.modules.attention")
        attention_module.optimized_attention = attention("optimized")
        attention_module.attention_pytorch = attention("pytorch")
        class VAEModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.block = torch.nn.Module()
                self.block.optimized_attention = None

            def encode(self, q, k, v):
                return self.block.optimized_attention(q, k, v)

        model = VAEModel()
        self.distorch.configure_mixed_vae(model, "cpu", "cpu", int8=True)

        # Kitchen attention takes head dims up to 256; VAE heads span all channels.
        for shape, name in (((2, 4, 3, 3), "optimized"), ((1, 257, 2, 2), "pytorch")):
            q, k, v = (torch.randn(shape) for _ in range(3))
            calls.clear()
            with (
                mock.patch.dict(sys.modules, {attention_module.__name__: attention_module}),
                mock.patch.object(self.distorch, "_ATTENTION_RESERVE_BYTES", 0),
                mock.patch.object(self.distorch.mm, "get_free_memory", lambda *_: 1 << 30),
            ):
                output = model.encode(q, k, v)

            q, k, v = (t.reshape(shape[0], 1, shape[1], -1).transpose(2, 3) for t in (q, k, v))
            expected = functional.scaled_dot_product_attention(q, k, v).transpose(2, 3).reshape(shape)
            self.assertTrue(torch.allclose(output, expected, atol=1e-5))
            self.assertEqual(calls, shape[0] * [(name, 1)])

    def test_mixed_int8_vae_routes_direct_attention_calls_to_compute(self):
        heads_seen = []

        def attention(q, k, v, heads, mask=None, skip_reshape=False, skip_output_reshape=False):
            heads_seen.append(heads)
            out = functional.scaled_dot_product_attention(q, k, v)
            return out if skip_output_reshape else out.transpose(1, 2).flatten(2)

        vae_module = types.ModuleType("mgpu_test_vae")
        vae_module.optimized_attention = attention

        class VAEModel(torch.nn.Module):
            __module__ = vae_module.__name__

            def decode(self, q, k, v):
                return vae_module.optimized_attention(q, k, v, 2, skip_reshape=True)

        model = VAEModel()
        q, k, v = (torch.randn(1, 2, 5, 4) for _ in range(3))
        with (
            mock.patch.dict(sys.modules, {vae_module.__name__: vae_module}),
            mock.patch.object(self.distorch, "_ATTENTION_RESERVE_BYTES", 0),
            mock.patch.object(self.distorch.mm, "get_free_memory", lambda *_: 1 << 30),
        ):
            self.distorch.configure_mixed_vae(model, "cpu", "cpu", int8=True)
            output = model.decode(q, k, v)
            outside = vae_module.optimized_attention(q, k, v, 2, skip_reshape=True)

        expected = attention(q, k, v, 2, skip_reshape=True)
        self.assertTrue(torch.allclose(output, expected, atol=1e-6))
        self.assertTrue(torch.equal(outside, expected))
        # The mixed call runs head chunks on the compute GPU; a call outside it runs unchanged.
        self.assertEqual(heads_seen, [1, 1, 2, 2])

    def test_attention_chunks_heads_over_the_whole_sequence(self):
        q, k, v = (torch.randn(1, 2, 9, 4) for _ in range(3))
        calls = []

        def attention(q, k, v, heads, mask=None, skip_reshape=False, **kwargs):
            calls.append(q.shape[2])
            out = functional.scaled_dot_product_attention(q, k, v)
            return out.transpose(1, 2).reshape(q.shape[0], -1, heads * q.shape[-1])

        with mock.patch.object(self.distorch, "_ATTENTION_RESERVE_BYTES", 0), mock.patch.object(
            self.distorch.mm, "get_free_memory", lambda *_: 700
        ):
            output = self.distorch._run_attention_on_compute(
                attention, q, k, v, 2, compute_device=torch.device("cpu"), plan={}, skip_reshape=True
            )

        self.assertEqual(calls, [9, 9])
        self.assertTrue(
            torch.allclose(output, attention(q, k, v, 2, skip_reshape=True), atol=1e-6)
        )

    def test_attention_chunks_heads_with_grouped_kv(self):
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
                    attention, q, k, v, 4, compute_device=torch.device("cpu"), plan={},
                    skip_output_reshape=skip_output_reshape,
                )
            expected = attention(q, k, v, 4, skip_output_reshape=skip_output_reshape)
            self.assertTrue(torch.allclose(output, expected, atol=1e-6))
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

    def test_attention_falls_back_to_donor_when_nothing_fits(self):
        q, k, v = (torch.randn(1, 2, 4, 3) for _ in range(3))
        plan = {}
        donor_calls = []

        def attention(q_in, k_in, v_in, heads, mask=None, **kwargs):
            # Chunks are views, so only the donor fallback sees the original tensors.
            if q_in is not q:
                raise torch.OutOfMemoryError("out of memory")
            donor_calls.append(q_in.shape)
            return functional.scaled_dot_product_attention(q_in, k_in, v_in)

        with mock.patch.object(self.distorch.mm, "get_free_memory", lambda *_: 1 << 30):
            for _ in range(2):
                output = self.distorch._run_attention_on_compute(
                    attention, q, k, v, 2, compute_device=torch.device("cpu"), plan=plan,
                    skip_reshape=True, skip_output_reshape=True,
                )

        self.assertTrue(
            torch.allclose(output, functional.scaled_dot_product_attention(q, k, v), atol=1e-6)
        )
        self.assertIsNone(plan[(q.shape, k.shape)])
        self.assertEqual(donor_calls[-2:], [q.shape, q.shape])

    def test_mixed_execution_moves_model_to_donor_and_routes_attention(self):
        calls = []

        class DiffusionModel(torch.nn.Module):
            def forward(self, x, timestep, transformer_options={}):
                calls.append(transformer_options)
                return [x[0] * 2]

        model = DiffusionModel()
        transformer_options = {"patches": {}}
        schedule = types.SimpleNamespace(
            start_forward=lambda: calls.append("start_forward"), mlp_linears={}
        )
        self.distorch.configure_mixed_execution(model, "cpu", "cpu", schedule)
        output = model.forward(
            [torch.ones(2)], torch.zeros(1), transformer_options=transformer_options
        )

        self.assertTrue(torch.equal(output[0], torch.full((2,), 2.0)))
        self.assertEqual(calls[0], "start_forward")
        self.assertIn("optimized_attention_override", calls[1])
        self.assertNotIn("optimized_attention_override", transformer_options)

    def test_donor_prepared_linear_matches_full_linear(self):
        module = torch.nn.Linear(3, 5, bias=True).requires_grad_(False)
        input_tensor = torch.randn(2, 3)

        with self.assertLogs("MultiGPU", level="INFO") as logs:
            output = self.distorch._run_donor_prepared_linear_on_compute(
                input_tensor, module, "cpu", "cpu"
            )

        self.assertTrue(torch.allclose(output, module(input_tensor)))
        self.assertEqual(output.shape, (2, 5))
        self.assertIn("Donor preparation confirmed on cpu", logs.output[0])
        self.assertIn("prepared weight is on cpu", logs.output[0])

    def test_materializes_cast_weights_on_donor(self):
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
            output = self.distorch._run_donor_prepared_linear_on_compute(
                input_tensor, module, "cpu", "cpu"
            )

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

    def test_donor_prepared_linear_moves_plain_lora_off_the_donor(self):
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
            output = self.distorch._run_donor_prepared_linear_on_compute(
                input_tensor, module, "cpu", "cpu"
            )

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

    def test_mixed_mode_leaves_donor_memory_to_activations(self):
        donor = torch.device("cuda:1")
        self.assertEqual(self.distorch._mixed_weight_device(AllocationModule(), donor), "cpu")
        self.assertEqual(self.distorch._mixed_weight_device(torch.nn.LayerNorm(2), donor), donor)

    def test_pinning_keeps_quantized_weight_metadata(self):
        class PackedTensor(torch.Tensor):
            pass

        module = torch.nn.Linear(1, 1, bias=False)
        weight = torch.randint(0, 255, (4, 144), dtype=torch.uint8).as_subclass(PackedTensor)
        module.weight = torch.nn.Parameter(weight, requires_grad=False)
        module.weight.tensor_type = "Q4_K"
        hip = types.ModuleType("comfy_kitchen.backends.hip")
        hip.offload_weight = lambda value: value.clone()
        backends = types.ModuleType("comfy_kitchen.backends")
        backends.hip = hip

        with (
            mock.patch.object(torch.version, "hip", "6.4"),
            mock.patch.dict(
                sys.modules,
                {"comfy_kitchen.backends": backends, "comfy_kitchen.backends.hip": hip},
            ),
        ):
            self.assertTrue(self.distorch.pin_hip_offloaded_weight(module))

        self.assertIsInstance(module.weight, PackedTensor)
        self.assertEqual(module.weight.tensor_type, "Q4_K")
        self.assertTrue(torch.equal(module.weight, weight))

    def test_mixed_mode_requires_standard_linear_tiling(self):
        self.assertFalse(self.distorch.configure_mixed_gemm(DonorModule(), "cuda:0", "cuda:1"))

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
        input_tensor = torch.randn(2, 4, 3)
        expected = module(input_tensor)
        chunks.clear()

        free_bytes = self.distorch._COMPUTE_GPU_RESERVE_BYTES + 72 + 3 * 2 * (3 + 5) * 4 * 2
        with (
            mock.patch.object(self.distorch, "_PIPELINE_CHUNKS", 4),
            mock.patch.object(self.distorch.mm, "get_free_memory", lambda *_: free_bytes, create=True),
            mock.patch.object(self.distorch.mm, "module_size", lambda _: 72, create=True),
        ):
            self.assertTrue(self.distorch.configure_mixed_gemm(module, "cpu", "cpu"))
            output = module.forward(input_tensor)

        # Three chunks of 2 tokens fit what the weight (72 bytes) leaves, and 8 tokens
        # already make _PIPELINE_CHUNKS chunks at that size.
        self.assertEqual(chunks, [2, 2, 2, 2])
        self.assertTrue(torch.allclose(output, expected))

    def test_mixed_int8_linear_runs_stored_weight_and_keeps_lora_low_rank(self):
        module = torch.nn.Linear(3, 5).requires_grad_(False)
        up, down = torch.randn(5, 2), torch.randn(2, 3)
        self.assertTrue(self.distorch.configure_mixed_gemm(module, "cpu", "cpu", int8=True))
        module._mgpu_int8_weight = module.weight + 1
        module._mgpu_int8_bias = module.bias
        module._mgpu_int8_lora = [(down, up, 0.5)]
        input_tensor = torch.randn(2, 3, 3)
        free_bytes = self.distorch._COMPUTE_GPU_RESERVE_BYTES + 2 * (3 + 5 + 2) * 4 * 2

        with mock.patch.object(
            self.distorch.mm, "get_free_memory", lambda *_: free_bytes, create=True
        ):
            output = module(input_tensor)

        expected = functional.linear(
            input_tensor, module.weight + 1 + 0.5 * (up @ down), module.bias
        )
        self.assertTrue(torch.allclose(output, expected, atol=1e-5))

    @unittest.skipUnless(torch.cuda.is_available(), "weight prefetch uses CUDA streams")
    def test_mixed_linears_learn_their_order_and_stage_the_next_weight(self):
        device = torch.device("cuda:0")
        schedule = self.distorch._MixedSchedule(device, device)
        first, second = (torch.nn.Linear(256, 256).requires_grad_(False) for _ in range(2))
        for module in (first, second):
            self.assertTrue(
                self.distorch.configure_mixed_gemm(module, device, device, schedule=schedule)
            )
        x = torch.randn(64, 256, device=device)
        expected = x
        for module in (first, second):
            expected = functional.linear(expected, module.weight.to(device), module.bias.to(device))

        with mock.patch.object(self.distorch.mm, "get_free_memory", lambda *_: 1 << 30, create=True):
            for _ in range(2):
                schedule.start_forward()
                output = second(first(x))
                self.assertTrue(torch.allclose(output, expected, atol=1e-4))
                self.assertIs(schedule.following[first], second)
            schedule.start_forward()
            first(x)
            self.assertIsNotNone(second._mgpu_staged)
            schedule.start_forward()

        self.assertIsNone(second._mgpu_staged)
        self.assertEqual(list(first.children()), [])

    @unittest.skipUnless(torch.cuda.is_available(), "weight prefetch uses CUDA streams")
    def test_mixed_units_measure_the_budget_once_per_forward(self):
        device = torch.device("cuda:0")
        schedule = self.distorch._MixedSchedule(device, device)
        linears = [torch.nn.Linear(256, 256).requires_grad_(False) for _ in range(3)]
        for module in linears:
            self.distorch.configure_mixed_gemm(module, device, device, schedule=schedule)
        x = torch.randn(64, 256, device=device)
        measured = []

        def budget(*_):
            measured.append(None)
            return 1 << 30

        with mock.patch.object(self.distorch.mm, "get_free_memory", lambda *_: 1 << 30, create=True), \
                mock.patch.object(self.distorch, "_compute_budget", budget):
            for _ in range(2):
                schedule.start_forward()
                for module in linears:
                    x = module(x)

        self.assertEqual(len(measured), 2)

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

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(256, 256)
                self.feed_forward = FeedForward()
                self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(256, 512), torch.nn.GELU(), torch.nn.Linear(512, 256)
                )

            def forward(self, x, timestep, transformer_options={}):
                return self.mlp(self.feed_forward(self.proj(x)))

        model = Model().requires_grad_(False)
        x = torch.randn(2, 64, 256)
        expected = model.forward(x, None)
        schedule = self.distorch._MixedSchedule(device, device)
        for module in model.modules():
            self.distorch.configure_mixed_gemm(module, device, "cpu", schedule=schedule)
        self.distorch.configure_mixed_execution(model, "cpu", device, schedule)
        self.assertEqual(
            schedule.mlp_linears,
            {
                model.feed_forward: [model.feed_forward.w1, model.feed_forward.w3, model.feed_forward.w2],
                model.mlp: [model.mlp[0], model.mlp[2]],
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

    def test_mixed_int8_needs_convrot_group_aligned_features(self):
        self.assertTrue(self.distorch._is_mixed_int8_linear(torch.nn.Linear(512, 4)))
        self.assertFalse(self.distorch._is_mixed_int8_linear(torch.nn.Linear(96, 4)))
        quantized = torch.nn.Linear(512, 4)
        quantized.layout_type = "TensorWiseINT8Layout"
        self.assertFalse(self.distorch._is_mixed_int8_linear(quantized))

    def int8_modules(self, merged):
        quant_ops = types.ModuleType("comfy.quant_ops")
        lora = types.ModuleType("comfy.lora")
        quantized = []

        class QuantizedTensor:
            @staticmethod
            def from_float(tensor, layout, **kwargs):
                quantized.append((layout, kwargs))
                return tensor.clone()

        def calculate_weight(patches, weight, key):
            merged.append(key)
            return weight + 1

        quant_ops.QuantizedTensor = QuantizedTensor
        lora.calculate_weight = calculate_weight
        return {
            "comfy.quant_ops": quant_ops,
            "comfy.lora": lora,
            "comfy.weight_adapter.lora": self.lora_adapter_module(),
        }, quantized

    def test_prepares_int8_weight_once_per_patch_set(self):
        merged = []
        modules, quantized = self.int8_modules(merged)
        module = torch.nn.Linear(3, 5).requires_grad_(False)
        up, down = torch.randn(5, 2), torch.randn(2, 3)
        adapter = modules["comfy.weight_adapter.lora"].LoRAAdapter((up, down, None, None, None, None))
        patches = {"linear.weight": [(0.5, adapter, 1.0, None, None)]}

        with mock.patch.dict(sys.modules, modules):
            prepare = self.distorch._prepare_mixed_int8_weight
            prepare(module, "linear", patches, torch.float32, "cpu", "uuid-1")
            prepare(module, "linear", patches, torch.float32, "cpu", "uuid-1")
            self.assertEqual(len(quantized), 1)
            prepare(module, "linear", patches, torch.float32, "cpu", "uuid-2")

        self.assertEqual(
            quantized,
            2 * [("TensorWiseINT8Layout", {"is_weight": True, "per_channel": True, "convrot": True})],
        )
        # The plain LoRA stays low rank; the stored weight is the unpatched original.
        self.assertEqual(merged, [])
        self.assertTrue(torch.equal(module._mgpu_int8_weight, module.weight))
        self.assertTrue(torch.equal(module._mgpu_int8_bias, module.bias))
        (lora_down, lora_up, scale), = module._mgpu_int8_lora
        self.assertIs(lora_down, down)
        self.assertEqual(scale, 0.5)
        self.assertNotIn("_mgpu_int8_weight", module.state_dict())

    def test_merges_non_plain_patches_into_int8_weight(self):
        merged = []
        modules, _ = self.int8_modules(merged)
        module = torch.nn.Linear(3, 5).requires_grad_(False)
        patches = {
            "linear.weight": [(1.0, object(), 1.0, None, None)],
            "linear.bias": [(1.0, object(), 1.0, None, None)],
        }

        with mock.patch.dict(sys.modules, modules):
            self.distorch._prepare_mixed_int8_weight(
                module, "linear", patches, torch.float32, "cpu", "uuid"
            )

        self.assertEqual(merged, ["linear.weight", "linear.bias"])
        self.assertTrue(torch.equal(module._mgpu_int8_weight, module.weight + 1))
        self.assertTrue(torch.equal(module._mgpu_int8_bias, module.bias + 1))
        self.assertEqual(module._mgpu_int8_lora, ())

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
            self.assertFalse(self.distorch.configure_mixed_gemm(module, "cuda:0", "cuda:1"))

        self.assertIn("Mixed compute GEMM skipped", logs.output[0])

    def test_invalid_expert_allocation_uses_virtual_vram_donor(self):
        module = AllocationModule()
        patcher = AllocationPatcher(module)
        gibibyte = 1024**3

        def total_memory(device):
            return {"cuda:0": 6 * gibibyte, "cuda:1": 32 * gibibyte}.get(
                str(device), 64 * gibibyte
            )

        with (
            mock.patch.object(
                self.distorch,
                "get_device_list",
                return_value=["cuda:0", "cuda:1", "cpu"],
            ),
            mock.patch.object(
                self.distorch.mm,
                "get_total_memory",
                side_effect=total_memory,
                create=True,
            ),
        ):
            assignments = self.distorch.analyze_safetensor_loading(
                patcher, "True#cuda:0;24.0;cuda:1"
            )

        self.assertEqual(assignments["block_assignments"]["linear"], "cuda:1")

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

    def test_virtual_vram_sizes_model_by_stored_bytes(self):
        # A quantized weight reports its logical dtype: 4 MiB as float32, 1 MiB stored as int8.
        module = AllocationModule()
        module.weight = torch.empty(1024, 1024)
        patcher = AllocationPatcher(module)
        patcher._load_list = lambda: ((1024 * 1024, "linear", module, {}),)
        mebibyte = 1024**2

        with (
            mock.patch.object(
                self.distorch, "get_device_list", return_value=["cuda:0", "cuda:1", "cpu"]
            ),
            mock.patch.object(
                self.distorch.mm,
                "get_total_memory",
                side_effect=lambda device: {"cuda:0": 2 * mebibyte, "cuda:1": 32 * mebibyte}.get(
                    str(device), 64 * mebibyte
                ),
                create=True,
            ),
        ):
            assignments = self.distorch.analyze_safetensor_loading(
                patcher, "#cuda:0;0.001;cuda:1"
            )

        self.assertEqual(assignments["block_assignments"]["linear"], "cuda:1")

    def test_all_mode_executes_large_gemms_on_donor(self):
        module = DonorModule()
        module.weight = torch.empty(
            16 * 1024 * 1024 // 4 + 1,
            dtype=torch.float32,
        ).reshape(-1, 1)

        with (
            mock.patch.object(
                self.distorch, "_is_hip_software_gemm_device", return_value=True
            ),
            mock.patch.object(
                self.distorch.mm,
                "comfy_kitchen_attention_enabled",
                return_value=True,
                create=True,
            ),
        ):
            self.assertTrue(
                self.distorch.configure_hip_donor_gemm_offload(
                    module, "cuda:0", "cuda:1", execution_mode="all"
                )
            )

    def test_all_mode_requires_comfy_kitchen_attention(self):
        module = DonorModule()

        with mock.patch.object(
            self.distorch, "_is_hip_software_gemm_device", return_value=True
        ):
            self.assertFalse(
                self.distorch.configure_hip_donor_gemm_offload(
                    module, "cuda:0", "cuda:1", execution_mode="all"
                )
            )

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

    def test_rocm_donor_linear_execution(self):
        if not getattr(torch.version, "hip", None) or torch.cuda.device_count() < 2:
            self.skipTest("requires two ROCm GPUs")
        if not (
            self.distorch._is_hip_software_gemm_device("cuda:0")
            and self.distorch._is_hip_software_gemm_device("cuda:1")
        ):
            self.skipTest("requires two HIP software-GEMM GPUs")
        if not self.distorch._is_comfy_kitchen_attention_enabled():
            self.skipTest("requires Comfy Kitchen attention")

        compute_device = torch.device("cuda:0")
        donor_device = torch.device("cuda:1")
        module = DonorLinearModule(donor_device)
        input_tensor = torch.tensor([[2.0, 3.0]], device=compute_device)

        self.assertTrue(
            self.distorch.configure_hip_donor_gemm_offload(
                module, compute_device, donor_device, execution_mode="all"
            )
        )
        result = module.forward(input_tensor)
        torch.cuda.synchronize(compute_device)
        torch.cuda.synchronize(donor_device)

        self.assertEqual(module.input_device, donor_device)
        self.assertEqual(result.device, compute_device)
        self.assertTrue(torch.equal(result.cpu(), input_tensor.cpu()))
        self.assertEqual(module._mgpu_donor_gemm_calls, 1)
