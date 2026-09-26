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
    comfy.model_management = model_management
    comfy.model_patcher = model_patcher
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = model_management
    sys.modules["comfy.model_patcher"] = model_patcher

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

    def test_attention_runs_in_query_chunks(self):
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
                attention, q, k, v, 2, compute_device=torch.device("cpu"), skip_reshape=True
            )

        self.assertGreater(len(calls), 1)
        self.assertTrue(
            torch.allclose(output, attention(q, k, v, 2, skip_reshape=True), atol=1e-6)
        )

    def test_mixed_execution_moves_model_to_donor_and_routes_attention(self):
        calls = []

        class DiffusionModel:
            def forward(self, x, timestep, transformer_options={}):
                calls.append(transformer_options)
                return [x[0] * 2]

        model = DiffusionModel()
        transformer_options = {"patches": {}}
        self.distorch.configure_mixed_execution(model, "cpu", "cpu")
        output = model.forward(
            [torch.ones(2)], torch.zeros(1), transformer_options=transformer_options
        )

        self.assertTrue(torch.equal(output[0], torch.full((2,), 2.0)))
        self.assertIn("optimized_attention_override", calls[0])
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

    def test_mixed_mode_requires_standard_linear_tiling(self):
        self.assertFalse(self.distorch.configure_mixed_gemm(DonorModule(), "cuda:0", "cuda:1"))

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
