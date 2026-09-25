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
            mock.patch.object(self.distorch, "_move_tensors", side_effect=record_move),
        ):
            self.assertTrue(
                self.distorch.configure_hip_donor_gemm_offload(
                    module, "cuda:0", "cuda:1"
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
        ):
            self.distorch.configure_hip_donor_gemm_offload(module, "cuda:0", "cuda:1")
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
        ):
            self.assertFalse(
                self.distorch.configure_hip_donor_gemm_offload(
                    module, "cuda:0", "cuda:1"
                )
            )

    def test_mixed_mode_accepts_large_gemms_for_tiled_execution(self):
        module = DonorModule()
        module.weight = torch.empty(
            16 * 1024 * 1024 // 4 + 1,
            dtype=torch.float32,
        ).reshape(-1, 1)

        with mock.patch.object(
            self.distorch, "_is_hip_software_gemm_device", return_value=True
        ):
            self.assertTrue(
                self.distorch.configure_hip_donor_gemm_offload(
                    module, "cuda:0", "cuda:1"
                )
            )

    def test_tiled_linear_limits_compute_weight_tile_size(self):
        module = torch.nn.Linear(3, 5, bias=True)
        input_tensor = torch.randn(2, 3)

        with mock.patch.object(self.distorch, "_HIP_DONOR_GEMM_MAX_WEIGHT_BYTES", 24):
            output = self.distorch._run_tiled_linear_on_compute(
                input_tensor, module, "cpu"
            )

        self.assertTrue(torch.allclose(output, module(input_tensor)))
        self.assertEqual(output.shape, (2, 5))

    def test_rejects_packed_linear_weight_for_compute_tiling(self):
        module = torch.nn.Linear(1, 1, bias=False)
        module.in_features = 5376
        module.out_features = 16128
        module.weight = torch.nn.Parameter(torch.empty(3024, 3120, device="meta"))
        module.weight.tensor_type = "Q4_K"

        self.assertFalse(self.distorch._can_tile_linear_on_compute(module))

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

        with mock.patch.object(
            self.distorch, "_is_hip_software_gemm_device", return_value=True
        ):
            self.assertTrue(
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

        compute_device = torch.device("cuda:0")
        donor_device = torch.device("cuda:1")
        module = DonorLinearModule(donor_device)
        input_tensor = torch.tensor([[2.0, 3.0]], device=compute_device)

        self.assertTrue(
            self.distorch.configure_hip_donor_gemm_offload(
                module, compute_device, donor_device
            )
        )
        result = module.forward(input_tensor)
        torch.cuda.synchronize(compute_device)
        torch.cuda.synchronize(donor_device)

        self.assertEqual(module.input_device, donor_device)
        self.assertEqual(result.device, compute_device)
        self.assertTrue(torch.equal(result.cpu(), input_tensor.cpu()))
        self.assertEqual(module._mgpu_donor_gemm_calls, 1)
