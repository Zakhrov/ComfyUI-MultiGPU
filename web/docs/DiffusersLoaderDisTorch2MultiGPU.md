# DiffusersLoaderDisTorch2MultiGPU

The `DiffusersLoaderDisTorch2MultiGPU` node is used to load Diffusers models (HuggingFace Hub repositories) with DisTorch2 distributed tensor allocation, enabling advanced multi-device VRAM management to handle larger diffusion models across multiple GPUs.

This node loads models directly from HuggingFace model repositories by specifying the repository ID (e.g., "stabilityai/stable-diffusion-xl-base-1.0").

## Inputs

| Parameter | Data Type | Description |
| --- | --- | --- |
| `model_path` | `STRING` | The HuggingFace repository ID or local path of the diffusers model to load (e.g., 'stabilityai/stable-diffusion-xl-base-1.0'). |
| `compute_device` | `STRING` | Target device for compute operations (e.g., 'cuda:0', 'cuda:1', 'cpu'). Selected from available devices on your system. |
| `eject_models` | `BOOLEAN` | Whether to unload ALL models from the target device before loading this model, enabling deterministic model eviction for testing and memory management (default: true). |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `MODEL` | `MODEL` | The loaded diffusers model with DisTorch2 distributed allocation applied. |

## DisTorch2 Distributed Loading

DisTorch2 places the model's weights automatically when it loads. The compute device keeps the layers it must compute with directly; the rest are spread over every other device, system RAM included, in proportion to their free memory, and weights in system RAM are pinned so they never page in from disk. Large diffusion models on supported AMD GPUs also run streamed: activations stay on the compute device and weights are streamed in. See the repository README for details.
