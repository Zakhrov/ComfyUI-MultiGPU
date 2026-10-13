# UnetLoaderGGUFDisTorch2MultiGPU

The `UnetLoaderGGUFDisTorch2MultiGPU` node is used to load GGUF format UNet models with DisTorch2 distributed tensor allocation, enabling advanced multi-device VRAM management to handle larger models across multiple GPUs.

This node automatically detects models located in the `ComfyUI/models/unet_gguf` folder, and it will also read models from additional paths configured in the `extra_model_paths.yaml` file. Sometimes, you may need to **refresh the ComfyUI interface** to allow it to read the model files from the corresponding folder.

## Inputs

| Parameter | Data Type | Description |
| --- | --- | --- |
| `unet_name` | `STRING` | The name of the GGUF format UNet model to load. |
| `compute_device` | `STRING` | Target device for compute operations (e.g., 'cuda:0', 'cuda:1', 'cpu'). Selected from available devices on your system. |
| `eject_models` | `BOOLEAN` | Whether to unload ALL models from the target device before loading this model, enabling deterministic model eviction for testing and memory management (default: true). |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `MODEL` | `MODEL` | The loaded GGUF format UNet model with DisTorch2 distributed allocation applied. |

## DisTorch2 Distributed Loading

DisTorch2 places the model's weights automatically when it loads. The compute device keeps the layers it must compute with directly; the rest are spread over every other device, system RAM included, in proportion to their free memory, and weights in system RAM are pinned so they never page in from disk. Large diffusion models on supported AMD GPUs also run streamed: activations stay on the compute device and weights are streamed in. See the repository README for details.
