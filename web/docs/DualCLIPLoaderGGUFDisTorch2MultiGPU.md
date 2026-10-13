# DualCLIPLoaderGGUFDisTorch2MultiGPU

The `DualCLIPLoaderGGUFDisTorch2MultiGPU` node is used to load dual GGUF format CLIP text encoder models with DisTorch2 distributed tensor allocation, enabling advanced multi-device VRAM management to handle larger text encoding models across multiple GPUs.

This node automatically detects models located in the `ComfyUI/models/clip` and `ComfyUI/models/clip_gguf` folders, and it will also read models from additional paths configured in the `extra_model_paths.yaml` file. Sometimes, you may need to **refresh the ComfyUI interface** to allow it to read the model files from the corresponding folder.

## Inputs

| Parameter | Data Type | Description |
| --- | --- | --- |
| `clip_name1` | `STRING` | The name of the first CLIP model to load from combined clip and clip_gguf folders. |
| `clip_name2` | `STRING` | The name of the second CLIP model to load from combined clip and clip_gguf folders. |
| `type` | `STRING` | The type of CLIP model configuration for dual loading. |
| `device` | `STRING` | Target device for text encoder compute operations (e.g., 'cuda:0', 'cuda:1', 'cpu'). Selected from available devices on your system. |
| `eject_models` | `BOOLEAN` | Whether to unload ALL models from the target device before loading this model, enabling deterministic model eviction for testing and memory management (default: false for CLIP loaders). |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `CLIP` | `CLIP` | The loaded dual CLIP text encoder models with DisTorch2 distributed allocation applied. |

## DisTorch2 Distributed Loading

DisTorch2 places the model's weights automatically when it loads. The compute device keeps the layers it must compute with directly; the rest are spread over every other device, system RAM included, in proportion to their free memory, and weights in system RAM are pinned so they never page in from disk. Large diffusion models on supported AMD GPUs also run streamed: activations stay on the compute device and weights are streamed in. See the repository README for details.
