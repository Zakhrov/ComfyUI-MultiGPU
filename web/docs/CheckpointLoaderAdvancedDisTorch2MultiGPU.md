# CheckpointLoaderAdvancedDisTorch2MultiGPU

The `CheckpointLoaderAdvancedDisTorch2MultiGPU` node is used to load checkpoint models with advanced DisTorch2 distributed tensor allocation, providing granular control over UNet, CLIP, and VAE component allocation across multiple devices with independent virtual VRAM management.

This node automatically detects models located in the `ComfyUI/models/checkpoints` folder, and it will also read models from additional paths configured in the `extra_model_paths.yaml` file. Sometimes, you may need to **refresh the ComfyUI interface** to allow it to read the model files from the corresponding folder.

## Inputs

| Parameter | Data Type | Description |
| --- | --- | --- |
| `ckpt_name` | `STRING` | The name of the checkpoint model to load. |
| `unet_compute_device` | `STRING` | Target compute device for UNet distributed allocation (e.g., 'cuda:0', 'cuda:1', 'cpu'). |
| `clip_compute_device` | `STRING` | Target compute device for CLIP distributed allocation (default: 'cpu'). |
| `vae_device` | `STRING` | Target device for the VAE component (e.g., 'cuda:0', 'cuda:1', 'cpu'). |
| `high_precision_loras` | `BOOLEAN` | Whether to use high-precision LoRA patches (default: true). |

## Outputs

| Output Name | Data Type | Description |
| --- | --- | --- |
| `MODEL` | `MODEL` | The loaded UNet diffusion model with DisTorch2 distributed allocation. |
| `CLIP` | `CLIP` | The loaded CLIP text encoder model with DisTorch2 distributed allocation. |
| `VAE` | `VAE` | The loaded VAE decoder/encoder model. |

## DisTorch2 Distributed Loading

DisTorch2 places the model's weights automatically when it loads. The compute device keeps the layers it must compute with directly; the rest are spread over every other device, system RAM included, in proportion to their free memory, and weights in system RAM are pinned so they never page in from disk. Large diffusion models on supported AMD GPUs also run streamed: activations stay on the compute device and weights are streamed in. See the repository README for details.
