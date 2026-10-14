# ComfyUI-MultiGPU (AMD ROCm fork)

A fork of [pollockjj/ComfyUI-MultiGPU](https://github.com/pollockjj/ComfyUI-MultiGPU) for multi-GPU AMD ROCm systems: DisTorch2 places model weights automatically across every device, and diffusion models run with their activations on the compute GPU and their weights streamed in.

> [!WARNING]
> **AMD ROCm only.** The changes in this fork are developed and tested exclusively on AMD GPUs with ROCm (HIP builds of PyTorch). They are **not** tested on, and should be assumed **not to work** on, NVIDIA CUDA, Intel XPU, Apple MPS, DirectML or any other torch backend. If you are not on ROCm, use [upstream ComfyUI-MultiGPU](https://github.com/pollockjj/ComfyUI-MultiGPU) instead.

> [!CAUTION]
> **Requires a custom, unofficial Comfy Kitchen HIP backend.** Streamed execution depends on the `rdna1` backend in the `rdna1-backend` branch of a Comfy Kitchen fork: [Zakhrov/comfy-kitchen@rdna1-backend](https://github.com/Zakhrov/comfy-kitchen/tree/rdna1-backend), which the [RDNA1 ComfyUI fork](https://github.com/Zakhrov/ComfyUI-RDNA1-ROCM) carries as a submodule. This backend is experimental, is not part of official Comfy Kitchen releases, and is not supported by the Comfy Kitchen or ComfyUI maintainers. Expect breakage when ComfyUI or Comfy Kitchen update. Do not report problems caused by it to those projects.

## What's different from upstream

| Area | Change |
| --- | --- |
| Placement | Automatic. DisTorch2 loaders have only a compute device and `eject_models`; `virtual_vram_gb`, `donor_device` and expert allocation strings are gone. See [Automatic placement](#automatic-placement). |
| Streamed execution | Large diffusion models keep their activations on the compute GPU, spill them to pinned system RAM only when it is full, and stream their weights in. See [Streamed execution](#streamed-execution). |
| GGUF text encoder | New `CCTechClipProjLoaderDisTorch2MultiGPU` for [ComfyUI-GGUF-Loader](https://github.com/ChrisColeTech/ComfyUI-GGUF-Loader)'s Text Encoder + ClipProj Loader. |
| WanVideoWrapper | The WanVideo model loader falls back to the offload-aware default RMSNorm when PyTorch RMSNorm is selected, since the native one is incompatible with WanVideo VRAM management. |
| Tests | `tests/test_distorch_hip_donor.py` covers placement and streamed execution. |

Workflows saved with the old DisTorch2 loader inputs need their loader widgets checked once: ComfyUI stores widget values by position.

## Automatic placement

When a DisTorch2 model loads, every layer is placed by free memory:

- The compute GPU keeps the layers ComfyUI cannot cast at runtime, tiny layers and CLIP embeddings.
- A streamed diffusion model leaves the rest of the compute GPU to activations. Any other model (CLIP, VAE, small diffusion models) fills it from the head, leaving ComfyUI's inference reserve and room to cast its largest weight in float32.
- Everything else is spread over all other devices, the CPU included, in proportion to their free memory.
- Weights on the CPU are copied into pinned RAM, so they are never paged in from a memory-mapped file during a run and the compute GPU's copy engine reads them asynchronously.

The log shows the result, for example `MiniMaxH3 weights (streamed): cuda:0 213 layers 0 MB, cuda:1 140 layers 5642 MB, cpu 121 layers 5503 MB`.

On ComfyUI builds with DynamicVRAM/comfy-aimdo enabled, MultiGPU keeps DynamicVRAM on devices comfy-aimdo initialized and falls back to legacy model patching for other MultiGPU devices.

## Streamed execution

A diffusion model streams when it would take more than half of the compute GPU's free memory, the compute GPU uses the HIP software-GEMM path and Comfy Kitchen attention is enabled. Otherwise it loads like upstream DisTorch2 and `Streamed execution disabled: <reasons>` is logged.

### Relationship to comfy-kitchen's rdna1 backend

On an RDNA1 compute GPU, streamed execution runs on comfy-kitchen's `rdna1` backend, not on the plain eager backend:

- it is enabled only on the HIP software-GEMM path with `--use-ck-attention`, which needs the backend's INT8 attention; the eager backend has none on gfx1010;
- streamed linears run the backend's packed fp16 GEMM (GGUF Q8_0 and Q4_K weights its `q8_0_linear` and `q4_k_linear`), convs its implicit-GEMM conv, and SLA its `int8_block_sparse_attention`.

The RDNA1 ComfyUI fork routes the dense layers to the same kernels when a model fits without streaming. A donor GPU only stores weights; every op runs on the compute GPU.

### Supported GPUs

The compute GPU must use the HIP software-GEMM path:

- Vega / GCN5: `gfx900`, `gfx906`, `gfx90c`
- RDNA1: `gfx1010`–`gfx1012`
- RDNA2: `gfx1030`–`gfx1036`

Comfy Kitchen's `rdna1` backend only has kernels for `gfx1010`; on the others streamed layers run on torch ops.

### How it works

- Activations live on the compute GPU. An op whose result does not fit in its free VRAM runs in slices whose results are copied to pinned system RAM as they complete. An op that reads a spilled tensor streams it back in slices and keeps its result on the compute GPU when it fits. Spills never go to another GPU: an APU would copy them with its shaders, where the compute GPU's copy engine does it for free.
- When a step's forward completes, the staged weights are dropped and the allocator's cached memory is released, so the next step, and anything between steps such as a preview decode, starts with the compute GPU free. That is the only synchronization per step.
- Within a step, nothing synchronizes the host with the GPUs. Copy and compute streams wait for each other on the GPU, and free VRAM is measured without a sync. After an out-of-memory error, the allocator's cache is released and the work retried in smaller pieces.
- Each linear stages the next linear's weight on a copy stream while its own GEMM runs. Weights are copied to the compute GPU as stored (quantized) and dequantized and patched there. Float GEMMs use ComfyUI's `dense_linear`, which runs Comfy Kitchen's packed-fp16 GEMM where it is faster (`gfx1010`, `gfx90c`: about 2.5× rocBLAS).
- Token-wise MLPs (SwiGLU `w1`/`w2`/`w3`, `fc1`/`fc2` pairs and `Sequential` linear/activation MLPs) run whole on the compute GPU in token chunks, so their wide hidden activation is never materialized in full.
- Attention runs in one call when Q, K and V are on the compute GPU. On out-of-memory, or for spilled tensors, it runs in chunks of whole heads, and the chunk size is kept until the model reloads.
- ComfyUI quantized safetensors layers (int8, fp8) send their packed weight to the compute GPU and run their own quantized matmul there. Plain LoRAs on GGUF models run as two thin GEMMs instead of being merged into the dequantized weight on every call.
- GGUF Q8_0 and Q4_K linears are staged as their raw blocks (Q8_0 half, Q4_K under a third of the float16 weight's bytes) and run on Comfy Kitchen's `q8_0_linear` / `q4_k_linear`, which read the blocks in the packed GEMM without dequantizing the weight: 1.15-1.2x faster than the float16 GEMM on `gfx1010`. Plain LoRAs on these layers run beside them as two thin GEMMs; any other patch makes the layer dequantize so the loader can merge it.
- Conv2d and Conv3d run in tiles of at most 128 MiB on the compute GPU, each tile through ComfyUI's `comfy.ops.conv_nd`, so they take the same Comfy Kitchen conv kernels as an unstreamed model on the RDNA1 ComfyUI fork.
- VAEs and text encoders are never streamed; their weights are placed as above and cast layer by layer.

### Benchmark

MiniMax H3 ref2va Q4_K GGUF with the 4-step turbo LoRA, 832×480, 124 frames (15.3k tokens), SLA attention at 90% sparsity on steps 3–4, on a Dell G5 15 SE: RX 5600M 6 GB (`gfx1010`, compute) + Ryzen Vega APU (`gfx90c`). PyTorch 2.12 ROCm nightly, `--cuda-malloc --disable-dynamic-vram --use-ck-attention`.

| Execution | Dense step | Sparse step | Sampling | Workflow |
| --- | --- | --- | --- | --- |
| Previous design: activations on a CPU donor, every op round-tripped | 423–435 s | 346–352 s | 25:56 | 30:26 |
| Streamed execution | 193–253 s | 181–207 s | 13:02–14:25 | 17:50–19:43 |

Both produced matching videos.

### Text-to-image benchmark

512x512, one image, on the same laptop, 2026-10-09: the RDNA1 ComfyUI fork with comfy-kitchen's `rdna1` backend, `--use-ck-attention --disable-cuda-malloc --disable-dynamic-vram`, normal VRAM. "Without MultiGPU" uses the core loaders on the 5600M alone; "MultiGPU" uses the DisTorch2 loaders for every model and VAE with both GPUs visible (compute on the 5600M, donors: the Vega iGPU and system RAM). Cold includes model loading; warm is a second prompt with a new seed.

| Model | Without MultiGPU | MultiGPU | Weight placement under MultiGPU |
| --- | --- | --- | --- |
| SD1.5, 20 steps | 14.9 / 12.7 s, 1.86 it/s, 2.6-2.7 GB | 16.8 / 13.0 s, 1.85 it/s, 2.4 GB | all on the 5600M (not streamed) |
| SDXL, 20 steps | 35.3 / 30.8 s, 1.38 s/it, 5.3-5.5 GB | 45.3 / 35.4 s, 1.65 s/it, 1.9 GB | streamed: 0 MB on the 5600M, 2.1 GB on the iGPU, 2.7 GB in RAM |
| Z-Image Turbo, 9 steps | out of memory in text encode | 74.6 / 29.6 s, 2.87 s/it, 5.2 GB cold / 2.6 GB warm | UNet streamed: 2.6 GB on the iGPU, 3.3 GB in RAM; text encoder 3.2 GB on the 5600M, 0.5 GB on the iGPU, 0.7 GB in RAM |

- Times are cold / warm; VRAM is the 5600M's device-wide peak.
- Without MultiGPU, Z-Image Turbo only runs with `--lowvram`: 111.8 / 33.2 s, 3.37 s/it (Q8_0 GGUF UNet). The MultiGPU run loads `z_image_turbo_int8_convrot.safetensors` instead; both use the Qwen3-4B Q8_0 GGUF text encoder.
- With the 5600M alone (CPU donor), the MultiGPU Z-Image run took 91.9 / 28.7 s at 3.07 s/it. With sub-quadratic attention instead of ck-attention streaming is disabled, 4.4 GB of the UNet stays on the GPU, and it took 64.8 / 29.5 s at 2.90 s/it.
- With the Q8_0 GGUF UNet on `UnetLoaderGGUFDisTorch2MultiGPU` instead, the MultiGPU Z-Image run took 56.4 / 30.3 s at 3.00 s/it, 3.0 GB warm (UNet streamed: about 3 GB on the iGPU, 3.4-3.9 GB in RAM): within 5% of the safetensors run once loaded.
- The iGPU only holds weights. Every op runs on the compute GPU.
- SD1.5 and SDXL checkpoints carry their VAE; for the MultiGPU VAE loader each was extracted unchanged to its own file.

## Installation

ComfyUI-Manager installs upstream, not this fork. Install manually:

1. Clone this repository into `ComfyUI/custom_nodes/`:
   ```bash
   git clone https://github.com/Zakhrov/ComfyUI-MultiGPU
   ```
2. For streamed execution, use the [RDNA1 ComfyUI fork](https://github.com/Zakhrov/ComfyUI-RDNA1-ROCM), whose install builds the Comfy Kitchen `rdna1` backend from its submodule, or install [Zakhrov/comfy-kitchen@rdna1-backend](https://github.com/Zakhrov/comfy-kitchen/tree/rdna1-backend) with `COMFY_HIP_ARCHS=gfx1010` into the same Python environment as ComfyUI, replacing any official `comfy-kitchen` package.
3. Run ComfyUI with a ROCm build of PyTorch.

## Nodes

The extension automatically creates MultiGPU versions of loader nodes. Each MultiGPU node has the same functionality as its original counterpart but adds a `device` parameter that allows you to specify the GPU to use.

Currently supported nodes (automatically detected if available):

- Standard [ComfyUI](https://github.com/comfyanonymous/ComfyUI) model loaders:
  - [CheckpointLoaderAdvancedMultiGPU](web/docs/CheckpointLoaderAdvancedMultiGPU.md) / [CheckpointLoaderAdvancedDisTorch2MultiGPU](web/docs/CheckpointLoaderAdvancedDisTorch2MultiGPU.md)
  - [CheckpointLoaderSimpleMultiGPU](web/docs/CheckpointLoaderSimpleMultiGPU.md) / [CheckpointLoaderSimpleDisTorch2MultiGPU](web/docs/CheckpointLoaderSimpleDisTorch2MultiGPU.md)
  - [UNETLoaderMultiGPU](web/docs/UNETLoaderMultiGPU.md) / [UNETLoaderDisTorch2MultiGPU](web/docs/UNETLoaderDisTorch2MultiGPU.md)
  - [UNetLoaderLP](web/docs/UNetLoaderLP.md)
  - [VAELoaderMultiGPU](web/docs/VAELoaderMultiGPU.md) / [VAELoaderDisTorch2MultiGPU](web/docs/VAELoaderDisTorch2MultiGPU.md)
  - [CLIPLoaderMultiGPU](web/docs/CLIPLoaderMultiGPU.md) / [CLIPLoaderDisTorch2MultiGPU](web/docs/CLIPLoaderDisTorch2MultiGPU.md)
  - [DualCLIPLoaderMultiGPU](web/docs/DualCLIPLoaderMultiGPU.md) / [DualCLIPLoaderDisTorch2MultiGPU](web/docs/DualCLIPLoaderDisTorch2MultiGPU.md)
  - [TripleCLIPLoaderMultiGPU](web/docs/TripleCLIPLoaderMultiGPU.md) / [TripleCLIPLoaderDisTorch2MultiGPU](web/docs/TripleCLIPLoaderDisTorch2MultiGPU.md)
  - [QuadrupleCLIPLoaderMultiGPU](web/docs/QuadrupleCLIPLoaderMultiGPU.md) / [QuadrupleCLIPLoaderDisTorch2MultiGPU](web/docs/QuadrupleCLIPLoaderDisTorch2MultiGPU.md)
  - [CLIPVisionLoaderMultiGPU](web/docs/CLIPVisionLoaderMultiGPU.md) / [CLIPVisionLoaderDisTorch2MultiGPU](web/docs/CLIPVisionLoaderDisTorch2MultiGPU.md)
  - [ControlNetLoaderMultiGPU](web/docs/ControlNetLoaderMultiGPU.md) / [ControlNetLoaderDisTorch2MultiGPU](web/docs/ControlNetLoaderDisTorch2MultiGPU.md)
  - [DiffusersLoaderMultiGPU](web/docs/DiffusersLoaderMultiGPU.md) / [DiffusersLoaderDisTorch2MultiGPU](web/docs/DiffusersLoaderDisTorch2MultiGPU.md)
  - [DiffControlNetLoaderMultiGPU](web/docs/DiffControlNetLoaderMultiGPU.md) / [DiffControlNetLoaderDisTorch2MultiGPU](web/docs/DiffControlNetLoaderDisTorch2MultiGPU.md)
- WanVideoWrapper (requires [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)):
  - [WanVideoModelLoaderMultiGPU](web/docs/WanVideoModelLoaderMultiGPU.md)
  - [WanVideoVAELoaderMultiGPU](web/docs/WanVideoVAELoaderMultiGPU.md)
  - [WanVideoTinyVAELoaderMultiGPU](web/docs/WanVideoTinyVAELoaderMultiGPU.md)
  - [WanVideoBlockSwapMultiGPU](web/docs/WanVideoBlockSwapMultiGPU.md)
  - [WanVideoImageToVideoEncodeMultiGPU](web/docs/WanVideoImageToVideoEncodeMultiGPU.md)
  - [WanVideoEncodeMultiGPU](web/docs/WanVideoEncodeMultiGPU.md)
  - [WanVideoDecodeMultiGPU](web/docs/WanVideoDecodeMultiGPU.md)
  - [WanVideoSamplerMultiGPU](web/docs/WanVideoSamplerMultiGPU.md)
  - [WanVideoVACEEncodeMultiGPU](web/docs/WanVideoVACEEncodeMultiGPU.md)
  - [WanVideoClipVisionEncodeMultiGPU](web/docs/WanVideoClipVisionEncodeMultiGPU.md)
  - [WanVideoControlnetLoaderMultiGPU](web/docs/WanVideoControlnetLoaderMultiGPU.md)
  - [WanVideoUni3C_ControlnetLoaderMultiGPU](web/docs/WanVideoUni3C_ControlnetLoaderMultiGPU.md)
  - [WanVideoTextEncodeMultiGPU](web/docs/WanVideoTextEncodeMultiGPU.md)
  - [WanVideoTextEncodeCachedMultiGPU](web/docs/WanVideoTextEncodeCachedMultiGPU.md)
  - [WanVideoTextEncodeSingleMultiGPU](web/docs/WanVideoTextEncodeSingleMultiGPU.md)
  - [LoadWanVideoT5TextEncoderMultiGPU](web/docs/LoadWanVideoT5TextEncoderMultiGPU.md)
  - [LoadWanVideoClipTextEncoderMultiGPU](web/docs/LoadWanVideoClipTextEncoderMultiGPU.md)
  - [FantasyTalkingModelLoaderMultiGPU](web/docs/FantasyTalkingModelLoaderMultiGPU.md)
  - [Wav2VecModelLoaderMultiGPU](web/docs/Wav2VecModelLoaderMultiGPU.md) / [DownloadAndLoadWav2VecModelMultiGPU](web/docs/DownloadAndLoadWav2VecModelMultiGPU.md)
- GGUF loaders (requires [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) or [ComfyUI-GGUF-Loader](https://github.com/ChrisColeTech/ComfyUI-GGUF-Loader)):
  - UNet family: [UnetLoaderGGUFMultiGPU](web/docs/UnetLoaderGGUFMultiGPU.md) / [UnetLoaderGGUFDisTorch2MultiGPU](web/docs/UnetLoaderGGUFDisTorch2MultiGPU.md)
  - UNet Advanced bundles: [UnetLoaderGGUFAdvancedMultiGPU](web/docs/UnetLoaderGGUFAdvancedMultiGPU.md) / [UnetLoaderGGUFAdvancedDisTorch2MultiGPU](web/docs/UnetLoaderGGUFAdvancedDisTorch2MultiGPU.md)
  - CLIP family: [CLIPLoaderGGUFMultiGPU](web/docs/CLIPLoaderGGUFMultiGPU.md) / [CLIPLoaderGGUFDisTorch2MultiGPU](web/docs/CLIPLoaderGGUFDisTorch2MultiGPU.md)
  - Dual CLIP: [DualCLIPLoaderGGUFMultiGPU](web/docs/DualCLIPLoaderGGUFMultiGPU.md) / [DualCLIPLoaderGGUFDisTorch2MultiGPU](web/docs/DualCLIPLoaderGGUFDisTorch2MultiGPU.md)
  - Triple CLIP: [TripleCLIPLoaderGGUFMultiGPU](web/docs/TripleCLIPLoaderGGUFMultiGPU.md) / [TripleCLIPLoaderGGUFDisTorch2MultiGPU](web/docs/TripleCLIPLoaderGGUFDisTorch2MultiGPU.md)
  - Quadruple CLIP: [QuadrupleCLIPLoaderGGUFMultiGPU](web/docs/QuadrupleCLIPLoaderGGUFMultiGPU.md) / [QuadrupleCLIPLoaderGGUFDisTorch2MultiGPU](web/docs/QuadrupleCLIPLoaderGGUFDisTorch2MultiGPU.md)
  - Text Encoder + ClipProj (ComfyUI-GGUF-Loader only): CCTechClipProjLoaderDisTorch2MultiGPU. Layers the allocation leaves on `cpu` stay memory-mapped on disk for a GGUF encoder and are streamed to the compute device one layer at a time.
- XLabAI FLUX ControlNet (requires [x-flux-comfy](https://github.com/XLabAI/x-flux-comfyui)):
  - [LoadFluxControlNetMultiGPU](web/docs/LoadFluxControlNetMultiGPU.md)
- Florence2 (requires [ComfyUI-Florence2](https://github.com/kijai/ComfyUI-Florence2)):
  - [Florence2ModelLoaderMultiGPU](web/docs/Florence2ModelLoaderMultiGPU.md)
  - [DownloadAndLoadFlorence2ModelMultiGPU](web/docs/DownloadAndLoadFlorence2ModelMultiGPU.md)
- LTX Video Custom Checkpoint Loader (requires [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo)):
  - [LTXVLoaderMultiGPU](web/docs/LTXVLoaderMultiGPU.md)
- NF4 Checkpoint Format Loader (requires [ComfyUI_bitsandbytes_NF4](https://github.com/comfyanonymous/ComfyUI_bitsandbytes_NF4)):
  - [CheckpointLoaderNF4MultiGPU](web/docs/CheckpointLoaderNF4MultiGPU.md)
- MMAudio (requires [ComfyUI-MMAudio](https://github.com/comfyanonymous/ComfyUI-MMAudio)):
  - [MMAudioModelLoaderMultiGPU](web/docs/MMAudioModelLoaderMultiGPU.md)
  - [MMAudioFeatureUtilsLoaderMultiGPU](web/docs/MMAudioFeatureUtilsLoaderMultiGPU.md)
  - [MMAudioSamplerMultiGPU](web/docs/MMAudioSamplerMultiGPU.md)
- Pulid (requires [PuLID_ComfyUI](https://github.com/cubiq/PuLID_ComfyUI)):
  - [PulidModelLoaderMultiGPU](web/docs/PulidModelLoaderMultiGPU.md)
  - [PulidInsightFaceLoaderMultiGPU](web/docs/PulidInsightFaceLoaderMultiGPU.md)
  - [PulidEvaClipLoaderMultiGPU](web/docs/PulidEvaClipLoaderMultiGPU.md)

All MultiGPU nodes available for your install can be found in the "multigpu" category in the node menu.

## Node Documentation

Detailed technical documentation is available for all **automatically-detected core MultiGPU and DisTorch2 nodes**, covering 70+ documented nodes with comprehensive parameter details, output specifications, and DisTorch2 allocation guidance where applicable.

- **To access documentation**: Click on any core MultiGPU or DisTorch2 node in ComfyUI and select "Help" (question mark inside a circle) from the resultant menu
- **Coverage**: All standard ComfyUI loader nodes (UNet, VAE, Checkpoints, CLIP, ControlNet, Diffusers) plus popular GGUF loader variants
- **Contents**: Input parameters with data types and descriptions, output specifications, usage examples, and DisTorch2 distributed loading explanations with allocation modes and strategies
- **Note**: Documentation covers core ComfyUI-MultiGPU functionality only. Third-party custom node integrations (WanVideoWrapper, Florence2, etc.) have their own separate documentation.

## Example workflows

These workflows come from upstream, where they were tested on NVIDIA setups (2x 3090 + 1060 Ti Linux, 4070 Windows 11, 3090 + 1070 Ti Linux). They have not all been re-tested on ROCm in this fork.

### DisTorch2

<table>
  <tr>
    <td align="center">
      <a href="example_workflows/ltxvideo%20checkpointloadersimple%20distorch2.json">
        <img src="example_workflows/ltxvideo%20checkpointloadersimple%20distorch2.jpg" alt="LTX Video + CheckpointLoaderSimple (DisTorch2)" style="max-width:160px; max-height:160px;">
        <div>LTX Video + CheckpointLoaderSimple (DisTorch2)</div>
      </a>
    </td>
    <td align="center">
      <a href="example_workflows/mochi%20checkpointloaderadvanced%20distorch2.json">
        <img src="example_workflows/mochi%20checkpointloaderadvanced%20distorch2.jpg" alt="Mochi + CheckpointLoaderAdvanced (DisTorch2)" style="max-width:160px; max-height:160px;">
        <div>Mochi + CheckpointLoaderAdvanced (DisTorch2)</div>
      </a>
    </td>
    <td align="center">
      <a href="example_workflows/qwen_image%20unet%20clip%20distorch2.json">
        <img src="example_workflows/qwen_image%20unet%20clip%20distorch2.jpg" alt="Qwen Image UNet + CLIP (DisTorch2)" style="max-width:160px; max-height:160px;">
        <div>Qwen Image UNet + CLIP (DisTorch2)</div>
      </a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="example_workflows/qwen_image_edit_2509%20unet%20clip%20distorch2.json">
        <img src="example_workflows/qwen_image_edit_2509%20unet%20clip%20distorch2.jpg" alt="Qwen Image Edit UNet + CLIP (DisTorch2)" style="max-width:160px; max-height:160px;">
        <div>Qwen Image Edit UNet + CLIP (DisTorch2)</div>
      </a>
    </td>
    <td align="center">
      <a href="example_workflows/wan2_2%20distorch2%20double_unet%20no_cpu.json">
        <img src="example_workflows/wan2_2%20distorch2%20double_unet%20no_cpu.jpg" alt="WanVideo 2.2 Double UNet, No CPU (DisTorch2)" style="max-width:160px; max-height:160px;">
        <div>WanVideo 2.2 Double UNet, No CPU (DisTorch2)</div>
      </a>
    </td>
    <td align="center">
      <a href="example_workflows/wan2_2%20t2i%20lightx2v%20lora%20distorch2.json">
        <img src="example_workflows/wan2_2%20t2i%20lightx2v%20lora%20distorch2.jpg" alt="WanVideo 2.2 T2I LightX2V LoRA (DisTorch2)" style="max-width:160px; max-height:160px;">
        <div>WanVideo 2.2 T2I LightX2V LoRA (DisTorch2)</div>
      </a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="example_workflows/wan2_2%20t2v%20lightx2v%20lora%20distorch2.json">
        <img src="example_workflows/wan2_2%20t2v%20lightx2v%20lora%20distorch2.jpg" alt="WanVideo 2.2 T2V LightX2V LoRA (DisTorch2)" style="max-width:160px; max-height:160px;">
        <div>WanVideo 2.2 T2V LightX2V LoRA (DisTorch2)</div>
      </a>
    </td>
    <td></td>
    <td></td>
  </tr>
</table>

### WanVideoWrapper

<table>
  <tr>
    <td align="center">
      <a href="example_workflows/ComfyUI-WanVideoWrapper%20wanvideo_T2V.json">
        <img src="example_workflows/ComfyUI-WanVideoWrapper%20wanvideo_T2V.jpg" alt="WanVideoWrapper T2V" style="max-width:160px; max-height:160px;">
        <div>WanVideoWrapper T2V</div>
      </a>
    </td>
    <td align="center">
      <a href="example_workflows/ComfyUI-WanVideoWrapper%20wanvideo_1_3B%20control_lora.json">
        <img src="example_workflows/ComfyUI-WanVideoWrapper%20wanvideo_1_3B%20control_lora.jpg" alt="WanVideoWrapper 1.3B Control LoRA" style="max-width:160px; max-height:160px;">
        <div>WanVideoWrapper 1.3B Control LoRA</div>
      </a>
    </td>
    <td align="center">
      <a href="example_workflows/ComfyUI-WanVideoWrapper%20wanvideo2_2%20I2V%20A14B%20GGUF.json">
        <img src="example_workflows/ComfyUI-WanVideoWrapper%20wanvideo2_2%20I2V%20A14B%20GGUF.jpg" alt="WanVideoWrapper 2.2 I2V A14B GGUF" style="max-width:160px; max-height:160px;">
        <div>WanVideoWrapper 2.2 I2V A14B GGUF</div>
      </a>
    </td>
  </tr>
</table>

### MultiGPU

<table>
  <tr>
    <td align="center">
      <a href="example_workflows/flux%20unet%20dual_clip%20vae%20loaders.json">
        <img src="example_workflows/flux%20unet%20dual_clip%20vae%20loaders.jpg" alt="FLUX UNet + Dual CLIP + VAE Loaders (MultiGPU)" style="max-width:160px; max-height:160px;">
        <div>FLUX UNet + Dual CLIP + VAE Loaders (MultiGPU)</div>
      </a>
    </td>
    <td align="center">
      <a href="example_workflows/sd15%20checkpoint%20loader%20simple.json">
        <img src="example_workflows/sd15%20checkpoint%20loader%20simple.jpg" alt="SD15 CheckpointLoaderSimple (MultiGPU)" style="max-width:160px; max-height:160px;">
        <div>SD15 CheckpointLoaderSimple (MultiGPU)</div>
      </a>
    </td>
    <td align="center">
      <a href="example_workflows/sdxl%20checkpoint%20loader%20advanced.json">
        <img src="example_workflows/sdxl%20checkpoint%20loader%20advanced.jpg" alt="SDXL CheckpointLoaderAdvanced (MultiGPU)" style="max-width:160px; max-height:160px;">
        <div>SDXL CheckpointLoaderAdvanced (MultiGPU)</div>
      </a>
    </td>
  </tr>
</table>

### GGUF

<table>
  <tr>
    <td align="center">
      <a href="example_workflows/ComfyUI-GGUF%20flux%20unet%20dual_clip%20loaders.json">
        <img src="example_workflows/ComfyUI-GGUF%20flux%20unet%20dual_clip%20loaders.jpg" alt="FLUX UNet + Dual CLIP GGUF" style="max-width:160px; max-height:160px;">
        <div>FLUX UNet + Dual CLIP GGUF</div>
      </a>
    </td>
    <td align="center">
      <a href="example_workflows/ComfyUI-GGUF%20qwen_image%20unet%20distorch2%20cliploader.json">
        <img src="example_workflows/ComfyUI-GGUF%20qwen_image%20unet%20distorch2%20cliploader.jpg" alt="Qwen Image UNet DisTorch2 GGUF" style="max-width:160px; max-height:160px;">
        <div>Qwen Image UNet DisTorch2 GGUF</div>
      </a>
    </td>
    <td></td>
  </tr>
</table>

### HunyuanVideoWrapper / Florence2

<table>
  <tr>
    <td align="center">
      <a href="example_workflows/hunyuanvideo%20distorch%20DEPRECATED.json">
        <img src="example_workflows/hunyuanvideo%20distorch%20DEPRECATED.jpg" alt="HunyuanVideoWrapper DisTorch (Legacy, Deprecated)" style="max-width:160px; max-height:160px;">
        <div>HunyuanVideoWrapper DisTorch (Legacy, Deprecated)</div>
      </a>
    </td>
    <td align="center">
      <a href="example_workflows/ComfyUI-Florence2%20detailed_caption%20to%20flux.json">
        <img src="example_workflows/ComfyUI-Florence2%20detailed_caption%20to%20flux.jpg" alt="Florence2 Detailed Caption to FLUX Pipeline" style="max-width:160px; max-height:160px;">
        <div>Florence2 Detailed Caption to FLUX Pipeline</div>
      </a>
    </td>
  </tr>
</table>

## Support

This is a personal fork, maintained on a best-effort basis and tested only on AMD ROCm. Report issues with the ROCm changes at [Zakhrov/ComfyUI-MultiGPU](https://github.com/Zakhrov/ComfyUI-MultiGPU/issues), not upstream. Upstream maintenance has ended and the upstream repository is being archived on 30 September 2026; see its [pinned issue](https://github.com/pollockjj/ComfyUI-MultiGPU/issues/223) for fork coordination.

## Credits

ROCm donor GEMM changes by [Aaron Zakhrov](https://github.com/Zakhrov).
Upstream maintained by [pollockjj](https://github.com/pollockjj) until September 2026.
Originally created by [Alexander Dzhoganov](https://github.com/AlexanderDzhoganov).
With deepest thanks to [City96](https://v100s.net/).
