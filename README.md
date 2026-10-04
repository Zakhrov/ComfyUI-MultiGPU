# ComfyUI-MultiGPU (AMD ROCm fork)

A fork of [pollockjj/ComfyUI-MultiGPU](https://github.com/pollockjj/ComfyUI-MultiGPU) that adds donor-GPU GEMM execution for multi-GPU AMD ROCm systems on top of the upstream DisTorch2 model distribution.

> [!WARNING]
> **AMD ROCm only.** The changes in this fork are developed and tested exclusively on AMD GPUs with ROCm (HIP builds of PyTorch). They are **not** tested on, and should be assumed **not to work** on, NVIDIA CUDA, Intel XPU, Apple MPS, DirectML or any other torch backend. If you are not on ROCm, use [upstream ComfyUI-MultiGPU](https://github.com/pollockjj/ComfyUI-MultiGPU) instead.

> [!CAUTION]
> **Requires a custom, unofficial Comfy Kitchen HIP backend.** The donor GEMM modes depend on the `hip-bringup` branch of a Comfy Kitchen fork: [Zakhrov/comfy-kitchen@hip-bringup](https://github.com/Zakhrov/comfy-kitchen/tree/hip-bringup). This backend is experimental, is not part of official Comfy Kitchen releases, and is not supported by the Comfy Kitchen or ComfyUI maintainers. Expect breakage when ComfyUI or Comfy Kitchen update. Do not report problems caused by it to those projects.

## What's different from upstream

| Area | Change |
| --- | --- |
| Donor GEMM execution | New `donor_gemm_execution_mode` input (`disabled`, `mixed`, `mixed_int8`, `all`) on DisTorch2 loaders and the DisTorch2 checkpoint loader. Runs linear, conv and attention work split across a compute GPU and a donor GPU instead of copying donor weights back for every op. See [ROCm donor GEMM execution](#rocm-donor-gemm-execution). |
| VAE | `VAELoaderDisTorch2MultiGPU` supports `mixed` mode for image and video VAE encode/decode, including temporal caches such as Wan's. |
| GGUF text encoder | New `CCTechClipProjLoaderDisTorch2MultiGPU` for [ComfyUI-GGUF-Loader](https://github.com/ChrisColeTech/ComfyUI-GGUF-Loader)'s Text Encoder + ClipProj Loader. Layers the allocation leaves on `cpu` stay memory-mapped on disk and are streamed one layer at a time. |
| WanVideoWrapper | The WanVideo model loader falls back to the offload-aware default RMSNorm when PyTorch RMSNorm is selected, since the native one is incompatible with WanVideo VRAM management. |
| Tests | `tests/test_distorch_hip_donor.py` covers the donor GEMM paths. |

With `donor_gemm_execution_mode` left at `disabled` (the default), DisTorch2 behaves like upstream. Even so, the rest of this fork has only been exercised on ROCm.

## About DisTorch2

DisTorch ("distributed torch") moves the static parts of your main model off your compute GPU to a donor device — system RAM or another GPU's VRAM — so the compute GPU's VRAM is free for latents, longer videos or bigger batches. Pick one or more donor devices and how much of the model to put on each, and DisTorch handles the rest. It works with all `.safetensors` and GGUF models.

- **Normal mode**: the `virtual_vram_gb` slider picks how much of the model moves to a single donor device (such as system RAM).
- **Expert mode**: an allocation string that sets exactly how the model is split across devices:
  - **Bytes (recommended)**: gigabytes or megabytes per device, like Hugging Face's `device_map`. `*` assigns the remainder (the CPU is the default wildcard).
    - `cuda:0,2.5gb;cpu,*` — first 2.5 GB on `cuda:0`, the rest on `cpu`.
    - `cuda:0,500mb;cuda:1,3.0g;cpu,5gb*` — 0.5 GB on `cuda:0`, 3 GB on `cuda:1`, the remainder on `cpu`.
  - **Ratio**: like `llama.cpp`'s `tensor_split`.
    - `cuda:0,25%;cpu,75%` — a 1:3 split.
    - `cuda:0,8%;cuda:1,8%;cpu,4%` — 40% / 40% / 20%.
  - **Fraction**: the original DisTorch mode, a fraction of each device's *total* memory.
    - `cuda:0,0.1;cpu,0.5` — 10% of `cuda:0`'s VRAM and 50% of system RAM.

On ROCm, PyTorch still names AMD GPUs `cuda:N`, so these strings are unchanged.

On ComfyUI builds with DynamicVRAM/comfy-aimdo enabled, MultiGPU keeps DynamicVRAM on devices comfy-aimdo initialized and falls back to legacy model patching for other MultiGPU devices.

## ROCm donor GEMM execution

With the [Comfy Kitchen HIP backend](https://github.com/Zakhrov/comfy-kitchen/tree/hip-bringup) installed, DisTorch2 can run donor-assigned layers across two AMD GPUs instead of copying their weights to the compute GPU for every op.

### Supported GPUs

Both the compute and the donor GPU must use the HIP software-GEMM path:

- Vega / GCN5: `gfx900`, `gfx906`, `gfx90c`
- RDNA1: `gfx1010`–`gfx1012`
- RDNA2: `gfx1030`–`gfx1036`

RDNA3 and later, and CDNA (native WMMA) GPUs keep normal DisTorch2 placement. To confirm the donor path is in use, check the ComfyUI log:

- `mixed` / `mixed_int8`: `[MultiGPU DisTorch V2] Mixed execution: activations on cuda:1, GEMMs and attention on cuda:0`. If a requirement isn't met, `Mixed mode disabled: <reasons>` is logged and the model loads normally.
- `all`: `[MultiGPU DisTorch V2] Donor GEMM active: cuda:0 -> cuda:1`.

Donor execution trades PCIe transfer bandwidth and latency for lower peak VRAM on the compute GPU. Use Expert mode to put a meaningful group of layers on the donor, for example `cuda:0,2gb;cuda:1,4gb;cpu,*`.

### Modes

- **`disabled`** (default): standard ComfyUI/DisTorch2 execution.

- **`mixed`**: the donor GPU runs the model and the compute GPU does only the heavy math. Built for AMD+AMD laptops and systems; tested on a Dell G5 15 SE (RX 5600M 6 GB + Ryzen Vega APU).
  - Activations, norms, modulation, RoPE, residuals, dequantization and weight patches stay on the donor. The donor's memory holds activations, so high-resolution images and long videos are limited by donor memory (for example an APU using shared system RAM), not compute-GPU VRAM.
  - Linear GEMMs are sent to the compute GPU in token and weight tiles sized to its free VRAM, and each result tile is copied back to the donor. Work is split into at least 8 chunks, and the copies run on a side stream overlapping neighbouring chunks' compute. At most two chunks are in flight.
  - Each linear prepares the next linear's weight on a separate donor stream while the current GEMM runs. Nothing is staged across forwards.
  - Attention runs on the compute GPU in chunks of whole heads. On OOM the chunk is halved and the smaller size is kept until the model reloads; attention falls back to the donor only if a single head doesn't fit. Requires Comfy Kitchen attention.
  - Weights stay where they were loaded (GGUF memory-mapped on disk, safetensors in RAM) and are prepared on the donor one layer at a time. The compute GPU holds no weights, only GEMM tiles and attention chunks. The allocation only selects the donor (the first GPU donor listed).
  - ComfyUI quantized safetensors layers (int8, fp8) send their packed weight to the compute GPU and run their own quantized matmul there.
  - Plain LoRAs on GGUF models run as two thin GEMMs on the compute GPU instead of being merged into the dequantized weight on every call. DoRA, LoCon and other patch types are still merged.
  - Conv2d and Conv3d: the compute GPU receives only the input each tile needs, in tiles of at most 128 MiB split across height (and frames for Conv3d). Grouped convs run on the donor.
  - With `VAELoaderDisTorch2MultiGPU`, image and video VAE encode/decode run on the donor with their conv and linear GEMMs on the compute GPU. VAE attention stays on the donor in `mixed`, and VAEs don't require Comfy Kitchen attention.

- **`mixed_int8`**: `mixed`, plus a one-time conversion at load of every linear (GGUF Q4/Q5/Q8 and others, fp16/bf16 safetensors) to ComfyUI's int8 ConvRot layout, run with Comfy Kitchen's int8 GEMM. Requires Comfy Kitchen attention.
  - The int8 weights live in system RAM (about 1 byte per weight, roughly twice a Q4 model) while the model is loaded or cached. The donor keeps only activations, so it has more room than in `mixed`.
  - Reloading with the same LoRAs reuses the converted weights; changing LoRAs converts again.
  - Requantizing a GGUF weight adds a second rounding on top of its own quantization.
  - Plain LoRAs, linears whose input features aren't a multiple of 256, and convs run at full precision as in `mixed`.
  - VAEs convert their linears too, and their attention runs on the compute GPU in chunks of frames or heads, with ComfyUI's selected attention (Comfy Kitchen) where the head dim is at most 256 and PyTorch SDPA otherwise. Everything else stays on the donor as in `mixed`. On the RX 5600M + Vega APU pair: MiniMax H3 decode 13.5 s → 4.8 s (17 frames, 256²), Flux encode 5.0 s → 3.7 s and decode 8.2 s → 6.9 s (1024², identical output), Wan 2.1 encode 36.9 s → 34.5 s (17 frames, 480×832).

- **`all`**: every eligible donor-resident linear GEMM runs entirely on the donor. A crude SLI/Crossfire for inference; best with a roughly 50/50 split on identical or similar GPUs (for example 2× RX 5700 XT or 2× Radeon VII). Only this mode pins CPU weights, since only donor GEMMs read host memory directly.

### Benchmarks

Z-Image Turbo, 1024×1024, 8 steps, `res_multistep`/`simple`, CFG 1, on a Dell G5 15 SE: RX 5600M 6 GB (`gfx1010`, `cuda:0`, compute) + Ryzen Vega APU (`gfx90c`, `cuda:1`, donor, shared system RAM). PyTorch 2.12 ROCm nightly (HIP 7.17), `--cuda-malloc --disable-dynamic-vram --use-ck-attention`. Runs use `UnetLoaderGGUFDisTorch2MultiGPU` (GGUF) or `UNETLoaderDisTorch2MultiGPU` (int8_convrot) with `virtual_vram_gb` 4.0. The `disabled` baseline uses the default `cpu` donor and the other modes use `cuda:1`. The text encoder and VAE run on the CPU and are not counted.

Time is the KSampler node's wall time (including per-run weight loading), averaged over 2–3 runs. VRAM is the peak from sysfs during sampling. The APU's 512 MB VRAM carve-out is always full, so its usage is shown as GTT (system RAM mapped to the GPU), which includes about 0.5–0.8 GB of desktop usage at idle. The first `disabled` GGUF run, which also loaded the CPU text encoder, peaked at 4.89 GB GTT and is left out of that column. All modes produced matching images.

| Format | Mode | Time | vs. `disabled` | RX 5600M peak VRAM | APU peak GTT |
| --- | --- | --- | --- | --- | --- |
| GGUF Q8_0 (7.2 GB) | `disabled` | 255 s | baseline | 4.98 GB | 0.84 GB |
| | `mixed` | 330 s | 0.77× speed | 0.73 GB (−85%) | 1.81 GB |
| | `mixed_int8` | 220 s | **1.16× speed** | **0.62 GB (−88%)** | 1.37 GB |
| | `all` | 744 s | 0.34× speed | 4.24 GB (−15%) | 5.98 GB |
| int8_convrot (6.2 GB) | `disabled` | 156 s | baseline | 3.43 GB | 0.85 GB |
| | `mixed` | 274 s | 0.57× speed | 0.63 GB (−82%) | 1.42 GB |
| | `mixed_int8` | 277 s | 0.56× speed | 0.64 GB (−81%) | 1.27 GB |
| | `all` | 483 s | 0.32× speed | 3.23 GB (−6%) | 5.48 GB |

- `mixed` and `mixed_int8` cut peak compute-GPU VRAM by over 80% for both formats. That room can go to larger latents or longer videos.
- For GGUF, `mixed_int8` is also faster than the baseline, because Kitchen's int8 GEMM replaces per-layer dequantization. An int8_convrot checkpoint is already int8, so `mixed_int8` behaves like `mixed`, and the `disabled` baseline is already fast.
- `all` is slow on this pair because the Vega APU is much slower than the RX 5600M at GEMMs. It is meant for two similar GPUs.

## Installation

ComfyUI-Manager installs upstream, not this fork. Install manually:

1. Clone this repository into `ComfyUI/custom_nodes/`:
   ```bash
   git clone https://github.com/Zakhrov/ComfyUI-MultiGPU
   ```
2. For donor GEMM modes, install the Comfy Kitchen HIP backend from [Zakhrov/comfy-kitchen@hip-bringup](https://github.com/Zakhrov/comfy-kitchen/tree/hip-bringup) into the same Python environment as ComfyUI, replacing any official `comfy-kitchen` package.
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
