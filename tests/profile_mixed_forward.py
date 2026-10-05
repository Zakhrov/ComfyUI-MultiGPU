"""Profile one steady-state sampling step of a DisTorch2 model on both GPUs.

Run from anywhere with ComfyUI's venv:
    .venv/bin/python custom_nodes/ComfyUI-MultiGPU/tests/profile_mixed_forward.py --mode mixed_int8
Re-analyze a saved trace without a GPU:
    ... --analyze /tmp/mixed_int8.trace.json

The step after the first is profiled (the first loads and converts weights), and
the step after that is timed without the profiler.
"""
import argparse
import collections
import json
import os
import sys
import time

COMFY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def total(intervals):
    return sum(end - start for start, end in intervals)


def intersect(a, b):
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        start, end = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if start < end:
            out.append([start, end])
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def complement(a, start, end):
    out, cursor = [], start
    for s, e in a:
        if s > cursor:
            out.append([cursor, s])
        cursor = max(cursor, e)
    if cursor < end:
        out.append([cursor, end])
    return out


def classify_kernel(name):
    lowered = name.lower()
    if "cijk" in lowered or "gemm" in lowered or "int8" in lowered or "matmul" in lowered:
        return "gemm"
    if "attn" in lowered or "attention" in lowered or "fmha" in lowered or "flash" in lowered or "ck_tile" in lowered:
        return "attention"
    return "other"


def copy_kind(name):
    for kind in ("HtoD", "DtoH", "DtoD", "PtoP"):
        if kind in name:
            return kind
    return "other"


def innermost_label(annotations):
    """Return a lookup from a timestamp to the innermost mgpu annotation covering it."""
    events = sorted(annotations, key=lambda a: (a[0], -a[1]))
    starts = [e[0] for e in events]
    import bisect

    def lookup(ts):
        index = bisect.bisect_right(starts, ts)
        best = None
        for candidate in events[max(0, index - 64):index]:
            if candidate[0] <= ts <= candidate[1] and (best is None or candidate[0] >= best[0]):
                best = candidate
        return best[2] if best else "(outside mgpu units)"

    return lookup


def analyze(trace_path, compute=0, donor=1):
    with open(trace_path) as f:
        events = json.load(f)["traceEvents"]
    kernels = collections.defaultdict(list)
    copies = collections.defaultdict(list)
    kernel_names = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0.0]))
    copy_stats = collections.defaultdict(lambda: [0, 0.0, 0])
    runtime = {}
    runtime_by_api = collections.defaultdict(lambda: [0, 0.0])
    annotations = []
    kernel_events = []
    main_thread = None
    for e in events:
        if e.get("ph") != "X":
            continue
        cat, name, ts, dur = e.get("cat"), e.get("name", ""), e["ts"], e.get("dur", 0)
        args = e.get("args", {})
        if cat == "kernel":
            device = args.get("device", e["pid"])
            kernels[device].append((ts, ts + dur))
            stats = kernel_names[device][name]
            stats[0] += 1
            stats[1] += dur
            kernel_events.append((ts, dur, device, args.get("correlation"), "kernel", name))
        elif cat in ("gpu_memcpy", "gpu_memset"):
            device = args.get("device", e["pid"])
            copies[device].append((ts, ts + dur))
            stats = copy_stats[(device, copy_kind(name))]
            stats[0] += 1
            stats[1] += dur
            stats[2] += args.get("bytes", 0)
            kernel_events.append((ts, dur, device, args.get("correlation"), "copy", name))
        elif cat in ("cuda_runtime", "cuda_driver"):
            runtime[args.get("correlation")] = (ts, e["tid"])
            api = runtime_by_api[name]
            api[0] += 1
            api[1] += dur
        elif cat == "user_annotation" and name.startswith("mgpu::"):
            annotations.append((ts, ts + dur, name))
            main_thread = e["tid"]
    if not kernels:
        raise SystemExit("no kernel events in trace")

    all_busy = [i for d in kernels for i in kernels[d]] + [i for d in copies for i in copies[d]]
    t0, t1 = min(s for s, _ in all_busy), max(e for _, e in all_busy)
    window = t1 - t0
    ms = lambda us: us / 1000.0

    k = {d: union(kernels[d]) for d in kernels}
    c = {d: union(copies[d]) for d in copies}
    all_copy = union([i for d in c for i in c[d]])

    print(f"\n=== window {ms(window):.1f} ms, {sum(len(v) for v in kernels.values())} kernels, "
          f"{sum(len(v) for v in copies.values())} copies/memsets ===")
    for d in sorted(kernels):
        by_class = collections.defaultdict(float)
        for name, (count, dur) in kernel_names[d].items():
            by_class[classify_kernel(name)] += dur
        print(f"cuda:{d}: kernel-busy {ms(total(k[d])):.1f} ms ({100*total(k[d])/window:.0f}%)  "
              f"copy-engine {ms(total(c.get(d, []))):.1f} ms  "
              + "  ".join(f"{cls}={ms(v):.0f}ms" for cls, v in sorted(by_class.items())))

    comp, don = k.get(compute, []), k.get(donor, [])
    both = intersect(comp, don)
    print(f"\nboth GPUs running kernels at once: {ms(total(both)):.1f} ms ({100*total(both)/window:.0f}% of window)")
    for label, idle_gpu, other_busy, other_name in (
        ("compute GPU", complement(comp, t0, t1), don, "donor"),
        ("donor GPU", complement(don, t0, t1), comp, "compute"),
    ):
        idle_total = total(idle_gpu)
        with_other = intersect(idle_gpu, other_busy)
        remaining = complement(union(with_other), t0, t1)
        with_copy = intersect(intersect(idle_gpu, all_copy), remaining)
        rest = total(idle_gpu) - total(with_other) - total(with_copy)
        print(f"{label} idle {ms(idle_total):.1f} ms ({100*idle_total/window:.0f}%): "
              f"while {other_name} runs kernels {ms(total(with_other)):.1f} | "
              f"copies only {ms(total(with_copy)):.1f} | nothing on GPU (host/sync gaps) {ms(rest):.1f}")

    print("\ncopies (device, kind): count, total ms, GB, GB/s while copying")
    for (device, kind), (count, dur, nbytes) in sorted(copy_stats.items()):
        rate = nbytes / dur * 1e-3 if dur else 0
        print(f"  cuda:{device} {kind:5s} {count:6d}  {ms(dur):9.1f} ms  {nbytes/1e9:7.2f} GB  {rate:6.2f} GB/s")

    lookup = innermost_label(annotations)
    per_label = collections.defaultdict(lambda: collections.defaultdict(float))
    for ts, dur, device, correlation, kind, name in kernel_events:
        launch = runtime.get(correlation)
        label = lookup(launch[0]) if launch else "(unattributed)"
        key = f"cuda:{device} {kind}"
        per_label[label][key] += dur
    if annotations:
        print("\nGPU time by innermost mgpu annotation (ms, summed over kernels, not unions):")
        columns = sorted({key for v in per_label.values() for key in v})
        print("  " + "label".ljust(34) + "".join(col.rjust(16) for col in columns))
        for label, values in sorted(per_label.items(), key=lambda item: -sum(item[1].values())):
            print("  " + label.ljust(34) + "".join(f"{ms(values.get(col, 0)):16.1f}" for col in columns))
        counts = collections.Counter(a[2] for a in annotations)
        host = collections.defaultdict(float)
        for s, e, name in annotations:
            host[name] += e - s
        print("\nhost-side time per annotation (inclusive, ms) and calls:")
        for name in sorted(host, key=host.get, reverse=True):
            print(f"  {name:34s} {ms(host[name]):9.1f} ms  {counts[name]:6d} calls")

    print("\ntop runtime API calls on the host (ms total, calls):")
    for name, (count, dur) in sorted(runtime_by_api.items(), key=lambda item: -item[1][1])[:10]:
        print(f"  {name:40s} {ms(dur):9.1f} ms  {count:7d}")

    for d in sorted(kernel_names):
        print(f"\ntop kernels on cuda:{d} (count, total ms):")
        for name, (count, dur) in sorted(kernel_names[d].items(), key=lambda item: -item[1][1])[:8]:
            print(f"  {count:6d} {ms(dur):9.1f}  {name[:110]}")


def run(args):
    os.chdir(COMFY_ROOT)
    sys.path.insert(0, COMFY_ROOT)
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
    sys.argv = [
        "main.py", "--cuda-malloc", "--disable-dynamic-vram", "--use-ck-attention",
        "--disable-all-custom-nodes", "--whitelist-custom-nodes", "ComfyUI-MultiGPU", "ComfyUI-GGUF-Loader",
        "--disable-api-nodes",
    ]
    import comfy.options

    comfy.options.enable_args_parsing()
    import cuda_malloc  # noqa: F401  (sets the allocator before torch initializes)
    import asyncio
    import torch
    import comfy.sample
    import comfy.utils
    import comfy.model_management
    import nodes

    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False))

    distorch = next(m for name, m in sys.modules.items() if name.endswith("distorch_2") and hasattr(m, "_run_fused_mlp"))
    from torch.profiler import ProfilerActivity, profile, record_function

    def annotate(name, label):
        original = getattr(distorch, name)

        def wrapped(*a, **kw):
            with record_function(label):
                return original(*a, **kw)

        setattr(distorch, name, wrapped)

    for name, label in (
        ("_run_staged_linear", "mgpu::linear_gemm"),
        ("_run_fused_mlp", "mgpu::fused_mlp"),
        ("_run_quantized_linear_on_compute", "mgpu::quant_linear"),
        ("_run_donor_prepared_linear_on_compute", "mgpu::donor_prepared_linear"),
        ("_run_tiled_linear_on_compute", "mgpu::tiled_linear"),
        ("_run_attention_on_compute", "mgpu::attention"),
        ("_stage_linear", "mgpu::stage_next_weights"),
        ("_compute_budget", "mgpu::budget_sync"),
        ("_run_tiled_conv_on_compute", "mgpu::conv"),
    ):
        annotate(name, label)

    if args.gguf:
        loader = nodes.NODE_CLASS_MAPPINGS["UnetLoaderGGUFDisTorch2MultiGPU"]()
        (model,) = loader.override(
            unet_name=args.gguf, compute_device="cuda:0", virtual_vram_gb=args.virtual_vram,
            donor_device="cuda:1", donor_gemm_execution_mode=args.mode, expert_mode_allocations="", eject_models=True,
        )
    else:
        loader = nodes.NODE_CLASS_MAPPINGS["UNETLoaderDisTorch2MultiGPU"]()
        (model,) = loader.override(
            unet_name=args.unet, weight_dtype="default", compute_device="cuda:0",
            virtual_vram_gb=args.virtual_vram, donor_device="cuda:1" if args.mode != "disabled" else "cpu",
            donor_gemm_execution_mode=args.mode, expert_mode_allocations="", eject_models=True,
        )

    latent = torch.zeros(1, 16, args.size // 8, args.size // 8)
    noise = comfy.sample.prepare_noise(latent, 0)
    context = torch.randn(1, args.ctx_len, args.ctx_dim) * 0.1
    conditioning = [[context, {}]]

    clock = {}
    step_marks = []
    prof = {"p": None}

    def sync():
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)

    def callback(step, x0, x, total_steps):
        sync()
        now = time.perf_counter()
        step_marks.append(now)
        print(f"step {step} finished, +{now - clock['last']:.1f}s", flush=True)
        clock["last"] = now
        if step == 0 and not args.no_profile:
            prof["p"] = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False, with_stack=args.stacks)
            prof["p"].__enter__()
            clock["profile_start"] = time.perf_counter()
        elif step == 1 and prof["p"] is not None:
            prof["p"].__exit__(None, None, None)
            clock["profile_wall"] = time.perf_counter() - clock["profile_start"]
            prof["p"].export_chrome_trace(args.trace)
            clock["last"] = time.perf_counter()
            print(f"profiled step wall {clock['profile_wall']:.1f}s, trace -> {args.trace}", flush=True)

    clock["last"] = time.perf_counter()
    result = comfy.sample.sample(
        model, noise, 3, 1.0, "euler", "simple", conditioning, conditioning, latent,
        callback=callback, disable_pbar=True, seed=0,
    )
    print(f"result mean {result.float().mean():.6f} std {result.float().std():.6f}", flush=True)
    print(f"\nunprofiled step 2 wall: see the 'step 2 finished' line above (mode={args.mode})")
    if not args.no_profile:
        analyze(args.trace)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="mixed_int8", choices=("disabled", "mixed", "mixed_int8"))
    parser.add_argument("--unet", default="z_image_turbo_int8_convrot.safetensors")
    parser.add_argument("--gguf", default=None, help="GGUF unet name, e.g. unsloth/z-image-turbo-Q8_0.gguf")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--virtual-vram", type=float, default=4.0)
    parser.add_argument("--ctx-len", type=int, default=64)
    parser.add_argument("--ctx-dim", type=int, default=2560)
    parser.add_argument("--trace", default="/tmp/mixed_forward.trace.json")
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--stacks", action="store_true", help="record Python stacks so slow runtime calls can be attributed to source lines")
    parser.add_argument("--analyze", default=None)
    parsed = parser.parse_args()
    if parsed.analyze:
        analyze(parsed.analyze)
    else:
        run(parsed)
