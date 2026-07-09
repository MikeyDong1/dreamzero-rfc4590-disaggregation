#!/usr/bin/env python3
"""Profile the DreamZero Wan-VAE encode under torch.compile(inductor) vs eager,
and COUNT how many XPU kernels are launched in each setting.

Reproduces the exact obs#1 VAE encode DreamZeroPipeline._encode_image performs:
  stitched first frame -> (1,3,1,352,640) bf16 -> concat 32 zero frames ->
  (1,3,33,352,640) -> vae._encode -> chunk -> latent (1,16,9,44,80),
  faithful path = autocast(bf16).

Uses STOCK diffusers AutoencoderKLWan (default Wan2.1 config). Per prior work this
is graph-/timing-identical to vllm_omni's DistributedAutoencoderKLWan._encode on
the single-card non-distributed path, and kernel COUNT is purely structural
(weight-value-independent). Critically, stock diffusers avoids `import vllm_omni`,
which disables triton-xpu and would break the inductor backend.

Two settings, each run TWICE (the two passes verify the kernel count is stable):
  - "inductor": encode wrapped in torch.compile(backend="inductor")
  - "eager":    plain eager encode (== enforce_eager)

Kernel-launch counting: torch.profiler with ProfilerActivity.CPU+XPU records the
host-side Level-Zero/UR launch API (`urEnqueueKernelLaunch`) once per kernel
launched, plus the device-side kernel executions. We report BOTH:
  - LAUNCHES/encode  = count of the launch-API event / profiled reps  (headline)
  - DEVICE_KERNELS/encode = count of device kernel-category events / profiled reps
  - distinct kernel names (fusion indicator)
We warm up (compile + allocator + JIT reach steady state) BEFORE the profiled
window so compilation is never counted as launches.

Single process; ZE_AFFINITY_MASK pins one card.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLWan

from torch.profiler import profile, ProfilerActivity, schedule

DEVICE = "xpu"
NUM_FRAMES = 33  # ah_config["num_frames"]
H, W = 352, 640


def log(m):
    print(m, flush=True)


def build_input(device, stitched_npz=None):
    """Recreate the (1,3,33,352,640) bf16 VAE input for obs#1 (kernel count is
    value-independent, so a synthetic frame is fine if the npz is absent)."""
    if stitched_npz and Path(stitched_npz).exists():
        z = np.load(stitched_npz)
        stitched = z["images"]  # (1,352,640,3) uint8
        if stitched.ndim == 3:
            stitched = stitched[None]
        v = torch.from_numpy(stitched).unsqueeze(0).to(device)      # B,T,H,W,C
        v = v.permute(0, 4, 1, 2, 3).float() / 255.0                # B,C,T,H,W
        v = v.to(torch.bfloat16) * 2.0 - 1.0                        # [-1,1]
        first = v[:, :, :1]                                         # (1,3,1,H,W)
        src = "npz"
    else:
        g = torch.Generator(device="cpu").manual_seed(0)
        first = (torch.rand(1, 3, 1, H, W, generator=g) * 2 - 1).to(device=device, dtype=torch.bfloat16)
        src = "synthetic"
    zeros = torch.zeros(1, 3, NUM_FRAMES - 1, first.shape[-2], first.shape[-1],
                        dtype=first.dtype, device=device)
    return torch.concat([first, zeros], dim=2), src  # (1,3,33,H,W)


# Host-side Level-Zero/UR runtime rows. On this XPU build these are CPU-typed;
# the equal-count urEnqueueKernelLaunch and zeCommandListAppendLaunchKernel each
# fire exactly once per compute-kernel dispatch -> the literal "kernels launched".
KERNEL_LAUNCH_ROWS = ("urEnqueueKernelLaunch", "zeCommandListAppendLaunchKernel")
MEMCPY_ROWS = ("urEnqueueUSMMemcpy", "urEnqueueUSMMemcpy2D")
# runtime-API name prefixes to EXCLUDE when tallying true device-side kernels
RUNTIME_PREFIXES = ("ur", "ze")


def _rowcount(ka, names):
    return sum(int(e.count) for e in ka if e.key in names)


def count_events(prof, reps):
    """Per-encode kernel counts from a completed profiler.

    Headline = `urEnqueueKernelLaunch` count (compute-kernel dispatches). Also
    reports memcpy launches and the true device-side kernel-execution total
    (events tagged DeviceType.XPU that are actual kernels, not runtime API)."""
    ka = prof.key_averages()

    launch_total = _rowcount(ka, ("urEnqueueKernelLaunch",))
    ze_launch_total = _rowcount(ka, ("zeCommandListAppendLaunchKernel",))
    memcpy_total = _rowcount(ka, MEMCPY_ROWS)

    # true device-side kernel executions: DeviceType.XPU events that are not
    # runtime-API rows (ur*/ze*) and not memcpy. Tally distinct kernel symbols.
    dev_kernel_total = 0
    dev_kernel_names = {}
    xpu_device_events = 0
    try:
        from torch.autograd import DeviceType
        xpu_dt = getattr(DeviceType, "XPU", None)
        for ev in prof.events():
            if getattr(ev, "device_type", None) != xpu_dt:
                continue
            xpu_device_events += 1
            name = ev.name
            if name.startswith(RUNTIME_PREFIXES) or "Memcpy" in name or "Memset" in name:
                continue
            dev_kernel_total += 1
            dev_kernel_names[name] = dev_kernel_names.get(name, 0) + 1
    except Exception as exc:  # noqa: BLE001
        log(f"[warn] device-kernel enumeration failed: {exc!r}")

    top_dev = sorted(dev_kernel_names.items(), key=lambda kv: kv[1], reverse=True)[:25]
    return {
        "launches_total": launch_total,
        "launches_per_encode": launch_total / reps if reps else 0.0,
        "ze_launch_total": ze_launch_total,          # sanity: should == launches_total
        "memcpy_launches_total": memcpy_total,
        "memcpy_per_encode": memcpy_total / reps if reps else 0.0,
        "xpu_device_events_total": xpu_device_events,
        "device_kernel_events_total": dev_kernel_total,
        "device_kernels_per_encode": dev_kernel_total / reps if reps else 0.0,
        "distinct_device_kernel_names": len(dev_kernel_names),
        "top_device_kernels": [{"name": n[:90], "count": c} for n, c in top_dev],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stitched-npz", default=None)
    ap.add_argument("--outdir", default="/work/vae_inductor_vs_eager")
    ap.add_argument("--device-name", default="")
    ap.add_argument("--warmup", type=int, default=4, help="warm reps (compile+JIT) before profiling")
    ap.add_argument("--active", type=int, default=5, help="profiled reps per pass")
    ap.add_argument("--passes", type=int, default=2, help="how many times to repeat each setting")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(DEVICE, 0)
    assert torch.xpu.device_count() >= 1, "no XPU visible"
    dev_name = args.device_name or torch.xpu.get_device_name(0)

    t0 = time.perf_counter()
    vae = AutoencoderKLWan().eval().to(device=device, dtype=torch.float32)
    latents_mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1).to(device)
    latents_inv_std = (1.0 / torch.tensor(vae.config.latents_std, dtype=torch.float32)).view(1, -1, 1, 1, 1).to(device)
    torch.xpu.synchronize()
    build_s = time.perf_counter() - t0

    vae_input, src = build_input(device, args.stitched_npz)
    log(f"[vae] device='{dev_name}' torch={torch.__version__} built in {build_s:.2f}s "
        f"input={tuple(vae_input.shape)} src={src}")

    def _encode_core(inp):
        with torch.amp.autocast(dtype=torch.bfloat16, device_type="xpu"):
            h = vae._encode(inp)
            mu, _ = h.chunk(2, dim=1)
            mu = (mu - latents_mean) * latents_inv_std
        return mu

    def make_fn(mode):
        if mode == "inductor":
            compiled = torch.compile(_encode_core, backend="inductor")
            def fn(inp):
                with torch.no_grad():
                    return compiled(inp)
            return fn
        else:  # eager / enforce_eager
            def fn(inp):
                with torch.no_grad():
                    return _encode_core(inp)
            return fn

    activities = [ProfilerActivity.CPU]
    xpu_activity = getattr(ProfilerActivity, "XPU", None)
    if xpu_activity is not None:
        activities.append(xpu_activity)
        device_sort = "self_xpu_time_total"
    else:
        device_sort = "self_cpu_time_total"
    log(f"[prof] activities={[str(a) for a in activities]}")

    def steady_wall_ms(fn, reps):
        torch.xpu.synchronize()
        t = time.perf_counter()
        for _ in range(reps):
            fn(vae_input)
        torch.xpu.synchronize()
        return (time.perf_counter() - t) / reps * 1000.0

    def run_pass(mode, pass_ix):
        tag = f"{mode}_pass{pass_ix}"
        fn = make_fn(mode)  # fresh compile per pass so each pass is self-contained
        # warmup: compile + allocator + JIT reach steady state (NOT profiled)
        for _ in range(args.warmup):
            fn(vae_input)
        torch.xpu.synchronize()
        out = fn(vae_input)
        torch.xpu.synchronize()
        finite = bool(torch.isfinite(out.float().cpu()).all())
        warm_ms = steady_wall_ms(fn, max(5, args.active))
        log(f"[{tag}] steady wall = {warm_ms:.1f} ms/encode  latent={tuple(out.shape)} finite={finite}")

        prof_schedule = schedule(wait=0, warmup=1, active=args.active, repeat=1)
        with profile(activities=activities, record_shapes=False,
                     profile_memory=False, with_stack=False,
                     schedule=prof_schedule) as prof:
            for _ in range(args.active + 1):
                fn(vae_input)
                torch.xpu.synchronize()
                prof.step()

        counts = count_events(prof, args.active)
        # dump the device-time table for the record
        table = prof.key_averages().table(sort_by=device_sort, row_limit=40)
        (outdir / f"keyavg_{tag}.txt").write_text(table)
        log(f"[{tag}] LAUNCHES/encode={counts['launches_per_encode']:.1f} "
            f"(urEnqueueKernelLaunch total={counts['launches_total']}, "
            f"ze={counts['ze_launch_total']}, memcpy/enc={counts['memcpy_per_encode']:.1f}) | "
            f"DEVICE_KERNELS/encode={counts['device_kernels_per_encode']:.1f} "
            f"(distinct={counts['distinct_device_kernel_names']})")
        return {
            "tag": tag, "mode": mode, "pass": pass_ix,
            "steady_wall_ms_per_encode": warm_ms,
            "finite": finite, "latent_shape": list(out.shape),
            **counts,
        }

    summary = {
        "device_name": dev_name,
        "torch_version": torch.__version__,
        "vae_source": "diffusers.AutoencoderKLWan (default Wan2.1; == vllm_omni encode path)",
        "vae_input_shape": list(vae_input.shape),
        "input_src": src,
        "num_frames": NUM_FRAMES,
        "warmup_reps": args.warmup,
        "active_reps": args.active,
        "passes": args.passes,
        "results": [],
    }

    for mode in ("eager", "inductor"):
        for p in range(1, args.passes + 1):
            summary["results"].append(run_pass(mode, p))

    peak = torch.xpu.max_memory_allocated() / 1024**3
    summary["peak_xpu_gib"] = peak
    log(f"[vae] PEAK_XPU_GIB={peak:.3f}")

    # concise headline table
    log("\n===== KERNELS LAUNCHED PER ENCODE =====")
    log(f"{'setting':<18}{'launches/enc':>14}{'dev_kernels/enc':>18}{'distinct':>10}{'wall_ms':>10}")
    for r in summary["results"]:
        log(f"{r['tag']:<18}{r['launches_per_encode']:>14.1f}"
            f"{r['device_kernels_per_encode']:>18.1f}{r['distinct_device_kernel_names']:>10}"
            f"{r['steady_wall_ms_per_encode']:>10.1f}")

    out_json = outdir / "vae_inductor_vs_eager_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    log(f"SAVED={out_json}")
    log("DONE")


if __name__ == "__main__":
    main()
