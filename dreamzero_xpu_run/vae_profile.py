#!/usr/bin/env python3
"""Profile the DreamZero Wan-VAE encode on ONE XPU card with torch.profiler.

Runs the EXACT VAE encode DreamZeroPipeline._encode_image performs for obs#1
(session reset, first frame): the saved stitched frame -> (1,3,1,352,640) bf16
-> concat with 32 zero frames -> (1,3,33,352,640) -> vae._encode -> chunk ->
normalize -> latent (1,16,9,44,80).  Faithful path = autocast(bf16).

On top of vae_only_bench.py this attaches torch.profiler with CPU+XPU activities,
record_shapes, profile_memory, with_stack, and:
  - warms up the kernels (JIT/allocator) first (profiler measures the WARM state),
  - profiles N steady-state reps,
  - exports a chrome trace (perfetto/chrome://tracing),
  - dumps key_averages tables sorted by device-time and host-time,
  - writes a JSON summary with the top ops (self device time, count, shapes)
    for offline analysis.

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

import vllm_omni  # noqa: F401  (triton-xpu disable side effect)

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
    DistributedAutoencoderKLWan,
)
from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline

from torch.profiler import profile, ProfilerActivity, schedule

DEVICE = "xpu"
NUM_FRAMES = 33  # ah_config["num_frames"]


def log(m):
    print(m, flush=True)


def _preprocess_video(videos: torch.Tensor) -> torch.Tensor:
    """uint8 [B,T,H,W,C] -> bf16 [B,C,T,H,W] normalized to [-1,1] (copy of pipeline)."""
    videos = videos.permute(0, 4, 1, 2, 3)
    if videos.dtype == torch.uint8:
        videos = videos.float() / 255.0
        videos = videos.to(dtype=torch.bfloat16)
        b, c, t, h, w = videos.shape
        videos = videos.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        videos = videos * 2.0 - 1.0
        videos = videos.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    return videos.to(dtype=torch.bfloat16)


def stream_vae_weights(model_path: str):
    from safetensors import safe_open
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    want = {k: s for k, s in weight_map.items() if k.startswith("action_head.vae.")}
    shard_to_keys: dict[str, list[str]] = {}
    for k, s in want.items():
        shard_to_keys.setdefault(s, []).append(k)
    for shard, keys in shard_to_keys.items():
        with safe_open(os.path.join(model_path, shard), framework="pt", device="cpu") as f:
            for name in keys:
                yield name, f.get_tensor(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--stitched-npz", required=True,
                    help="model_input_stitched.npz (images:(1,352,640,3) uint8)")
    ap.add_argument("--warmup", type=int, default=3, help="warm reps before profiling")
    ap.add_argument("--active", type=int, default=5, help="profiled reps")
    ap.add_argument("--outdir", default="/work/vae_profile")
    ap.add_argument("--device-name", default="")
    ap.add_argument("--fp32-too", action="store_true",
                    help="also profile pure fp32 (no autocast) for reference")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(DEVICE, 0)
    assert torch.xpu.device_count() >= 1, "no XPU visible"
    dev_name = args.device_name or torch.xpu.get_device_name(0)

    # ---- Build ONLY the VAE + fill real action_head.vae.* weights ----
    t0 = time.perf_counter()
    vae = DistributedAutoencoderKLWan()
    vae.eval()
    params = dict(vae.named_parameters())
    loaded = 0
    for name, tensor in stream_vae_weights(args.model_path):
        mapped = DreamZeroPipeline._remap_vae_key(name)
        if mapped is None:
            continue
        if mapped in params:
            params[mapped].data.copy_(tensor)
            loaded += 1
    vae.to(device=device, dtype=torch.float32)
    latents_mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1).to(device)
    latents_inv_std = (1.0 / torch.tensor(vae.config.latents_std, dtype=torch.float32)).view(1, -1, 1, 1, 1).to(device)
    torch.xpu.synchronize()
    build_s = time.perf_counter() - t0
    log(f"[vae] device='{dev_name}' torch={torch.__version__} built + filled {loaded} weights in {build_s:.2f}s")

    # ---- Reconstruct the EXACT VAE input for obs#1 ----
    z = np.load(args.stitched_npz)
    stitched = z["images"]  # (1,352,640,3) uint8
    if stitched.ndim == 3:
        stitched = stitched[None]
    videos = torch.from_numpy(stitched).unsqueeze(0).to(device)  # (B=1,T=1,H,W,C)
    videos = _preprocess_video(videos)  # (1,3,1,352,640) bf16
    _, _, _, height, width = videos.shape
    image = videos[:, :, :1].transpose(1, 2)  # (B,1,C,H,W)
    image_input = image.transpose(1, 2)       # (B,C,1,H,W)
    image_zeros = torch.zeros(1, 3, NUM_FRAMES - 1, height, width, dtype=image_input.dtype, device=device)
    vae_input = torch.concat([image_input, image_zeros], dim=2)  # (1,3,33,352,640)
    log(f"[vae] vae_input shape={tuple(vae_input.shape)} dtype={vae_input.dtype}")

    def encode_once(use_autocast: bool):
        with torch.no_grad():
            if use_autocast:
                with torch.amp.autocast(dtype=torch.bfloat16, device_type="xpu"):
                    hidden = vae._encode(vae_input.to(dtype=vae.dtype))
                    mu, _ = hidden.chunk(2, dim=1)
                    mu = (mu - latents_mean) * latents_inv_std
            else:
                hidden = vae._encode(vae_input.to(dtype=vae.dtype))
                mu, _ = hidden.chunk(2, dim=1)
                mu = (mu - latents_mean) * latents_inv_std
        return mu

    # Which activities are supported on this build?
    activities = [ProfilerActivity.CPU]
    xpu_activity = getattr(ProfilerActivity, "XPU", None)
    if xpu_activity is not None:
        activities.append(xpu_activity)
        device_sort = "self_xpu_time_total"
        device_col = "xpu_time_total"
    else:
        device_sort = "self_cpu_time_total"
        device_col = "cpu_time_total"
    log(f"[prof] activities={[str(a) for a in activities]} device_sort={device_sort}")

    def timed_wallclock(use_autocast, reps):
        torch.xpu.synchronize()
        t = time.perf_counter()
        for _ in range(reps):
            encode_once(use_autocast)
        torch.xpu.synchronize()
        return (time.perf_counter() - t) / reps * 1000.0

    def run_profiled(use_autocast, tag):
        # --- warmup (kernels + allocator reach steady state) ---
        for _ in range(args.warmup):
            encode_once(use_autocast)
        torch.xpu.synchronize()
        warm_ms = timed_wallclock(use_autocast, max(3, args.active))
        log(f"[{tag}] steady-state wall-clock per-encode = {warm_ms:.1f} ms")

        # --- profiled window ---
        prof_schedule = schedule(wait=0, warmup=1, active=args.active, repeat=1)
        trace_path = outdir / f"vae_trace_{tag}.json"
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,   # stacks explode trace size; op names + shapes suffice
            schedule=prof_schedule,
        ) as prof:
            for _ in range(args.active + 1):  # +1 for the schedule warmup step
                encode_once(use_autocast)
                torch.xpu.synchronize()
                prof.step()

        prof.export_chrome_trace(str(trace_path))
        log(f"[{tag}] chrome trace -> {trace_path}")

        ka = prof.key_averages(group_by_input_shape=False)
        table_device = ka.table(sort_by=device_sort, row_limit=40)
        table_host = ka.table(sort_by="self_cpu_time_total", row_limit=40)
        (outdir / f"vae_keyavg_{tag}_by_device.txt").write_text(table_device)
        (outdir / f"vae_keyavg_{tag}_by_host.txt").write_text(table_host)
        log(f"[{tag}] ===== key_averages sorted by {device_sort} (top 40) =====")
        log(table_device)

        # also group by input shape to see which conv shapes dominate
        ka_shapes = prof.key_averages(group_by_input_shape=True)
        table_shapes = ka_shapes.table(sort_by=device_sort, row_limit=40)
        (outdir / f"vae_keyavg_{tag}_by_shape.txt").write_text(table_shapes)

        # --- machine-readable summary of top ops ---
        def evt_row(e):
            dev_self = getattr(e, "self_xpu_time_total", 0.0) or 0.0
            dev_total = getattr(e, "xpu_time_total", 0.0) or 0.0
            if xpu_activity is None:
                dev_self = e.self_cpu_time_total
                dev_total = e.cpu_time_total
            return {
                "name": e.key,
                "count": int(e.count),
                "self_device_us": float(dev_self),
                "device_total_us": float(dev_total),
                "self_host_us": float(e.self_cpu_time_total),
                "host_total_us": float(e.cpu_time_total),
            }

        rows = [evt_row(e) for e in ka]
        rows_by_dev = sorted(rows, key=lambda r: r["self_device_us"], reverse=True)
        rows_by_host = sorted(rows, key=lambda r: r["self_host_us"], reverse=True)
        total_self_dev = sum(r["self_device_us"] for r in rows)
        total_self_host = sum(r["self_host_us"] for r in rows)

        return {
            "tag": tag,
            "steady_state_wall_ms_per_encode": warm_ms,
            "profiled_reps": args.active,
            "total_self_device_us": total_self_dev,
            "total_self_host_us": total_self_host,
            "device_time_measured": xpu_activity is not None,
            "top_by_device": rows_by_dev[:25],
            "top_by_host": rows_by_host[:25],
            "trace": str(trace_path.name),
        }

    summary = {
        "device_name": dev_name,
        "torch_version": torch.__version__,
        "vae_input_shape": list(vae_input.shape),
        "latent_shape": [1, 16, 9, 44, 80],
        "num_frames": NUM_FRAMES,
        "weights_loaded": loaded,
        "warmup_reps": args.warmup,
        "active_reps": args.active,
    }

    summary["autocast_bf16"] = run_profiled(True, "autocast_bf16")
    if args.fp32_too:
        summary["pure_fp32"] = run_profiled(False, "pure_fp32")

    peak = torch.xpu.max_memory_allocated() / 1024**3
    summary["peak_xpu_gib"] = peak
    log(f"[vae] PEAK_XPU_GIB={peak:.3f}")

    out_json = outdir / "vae_profile_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    log(f"SAVED={out_json}")
    log("DONE")


if __name__ == "__main__":
    main()
