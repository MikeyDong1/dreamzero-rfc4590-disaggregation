#!/usr/bin/env python3
"""Isolated Wan-VAE encode benchmark using STOCK diffusers AutoencoderKLWan.

For TIMING, this is identical to vllm_omni's DistributedAutoencoderKLWan on the
non-distributed path: OmniAutoencoderKLWan subclasses diffusers AutoencoderKLWan
with NO config/_encode/forward override (encode() just wraps super().encode();
_encode is pure diffusers), and is_distributed_enabled() is False here (no
tiling executor). Default ctor config matches (z_dim=16, base_dim=96,
dim_mult=[1,2,4,4], temperal_downsample=[F,T,T], fp32) — same as the B60 run's
default-constructed VAE.

Reproduces the exact obs#1 VAE input:
  first stitched frame -> preprocess -> (1,3,1,352,640) bf16 in [-1,1];
  _encode_image builds concat([first_frame, zeros(1,3,32,352,640)]) = (1,3,33,352,640).
Because VAE encode wall-clock depends only on input shape + architecture (not
weight VALUES), random-init weights give accurate timing (model dl was wiped
from this shared node).

Runs COLD then N WARM reps, autocast-bf16 (faithful pipeline path) AND pure fp32.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLWan

DEVICE = "xpu"
NUM_FRAMES = 33  # ah_config["num_frames"]
H, W = 352, 640  # stitched OXE_DROID frame (2x2 of 176x320)


def log(m):
    print(m, flush=True)


def build_input(device, stitched_npz=None):
    """Recreate the (1,3,33,352,640) bf16 VAE input for obs#1.

    If a real stitched npz is given, use its first frame; else synthesize a frame
    in the same [-1,1] range (timing is value-independent). Rest = zero frames,
    exactly as pipeline._encode_image constructs vae_input.
    """
    if stitched_npz and Path(stitched_npz).exists():
        z = np.load(stitched_npz)
        stitched = z["images"]  # (1,352,640,3) uint8
        if stitched.ndim == 3:
            stitched = stitched[None]
        v = torch.from_numpy(stitched).unsqueeze(0).to(device)  # (1,1,352,640,3)? -> B,T,H,W,C
        v = v.permute(0, 4, 1, 2, 3).float() / 255.0            # B,C,T,H,W
        v = v.to(torch.bfloat16) * 2.0 - 1.0                    # [-1,1]
        first = v[:, :, :1]                                     # (1,3,1,H,W)
    else:
        # synthetic first frame in [-1,1]
        g = torch.Generator(device="cpu").manual_seed(0)
        first = (torch.rand(1, 3, 1, H, W, generator=g) * 2 - 1).to(device=device, dtype=torch.bfloat16)
    zeros = torch.zeros(1, 3, NUM_FRAMES - 1, first.shape[-2], first.shape[-1],
                        dtype=first.dtype, device=device)
    vae_input = torch.concat([first, zeros], dim=2)  # (1,3,33,H,W)
    return vae_input


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--stitched-npz", default=None)
    ap.add_argument("--out", default="/work/vae_only_results_b70.json")
    ap.add_argument("--device-name", default="")
    args = ap.parse_args()

    device = torch.device(DEVICE, 0)
    assert torch.xpu.device_count() >= 1, "no XPU visible"
    dev_name = args.device_name or torch.xpu.get_device_name(0)

    t0 = time.perf_counter()
    vae = AutoencoderKLWan()  # default Wan2.1 config; random init (fine for timing)
    vae.eval().to(device=device, dtype=torch.float32)
    latents_mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1).to(device)
    latents_inv_std = (1.0 / torch.tensor(vae.config.latents_std, dtype=torch.float32)).view(1, -1, 1, 1, 1).to(device)
    torch.xpu.synchronize()
    build_s = time.perf_counter() - t0
    log(f"[vae] device='{dev_name}' built VAE in {build_s:.2f}s")

    vae_input = build_input(device, args.stitched_npz)
    log(f"[vae] vae_input shape={tuple(vae_input.shape)} dtype={vae_input.dtype}")

    def encode_once(use_autocast):
        with torch.no_grad():
            if use_autocast:
                with torch.amp.autocast(dtype=torch.bfloat16, device_type="xpu"):
                    h = vae._encode(vae_input.to(dtype=vae.dtype))
                    mu, _ = h.chunk(2, dim=1)
                    mu = (mu - latents_mean) * latents_inv_std
            else:
                h = vae._encode(vae_input.to(dtype=vae.dtype))
                mu, _ = h.chunk(2, dim=1)
                mu = (mu - latents_mean) * latents_inv_std
        return mu

    def bench(use_autocast, tag):
        torch.xpu.synchronize()
        tc = time.perf_counter()
        out = encode_once(use_autocast)
        torch.xpu.synchronize()
        cold = time.perf_counter() - tc
        finite = bool(torch.isfinite(out.float().cpu()).all())
        log(f"[vae:{tag}] latent shape={tuple(out.shape)} dtype={out.dtype} finite={finite}  COLD_MS={cold*1000:.1f}")
        warm = []
        for _ in range(args.reps):
            torch.xpu.synchronize()
            tw = time.perf_counter()
            encode_once(use_autocast)
            torch.xpu.synchronize()
            warm.append(time.perf_counter() - tw)
        warm_ms = [w * 1000 for w in warm]
        mean_ms = sum(warm_ms) / len(warm_ms)
        log(f"[vae:{tag}] WARM_MS per-rep={[f'{m:.1f}' for m in warm_ms]}")
        log(f"[vae:{tag}] WARM_MEAN_MS={mean_ms:.1f}  min={min(warm_ms):.1f}  max={max(warm_ms):.1f}")
        return {"cold_ms": cold * 1000, "warm_ms": warm_ms, "warm_mean_ms": mean_ms,
                "latent_shape": list(out.shape), "finite": finite}

    autocast_res = bench(True, "autocast_bf16")
    fp32_res = bench(False, "pure_fp32")

    peak = torch.xpu.max_memory_allocated() / 1024**3
    log(f"[vae] PEAK_XPU_GIB={peak:.3f}")

    res = {
        "device_name": dev_name,
        "torch_version": torch.__version__,
        "vae_source": "diffusers.AutoencoderKLWan (default Wan2.1 config; == vllm_omni DistributedAutoencoderKLWan encode path)",
        "vae_input_shape": list(vae_input.shape),
        "num_frames": NUM_FRAMES,
        "build_s": build_s,
        "autocast_bf16": autocast_res,
        "pure_fp32": fp32_res,
        "peak_xpu_gib": peak,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    log(f"SAVED={args.out}")
    log("DONE")


if __name__ == "__main__":
    main()
