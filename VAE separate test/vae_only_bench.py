#!/usr/bin/env python3
"""Isolated Wan-VAE encode benchmark on ONE XPU card (no UMT5, no DiT, no TP).

Reproduces EXACTLY the VAE encode that DreamZeroPipeline._encode_image does for
observation #1 (session reset, first frame):

  stitched first frame (352x640 uint8) -> _preprocess_video -> (1,3,1,352,640) bf16
    -> _encode_image builds vae_input = concat([first_frame,
         zeros(1,3,num_frames-1,352,640)], dim=2)  =>  (1,3,33,352,640) fp32
    -> _encode_vae_latents(vae_input):  mu = vae._encode(x).chunk(2)[0];
       normalized by (mu-mean)*inv_std  ->  latent (1,16,9,44,80)

Only the VAE is built + filled (action_head.vae.* weights, ~homework of load_weights'
_remap_vae_key). Runs COLD (1st call, pays XPU kernel JIT) then N WARM reps and
reports per-call ms. Single process, ZE_AFFINITY_MASK pins one card.
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
    # only shards holding action_head.vae.* keys
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
                    help="model_input_stitched.npz from the preprocessing run (images:(1,352,640,3) uint8)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="/mnt/data/vllm-omni/outputs/dreamzero_preproc_tp8_timed/vae_only_results.json")
    args = ap.parse_args()

    device = torch.device(DEVICE, 0)

    # ---- Build ONLY the VAE (default ctor, matches pipeline when no vae/ dir) ----
    t0 = time.perf_counter()
    vae = DistributedAutoencoderKLWan()
    # NOTE: intentionally NOT calling vae.init_distributed() — it needs the DIT
    # process group (single-card, no TP here). The distributed tiling executor is
    # only used when is_distributed_enabled() (use_tiling=True AND parallel_size>1),
    # which is never the case in this pipeline path, so _encode runs plain/replicated
    # exactly as it does per-rank in the real run.
    vae.eval()

    # Fill action_head.vae.* -> vae.* using the pipeline's own remap.
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

    # VAE normalization buffers (same as pipeline registers)
    latents_mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1).to(device)
    latents_inv_std = (1.0 / torch.tensor(vae.config.latents_std, dtype=torch.float32)).view(1, -1, 1, 1, 1).to(device)
    torch.xpu.synchronize()
    build_s = time.perf_counter() - t0
    log(f"[vae] built + filled {loaded} weights in {build_s:.2f}s")

    # ---- Reconstruct the EXACT VAE input for obs#1 ----
    z = np.load(args.stitched_npz)
    stitched = z["images"]  # (1,352,640,3) uint8 (T=1 first frame)
    if stitched.ndim == 3:
        stitched = stitched[None]
    videos = torch.from_numpy(stitched).unsqueeze(0).to(device)  # (B=1,T=1,H,W,C)
    videos = _preprocess_video(videos)  # (B,C,T,H,W) bf16 = (1,3,1,352,640)
    _, _, _, height, width = videos.shape
    # Mirror pipeline._encode_image: image = videos[:,:,:1].transpose(1,2) -> (B,1,C,H,W),
    # then image_input = image.transpose(1,2) -> (B,C,1,H,W).
    image = videos[:, :, :1].transpose(1, 2)  # (B,1,C,H,W)
    image_input = image.transpose(1, 2)       # (B,C,1,H,W)  == pipeline's image_input
    image_zeros = torch.zeros(1, 3, NUM_FRAMES - 1, height, width, dtype=image_input.dtype, device=device)
    vae_input = torch.concat([image_input, image_zeros], dim=2)  # (1,3,33,352,640)
    log(f"[vae] vae_input shape={tuple(vae_input.shape)} dtype={vae_input.dtype}")

    def encode_once(use_autocast: bool):
        with torch.no_grad():
            if use_autocast:
                # EXACTLY matches pipeline._encode_image, which wraps the VAE
                # encode in autocast(bf16) -> convs run bf16 despite fp32 weights.
                with torch.amp.autocast(dtype=torch.bfloat16, device_type="xpu"):
                    hidden = vae._encode(vae_input.to(dtype=vae.dtype))
                    mu, _ = hidden.chunk(2, dim=1)
                    mu = (mu - latents_mean) * latents_inv_std
            else:
                hidden = vae._encode(vae_input.to(dtype=vae.dtype))
                mu, _ = hidden.chunk(2, dim=1)
                mu = (mu - latents_mean) * latents_inv_std
        return mu

    def bench(use_autocast: bool, tag: str):
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

    # (1) autocast bf16 == faithful pipeline path; (2) pure fp32 reference.
    autocast_res = bench(True, "autocast_bf16")
    fp32_res = bench(False, "pure_fp32")

    peak = torch.xpu.max_memory_allocated() / 1024**3
    log(f"[vae] PEAK_XPU_GIB={peak:.3f}")

    res = {
        "vae_input_shape": list(vae_input.shape),
        "num_frames": NUM_FRAMES,
        "weights_loaded": loaded,
        "build_s": build_s,
        "autocast_bf16": autocast_res,   # matches pipeline _encode_image
        "pure_fp32": fp32_res,           # reference (no autocast)
        "peak_xpu_gib": peak,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    log(f"SAVED={args.out}")
    log("DONE")


if __name__ == "__main__":
    main()
