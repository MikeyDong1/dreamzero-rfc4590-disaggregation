#!/usr/bin/env python3
"""Standalone REAL DreamZeroPipeline on ONE XPU card with K-resident layerwise offload.

Unlike `standalone_dreamzero_layerwise.py` (which drives the bare CausalWanModel with
SYNTHETIC conditioning and emits raw noise-prediction latents), this harness runs the
**real** `DreamZeroPipeline.forward()` end-to-end on the repo's real camera input:

  real camera mp4s (YangshenDeng/vllm-omni-dreamzero-assets)
    -> DROID transform (stitch 3 views -> 352x640, templated prompt, state)
      -> UMT5-xxl text encode + CLIP image encode + Wan VAE encode
        -> CausalWanModel DiT denoise loop (num_inference_steps=16, CFG)
          -> Wan VAE decode -> RGB frames -> mp4/gif (ASSESSABLE video)

It bypasses ONLY the Omni serving stack (no Omni / AsyncOmniEngine / DiffusionWorker /
orchestrator) -- we construct DreamZeroPipeline directly and call forward(req).

Two things differ from the production LayerWiseOffloadBackend (which has no K knob and
keeps ALL encoders resident -> would OOM a 24.5GB card with the 11.4GB UMT5 text encoder):
  1. K-RESIDENT layerwise offload on transformer.blocks (K resident, rest sliding-window),
     ported from standalone_dreamzero_layerwise.enable_layerwise_offload.
  2. MEMORY STAGING: encoders (text/image/VAE) live on XPU only while encoding, and are
     evicted to CPU before the DiT denoise loop so K resident DiT blocks + activations fit.

Reports MODEL_LOAD_S, TIME_TO_FIRST_OUTPUT_S (obs#1 done), TIME_TO_OUTPUT_FINISHED_S
(obs#2 done), DECODE_S, and peak XPU memory. Saves mp4 + gif + actions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch

# Import vllm_omni FIRST: its package init applies the vLLM patch that disables
# the broken triton XPU import (triton.tools.disasm.get_spvdis). Importing
# `vllm.config` before this triggers that ImportError.
import vllm_omni  # noqa: F401  (side-effect: triton disable before vllm import)

# ---- vLLM-Omni bootstrap imports -----------------------------------------
from vllm.config import CompilationConfig, DeviceConfig, VllmConfig

from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.offloader.layerwise_backend import apply_block_hook
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

DEVICE = "xpu"

# Same fixed inputs the repo example uses (export_prediction_video.py).
RELATIVE_OFFSETS = [-23, -16, -8, 0]
ACTION_HORIZON = 24
CAMERA_FILES = {
    "observation/exterior_image_0_left": "exterior_image_1_left.mp4",
    "observation/exterior_image_1_left": "exterior_image_2_left.mp4",
    "observation/wrist_image_left": "wrist_image_left.mp4",
}
DEFAULT_PROMPT = (
    "Move the pan forward and use the brush in the middle of the plates "
    "to brush the inside of the pan"
)


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Input building (mirrors examples/offline_inference/dreamzero/export_prediction_video.py)
# ---------------------------------------------------------------------------
def _load_all_frames(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames loaded from {video_path}")
    return np.stack(frames, axis=0)


def _load_camera_frames(video_dir: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for cam_key, fname in CAMERA_FILES.items():
        p = video_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"Missing camera asset: {p}")
        out[cam_key] = _load_all_frames(p)
    return out


def _build_frame_schedule(total_frames: int, num_chunks: int) -> list[list[int]]:
    chunks: list[list[int]] = []
    current_frame = 23
    for _ in range(num_chunks):
        indices = [max(current_frame + off, 0) for off in RELATIVE_OFFSETS]
        if indices[-1] >= total_frames:
            break
        chunks.append(indices)
        current_frame += ACTION_HORIZON
    return chunks


def _make_obs(camera_frames, frame_indices, *, prompt, session_id) -> dict:
    obs: dict = {}
    for cam_key, all_frames in camera_frames.items():
        selected = all_frames[frame_indices]
        obs[cam_key] = selected[0] if len(frame_indices) == 1 else selected
    obs["observation/joint_position"] = np.zeros(7, dtype=np.float32)
    obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
    obs["observation/gripper_position"] = np.zeros(1, dtype=np.float32)
    obs["prompt"] = prompt
    obs["session_id"] = session_id
    # DROID embodiment (1-indexed exterior cameras) — matches the DROID checkpoint.
    obs["embodiment"] = "droid"
    return obs


def build_observations(video_dir: Path, prompt: str, session_id: str):
    camera_frames = _load_camera_frames(video_dir)
    total = min(f.shape[0] for f in camera_frames.values())
    chunks = _build_frame_schedule(total, 1)
    observations = [_make_obs(camera_frames, [0], prompt=prompt, session_id=session_id)]
    if chunks:
        observations.append(_make_obs(camera_frames, chunks[0], prompt=prompt, session_id=session_id))
    if len(observations) < 2:
        raise RuntimeError("Need >=2 observations to export a prediction video.")
    return camera_frames, observations[:2]


# ---------------------------------------------------------------------------
# Weight streaming from the 10 root safetensors
# ---------------------------------------------------------------------------
def stream_root_weights(model_path: str):
    from safetensors import safe_open

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    shard_to_keys: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        shard_to_keys.setdefault(shard, []).append(key)
    for shard, keys in shard_to_keys.items():
        with safe_open(os.path.join(model_path, shard), framework="pt", device="cpu") as f:
            for name in keys:
                yield name, f.get_tensor(name)


# ---------------------------------------------------------------------------
# K-resident layerwise offload on the DiT blocks (ported from standalone harness)
# ---------------------------------------------------------------------------
def _block_bytes(block: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in block.parameters()) + sum(
        b.numel() * b.element_size() for b in block.buffers()
    )


def enable_k_resident_offload(dit: torch.nn.Module, device: torch.device,
                              resident_blocks: int, pin_memory: bool = True):
    """Keep the first K DiT blocks permanently resident on device; sliding-window
    offload the remaining (num_blocks - K). Non-block submodules + top-level params
    are moved resident. Mirrors LayerWiseOffloadBackend but with a K knob."""
    blocks = list(dit.blocks)
    num_blocks = len(blocks)

    # Non-block children resident (img_emb, patch_embedding, head, time_embedding, ...).
    for name, child in dit.named_children():
        if name == "blocks":
            continue
        child.to(device)
    for p in dit._parameters.values():
        if p is not None:
            p.data = p.data.to(device)
    for b in dit._buffers.values():
        if b is not None:
            b.data = b.data.to(device)

    K = max(0, min(resident_blocks, num_blocks - 2))  # keep >=2 offloaded for a valid window
    blk_mib = _block_bytes(blocks[0]) / 1024**2
    log(f"[offload] per-block ~{blk_mib:.0f} MiB; K={K}/{num_blocks} resident, "
        f"{num_blocks - K} sliding-window offloaded.")

    for blk in blocks[:K]:
        blk.to(device)

    off_blocks = blocks[K:]
    n_off = len(off_blocks)
    copy_stream = current_omni_platform.Stream()
    last_block, first_off = off_blocks[-1], off_blocks[0]
    last_hook = apply_block_hook(last_block, first_off, device, copy_stream, pin_memory)
    last_hook.prefetch_layer(non_blocking=False)
    hooks = [last_hook]
    for i, block in enumerate(off_blocks[:-1]):
        nxt = off_blocks[(i + 1) % n_off]
        hooks.append(apply_block_hook(block, nxt, device, copy_stream, pin_memory))
    for i in range(len(hooks)):
        hooks[i]._prev_hook = hooks[i - 1]
    return hooks


def _move_module(m: torch.nn.Module, device, dtype=None):
    if m is None:
        return
    if dtype is not None:
        m.to(device=device, dtype=dtype)
    else:
        m.to(device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def bootstrap(device, od_config) -> VllmConfig:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29556")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    current_omni_platform.set_device(device)
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(),
        device_config=DeviceConfig(device=device),
    )
    vllm_config.parallel_config.tensor_parallel_size = 1
    vllm_config.parallel_config.data_parallel_size = 1
    init_distributed_environment(world_size=1, rank=0, local_rank=0)
    initialize_model_parallel(
        data_parallel_size=1, cfg_parallel_size=1, sequence_parallel_size=1,
        ulysses_degree=1, ring_degree=1, tensor_parallel_size=1, pipeline_parallel_size=1,
    )
    log("[bootstrap] distributed + model-parallel initialized (world=1, tp=1)")
    return vllm_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out-dir", default="/workspace/out/dreamzero_real_k")
    ap.add_argument("--resident-blocks", type=int, default=10, help="K resident DiT blocks")
    ap.add_argument("--num-inference-steps", type=int, default=16)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--fps", type=int, default=5)
    args = ap.parse_args()

    device = torch.device(DEVICE, 0)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    od_config = OmniDiffusionConfig(
        model=args.model_path,
        dtype=torch.bfloat16,
        enforce_eager=True,
        # Tell the pipeline NOT to eagerly move VAE to device (we stage manually),
        # and set the denoise step count (forward() reads model_config).
        enable_layerwise_offload=True,
        model_config={"num_inference_steps": args.num_inference_steps},
    )
    vllm_config = bootstrap(device, od_config)

    results: dict = {}
    with set_forward_context(vllm_config=vllm_config, omni_diffusion_config=od_config), \
         set_current_diffusion_config(od_config):
        # ---- Build pipeline (all on CPU; offload flag keeps VAE off-device) ----
        t_load0 = time.perf_counter()
        log("[load] constructing DreamZeroPipeline (CPU) ...")
        pipeline = DreamZeroPipeline(od_config=od_config)
        pipeline.eval()

        log("[load] streaming weights from root safetensors ...")
        loaded = pipeline.load_weights(stream_root_weights(args.model_path))
        log(f"[load] loaded {len(loaded)} weights into pipeline")

        # ---- K-resident layerwise offload on the DiT ----
        enable_k_resident_offload(pipeline.transformer, device,
                                  resident_blocks=args.resident_blocks)

        # ---- MEMORY STAGING (so K=10 fits a 24.5GB card) -----------------
        # The UMT5-xxl text encoder is 11.4GB bf16 and CLIP is 1.3GB. They are
        # only used in forward()'s ENCODE phase (_encode_text / _encode_image),
        # before the DiT prefill+denoise. If left resident alongside K=10 DiT
        # blocks (~8.1GB) + denoise activations, peak would hit ~28GB → OOM.
        # So: VAE stays resident (0.5GB, used for encode AND final decode), but
        # the text + image encoders live on CPU and are moved to device only
        # during their encode call, then evicted to CPU at _prefill_kv_cache
        # entry (the first DiT step). Result: encode-peak ~21GB, denoise-peak
        # ~15GB — both under 24.5GB.
        _move_module(pipeline.vae, device, dtype=torch.float32)
        pipeline.text_encoder.to("cpu")
        pipeline.image_encoder.to("cpu")
        # vae normalization buffers follow .to() — push pipeline buffers to device.
        for _b_name, b in list(pipeline.named_buffers()):
            try:
                b.data = b.data.to(device)
            except Exception:
                pass

        # --- wrap encode methods to stage the big encoders on-demand ---
        _orig_encode_text = pipeline._encode_text
        _orig_encode_image = pipeline._encode_image
        _orig_prefill = pipeline._prefill_kv_cache

        def _staged_encode_text(text_tokens, attention_mask):
            if next(pipeline.text_encoder.parameters()).device.type != device.type:
                pipeline.text_encoder.to(device)
            return _orig_encode_text(text_tokens, attention_mask)

        def _staged_encode_image(image, num_frames, height, width):
            if next(pipeline.image_encoder.parameters()).device.type != device.type:
                pipeline.image_encoder.to(device)
            return _orig_encode_image(image, num_frames, height, width)

        def _staged_prefill(*a, **kw):
            # Encoders are done for this forward(): evict to CPU before the DiT
            # prefill/denoise so K resident blocks + activations fit.
            pipeline.text_encoder.to("cpu")
            pipeline.image_encoder.to("cpu")
            current_omni_platform.empty_cache()
            return _orig_prefill(*a, **kw)

        pipeline._encode_text = _staged_encode_text
        pipeline._encode_image = _staged_encode_image
        pipeline._prefill_kv_cache = _staged_prefill

        current_omni_platform.synchronize()
        model_load_s = time.perf_counter() - t_load0
        log(f"[load] MODEL_LOAD_S={model_load_s:.3f}")

        try:
            current_omni_platform.reset_peak_memory_stats()
        except Exception:
            pass

        # ---- Build the two real observations ----
        session_id = f"dreamzero-standalone-{uuid.uuid4()}"
        camera_frames, observations = build_observations(Path(args.video_dir), args.prompt, session_id)
        log(f"[input] built {len(observations)} observations from real camera mp4s "
            f"(stitched frame shape from view set)")

        # ---- Run forward() per observation (encode -> prefill -> denoise) ----
        outputs = []
        per_obs_s = []
        t_first = None
        t_runs0 = time.perf_counter()
        for idx, obs in enumerate(observations):
            sp = OmniDiffusionSamplingParams(
                num_inference_steps=args.num_inference_steps,
                extra_args={"reset": idx == 0, "session_id": obs["session_id"], "robot_obs": obs},
            )
            req = OmniDiffusionRequest(
                prompts=[obs["prompt"]],
                sampling_params=sp,
                request_id=f"req-{idx}",
            )
            t0 = time.perf_counter()
            out = pipeline.forward(req)
            current_omni_platform.synchronize()
            dt = time.perf_counter() - t0
            per_obs_s.append(dt)
            outputs.append(out)
            if idx == 0:
                t_first = time.perf_counter() - t_runs0
                log(f"[run] TIME_TO_FIRST_OUTPUT_S={t_first:.3f}")
            log(f"[run] observation {idx+1}/{len(observations)} done in {dt:.3f}s")
        t_finished = time.perf_counter() - t_runs0
        log(f"[run] TIME_TO_OUTPUT_FINISHED_S={t_finished:.3f}")

        # ---- Decode latents -> RGB frames -> mp4/gif ----
        def extract_latents(o):
            v = o.output["video"]  # (B, T, C, H, W) latent
            v = v.detach().cpu()
            if v.dim() == 4:
                v = v.unsqueeze(0)
            if v.shape[1] < v.shape[2]:  # ensure (B,C,T,H,W) for decode
                v = v.transpose(1, 2).contiguous()
            return v

        latent_steps = [extract_latents(o) for o in outputs]
        full_latents = torch.cat(latent_steps, dim=2)  # concat along time
        log(f"[decode] full latent shape {tuple(full_latents.shape)}; decoding ...")
        t_dec0 = time.perf_counter()
        with torch.no_grad():
            decoded = pipeline.decode_video_latents(full_latents.to(device))
        decoded = decoded.squeeze(0).permute(1, 2, 3, 0).contiguous()  # (T,H,W,C)
        decoded = decoded.clamp(-1, 1) * 0.5 + 0.5
        frames = (decoded * 255.0).round().to(torch.uint8).cpu().numpy()
        current_omni_platform.synchronize()
        decode_s = time.perf_counter() - t_dec0
        log(f"[decode] DECODE_S={decode_s:.3f}; frames {frames.shape}")

        # ---- Write mp4 + gif ----
        mp4_path = out_dir / "dreamzero_prediction.mp4"
        h, w = frames.shape[1:3]
        writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 float(args.fps), (w, h))
        for fr in frames:
            writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        writer.release()
        log(f"SAVED_MP4={mp4_path}")

        try:
            from PIL import Image
            gif_path = out_dir / "dreamzero_prediction.gif"
            imgs = [Image.fromarray(fr) for fr in frames]
            imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                         duration=max(int(round(1000 / max(args.fps, 1))), 1), loop=0)
            log(f"SAVED_GIF={gif_path}")
        except Exception as exc:
            log(f"[gif] skipped: {exc}")

        # ---- Actions + finiteness ----
        actions = [np.asarray(o.output.get("actions")) for o in outputs]
        np.savez(out_dir / "actions.npz", step0=actions[0], step1=actions[1])
        video_finite = bool(np.isfinite(frames).all())
        actions_finite = bool(all(np.isfinite(a).all() for a in actions))

        peak = current_omni_platform.max_memory_allocated() / 1024**3
        results = {
            "model_load_s": model_load_s,
            "time_to_first_output_s": t_first,
            "time_to_output_finished_s": t_finished,
            "per_observation_s": per_obs_s,
            "decode_s": decode_s,
            "num_inference_steps": args.num_inference_steps,
            "resident_blocks": args.resident_blocks,
            "num_dit_blocks": len(pipeline.transformer.blocks),
            "frames_shape": list(frames.shape),
            "video_finite": video_finite,
            "actions_finite": actions_finite,
            "action_shapes": [list(a.shape) for a in actions],
            "video_pixel_min": float(frames.min()),
            "video_pixel_max": float(frames.max()),
            "video_pixel_mean": float(frames.mean()),
            "peak_xpu_gib": peak,
        }

    log("=========== RESULTS ===========")
    for k, v in results.items():
        log(f"{k} = {v}")
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    log("DONE")


if __name__ == "__main__":
    main()
