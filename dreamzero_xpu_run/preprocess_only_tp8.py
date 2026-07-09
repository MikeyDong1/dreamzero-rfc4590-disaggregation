#!/usr/bin/env python3
"""DreamZero PREPROCESSING-ONLY harness (TP=8) — encode stages, NO DiT.

Runs the *real* `DreamZeroPipeline` preprocessing path on the real 3-camera mp4
inputs under a genuine 8-rank Tensor-Parallel world (torchrun --nproc_per_node=8,
matching the `dreamzero_tp8.yaml` serving deployment), and STOPS exactly at the
DiT boundary:

  real camera mp4s -> DROID/RoboArena transform (stitch 3 views, template prompt,
    state) -> tokenize -> UMT5-xxl text encode (+ negative for CFG)
      -> CLIP image encode + Wan VAE encode  ============= DiT INPUTS READY =====
        -> [ _prefill_kv_cache -> diffuse ]   <-- DiT.  NOT RUN (intercepted).

Why this faithfully reproduces the serving preprocessing at TP=8:
  * `load_weights` shards ONLY the DiT (CausalWanModel) across the 8 ranks
    (QKV/Column/RowParallelLinear, DistributedRMSNorm). The UMT5 text encoder,
    CLIP image encoder and Wan VAE are REPLICATED on every rank (plain copy_),
    so the text/video embeddings that feed the DiT are TP-invariant and are
    produced with no cross-rank collective. We still init the full 8-rank world
    + TP group so the pipeline builds exactly as it does in serving.
  * The DiT boundary is the first `self.transformer(...)` call, which lives
    inside `_prefill_kv_cache`. We monkeypatch `_prefill_kv_cache` (capture its
    args, then return without touching the DiT) and `diffuse` (capture ALL its
    inputs + state.clip_feas/state.ys, then raise a sentinel) so ZERO DiT
    forward runs.

Rank 0 saves: original inputs, stitched/templated model inputs, text encoder
outputs (prompt + negative embeds), VAE/image latents (image, state.ys,
state.clip_feas), the noise inits, and the complete DiT-input bundle. It reports
PREPROCESS_S measured from the start of encode to DiT-inputs-ready.
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

# Import vllm_omni FIRST (side effect: disables broken triton-xpu import before
# vllm.config is imported).
import vllm_omni  # noqa: F401

from vllm.config import CompilationConfig, DeviceConfig, VllmConfig

from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
# TP rank/size getters live in vllm.distributed (vllm_omni's initialize_model_parallel
# populates vllm's _TP group; the DiT shards off these).
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

DEVICE = "xpu"

# Same fixed inputs the repo example (export_prediction_video.py) uses.
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
    rank = int(os.environ.get("RANK", "0"))
    print(f"[rank{rank}] {msg}", flush=True)


class _StopBeforeDiT(Exception):
    """Sentinel raised right after DiT inputs are captured, before any DiT run."""


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


def _make_obs(camera_frames, frame_indices, *, prompt, session_id, embodiment) -> dict:
    obs: dict = {}
    for cam_key, all_frames in camera_frames.items():
        selected = all_frames[frame_indices]
        obs[cam_key] = selected[0] if len(frame_indices) == 1 else selected
    obs["observation/joint_position"] = np.zeros(7, dtype=np.float32)
    obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
    obs["observation/gripper_position"] = np.zeros(1, dtype=np.float32)
    obs["prompt"] = prompt
    obs["session_id"] = session_id
    obs["embodiment"] = embodiment
    return obs


def build_first_observation(video_dir: Path, prompt: str, session_id: str, embodiment: str):
    """Build ONLY observation #1 (reset, frame [0], current_start_frame==0).

    This is the observation that does the full-from-scratch CLIP+VAE first-frame
    encode + text encode -> exactly the set of DiT inputs the DiT prefill would
    consume. Observation #2 is a DiT-state continuation (needs the DiT to have
    run on obs#1) so it is intentionally out of scope for a preprocessing-only,
    no-DiT run.
    """
    camera_frames = _load_camera_frames(video_dir)
    obs = _make_obs(camera_frames, [0], prompt=prompt, session_id=session_id, embodiment=embodiment)
    return camera_frames, obs


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
# Distributed bootstrap (real 8-rank TP world)
# ---------------------------------------------------------------------------
def bootstrap(device, tp_size: int) -> VllmConfig:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    current_omni_platform.set_device(current_omni_platform.get_torch_device(local_rank))

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(),
        device_config=DeviceConfig(device=DEVICE),
    )
    vllm_config.parallel_config.tensor_parallel_size = tp_size
    vllm_config.parallel_config.data_parallel_size = 1

    init_distributed_environment(world_size=world_size, rank=rank, local_rank=local_rank)
    initialize_model_parallel(
        data_parallel_size=1, cfg_parallel_size=1, sequence_parallel_size=1,
        ulysses_degree=1, ring_degree=1, tensor_parallel_size=tp_size, pipeline_parallel_size=1,
    )
    log(f"[bootstrap] world={world_size} rank={rank} local_rank={local_rank} "
        f"tp={get_tensor_model_parallel_world_size()} tp_rank={get_tensor_model_parallel_rank()}")
    return vllm_config


def _t(x):
    """CPU-detach a tensor for saving; pass through non-tensors."""
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu")
    return x


def _describe(name, x):
    if isinstance(x, torch.Tensor):
        return f"{name}: shape={tuple(x.shape)} dtype={x.dtype} device={x.device}"
    return f"{name}: {type(x).__name__}={x!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out-dir", default="/mnt/data/vllm-omni/outputs/dreamzero_preproc_tp8")
    ap.add_argument("--tp-size", type=int, default=8)
    ap.add_argument("--num-inference-steps", type=int, default=16)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--embodiment", default="roboarena")
    args = ap.parse_args()

    device = current_omni_platform.get_torch_device(int(os.environ.get("LOCAL_RANK", "0")))
    out_dir = Path(args.out_dir)
    is_rank0 = int(os.environ.get("RANK", "0")) == 0
    if is_rank0:
        out_dir.mkdir(parents=True, exist_ok=True)

    od_config = OmniDiffusionConfig(
        model=args.model_path,
        dtype=torch.bfloat16,
        enforce_eager=True,
        model_config={"num_inference_steps": args.num_inference_steps},
    )
    vllm_config = bootstrap(device, args.tp_size)

    captured: dict = {}
    results: dict = {}

    with set_forward_context(vllm_config=vllm_config, omni_diffusion_config=od_config), \
         set_current_diffusion_config(od_config):
        # ---- Build pipeline (CPU) + load weights (DiT sharded 8-way per rank) ----
        t_load0 = time.perf_counter()
        log("[load] constructing DreamZeroPipeline ...")
        pipeline = DreamZeroPipeline(od_config=od_config)
        pipeline.eval()
        log("[load] streaming weights from root safetensors ...")
        loaded = pipeline.load_weights(stream_root_weights(args.model_path))
        log(f"[load] loaded {len(loaded)} weights")

        # dtype: model built at fp32-default; upcast checkpoint -> must recast to
        # bf16 (checkpoint native) so UMT5 (11.4GB bf16 vs 22.8GB fp32) fits.
        # VAE stays fp32 (its design dtype).
        pipeline.transformer.to(device=device, dtype=torch.bfloat16)
        pipeline.text_encoder.to(device=device, dtype=torch.bfloat16)
        pipeline.image_encoder.to(device=device, dtype=torch.bfloat16)
        pipeline.vae.to(device=device, dtype=torch.float32)
        for _b_name, b in list(pipeline.named_buffers()):
            try:
                b.data = b.data.to(device)
            except Exception:
                pass

        current_omni_platform.synchronize()
        model_load_s = time.perf_counter() - t_load0
        log(f"[load] MODEL_LOAD_S={model_load_s:.3f}")
        try:
            current_omni_platform.reset_peak_memory_stats()
        except Exception:
            pass

        # ---- Intercept the DiT boundary --------------------------------------
        # forward() calls _prefill_kv_cache(image, prompt_embeds, neg_embeds,
        # frame_seqlen, seq_len, do_true_cfg, state) as the FIRST DiT touch, then
        # diffuse(...). We capture at _prefill (all the prefill-time DiT inputs)
        # AND at diffuse (the denoise-loop DiT inputs: noise latents, embeds,
        # state_features, embodiment_id, seq_len) and then abort — no DiT runs.
        def _capture_prefill(image_latents, prompt_embeds, negative_prompt_embeds,
                             frame_seqlen, seq_len, do_true_cfg, state):
            captured["prefill"] = dict(
                image_latents=_t(image_latents),
                prompt_embeds=_t(prompt_embeds),
                negative_prompt_embeds=_t(negative_prompt_embeds),
                frame_seqlen=int(frame_seqlen),
                seq_len=int(seq_len),
                do_true_cfg=bool(do_true_cfg),
            )
            captured["state_clip_feas"] = _t(getattr(state, "clip_feas", None))
            captured["state_ys"] = _t(getattr(state, "ys", None))
            captured["current_start_frame"] = int(state.current_start_frame)
            # Return without running the DiT KV-cache prefill.
            return None

        def _capture_diffuse(video_latents, action_latents, timesteps_video,
                             timesteps_action, prompt_embeds, negative_prompt_embeds,
                             video_action_scheduler, do_true_cfg, state, **kwargs):
            captured["diffuse"] = dict(
                video_latents=_t(video_latents),       # = noise_obs (transposed)
                action_latents=_t(action_latents),     # = noise_action
                timesteps_video=_t(timesteps_video),
                timesteps_action=_t(timesteps_action),
                prompt_embeds=_t(prompt_embeds),
                negative_prompt_embeds=_t(negative_prompt_embeds),
                do_true_cfg=bool(do_true_cfg),
                seq_len=int(kwargs.get("seq_len")),
                state_features=_t(kwargs.get("state_features")),
                embodiment_id=_t(kwargs.get("embodiment_id")),
            )
            raise _StopBeforeDiT()

        pipeline._prefill_kv_cache = _capture_prefill
        pipeline.diffuse = _capture_diffuse

        # ---- Per-stage timers (XPU-sync'd, so async work is attributed) -------
        # _encode_text is called twice (positive + negative prompt for CFG).
        # _encode_image internally calls CLIP (image_encoder.encode_image) then
        # the Wan VAE (_encode_vae_latents); time each separately so we can split
        # CLIP vs VAE inside the image-encode.
        stage_times: dict[str, list] = {}

        def _timed(name, fn):
            def wrapper(*a, **kw):
                current_omni_platform.synchronize()
                t0 = time.perf_counter()
                out = fn(*a, **kw)
                current_omni_platform.synchronize()
                stage_times.setdefault(name, []).append(time.perf_counter() - t0)
                return out
            return wrapper

        pipeline._encode_text = _timed("text_encode_umt5", pipeline._encode_text)
        pipeline._encode_image = _timed("image_encode_total", pipeline._encode_image)
        pipeline._encode_vae_latents = _timed("vae_encode", pipeline._encode_vae_latents)
        pipeline.image_encoder.encode_image = _timed(
            "clip_encode", pipeline.image_encoder.encode_image
        )

        # ---- Build the real observation (obs #1: reset, current_start_frame==0)
        session_id = f"dreamzero-preproc-{args.embodiment}"
        camera_frames, obs = build_first_observation(Path(args.video_dir), args.prompt, session_id, args.embodiment)
        log(f"[input] built observation from real camera mp4s, embodiment={args.embodiment}")

        def _make_req(sess):
            sp = OmniDiffusionSamplingParams(
                num_inference_steps=args.num_inference_steps,
                extra_args={"reset": True, "session_id": sess, "robot_obs": obs},
            )
            return OmniDiffusionRequest(prompts=[obs["prompt"]], sampling_params=sp, request_id=sess)

        def _run_once(sess):
            # Fresh session each call so current_start_frame==0 -> full from-scratch
            # CLIP+VAE first-frame encode path runs (not the continuation path).
            current_omni_platform.synchronize()
            t0 = time.perf_counter()
            try:
                pipeline.forward(_make_req(sess))
            except _StopBeforeDiT:
                pass
            current_omni_platform.synchronize()
            return time.perf_counter() - t0

        # ---- COLD pass: first call pays one-time XPU kernel JIT/compile cost ----
        cold_s = _run_once("cold")
        cold_stage_times = {k: list(v) for k, v in stage_times.items()}
        log(f"[run] PREPROCESS_COLD_S={cold_s:.4f}")

        # ---- WARM pass: kernels cached -> steady-state encode cost -------------
        captured.clear()
        stage_times.clear()
        warm_s = _run_once("warm")
        log(f"[run] PREPROCESS_WARM_S={warm_s:.4f}")

        # Report WARM as the representative preprocessing time.
        preprocess_s = warm_s
        log(f"[run] PREPROCESS_S={preprocess_s:.4f}  (warm; cold={cold_s:.4f})")

        if "prefill" not in captured or "diffuse" not in captured:
            raise RuntimeError(f"Did not capture DiT inputs; keys={list(captured)}")

        # --- Stage breakdown (sequential, single-threaded; sums ~= PREPROCESS_S) ---
        # text_encode_umt5 fires twice (positive + negative prompt); vae_encode &
        # clip_encode fire once, both nested inside image_encode_total.
        def _summ(st):
            out = {}
            for name, times in st.items():
                out[name] = {"calls": len(times), "total_s": sum(times), "per_call_s": times}
            return out

        def _report(tag, st, total_s):
            text_tot = sum(st.get("text_encode_umt5", []))
            img_tot = sum(st.get("image_encode_total", []))
            clip_tot = sum(st.get("clip_encode", []))
            vae_tot = sum(st.get("vae_encode", []))
            log(f"=========== STAGE BREAKDOWN [{tag}] (rank0) ===========")
            log(f"text_encode_umt5   : {text_tot*1000:8.1f} ms  ({len(st.get('text_encode_umt5', []))} calls: "
                f"{[f'{t*1000:.1f}ms' for t in st.get('text_encode_umt5', [])]})")
            log(f"  image_encode_total: {img_tot*1000:8.1f} ms  (1 call; contains CLIP + VAE + overhead)")
            log(f"    clip_encode     : {clip_tot*1000:8.1f} ms")
            log(f"    vae_encode      : {vae_tot*1000:8.1f} ms")
            log(f"  (image overhead   : {(img_tot-clip_tot-vae_tot)*1000:8.1f} ms  = mask build/concat/transpose)")
            log(f"SUM(text+image)     : {(text_tot+img_tot)*1000:8.1f} ms   vs PREPROCESS[{tag}]={total_s*1000:.1f} ms")

        _report("COLD", cold_stage_times, cold_s)
        _report("WARM", stage_times, warm_s)
        stage_summary = {"cold": _summ(cold_stage_times), "warm": _summ(stage_times)}

        peak = current_omni_platform.max_memory_allocated() / 1024**3

        results = {
            "tp_size": get_tensor_model_parallel_world_size(),
            "world_size": int(os.environ["WORLD_SIZE"]),
            "embodiment": args.embodiment,
            "prompt": args.prompt,
            "num_inference_steps": args.num_inference_steps,
            "model_load_s": model_load_s,
            "preprocess_s": preprocess_s,
            "preprocess_cold_s": cold_s,
            "preprocess_warm_s": warm_s,
            "stage_breakdown_s": stage_summary,
            "peak_xpu_gib": peak,
            "num_weights_loaded": len(loaded),
            "num_dit_blocks": len(pipeline.transformer.blocks),
        }

    # -------- Rank-0 saves everything --------
    if is_rank0:
        pf = captured["prefill"]
        df = captured["diffuse"]

        # Re-derive the human-facing model inputs (stitched frames + templated
        # prompt) via the transform, for the record.
        from vllm_omni.diffusion.models.dreamzero.transform.base import get_transform
        transform = get_transform(args.embodiment)
        unified = transform.transform_input(obs)

        # 1) original inputs
        np.savez(
            out_dir / "original_inputs.npz",
            **{k.replace("/", "__"): np.asarray(v) for k, v in obs.items() if isinstance(v, np.ndarray)},
        )
        # raw camera arrays (all frames per view)
        np.savez(
            out_dir / "original_camera_frames.npz",
            **{k.replace("/", "__"): v for k, v in camera_frames.items()},
        )
        with open(out_dir / "original_prompt.txt", "w") as f:
            f.write(args.prompt + "\n")
        with open(out_dir / "templated_prompt.txt", "w") as f:
            f.write(str(unified.get("prompt", "")) + "\n")
        # stitched multi-view frame(s) actually fed to the model
        stitched = unified.get("images")
        if stitched is not None:
            np.savez(out_dir / "model_input_stitched.npz", images=np.asarray(stitched))

        # 2) text encoder outputs
        torch.save(
            {
                "prompt_embeds": pf["prompt_embeds"],
                "negative_prompt_embeds": pf["negative_prompt_embeds"],
            },
            out_dir / "text_encoder_outputs.pt",
        )
        # 3) VAE / image embeddings
        torch.save(
            {
                "image_latents": pf["image_latents"],       # VAE first-frame latent (DiT hidden init)
                "state_ys": captured["state_ys"],            # VAE cond latents + mask (concat)
                "state_clip_feas": captured["state_clip_feas"],  # CLIP image features
            },
            out_dir / "vae_video_embeddings.pt",
        )
        # 4) full DiT-input bundle (everything the DiT would receive)
        torch.save(
            {
                # from _prefill_kv_cache
                "prefill_image_latents": pf["image_latents"],
                "prefill_prompt_embeds": pf["prompt_embeds"],
                "prefill_negative_prompt_embeds": pf["negative_prompt_embeds"],
                "prefill_frame_seqlen": pf["frame_seqlen"],
                "prefill_seq_len": pf["seq_len"],
                "prefill_do_true_cfg": pf["do_true_cfg"],
                # from diffuse (denoise-loop DiT inputs)
                "video_latents_noise": df["video_latents"],
                "action_latents_noise": df["action_latents"],
                "timesteps_video": df["timesteps_video"],
                "timesteps_action": df["timesteps_action"],
                "diffuse_prompt_embeds": df["prompt_embeds"],
                "diffuse_negative_prompt_embeds": df["negative_prompt_embeds"],
                "diffuse_seq_len": df["seq_len"],
                "state_features": df["state_features"],
                "embodiment_id": df["embodiment_id"],
                # conditioning carried on state
                "state_clip_feas": captured["state_clip_feas"],
                "state_ys": captured["state_ys"],
            },
            out_dir / "dit_inputs.pt",
        )

        # shapes/types manifest
        def _meta(x):
            if isinstance(x, torch.Tensor):
                return {"shape": list(x.shape), "dtype": str(x.dtype)}
            return {"value": x, "type": type(x).__name__}

        manifest = {
            "text_encoder_outputs": {
                "prompt_embeds": _meta(pf["prompt_embeds"]),
                "negative_prompt_embeds": _meta(pf["negative_prompt_embeds"]),
            },
            "vae_video_embeddings": {
                "image_latents": _meta(pf["image_latents"]),
                "state_ys": _meta(captured["state_ys"]),
                "state_clip_feas": _meta(captured["state_clip_feas"]),
            },
            "dit_inputs": {
                "video_latents_noise": _meta(df["video_latents"]),
                "action_latents_noise": _meta(df["action_latents"]),
                "timesteps_video": _meta(df["timesteps_video"]),
                "timesteps_action": _meta(df["timesteps_action"]),
                "state_features": _meta(df["state_features"]),
                "embodiment_id": _meta(df["embodiment_id"]),
                "prefill_frame_seqlen": _meta(pf["frame_seqlen"]),
                "prefill_seq_len": _meta(pf["seq_len"]),
                "diffuse_seq_len": _meta(df["seq_len"]),
                "prefill_do_true_cfg": _meta(pf["do_true_cfg"]),
            },
            "results": results,
        }
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        with open(out_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        log("=========== DiT INPUT SHAPES ===========")
        for grp in ("text_encoder_outputs", "vae_video_embeddings", "dit_inputs"):
            for k, v in manifest[grp].items():
                log(f"{grp}.{k} = {v}")
        log("=========== RESULTS ===========")
        for k, v in results.items():
            log(f"{k} = {v}")
        log(f"SAVED_DIR={out_dir}")
    log("DONE")


if __name__ == "__main__":
    main()
