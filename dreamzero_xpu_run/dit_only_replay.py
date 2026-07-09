#!/usr/bin/env python3
"""DreamZero DiT-ONLY replay on ONE XPU card (TP=1) with K-resident layerwise offload.

Runs *only* the DiT part of DreamZero-DROID -- the CausalWanModel prefill + denoise
loop -- fed by PRE-COMPUTED, SAVED preprocessing outputs (the embodiment/text/VAE
encodings captured earlier at the true DiT boundary by `preprocess_only_tp8.py`).
No UMT5 / CLIP / VAE encode runs here, and no VAE decode: this isolates the DiT.

Why build the *real* DreamZeroPipeline instead of a bare CausalWanModel?
  So we reuse the pipeline's EXACT DiT-path logic -- `_prefill_kv_cache`, `diffuse`,
  the CFG mixin (`predict_noise_maybe_with_cfg`), and the FlowUniPC scheduler setup
  from `forward()`. We just skip the encode/decode halves and inject the saved
  tensors. Only `action_head.model.*` (DiT) weights are loaded; the encoder/VAE
  modules are built (random) but never touched, so they stay on CPU.

Layerwise offload: the first K DiT blocks stay permanently resident on the XPU; the
remaining (num_blocks - K) are sliding-window CPU<->XPU offloaded (repo hooks). This
lets the ~28GB DiT run on a single card.

Sweeps num_inference_steps in {4,8,16} in ONE process (DiT loaded once). Per N: a
fresh state + KV cache, the allocator peak is reset, and the timestep schedule is
regenerated exactly as pipeline.forward() does -- so step count is the only variable
(the initial noise is the saved one, identical across N). Reports MODEL_LOAD_S and,
per N: PREFILL_S, TIME_TO_FIRST_OUTPUT_S, TIME_TO_COMPLETE_OUTPUT_S, per-step times,
allocator peak, and output finiteness. Emits epoch-time RUN_START/RUN_END markers so a
host-side xpu-smi sampler can slice whole-device peak per N.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

# Import vllm_omni FIRST (side effect: disables the broken triton-xpu import before
# vllm.config is imported).
import vllm_omni  # noqa: F401

from vllm.config import CompilationConfig, DeviceConfig, VllmConfig

from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import (
    DreamZeroPipeline,
    VideoActionScheduler,
)
from vllm_omni.diffusion.offloader.layerwise_backend import apply_block_hook
from vllm_omni.platforms import current_omni_platform

DEVICE = "xpu"


def log(msg: str) -> None:
    print(msg, flush=True)


def now() -> float:
    """Epoch seconds (wall clock) -- for cross-process CSV correlation with xpu-smi."""
    return time.time()


# ---------------------------------------------------------------------------
# Distributed bootstrap (world_size=1, tp=1)
# ---------------------------------------------------------------------------
def bootstrap(device) -> VllmConfig:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29557")
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


# ---------------------------------------------------------------------------
# Weight streaming -- DiT ONLY (action_head.model.*)
# ---------------------------------------------------------------------------
def stream_dit_weights(model_path: str):
    from safetensors import safe_open

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    shard_to_keys: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        if key.startswith("action_head.model."):  # DiT weights only
            shard_to_keys.setdefault(shard, []).append(key)
    for shard, keys in shard_to_keys.items():
        with safe_open(os.path.join(model_path, shard), framework="pt", device="cpu") as f:
            for name in keys:
                yield name, f.get_tensor(name)


# ---------------------------------------------------------------------------
# K-resident layerwise offload on the DiT blocks
# ---------------------------------------------------------------------------
def _block_bytes(block: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in block.parameters()) + sum(
        b.numel() * b.element_size() for b in block.buffers()
    )


def enable_k_resident_offload(dit: torch.nn.Module, device, resident_blocks: int,
                              pin_memory: bool = True):
    blocks = list(dit.blocks)
    num_blocks = len(blocks)

    # Non-block children resident (img_emb, patch_embedding, head, time_embedding...).
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
        f"{num_blocks - K} sliding-window offloaded (per-step H2D cut ~{100*K/num_blocks:.0f}%).")

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
    return hooks, K


# ---------------------------------------------------------------------------
# Load saved DiT inputs and move to device with correct dtypes
# ---------------------------------------------------------------------------
def load_saved_inputs(test_data_dir: Path, device):
    dit = torch.load(test_data_dir / "dit_inputs.pt", map_location="cpu", weights_only=False)

    def to_dev(x, dtype=None):
        if not isinstance(x, torch.Tensor):
            return x
        if dtype is not None:
            return x.to(device=device, dtype=dtype)
        return x.to(device=device)

    saved = {
        # prefill args
        "image_latents": to_dev(dit["prefill_image_latents"], torch.bfloat16),
        "prompt_embeds": to_dev(dit["prefill_prompt_embeds"], torch.bfloat16),
        "negative_prompt_embeds": to_dev(dit["prefill_negative_prompt_embeds"], torch.bfloat16),
        "frame_seqlen": int(dit["prefill_frame_seqlen"]),
        "prefill_seq_len": int(dit["prefill_seq_len"]),
        "do_true_cfg": bool(dit["prefill_do_true_cfg"]),
        # diffuse args
        "video_latents_noise": to_dev(dit["video_latents_noise"], torch.bfloat16),
        "action_latents_noise": to_dev(dit["action_latents_noise"], torch.bfloat16),
        "diffuse_seq_len": int(dit["diffuse_seq_len"]),
        "state_features": to_dev(dit["state_features"], torch.bfloat16),
        "embodiment_id": to_dev(dit["embodiment_id"], torch.long),
        # conditioning carried on state
        "state_clip_feas": to_dev(dit["state_clip_feas"], torch.bfloat16),
        "state_ys": to_dev(dit["state_ys"], torch.bfloat16),
    }
    return saved


# ---------------------------------------------------------------------------
# One DiT run (prefill + N-step denoise) using saved inputs
# ---------------------------------------------------------------------------
def run_one(pipeline: DreamZeroPipeline, saved: dict, num_steps: int, device, tag: str):
    # Evict ALL prior per-session states first so their on-device KV caches are
    # freed — otherwise each run's peak accumulates the previous runs' caches and
    # the per-N peak is inflated (not isolated). Then empty the allocator cache.
    try:
        pipeline._states.clear()
    except Exception:
        pass
    try:
        current_omni_platform.empty_cache()
    except Exception:
        pass

    # Fresh state per run (isolates KV cache + peak between N values).
    state = pipeline._get_or_create_state(f"dit-replay-{tag}")
    state.reset()
    state.current_start_frame = 0
    state.clip_feas = saved["state_clip_feas"]
    state.ys = saved["state_ys"]
    pipeline.state = state

    try:
        current_omni_platform.reset_peak_memory_stats()
    except Exception:
        pass
    current_omni_platform.synchronize()

    log(f"RUN_START N={num_steps} epoch={now():.3f}")

    # ---- PREFILL (current_start_frame 0 -> 1: fills KV cache via a DiT forward) ----
    # NOTE: pipeline.forward() is decorated @torch.no_grad(); we call the internal
    # DiT methods directly, so we MUST re-establish no_grad ourselves or autograd
    # retains activations for all 40 blocks (x2 CFG passes) and device memory
    # explodes to the ceiling (fragmentation hang / DEVICE_LOST).
    t_pref0 = time.perf_counter()
    with torch.no_grad():
        pipeline._prefill_kv_cache(
            saved["image_latents"],
            saved["prompt_embeds"],
            saved["negative_prompt_embeds"],
            saved["frame_seqlen"],
            saved["prefill_seq_len"],
            saved["do_true_cfg"],
            state,
        )
    current_omni_platform.synchronize()
    prefill_s = time.perf_counter() - t_pref0
    log(f"[N={num_steps}] PREFILL_S={prefill_s:.3f} (csf now {state.current_start_frame})")

    # ---- Scheduler setup: regenerate timesteps for THIS N (mirror forward()) ----
    pipeline.num_inference_steps = num_steps
    sample_scheduler = copy.deepcopy(pipeline.scheduler)
    sample_scheduler_action = copy.deepcopy(pipeline.scheduler)
    sample_scheduler.set_timesteps(num_steps, device=device, shift=pipeline.sigma_shift)
    sample_scheduler_action.set_timesteps(num_steps, device=device, shift=pipeline.sigma_shift)
    if pipeline.decouple_inference_noise:
        video_final_noise = pipeline.video_inference_final_noise
        sigma_max = sample_scheduler.sigmas[0].item()
        sample_scheduler.sigmas = (
            sample_scheduler.sigmas * (sigma_max - video_final_noise) / sigma_max + video_final_noise
        )
        sample_scheduler.timesteps = (sample_scheduler.sigmas[:-1] * 1000).to(torch.int64)
    video_action_scheduler = VideoActionScheduler(sample_scheduler, sample_scheduler_action)
    ts_v = sample_scheduler.timesteps
    ts_a = sample_scheduler_action.timesteps
    log(f"[N={num_steps}] timesteps_video(len={len(ts_v)})={ts_v.tolist()}")

    # ---- DENOISE LOOP -- instrument per-step by monkeypatching the scheduler.step
    #      boundary. We wrap diffuse's per-step timing via a step counter hook. ----
    step_times: list[float] = []
    first_output_wall = {"t": None}
    t_loop0 = time.perf_counter()

    orig_step = video_action_scheduler.step
    _state = {"t_step0": time.perf_counter()}

    def timed_step(*a, **kw):
        out = orig_step(*a, **kw)
        current_omni_platform.synchronize()
        tnow = time.perf_counter()
        step_times.append(tnow - _state["t_step0"])
        if first_output_wall["t"] is None:
            first_output_wall["t"] = tnow
        _state["t_step0"] = tnow
        return out

    video_action_scheduler.step = timed_step

    with torch.no_grad():
      video_out, action_out = pipeline.diffuse(
        video_latents=saved["video_latents_noise"],
        action_latents=saved["action_latents_noise"],
        timesteps_video=ts_v,
        timesteps_action=ts_a,
        prompt_embeds=saved["prompt_embeds"],
        negative_prompt_embeds=saved["negative_prompt_embeds"],
        video_action_scheduler=video_action_scheduler,
        do_true_cfg=saved["do_true_cfg"],
        state=state,
        seq_len=saved["diffuse_seq_len"],
        state_features=saved["state_features"],
        embodiment_id=saved["embodiment_id"],
    )
    current_omni_platform.synchronize()
    loop_s = time.perf_counter() - t_loop0
    log(f"RUN_END N={num_steps} epoch={now():.3f}")

    ttfo = prefill_s + (first_output_wall["t"] - t_loop0) if first_output_wall["t"] else None
    complete = prefill_s + loop_s

    v = video_out.detach().float().cpu().numpy()
    a = action_out.detach().float().cpu().numpy()
    peak = current_omni_platform.max_memory_allocated() / 1024**3
    try:
        peak_reserved = torch.xpu.max_memory_reserved() / 1024**3
    except Exception:
        peak_reserved = None

    return {
        "num_steps": num_steps,
        "prefill_s": prefill_s,
        "denoise_loop_s": loop_s,
        "time_to_first_output_s": ttfo,
        "time_to_complete_output_s": complete,
        "per_step_s": step_times,
        "video_out_shape": list(v.shape),
        "action_out_shape": list(a.shape),
        "video_finite": bool(np.isfinite(v).all()),
        "action_finite": bool(np.isfinite(a).all()),
        "video_min": float(v.min()), "video_max": float(v.max()),
        "video_mean": float(v.mean()), "video_std": float(v.std()),
        "action_min": float(a.min()), "action_max": float(a.max()),
        "peak_xpu_alloc_gib": peak,
        "peak_xpu_reserved_gib": peak_reserved,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--test-data-dir", required=True)
    ap.add_argument("--out-dir", default="/workspace/out/dit_only_replay")
    ap.add_argument("--resident-blocks", type=int, default=10, help="K resident DiT blocks")
    ap.add_argument("--steps", default="4,8,16", help="comma-separated denoise step counts")
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="If >0, run a throwaway denoise of this many steps BEFORE the measured "
                         "sweep to absorb cold-start cost (kernel JIT, first-touch offload H2D, "
                         "allocator warmup). Its metrics are discarded. Makes every measured N warm.")
    args = ap.parse_args()

    steps_list = [int(s) for s in args.steps.split(",") if s.strip()]
    device = torch.device(DEVICE, 0)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    od_config = OmniDiffusionConfig(
        model=args.model_path,
        dtype=torch.bfloat16,
        enforce_eager=True,
        enable_layerwise_offload=True,  # keeps VAE/encoders off-device at build
    )
    vllm_config = bootstrap(device)

    all_results: dict = {}
    with set_forward_context(vllm_config=vllm_config, omni_diffusion_config=od_config), \
         set_current_diffusion_config(od_config):
        # ---- Build pipeline (CPU) + load DiT weights only ----
        t_load0 = time.perf_counter()
        log("[load] constructing DreamZeroPipeline (CPU) ...")
        pipeline = DreamZeroPipeline(od_config=od_config)
        pipeline.eval()
        # Cast DiT to bf16 BEFORE load so resident blocks are bf16 (K=10 fits card).
        pipeline.transformer.to(dtype=torch.bfloat16)
        log("[load] streaming DiT weights (action_head.model.*) ...")
        loaded = pipeline.load_weights(stream_dit_weights(args.model_path))
        log(f"[load] loaded {len(loaded)} DiT weights")

        # ---- K-resident layerwise offload on the DiT ----
        _hooks, K = enable_k_resident_offload(pipeline.transformer, device,
                                              resident_blocks=args.resident_blocks)
        # DiT top-level buffers (rope freqs etc.) already moved; pipeline VAE/encoders
        # stay on CPU (never used). Push small pipeline buffers to device just in case.
        for _bn, b in list(pipeline.named_buffers()):
            if _bn.startswith("transformer."):
                continue
            # leave encoder/vae buffers on CPU
        current_omni_platform.synchronize()
        model_load_s = time.perf_counter() - t_load0
        log(f"[load] MODEL_LOAD_S={model_load_s:.3f}  (K={K}, num_blocks={len(pipeline.transformer.blocks)})")

        # ---- Load saved DiT inputs ----
        saved = load_saved_inputs(Path(args.test_data_dir), device)
        log("[input] saved DiT inputs loaded: "
            f"image_latents={tuple(saved['image_latents'].shape)}, "
            f"video_noise={tuple(saved['video_latents_noise'].shape)}, "
            f"ys={tuple(saved['state_ys'].shape)}, clip={tuple(saved['state_clip_feas'].shape)}, "
            f"embodiment_id={saved['embodiment_id'].tolist()}, do_true_cfg={saved['do_true_cfg']}")

        # ---- WARMUP (discarded): absorb cold-start so every measured N is warm ----
        if args.warmup_steps > 0:
            log(f"[warmup] running throwaway {args.warmup_steps}-step denoise to warm kernels/offload/allocator ...")
            _w = run_one(pipeline, saved, args.warmup_steps, device, tag="warmup")
            log(f"[warmup] done: prefill={_w['prefill_s']:.3f}s complete={_w['time_to_complete_output_s']:.3f}s "
                f"(DISCARDED). Measured runs below are WARM.")

        # ---- Sweep denoise steps ----
        runs = []
        for n in steps_list:
            res = run_one(pipeline, saved, n, device, tag=str(n))
            log(f"=== N={n}: prefill={res['prefill_s']:.3f}s ttfo={res['time_to_first_output_s']:.3f}s "
                f"complete={res['time_to_complete_output_s']:.3f}s peak_alloc={res['peak_xpu_alloc_gib']:.3f}GiB "
                f"finite(v/a)={res['video_finite']}/{res['action_finite']}")
            runs.append(res)

        try:
            _dp = torch.xpu.get_device_properties(0)
            _dev_str = f"xpu:0 (card0, {_dp.name}, {_dp.total_memory // 1024 // 1024} MiB)"
        except Exception:
            _dev_str = "xpu:0 (card0)"
        all_results = {
            "model_load_s": model_load_s,
            "resident_blocks_K": K,
            "num_dit_blocks": len(pipeline.transformer.blocks),
            "num_dit_weights_loaded": len(loaded),
            "device": _dev_str,
            "warmup_steps": args.warmup_steps,
            "measurement": "warm" if args.warmup_steps > 0 else "cold",
            "runs": runs,
        }

    log("=========== RESULTS ===========")
    log(json.dumps(all_results, indent=2))
    with open(out_dir / "dit_only_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"SAVED_RESULTS={out_dir / 'dit_only_results.json'}")
    log("DONE")


if __name__ == "__main__":
    main()
