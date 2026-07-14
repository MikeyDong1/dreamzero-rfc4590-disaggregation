#!/usr/bin/env python3
"""PROFILED DreamZero DiT-only replay under REAL Tensor Parallelism (TP=N).

Multi-process sibling of `dit_only_profile.py`. Launched with torchrun
(--nproc_per_node=N); every rank builds the pipeline, `load_weights` shards ONLY
the DiT (CausalWanModel: QKV/Column/RowParallelLinear + DistributedRMSNorm) across
the N ranks, and each rank keeps its whole DiT shard RESIDENT on its own XPU card
(no layerwise offload -- the point of TP is that the 1/N shard fits, removing the
TP=1 H2D-offload bottleneck).

DiT ONLY: we stream just `action_head.model.*` weights and move only
`pipeline.transformer` to the device. The UMT5 text encoder, CLIP image encoder,
and Wan VAE are NEVER moved to the XPU (`enable_layerwise_offload=True` also stops
the constructor from placing the VAE on-device). Inputs are the pre-encoded DiT
inputs (`dit_inputs.pt`) produced by the preprocessing stage, so no encoder runs.

Measurement (per skill): we separate timing from profiling.
  * WARMUP run (discarded) -> warms kernels / allocator.
  * TIMED run (NO profiler) -> the reported timing metrics (clean).
  * PROFILED run (rank0 wraps torch.profiler; all ranks execute the collective) ->
    chrome trace + op tables. Its wall time is labelled profiled (has overhead).

Rank 0 owns all artifact writing:
  profile/chrome_trace.json.gz, profile/op_table_self_{xpu,cpu}.txt,
  profile/profile_summary.json  (timing from the TIMED run; op tables from the
  PROFILED run; per-rank peak alloc all_gathered).

Usage (inside torchrun):
  torchrun --nproc_per_node=4 dit_tp_profile.py \
    --model-path ... --test-data-dir ... --out-dir ... \
    --tp-size 4 --profile-steps 16 --warmup-steps 4
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import torch

import vllm_omni  # noqa: F401  (side effect: disables broken triton-xpu import)

from vllm.config import CompilationConfig, DeviceConfig, VllmConfig

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
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

RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", RANK))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
IS_RANK0 = RANK == 0


def log(msg: str) -> None:
    print(f"[rank{RANK}] {msg}", flush=True)


def now() -> float:
    return time.time()


def barrier(tag: str = "") -> None:
    """Cross-rank sync so phase timers on rank0 are not skewed by stragglers."""
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
    except Exception as e:
        log(f"[barrier] ({tag}) skipped: {e}")


# --------------------------------------------------------------------------- #
# Distributed bootstrap (REAL N-rank TP world, driven by torchrun env)
# --------------------------------------------------------------------------- #
def bootstrap(device, tp_size: int) -> VllmConfig:
    current_omni_platform.set_device(device)
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(),
        device_config=DeviceConfig(device=DEVICE),
    )
    vllm_config.parallel_config.tensor_parallel_size = tp_size
    vllm_config.parallel_config.data_parallel_size = 1
    init_distributed_environment(world_size=WORLD_SIZE, rank=RANK, local_rank=LOCAL_RANK)
    initialize_model_parallel(
        data_parallel_size=1, cfg_parallel_size=1, sequence_parallel_size=1,
        ulysses_degree=1, ring_degree=1, tensor_parallel_size=tp_size, pipeline_parallel_size=1,
    )
    log(f"[bootstrap] world={WORLD_SIZE} rank={RANK} local_rank={LOCAL_RANK} "
        f"tp={get_tensor_model_parallel_world_size()} tp_rank={get_tensor_model_parallel_rank()} "
        f"device={device}")
    return vllm_config


# --------------------------------------------------------------------------- #
# Weight streaming -- DiT ONLY (action_head.model.*). The parallel-linear
# weight_loaders slice this rank's shard out of each full CPU tensor.
# --------------------------------------------------------------------------- #
def stream_dit_weights(model_path: str):
    from safetensors import safe_open

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    shard_to_keys: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        if key.startswith("action_head.model."):
            shard_to_keys.setdefault(shard, []).append(key)
    for shard, keys in shard_to_keys.items():
        with safe_open(os.path.join(model_path, shard), framework="pt", device="cpu") as f:
            for name in keys:
                yield name, f.get_tensor(name)


def _block_bytes(block: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in block.parameters()) + sum(
        b.numel() * b.element_size() for b in block.buffers()
    )


def move_dit_resident(dit, device):
    """Move the ENTIRE (already TP-sharded) DiT to the device -- no offload hooks.

    Each rank only holds its 1/TP shard of every parallel-linear, so the whole
    model is resident. Returns (num_blocks, resident_shard_mib)."""
    blocks = list(dit.blocks)
    num_blocks = len(blocks)
    for name, child in dit.named_children():
        child.to(device)
    for p in dit._parameters.values():
        if p is not None:
            p.data = p.data.to(device)
    for b in dit._buffers.values():
        if b is not None:
            b.data = b.data.to(device)
    shard_mib = sum(_block_bytes(b) for b in blocks) / 1024**2
    blk0 = _block_bytes(blocks[0]) / 1024**2
    log(f"[resident] {num_blocks} blocks ALL resident on {device}; "
        f"per-block shard ~{blk0:.0f} MiB, total block shard ~{shard_mib:.0f} MiB")
    return num_blocks, shard_mib


def enable_k_resident_offload(dit, device, resident_blocks, pin_memory=True):
    """OOM fallback only: sliding-window offload identical to dit_only_profile.py."""
    blocks = list(dit.blocks)
    num_blocks = len(blocks)
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
    K = max(0, min(resident_blocks, num_blocks - 2))
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
    log(f"[offload] K={K}/{num_blocks} resident, {n_off} sliding-window offloaded.")
    return num_blocks, K


def load_saved_inputs(test_data_dir: Path, device):
    dit = torch.load(test_data_dir / "dit_inputs.pt", map_location="cpu", weights_only=False)

    def to_dev(x, dtype=None):
        if not isinstance(x, torch.Tensor):
            return x
        return x.to(device=device, dtype=dtype) if dtype is not None else x.to(device=device)

    return {
        "image_latents": to_dev(dit["prefill_image_latents"], torch.bfloat16),
        "prompt_embeds": to_dev(dit["prefill_prompt_embeds"], torch.bfloat16),
        "negative_prompt_embeds": to_dev(dit["prefill_negative_prompt_embeds"], torch.bfloat16),
        "frame_seqlen": int(dit["prefill_frame_seqlen"]),
        "prefill_seq_len": int(dit["prefill_seq_len"]),
        "do_true_cfg": bool(dit["prefill_do_true_cfg"]),
        "video_latents_noise": to_dev(dit["video_latents_noise"], torch.bfloat16),
        "action_latents_noise": to_dev(dit["action_latents_noise"], torch.bfloat16),
        "diffuse_seq_len": int(dit["diffuse_seq_len"]),
        "state_features": to_dev(dit["state_features"], torch.bfloat16),
        "embodiment_id": to_dev(dit["embodiment_id"], torch.long),
        "state_clip_feas": to_dev(dit["state_clip_feas"], torch.bfloat16),
        "state_ys": to_dev(dit["state_ys"], torch.bfloat16),
    }


def _setup_state(pipeline, saved, tag):
    try:
        pipeline._states.clear()
    except Exception:
        pass
    try:
        current_omni_platform.empty_cache()
    except Exception:
        pass
    state = pipeline._get_or_create_state(f"dit-tp-{tag}")
    state.reset()
    state.current_start_frame = 0
    state.clip_feas = saved["state_clip_feas"]
    state.ys = saved["state_ys"]
    pipeline.state = state
    return state


def _regen_scheduler(pipeline, num_steps, device):
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
    return VideoActionScheduler(sample_scheduler, sample_scheduler_action), \
        sample_scheduler.timesteps, sample_scheduler_action.timesteps


def run_one(pipeline, saved, num_steps, device, tag):
    """One prefill + N-step denoise (one request). All ranks call this together."""
    state = _setup_state(pipeline, saved, tag)
    try:
        current_omni_platform.reset_peak_memory_stats()
    except Exception:
        pass
    barrier(f"pre-{tag}")
    current_omni_platform.synchronize()
    log(f"RUN_START tag={tag} N={num_steps} epoch={now():.3f}")

    t_pref0 = time.perf_counter()
    with torch.no_grad():
        pipeline._prefill_kv_cache(
            saved["image_latents"], saved["prompt_embeds"], saved["negative_prompt_embeds"],
            saved["frame_seqlen"], saved["prefill_seq_len"], saved["do_true_cfg"], state,
        )
    current_omni_platform.synchronize()
    prefill_s = time.perf_counter() - t_pref0
    log(f"[{tag} N={num_steps}] PREFILL_S={prefill_s:.3f}")

    vas, ts_v, ts_a = _regen_scheduler(pipeline, num_steps, device)

    step_times: list[float] = []
    first_output_wall = {"t": None}
    orig_step = vas.step
    _s = {"t0": time.perf_counter()}

    def timed_step(*a, **kw):
        out = orig_step(*a, **kw)
        current_omni_platform.synchronize()
        tnow = time.perf_counter()
        step_times.append(tnow - _s["t0"])
        if first_output_wall["t"] is None:
            first_output_wall["t"] = tnow
        _s["t0"] = tnow
        return out

    vas.step = timed_step
    t_loop0 = time.perf_counter()
    with torch.no_grad():
        video_out, action_out = pipeline.diffuse(
            video_latents=saved["video_latents_noise"],
            action_latents=saved["action_latents_noise"],
            timesteps_video=ts_v, timesteps_action=ts_a,
            prompt_embeds=saved["prompt_embeds"],
            negative_prompt_embeds=saved["negative_prompt_embeds"],
            video_action_scheduler=vas, do_true_cfg=saved["do_true_cfg"],
            state=state, seq_len=saved["diffuse_seq_len"],
            state_features=saved["state_features"], embodiment_id=saved["embodiment_id"],
        )
    current_omni_platform.synchronize()
    loop_s = time.perf_counter() - t_loop0
    log(f"RUN_END tag={tag} N={num_steps} epoch={now():.3f}")

    ttfo = prefill_s + (first_output_wall["t"] - t_loop0) if first_output_wall["t"] else None
    v = video_out.detach().float().cpu().numpy()
    a = action_out.detach().float().cpu().numpy()
    peak = current_omni_platform.max_memory_allocated() / 1024**2  # MiB
    try:
        peak_reserved = torch.xpu.max_memory_reserved() / 1024**2
    except Exception:
        peak_reserved = None
    return {
        "num_steps": num_steps, "prefill_s": prefill_s, "denoise_loop_s": loop_s,
        "time_to_first_output_s": ttfo, "time_to_complete_output_s": prefill_s + loop_s,
        "per_step_s": step_times, "video_out_shape": list(v.shape),
        "action_out_shape": list(a.shape),
        "video_finite": bool(np.isfinite(v).all()), "action_finite": bool(np.isfinite(a).all()),
        "peak_xpu_alloc_mib": peak, "peak_xpu_reserved_mib": peak_reserved,
    }


# --------------------------------------------------------------------------- #
# Profiler helpers (unchanged from dit_only_profile.py)
# --------------------------------------------------------------------------- #
def _activities():
    acts = [torch.profiler.ProfilerActivity.CPU]
    xpu_act = getattr(torch.profiler.ProfilerActivity, "XPU", None)
    if xpu_act is not None:
        acts.append(xpu_act)
        log("[profile] activities = CPU + XPU")
    else:
        log("[profile] WARNING: ProfilerActivity.XPU unavailable; CPU-only trace")
    return acts, (xpu_act is not None)


def _events_to_rows(ka, has_xpu, topn=40):
    rows = []
    for evt in ka:
        row = {
            "name": evt.key,
            "count": int(evt.count),
            "cpu_time_total_us": float(getattr(evt, "cpu_time_total", 0.0)),
            "self_cpu_time_total_us": float(getattr(evt, "self_cpu_time_total", 0.0)),
        }
        for cand in ("device_time_total", "xpu_time_total"):
            if hasattr(evt, cand):
                row["device_time_total_us"] = float(getattr(evt, cand))
                break
        for cand in ("self_device_time_total", "self_xpu_time_total"):
            if hasattr(evt, cand):
                row["self_device_time_total_us"] = float(getattr(evt, cand))
                break
        if hasattr(evt, "input_shapes"):
            row["input_shapes"] = str(evt.input_shapes)
        rows.append(row)
    keyfn = (lambda r: r.get("self_device_time_total_us", 0.0)) if has_xpu \
        else (lambda r: r["self_cpu_time_total_us"])
    rows.sort(key=keyfn, reverse=True)
    return rows[:topn]


def _table_sort_keys(ka, has_xpu):
    sample = ka[0] if len(ka) else None
    dev_key = None
    if has_xpu and sample is not None:
        for cand in ("self_xpu_time_total", "self_device_time_total"):
            if hasattr(sample, cand):
                dev_key = cand
                break
    return dev_key, "self_cpu_time_total"


def gather_peaks(local_peak_mib):
    """All-gather each rank's peak alloc (MiB). Returns list indexed by rank."""
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            buf = [None] * WORLD_SIZE
            torch.distributed.all_gather_object(buf, (RANK, local_peak_mib))
            out = [None] * WORLD_SIZE
            for r, v in buf:
                out[r] = v
            return out
    except Exception as e:
        log(f"[peaks] all_gather failed: {e}")
    out = [None] * WORLD_SIZE
    out[RANK] = local_peak_mib
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--test-data-dir", required=True)
    ap.add_argument("--out-dir", default="/workspace/out/dit_tp_profile")
    ap.add_argument("--tp-size", type=int, default=4)
    ap.add_argument("--profile-steps", type=int, default=16,
                    help="denoise steps (16 matches the encoded request)")
    ap.add_argument("--warmup-steps", type=int, default=4,
                    help="throwaway denoise before timing/profiling")
    ap.add_argument("--resident-blocks", type=int, default=-1,
                    help="-1 = full-resident (default, no offload); >=0 = OOM-fallback "
                         "sliding-window offload with K resident blocks")
    args = ap.parse_args()

    device = current_omni_platform.get_torch_device(LOCAL_RANK)
    out_dir = Path(args.out_dir)
    prof_dir = out_dir / "profile"
    if IS_RANK0:
        prof_dir.mkdir(parents=True, exist_ok=True)

    od_config = OmniDiffusionConfig(
        model=args.model_path, dtype=torch.bfloat16,
        enforce_eager=True, enable_layerwise_offload=True,
    )
    vllm_config = bootstrap(device, args.tp_size)

    summary: dict = {}
    with set_forward_context(vllm_config=vllm_config, omni_diffusion_config=od_config), \
         set_current_diffusion_config(od_config):
        t_load0 = time.perf_counter()
        log("[load] constructing DreamZeroPipeline (CPU) ...")
        pipeline = DreamZeroPipeline(od_config=od_config)
        pipeline.eval()
        pipeline.transformer.to(dtype=torch.bfloat16)
        log("[load] streaming DiT weights (action_head.model.*, TP-sharded) ...")
        loaded = pipeline.load_weights(stream_dit_weights(args.model_path))
        log(f"[load] loaded {len(loaded)} DiT weight params (this rank's shard)")

        offload_mode = "full_resident"
        K = None
        if args.resident_blocks < 0:
            num_blocks, shard_mib = move_dit_resident(pipeline.transformer, device)
        else:
            offload_mode = f"sliding_window_K{args.resident_blocks}"
            num_blocks, K = enable_k_resident_offload(
                pipeline.transformer, device, resident_blocks=args.resident_blocks)
            shard_mib = None

        current_omni_platform.synchronize()
        barrier("post-load")
        model_load_s = time.perf_counter() - t_load0
        log(f"[load] MODEL_LOAD_S={model_load_s:.3f} mode={offload_mode}")

        saved = load_saved_inputs(Path(args.test_data_dir), device)
        log("[input] pre-encoded DiT inputs loaded (no encoder run)")

        # ---- Warmup (discarded) ----
        if args.warmup_steps > 0:
            log(f"[warmup] throwaway {args.warmup_steps}-step denoise ...")
            _w = run_one(pipeline, saved, args.warmup_steps, device, tag="warmup")
            log(f"[warmup] done complete={_w['time_to_complete_output_s']:.3f}s (DISCARDED)")

        # ---- TIMED run (NO profiler): the reported timing metrics ----
        log(f"[timed] clean timed run (no profiler): N={args.profile_steps} ...")
        timed = run_one(pipeline, saved, args.profile_steps, device, tag="timed")
        log(f"[timed] complete={timed['time_to_complete_output_s']:.3f}s "
            f"prefill={timed['prefill_s']:.3f}s ttfo={timed['time_to_first_output_s']:.3f}s "
            f"peak_alloc={timed['peak_xpu_alloc_mib']:.0f}MiB "
            f"finite(v/a)={timed['video_finite']}/{timed['action_finite']}")

        # ---- PROFILED run: rank0 wraps profiler; all ranks execute collective ----
        acts, has_xpu = _activities()
        prof_res = None
        ka = None
        if IS_RANK0:
            prof_kwargs = dict(activities=acts, record_shapes=True, with_stack=False)
            try:
                prof_kwargs["profile_memory"] = True
            except Exception:
                pass
            log(f"[profile] profiling ONE warm run on rank0: N={args.profile_steps} ...")
            with torch.profiler.profile(**prof_kwargs) as prof:
                prof_res = run_one(pipeline, saved, args.profile_steps, device, tag="prof")
            ka = prof.key_averages(group_by_input_shape=False)
            log(f"[profile] profiled run complete={prof_res['time_to_complete_output_s']:.3f}s "
                f"(has profiler overhead)")
        else:
            prof_res = run_one(pipeline, saved, args.profile_steps, device, tag="prof")

        barrier("post-prof")
        peaks = gather_peaks(timed["peak_xpu_alloc_mib"])

        # ---- Rank0 writes all artifacts ----
        if IS_RANK0:
            dev_key, cpu_key = _table_sort_keys(ka, has_xpu)
            if dev_key:
                try:
                    (prof_dir / "op_table_self_xpu.txt").write_text(
                        ka.table(sort_by=dev_key, row_limit=50))
                    log("[profile] wrote op_table_self_xpu.txt")
                except Exception as e:
                    log(f"[profile] xpu table failed: {e}")
            try:
                (prof_dir / "op_table_self_cpu.txt").write_text(
                    ka.table(sort_by=cpu_key, row_limit=50))
                log("[profile] wrote op_table_self_cpu.txt")
            except Exception as e:
                log(f"[profile] cpu table failed: {e}")

            raw_trace = prof_dir / "chrome_trace.json"
            try:
                prof.export_chrome_trace(str(raw_trace))
                with open(raw_trace, "rb") as fin, gzip.open(str(raw_trace) + ".gz", "wb") as fout:
                    shutil.copyfileobj(fin, fout)
                os.remove(raw_trace)
                log("[profile] wrote chrome_trace.json.gz")
            except Exception as e:
                log(f"[profile] chrome trace failed: {e}")

            top_rows = _events_to_rows(ka, has_xpu, topn=40)
            total_self_dev = sum(r.get("self_device_time_total_us", 0.0)
                                 for r in _events_to_rows(ka, has_xpu, topn=100000))
            total_self_cpu = sum(r["self_cpu_time_total_us"]
                                 for r in _events_to_rows(ka, has_xpu, topn=100000))

            try:
                _dp = torch.xpu.get_device_properties(LOCAL_RANK)
                dev_str = f"xpu:{LOCAL_RANK} ({_dp.name}, {_dp.total_memory // 1024 // 1024} MiB)"
            except Exception:
                dev_str = f"xpu:{LOCAL_RANK}"

            valid_peaks = [p for p in peaks if p is not None]
            summary = {
                "test": "dreamzero_dit_only_TP",
                "tp_size": get_tensor_model_parallel_world_size(),
                "world_size": WORLD_SIZE,
                "device_rank0": dev_str,
                "cards": os.environ.get("ZE_AFFINITY_MASK", "unset"),
                "torch_version": torch.__version__,
                "python": platform.python_version(),
                "offload_mode": offload_mode,
                "resident_blocks_K": K,
                "dit_block_shard_mib_per_rank": shard_mib,
                "num_dit_blocks": num_blocks,
                "num_dit_weight_params_rank0": len(loaded),
                "profile_steps": args.profile_steps,
                "warmup_steps": args.warmup_steps,
                "measurement": "warm",
                "model_load_s": model_load_s,
                # timing metrics come from the CLEAN timed run (no profiler overhead)
                "timing_run": timed,
                # profiled run wall time carries profiler overhead -- kept separate
                "profiled_run": prof_res,
                "peak_xpu_alloc_mib_per_rank": peaks,
                "peak_xpu_alloc_mib_max": max(valid_peaks) if valid_peaks else None,
                "peak_xpu_alloc_mib_aggregate": sum(valid_peaks) if valid_peaks else None,
                "profiler": {
                    "activities": [str(a) for a in acts],
                    "has_xpu_activity": has_xpu,
                    "device_sort_key": dev_key,
                    "total_self_device_time_us": total_self_dev,
                    "total_self_cpu_time_us": total_self_cpu,
                    "top_ops": top_rows,
                },
                "metric_notes": {
                    "time_to_first_output_s": "prefill + first denoise step (first DiT latent output; no VAE decode in DiT-only test)",
                    "time_to_complete_output_s": "prefill + full N-step denoise loop (DiT latents+action; MP4/VAE decode NOT run)",
                    "decode_time_s": "null: VAE decode not run in DiT-only test",
                    "peak_xpu_alloc_mib": "per-rank torch allocator peak (process), not whole-device; see xpu_memory.csv for whole-device system peaks",
                    "timing_vs_profiled": "timing_run has NO profiler; profiled_run wall time includes profiler overhead",
                },
            }
            (prof_dir / "profile_summary.json").write_text(json.dumps(summary, indent=2))
            log("=========== TP PROFILE SUMMARY (rank0) ===========")
            log(json.dumps({k: v for k, v in summary.items() if k != "profiler"}, indent=2))
            log(f"SAVED_PROFILE_DIR={prof_dir}")
    barrier("done")
    log("DONE")


if __name__ == "__main__":
    main()
