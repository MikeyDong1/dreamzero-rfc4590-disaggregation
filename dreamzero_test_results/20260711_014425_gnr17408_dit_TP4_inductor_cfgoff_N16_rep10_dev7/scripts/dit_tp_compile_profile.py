#!/usr/bin/env python3
"""TP=4 DiT-only replay with CFG-toggle + eager/inductor toggle, WARM + profiled +
a 10x sequential-repeat completion-time distribution.

Combines `dit_tp_profile.py` (real N-rank TP world, torchrun launch, each rank's
1/N DiT shard fully resident, VAE/text-encoder never on XPU) with
`dit_cfg_compile_profile.py`'s two knobs:

  --cfg {on,off}       OFF => predict_noise runs only the positive branch => 1 DiT
                       forward/step instead of 2 => halves per-step compute AND
                       halves the tensor-parallel allreduce call count (each
                       forward does its own QKV/O allreduces). Output has no
                       classifier-free guidance -- expected, numerically different.
  --compile {eager,inductor}
                       inductor => torch.compile(block.forward, mode="default",
                       fullgraph=False, dynamic=False) on every DiT block BEFORE
                       moving to device. fullgraph=False is required: DiT
                       self-attention calls the XPU FlashAttention-2 custom op
                       (_vllm_fa2_C::varlen_fwd), which has no fake/meta kernel,
                       so Dynamo graph-breaks there and compiles everything else
                       (QKV/O/FFN projections, layernorms, RoPE, elementwise/cat/
                       gelu tail, and -- new vs the TP=1 harness -- the
                       tensor_model_parallel_all_reduce wrapper call itself, since
                       that's plain aten ops around a c10d op Dynamo treats as an
                       opaque call). NOT previously tested combined with TP>1.

Measurement sequence (per rank, all synchronized):
  1. WARMUP (discarded) -- also triggers inductor's first-call compilation so the
     compile cost is paid here, not in the timed runs.
  2. CLEAN timed run, N=profile-steps, NO profiler -- the reported single-request
     metrics.
  3. PROFILED run, N=profile-steps -- rank0 wraps torch.profiler (CPU+XPU) for the
     trace/op-tables; all ranks execute the collective.
  4. REPEAT: run_one() called `--repeat` times back-to-back (same session reset
     each time, same inputs) -- a completion-time distribution for repeated
     single-request serving, NOT a throughput/batching test.

Usage (inside torchrun):
  torchrun --nproc_per_node=4 dit_tp_compile_profile.py \
    --model-path ... --test-data-dir ... --out-dir ... \
    --tp-size 4 --cfg off --compile inductor \
    --profile-steps 16 --warmup-steps 4 --repeat 10
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
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
    except Exception as e:
        log(f"[barrier] ({tag}) skipped: {e}")


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


def maybe_compile_blocks(dit, mode: str) -> dict:
    """Compile each (already TP-sharded) DiT block.forward with inductor, BEFORE
    moving to device. Same fullgraph=False rationale as dit_cfg_compile_profile.py
    (attention custom op has no fake kernel) -- untested here combined with the
    tensor_model_parallel_all_reduce inside DistributedRMSNorm/attention."""
    if mode != "inductor":
        log("[compile] mode=eager (no torch.compile)")
        return {"mode": "eager", "compiled_partial": 0, "eager": len(dit.blocks)}
    partial = 0
    for block in dit.blocks:
        block.forward = torch.compile(block.forward, mode="default", fullgraph=False, dynamic=False)
        partial += 1
    log(f"[compile] mode=inductor (fullgraph=False, graph-break at attention custom op "
        f"+ possible break at tensor_model_parallel_all_reduce): {partial}/{len(dit.blocks)} blocks wrapped")
    return {"mode": "inductor", "fullgraph": False, "compiled_partial": partial, "eager": 0,
            "note": "graph break at _vllm_fa2_C::varlen_fwd (no fake impl); TP all_reduce combo untested prior to this run"}


def move_dit_resident(dit, device):
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
    state = pipeline._get_or_create_state(f"dit-tpc-{tag}")
    state.reset()
    state.current_start_frame = 0
    state.clip_feas = saved["state_clip_feas"]
    state.ys = saved["state_ys"]
    pipeline.state = state
    return state


def _regen_scheduler(pipeline, num_steps, device):
    pipeline.num_inference_steps = num_steps
    ss = copy.deepcopy(pipeline.scheduler)
    ssa = copy.deepcopy(pipeline.scheduler)
    ss.set_timesteps(num_steps, device=device, shift=pipeline.sigma_shift)
    ssa.set_timesteps(num_steps, device=device, shift=pipeline.sigma_shift)
    if pipeline.decouple_inference_noise:
        vfn = pipeline.video_inference_final_noise
        smax = ss.sigmas[0].item()
        ss.sigmas = ss.sigmas * (smax - vfn) / smax + vfn
        ss.timesteps = (ss.sigmas[:-1] * 1000).to(torch.int64)
    return VideoActionScheduler(ss, ssa), ss.timesteps, ssa.timesteps


def run_one(pipeline, saved, num_steps, device, tag, cfg_on: bool):
    """One prefill + N-step denoise (one request). All ranks call this together."""
    state = _setup_state(pipeline, saved, tag)
    try:
        current_omni_platform.reset_peak_memory_stats()
    except Exception:
        pass
    neg = saved["negative_prompt_embeds"] if cfg_on else None
    barrier(f"pre-{tag}")
    current_omni_platform.synchronize()
    log(f"RUN_START tag={tag} N={num_steps} cfg={cfg_on} epoch={now():.3f}")

    t_pref0 = time.perf_counter()
    with torch.no_grad():
        pipeline._prefill_kv_cache(
            saved["image_latents"], saved["prompt_embeds"], neg,
            saved["frame_seqlen"], saved["prefill_seq_len"], cfg_on, state,
        )
    current_omni_platform.synchronize()
    prefill_s = time.perf_counter() - t_pref0

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
            negative_prompt_embeds=neg,
            video_action_scheduler=vas, do_true_cfg=cfg_on,
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
        "num_steps": num_steps, "cfg": cfg_on, "prefill_s": prefill_s, "denoise_loop_s": loop_s,
        "time_to_first_output_s": ttfo, "time_to_complete_output_s": prefill_s + loop_s,
        "per_step_s": step_times, "video_out_shape": list(v.shape),
        "action_out_shape": list(a.shape),
        "video_finite": bool(np.isfinite(v).all()), "action_finite": bool(np.isfinite(a).all()),
        "peak_xpu_alloc_mib": peak, "peak_xpu_reserved_mib": peak_reserved,
    }


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
            "name": evt.key, "count": int(evt.count),
            "cpu_time_total_us": float(getattr(evt, "cpu_time_total", 0.0)),
            "self_cpu_time_total_us": float(getattr(evt, "self_cpu_time_total", 0.0)),
        }
        for cand in ("device_time_total", "xpu_time_total"):
            if hasattr(evt, cand):
                row["device_time_total_us"] = float(getattr(evt, cand)); break
        for cand in ("self_device_time_total", "self_xpu_time_total"):
            if hasattr(evt, cand):
                row["self_device_time_total_us"] = float(getattr(evt, cand)); break
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
    ap.add_argument("--out-dir", default="/workspace/out/dit_tp_compile_profile")
    ap.add_argument("--tp-size", type=int, default=4)
    ap.add_argument("--cfg", choices=["on", "off"], default="off")
    ap.add_argument("--compile", choices=["eager", "inductor"], default="inductor")
    ap.add_argument("--profile-steps", type=int, default=16)
    ap.add_argument("--warmup-steps", type=int, default=4)
    ap.add_argument("--repeat", type=int, default=10,
                    help="sequential same-input repeats AFTER the profiled run, for a completion-time distribution")
    args = ap.parse_args()

    cfg_on = (args.cfg == "on")
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

        compile_info = maybe_compile_blocks(pipeline.transformer, args.compile)
        num_blocks, shard_mib = move_dit_resident(pipeline.transformer, device)

        current_omni_platform.synchronize()
        barrier("post-load")
        model_load_s = time.perf_counter() - t_load0
        log(f"[load] MODEL_LOAD_S={model_load_s:.3f} cfg={args.cfg} compile={args.compile}")

        saved = load_saved_inputs(Path(args.test_data_dir), device)
        log(f"[input] pre-encoded DiT inputs loaded (saved do_true_cfg={saved['do_true_cfg']}, using cfg={cfg_on})")

        # ---- Warmup (discarded); also triggers inductor first-call compile ----
        if args.warmup_steps > 0:
            log(f"[warmup] throwaway {args.warmup_steps}-step denoise (compiles blocks if inductor) ...")
            t_w = time.perf_counter()
            _w = run_one(pipeline, saved, args.warmup_steps, device, "warmup", cfg_on)
            barrier("post-warmup")
            log(f"[warmup] done complete={_w['time_to_complete_output_s']:.3f}s "
                f"wall={time.perf_counter()-t_w:.1f}s (DISCARDED)")

        # ---- TIMED run (NO profiler): the reported single-request metrics ----
        log(f"[timed] clean timed run (no profiler): N={args.profile_steps} cfg={args.cfg} ...")
        timed = run_one(pipeline, saved, args.profile_steps, device, "timed", cfg_on)
        log(f"[timed] complete={timed['time_to_complete_output_s']:.3f}s "
            f"prefill={timed['prefill_s']:.3f}s ttfo={timed['time_to_first_output_s']:.3f}s "
            f"peak_alloc={timed['peak_xpu_alloc_mib']:.0f}MiB "
            f"finite(v/a)={timed['video_finite']}/{timed['action_finite']}")

        # ---- PROFILED run: rank0 wraps profiler; all ranks execute collective ----
        acts, has_xpu = _activities()
        ka = None
        if IS_RANK0:
            prof_kwargs = dict(activities=acts, record_shapes=True, with_stack=False, profile_memory=True)
            log(f"[profile] profiling ONE warm run on rank0: N={args.profile_steps} ...")
            with torch.profiler.profile(**prof_kwargs) as prof:
                prof_res = run_one(pipeline, saved, args.profile_steps, device, "prof", cfg_on)
            ka = prof.key_averages(group_by_input_shape=False)
            log(f"[profile] profiled run complete={prof_res['time_to_complete_output_s']:.3f}s (has profiler overhead)")
        else:
            prof_res = run_one(pipeline, saved, args.profile_steps, device, "prof", cfg_on)
        barrier("post-prof")

        # ---- REPEAT: same input, N sequential requests, NO profiler ----
        repeats = []
        if args.repeat > 0:
            log(f"[repeat] running {args.repeat}x sequential identical requests (N={args.profile_steps}) ...")
            for i in range(args.repeat):
                r = run_one(pipeline, saved, args.profile_steps, device, f"rep{i}", cfg_on)
                barrier(f"post-rep{i}")
                log(f"[repeat {i+1}/{args.repeat}] complete={r['time_to_complete_output_s']:.3f}s "
                    f"ttfo={r['time_to_first_output_s']:.3f}s peak={r['peak_xpu_alloc_mib']:.0f}MiB")
                repeats.append(r)

        peaks = gather_peaks(timed["peak_xpu_alloc_mib"])

        if IS_RANK0:
            dev_key, cpu_key = _table_sort_keys(ka, has_xpu)
            if dev_key:
                try:
                    (prof_dir / "op_table_self_xpu.txt").write_text(ka.table(sort_by=dev_key, row_limit=50))
                    log("[profile] wrote op_table_self_xpu.txt")
                except Exception as e:
                    log(f"[profile] xpu table failed: {e}")
            try:
                (prof_dir / "op_table_self_cpu.txt").write_text(ka.table(sort_by=cpu_key, row_limit=50))
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
            all_rows = _events_to_rows(ka, has_xpu, topn=100000)
            total_self_dev = sum(r.get("self_device_time_total_us", 0.0) for r in all_rows)
            total_self_cpu = sum(r["self_cpu_time_total_us"] for r in all_rows)

            try:
                _dp = torch.xpu.get_device_properties(LOCAL_RANK)
                dev_str = f"xpu:{LOCAL_RANK} ({_dp.name}, {_dp.total_memory // 1024 // 1024} MiB)"
            except Exception:
                dev_str = f"xpu:{LOCAL_RANK}"

            valid_peaks = [p for p in peaks if p is not None]
            rep_complete = [r["time_to_complete_output_s"] for r in repeats]
            rep_ttfo = [r["time_to_first_output_s"] for r in repeats]
            repeat_stats = None
            if rep_complete:
                arr = np.array(rep_complete)
                repeat_stats = {
                    "n": len(arr), "values_s": rep_complete, "ttfo_values_s": rep_ttfo,
                    "mean_s": float(arr.mean()), "std_s": float(arr.std()),
                    "min_s": float(arr.min()), "max_s": float(arr.max()),
                    "median_s": float(np.median(arr)),
                    "p90_s": float(np.percentile(arr, 90)),
                }

            summary = {
                "test": "dreamzero_dit_only_TP_compile_cfg",
                "tp_size": get_tensor_model_parallel_world_size(),
                "world_size": WORLD_SIZE,
                "device_rank0": dev_str,
                "cards": os.environ.get("ZE_AFFINITY_MASK", "unset"),
                "torch_version": torch.__version__,
                "python": platform.python_version(),
                "cfg": args.cfg,
                "compile": args.compile,
                "compile_info": compile_info,
                "offload_mode": "full_resident",
                "dit_block_shard_mib_per_rank": shard_mib,
                "num_dit_blocks": num_blocks,
                "num_dit_weight_params_rank0": len(loaded),
                "profile_steps": args.profile_steps,
                "warmup_steps": args.warmup_steps,
                "repeat_count": args.repeat,
                "measurement": "warm",
                "model_load_s": model_load_s,
                "timing_run": timed,
                "profiled_run": prof_res,
                "repeat_runs": repeats,
                "repeat_stats": repeat_stats,
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
                    "time_to_first_output_s": "prefill + first denoise step (first DiT latent output; no VAE decode)",
                    "time_to_complete_output_s": "prefill + full N-step denoise loop (DiT latents+action; no MP4/VAE decode)",
                    "decode_time_s": "null: VAE decode not run in DiT-only test",
                    "cfg_off_semantics": "no classifier-free guidance; raw conditional prediction, numerically different output than cfg=on",
                    "repeat_runs": "10 sequential identical single-request calls on the SAME warm pipeline/session-reset each time -- a completion-time distribution, not a batched-throughput test",
                    "timing_vs_profiled": "timing_run has NO profiler; profiled_run wall time includes profiler overhead",
                },
            }
            (prof_dir / "profile_summary.json").write_text(json.dumps(summary, indent=2))
            log("=========== TP+COMPILE+CFG PROFILE SUMMARY (rank0) ===========")
            log(json.dumps({k: v for k, v in summary.items() if k not in ("profiler", "repeat_runs")}, indent=2))
            log(f"SAVED_PROFILE_DIR={prof_dir}")
    barrier("done")
    log("DONE")


if __name__ == "__main__":
    main()
