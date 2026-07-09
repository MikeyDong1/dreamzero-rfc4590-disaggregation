#!/usr/bin/env python3
"""DreamZero DiT-only replay with CFG-toggle + eager/inductor toggle, WARM + profiled.

Same pipeline / K-resident layerwise offload / saved-input replay as
`dit_only_profile.py`, plus two knobs to A/B against the eager+CFG baseline:

  --cfg {on,off}       override the saved do_true_cfg. OFF => predict_noise runs
                       ONLY the positive branch => 1 DiT forward/step instead of 2
                       => halves the per-step H2D block streaming (the dominant
                       cost found in the eager+CFG profile). Output is the raw
                       conditional prediction (NO classifier-free guidance), so it
                       differs numerically from the CFG result -- expected for a
                       perf experiment.
  --compile {eager,inductor}
                       inductor => torch.compile(block.forward, mode="default",
                       fullgraph=True, dynamic=False) on EVERY DiT block BEFORE the
                       offload hooks are applied (mirrors the pipeline's own
                       setup_compile, which is otherwise CUDA-gated and never runs
                       on XPU). Compiling before hooking means the hook captures the
                       COMPILED block as `_omni_original_forward`; prefetch/offload
                       stay eager (they are @torch.compiler.disable). Per-block
                       fullgraph=True -> fullgraph=False -> eager fallback, logged.

Flow: build pipeline -> load DiT weights -> (optional) compile blocks -> K-resident
offload -> throwaway warmup denoise (also triggers block compilation so the timed
runs are WARM) -> CLEAN warm sweep (no profiler) for accurate TTC -> ONE profiled
warm run (torch.profiler CPU+XPU) for the analysis artifacts.

Usage:
  dit_cfg_compile_profile.py --model-path ... --test-data-dir ... --out-dir ...
    [--resident-blocks 10] [--cfg off] [--compile inductor]
    [--clean-steps 4,8,16] [--profile-steps 16] [--warmup-steps 4]
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
    return time.time()


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
    """Compile each DiT block.forward with inductor BEFORE offload hooks are applied.

    Mirrors DreamZeroPipeline.setup_compile (which is CUDA-gated). Per-block:
    fullgraph=True -> on failure retry fullgraph=False -> on failure leave eager.
    Returns a summary dict.
    """
    if mode != "inductor":
        log("[compile] mode=eager (no torch.compile)")
        return {"mode": "eager", "compiled_fullgraph": 0, "compiled_partial": 0, "eager": len(dit.blocks)}

    # NOTE: fullgraph=False is REQUIRED here. The DiT self-attention calls the XPU
    # FlashAttention-2 custom op `_vllm_fa2_C::varlen_fwd`, which has NO fake/meta
    # kernel registered, so Dynamo cannot trace THROUGH it. fullgraph=True raises
    # "Operator does not support running with fake tensors" at first call. With
    # fullgraph=False, Dynamo inserts a graph break at the attention op and still
    # compiles+fuses everything around it (QKV/O/FFN projections, layernorms, RoPE,
    # the elementwise/cat/gelu tail) — which is the bulk of the eager launch count.
    # torch.compile defers tracing to first call, so failures surface during warmup,
    # not here; fullgraph=False avoids them entirely.
    partial = 0
    for i, block in enumerate(dit.blocks):
        block.forward = torch.compile(block.forward, mode="default", fullgraph=False, dynamic=False)
        partial += 1
    log(f"[compile] mode=inductor (fullgraph=False, graph-break at attention custom op): "
        f"{partial} blocks wrapped (of {len(dit.blocks)})")
    return {"mode": "inductor", "fullgraph": False, "compiled_partial": partial, "eager": 0,
            "note": "graph break at _vllm_fa2_C::varlen_fwd (no fake impl)"}


def enable_k_resident_offload(dit, device, resident_blocks, pin_memory=True):
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
    return hooks, K


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
    state = pipeline._get_or_create_state(f"dit-cc-{tag}")
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


def run_one(pipeline, saved, num_steps, device, tag, cfg_override: bool):
    state = _setup_state(pipeline, saved, tag)
    try:
        current_omni_platform.reset_peak_memory_stats()
    except Exception:
        pass
    current_omni_platform.synchronize()
    log(f"RUN_START N={num_steps} epoch={now():.3f}")

    do_cfg = cfg_override
    neg = saved["negative_prompt_embeds"] if do_cfg else None

    t_pref0 = time.perf_counter()
    with torch.no_grad():
        pipeline._prefill_kv_cache(
            saved["image_latents"], saved["prompt_embeds"], neg,
            saved["frame_seqlen"], saved["prefill_seq_len"], do_cfg, state,
        )
    current_omni_platform.synchronize()
    prefill_s = time.perf_counter() - t_pref0
    log(f"[N={num_steps}] PREFILL_S={prefill_s:.3f}")

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
            video_action_scheduler=vas, do_true_cfg=do_cfg,
            state=state, seq_len=saved["diffuse_seq_len"],
            state_features=saved["state_features"], embodiment_id=saved["embodiment_id"],
        )
    current_omni_platform.synchronize()
    loop_s = time.perf_counter() - t_loop0
    log(f"RUN_END N={num_steps} epoch={now():.3f}")

    ttfo = prefill_s + (first_output_wall["t"] - t_loop0) if first_output_wall["t"] else None
    v = video_out.detach().float().cpu().numpy()
    a = action_out.detach().float().cpu().numpy()
    peak = current_omni_platform.max_memory_allocated() / 1024**3
    try:
        peak_reserved = torch.xpu.max_memory_reserved() / 1024**3
    except Exception:
        peak_reserved = None
    return {
        "num_steps": num_steps, "cfg": do_cfg, "prefill_s": prefill_s, "denoise_loop_s": loop_s,
        "time_to_first_output_s": ttfo, "time_to_complete_output_s": prefill_s + loop_s,
        "per_step_s": step_times, "video_out_shape": list(v.shape), "action_out_shape": list(a.shape),
        "video_finite": bool(np.isfinite(v).all()), "action_finite": bool(np.isfinite(a).all()),
        "video_mean": float(v.mean()), "video_std": float(v.std()),
        "action_min": float(a.min()), "action_max": float(a.max()),
        "peak_xpu_alloc_gib": peak, "peak_xpu_reserved_gib": peak_reserved,
    }


# ---- profiler helpers (same as dit_only_profile.py) ----
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
        rows.append(row)
    keyfn = (lambda r: r.get("self_device_time_total_us", 0.0)) if has_xpu \
        else (lambda r: r["self_cpu_time_total_us"])
    rows.sort(key=keyfn, reverse=True)
    return rows[:topn]


def _table_dev_key(ka, has_xpu):
    if has_xpu and len(ka):
        for cand in ("self_xpu_time_total", "self_device_time_total"):
            if hasattr(ka[0], cand):
                return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--test-data-dir", required=True)
    ap.add_argument("--out-dir", default="/workspace/out/dit_cfg_compile")
    ap.add_argument("--resident-blocks", type=int, default=10)
    ap.add_argument("--cfg", choices=["on", "off"], default="off")
    ap.add_argument("--compile", choices=["eager", "inductor"], default="inductor")
    ap.add_argument("--clean-steps", default="4,8,16", help="clean (non-profiled) warm sweep for TTC")
    ap.add_argument("--profile-steps", type=int, default=16, help="N for the single profiled run")
    ap.add_argument("--warmup-steps", type=int, default=4)
    args = ap.parse_args()

    cfg_on = (args.cfg == "on")
    clean_steps = [int(s) for s in args.clean_steps.split(",") if s.strip()]
    device = torch.device(DEVICE, 0)
    out_dir = Path(args.out_dir)
    prof_dir = out_dir / "profile"
    prof_dir.mkdir(parents=True, exist_ok=True)

    od_config = OmniDiffusionConfig(
        model=args.model_path, dtype=torch.bfloat16,
        enforce_eager=True, enable_layerwise_offload=True,
    )
    vllm_config = bootstrap(device)

    summary: dict = {}
    with set_forward_context(vllm_config=vllm_config, omni_diffusion_config=od_config), \
         set_current_diffusion_config(od_config):
        t_load0 = time.perf_counter()
        log("[load] constructing DreamZeroPipeline (CPU) ...")
        pipeline = DreamZeroPipeline(od_config=od_config)
        pipeline.eval()
        pipeline.transformer.to(dtype=torch.bfloat16)
        log("[load] streaming DiT weights (action_head.model.*) ...")
        loaded = pipeline.load_weights(stream_dit_weights(args.model_path))
        log(f"[load] loaded {len(loaded)} DiT weights")

        # ---- Compile blocks BEFORE offload (so hooks capture compiled forward) ----
        compile_info = maybe_compile_blocks(pipeline.transformer, args.compile)

        # ---- K-resident layerwise offload ----
        _hooks, K = enable_k_resident_offload(pipeline.transformer, device,
                                              resident_blocks=args.resident_blocks)
        current_omni_platform.synchronize()
        model_load_s = time.perf_counter() - t_load0
        log(f"[load] MODEL_LOAD_S={model_load_s:.3f} (K={K}, cfg={args.cfg}, compile={args.compile})")

        saved = load_saved_inputs(Path(args.test_data_dir), device)
        log(f"[input] saved DiT inputs loaded (saved do_true_cfg={saved['do_true_cfg']}, using cfg={cfg_on})")

        # ---- Warmup (discarded); ALSO triggers per-block inductor compilation ----
        if args.warmup_steps > 0:
            log(f"[warmup] throwaway {args.warmup_steps}-step denoise (also compiles blocks if inductor) ...")
            t_w = time.perf_counter()
            _w = run_one(pipeline, saved, args.warmup_steps, device, "warmup", cfg_on)
            log(f"[warmup] done complete={_w['time_to_complete_output_s']:.3f}s wall={time.perf_counter()-t_w:.1f}s (DISCARDED)")

        # ---- CLEAN warm sweep (no profiler) for accurate TTC ----
        clean_runs = []
        for n in clean_steps:
            r = run_one(pipeline, saved, n, device, f"clean{n}", cfg_on)
            log(f"=== CLEAN N={n}: prefill={r['prefill_s']:.3f}s ttfo={r['time_to_first_output_s']:.3f}s "
                f"complete={r['time_to_complete_output_s']:.3f}s per_step_mean={np.mean(r['per_step_s']):.3f}s "
                f"peak_alloc={r['peak_xpu_alloc_gib']:.3f}GiB finite(v/a)={r['video_finite']}/{r['action_finite']}")
            clean_runs.append(r)

        # ---- PROFILED warm run ----
        acts, has_xpu = _activities()
        log(f"[profile] profiling ONE warm run: N={args.profile_steps} (cfg={args.cfg}, compile={args.compile}) ...")
        prof_kwargs = dict(activities=acts, record_shapes=True, with_stack=False, profile_memory=True)
        with torch.profiler.profile(**prof_kwargs) as prof:
            prof_res = run_one(pipeline, saved, args.profile_steps, device, "prof", cfg_on)
        log(f"[profile] run done: complete={prof_res['time_to_complete_output_s']:.3f}s "
            f"peak_alloc={prof_res['peak_xpu_alloc_gib']:.3f}GiB "
            f"finite(v/a)={prof_res['video_finite']}/{prof_res['action_finite']}")

        ka = prof.key_averages(group_by_input_shape=False)
        dev_key = _table_dev_key(ka, has_xpu)
        if dev_key:
            try:
                (prof_dir / "op_table_self_xpu.txt").write_text(ka.table(sort_by=dev_key, row_limit=50))
                log("[profile] wrote op_table_self_xpu.txt")
            except Exception as e:
                log(f"[profile] xpu table failed: {e}")
        try:
            (prof_dir / "op_table_self_cpu.txt").write_text(ka.table(sort_by="self_cpu_time_total", row_limit=50))
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
            _dp = torch.xpu.get_device_properties(0)
            dev_str = f"xpu:0 (card0, {_dp.name}, {_dp.total_memory // 1024 // 1024} MiB)"
        except Exception:
            dev_str = "xpu:0 (card0)"

        summary = {
            "device": dev_str, "torch_version": torch.__version__, "python": platform.python_version(),
            "config": {"cfg": args.cfg, "compile": args.compile, "resident_blocks_K": K,
                       "num_dit_blocks": len(pipeline.transformer.blocks), "warmup_steps": args.warmup_steps,
                       "measurement": "warm", "saved_do_true_cfg": saved["do_true_cfg"]},
            "compile_info": compile_info,
            "model_load_s": model_load_s,
            "num_dit_weights_loaded": len(loaded),
            "clean_runs": clean_runs,
            "profiled_run": {"num_steps": args.profile_steps, "metrics": prof_res},
            "profiler": {"activities": [str(a) for a in acts], "has_xpu_activity": has_xpu,
                         "device_sort_key": dev_key,
                         "total_self_device_time_us": total_self_dev,
                         "total_self_cpu_time_us": total_self_cpu, "top_ops": top_rows},
        }

    (prof_dir / "profile_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "cfg_compile_results.json").write_text(
        json.dumps({k: v for k, v in summary.items() if k != "profiler"}, indent=2))
    log("=========== SUMMARY ===========")
    log(json.dumps({k: v for k, v in summary.items() if k not in ("profiler",)}, indent=2))
    log(f"SAVED_PROFILE_DIR={prof_dir}")
    log("DONE")


if __name__ == "__main__":
    main()
