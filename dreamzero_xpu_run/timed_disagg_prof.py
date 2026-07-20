#!/usr/bin/env python3
"""DreamZero DISAGGREGATED timed run on XPU — DEEP per-process profiling.

Config (this run): pipeline=dreamzero_disaggregated, encode on card 0, denoise
TP=4 on cards 4,5,6,7, DECODE ON CARD 3 (separate device, not co-resident with
encode), cfg_scale=1.0 (CFG OFF), num_inference_steps=16, denoise inductor ON
(enforce_eager False), raw 3-camera mp4 inputs, MULTI-REQUEST (one session).

Goal: the per-request omni.generate wall time (~17-24s) is longer than the sum
of DiT + encode + decode compute. Decompose it to find the drag. Three layers:

  1. Orchestrator-side per-stage timeline: for each request, when it ENTERS and
     EXITS each stage (stage_submit_ts / stage_gen_time_ms already computed by
     the orchestrator) — surfaced by scraping the orchestrator INFO logs and, as
     the reliable signal, the per-process phase events below.
  2. Per-PROCESS phase events: sitecustomize (on PYTHONPATH, auto-imported in
     every spawned worker) writes events.<pid>.jsonl with absolute-wall-clock
     start/end for encode/diffuse/postprocess and their sub-phases, plus per-DiT
     step count/self-time and TP all-reduce call counts. Aligning encode-end to
     denoise-start across processes (shared host clock) yields the inter-stage
     GAP = transport + queue-wait, which is exactly what the single generate wall
     time hides.
  3. Server-side denoise E2E via the ar_diffusion_perf_stats RPC (per-request
     forward time INSIDE the denoise worker, excludes engine<->worker IPC), and
     per-stage peak memory via gpu_mem_stats RPC.

Also keeps the torch.profiler worker traces (omni.start_profile) for a kernel-
level view on the measured requests.
"""
import time, uuid, sys, os, json, hashlib, glob
from pathlib import Path
import numpy as np, torch

# ---- XPU compat shims (belt-and-suspenders; sitecustomize also installs them) ----
if not torch.cuda.is_available() and torch.xpu.is_available():
    torch.cuda.mem_get_info = lambda device=None: torch.xpu.mem_get_info(device)

from vllm_omni.experimental.ar_diffusion.runner import ARDiffusionModelRunner
if not getattr(ARDiffusionModelRunner.execute_model, "__wrapped_for_kwargs__", False):
    _orig_exec = ARDiffusionModelRunner.execute_model
    def _exec_compat(self, req, kv_prefetch_jobs=None):
        return _orig_exec(self, req)
    ARDiffusionModelRunner.execute_model = _exec_compat

from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline
if not torch.cuda.is_available() and torch.xpu.is_available():
    for _m in ('setup_compile', 'warmup_compile'):
        _orig = getattr(DreamZeroPipeline, _m, None)
        if _orig is None or getattr(_orig, '__cuda_spoofed__', False):
            continue
        def _mk(fn):
            def _w(self, *a, **k):
                _r = torch.cuda.is_available
                torch.cuda.is_available = lambda: True
                try:
                    return fn(self, *a, **k)
                finally:
                    torch.cuda.is_available = _r
            _w.__cuda_spoofed__ = True
            return _w
        setattr(DreamZeroPipeline, _m, _mk(_orig))

sys.argv = ['x']
import export_prediction_video as E
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = 'GEAR-Dreams/DreamZero-DROID'
DEPLOY = Path(os.environ['DZ_DEPLOY'])
VIDEO_DIR = Path('/workspace/vllm-omni/outputs/dreamzero/assets')
PROMPT = 'Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan'
MODE = os.environ.get('DZ_MODE', 'multi')
NUM_CHUNKS = int(os.environ.get('DZ_NUM_CHUNKS', '6'))
METRICS = Path(os.environ.get('DZ_METRICS', '/workspace/out/metrics'))
PROF_DIR = os.environ.get('DZ_PROF_DIR', '')
METRICS.mkdir(parents=True, exist_ok=True)


def xsync():
    try:
        torch.xpu.synchronize()
    except Exception:
        pass


def reset_peak():
    try:
        for i in range(torch.xpu.device_count()):
            torch.xpu.reset_peak_memory_stats(i)
    except Exception:
        pass


def act_hash(o):
    a = o.multimodal_output.get('actions') if getattr(o, 'multimodal_output', None) else None
    if a is None:
        return None
    arr = np.asarray(a, dtype=np.float64)
    return {'sha': hashlib.sha1(arr.tobytes()).hexdigest()[:12], 'mean': float(arr.mean()), 'shape': list(arr.shape)}


def _rpc(omni, method, n_stages, **kw):
    try:
        return omni.engine.collective_rpc(method=method, stage_ids=list(range(n_stages)), **kw)
    except Exception as e:
        return {'error': repr(e)[:300]}


def main():
    sid = 'disagg-' + MODE + '-' + str(uuid.uuid4())
    n_chunks = 1 if MODE == 'single' else NUM_CHUNKS
    camera_frames, observations = E._build_observations(
        video_dir=VIDEO_DIR, prompt=PROMPT, session_id=sid,
        num_chunks=n_chunks, repeat_chunk_observations=True)
    print('[cfg] MODE=%s n_observations=%d deploy=%s prof_dir=%s' % (MODE, len(observations), DEPLOY, PROF_DIR), flush=True)

    reset_peak()
    tl0 = time.perf_counter()
    omni = Omni(model=MODEL, deploy_config=str(DEPLOY), enforce_eager=False, worker_extension_cls=E.WORKER_EXTENSION)
    xsync()
    tl1 = time.perf_counter()
    print('[timing] MODEL_LOAD_S=%.3f' % (tl1 - tl0), flush=True)

    n_stages = getattr(omni, 'num_stages', None) or getattr(getattr(omni, 'engine', None), 'num_stages', 1)
    print('[cfg] num_stages=%s' % n_stages, flush=True)

    prof_enabled = os.environ.get('DZ_PROFILE', '1') == '1'
    gen = []
    outs = []
    hashes = []
    req_marks = []   # orchestrator-side per-request wall clock (absolute) start/end

    def _run_one(i, obs):
        sp0 = OmniDiffusionSamplingParams(extra_args={'reset': i == 0, 'session_id': obs['session_id'], 'robot_obs': obs})
        if n_stages and n_stages > 1:
            sp_list = [sp0] + [OmniDiffusionSamplingParams(extra_args={'session_id': obs['session_id']})
                               for _ in range(n_stages - 1)]
        else:
            sp_list = [sp0]
        w0 = time.time()
        g0 = time.perf_counter()
        res = omni.generate(obs['prompt'], sampling_params_list=sp_list)
        xsync()
        g1 = time.perf_counter()
        w1 = time.time()
        if not res:
            raise RuntimeError('no output at request %d' % i)
        gen.append(g1 - g0)
        req_marks.append({'req': i, 'wall_start': w0, 'wall_end': w1, 'dur_s': g1 - g0})
        outs.append(res[0])
        hashes.append(act_hash(res[0]))
        tag = 'WARMUP(compile)' if i == 0 else ('WARM' if (MODE == 'single' and i == 1) else 'chunk%d' % i)
        print('[gen] req=%d %s wall=[%.6f,%.6f] time_s=%.3f act=%s' % (i, tag, w0, w1, g1 - g0, hashes[-1]), flush=True)

    # Reset server-side denoise perf list after warmup so the measured requests are clean.
    worker_profiled = False
    for i, obs in enumerate(observations):
        if i == 1:
            # drain warmup denoise timings, then (optionally) start worker torch.profiler
            _rpc(omni, 'ar_diffusion_perf_stats', n_stages, reset=True)
            if prof_enabled:
                try:
                    omni.start_profile(profile_prefix='disagg_%s' % MODE)
                    worker_profiled = True
                    print('[profile] worker profiler started before req 1', flush=True)
                except Exception as e:
                    print('[profile] start_profile failed: %s' % repr(e)[:200], flush=True)
        _run_one(i, obs)

    if worker_profiled:
        try:
            omni.stop_profile()
            print('[profile] worker profiler stopped', flush=True)
        except Exception as e:
            print('[profile] stop_profile failed: %s' % repr(e)[:200], flush=True)

    # Server-side denoise E2E per request (excludes engine<->worker IPC).
    server_perf = _rpc(omni, 'ar_diffusion_perf_stats', n_stages, reset=False)
    print('[server_perf] %s' % json.dumps(server_perf)[:800], flush=True)

    # Per-stage peak XPU memory (host xpu-smi reads 0 under ZE_AFFINITY_MASK).
    peak_mem = _rpc(omni, 'gpu_mem_stats', n_stages)
    print('[peakmem] %s' % json.dumps(peak_mem)[:800], flush=True)

    nonfirst = [h['sha'] for h in hashes[1:] if h]
    collapsed = len(nonfirst) >= 2 and len(set(nonfirst)) == 1
    result = {
        'mode': MODE, 'session_id': sid, 'n_requests': len(observations),
        'device_layout': {'encode': 'phys card 0', 'denoise_TP4': 'phys cards 4,5,6,7',
                          'decode': 'phys card 3 (separate)', 'ZE_AFFINITY_MASK': os.environ.get('ZE_AFFINITY_MASK')},
        'model_load_s': tl1 - tl0,
        'per_request_gen_s': gen,
        'req_wall_marks': req_marks,
        'warm_request_s': (gen[1] if MODE == 'single' and len(gen) > 1 else None),
        'mean_chunk_gen_s': (float(np.mean(gen[1:])) if len(gen) > 1 else None),
        'action_hashes': hashes,
        'server_denoise_perf': server_perf,
        'nonfirst_outputs_identical_collapse': collapsed,
        'peak_xpu_mem_per_stage': peak_mem,
        'prof_event_dir': PROF_DIR,
        'profiler_enabled': prof_enabled,
        'status': 'success',
    }
    out_path = METRICS / ('result_%s.json' % MODE)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print('[RESULT_JSON]=%s' % out_path, flush=True)
    print('[SUMMARY] mode=%s warm_s=%s mean_chunk_s=%s collapse=%s' %
          (MODE, result['warm_request_s'], result['mean_chunk_gen_s'], collapsed), flush=True)
    try:
        omni.close()
    except Exception:
        pass


if __name__ == '__main__':
    main()
