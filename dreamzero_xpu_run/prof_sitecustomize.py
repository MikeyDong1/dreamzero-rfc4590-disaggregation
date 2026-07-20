#!/usr/bin/env python3
"""sitecustomize.py — auto-imported at interpreter startup by EVERY Python
process that has this directory on PYTHONPATH.

Two responsibilities, both of which MUST run inside each spawned DiffusionWorker
child (workers are launched with multiprocessing 'spawn', so parent-process
monkeypatches do NOT propagate — see
vllm_omni/diffusion/executor/multiproc_executor.py _launch_workers
mp.set_start_method('spawn', force=True)):

  A. XPU compatibility shims required for the AR-Diffusion denoise engine +
     torch.compile/inductor to run on Intel XPU (3 known upstream CUDA-isms).

  B. Fine-grained per-PROCESS phase profiling of the disaggregated DreamZero
     pipeline. Every wrapped method emits ONE JSON record per call to
     $DZ_PROF_DIR/events.<pid>.jsonl with:
        pid, role (encode/denoise/decode/monolithic), method, seq,
        t_start (ABSOLUTE time.time()), t_end (absolute), dur_s (perf_counter,
        device-synchronized for coarse phase boundaries).
     Absolute timestamps share the single host wall clock across all processes,
     so an offline pass can align encode-end vs denoise-start and read the
     inter-stage GAP (transport + queue-wait) directly — the thing the
     per-omni.generate wall time hides.

Distortion control: coarse phase boundaries (encode/diffuse/postprocess and
their sub-phases) synchronize the device before stopping their timer (a handful
of syncs per request — negligible). The per-DiT-step forward (predict_noise) is
timed WITHOUT a forced sync per call (it would serialize the pipeline); its
count and cumulative self-time are recorded. TP all-reduce collectives are
COUNTED only (no per-call sync/timing) to confirm collective VOLUME without
distorting it; their wall cost comes from the torch.profiler kernel trace.

Everything is guarded so a failure here can never break interpreter startup or
the run; profiling is fully disabled unless DZ_PROF_DIR is set.
"""
from __future__ import annotations

import os
import sys
import json
import time
import threading

# ===========================================================================
# A. XPU compatibility shims (must load before any AR-Diffusion worker runs)
# ===========================================================================
try:
    import torch

    # Shim 1: torch.cuda.mem_get_info -> torch.xpu.mem_get_info for xpu devices.
    _orig_cuda_mem_get_info = torch.cuda.mem_get_info

    def _xpu_safe_mem_get_info(device=None):
        dev = None
        if device is not None:
            dev = device if isinstance(device, torch.device) else torch.device(device)
        if dev is not None and dev.type == "xpu":
            return torch.xpu.mem_get_info(dev)
        return _orig_cuda_mem_get_info(device)

    if torch.cuda.mem_get_info is not _xpu_safe_mem_get_info:
        torch.cuda.mem_get_info = _xpu_safe_mem_get_info
        print("[sitecustomize] patched torch.cuda.mem_get_info -> xpu", file=sys.stderr)
except Exception as _exc:  # pragma: no cover
    print(f"[sitecustomize] mem_get_info patch failed (non-fatal): {_exc}", file=sys.stderr)

# Shim 2: ARDiffusionModelRunner.execute_model signature drift (kv_prefetch_jobs).
try:
    from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
    from vllm_omni.experimental.ar_diffusion.runner import ARDiffusionModelRunner

    _orig_ar_execute_model = ARDiffusionModelRunner.execute_model

    def _ar_execute_model_kwargs_compat(self, req, **kwargs):
        if self.kv_cache is None:
            return DiffusionModelRunner.execute_model(self, req, **kwargs)
        return _orig_ar_execute_model(self, req)

    if not getattr(ARDiffusionModelRunner.execute_model, "__wrapped_for_kwargs__", False):
        _ar_execute_model_kwargs_compat.__wrapped_for_kwargs__ = True
        ARDiffusionModelRunner.execute_model = _ar_execute_model_kwargs_compat
        print("[sitecustomize] patched ARDiffusionModelRunner.execute_model kwargs", file=sys.stderr)
except Exception as _exc:  # pragma: no cover
    print(f"[sitecustomize] execute_model patch failed (non-fatal): {_exc}", file=sys.stderr)

# Shim 3: spoof torch.cuda.is_available()=True around setup_compile/warmup_compile
# so inductor actually engages on XPU.
try:
    from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline as _DZP_compile

    if not torch.cuda.is_available() and torch.xpu.is_available():
        for _m in ("setup_compile", "warmup_compile"):
            _orig = getattr(_DZP_compile, _m, None)
            if _orig is None or getattr(_orig, "__cuda_spoofed__", False):
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

            setattr(_DZP_compile, _m, _mk(_orig))
        print("[sitecustomize] patched setup_compile/warmup_compile cuda spoof", file=sys.stderr)
except Exception as _exc:  # pragma: no cover
    print(f"[sitecustomize] compile spoof patch failed (non-fatal): {_exc}", file=sys.stderr)


# ===========================================================================
# B. Per-process phase profiler
# ===========================================================================
_PROF_DIR = os.environ.get("DZ_PROF_DIR", "")

if _PROF_DIR:
    try:
        os.makedirs(_PROF_DIR, exist_ok=True)
    except Exception:
        pass

    _PID = os.getpid()
    _EV_PATH = os.path.join(_PROF_DIR, f"events.{_PID}.jsonl")
    _EV_LOCK = threading.Lock()
    _EV_FH = None
    _SEQ = {}

    def _fh():
        global _EV_FH
        if _EV_FH is None:
            _EV_FH = open(_EV_PATH, "a", buffering=1)  # line-buffered
        return _EV_FH

    def _emit(rec):
        try:
            with _EV_LOCK:
                _fh().write(json.dumps(rec) + "\n")
        except Exception:
            pass

    def _xsync():
        try:
            import torch as _t
            if _t.xpu.is_available():
                _t.xpu.synchronize()
        except Exception:
            pass

    def _role(self):
        try:
            return str(getattr(self, "_model_stage", "?"))
        except Exception:
            return "?"

    def _next_seq(method):
        n = _SEQ.get(method, 0)
        _SEQ[method] = n + 1
        return n

    def _wrap_phase(cls, method_name, *, sync=True, drain_counters=False):
        """Wrap a bound method with an absolute-timestamped, device-synced timer.

        sync=True  -> torch.xpu.synchronize() before stopping (coarse phases).
        All wrapped methods here run in EAGER land (outside the compiled DiT
        graph), so the timing wrapper is safe under torch.compile.
        """
        orig = getattr(cls, method_name, None)
        if orig is None or getattr(orig, "__dz_profiled__", False):
            return

        def _w(self, *a, **k):
            role = _role(self)
            seq = _next_seq(method_name)
            t0_wall = time.time()
            t0 = time.perf_counter()
            try:
                return orig(self, *a, **k)
            finally:
                if sync:
                    _xsync()
                t1 = time.perf_counter()
                t1_wall = time.time()
                _emit({
                    "pid": _PID, "role": role, "method": method_name, "seq": seq,
                    "t_start": t0_wall, "t_end": t1_wall, "dur_s": t1 - t0,
                })

        _w.__dz_profiled__ = True
        setattr(cls, method_name, _w)

    def _wrap_diffuse(cls):
        """Denoise entrypoint: coarse timer + csf/reset signal (bug #1 witness).

        Wraps the eager `diffuse` atom (NOT the compiled transformer inside it),
        so it is torch.compile-safe.
        """
        orig = getattr(cls, "diffuse", None)
        if orig is None or getattr(orig, "__dz_profiled__", False):
            return

        def _w(self, state, *a, **k):
            role = _role(self)
            seq = _next_seq("diffuse")
            carrier = None
            try:
                carrier = state.extra.get(self._CARRIER_KEY)
            except Exception:
                pass
            csf_in = getattr(carrier, "current_start_frame", None) if carrier is not None else None
            reset_reason = getattr(carrier, "reset_reason", None) if carrier is not None else None
            t0_wall = time.time()
            t0 = time.perf_counter()
            try:
                return orig(self, state, *a, **k)
            finally:
                _xsync()
                t1 = time.perf_counter()
                t1_wall = time.time()
                csf_after = None
                try:
                    csf_after = getattr(getattr(self, "state", None), "current_start_frame", None)
                except Exception:
                    pass
                _emit({
                    "pid": _PID, "role": role, "method": "diffuse", "seq": seq,
                    "t_start": t0_wall, "t_end": t1_wall, "dur_s": t1 - t0,
                    "carrier_csf_in": csf_in, "reset_reason": reset_reason,
                    "denoise_state_csf_after": csf_after,
                })

        _w.__dz_profiled__ = True
        setattr(cls, "diffuse", _w)

    try:
        from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline as _DZP

        # ---- Coarse per-request STAGE boundaries (device-synced) ----
        _wrap_phase(_DZP, "encode", sync=True)
        _wrap_diffuse(_DZP)
        _wrap_phase(_DZP, "postprocess", sync=True)

        # ---- Encode sub-phases ----
        _wrap_phase(_DZP, "_encode_text", sync=True)
        _wrap_phase(_DZP, "_encode_image", sync=True)
        _wrap_phase(_DZP, "_encode_observation_latents", sync=True)

        # ---- Denoise sub-phases ----
        _wrap_phase(_DZP, "_kv_populate_cross", sync=True)
        _wrap_phase(_DZP, "_prefill_kv_cache", sync=True)
        _wrap_phase(_DZP, "_run_dit_loop", sync=True)

        # ---- Decode sub-phases ----
        _wrap_phase(_DZP, "decode_video_latents", sync=True)
        _wrap_phase(_DZP, "_denormalize_action", sync=False)

        # ---- Transport (producer pack / consumer unpack) ----
        _wrap_phase(_DZP, "pack_stage_state", sync=False)
        _wrap_phase(_DZP, "unpack_stage_state", sync=True)  # includes H2D restore

        # NOTE: per-DiT-step forward (predict_noise) and the TP all-reduce
        # counter are DELIBERATELY NOT patched here. Both execute inside the
        # torch.compile(fullgraph=True) transformer region; wrapping the
        # module-level tensor_model_parallel_all_reduce in a Python fn that
        # mutates a global dict breaks dynamo's collective special-casing and
        # trips FailOnRecompileLimitHit. Per-step / per-collective detail comes
        # from the non-intrusive torch.profiler worker trace instead. The coarse
        # phase timers below are all in eager land (outside the compiled graph).

        print(f"[sitecustomize] DreamZero phase profiler armed pid={_PID} -> {_EV_PATH}", file=sys.stderr)
    except Exception as _exc:  # pragma: no cover
        print(f"[sitecustomize] pipeline profiler patch failed (non-fatal): {_exc}", file=sys.stderr)

    # ---- Transport D2H sanitize counter (bytes + time per stage boundary) ----
    try:
        import vllm_omni.diffusion.stage_payload as _sp

        _orig_sanitize = _sp.sanitize_transport_tensor

        def _timed_sanitize(tensor):
            t0 = time.perf_counter()
            try:
                return _orig_sanitize(tensor)
            finally:
                dt = time.perf_counter() - t0
                try:
                    nbytes = int(tensor.numel()) * int(tensor.element_size())
                except Exception:
                    nbytes = -1
                _emit({"pid": _PID, "role": "transport", "method": "sanitize_transport_tensor",
                       "seq": _next_seq("sanitize_transport_tensor"),
                       "t_start": None, "t_end": time.time(), "dur_s": dt, "nbytes": nbytes})

        if not getattr(_sp.sanitize_transport_tensor, "__dz_timed__", False):
            _timed_sanitize.__dz_timed__ = True
            _sp.sanitize_transport_tensor = _timed_sanitize
            print(f"[sitecustomize] transport sanitize timer armed pid={_PID}", file=sys.stderr)
    except Exception as _exc:  # pragma: no cover
        print(f"[sitecustomize] sanitize timer patch failed (non-fatal): {_exc}", file=sys.stderr)
