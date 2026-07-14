# SPDX-License-Identifier: Apache-2.0
"""Auto-imported by every Python interpreter that has this dir on PYTHONPATH.

Workaround for an upstream XPU-portability bug: ARDiffusionModelRunner.
_preallocate_kv_cache() (vllm_omni/experimental/ar_diffusion/runner.py) calls
``torch.cuda.mem_get_info(self.device)`` unconditionally -- this raises
``ValueError: Expected a cuda device, but got: xpu:N`` on any XPU run through
the AR-Diffusion engine (which DreamZero's denoise path requires). Every other
device access in that file is already platform-guarded (see the
``torch.cuda.is_available()`` check two lines below the buggy call); this one
call was missed.

This patches torch.cuda.mem_get_info to redirect XPU devices to
torch.xpu.mem_get_info, which has the identical (free_bytes, total_bytes)
return signature. CUDA devices/None are passed through unchanged. This is a
test-harness workaround only -- NOT a fix to the installed package -- so it is
scoped as narrowly as possible (redirect only, no other behavior change) and
logged loudly so it is never silently relied upon.

Because DiffusionWorker processes are spawned fresh (multiprocessing "spawn"),
an in-process monkeypatch in the parent script would not propagate to them;
placing this file on PYTHONPATH makes Python's site module auto-import it in
every worker process at interpreter startup.
"""
from __future__ import annotations

import sys

try:
    import torch

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
        print(
            "[sitecustomize] patched torch.cuda.mem_get_info to redirect xpu "
            "devices to torch.xpu.mem_get_info (workaround for ar_diffusion "
            "runner.py:199 hardcoded CUDA call)",
            file=sys.stderr,
        )
except Exception as _exc:  # pragma: no cover - must never break interpreter startup
    print(f"[sitecustomize] mem_get_info patch failed (non-fatal): {_exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Workaround 2: ARDiffusionModelRunner.execute_model() signature drift.
#
# The base DiffusionModelRunner.execute_model(req, kv_prefetch_jobs=None)
# accepts kv_prefetch_jobs, but ARDiffusionModelRunner.execute_model(self, req)
# -- an override in this checked-out revision -- was never updated to accept
# it, so any call site passing kv_prefetch_jobs=... raises "unexpected keyword
# argument". Patch: accept **kwargs and forward it only on the KV-disabled
# fallback branch (where it calls the base class directly); the AR-KV session
# branch below it never used kv_prefetch_jobs in the first place (session KV
# and KV-connector prefetch are two unrelated mechanisms), so kwargs is simply
# not forwarded there -- identical behavior to the unpatched branch, just no
# longer crashing on the extra keyword.
# ---------------------------------------------------------------------------
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
        print(
            "[sitecustomize] patched ARDiffusionModelRunner.execute_model to accept "
            "kv_prefetch_jobs kwarg (upstream override signature drift workaround)",
            file=sys.stderr,
        )
except Exception as _exc:  # pragma: no cover - must never break interpreter startup
    print(f"[sitecustomize] execute_model patch failed (non-fatal): {_exc}", file=sys.stderr)
