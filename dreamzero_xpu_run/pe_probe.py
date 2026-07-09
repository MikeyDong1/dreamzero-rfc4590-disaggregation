#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Two focused probes for the parallel-encoder POC on Intel XPU (B60), run inside
the v0240 container (which DOES import vllm_omni, unlike the old VAE-only test).

PROBE A — inductor viability under vllm_omni:
  Does torch.compile(backend="inductor") actually work once `import vllm_omni` has
  run (the POC needs vllm_omni for DistributedAutoencoderKLWan)? The older VAE-only
  report claimed importing vllm_omni disables triton-xpu and breaks inductor. We
  test the REAL vllm_omni VAE encode both eager and inductor-compiled, compare the
  latent for numerical parity, and report wall-clock.

PROBE B — one-card two-stream concurrency:
  The benchmark showed wall == vae_stream + text_stream (perfect serialization). Is
  that fundamental (single-tile serial execution) or fixable (in-order immediate
  command lists preventing cross-stream overlap)? We launch two INDEPENDENT VAE-ish
  compute streams and measure whether their wall-clock is ~max(a,b) (overlap) or
  ~a+b (serial), under whatever SYCL/L0 env the caller sets. Uses matmul chains so
  the test is pure device compute (no host sync inside the timed region).

Usage:
  python pe_probe.py --model-path <snap> --stitched-npz <npz> --which A,B --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# vllm_omni FIRST (mirrors the POC import order + triton-xpu handling).
import vllm_omni  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parallel_encoders as pe  # noqa: E402
import dreamzero_parallel_encoders as dz  # noqa: E402


def log(m):
    print(m, flush=True)


def _sync(dev):
    pe.sync_device(dev)


def probe_A(args, dev):
    """inductor vs eager for the real vllm_omni VAE encode."""
    log("\n===== PROBE A: inductor viability under vllm_omni =====")
    res = {"probe": "A"}
    vae = dz.build_vae(args.model_path)
    dz.load_encoder_weights(dz.build_text_encoder(torch.bfloat16), vae, args.model_path)  # fill vae (text discarded)
    vae.to(device=dev, dtype=torch.float32)
    mean, inv_std = dz.make_vae_buffers(vae)
    mean, inv_std = mean.to(dev), inv_std.to(dev)

    _, _, vae_input, _ = dz.build_inputs(args.model_path, dz.DEFAULT_PROMPT,
                                         "google/umt5-xxl", 352, 640, args.stitched_npz, dev)
    vae_input = vae_input.to(dev)

    def encode_eager():
        return pe.run_vae_encoder(vae_input, vae, mean, inv_std, autocast=True)

    # ---- eager reference ----
    with torch.inference_mode():
        for _ in range(3):
            encode_eager()
        _sync(dev)
        t = time.perf_counter()
        for _ in range(10):
            out_eager = encode_eager()
        _sync(dev)
        eager_ms = (time.perf_counter() - t) / 10 * 1000.0
    log(f"[A] eager: {eager_ms:.1f} ms/encode  latent={tuple(out_eager.shape)} "
        f"finite={bool(torch.isfinite(out_eager.float()).all())}")
    res["eager_ms"] = eager_ms
    res["latent_shape"] = list(out_eager.shape)

    # ---- inductor: compile the vae._encode core (same math path) ----
    input_dtype = vae_input.dtype

    def _encode_core(inp):
        with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16):
            h = vae._encode(inp.to(dtype=vae.dtype))
            mu, _ = h.chunk(2, dim=1)
            mu = (mu - mean.to(mu.dtype)) * inv_std.to(mu.dtype)
            return mu.to(input_dtype)

    try:
        compiled = torch.compile(_encode_core, backend="inductor")
        with torch.inference_mode():
            log("[A] compiling (first inductor call, may take a while)...")
            tc0 = time.perf_counter()
            out_ind = compiled(vae_input)
            _sync(dev)
            compile_s = time.perf_counter() - tc0
            for _ in range(3):
                compiled(vae_input)
            _sync(dev)
            t = time.perf_counter()
            for _ in range(10):
                out_ind = compiled(vae_input)
            _sync(dev)
            ind_ms = (time.perf_counter() - t) / 10 * 1000.0
        max_abs = (out_ind.float() - out_eager.float()).abs().max().item()
        res.update(inductor_ok=True, inductor_ms=ind_ms, compile_s=compile_s,
                   inductor_vs_eager_max_abs_diff=max_abs,
                   speedup=eager_ms / ind_ms if ind_ms else None,
                   inductor_finite=bool(torch.isfinite(out_ind.float()).all()))
        log(f"[A] inductor: {ind_ms:.1f} ms/encode  (compile {compile_s:.1f}s)  "
            f"speedup={eager_ms/ind_ms:.2f}x  max_abs_diff_vs_eager={max_abs:.3e}  "
            f"finite={res['inductor_finite']}")
    except Exception as exc:  # noqa: BLE001
        import traceback
        res.update(inductor_ok=False, error=repr(exc), traceback=traceback.format_exc()[-2000:])
        log(f"[A] inductor FAILED: {exc!r}")
    return res


def probe_B(args, dev):
    """Do two torch.xpu streams overlap on one card, or serialize?"""
    log("\n===== PROBE B: one-card two-stream concurrency =====")
    res = {"probe": "B", "env": {k: os.environ.get(k) for k in
           ("SYCL_UR_USE_LEVEL_ZERO_V2", "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS",
            "UR_L0_USE_IMMEDIATE_COMMANDLISTS")}}
    # two independent heavy matmul chains (pure device compute, no host sync inside)
    n = 4096
    reps = 60
    a = torch.randn(n, n, device=dev, dtype=torch.float32)
    b = torch.randn(n, n, device=dev, dtype=torch.float32)

    def chain(x):
        y = x
        for _ in range(reps):
            y = y @ b
            y = y * 1.0001
        return y

    def timed(fn, devs):
        for _ in range(2):
            fn()
        for d in devs:
            _sync(d)
        t = time.perf_counter()
        fn()
        for d in devs:
            _sync(d)
        return (time.perf_counter() - t) * 1000.0

    with torch.inference_mode():
        # isolated single-chain time
        iso_ms = timed(lambda: chain(a), (dev,))
        # two chains on the DEFAULT stream (baseline serial)
        def two_default():
            chain(a); chain(b)  # noqa: E702
        serial_ms = timed(two_default, (dev,))
        # two chains on TWO separate streams
        s1 = pe.make_stream(dev)
        s2 = pe.make_stream(dev)
        def two_stream():
            ds = pe.current_stream(dev)
            pe._stream_wait_stream(s1, ds)
            pe._stream_wait_stream(s2, ds)
            with pe.stream_context(dev, s1):
                chain(a)
            with pe.stream_context(dev, s2):
                chain(b)
            pe._stream_wait_stream(ds, s1)
            pe._stream_wait_stream(ds, s2)
        stream_ms = timed(two_stream, (dev,))

    overlap_ms = serial_ms - stream_ms
    res.update(isolated_chain_ms=iso_ms, two_chain_default_ms=serial_ms,
               two_chain_2stream_ms=stream_ms, stream_vs_default_gain_ms=overlap_ms,
               interpretation=("streams OVERLAP (gain>15%% of a chain)"
                               if overlap_ms > 0.15 * iso_ms else
                               "streams SERIALIZE (no meaningful overlap)"))
    log(f"[B] isolated 1 chain={iso_ms:.1f}ms | 2 chains default-stream={serial_ms:.1f}ms "
        f"| 2 chains 2-stream={stream_ms:.1f}ms | gain={overlap_ms:.1f}ms -> {res['interpretation']}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--stitched-npz", required=True)
    ap.add_argument("--which", default="A,B")
    ap.add_argument("--device", default="xpu:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = torch.device(args.device)
    pe.set_accel_device(dev)
    which = [w.strip().upper() for w in args.which.split(",") if w.strip()]
    out = {"device": str(dev), "torch": torch.__version__,
           "device_name": torch.xpu.get_device_properties(0).name if torch.xpu.is_available() else "cpu",
           "results": []}
    if "A" in which:
        out["results"].append(probe_A(args, dev))
    if "B" in which:
        out["results"].append(probe_B(args, dev))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    log(f"\nSAVED={args.out}")
    log("DONE")


if __name__ == "__main__":
    main()
