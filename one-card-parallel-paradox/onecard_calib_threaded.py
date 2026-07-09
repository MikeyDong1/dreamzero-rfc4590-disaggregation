#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Corrected calibration + one-card test — THREADED, device-bound.

The first matrix run's E0 calibration FAILED: the two-card positive control showed
R=2.07 (two separate GPUs appeared to "serialize"), proving the harness was BLIND.
Root causes: (1) tiny 256-matmul workload was host-launch-bound not device-bound,
(2) both cards were driven from ONE python thread so host submission serialized,
(3) the cross-stream event.elapsed_time overlap metric returned 0 on this XPU build.

This script fixes all three:
  * DEVICE-BOUND workload: chains sized so ONE chain ~= 180 ms of device work at a
    NON-SATURATING size (bf16 N=512, ~10% of peak per the E1 sweep) -> host launch is
    a negligible fraction, so wall-clock reflects device scheduling.
  * ONE THREAD PER STREAM/DEVICE: each stream's chain is launched from its own python
    thread, so the host truly submits both concurrently (removes GIL/host-serial doubt).
  * WALL-CLOCK ratio R as the metric (no fragile cross-stream events). R = wall(pair)/
    wall(one chain). R~=1 => the two ran concurrently; R~=2 => they serialized.

Three configurations, identical workload, identical threading:
  A) TWO CARDS  (xpu:0 + xpu:1)      -- POSITIVE control: MUST overlap (R~=1) or the
                                        instrument is blind and nothing is conclusive.
  B) ONE CARD, TWO STREAMS           -- the actual question.
  C) ONE CARD, SAME (default) stream -- NEGATIVE control: MUST serialize (R~=2).

Interpretation (only valid if A passes, i.e. R_A < ~1.4):
  * B R~=1  => one-card streams DO overlap  => H_hw refuted
  * B R~=2 (like C) => one-card streams serialize while two cards overlap => supports H_hw
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

import torch


def log(m):
    print(m, flush=True)


def sync(dev):
    torch.xpu.synchronize(dev)


def make_chain(dev, n, reps, dtype):
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)

    def run():
        y = a
        for _ in range(reps):
            y = y @ b
            y = y * 1.0001
        return y
    return run


def run_on_stream(run, dev, stream):
    """Run a chain inside a stream context (called from a worker thread)."""
    with torch.xpu.stream(stream):
        run()


def timed_pair_threaded(fnA, fnB, devs, warmup, reps):
    """Launch fnA and fnB from two threads; time wall of both completing."""
    def once():
        tA = threading.Thread(target=fnA)
        tB = threading.Thread(target=fnB)
        tA.start(); tB.start()
        tA.join(); tB.join()
        for d in devs:
            sync(d)
    for _ in range(warmup):
        once()
    walls = []
    for _ in range(reps):
        for d in devs:
            sync(d)
        t0 = time.perf_counter()
        once()
        walls.append((time.perf_counter() - t0) * 1000.0)
    walls.sort()
    return walls[len(walls) // 2]  # median


def timed_one(run, dev, warmup, reps):
    for _ in range(warmup):
        run()
    sync(dev)
    walls = []
    for _ in range(reps):
        sync(dev)
        t0 = time.perf_counter()
        run()
        sync(dev)
        walls.append((time.perf_counter() - t0) * 1000.0)
    walls.sort()
    return walls[len(walls) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512, help="non-saturating matmul size (bf16 S*)")
    ap.add_argument("--target-ms", type=float, default=180.0)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    d0 = torch.device("xpu:0")
    ndev = torch.xpu.device_count()
    log(f"[env] device_count={ndev} torch={torch.__version__} "
        f"name={torch.xpu.get_device_properties(0).name}")
    log(f"[env] knobs SYCL_UR_USE_LEVEL_ZERO_V2={os.environ.get('SYCL_UR_USE_LEVEL_ZERO_V2')} "
        f"IMM_CMDLIST={os.environ.get('UR_L0_USE_IMMEDIATE_COMMANDLISTS')} "
        f"USE_COMPUTE_ENGINE={os.environ.get('SYCL_PI_LEVEL_ZERO_USE_COMPUTE_ENGINE')} "
        f"CCS={os.environ.get('ZEX_NUMBER_OF_CCS')}")

    # calibrate reps so one chain ~= target_ms on xpu:0 at the non-saturating size
    probe = make_chain(d0, args.n, 100, dtype)
    iso100 = timed_one(probe, d0, args.warmup, 8)
    reps = max(20, int(100 * args.target_ms / iso100)) if iso100 else 100
    log(f"[calib] N={args.n} {args.dtype}: 100-rep chain={iso100:.1f}ms -> using reps={reps} for ~{args.target_ms:.0f}ms")

    runA0 = make_chain(d0, args.n, reps, dtype)
    iso = timed_one(runA0, d0, args.warmup, args.reps)
    log(f"[calib] isolated one chain on xpu:0 = {iso:.1f}ms  (this is R=1.0 baseline)")

    out = {
        "device_name": torch.xpu.get_device_properties(0).name,
        "torch": torch.__version__, "device_count": ndev,
        "n": args.n, "dtype": args.dtype, "reps": reps, "isolated_ms": iso,
        "env": {k: os.environ.get(k) for k in (
            "SYCL_UR_USE_LEVEL_ZERO_V2", "UR_L0_USE_IMMEDIATE_COMMANDLISTS",
            "SYCL_PI_LEVEL_ZERO_USE_COMPUTE_ENGINE", "ZEX_NUMBER_OF_CCS", "ZE_AFFINITY_MASK")},
        "configs": {},
    }

    # ---- C) negative control: one card, SAME default stream, two threads ----
    rA = make_chain(d0, args.n, reps, dtype)
    rB = make_chain(d0, args.n, reps, dtype)
    wall_same = timed_pair_threaded(rA, rB, [d0], args.warmup, args.reps)
    out["configs"]["C_one_card_same_stream"] = {"wall_ms": wall_same, "R": wall_same / iso}
    log(f"[C  neg-ctrl ] one-card same-stream 2 threads: wall={wall_same:.1f}ms R={wall_same/iso:.2f} (expect ~2)")

    # ---- B) one card, TWO streams, two threads ----
    s1 = torch.xpu.Stream(); s2 = torch.xpu.Stream()
    rA = make_chain(d0, args.n, reps, dtype)
    rB = make_chain(d0, args.n, reps, dtype)
    wall_2stream = timed_pair_threaded(lambda: run_on_stream(rA, d0, s1),
                                       lambda: run_on_stream(rB, d0, s2), [d0], args.warmup, args.reps)
    out["configs"]["B_one_card_two_stream"] = {"wall_ms": wall_2stream, "R": wall_2stream / iso}
    log(f"[B  TEST     ] one-card two-stream 2 threads: wall={wall_2stream:.1f}ms R={wall_2stream/iso:.2f}")

    # ---- A) positive control: TWO CARDS, two threads ----
    if ndev >= 2:
        d1 = torch.device("xpu:1")
        rA = make_chain(d0, args.n, reps, dtype)
        rB = make_chain(d1, args.n, reps, dtype)
        wall_2card = timed_pair_threaded(rA, rB, [d0, d1], args.warmup, args.reps)
        out["configs"]["A_two_card"] = {"wall_ms": wall_2card, "R": wall_2card / iso}
        log(f"[A  pos-ctrl ] two-card 2 threads: wall={wall_2card:.1f}ms R={wall_2card/iso:.2f} (MUST be ~1 for valid instrument)")

    # verdict
    RA = out["configs"].get("A_two_card", {}).get("R")
    RB = out["configs"]["B_one_card_two_stream"]["R"]
    RC = out["configs"]["C_one_card_same_stream"]["R"]
    instrument_valid = (RA is not None and RA < 1.4)
    if not instrument_valid:
        verdict = "INSTRUMENT-INVALID (two-card positive control did not overlap; cannot conclude)"
    elif RB < 1.4:
        verdict = "one-card streams OVERLAP (R_B~1) => H_hw REFUTED"
    elif RB > 1.7:
        verdict = "one-card streams SERIALIZE (R_B~2) while two cards overlap => SUPPORTS H_hw"
    else:
        verdict = "partial/inconclusive (1.4 < R_B < 1.7)"
    out["instrument_valid"] = instrument_valid
    out["verdict"] = verdict
    log(f"\n[VERDICT] instrument_valid={instrument_valid}  R_A(2card)={RA}  R_B(2stream)={RB:.2f}  R_C(same)={RC:.2f}")
    log(f"[VERDICT] {verdict}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    log(f"SAVED={args.out}")
    log("DONE")


if __name__ == "__main__":
    main()
