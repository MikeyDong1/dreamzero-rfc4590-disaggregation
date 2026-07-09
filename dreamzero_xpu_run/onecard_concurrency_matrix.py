#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
One-card multi-stream concurrency — corrected experiment matrix.

Separates two hypotheses for why the DreamZero POC's one_card_stream got ~0 overlap:
  H_sat = streams DO co-schedule, but the workload left no idle resource (saturation)
  H_hw  = this B60 + torch-xpu 2.12 stack does NOT run kernels from two torch.xpu
          streams concurrently on one card, regardless of workload

The first probe (two identical 4096-fp32 matmul chains) could not tell these apart
because ONE such chain already saturates the GPU. This matrix fixes that:

  E0  calibration: measurement must SEE overlap when it exists (two-CARD positive
      control) and ~0 when it doesn't (same-stream negative control). Uses the
      SAME event-based overlap metric everything else uses. If E0 fails -> instrument
      is blind -> no one-card verdict is admissible.
  E1  occupancy sweep (one stream): find a workload that provably leaves the tile
      idle (busy, i.e. non-saturating). Anchors N_sat (saturated) and S* (idle).
  E2  two independent compute streams at S* (non-saturating like-vs-like).
  E3  complementary pair: compute-bound stream A vs bandwidth-bound stream B (the
      real text||VAE signature) — the most decisive one-card test.

E5 (runtime-knob sweep) is done by re-invoking this script under different env from
an outer bash loop (env is read at zeInit), passing --experiments e0,e1,e2,e3.

OVERLAP METRIC (ground truth, not wall-clock):
  We time each stream's work with per-stream device Events (enable_timing) recording
  START and END on that stream, all enqueued sync-free, one final device sync. From
  the four event timestamps (Astart,Aend,Bstart,Bend) on a common device clock we
  compute the literal overlap interval:
       overlap_ms   = max(0, min(Aend,Bend) - max(Astart,Bstart))
       union_ms     = max(Aend,Bend) - min(Astart,Bstart)
       overlap_frac = overlap_ms / min(Adur, Bdur)      # 1.0 = fully concurrent, 0 = serial
  This directly answers "did the two streams' device work run at the same wall-clock
  time?" — independent of host launch queueing. We ALSO report wall ratio R and eff.

Decision (per the design's table):
  overlap_frac > 0.25 (or R<=1.5)                         -> OVERLAP (H_hw refuted)
  overlap_frac < 0.05 and R>=1.85 at a NON-SATURATING pt  -> SERIALIZE (supports H_hw)
  else                                                    -> partial/inconclusive
A verdict is only admissible if E0's two-card control showed overlap_frac>0.6 AND the
same-stream control showed <0.05 (proves the ruler works on THIS build).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


def log(m):
    print(m, flush=True)


def _accel(dev):
    if dev.type == "xpu" and hasattr(torch, "xpu"):
        return torch.xpu
    if dev.type == "cuda" and hasattr(torch, "cuda"):
        return torch.cuda
    return None


def _set_dev(accel, dev):
    if accel is not None and dev.index is not None and hasattr(accel, "set_device"):
        try:
            accel.set_device(dev)
        except Exception:  # noqa: BLE001
            pass


def _sync(accel, dev=None):
    if accel is None:
        return
    try:
        accel.synchronize(dev) if dev is not None else accel.synchronize()
    except (TypeError, RuntimeError):
        accel.synchronize()


def _stream(accel, dev):
    _set_dev(accel, dev)
    return accel.Stream() if accel is not None and hasattr(accel, "Stream") else None


def _event(accel):
    return accel.Event(enable_timing=True)


def _median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _iqr(xs):
    s = sorted(xs)
    n = len(s)
    if n < 4:
        return max(s) - min(s)
    return s[int(0.75 * (n - 1))] - s[int(0.25 * (n - 1))]


# ---------------------------------------------------------------------------
# Workload builders
# ---------------------------------------------------------------------------
def make_matmul_chain(dev, n, reps, dtype):
    """A dependent matmul chain: reps x (NxN @ NxN). Disjoint operands per call site."""
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)

    def run():
        y = a
        for _ in range(reps):
            y = y @ b
            y = y * 1.0001
        return y
    # flops per chain = reps * 2*N^3
    return run, (reps * 2.0 * n ** 3)


def make_bandwidth_chain(dev, numel, reps, dtype=torch.bfloat16):
    """Bandwidth-bound chain: elementwise over a >LLC tensor (low arithmetic intensity)."""
    x = torch.randn(numel, device=dev, dtype=dtype)
    y = torch.randn(numel, device=dev, dtype=dtype)

    def run():
        z = x
        for _ in range(reps):
            z = z + y
            z = z * 1.0001
        return z
    bytes_moved = reps * 3.0 * numel * torch.tensor([], dtype=dtype).element_size()
    return run, bytes_moved


# ---------------------------------------------------------------------------
# Core measurement: time one run, and time two runs on two streams with EVENT overlap
# ---------------------------------------------------------------------------
def time_isolated(accel, dev, run, warmup, reps):
    with torch.inference_mode():
        for _ in range(warmup):
            run()
        _sync(accel, dev)
        ts = []
        for _ in range(reps):
            _sync(accel, dev)
            t0 = time.perf_counter()
            run()
            _sync(accel, dev)
            ts.append((time.perf_counter() - t0) * 1000.0)
    return _median(ts), _iqr(ts), ts


def time_two_stream(accel, dev, runA, runB, s1, s2, warmup, reps, fenced=True):
    """Run A on s1 and B on s2, sync-free; measure wall + per-stream event overlap.

    Returns dict with wall_ms (median), and median event-derived overlap_frac.
    """
    walls, ov_fracs, aDurs, bDurs = [], [], [], []
    with torch.inference_mode():
        for _ in range(warmup):
            _run_pair(accel, dev, runA, runB, s1, s2, fenced, want_events=False)
        _sync(accel, dev)
        for _ in range(reps):
            _sync(accel, dev)
            t0 = time.perf_counter()
            evs = _run_pair(accel, dev, runA, runB, s1, s2, fenced, want_events=True)
            _sync(accel, dev)
            walls.append((time.perf_counter() - t0) * 1000.0)
            if evs is not None:
                aS, aE, bS, bE = evs
                # elapsed_time(other) = ms from self to other on the device clock.
                # Build a common timeline anchored at aS.
                a_start = 0.0
                a_end = aS.elapsed_time(aE)
                b_start = aS.elapsed_time(bS)
                b_end = aS.elapsed_time(bE)
                aDur = a_end - a_start
                bDur = b_end - b_start
                inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
                denom = min(aDur, bDur) if min(aDur, bDur) > 1e-6 else 1e-6
                ov_fracs.append(inter / denom)
                aDurs.append(aDur)
                bDurs.append(bDur)
    out = {"wall_ms": _median(walls), "wall_iqr": _iqr(walls), "walls": walls}
    if ov_fracs:
        out.update(overlap_frac=_median(ov_fracs), overlap_fracs=ov_fracs,
                   a_dur_ms=_median(aDurs), b_dur_ms=_median(bDurs))
    return out


def _run_pair(accel, dev, runA, runB, s1, s2, fenced, want_events):
    """Enqueue A->s1 and B->s2 sync-free. Optionally bracket each with timing events."""
    cur = accel.current_stream()
    evs = None
    if want_events:
        aS, aE, bS, bE = _event(accel), _event(accel), _event(accel), _event(accel)
    if fenced:
        s1.wait_stream(cur)
        s2.wait_stream(cur)
    with accel.stream(s1):
        if want_events:
            aS.record()
        runA()
        if want_events:
            aE.record()
    with accel.stream(s2):
        if want_events:
            bS.record()
        runB()
        if want_events:
            bE.record()
    if fenced:
        cur.wait_stream(s1)
        cur.wait_stream(s2)
    if want_events:
        evs = (aS, aE, bS, bE)
    return evs


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def exp_e0(accel, dev, args, out):
    """Calibration: two-card positive control + same-stream negative control +
    two-stream one-card — all with the SAME event-overlap metric + tiny workload."""
    log("\n===== E0: calibration (does our overlap metric WORK on this build?) =====")
    n, reps, dtype = 256, 40, torch.float32
    res = {"workload": f"matmul chain N={n} reps={reps} fp32 (tiny/launch-bound)"}

    # --- one-card streams ---
    runA, _ = make_matmul_chain(dev, n, reps, dtype)
    runB, _ = make_matmul_chain(dev, n, reps, dtype)
    s1, s2 = _stream(accel, dev), _stream(accel, dev)
    assert s1 is not s2
    iso, iso_iqr, _ = time_isolated(accel, dev, runA, args.warmup, args.reps)
    same = time_two_stream(accel, dev, runA, runB, accel.current_stream(),
                           accel.current_stream(), args.warmup, args.reps, fenced=False)
    two = time_two_stream(accel, dev, runA, runB, s1, s2, args.warmup, args.reps, fenced=True)
    res["one_card"] = {
        "isolated_ms": iso, "isolated_iqr": iso_iqr,
        "same_stream_wall_ms": same["wall_ms"],
        "same_stream_overlap_frac": same.get("overlap_frac"),
        "two_stream_wall_ms": two["wall_ms"],
        "two_stream_overlap_frac": two.get("overlap_frac"),
        "R": two["wall_ms"] / iso if iso else None,
    }
    log(f"[E0 one-card] iso={iso:.1f}ms same_stream_wall={same['wall_ms']:.1f} "
        f"(ovl={same.get('overlap_frac')}) two_stream_wall={two['wall_ms']:.1f} "
        f"(ovl={two.get('overlap_frac'):.3f}) R={two['wall_ms']/iso:.2f}")

    # --- two-card positive control ---
    if torch.xpu.device_count() >= 2:
        devB = torch.device("xpu:1")
        runA2, _ = make_matmul_chain(dev, n, reps, dtype)
        runB2, _ = make_matmul_chain(devB, n, reps, dtype)
        # events on two devices don't share a clock; use wall speedup as the 2-card signal
        with torch.inference_mode():
            for _ in range(args.warmup):
                runA2(); runB2()
            _sync(accel, dev); _sync(accel, devB)
            walls = []
            for _ in range(args.reps):
                _sync(accel, dev); _sync(accel, devB)
                t0 = time.perf_counter()
                runA2()  # different devices -> both queue then run concurrently
                runB2()
                _sync(accel, dev); _sync(accel, devB)
                walls.append((time.perf_counter() - t0) * 1000.0)
        two_card_wall = _median(walls)
        # positive control passes if 2-card wall ~ max(iso,iso) not 2*iso
        res["two_card"] = {"wall_ms": two_card_wall, "R_vs_one_chain": two_card_wall / iso if iso else None}
        log(f"[E0 two-card] wall={two_card_wall:.1f}ms  R_vs_one_chain={two_card_wall/iso:.2f} "
            f"(want ~1.0 = concurrency across cards works)")
    else:
        res["two_card"] = {"note": "only 1 device visible; positive control skipped"}

    # calibration verdicts
    same_ovl = res["one_card"]["same_stream_overlap_frac"] or 0.0
    twocard_R = (res.get("two_card") or {}).get("R_vs_one_chain")
    res["calibration_same_stream_low"] = same_ovl < 0.15
    res["calibration_two_card_overlap"] = (twocard_R is not None and twocard_R < 1.5)
    res["instrument_valid"] = bool(res["calibration_two_card_overlap"])
    log(f"[E0] instrument_valid={res['instrument_valid']} "
        f"(two-card concurrency seen={res['calibration_two_card_overlap']}, "
        f"same-stream low-overlap={res['calibration_same_stream_low']})")
    out["E0"] = res


def exp_e1(accel, dev, args, out):
    """Occupancy sweep: find non-saturating S* and saturated N_sat via TFLOP/s knee."""
    log("\n===== E1: occupancy sweep (find non-saturating size) =====")
    rows = []
    for n in [256, 512, 1024, 2048, 4096]:
        reps = max(4, int(round((4096.0 / n) ** 3 * 4)))  # keep ~constant flops/timed-region
        reps = min(reps, 4000)
        for dtype, dn in [(torch.float32, "fp32"), (torch.bfloat16, "bf16")]:
            run, flops = make_matmul_chain(dev, n, reps, dtype)
            iso, iqr, _ = time_isolated(accel, dev, run, args.warmup, max(8, args.reps // 2))
            tflops = flops / (iso / 1000.0) / 1e12
            rows.append({"n": n, "dtype": dn, "reps": reps, "iso_ms": iso,
                         "tflops": tflops})
            log(f"[E1] N={n:5d} {dn} reps={reps:4d} iso={iso:7.1f}ms  {tflops:7.2f} TFLOP/s")
    # peak per dtype -> S* = largest N with <=60% peak tflops; N_sat = smallest within 90%
    res = {"sweep": rows, "anchors": {}}
    for dn in ("fp32", "bf16"):
        sub = [r for r in rows if r["dtype"] == dn]
        peak = max(r["tflops"] for r in sub)
        nonsat = [r for r in sub if r["tflops"] <= 0.60 * peak]
        sat = [r for r in sub if r["tflops"] >= 0.90 * peak]
        s_star = max(nonsat, key=lambda r: r["n"]) if nonsat else min(sub, key=lambda r: r["tflops"])
        n_sat = min(sat, key=lambda r: r["n"]) if sat else max(sub, key=lambda r: r["tflops"])
        res["anchors"][dn] = {"peak_tflops": peak,
                              "S_star_n": s_star["n"], "S_star_frac_peak": s_star["tflops"] / peak,
                              "N_sat_n": n_sat["n"]}
        log(f"[E1] {dn}: peak={peak:.2f} TFLOP/s  S*(non-sat)=N{s_star['n']} "
            f"({100*s_star['tflops']/peak:.0f}% peak)  N_sat=N{n_sat['n']}")
    out["E1"] = res


def _pair_verdict(iso, two, nonsat):
    R = two["wall_ms"] / iso if iso else None
    ovl = two.get("overlap_frac")
    eff = None  # filled by caller when both durs known
    verdict = "inconclusive"
    if (ovl is not None and ovl > 0.25) or (R is not None and R <= 1.5):
        verdict = "OVERLAP (H_hw refuted)"
    elif R is not None and R >= 1.85 and (ovl is None or ovl < 0.05):
        verdict = "SERIALIZE" + (" @ non-saturating (supports H_hw)" if nonsat else " (saturated - expected)")
    return R, ovl, verdict


def exp_e2(accel, dev, args, out, s_star_n=512):
    """Two independent compute chains at the non-saturating S*."""
    log(f"\n===== E2: two compute streams at non-saturating N={s_star_n} =====")
    res = {}
    for dtype, dn in [(torch.float32, "fp32"), (torch.bfloat16, "bf16")]:
        # size chain so one chain ~180ms
        reps = 200
        run, _ = make_matmul_chain(dev, s_star_n, reps, dtype)
        iso, _, _ = time_isolated(accel, dev, run, args.warmup, args.reps)
        # scale reps to ~180ms
        if iso > 0:
            reps = max(20, int(reps * 180.0 / iso))
        runA, _ = make_matmul_chain(dev, s_star_n, reps, dtype)
        runB, _ = make_matmul_chain(dev, s_star_n, reps, dtype)
        iso, iso_iqr, _ = time_isolated(accel, dev, runA, args.warmup, args.reps)
        s1, s2 = _stream(accel, dev), _stream(accel, dev)
        for fenced in (True, False):
            two = time_two_stream(accel, dev, runA, runB, s1, s2, args.warmup, args.reps, fenced=fenced)
            R, ovl, verdict = _pair_verdict(iso, two, nonsat=True)
            key = f"{dn}_{'fenced' if fenced else 'fencefree'}"
            res[key] = {"n": s_star_n, "reps": reps, "isolated_ms": iso, "isolated_iqr": iso_iqr,
                        "two_stream_wall_ms": two["wall_ms"], "overlap_frac": ovl, "R": R,
                        "a_dur_ms": two.get("a_dur_ms"), "b_dur_ms": two.get("b_dur_ms"),
                        "verdict": verdict}
            log(f"[E2 {key}] iso={iso:.1f}ms two_stream={two['wall_ms']:.1f}ms R={R:.2f} "
                f"overlap_frac={ovl if ovl is None else round(ovl,3)} -> {verdict}")
    out["E2"] = res


def exp_e3(accel, dev, args, out, s_star_n=512):
    """Complementary pair: compute-bound A vs bandwidth-bound B (the text||VAE signature)."""
    log(f"\n===== E3: complementary compute(N={s_star_n}) || bandwidth pair =====")
    res = {}
    # A: bf16 compute chain ~180ms; B: bandwidth chain (>0.5 GiB) ~ matched-ish
    runA0, _ = make_matmul_chain(dev, s_star_n, 200, torch.bfloat16)
    isoA0, _, _ = time_isolated(accel, dev, runA0, args.warmup, args.reps)
    repsA = max(20, int(200 * 180.0 / isoA0)) if isoA0 else 200
    runA, _ = make_matmul_chain(dev, s_star_n, repsA, torch.bfloat16)

    numel = 160 * 1024 * 1024  # 160M bf16 = 320 MiB per tensor; 3 tensors touched > LLC
    runB0, _ = make_bandwidth_chain(dev, numel, 40)
    isoB0, _, _ = time_isolated(accel, dev, runB0, args.warmup, max(6, args.reps // 2))
    repsB = max(10, int(40 * 400.0 / isoB0)) if isoB0 else 40  # ~400ms bandwidth chain
    runB, _ = make_bandwidth_chain(dev, numel, repsB)

    isoA, _, _ = time_isolated(accel, dev, runA, args.warmup, args.reps)
    isoB, _, _ = time_isolated(accel, dev, runB, args.warmup, max(6, args.reps // 2))
    log(f"[E3] isolated compute-A={isoA:.1f}ms  bandwidth-B={isoB:.1f}ms")
    s1, s2 = _stream(accel, dev), _stream(accel, dev)
    for fenced in (True, False):
        two = time_two_stream(accel, dev, runA, runB, s1, s2, args.warmup, args.reps, fenced=fenced)
        wall = two["wall_ms"]
        eff = ((isoA + isoB) - wall) / min(isoA, isoB) if min(isoA, isoB) > 0 else None
        ovl = two.get("overlap_frac")
        if (ovl is not None and ovl > 0.25) or (eff is not None and eff >= 0.25):
            verdict = "OVERLAP (H_hw refuted; one-card CAN help the pair)"
        elif eff is not None and eff < 0.10 and (ovl is None or ovl < 0.05):
            verdict = "SERIALIZE (supports H_hw for the encoder pair)"
        else:
            verdict = "partial/inconclusive"
        key = "fenced" if fenced else "fencefree"
        res[key] = {"isoA_ms": isoA, "isoB_ms": isoB, "two_stream_wall_ms": wall,
                    "sum_ms": isoA + isoB, "eff": eff, "overlap_frac": ovl, "verdict": verdict}
        log(f"[E3 {key}] wall={wall:.1f}ms sum={isoA+isoB:.1f}ms eff={eff if eff is None else round(eff,3)} "
            f"overlap_frac={ovl if ovl is None else round(ovl,3)} -> {verdict}")
    out["E3"] = res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xpu:0")
    ap.add_argument("--experiments", default="e0,e1,e2,e3")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--s-star", type=int, default=0, help="override S* N for E2/E3 (0=auto from E1)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = torch.device(args.device)
    accel = _accel(dev)
    _set_dev(accel, dev)
    want = [e.strip().lower() for e in args.experiments.split(",") if e.strip()]

    out = {
        "device": str(dev),
        "device_name": torch.xpu.get_device_properties(dev.index or 0).name if torch.xpu.is_available() else "cpu",
        "device_count": torch.xpu.device_count() if torch.xpu.is_available() else 0,
        "torch": torch.__version__,
        "env": {k: os.environ.get(k) for k in (
            "SYCL_UR_USE_LEVEL_ZERO_V2", "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS",
            "UR_L0_USE_IMMEDIATE_COMMANDLISTS", "SYCL_PI_LEVEL_ZERO_USE_COMPUTE_ENGINE",
            "ZEX_NUMBER_OF_CCS", "ZE_AFFINITY_MASK")},
        "config": {"warmup": args.warmup, "reps": args.reps},
    }
    log(f"[env] device={dev} name={out['device_name']} count={out['device_count']} torch={out['torch']}")
    log(f"[env] knobs={out['env']}")

    if "e0" in want:
        exp_e0(accel, dev, args, out)
    if "e1" in want:
        exp_e1(accel, dev, args, out)
    # resolve S* for E2/E3
    s_star = args.s_star or 512
    if not args.s_star and "E1" in out:
        s_star = out["E1"]["anchors"].get("bf16", {}).get("S_star_n", 512)
    if "e2" in want:
        exp_e2(accel, dev, args, out, s_star_n=s_star)
    if "e3" in want:
        exp_e3(accel, dev, args, out, s_star_n=s_star)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    log(f"\nSAVED={args.out}")
    log("DONE")


if __name__ == "__main__":
    main()
