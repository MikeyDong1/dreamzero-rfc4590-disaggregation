#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
============================================================================
 THE ONE-CARD PARALLEL "PARADOX" — a KEPT-FOR-DOCUMENTATION FLAWED PROBE
============================================================================

*** WARNING — DO NOT CITE THIS SCRIPT'S OUTPUT AS PROOF. ***

This is the ORIGINAL probe that was used to (wrongly) claim "one-card multi-stream
is impossible on this hardware." It is kept here as a documented example of a
methodological mistake. See README.md for the full explanation and the corrected,
but ultimately INCONCLUSIVE, follow-up experiments.

THE FLAW
--------
This probe runs two IDENTICAL 4096x4096 fp32 matmul chains on two streams and
shows wall ~= 2x one chain, then concludes "streams serialize -> impossible."
But a single 4096-fp32 chain ALREADY SATURATES the B60's compute units, so two
of them serialize on ANY GPU (including ones that fully support concurrent
streams) — there is simply no idle resource for the second to use. Therefore the
2x result is equally consistent with:
   H_hw  = the hardware won't co-schedule streams, AND
   H_sat = it can, but this workload left no idle resource.
This probe CANNOT distinguish them, so it proves nothing about the hardware.

WHAT IS ACTUALLY TRUE
---------------------
Measured: for the DreamZero text+VAE encoder pair on this B60, one-card two-stream
gave no measurable speedup under the settings tested. That is a fact about the
WORKLOAD. Whether the cause is hardware (H_hw) or saturation (H_sat) was NOT
resolved — the corrected experiments' measurement harness failed its own two-card
positive control (see README.md, onecard_calib_threaded.py). The strong "impossible"
claim is WITHDRAWN.

The lines below are the original (flawed) framing, left intact for the record:
----------------------------------------------------------------------------
[original claim] "On the Intel Arc Pro B60, you CANNOT make two independent pieces
of work run concurrently via two torch.xpu streams of one card." <-- NOT PROVEN.

WHY THIS TEST IS CONCLUSIVE
---------------------------
The POC's own encoders (UMT5 text vs Wan VAE) are unequal in size and share the
device's memory system, so "no overlap" there could be blamed on the VAE
saturating bandwidth. This demonstrator removes every such confound:

  * Two IDENTICAL, INDEPENDENT matmul chains. No shared tensors, no data
    dependency between them -> nothing FORCES them to serialize except the
    hardware/runtime itself.
  * Pure on-device compute (a chain of large fp32 matmuls). No host<->device
    syncs, no .cpu()/.item()/print inside the timed region.
  * We compare three wall-clock times on ONE card:
        (1) one chain alone                      -> T1
        (2) two chains on the SAME (default) stream
        (3) two chains on TWO SEPARATE streams

THE DECISIVE LOGIC (read this before the numbers)
-------------------------------------------------
If two streams on one card COULD overlap, then:
        two-stream time  ->  ~max(chainA, chainB)  ~= T1        (work hidden)
If they SERIALIZE, then:
        two-stream time  ->  chainA + chainB       ~= 2 * T1    (work stacked)

So the test is a clean binary:
   * two_stream ~= T1     => overlap EXISTS (claim refuted)
   * two_stream ~= 2*T1   => serialization  (claim confirmed, one-card is impossible)

We also print the "gain" = (two-chains-same-stream) - (two-chains-two-streams).
A gain of ~0 means switching to two streams bought nothing.

MEASURED RESULT ON gnr17408 (Arc Pro B60, torch 2.12.0+xpu), 2026-07-08
----------------------------------------------------------------------
   1 chain alone .................... 702.6 ms
   2 chains, same (default) stream .. 1405.1 ms   (~= 2 * 702.6)
   2 chains, two separate streams ... 1405.5 ms   (~= 2 * 702.6, NOT ~702)
   gain from two streams ............ -0.3 ms  ->  streams SERIALIZE

Re-run with immediate command lists OFF
(SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0, UR_L0_USE_IMMEDIATE_COMMANDLISTS=0):
   two-stream = 1405.3 ms, gain -0.7 ms  ->  identical conclusion.

=> The two-stream wall-clock equals the SUM of the two chains, exactly as if they
   ran back-to-back. Two streams on one B60 do not overlap. QED.

HOW TO RUN (inside the vllm-omni XPU container on the node)
-----------------------------------------------------------
    # default environment:
    python demonstrate_one_card_serialization.py

    # prove the immediate-command-list setting doesn't change it:
    SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 \
    UR_L0_USE_IMMEDIATE_COMMANDLISTS=0 \
    python demonstrate_one_card_serialization.py

    # (optional) also works on CUDA for contrast, where you WILL see overlap:
    python demonstrate_one_card_serialization.py --device cuda:0

SCOPE / CAVEAT
--------------
"Impossible" is scoped to the STREAM-OVERLAP approach on this stack (Arc Pro B60
+ torch-xpu 2.12). It does not claim that no form of intra-card sharing can ever
exist (e.g. MPS-style partitioning or multi-process); it shows that the mechanism
the POC used — two torch.xpu streams on one card — cannot produce concurrency here.
"""
from __future__ import annotations

import argparse
import json
import os
import time


def log(m):
    print(m, flush=True)


def _accel(device):
    """Return torch.xpu / torch.cuda for the device (or None for CPU)."""
    import torch
    if device.type == "xpu" and hasattr(torch, "xpu"):
        return torch.xpu
    if device.type == "cuda" and hasattr(torch, "cuda"):
        return torch.cuda
    return None


def _sync(accel):
    if accel is not None:
        accel.synchronize()


def _make_stream(accel):
    # Both torch.xpu.Stream and torch.cuda.Stream create on the *current* device.
    return accel.Stream() if accel is not None and hasattr(accel, "Stream") else None


def main():
    ap = argparse.ArgumentParser(description="One-card two-stream serialization demonstrator.")
    ap.add_argument("--device", default="xpu:0", help="xpu:0 (default) / cuda:0 / cpu")
    ap.add_argument("--matrix", type=int, default=4096, help="NxN matmul size (compute knob).")
    ap.add_argument("--chain", type=int, default=60, help="matmuls per chain (chain length).")
    ap.add_argument("--out", default=None, help="optional path to write a result JSON.")
    args = ap.parse_args()

    import torch  # imported after argparse so --help works without torch

    device = torch.device(args.device)
    accel = _accel(device)
    if accel is not None and device.index is not None and hasattr(accel, "set_device"):
        accel.set_device(device)  # xpu stream/memory APIs are current-device scoped

    dev_name = "cpu"
    if accel is not None and hasattr(accel, "get_device_properties"):
        try:
            dev_name = accel.get_device_properties(device.index or 0).name
        except Exception:  # noqa: BLE001
            pass
    log(f"[env] device={device} name='{dev_name}' torch={torch.__version__}")
    log(f"[env] SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS="
        f"{os.environ.get('SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS')} "
        f"UR_L0_USE_IMMEDIATE_COMMANDLISTS={os.environ.get('UR_L0_USE_IMMEDIATE_COMMANDLISTS')}")

    n, reps = args.matrix, args.chain
    # Two INDEPENDENT operands. `a` seeds chain A, `b` seeds chain B and is the
    # shared multiplicand — but the two chains write disjoint outputs and never
    # read each other, so there is no data dependency linking them.
    a = torch.randn(n, n, device=device, dtype=torch.float32)
    b = torch.randn(n, n, device=device, dtype=torch.float32)

    def chain(x):
        """A long dependent matmul chain = a big blob of pure device compute.

        Each step depends on the previous (so it can't be reordered away), but the
        chain as a whole is independent of the other chain. The *1.0001 keeps the
        values bounded without adding a sync."""
        y = x
        for _ in range(reps):
            y = y @ b
            y = y * 1.0001
        return y

    def timed(fn, warmups=2):
        """Wall-clock of fn(): warm up, sync, time one call, sync. No sync inside."""
        for _ in range(warmups):
            fn()
        _sync(accel)
        t0 = time.perf_counter()
        fn()
        _sync(accel)
        return (time.perf_counter() - t0) * 1000.0

    with torch.inference_mode():
        # ---- (1) one chain alone: the baseline unit of work T1 ----
        iso_ms = timed(lambda: chain(a))

        # ---- (2) two chains on the DEFAULT stream: guaranteed serial ----
        def two_default():
            chain(a)
            chain(b)
        serial_ms = timed(two_default)

        # ---- (3) two chains on TWO SEPARATE streams: the actual test ----
        # Enqueue each chain onto its own stream. If the card co-schedules
        # streams, these two blobs overlap and wall -> ~T1. We add cross-stream
        # wait edges only at the boundaries (so the side streams see the input
        # allocations and the host waits for both) — these are device-side
        # dependency edges, NOT host syncs, so they don't serialize the CPU.
        s1 = _make_stream(accel)
        s2 = _make_stream(accel)
        if s1 is None or s2 is None:
            log("[warn] no stream support on this device; two-stream == default.")
            stream_ms = serial_ms
        else:
            def two_stream():
                cur = accel.current_stream()
                s1.wait_stream(cur)              # s1 waits for pending default-stream work
                s2.wait_stream(cur)              # s2 waits for pending default-stream work
                with accel.stream(s1):
                    chain(a)                     # chain A -> stream 1
                with accel.stream(s2):
                    chain(b)                     # chain B -> stream 2 (independent of A)
                cur.wait_stream(s1)              # default waits for s1
                cur.wait_stream(s2)              # default waits for s2
            stream_ms = timed(two_stream)

    gain = serial_ms - stream_ms                 # >0 would mean streams helped
    # Decision rule: call it overlap only if two streams recovered at least 15%
    # of one chain's time vs the same-stream baseline.
    overlaps = gain > 0.15 * iso_ms
    verdict = "OVERLAP (streams run concurrently)" if overlaps else \
              "SERIALIZE (two streams == one stream; one-card parallel impossible)"

    log("")
    log("===================== RESULT =====================")
    log(f"  1 chain alone .................... {iso_ms:8.1f} ms   (= T1, one unit of work)")
    log(f"  2 chains, same (default) stream .. {serial_ms:8.1f} ms   (~ 2*T1 by construction)")
    log(f"  2 chains, two separate streams ... {stream_ms:8.1f} ms")
    log(f"  gain from two streams ............ {gain:8.1f} ms")
    log(f"  ratio two-stream / one-chain ..... {stream_ms/iso_ms:8.2f}x   "
        f"(~1.0 => overlap, ~2.0 => serialize)")
    log(f"  VERDICT: {verdict}")
    log("==================================================")

    if args.out:
        payload = {
            "device": str(device), "device_name": dev_name, "torch": torch.__version__,
            "env": {k: os.environ.get(k) for k in
                    ("SYCL_UR_USE_LEVEL_ZERO_V2",
                     "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS",
                     "UR_L0_USE_IMMEDIATE_COMMANDLISTS")},
            "matrix": n, "chain_len": reps,
            "isolated_chain_ms": iso_ms,
            "two_chain_default_ms": serial_ms,
            "two_chain_2stream_ms": stream_ms,
            "stream_vs_default_gain_ms": gain,
            "two_stream_over_one_chain_ratio": stream_ms / iso_ms,
            "overlaps": overlaps,
            "verdict": verdict,
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        log(f"SAVED={args.out}")


if __name__ == "__main__":
    main()
