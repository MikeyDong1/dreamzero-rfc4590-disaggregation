#!/usr/bin/env python3
"""Whole-device XPU memory sampler using torch.xpu.mem_get_info (works INSIDE the
container on gnr17408, where host xpu-smi / Level-Zero is unavailable and the dev7
image ships no xpu-smi binary).

Writes CSV rows: epoch_s,device_id,mem_used_mib  where mem_used_mib = total - free
for the WHOLE device (includes every process on that card, not just ours -- same
semantics as the xpu-smi "GPU Memory Used" the 797635 run sampled). Runs until
SIGTERM/SIGINT.

Usage: xpu_mem_sampler_torch.py <device_id> <interval_ms> <out_csv>

Note: this sampler process opens its own tiny XPU context; that overhead is a small
constant present in every sample (including the pre-run baseline), so per-N *peaks*
are unaffected. The card is pinned via ZE_AFFINITY_MASK=0 so device 0 == card0.
"""
import os
import signal
import sys
import time

import torch

DEV = int(sys.argv[1]) if len(sys.argv) > 1 else 0
INTERVAL_S = (int(sys.argv[2]) if len(sys.argv) > 2 else 400) / 1000.0
OUT = sys.argv[3] if len(sys.argv) > 3 else "/tmp/xpu_mem.csv"

_run = {"go": True}


def _stop(*_a):
    _run["go"] = False


signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)

torch.xpu.init()
with open(OUT, "w", buffering=1) as f:
    f.write("epoch_s,device_id,mem_used_mib\n")
    while _run["go"]:
        ts = time.time()
        try:
            free, total = torch.xpu.mem_get_info(DEV)
            used_mib = (total - free) // 1024 // 1024
            f.write(f"{ts:.3f},{DEV},{used_mib}\n")
        except Exception:
            pass
        time.sleep(INTERVAL_S)
