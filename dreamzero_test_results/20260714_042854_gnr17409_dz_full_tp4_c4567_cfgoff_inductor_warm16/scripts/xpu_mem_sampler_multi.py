#!/usr/bin/env python3
"""Multi-device whole-device XPU memory sampler (torch.xpu.mem_get_info).

Samples EVERY visible XPU device (0..N-1 in-container, mapped to the physical
cards by ZE_AFFINITY_MASK) so a TP=N run's per-card whole-device memory is
captured. Writes CSV rows: epoch_s,device_id,mem_used_mib  (mem_used = total-free
for the whole device, all processes). Runs until SIGTERM/SIGINT.

Usage: xpu_mem_sampler_multi.py <interval_ms> <out_csv>
"""
import signal
import sys
import time

import torch

INTERVAL_S = (int(sys.argv[1]) if len(sys.argv) > 1 else 400) / 1000.0
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/xpu_mem_multi.csv"

_run = {"go": True}


def _stop(*_a):
    _run["go"] = False


signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)

torch.xpu.init()
ndev = torch.xpu.device_count()
with open(OUT, "w", buffering=1) as f:
    f.write("epoch_s,device_id,mem_used_mib\n")
    while _run["go"]:
        ts = time.time()
        for d in range(ndev):
            try:
                free, total = torch.xpu.mem_get_info(d)
                used_mib = (total - free) // 1024 // 1024
                f.write(f"{ts:.3f},{d},{used_mib}\n")
            except Exception:
                pass
        time.sleep(INTERVAL_S)
