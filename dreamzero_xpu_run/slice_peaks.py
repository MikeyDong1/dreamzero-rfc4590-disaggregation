#!/usr/bin/env python3
"""Slice per-N whole-device XPU peak memory from the sampler CSV using the
RUN_START/RUN_END epoch markers the harness printed to the run log.

Outputs metrics/comparison.csv and merges whole-device peaks into
output/dit_only_results.json (adds `peak_xpu_device_mib` per run).
"""
import csv
import json
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
log = (run_dir / "logs" / "run.log").read_text(errors="replace")
csv_path = run_dir / "metrics" / "xpu_memory.csv"
results_path = run_dir / "output" / "dit_only_results.json"

# Parse markers: "RUN_START N=4 epoch=1712345678.123"
starts, ends = {}, {}
for m in re.finditer(r"RUN_START N=(\d+) epoch=([\d.]+)", log):
    starts[int(m.group(1))] = float(m.group(2))
for m in re.finditer(r"RUN_END N=(\d+) epoch=([\d.]+)", log):
    ends[int(m.group(1))] = float(m.group(2))

# Load sampler rows.
samples = []
if csv_path.exists():
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                samples.append((float(row["epoch_s"]), int(row["mem_used_mib"])))
            except (ValueError, KeyError):
                pass

def peak_between(t0, t1):
    vals = [mib for (ts, mib) in samples if t0 <= ts <= t1]
    return max(vals) if vals else None

# Baseline before first run (device idle-ish with our process loading).
baseline = None
if starts:
    first_start = min(starts.values())
    pre = [mib for (ts, mib) in samples if ts < first_start]
    baseline = max(pre) if pre else None

device_peaks = {}
for n in sorted(starts):
    if n in ends:
        device_peaks[n] = peak_between(starts[n], ends[n])

# Merge into results json.
results = json.loads(results_path.read_text()) if results_path.exists() else {"runs": []}
for run in results.get("runs", []):
    n = run["num_steps"]
    run["peak_xpu_device_mib"] = device_peaks.get(n)
    run["peak_xpu_device_gib"] = (device_peaks[n] / 1024.0) if device_peaks.get(n) else None
results["baseline_device_mib_before_runs"] = baseline
results["device_peak_source"] = "xpu-smi host sampler (whole-device, 400ms)"
results_path.write_text(json.dumps(results, indent=2))

# comparison.csv
comp = run_dir / "metrics" / "comparison.csv"
with open(comp, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "denoise_steps", "prefill_s", "denoise_loop_s", "time_to_first_output_s",
        "time_to_complete_output_s", "peak_xpu_alloc_gib", "peak_xpu_device_mib",
        "peak_xpu_device_gib", "video_finite", "action_finite",
    ])
    for run in results.get("runs", []):
        w.writerow([
            run["num_steps"], f'{run["prefill_s"]:.3f}', f'{run["denoise_loop_s"]:.3f}',
            f'{run["time_to_first_output_s"]:.3f}' if run["time_to_first_output_s"] else "",
            f'{run["time_to_complete_output_s"]:.3f}',
            f'{run["peak_xpu_alloc_gib"]:.3f}',
            run.get("peak_xpu_device_mib") or "",
            f'{run["peak_xpu_device_gib"]:.3f}' if run.get("peak_xpu_device_gib") else "",
            run["video_finite"], run["action_finite"],
        ])

print("baseline_device_mib:", baseline)
print("per-N device peaks (MiB):", device_peaks)
print("wrote", comp)
print("merged into", results_path)
