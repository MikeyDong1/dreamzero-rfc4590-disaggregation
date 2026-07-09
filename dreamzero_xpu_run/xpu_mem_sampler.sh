#!/usr/bin/env bash
# Sample per-device XPU memory-used (MiB) via xpu-smi at a fixed interval, writing
# CSV rows: epoch_seconds,device_id,mem_used_mib. Runs until killed (SIGTERM/SIGINT).
# Usage: xpu_mem_sampler.sh <device_id> <interval_ms> <out_csv>
set -u
DEV="${1:-0}"
INTERVAL_MS="${2:-400}"
OUT="${3:-/tmp/xpu_mem.csv}"
INTERVAL_S=$(awk "BEGIN{printf \"%.3f\", ${INTERVAL_MS}/1000.0}")

echo "epoch_s,device_id,mem_used_mib" > "$OUT"
trap 'exit 0' TERM INT
while true; do
  TS=$(date +%s.%N)
  # xpu-smi stats -d N prints a "GPU Memory Used (MiB)" row; grab the integer.
  MEM=$(xpu-smi stats -d "$DEV" 2>/dev/null | awk -F'|' '/GPU Memory Used/ {gsub(/[^0-9]/,"",$3); print $3; exit}')
  if [ -n "$MEM" ]; then
    echo "${TS},${DEV},${MEM}" >> "$OUT"
  fi
  sleep "$INTERVAL_S"
done
