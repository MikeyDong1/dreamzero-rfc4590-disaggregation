#!/usr/bin/env bash
OUT="$1"
echo "Timestamp, DeviceId, GPU Memory Used (MiB)" > "$OUT"
while true; do
  xpu-smi dump -d 0,1,2,3 -m 18 -n 1 --date 2>/dev/null | tail -n +2 >> "$OUT"
  sleep 0.5
done
