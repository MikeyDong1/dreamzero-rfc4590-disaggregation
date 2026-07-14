#!/usr/bin/env bash
# Real-serving-path DreamZero test: TP=4 on cards 4,5,6,7, RAW camera MP4 input,
# on gnr17409. Uses the maintainer-proven standard-engine path (KV-connector,
# NOT the AR-Diffusion engine -- that combination crashed with DEVICE_LOST on
# gnr17408; see /data/vllm-omni/logs/dreamzero_tp8_bmg_proxy0.log for the
# working reference: standard engine + TP=8 produced finite actions there).
#
# Usage: run_dz_serve_tp4_gnr17409.sh <run_id> [image] [container]
set -Eeuo pipefail

RUN_ID="${1:?run_id required}"
IMAGE="${2:-vllm-omni-xpu:v0240}"
CONTAINER="${3:-xianzhed-dz-serve-tp4-gnr17409}"
CARDS="4,5,6,7"
TP=4

WS_HOME="$HOME/xianzhed_dz_run"
RUN_DIR="$WS_HOME/$RUN_ID"
VLLM_OMNI_HOST="/data/vllm-omni"
MODEL_HOST="/data/hub/models--GEAR-Dreams--DreamZero-DROID"
MODEL_SNAP="$(ls -d "$MODEL_HOST"/snapshots/*/ 2>/dev/null | head -1)"
MODEL_SNAP="${MODEL_SNAP%/}"
SNAP_HASH="$(basename "$MODEL_SNAP")"
MODEL_PATH_INCTR="/workspace/model_repo/snapshots/$SNAP_HASH"
ASSETS_HOST="/data/vllm-omni/outputs/dreamzero/assets"
LOG="$RUN_DIR/logs/run.log"
MEM_CSV="$RUN_DIR/metrics/xpu_memory.csv"
REPORT_DIR_INCTR="/workspace/probe_reports"

if [ "$(hostname)" != "gnr17409" ]; then
  echo "FATAL: expected gnr17409, got $(hostname)"; exit 2
fi
if [ -z "$MODEL_SNAP" ] || [ ! -f "$MODEL_SNAP/model.safetensors.index.json" ]; then
  echo "FATAL: model snapshot not found under $MODEL_HOST/snapshots"; exit 3
fi
if [ ! -f "$ASSETS_HOST/exterior_image_1_left.mp4" ]; then
  echo "FATAL: expected camera assets under $ASSETS_HOST"; exit 7
fi
if [ ! -f "$RUN_DIR/config/dreamzero_tp4_cards4567.yaml" ]; then
  echo "FATAL: expected deploy config at $RUN_DIR/config/dreamzero_tp4_cards4567.yaml"; exit 6
fi
echo "MODEL_PATH_INCTR=$MODEL_PATH_INCTR  IMAGE=$IMAGE  CARDS=$CARDS  TP=$TP"

echo "=== ensure DEDICATED container up (cards $CARDS) ==="
if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  docker rm -f "$CONTAINER" 2>/dev/null || true
  docker run -it -d \
    --shm-size 32g --name "$CONTAINER" \
    --net=host --ipc=host --privileged \
    -v /dev/dri/by-path:/dev/dri/by-path \
    --device /dev/dri:/dev/dri \
    --env ZE_AFFINITY_MASK=$CARDS \
    --env SYCL_UR_USE_LEVEL_ZERO_V2=0 \
    --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --env HF_HOME=/root/hf_home --env HF_HUB_CACHE=/root/hf_home/hub \
    --env TP_PROBE_REPORT_DIR=$REPORT_DIR_INCTR \
    -v "$VLLM_OMNI_HOST":/workspace/vllm-omni \
    -v "$RUN_DIR/scripts":/workspace/scripts_run \
    -v "$RUN_DIR/config":/workspace/config_run \
    -v "$ASSETS_HOST":/workspace/assets_run \
    -v "$RUN_DIR/output":/workspace/output_run \
    -v "$RUN_DIR/probe_reports":$REPORT_DIR_INCTR \
    -v "$MODEL_HOST":/workspace/model_repo \
    -v /data/hub:/root/hf_home/hub \
    --entrypoint "" \
    "$IMAGE" /bin/bash
fi
docker ps --format '{{.Names}} :: {{.Image}} :: {{.Status}}' | grep "$CONTAINER"

echo "=== PRE-FLIGHT: verify the 4 visible cards are free (in-container) ==="
docker exec \
  --env ZE_AFFINITY_MASK=$CARDS --env SYCL_UR_USE_LEVEL_ZERO_V2=0 \
  "$CONTAINER" /bin/bash -c \
  "python - <<'PY'
import torch, sys
torch.xpu.init()
n = torch.xpu.device_count()
print(f'visible XPU devices: {n}')
if n < 4:
    print(f'FATAL: expected >=4 visible cards, got {n}'); sys.exit(4)
bad = 0
for d in range(n):
    free, total = torch.xpu.mem_get_info(d)
    used_gib = (total-free)/1024**3; free_gib = free/1024**3
    print(f'  xpu:{d} used={used_gib:5.2f} GiB free={free_gib:5.2f} GiB')
    if free_gib < 15.0:
        bad += 1
if bad:
    print(f'FATAL: {bad} card(s) have <15 GiB free'); sys.exit(5)
print('PREFLIGHT_OK: all 4 cards free')
PY"

echo "=== start IN-CONTAINER multi-device torch mem sampler (400ms, all 4 cards) ==="
docker exec -d \
  --env ZE_AFFINITY_MASK=$CARDS --env SYCL_UR_USE_LEVEL_ZERO_V2=0 \
  "$CONTAINER" /bin/bash -c \
  "python /workspace/scripts_run/xpu_mem_sampler_multi.py 400 /workspace/scripts_run/xpu_memory.csv"
sleep 3

echo "=== run DreamZero serving-path probe (TP=$TP, raw camera input) ==="
echo "=== Command ===" > "$LOG"
CMD="python /workspace/scripts_run/dz_serve_tp_probe_17409.py \
  --model $MODEL_PATH_INCTR \
  --deploy-config /workspace/config_run/dreamzero_tp4_cards4567.yaml \
  --video-dir /workspace/assets_run \
  --output-dir /workspace/output_run \
  --report-dir $REPORT_DIR_INCTR \
  --tp-size $TP \
  --dreamzero-example-dir /workspace/vllm-omni/examples/offline_inference/dreamzero"
echo "$CMD" >> "$LOG"
echo "=== Output ===" >> "$LOG"

set +e
docker exec -i \
  --env ZE_AFFINITY_MASK=$CARDS \
  --env SYCL_UR_USE_LEVEL_ZERO_V2=0 \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env HF_HOME=/root/hf_home --env HF_HUB_CACHE=/root/hf_home/hub \
  --env TP_PROBE_REPORT_DIR=$REPORT_DIR_INCTR \
  --env PYTHONPATH=/workspace/scripts_run \
  -w /workspace/scripts_run "$CONTAINER" \
  /bin/bash -c "source /root/.bashrc 2>/dev/null; $CMD" \
  2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
set -e

echo "=== stop sampler ==="
docker exec "$CONTAINER" /bin/bash -c "pkill -f xpu_mem_sampler_multi" 2>/dev/null || true
sleep 1

echo "=== copy artifacts ==="
cp "$RUN_DIR/scripts/xpu_memory.csv" "$MEM_CSV" 2>/dev/null || echo "no xpu_memory.csv"
echo "output dir contents:"
ls -la "$RUN_DIR/output/" 2>/dev/null || true
echo "probe reports:"
ls -la "$RUN_DIR/probe_reports/"

echo "harness exit code: $RC" | tee -a "$LOG"
exit $RC
