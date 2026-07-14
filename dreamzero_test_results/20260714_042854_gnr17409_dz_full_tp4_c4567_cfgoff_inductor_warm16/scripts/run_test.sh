#!/usr/bin/env bash
# FULL/monolithic DreamZero: TP=4 cards 4,5,6,7, CFG off, inductor ON, warm-only,
# denoise_steps=16, raw camera MP4 input. gnr17409.
set -Eeuo pipefail

RUN_ID="20260714_042854_gnr17409_dz_full_tp4_c4567_cfgoff_inductor_warm16"
IMAGE="vllm-omni-xpu:v0240"
CONTAINER="xianzhed-dz-full-tp4-c4567-cfgoff-inductor"
CARDS="4,5,6,7"

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

if [ "$(hostname)" != "gnr17409" ]; then
  echo "FATAL: expected gnr17409, got $(hostname)"; exit 2
fi
if [ ! -f "$MODEL_SNAP/model.safetensors.index.json" ]; then
  echo "FATAL: model snapshot not found under $MODEL_HOST/snapshots"; exit 3
fi
if [ ! -f "$ASSETS_HOST/exterior_image_1_left.mp4" ]; then
  echo "FATAL: expected camera assets under $ASSETS_HOST"; exit 7
fi
echo "MODEL_PATH_INCTR=$MODEL_PATH_INCTR  IMAGE=$IMAGE  CARDS=$CARDS"

echo "=== ensure DEDICATED container up (cards $CARDS) ==="
docker rm -f "$CONTAINER" 2>/dev/null || true
docker run -it -d   --shm-size 32g --name "$CONTAINER"   --net=host --ipc=host --privileged   -v /dev/dri/by-path:/dev/dri/by-path   --device /dev/dri:/dev/dri   --env ZE_AFFINITY_MASK=$CARDS   --env SYCL_UR_USE_LEVEL_ZERO_V2=0   --env VLLM_WORKER_MULTIPROC_METHOD=spawn   --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1   --env HF_HOME=/root/hf_home --env HF_HUB_CACHE=/root/hf_home/hub   -v "$VLLM_OMNI_HOST":/workspace/vllm-omni   -v "$RUN_DIR/scripts":/workspace/scripts_run   -v "$RUN_DIR/config":/workspace/config_run   -v "$ASSETS_HOST":/workspace/assets_run   -v "$RUN_DIR/output":/workspace/output_run   -v "$RUN_DIR/metrics":/workspace/metrics_run   -v "$MODEL_HOST":/workspace/model_repo   -v /data/hub:/root/hf_home/hub   --entrypoint ""   "$IMAGE" /bin/bash
docker ps --format '{{.Names}} :: {{.Image}} :: {{.Status}}' | grep "$CONTAINER"

echo "=== PRE-FLIGHT: verify the 4 visible cards are free (in-container) ==="
docker exec   --env ZE_AFFINITY_MASK=$CARDS --env SYCL_UR_USE_LEVEL_ZERO_V2=0   "$CONTAINER" /bin/bash -c   "python - <<'PY'
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
docker exec -d   --env ZE_AFFINITY_MASK=$CARDS --env SYCL_UR_USE_LEVEL_ZERO_V2=0   "$CONTAINER" /bin/bash -c   "python /workspace/scripts_run/xpu_mem_sampler_multi.py 400 /workspace/scripts_run/xpu_memory.csv"
sleep 3

echo "=== run DreamZero FULL/monolithic TP4 probe (cards 4-7, cfg off, inductor on, warm) ==="
echo "=== Command ===" > "$LOG"
CMD="python /workspace/scripts_run/timed_export_full_tp4_c4567_cfgoff_inductor_warm.py"
echo "$CMD" >> "$LOG"
echo "=== Output ===" >> "$LOG"

set +e
docker exec -i   --env ZE_AFFINITY_MASK=$CARDS   --env SYCL_UR_USE_LEVEL_ZERO_V2=0   --env VLLM_WORKER_MULTIPROC_METHOD=spawn   --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1   --env HF_HOME=/root/hf_home --env HF_HUB_CACHE=/root/hf_home/hub   --env PYTHONPATH=/workspace/scripts_run   -w /workspace/scripts_run "$CONTAINER"   /bin/bash -c "source /root/.bashrc 2>/dev/null; $CMD"   2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
set -e

echo "=== stop sampler ==="
docker exec "$CONTAINER" /bin/bash -c "pkill -f xpu_mem_sampler_multi" 2>/dev/null || true
sleep 1

echo "=== copy artifacts ==="
cp "$RUN_DIR/scripts/xpu_memory.csv" "$RUN_DIR/metrics/xpu_memory.csv" 2>/dev/null || echo "no xpu_memory.csv"
ls -la "$RUN_DIR/output/" 2>/dev/null || true
ls -la "$RUN_DIR/metrics/" 2>/dev/null || true

echo "harness exit code: $RC" | tee -a "$LOG"
exit $RC
