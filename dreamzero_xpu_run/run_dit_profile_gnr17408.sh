#!/usr/bin/env bash
# Orchestrate a PROFILED DiT-only DreamZero replay on gnr17408 card0 (TP=1) inside
# the vllm 0.24 / torch 2.12 image (vllm-omni-xpu:latest). Mirrors
# run_dit_only_gnr17408.sh but runs dit_only_profile.py (torch.profiler CPU+XPU)
# for ONE warm N and exports chrome trace + op tables + profile_summary.json.
#
# Usage: run_dit_profile_gnr17408.sh <run_id> <K> <profile_steps> [image] [container]
set -Eeuo pipefail

RUN_ID="${1:?run_id required}"
K="${2:?resident-blocks K required}"
PSTEPS="${3:-8}"
IMAGE="${4:-vllm-omni-xpu:latest}"
CONTAINER="${5:-vllm-omni-prof-mikey-gnr17408-024}"
WS="/data/sdp/mikey_dreamzero"
RUN_DIR="$WS/runs/$RUN_ID"
HF_CACHE="$WS/hf_cache"
MODEL_REPO="$HF_CACHE/hub/models--GEAR-Dreams--DreamZero-DROID"
MODEL_SNAP="$(ls -d "$MODEL_REPO"/snapshots/*/ 2>/dev/null | head -1)"
MODEL_SNAP="${MODEL_SNAP%/}"
SNAP_HASH="$(basename "$MODEL_SNAP")"
MODEL_PATH_INCTR="/workspace/model_repo/snapshots/$SNAP_HASH"
LOG="$RUN_DIR/logs/run.log"

if [ -z "$MODEL_SNAP" ] || [ ! -f "$MODEL_SNAP/model.safetensors.index.json" ]; then
  echo "FATAL: model snapshot not found under $MODEL_REPO/snapshots"; exit 3
fi
echo "IMAGE=$IMAGE"
echo "MODEL_PATH_INCTR=$MODEL_PATH_INCTR"

mkdir -p "$RUN_DIR"/{logs,metrics,output,config,scripts}
cp "$WS/scripts/dit_only_profile.py" "$WS/scripts/xpu_mem_sampler_torch.py" "$RUN_DIR/scripts/"

echo "=== ensure container up (card0, ZE_AFFINITY_MASK=0) ==="
if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  docker rm -f "$CONTAINER" 2>/dev/null || true
  docker run -it -d \
    --shm-size 10g --name "$CONTAINER" \
    --net=host --ipc=host --privileged \
    -v /dev/dri/by-path:/dev/dri/by-path \
    --device /dev/dri:/dev/dri \
    --env ZE_AFFINITY_MASK=0 \
    --env SYCL_UR_USE_LEVEL_ZERO_V2=0 \
    --env HF_HOME=/root/hf_cache \
    --env HF_HUB_CACHE=/root/hf_cache/hub \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    -v "$HF_CACHE":/root/hf_cache \
    -v "$WS":/workspace/mikey \
    -v "$MODEL_REPO":/workspace/model_repo \
    --entrypoint "" \
    "$IMAGE" /bin/bash
fi
docker ps --format '{{.Names}} :: {{.Image}} :: {{.Status}}' | grep "$CONTAINER"

echo "=== start IN-CONTAINER torch mem sampler (device 0, 400ms) ==="
docker exec -d \
  --env ZE_AFFINITY_MASK=0 --env SYCL_UR_USE_LEVEL_ZERO_V2=0 \
  "$CONTAINER" /bin/bash -c \
  "export LD_LIBRARY_PATH=\"/opt/intel/oneapi/ccl/2021.15/lib:\$LD_LIBRARY_PATH:/usr/local/lib/\"; python /workspace/mikey/runs/$RUN_ID/scripts/xpu_mem_sampler_torch.py 0 400 /workspace/mikey/runs/$RUN_ID/metrics/xpu_memory.csv"
sleep 5

echo "=== run PROFILED DiT harness in container (K=$K, profile_steps=$PSTEPS) ==="
echo "=== Command (K=$K profile_steps=$PSTEPS) ===" > "$LOG"
CMD="python /workspace/mikey/runs/$RUN_ID/scripts/dit_only_profile.py \
  --model-path $MODEL_PATH_INCTR \
  --test-data-dir /workspace/mikey/test_data \
  --out-dir /workspace/mikey/runs/$RUN_ID/output \
  --resident-blocks $K --profile-steps $PSTEPS --warmup-steps 4"
echo "$CMD" >> "$LOG"
echo "=== Output ===" >> "$LOG"

set +e
docker exec -i \
  --env ZE_AFFINITY_MASK=0 \
  --env SYCL_UR_USE_LEVEL_ZERO_V2=0 \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env HF_HOME=/root/hf_cache --env HF_HUB_CACHE=/root/hf_cache/hub \
  -w /workspace "$CONTAINER" \
  /bin/bash -c "source /root/.bashrc 2>/dev/null; export LD_LIBRARY_PATH=\"/opt/intel/oneapi/ccl/2021.15/lib:\$LD_LIBRARY_PATH:/usr/local/lib/\"; $CMD" \
  2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
set -e

echo "=== stop sampler ==="
docker exec "$CONTAINER" /bin/bash -c "pkill -f xpu_mem_sampler" 2>/dev/null || true
sleep 1
echo "harness exit code: $RC" | tee -a "$LOG"
echo "sampler rows: $(wc -l < "$RUN_DIR/metrics/xpu_memory.csv" 2>/dev/null || echo 0)"
exit $RC
