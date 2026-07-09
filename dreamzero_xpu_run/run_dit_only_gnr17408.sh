#!/usr/bin/env bash
# Orchestrate the DiT-only DreamZero replay on node gnr17408, card0 (TP=1).
# gnr17408 differs from 797635: the HOST has no working Level-Zero/xpu-smi, so the
# whole-device memory sampler must run INSIDE the container (docker exec -d). Card0
# is a 23256 MiB Arc Pro B60 (smaller than 797635's 32656 MiB) -> K is a parameter
# so the caller can drive the K=10 -> 8 -> 5 fallback on OOM.
#
# Usage: run_dit_only_gnr17408.sh <run_id> <K> [steps]
set -Eeuo pipefail

RUN_ID="${1:?run_id required}"
K="${2:?resident-blocks K required}"
STEPS="${3:-4,8,16}"
IMAGE="${4:-vllm-omni-xpu:mikey-dev7}"
CONTAINER="${5:-vllm-omni-dev-mikey-gnr17408}"
WS="/data/sdp/mikey_dreamzero"
RUN_DIR="$WS/runs/$RUN_ID"
TEST_DATA="$WS/test_data"
HF_CACHE="$WS/hf_cache"
# HF cache stores config.json etc. as ../../blobs symlinks relative to the snapshot
# dir, so we must mount the WHOLE repo dir (not just the snapshot subdir) or those
# symlinks dangle in-container. Mount repo at /workspace/model_repo and point
# --model-path at the snapshot subpath INSIDE it.
MODEL_REPO="$HF_CACHE/hub/models--GEAR-Dreams--DreamZero-DROID"
MODEL_SNAP="$(ls -d "$MODEL_REPO"/snapshots/*/ 2>/dev/null | head -1)"
MODEL_SNAP="${MODEL_SNAP%/}"
SNAP_HASH="$(basename "$MODEL_SNAP")"
MODEL_PATH_INCTR="/workspace/model_repo/snapshots/$SNAP_HASH"
LOG="$RUN_DIR/logs/run.log"
MEM_CSV="$RUN_DIR/metrics/xpu_memory.csv"

if [ -z "$MODEL_SNAP" ] || [ ! -f "$MODEL_SNAP/model.safetensors.index.json" ]; then
  echo "FATAL: model snapshot not found under $MODEL_REPO/snapshots (index.json missing)"; exit 3
fi
echo "MODEL_SNAP=$MODEL_SNAP"
echo "MODEL_PATH_INCTR=$MODEL_PATH_INCTR"

mkdir -p "$RUN_DIR"/{logs,metrics,output,config,scripts}
cp "$WS/scripts/dit_only_replay.py" "$WS/scripts/slice_peaks.py" \
   "$WS/scripts/xpu_mem_sampler.sh" "$WS/scripts/xpu_mem_sampler_torch.py" "$RUN_DIR/scripts/"

echo "=== ensure container up (card0, ZE_AFFINITY_MASK=0) ==="
if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  docker rm -f "$CONTAINER" 2>/dev/null || true
  docker run -it -d \
    --shm-size 10g \
    --name "$CONTAINER" \
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
# host xpu-smi/L0 is dead here AND the dev7 image ships no xpu-smi binary, so sample
# whole-device memory via torch.xpu.mem_get_info (total-free) from inside the container.
docker exec -d \
  --env ZE_AFFINITY_MASK=0 --env SYCL_UR_USE_LEVEL_ZERO_V2=0 \
  "$CONTAINER" /bin/bash -c \
  "export LD_LIBRARY_PATH=\"/opt/intel/oneapi/ccl/2021.15/lib:\$LD_LIBRARY_PATH:/usr/local/lib/\"; python /workspace/mikey/runs/$RUN_ID/scripts/xpu_mem_sampler_torch.py 0 400 /workspace/mikey/runs/$RUN_ID/metrics/xpu_memory.csv"
sleep 5
echo "sampler rows so far: $(wc -l < "$MEM_CSV" 2>/dev/null || echo 0)"

echo "=== run DiT-only harness in container (K=$K, steps=$STEPS) ==="
echo "=== Command (K=$K steps=$STEPS) ===" > "$LOG"
CMD="python /workspace/mikey/runs/$RUN_ID/scripts/dit_only_replay.py \
  --model-path $MODEL_PATH_INCTR \
  --test-data-dir /workspace/mikey/test_data \
  --out-dir /workspace/mikey/runs/$RUN_ID/output \
  --resident-blocks $K --steps $STEPS --warmup-steps 4"
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
echo "sampler rows: $(wc -l < "$MEM_CSV" 2>/dev/null || echo 0)"
exit $RC
