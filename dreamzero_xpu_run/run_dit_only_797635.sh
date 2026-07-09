#!/usr/bin/env bash
# Orchestrate the DiT-only DreamZero replay on node 797635, card0 (TP=1), K=10.
# - starts a host-side xpu-smi memory sampler (device 0),
# - runs the DiT-only harness inside the container (sweeping steps 4,8,16),
# - stops the sampler,
# - slices per-N whole-device peak from the sampler CSV using RUN_START/RUN_END markers.
set -Eeuo pipefail

RUN_ID="${1:?run_id required}"
CONTAINER="vllm-omni-dev-mikey"
IMAGE="vllm-omni-xpu:latest"
HOME_DIR="$HOME"
RUN_DIR="$HOME_DIR/mikey_dreamzero/runs/$RUN_ID"
TEST_DATA="$HOME_DIR/mikey_dreamzero/test_data"
MODEL_SNAP="$HOME_DIR/.cache/huggingface/hub/models--GEAR-Dreams--DreamZero-DROID/snapshots/96ad344138c66e82536422432ad742f015784942"
LOG="$RUN_DIR/logs/run.log"
MEM_CSV="$RUN_DIR/metrics/xpu_memory.csv"
SAMPLER="$RUN_DIR/scripts/xpu_mem_sampler.sh"

mkdir -p "$RUN_DIR"/{logs,metrics,output,config}

echo "=== ensure container up ==="
if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  docker rm -f "$CONTAINER" 2>/dev/null || true
  docker run -it -d \
    --shm-size 10g \
    --name "$CONTAINER" \
    --net=host --ipc=host --privileged \
    -v /dev/dri/by-path:/dev/dri/by-path \
    --device /dev/dri:/dev/dri \
    -v "$HOME_DIR/.cache/huggingface":/root/hf_cache \
    -v "$HOME_DIR/mikey_dreamzero":/workspace/mikey \
    -v "$MODEL_SNAP":/workspace/model \
    --env HF_HOME=/root/hf_cache \
    --env HF_HUB_CACHE=/root/hf_cache/hub \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --entrypoint "" \
    "$IMAGE" /bin/bash
fi
docker ps --format '{{.Names}} :: {{.Image}} :: {{.Status}}' | grep "$CONTAINER"

echo "=== start host xpu-smi sampler (device 0, 400ms) ==="
chmod +x "$SAMPLER"
nohup bash "$SAMPLER" 0 400 "$MEM_CSV" > "$RUN_DIR/logs/sampler.log" 2>&1 &
SAMPLER_PID=$!
echo "sampler PID=$SAMPLER_PID"
sleep 2

echo "=== run DiT-only harness in container ==="
echo "=== Command ===" > "$LOG"
CMD="python /workspace/mikey/runs/$RUN_ID/scripts/dit_only_replay.py \
  --model-path /workspace/model \
  --test-data-dir /workspace/mikey/test_data \
  --out-dir /workspace/mikey/runs/$RUN_ID/output \
  --resident-blocks 10 --steps 4,8,16"
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
  /bin/bash -c "source /root/.bashrc && export LD_LIBRARY_PATH=\"/opt/intel/oneapi/ccl/2021.15/lib:\$LD_LIBRARY_PATH:/usr/local/lib/\" && $CMD" \
  2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
set -e

echo "=== stop sampler ==="
kill -TERM "$SAMPLER_PID" 2>/dev/null || true
sleep 1
echo "harness exit code: $RC" | tee -a "$LOG"
echo "sampler rows: $(wc -l < "$MEM_CSV" 2>/dev/null || echo 0)"
exit $RC
