#!/usr/bin/env bash
# Orchestrate a PROFILED DiT-only DreamZero replay under REAL TP=4 on gnr17408,
# physical cards 4,5,6,7 (ZE_AFFINITY_MASK=4,5,6,7 -> in-container xpu:0..3).
# One request (N=16 denoise steps, matching the encoded input). DiT model ONLY:
# streams action_head.model.* and keeps each rank's 1/4 shard resident (no offload);
# the UMT5 text encoder / Wan VAE are NEVER moved to the XPU.
#
# Launches vllm-omni-xpu:latest (v0240, torch 2.12+xpu) via torchrun --nproc_per_node=4.
# Uses a DEDICATED container (does not touch the existing card0/1 pe container, nor
# any other user's container). Verifies cards 4-7 are actually free before running.
#
# Usage: run_dit_tp4_gnr17408.sh <run_id> [profile_steps] [image] [container]
set -Eeuo pipefail

# Use mikey-dev7 (torch 2.11+xpu): the PROVEN image for the direct-drive DiT-only
# harness and the one the TP=1 baseline was measured on. The v0240 :latest tag adds
# a fail-fast requiring the AR-Diffusion engine (DiT there is engine-only), which
# this harness -- driving _prefill_kv_cache/diffuse directly -- does not initialize.
RUN_ID="${1:?run_id required}"
PSTEPS="${2:-16}"
IMAGE="${3:-vllm-omni-xpu:mikey-dev7}"
CONTAINER="${4:-vllm-omni-tp4-mikey-gnr17408}"
CARDS="4,5,6,7"
TP=4
MASTER_PORT="29601"

WS="/data/sdp/mikey_dreamzero"
RUN_DIR="$WS/runs/$RUN_ID"
HF_CACHE="$WS/hf_cache"
MODEL_REPO="$HF_CACHE/hub/models--GEAR-Dreams--DreamZero-DROID"
MODEL_SNAP="$(ls -d "$MODEL_REPO"/snapshots/*/ 2>/dev/null | head -1)"
MODEL_SNAP="${MODEL_SNAP%/}"
SNAP_HASH="$(basename "$MODEL_SNAP")"
MODEL_PATH_INCTR="/workspace/model_repo/snapshots/$SNAP_HASH"
LOG="$RUN_DIR/logs/run.log"
MEM_CSV="$RUN_DIR/metrics/xpu_memory.csv"
CCL_LD="/opt/intel/oneapi/ccl/2021.15/lib:\$LD_LIBRARY_PATH:/usr/local/lib/"

if [ "$(hostname)" != "gnr17408.jf.intel.com" ] && [ "$(hostname)" != "gnr17408" ]; then
  echo "FATAL: expected gnr17408, got $(hostname)"; exit 2
fi
if [ -z "$MODEL_SNAP" ] || [ ! -f "$MODEL_SNAP/model.safetensors.index.json" ]; then
  echo "FATAL: model snapshot not found under $MODEL_REPO/snapshots"; exit 3
fi
echo "IMAGE=$IMAGE  CARDS=$CARDS  TP=$TP  PSTEPS=$PSTEPS"
echo "MODEL_PATH_INCTR=$MODEL_PATH_INCTR"

mkdir -p "$RUN_DIR"/{logs,metrics,output,config,scripts}
cp "$WS/scripts/dit_tp_profile.py" "$WS/scripts/xpu_mem_sampler_multi.py" "$RUN_DIR/scripts/"

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

echo "=== PRE-FLIGHT: verify the 4 visible cards are free (in-container) ==="
docker exec \
  --env ZE_AFFINITY_MASK=$CARDS --env SYCL_UR_USE_LEVEL_ZERO_V2=0 \
  "$CONTAINER" /bin/bash -c \
  "export LD_LIBRARY_PATH=\"$CCL_LD\"; python - <<'PY'
import torch, sys
torch.xpu.init()
n = torch.xpu.device_count()
print(f'visible XPU devices: {n}')
if n < 4:
    print(f'FATAL: expected >=4 visible cards, got {n}'); sys.exit(4)
bad = 0
for d in range(n):
    free, total = torch.xpu.mem_get_info(d)
    used_gib = (total-free)/1024**3; tot_gib = total/1024**3; free_gib = free/1024**3
    print(f'  xpu:{d}  used={used_gib:5.2f} GiB  free={free_gib:5.2f} GiB  total={tot_gib:5.2f} GiB')
    if free_gib < 15.0:
        bad += 1
if bad:
    print(f'FATAL: {bad} card(s) have <15 GiB free -- someone else is using cards {4,5,6,7}?'); sys.exit(5)
print('PREFLIGHT_OK: all 4 cards free')
PY"

echo "=== start IN-CONTAINER multi-device torch mem sampler (400ms, all 4 cards) ==="
docker exec -d \
  --env ZE_AFFINITY_MASK=$CARDS --env SYCL_UR_USE_LEVEL_ZERO_V2=0 \
  "$CONTAINER" /bin/bash -c \
  "export LD_LIBRARY_PATH=\"$CCL_LD\"; python /workspace/mikey/runs/$RUN_ID/scripts/xpu_mem_sampler_multi.py 400 /workspace/mikey/runs/$RUN_ID/metrics/xpu_memory.csv"
sleep 5
echo "sampler rows so far: $(wc -l < "$MEM_CSV" 2>/dev/null || echo 0)"

echo "=== run TP=$TP DiT harness via torchrun (nproc_per_node=$TP) ==="
echo "=== Command (TP=$TP profile_steps=$PSTEPS cards=$CARDS) ===" > "$LOG"
CMD="torchrun --nproc_per_node=$TP --master_port=$MASTER_PORT \
  /workspace/mikey/runs/$RUN_ID/scripts/dit_tp_profile.py \
  --model-path $MODEL_PATH_INCTR \
  --test-data-dir /workspace/mikey/test_data \
  --out-dir /workspace/mikey/runs/$RUN_ID/output \
  --tp-size $TP --profile-steps $PSTEPS --warmup-steps 4 --resident-blocks -1"
echo "$CMD" >> "$LOG"
echo "=== Output ===" >> "$LOG"

set +e
docker exec -i \
  --env ZE_AFFINITY_MASK=$CARDS \
  --env SYCL_UR_USE_LEVEL_ZERO_V2=0 \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env HF_HOME=/root/hf_cache --env HF_HUB_CACHE=/root/hf_cache/hub \
  --env CCL_ZE_IPC_EXCHANGE=sockets \
  -w /workspace "$CONTAINER" \
  /bin/bash -c "source /root/.bashrc 2>/dev/null; export LD_LIBRARY_PATH=\"$CCL_LD\"; $CMD" \
  2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
set -e

echo "=== stop sampler ==="
docker exec "$CONTAINER" /bin/bash -c "pkill -f xpu_mem_sampler_multi" 2>/dev/null || true
sleep 1
echo "harness exit code: $RC" | tee -a "$LOG"
echo "sampler rows: $(wc -l < "$MEM_CSV" 2>/dev/null || echo 0)"
exit $RC
