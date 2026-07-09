#!/usr/bin/env bash
# Orchestrate a CFG/COMPILE DiT-only DreamZero replay on node 797635 (srf797635),
# card0 (BDF 12:00.0 = renderD128 = ZE_AFFINITY_MASK=0), inside the torch 2.12 /
# vllm 0.24 image (docker-image-comp, id 484643ceab2c -- the SAME image as the
# gnr17408 runs, so the only cross-node variable is the hardware/host).
#
# 797635 has 2x ~32 GB B60 (32656 MiB, BIGGER than gnr17408's 23256 MiB) and host
# xpu-smi works, but we sample whole-device memory IN-CONTAINER via
# torch.xpu.mem_get_info for parity with the gnr17408 bundles.
#
# Usage: run_dit_cfg_compile_797635.sh <run_id> <K> <cfg on|off> <compile eager|inductor> <profile_steps> [image] [container]
set -Eeuo pipefail

RUN_ID="${1:?run_id required}"
K="${2:?K required}"
CFG="${3:-off}"
COMPILE="${4:-inductor}"
PSTEPS="${5:-16}"
IMAGE="${6:-vllm-omni-xpu:mikey-024}"
CONTAINER="${7:-vllm-omni-cc-mikey-797635-024}"
WS="/home/sdp/mikey_dreamzero"
RUN_DIR="$WS/runs/$RUN_ID"
HF_CACHE="$WS/hf_cache"
MODEL_REPO="$HF_CACHE/hub/models--GEAR-Dreams--DreamZero-DROID"
MODEL_SNAP="$(ls -d "$MODEL_REPO"/snapshots/*/ 2>/dev/null | head -1)"; MODEL_SNAP="${MODEL_SNAP%/}"
SNAP_HASH="$(basename "$MODEL_SNAP")"
MODEL_PATH_INCTR="/workspace/model_repo/snapshots/$SNAP_HASH"
LOG="$RUN_DIR/logs/run.log"
IND_CACHE="/workspace/mikey/inductor_cache"

if [ -z "$MODEL_SNAP" ] || [ ! -f "$MODEL_SNAP/model.safetensors.index.json" ]; then
  echo "FATAL: model snapshot not found under $MODEL_REPO/snapshots"; exit 3
fi
echo "IMAGE=$IMAGE  CFG=$CFG  COMPILE=$COMPILE  K=$K  PSTEPS=$PSTEPS"

mkdir -p "$RUN_DIR"/{logs,metrics,output,config,scripts}
mkdir -p "$WS/inductor_cache"
cp "$WS/scripts/dit_cfg_compile_profile.py" "$WS/scripts/xpu_mem_sampler_torch.py" "$RUN_DIR/scripts/"

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

echo "=== run CFG/COMPILE DiT harness in container ==="
echo "=== Command (K=$K cfg=$CFG compile=$COMPILE psteps=$PSTEPS) ===" > "$LOG"
CMD="python /workspace/mikey/runs/$RUN_ID/scripts/dit_cfg_compile_profile.py \
  --model-path $MODEL_PATH_INCTR \
  --test-data-dir /workspace/mikey/test_data \
  --out-dir /workspace/mikey/runs/$RUN_ID/output \
  --resident-blocks $K --cfg $CFG --compile $COMPILE \
  --clean-steps 4,8,16 --profile-steps $PSTEPS --warmup-steps 4"
echo "$CMD" >> "$LOG"
echo "=== Output ===" >> "$LOG"

set +e
docker exec -i \
  --env ZE_AFFINITY_MASK=0 \
  --env SYCL_UR_USE_LEVEL_ZERO_V2=0 \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env HF_HOME=/root/hf_cache --env HF_HUB_CACHE=/root/hf_cache/hub \
  --env TORCHINDUCTOR_CACHE_DIR="$IND_CACHE" \
  --env TORCHINDUCTOR_FX_GRAPH_CACHE=1 \
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
