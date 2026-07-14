#!/usr/bin/env bash
set -Eeuo pipefail
RUN_ID="20260714_062113_gnr17409_dz_DISAGG_denoisetp4_c4567_cfgoff_inductor_warm16"
IMAGE="vllm-omni-xpu:v0240"
CONTAINER="xianzhed-dz-disagg-denoisetp4"
# Physical cards: encode=1, denoise TP4=4,5,6,7, decode=0. Container renumbers to 0..5.
CARDS="1,4,5,6,7,0"
WS_HOME="$HOME/xianzhed_dz_run"; RUN_DIR="$WS_HOME/$RUN_ID"
VLLM_OMNI_HOST="/data/vllm-omni"
MODEL_HOST="/data/hub/models--GEAR-Dreams--DreamZero-DROID"
ASSETS_HOST="/data/vllm-omni/outputs/dreamzero/assets"; LOG="$RUN_DIR/logs/run.log"

[ "$(hostname)" = gnr17409 ] || { echo FATAL_wrong_host; exit 2; }
[ -f "$MODEL_HOST"/snapshots/*/model.safetensors.index.json ] || { echo FATAL_no_model; exit 3; }
[ -f "$ASSETS_HOST/exterior_image_1_left.mp4" ] || { echo FATAL_no_assets; exit 7; }

echo "=== fresh container (cards phys $CARDS -> container 0..5) ==="
docker rm -f "$CONTAINER" 2>/dev/null || true
docker run -it -d --shm-size 64g --name "$CONTAINER"   --net=host --ipc=host --privileged   -v /dev/dri/by-path:/dev/dri/by-path --device /dev/dri:/dev/dri   --env ZE_AFFINITY_MASK=$CARDS --env SYCL_UR_USE_LEVEL_ZERO_V2=0   --env VLLM_WORKER_MULTIPROC_METHOD=spawn   --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1   --env HF_HOME=/root/hf_home --env HF_HUB_CACHE=/root/hf_home/hub   --env PYTHONPATH=/workspace/vllm-omni   -v "$VLLM_OMNI_HOST":/workspace/vllm-omni   -v "$RUN_DIR/scripts":/workspace/scripts_run   -v "$RUN_DIR/config":/workspace/config_run   -v "$ASSETS_HOST":/workspace/assets_run   -v "$RUN_DIR/output":/workspace/output_run   -v "$RUN_DIR/metrics":/workspace/metrics_run   -v "$MODEL_HOST":/workspace/model_repo   -v /data/hub:/root/hf_home/hub   --entrypoint "" "$IMAGE" /bin/bash
docker ps --format '{{.Names}} :: {{.Status}}' | grep "$CONTAINER"

echo "=== preflight: 6 cards free ==="
docker exec --env ZE_AFFINITY_MASK=$CARDS --env SYCL_UR_USE_LEVEL_ZERO_V2=0 "$CONTAINER" python - <<'PY'
import torch,sys
torch.xpu.init(); n=torch.xpu.device_count(); print('visible',n)
[print(f'  xpu:{d} free={torch.xpu.mem_get_info(d)[0]/1024**3:.1f}GiB') for d in range(n)]
sys.exit(0 if n>=6 else 4)
PY

echo "=== start per-device mem sampler (400ms, all 6 cards) ==="
docker exec -d --env ZE_AFFINITY_MASK=$CARDS --env SYCL_UR_USE_LEVEL_ZERO_V2=0 "$CONTAINER"   python /workspace/scripts_run/xpu_mem_sampler_multi.py 400 /workspace/scripts_run/xpu_memory.csv
sleep 3

echo "=== run DISAGGREGATED (encode|denoise-TP4|decode, cfg off, inductor on, warm) ===" | tee "$LOG"
set +e
docker exec -i --env ZE_AFFINITY_MASK=$CARDS --env SYCL_UR_USE_LEVEL_ZERO_V2=0   --env VLLM_WORKER_MULTIPROC_METHOD=spawn --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1   --env HF_HOME=/root/hf_home --env HF_HUB_CACHE=/root/hf_home/hub   --env PYTHONPATH=/workspace/vllm-omni:/workspace/scripts_run   -w /workspace/scripts_run "$CONTAINER"   /bin/bash -c 'python /workspace/scripts_run/timed_export_disagg_denoisetp4_warm.py' 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
set -e
docker exec "$CONTAINER" pkill -f xpu_mem_sampler_multi 2>/dev/null || true
sleep 1
cp "$RUN_DIR/scripts/xpu_memory.csv" "$RUN_DIR/metrics/xpu_memory.csv" 2>/dev/null || true
echo "harness exit code: $RC" | tee -a "$LOG"
exit $RC
