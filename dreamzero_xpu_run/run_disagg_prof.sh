#!/bin/bash
set -Eeuo pipefail

# DreamZero disaggregated (encode/denoise/decode) DEEP-PROFILED run on gnr17405.
# Device layout: ZE_AFFINITY_MASK=0,3,4,5,6,7 renumbers physical cards to
# container xpu:0..5:
#   xpu:0 -> phys0 (encode), xpu:1 -> phys3 (DECODE, separate),
#   xpu:2..5 -> phys4,5,6,7 (denoise TP4).
# cfg off, inductor on (denoise), 16 steps, multi-request.
#
# Per-process phase profiling: prof/sitecustomize.py is mounted and prepended to
# PYTHONPATH so Python's site module auto-imports it in the orchestrator AND in
# every spawned stage/rank worker. It writes events.<pid>.jsonl to DZ_PROF_DIR.

RUN_ID="${RUN_ID:?RUN_ID must be set}"
DZ_MODE="${DZ_MODE:-multi}"
DZ_NUM_CHUNKS="${DZ_NUM_CHUNKS:-6}"
IMAGE="vllm-omni-xpu:latest"
CONTAINER_NAME="dreamzero-disagg-prof-${DZ_MODE}-${RUN_ID}"

RUN_DIR="$HOME/dreamzero-vllm-omni-runs/${RUN_ID}"
CKPT="${RUN_DIR}/vllm-omni"
PROF_HOST="${RUN_DIR}/metrics/prof"
INDUCTOR_CACHE_DIR="$HOME/dreamzero-vllm-omni-runs/inductor_cache_disagg"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/metrics" "${RUN_DIR}/output" "${PROF_HOST}" "${INDUCTOR_CACHE_DIR}"

docker run --rm --name "${CONTAINER_NAME}" \
  --device /dev/dri -v /dev/dri:/dev/dri --group-add 44 --group-add 992 \
  --shm-size=32g \
  -v "${CKPT}":/workspace/vllm-omni \
  -v "${RUN_DIR}/prof":/workspace/prof:ro \
  -v /data/sdp_dreamzero/hf_home:/mnt/hf_cache \
  -v /data/sdp_dreamzero/assets:/workspace/vllm-omni/outputs/dreamzero/assets:ro \
  -v "${RUN_DIR}/output":/workspace/out/output \
  -v "${RUN_DIR}/metrics":/workspace/out/metrics \
  -v "${RUN_DIR}/config":/workspace/out/config:ro \
  -v "${INDUCTOR_CACHE_DIR}":/mnt/inductor_cache \
  -w /workspace/vllm-omni/examples/offline_inference/dreamzero \
  -e PYTHONPATH=/workspace/prof:/workspace/vllm-omni \
  -e ZE_AFFINITY_MASK=0,3,4,5,6,7 \
  -e SYCL_UR_USE_LEVEL_ZERO_V2=0 \
  -e SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0 \
  -e UR_L0_USE_IMMEDIATE_COMMANDLISTS=0 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e HF_HOME=/mnt/hf_cache -e HF_HUB_CACHE=/mnt/hf_cache/hub \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e http_proxy= -e https_proxy= -e HTTP_PROXY= -e HTTPS_PROXY= -e proxy=0 \
  -e LD_LIBRARY_PATH="/opt/intel/oneapi/ccl/2021.15/lib:/opt/intel/oneapi/tcm/1.4/lib:/opt/intel/oneapi/umf/1.0/lib:/opt/intel/oneapi/tbb/2022.3/env/../lib/intel64/gcc4.8:/opt/intel/oneapi/pti/0.16/lib:/opt/intel/oneapi/mpi/2021.17/opt/mpi/libfabric/lib:/opt/intel/oneapi/mpi/2021.17/lib:/opt/intel/oneapi/mkl/2025.3/lib:/opt/intel/oneapi/dnnl/2025.3/lib:/opt/intel/oneapi/debugger/2025.3/opt/debugger/lib:/opt/intel/oneapi/compiler/2025.3/opt/compiler/lib:/opt/intel/oneapi/compiler/2025.3/lib:/usr/local/lib/" \
  -e CCL_SYCL_ALLGATHERV_TMP_BUF=0 \
  -e TORCHINDUCTOR_CACHE_DIR=/mnt/inductor_cache/torchinductor \
  -e TRITON_CACHE_DIR=/mnt/inductor_cache/triton \
  -e DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA \
  -e DZ_DEPLOY=/workspace/out/config/deploy_config_used.yaml \
  -e DZ_MODE="${DZ_MODE}" \
  -e DZ_NUM_CHUNKS="${DZ_NUM_CHUNKS}" \
  -e DZ_METRICS=/workspace/out/metrics \
  -e DZ_PROF_DIR=/workspace/out/metrics/prof \
  -e DZ_PROFILE="${DZ_PROFILE:-1}" \
  --entrypoint bash \
  "${IMAGE}" \
  -c 'python -u timed_disagg_prof.py'
echo "WRAPPER_EXIT=$?"
