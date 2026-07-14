#!/usr/bin/env bash
set -Eeuo pipefail
cd /workspace/vllm-omni/examples/offline_inference/dreamzero
export ZE_AFFINITY_MASK=0,1,2,3
export SYCL_UR_USE_LEVEL_ZERO_V2=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=/mnt/data
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0
export UR_L0_USE_IMMEDIATE_COMMANDLISTS=0
echo "=== TP4 raw-input start $(date) ==="
python -u timed_export.py
echo "WRAPPER_EXIT=$?"
