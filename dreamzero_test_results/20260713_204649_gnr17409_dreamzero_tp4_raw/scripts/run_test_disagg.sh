#!/usr/bin/env bash
source /root/.bashrc
export LD_LIBRARY_PATH="/opt/intel/oneapi/ccl/2021.15/lib:$LD_LIBRARY_PATH:/usr/local/lib/"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ZE_AFFINITY_MASK=0,1,2,3,4,5
export SYCL_UR_USE_LEVEL_ZERO_V2=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0
export UR_L0_USE_IMMEDIATE_COMMANDLISTS=0
cd /workspace/vllm-omni/examples/offline_inference/dreamzero
echo "START_$(date +%s)"
timeout 600 python -u timed_export_disagg.py
echo "WRAPPER_EXIT=$?"
