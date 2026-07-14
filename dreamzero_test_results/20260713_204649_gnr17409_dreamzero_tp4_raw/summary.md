# DreamZero TP=4 raw-input test on gnr17409 - FAILED (OOM)

**Status:** FAILED - out-of-memory device crash
**Node:** gnr17409 (10.54.109.214), 8x Intel Arc Pro B60 (24480 MiB each)
**Model:** GEAR-Dreams/DreamZero-DROID (source: /mnt/data/hub/models--GEAR-Dreams--DreamZero-DROID)
**Container:** dz_tp4_dev, base image vllm-omni-xpu:v0240 (vLLM 0.24.0+xpu) with /data/vllm-omni source mounted over /workspace/vllm-omni (editable install)
**Deploy config:** vllm_omni/deploy/dreamzero_tp4.yaml — TP=4, devices 0-3, no CFG parallel, no layer offload, enforce_eager=true
**Input:** raw video (three fixed camera MP4s decoded to frames via robot_obs, NOT pre-encoded latents) via examples/offline_inference/dreamzero/timed_export.py + export_prediction_video.py:_build_observations
**Denoise steps:** 16 (repo default, unmodified)
**Proxy:** 0 (offline; container confirmed to have no internet egress)

## Environment fixes required before any test could run

The current checkout on gnr17409 (/data/vllm-omni, branch prepare-dreamzero-xpu-ssh) predated three
vLLM-0.24.0-compatibility fixes already present in the reviewer's local repo. Copied over:
1. vllm_omni/platforms/__init__.py — supports_xccl was removed from vllm.utils.torch_utils in v0.24.0
2. vllm_omni/diffusion/model_loader/diffusers_loader.py + gguf_adapters/ — download_gguf removed (GGUF migrated to plugin)
3. vllm_omni/quantization/gguf_config.py — vllm.model_executor.layers.quantization.gguf module removed

Also required HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1: without it, vLLM's model-type auto-detection
(is_mistral_model_repo) makes a live Hugging Face Hub API call that hangs indefinitely because this
container has no internet egress, even though the model is fully cached locally.

## Result

After the above fixes, the run reproducibly (3/3 attempts) crashed at the same point:

- Weights loaded successfully (~20.2 GiB/replica, ~11-14s each), all 4 workers initialized, KV-cache
  reset/init completed normally.
- Crash occurred inside DreamZero's first generation request, specifically in
  pipeline_dreamzero.py::_prefill_kv_cache -> predict_noise -> causal_wan_model.py forward -> linear.py
  tensor_model_parallel_all_reduce — i.e. the first cross-device collective communication.
- Error: level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST) on all 4 ranks
  simultaneously.
- **Peak memory immediately before the crash, per device (via sudo xpu-smi dump -m 18):**
  - Device 0: 24408.04 MiB
  - Device 1: 24479.97 MiB  <- 0.03 MiB from the card's 24480 MiB physical capacity
  - Device 2: 24455.72 MiB
  - Device 3: 24472.00 MiB
  - Aggregate: 97815.73 MiB across 4 cards

**Assessment:** this is an OOM-induced device crash, not a code bug in the disaggregation feature and
not a transient driver flake — all 4 ranks pegged within 0.03-72 MiB of the hardware ceiling at the
exact moment of failure, reproduced identically 3 times. The plain (non-offload) TP=4 deploy config
leaves effectively zero memory headroom for the KV-cache prefill's collective-communication buffers on
24GB B60 cards. The repo already contains several offload-variant TP=4 configs
(dreamzero_tp4_offload2.yaml, dreamzero_tp4_offload_nopin.yaml) which likely exist specifically to work
around this; a follow-up run with one of those is the natural next step, or reducing max_num_batched_tokens/precision.

Time to completion / decode time could not be measured since generation never completed.

## Artifacts (on gnr17409, ~/dreamzero-vllm-omni-runs/20260713_204649_gnr17409_dreamzero_tp4_raw/)

- logs/run7.log — full log of the failing run (memory-sampler-matched attempt)
- logs/run3.log through run6.log — earlier attempts (env-fix iterations; run3 hung on network, run4-6 hit the same OOM crash)
- metrics/xpu_memory_run7.csv — sudo-sourced per-device memory samples (0.5s interval) for the successful-sampling attempt; treat run1-6 CSVs as unreliable (0.00 MiB throughout - non-sudo xpu-smi is permission-scoped to 0 inside this container)
- metrics/metrics.json — structured summary
- config/effective_config.yaml, config/deploy_config_used.yaml
- scripts/run_test.sh, scripts/sample_loop.sh

## Note on memory sampling gotcha

Non-privileged `xpu-smi dump -m 18` returns 0.00 MiB inside this container/host combination regardless
of actual usage; `sudo xpu-smi dump -m 18` returns correct values. Any future profiling on this node
must use sudo.
