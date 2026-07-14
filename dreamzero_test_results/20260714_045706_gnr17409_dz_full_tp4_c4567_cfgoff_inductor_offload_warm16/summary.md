# DreamZero FULL/Monolithic TP=4 Test - gnr17409 - SUCCESS

## Status: SUCCESS (first full end-to-end DreamZero run on this node)

- Node: gnr17409 | Model: GEAR-Dreams/DreamZero-DROID
- Docker image: vllm-omni-xpu:v0240 (sha256:8dbf8a877f8e), reused, not rebuilt
- Code: /data/vllm-omni @ prepare-dreamzero-xpu-ssh (d1bad6c) + 3 targeted source fixes (below)

## Configuration (standing setup)

| Setting | Value |
|---|---|
| Pipeline | dreamzero (full/monolithic, single stage) |
| Tensor parallel | 4 |
| Devices | physical cards 4,5,6,7 (ZE_AFFINITY_MASK) |
| CFG | OFF (cfg_scale=1.0) |
| torch.compile / inductor | ON (enforce_eager=false) |
| Denoise steps | 16 |
| Input | raw/unencoded camera MP4 (cv2, no pre-encode) |
| Layerwise CPU offload | ON (required - see root cause) |
| AR-Diffusion cudagraph warm-up | OFF (memory-spike; failed over to lazy anyway) |
| Engine | ARDiffusionEngine (session-scoped paged KV) |

## RESULTS - required deliverables

| Metric | Value |
|---|---|
| Time to completion (warm, per request) | 94.9 s  (gen 94.43s + VAE decode 0.47s) |
| Peak XPU memory | 23,255 MiB = 22.71 GiB per card (all 4 identical) |
| Warm generation (mean of 2) | 98.0 s |
| Model load (incl. offload staging + KV prealloc) | 236.8 s |
| Cold first request (load-warm + first-call inductor compile) | 116.8 s |
| Batched warm decode (2 requests) | 3.25 s |
| Output | 13-frame 640x352 mpeg4 MP4 (2.6 s), actions finite on all 3 requests |

Warm-only per spec: request 0 (cold: session init + first-call compile) excluded; requests 1-2 are the steady-state warm window.

## Root cause of the earlier DEVICE_LOST crashes (3 today, all cards 4-7 TP=4)

Out of memory, not flaky hardware. Chain: xe driver "VM worker error: -12" (-ENOMEM, GPU page-table bind failed) -> exec-queue reset -> UR_RESULT_ERROR_DEVICE_LOST. At TP=4 each B60 (22.71 GiB) holds a 2x larger DiT shard than the prior TP=8 preproc runs (which already peaked at 19.85 GiB). Plain TP=4 overcommits the card. The proven config header (dreamzero_tp4_offload_nopin.yaml) states it: "TP=4 alone OOMs the 24.5GB B60 in the attention forward; layerwise offload keeps only ~1 DiT block resident per rank so it fits." No prior full end-to-end run had ever succeeded on this node - the earlier "successful" runs were VAE/preproc-only profiling.

## Fixes applied to make it work (in /data/vllm-omni; backups in /tmp)

1. Config: enable layerwise CPU offload (enable_layerwise_offload: true, pin_cpu_memory: false). THE memory fix - keeps ~1 DiT block resident/rank so TP=4 fits in 22.71 GiB.
2. Config: disable AR-Diffusion cudagraph warm-up (ar_diffusion_kv_config.warmup_cudagraph: false). Its extra synthetic rollout spiked memory over the ceiling (error 40 = OUT_OF_RESOURCES -> DEVICE_LOST); it was already failing over to lazy capture, so disabling only removes the spike.
3. Code: inline_stage_diffusion_client.py + ar_diffusion/runner.py - OmniDiffusionRequest(prompts=[...]) -> prompt=... (the checkout client/request dataclass were out of sync; field is singular 'prompt').
4. Code: pipeline_dreamzero.py::_kv_populate_cross - the eager cross-attn KV precompute reaches into block.cross_attn.{k,v,...} WITHOUT calling block.forward(), so layerwise-offload pre/post-forward hooks never materialized those weights -> "mat1 on xpu, weight on cpu" addmm error. Added _layerwise_offload_hook() helper + onload-before / offload-after around the per-block loop (no-op when offload disabled).

Also: the test now runs the fixed /data checkout (PYTHONPATH=/workspace/vllm-omni), not the stale baked-in package (git g0807dda96, Jul 7, lacking the disaggregation refactor) that earlier runs silently imported.

## Notes / limitations

- TP=4 sits exactly at the memory ceiling (22.71 GiB). Offload is mandatory; essentially no headroom. To drop offload, use TP=8 (halves per-card shard).
- Fixes 3/4 are real bugs in the checkout's current committed state (this path had never run end-to-end). They edit shared /data/vllm-omni used by others - consider upstreaming; backups in /tmp.
- Warm gen times (94-102s) are dominated by layerwise-offload host<->device weight streaming (40 DiT blocks x 16 steps x chunks). A tradeoff forced by the 22.7 GiB cards at TP=4.
