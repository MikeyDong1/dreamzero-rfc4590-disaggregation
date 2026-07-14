# DreamZero FULL/Monolithic TP=4 Test -- gnr17409 -- FAILED

## Status: FAILED (hardware-level GPU device loss)

- **Node**: gnr17409
- **Model**: GEAR-Dreams/DreamZero-DROID (snapshot 96ad344138c66e82536422432ad742f015784942)
- **vLLM-Omni checkout**: /data/vllm-omni, branch `prepare-dreamzero-xpu-ssh`, commit `d1bad6c57d2c8f663a21adc78817ce28a34e9a86` (dirty -- local uncommitted changes already present on the node from prior sessions)
- **Docker image**: `vllm-omni-xpu:v0240` (`sha256:8dbf8a877f8e8f591399f1a6609d820f2fd6132300e1ee0f7a4d93d3e37e542a`), reused (not rebuilt)

## Configuration (standing test setup)

| Setting | Value |
|---|---|
| Pipeline | `dreamzero` (full/monolithic, single stage) |
| Tensor parallel size | 4 |
| Physical devices | cards 4, 5, 6, 7 (`ZE_AFFINITY_MASK=4,5,6,7`) |
| CFG | OFF (`cfg_scale: 1.0`) |
| torch.compile / inductor | ON (`enforce_eager: false`) |
| Denoise steps | 16 (`num_inference_steps: 16`) |
| Input | raw/unencoded camera MP4s (loaded directly via cv2, no pre-encoding cache) |
| Measurement intent | warm-only (request 0 discarded as cold/compile; requests 1-2 as warm) |
| Engine backend | `vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine` (required for monolithic DreamZero -- session-scoped paged KV) |

## Required metrics

| Metric | Value |
|---|---|
| Time to completion (warm) | **N/A -- run crashed before any generation completed** |
| Model load time | 28.71 s (succeeded) |
| Time to first output | N/A (crashed on first request) |
| Decode time | N/A |
| Peak XPU memory | ~0 MiB observed (crash occurred before inference tensors were allocated; sampler only captured idle post-load memory) |

## Failure

```
RuntimeError: level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)
```
Raised inside `DreamZeroPipeline.forward()` at the very first `torch.tensor()` allocation (`embodiment_id`) immediately after a successful model load -- i.e. before any meaningful compute or the inductor-compiled path was ever exercised.

**This is a reproducible hardware/driver-level fault, not a config or script bug:**

1. **18:31 UTC today** (prior session, before this one): identical crash, same 4 physical cards (4,5,6,7), TP=4, AR-Diffusion engine, `enforce_eager: true` (no inductor).
2. **04:40 UTC this session** (1st attempt, reused a stale container from an earlier failed import-path attempt): identical crash.
3. **04:41 UTC this session** (2nd attempt, fully fresh container + preflight-verified healthy devices): identical crash again.

`dmesg` confirms real GPU hardware events at each crash timestamp:
```
xe 0000:97:00.0 (card4): [drm] VM worker error: -12
xe 0000:a9:00.0 (card5): [drm] VM worker error: -12
xe 0000:cb:00.0 (card6): [drm] VM worker error: -12
xe 0000:ba:00.0 (card7): [drm] VM worker error: -12
... [drm] exec queue reset detected  (x multiple, same 4 devices)
```

Between crashes, all 4 devices report healthy in isolation (idle matmul + `torch.xpu.synchronize` succeed, 22.71 GiB free each). The fault appears specific to this 4-GPU set (cards 4-7) under TP=4 + AR-Diffusion-engine collective/distributed load on this node, at this point in time.

Per the one-targeted-retry policy, no further retries were attempted after the 2nd (fresh-container) reproduction. **Recommended next step**: retry later (driver/firmware-level GPU resets can be transient across hours) or on a different card set (e.g. cards 0-3, which also showed a reset event during a separate incident earlier today but were not exercised in this session) or a different node (gnr17408 / gnr17405) to isolate whether this is node-specific.

## Artifacts

- `logs/run.log` -- full container stdout/stderr, including 3 attempts' worth of tracebacks
- `logs/driver.log` -- shell orchestration log (container creation, preflight, sampler start)
- `metrics/metrics.json` -- machine-readable metrics (status=failed, partial data)
- `metrics/xpu_memory.csv` -- raw per-device memory samples (400ms interval) up to the crash
- `config/effective_config.yaml`, `config/deploy_config_used.yaml` -- the exact deploy YAML used
- `scripts/run_test.sh` -- exact reproducible launch command
- `scripts/timed_export_full_tp4_c4567_cfgoff_inductor_warm.py` -- exact driver script
- `environment.txt` -- image/torch/GPU identity
- `checksums.sha256` -- checksums of the fixed input assets

## Interpretation notes

- **proxy=0**: not a supported field on this vLLM-Omni revision's offline DreamZero path; no proxy layer is used for offline inference, so this is a no-op / not-applicable setting here.
- **devices field in deploy YAML**: container-relative indices (`0,1,2,3`), NOT physical card numbers -- the container restricts + renumbers physical cards 4,5,6,7 to xpu:0-3 internally via `ZE_AFFINITY_MASK`.
