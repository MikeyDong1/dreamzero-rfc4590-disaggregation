# DreamZero disaggregated TP=4 (denoise), CFG OFF, WARM measurement - SUCCESS

**Status:** SUCCESS
**Node:** gnr17409 (10.54.109.214), 6 of 8x Intel Arc Pro B60 used (24480 MiB each)
**Model:** GEAR-Dreams/DreamZero-DROID
**Deploy config:** vllm_omni/deploy/dreamzero_disaggregated_tp4denoise_cfgoff.yaml (cfg_scale=1.0,
  num_inference_steps=16, otherwise identical topology to dreamzero_disaggregated_tp4denoise.yaml:
  stage 0 encode/device 0, stage 1 denoise/devices 1-4 TP=4, stage 2 decode/device 5)
**Run script:** examples/offline_inference/dreamzero/timed_export_disagg_cfgoff_warm.py (2 chunks;
  chunk 0 = cold, chunk 1 = warm measurement)
**Input:** raw video (three fixed camera MP4s), denoise_steps=16, CFG off (single branch per step)

## Confirmed CFG is actually off

Log shows only "AR-Diffusion CROSS POPULATE [pos]" lines during the denoise forward, no "[neg]"
lines at all -- the negative/unconditional branch never ran. `do_true_cfg = cfg_scale > 1.0 and
negative_prompt_embeds is not None` in pipeline_dreamzero.py evaluates False with cfg_scale=1.0.

## Time to completion (warm, chunk 1 only -- excludes model load and the cold first chunk)

| Phase | Time |
|---|---|
| Model load (cold container start, once) | 105.078 s |
| Chunk 0 (cold) generation | 45.870 s |
| **Chunk 1 (warm) generation** | **35.842 s** |
| Decode (VAE, both chunks' latents) | 4.197 s |
| **WARM time-to-completion (warm gen + decode)** | **40.039 s** |

Cold-vs-warm speedup: 1.28x (45.87s -> 35.84s), consistent with expected first-call overhead
(CCL topology negotiation, lazy buffer allocation) amortizing away by the second chunk.

Compare to the CFG-ON run (same topology, same denoise_steps=16): gen times were [77.9s, 69.4s]
per chunk -- CFG-off's warm chunk (35.8s) is roughly half of CFG-on's warm chunk (69.4s), matching
the expectation that turning off the negative branch roughly halves per-chunk denoise compute
(one branch's forward passes instead of two).

## Memory usage (warm window only, via sudo xpu-smi dump -m 18, 0.5s samples,
## restricted to the exact [WARM_WINDOW_START_UNIX, DECODE_WINDOW_END_UNIX] interval)

| Device (denoise, TP=4) | Peak (MiB) |
|---|---|
| 0 | 17254.12 |
| 1 | 18985.72 |
| 2 | 18983.94 |
| 3 | 18978.92 |

**This is essentially identical to the CFG-ON run's peak (17369/19001/18979/19077 MiB).** Root
cause: the AR-Diffusion KV pool is sized at model-load time from `local_branches` (1 if
cfg_parallel_size>=2, else 2) -- a static topology decision independent of whether `do_true_cfg`
skips the negative branch at runtime. Setting cfg_scale=1.0 stops the negative branch's *forward
pass* but does not shrink the *KV pool allocation*, since the pool is provisioned for capacity
(2 branches) up front regardless of how many actually run. Turning CFG off saves ~45% of compute
time (per the timing above) but does NOT reduce peak per-device memory in the current
implementation.

To actually reduce memory here, either (a) size the KV pool from `do_true_cfg`/a
runtime-CFG-aware knob at construction time instead of the topology-derived `local_branches`, or
(b) use CFG-parallel (cfg_parallel_size=2) so each rank's pool is sized for exactly the one branch
it will ever run -- as already noted in the prior test's follow-up recommendation.

## Output sanity check

- FRAMES_SHAPE=(21, 352, 640, 3), dtype=uint8, min=0, max=255 -- valid RGB video, same shape as
  the CFG-on run.
- MP4: 312538 bytes (vs 316090 bytes CFG-on) -- comparable size, valid file.
- GIF: 3061751 bytes (vs 3056710 bytes CFG-on) -- comparable.
- ACTION[0]: shape=(24,8), min=-0.1119, max=0.4672, mean=0.1015, std=0.1473, all finite.
- ACTION[1]: shape=(24,8), min=-0.1757, max=0.5593, mean=0.1139, std=0.1972, all finite.

Action ranges/means/stds are close to the CFG-on run's action outputs (CFG-on ACTION[0] range was
-0.0996 to 0.5634; CFG-on ACTION[1] was -0.1175 to 0.4647) -- same order of magnitude, same sign
pattern, no NaN/Inf, no degenerate all-zero or saturated output. **The output looks sane** for a
CFG-off run: without classifier-free guidance the predictions are expected to differ somewhat from
the CFG=5.0 run (less sharply steered toward the prompt), but not to collapse or diverge, which
matches what was observed.

## Artifacts (on gnr17409, ~/dreamzero-vllm-omni-runs/20260713_204649_gnr17409_dreamzero_tp4_raw/)

- logs/run_cfgoff.log -- full run log
- metrics/xpu_memory_cfgoff.csv -- per-device memory samples (devices 0-3), 0.5s interval
- config/deploy_config_used_cfgoff.yaml
- scripts/timed_export_disagg_cfgoff_warm.py

Output video/gif left on node at
/data/vllm-omni/outputs/dreamzero/generated_predictions_disagg_tp4denoise_cfgoff/
