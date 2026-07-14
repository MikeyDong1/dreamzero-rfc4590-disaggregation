# DreamZero disaggregated TP=4 (denoise) test on gnr17409 - SUCCESS

**Status:** SUCCESS (after fixing multiple bugs found along the way)
**Node:** gnr17409 (10.54.109.214), 6 of 8x Intel Arc Pro B60 used (24480 MiB each)
**Model:** GEAR-Dreams/DreamZero-DROID
**Deploy config:** vllm_omni/deploy/dreamzero_disaggregated_tp4denoise.yaml (new, authored for this test)
  - stage 0 (encode): device 0, standard diffusion engine
  - stage 1 (denoise): devices 1-4, TP=4, ARDiffusionEngine (session-scoped paged KV)
  - stage 2 (decode): device 5, standard diffusion engine
**Run script:** examples/offline_inference/dreamzero/timed_export_disagg.py (new; timed_export.py adapted for 3-stage sampling_params_list and out-of-process decode RPC)
**Input:** raw video (three fixed camera MP4s), 2 chunks, denoise_steps=16 (default)

## Question answered: does the disaggregation fix actually cut denoise memory?

**Yes.** Peak per-device memory on the denoise stage (TP=4, devices 0-3 in this run):

| Device | Peak (MiB) | vs. monolithic TP=4 baseline |
|---|---|---|
| 0 | 17369.29 | -7086 MiB |
| 1 | 19001.49 | -5478 MiB |
| 2 | 18979.43 | -5476 MiB |
| 3 | 19077.30 | -5378 MiB |

The monolithic dreamzero_tp4.yaml run (see summary_gnr17409_tp4_raw.md) pinned every device at
~24408-24480 MiB (essentially the B60's 24480 MiB hardware ceiling), causing a reproducible
device-lost OOM crash mid-generation. With the disaggregated denoise stage carrying only
dit+scheduler+action_modules (no UMT5 text encoder, no image encoder, no VAE), peak usage dropped
~5.4-7.1 GB per device, comfortably under the ceiling with real headroom.

Encode (device 4 in the affinity mask / device "0" in-config) and decode (device 5) were not
covered by this run's memory sampler (hardcoded to devices 0-3) - not needed for the comparison,
since encode/decode are lightweight single-device stages by design (0.24-8.2 GiB model weights per
earlier logs), not the OOM-prone stage.

## Timing

- MODEL_LOAD_S: 119.4s (cold container start incl. weight load across 3 stages)
- TIME_TO_FIRST_OUTPUT_S: 77.9s
- GEN_TIMES_S: [77.9, 69.4] (2 chunks)
- DECODE_S: 4.6s
- TIME_TO_OUTPUT_FINISHED_S: 156.2s
- Output: FRAMES_SHAPE=(21, 352, 640, 3) uint8; ACTION shapes (24, 8), finite, sane ranges (-0.12 to 0.56)

Not directly comparable to the monolithic TP4 baseline's timing (different topology, device count,
and this is TP=4-on-denoise-only vs TP=4-monolithic), but included for completeness.

## Bugs found and fixed along the way (all fixed in the local repo, then copied to the node)

Two categories: (A) environment/checkout version-skew (node's checkout predated many unrelated
upstream changes - not this feature's fault), and (B) real bugs in the RFC #4590 disaggregation
implementation itself, found and fixed during this test:

### (B) Real disaggregation-feature bugs fixed

1. **pipeline_dreamzero.py `__init__`**: the AR-Diffusion engine_backend requirement check ran
   unconditionally for every stage role, wrongly rejecting encode/decode workers (which correctly
   own no DiT/KV and should use the standard engine). Fixed: only require ar_diffusion for
   denoise/monolithic roles.
2. **pipeline_dreamzero.py `_kv_reset`**: called `self._ar_diffusion_kv_state.reset(...)`
   unconditionally; encode workers have `_ar_diffusion_kv_state=None` by design (per the method's
   own docstring), so this crashed with AttributeError on every encode-stage reset. Fixed: guard
   with `is not None`.
2. **pipeline_dreamzero.py**: no `dummy_run_num_frames = 0` class attribute, so the engine's
   generic dummy-warmup request (no robot_obs) reached `check_inputs`/`encode_conditions` on every
   role and crashed - the monolithic path had a special-case for this in `forward()` but the
   disaggregated atom methods (`check_inputs`, `encode_conditions`) did not. Fixed: set
   `dummy_run_num_frames = 0` so no stage ever runs the warmup (simpler and more robust than
   special-casing each atom).
3. **diffusion_engine.py `postprocess_output`**: the `output.output is None` fast path (used by
   the intermediate encode/denoise outputs, whose real payload rides in `custom_output`) called
   `format_empty_diffusion_outputs()` without passing `custom_output` through, silently dropping the
   `DiffusionStagePayload` before the stage-transition processor could see it. Fixed: forward
   `custom_output` through `format_empty_diffusion_outputs` -> `OmniRequestOutput.from_diffusion`.
4. **stage_payload.py / stage_input_processors/diffusion.py / diffusion_model_runner.py**: the
   `DiffusionStagePayload` dataclass does not survive the out-of-process (multiproc) stage
   client's msgpack transport - it arrives on the far side as a plain `dict`, not the dataclass, so
   `.validate()` (called by both the transition processor and the receiving worker) crashed with
   `AttributeError`/"payload is dict, expected DiffusionStagePayload". The payload-transport docstring
   incorrectly claimed this "just worked" via the existing msgpack contract. Fixed: added
   `DiffusionStagePayload.from_dict()` and call it at both unwrap points (processor and worker).
5. **diffusion_worker.py running on the node was pre-disaggregation** (see below) and had no
   engine_backend-based model-runner selection at all, so the denoise stage's worker always
   constructed a plain `DiffusionModelRunner` instead of `ARDiffusionModelRunner`, silently
   skipping all AR-Diffusion KV attachment. This turned out to be a stale-checkout issue (fixed
   file already existed correctly in the source repo), not a new bug, but is listed here because it
   was the most confusing symptom to debug (looked like a config-threading bug for a long time).
6. **experimental/ar_diffusion/runner.py `_preallocate_kv_cache`**: hardcoded
   `torch.cuda.mem_get_info(self.device)`, which raises `ValueError: Expected a cuda device` on
   XPU. Fixed: use the existing platform-agnostic `current_omni_platform.get_free_memory(device)`.

### (A) Environment/checkout version-skew (node predated this feature or unrelated upstream work)

The node's /data/vllm-omni checkout was an independent, older "Download ZIP" import that predated
essentially this entire feature plus several unrelated upstream refactors. Files copied from this
repo (already fixed / already current) to the node to make the test possible at all:
config/pipeline_registry.py (trimmed to DreamZero-only to avoid an unrelated ming_flash_omni
import break), config/stage_config.py, config/__init__.py, config/config_factory.py,
config/omni_config.py, config/server_settings.py, diffusion/data.py, diffusion/registry.py,
diffusion/request.py, diffusion/output_formatter.py, diffusion/diffusion_engine.py,
diffusion/stage_payload.py, diffusion/stage_roles.py, diffusion/worker/diffusion_model_runner.py,
diffusion/worker/diffusion_worker.py, diffusion/worker/utils.py, diffusion/worker/request_batch.py,
diffusion/sched/* (whole dir), diffusion/cache/stepcache/* (whole dir),
diffusion/distributed/autoencoders/distributed_vae_executor.py,
diffusion/model_loader/diffusers_loader.py + gguf_adapters/, diffusion/models/dreamzero/
pipeline_dreamzero.py + causal_wan_model.py, diffusion/models/interface.py, engine/stage_init_utils.py,
engine/stage_engine_core_proc_manager.py, entrypoints/utils.py, experimental/ar_diffusion/* (whole
new module - didn't exist on the node at all), model_executor/models/dreamzero/pipeline.py,
model_executor/models/aura_omni/* + gr00t/* + indextts2/* (whole dirs, needed transitively by the
full pipeline_registry.py before it was trimmed), model_executor/stage_input_processors/diffusion.py,
outputs.py, platforms/__init__.py, platforms/interface.py, quantization/gguf_config.py,
distributed/omni_connectors/kv_transfer_manager.py.

## Artifacts (on gnr17409, ~/dreamzero-vllm-omni-runs/20260713_204649_gnr17409_dreamzero_tp4_raw/)

- logs/run_disagg21.log - the final successful run
- logs/run_disagg1.log through run_disagg20.log - the debugging iterations (see bugs above)
- metrics/xpu_memory_disagg21.csv - per-device memory samples for the successful run (devices 0-3
  only; sampler script did not cover devices 4-5)
- config/deploy_config_used_disagg.yaml - the new TP=4-denoise disaggregated deploy config
- scripts/timed_export_disagg.py - the new disaggregation-aware offline test script
- scripts/run_test_disagg.sh - exact env vars used to launch the container exec

## Output video/action files

Left on the node at
/data/vllm-omni/outputs/dreamzero/generated_predictions_disagg_tp4denoise/timed_prediction_disagg.mp4
(316090 bytes) and .gif (3056710 bytes) - not copied to local bundle by default; ask if you want them
pulled down.
