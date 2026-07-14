# How to enable a new diffusion model for RFC #4590 disaggregation

This describes how to make a new diffusion pipeline support the encode → denoise → decode disaggregated
execution mode, using the DreamZero implementation as the worked example. See
[dreamzero-tp4-disaggregation-results.md](dreamzero-tp4-disaggregation-results.md) for why this is worth
doing (encoder duplication across TP ranks; ~2.5x speedup, ~5 GiB/card memory saved for DreamZero).

## Why disaggregate at all

A monolithic pipeline builds every component (tokenizer, text/image encoders, VAE, DiT, scheduler) on
every worker, including every tensor-parallel rank of the DiT. If the encoder/VAE stack is non-trivial in
size, it gets replicated N times for an N-way TP DiT, wasting memory and often forcing CPU weight offload
that wouldn't otherwise be needed. Disaggregation splits the pipeline into three independently-scheduled
stages so encode and decode run on their own (non-TP) workers and the DiT's TP ranks hold only the DiT.

## The three roles

| Role | `model_stage` | Typically owns | Typically runs on |
|---|---|---|---|
| Encode | `"encode"` | tokenizer, text encoder, image/observation encoder, VAE encoder, initial-latent/timestep setup | 1 device, standard diffusion engine |
| Denoise | `"denoise"` | the transformer/DiT, scheduler, any KV/session state | N devices (tensor/sequence parallel), possibly a specialized engine |
| Decode | `"decode"` | VAE decoder, output postprocessing | 1 device, standard diffusion engine |

A model that doesn't disaggregate still works unchanged: `model_stage="diffusion"` (or unset) is the
monolithic fallback and takes the original code path. Disaggregation is purely additive.

## Step 1 — Implement the `SupportsDisaggregatedDiffusionExecution` protocol

File: `vllm_omni/diffusion/models/interface.py`. Your pipeline class needs:

```python
supports_disaggregated_execution: ClassVar[bool] = True

@classmethod
def required_components_for_stage(cls, model_stage: str) -> StageComponentSpec: ...

def export_stage_payload(self, state: DiffusionRequestState, *, source_stage: str, target_stage: str) -> DiffusionStagePayload: ...

def import_stage_payload(self, payload: DiffusionStagePayload, *, target_stage: str, request: OmniDiffusionRequest | None = None) -> DiffusionRequestState: ...
```

The runner probes for this with `isinstance(pipeline, SupportsDisaggregatedDiffusionExecution)` plus the
explicit `supports_disaggregated_execution` flag (`vllm_omni/diffusion/models/interface.py:supports_disaggregated_execution`).
Both must be true — the flag lets a model opt out even if it happens to define same-named methods.

### 1a. `required_components_for_stage` — declare what each role builds

`StageComponentSpec` (`vllm_omni/diffusion/stage_roles.py`) is a set of booleans:
`tokenizer`, `text_encoder`, `image_encoder`, `vae_encoder`, `dit`, `scheduler`, `vae_decoder`,
`action_modules`. This is queried **before** module construction, so returning `False` for a component
means it is never built or weight-loaded for that stage — not just unused.

DreamZero's implementation (`vllm_omni/diffusion/models/dreamzero/pipeline_dreamzero.py::required_components_for_stage`):

```python
@classmethod
def required_components_for_stage(cls, model_stage: str) -> StageComponentSpec:
    role = normalize_stage_role(model_stage)
    if role == ENCODE:
        return StageComponentSpec(tokenizer=True, text_encoder=True, image_encoder=True, vae_encoder=True)
    if role == DENOISE:
        return StageComponentSpec(dit=True, scheduler=True, action_modules=True)
    if role == DECODE:
        return StageComponentSpec(vae_decoder=True)
    # Monolithic / unknown: everything.
    return ALL_COMPONENTS
```

If your model has a component this vocabulary doesn't name, don't stretch an existing field to mean
something else — the fields are intentionally generic. Leave it unmodeled (built by every stage that needs
it) unless it's large enough to be worth a follow-up field.

### 1b. `export_stage_payload` — pack what the next stage needs

Called at the end of a stage's work. Convert your pipeline's runner-local state into a
`DiffusionStagePayload` (`vllm_omni/diffusion/stage_payload.py`):

```python
DiffusionStagePayload.create(
    request_id=state.request_id,
    source_stage=source_stage,          # "encode" or "denoise"
    target_stage=target_stage,          # "denoise" or "decode"
    payload_type=f"{source_stage}_to_{target_stage}",
    tensors={...},    # torch.Tensor values only
    metadata={...},   # str/bool/number/None, nested list/dict/tuple of those, or numpy
)
```

Rules the payload enforces (see `DiffusionStagePayload.validate` / `NonTransportableValueError`):

- **Tensors go in `tensors`, everything else transportable goes in `metadata`.** A tensor placed directly
  in `metadata` is rejected.
- **No live modules, generators, CUDA/XPU streams, schedulers, or session/KV state objects** — those are
  process-local and must never cross a stage boundary. If your model has session state analogous to
  DreamZero's AR-Diffusion KV, do NOT put it in the payload; instead carry a `session_id` in `metadata` and
  let the target stage re-acquire its own live state by that id (see 1c).
- `create(..., sanitize=True)` (the default) automatically calls `sanitize_transport_tensor` on every tensor
  — detach, move to host, make contiguous. You don't need to do this yourself.
- Only pack what the *next* stage actually reads. The runner and the generic transition processor never
  interpret payload contents — they're opaque to everyone except your own `export_stage_payload` /
  `import_stage_payload`.

DreamZero packs a private `DreamZeroStageCarrier` dataclass (session id, geometry, encoded conditions,
initial noise, timestep schedule params) into the payload's `tensors`/`metadata`, keyed by which transition
is being packed (`encode_to_denoise` carries encoded conditions + initial noise; `denoise_to_decode` carries
`video_out`/`action_out`). Different transitions can carry different fields — the payload's own
`source_stage`/`target_stage` disambiguate.

### 1c. `import_stage_payload` — rebuild runner-local state on the next stage

Called at the start of a stage's work, with the payload the previous stage exported:

```python
def import_stage_payload(self, payload, *, target_stage, request=None):
    payload.validate()
    # move payload.tensors back to this stage's device, reconstruct whatever
    # private carrier/state your export_stage_payload packed
    ...
    state = DiffusionRequestState(request_id=payload.request_id, sampling=...)
    state.extra["your_carrier_key"] = carrier
    return state
```

**Do not reconstruct live session/KV state from the payload.** If the target stage owns session state (like
DreamZero's denoise stage owns the AR-Diffusion KV pool), acquire it the same way the monolithic path
always did — by session id, through the engine's normal session map — and only use
`import_stage_payload` to restore the model-private carrier fields (geometry, encoded conditions, etc.).

## Step 2 — Register the disaggregated topology

File: your model's `pipeline.py` (e.g. `vllm_omni/model_executor/models/dreamzero/pipeline.py`). Declare a
second `PipelineConfig` alongside the existing monolithic one, with three `StagePipelineConfig` entries:

```python
GENERIC_DIFFUSION_PROCESSOR = (
    "vllm_omni.model_executor.stage_input_processors.diffusion.diffusion_stage_transition"
)

YOUR_MODEL_DISAGGREGATED_PIPELINE = PipelineConfig(
    model_type="your_model_disaggregated",
    model_arch="YourModelPipeline",
    stages=(
        StagePipelineConfig(
            stage_id=0, model_stage="encode", execution_type=StageExecutionType.DIFFUSION,
            input_sources=(), final_output=False, model_arch="YourModelPipeline",
            custom_process_next_stage_input_func=GENERIC_DIFFUSION_PROCESSOR,
        ),
        StagePipelineConfig(
            stage_id=1, model_stage="denoise", execution_type=StageExecutionType.DIFFUSION,
            input_sources=(0,), final_output=False, model_arch="YourModelPipeline",
            custom_process_input_func=GENERIC_DIFFUSION_PROCESSOR,
            custom_process_next_stage_input_func=GENERIC_DIFFUSION_PROCESSOR,
        ),
        StagePipelineConfig(
            stage_id=2, model_stage="decode", execution_type=StageExecutionType.DIFFUSION,
            input_sources=(1,), final_output=True, final_output_type="image",  # or your output type
            model_arch="YourModelPipeline",
            custom_process_input_func=GENERIC_DIFFUSION_PROCESSOR,
        ),
    ),
)
```

`GENERIC_DIFFUSION_PROCESSOR` is model-agnostic — it just moves the `DiffusionStagePayload` from one
stage's output into the next stage's request prompt. You wire it in, you never implement it.

Register the new `model_type` in `vllm_omni/config/pipeline_registry.py`'s `OMNI_PIPELINES` dict alongside
your existing monolithic entry.

Topology rules that `PipelineConfig.validate()` / `validate_linear_diffusion_topology`
(`vllm_omni/diffusion/stage_roles.py`) will enforce automatically:

- exactly one `encode` stage, entry point (`input_sources=()`);
- `denoise` must source from an `encode` (or upstream `denoise`) stage, exactly one source;
- `decode` must source from a `denoise` stage, exactly one source;
- `final_output=True` must be on the stage that owns the user-visible result (normally `decode`).

## Step 3 — If your denoise stage needs session/KV state, wire the right engine

If your model's denoise loop needs persistent session state across forwards (autoregressive KV, a rolling
window, etc.), that lives on the denoise worker only — never in `DiffusionRequestState`, never in the
payload. Look at how DreamZero's monolithic-vs-denoise role gate works in `__init__`
(`pipeline_dreamzero.py`):

```python
model_stage_role = normalize_stage_role(getattr(od_config, "model_stage", None))
if model_stage_role in (DENOISE, MONOLITHIC):
    engine_backend = str(getattr(od_config, "engine_backend", "") or "")
    if "ar_diffusion" not in engine_backend.lower().replace("-", "_"):
        raise ValueError("... requires the AR-Diffusion engine for the denoise role ...")
```

Encode/decode are exempt because they own no KV/session state and run on the standard diffusion engine.
This kind of fail-fast at `__init__` time (rather than crashing mid-forward on first KV access) is worth
copying if your model has an analogous "this role needs a special engine" requirement.

## Step 4 — Author the deploy YAML

Three `stages:` entries, one per role, each with its own `devices:` and (for denoise) `parallel_config`:

```yaml
pipeline: your_model_disaggregated
async_chunk: false
distributed_executor_backend: mp
dtype: bfloat16

stages:
  - stage_id: 0                    # encode
    devices: "0"
    enforce_eager: true
    model_class_name: YourModelPipeline
    model_config: {...}

  - stage_id: 1                    # denoise
    devices: "1,2,3,4"
    parallel_config:
      tensor_parallel_size: 4
    engine_backend: <your special engine, if any>
    model_config: {...}

  - stage_id: 2                    # decode
    devices: "5"
    enforce_eager: true
    model_class_name: YourModelPipeline
    model_config: {...}
```

`devices:` values are container-relative indices, not physical card numbers — if you restrict a container
to specific physical GPUs (e.g. via `ZE_AFFINITY_MASK=1,4,5,6,7,0` on Intel XPU), the container renumbers
them starting at 0 in mask order. Map deliberately if you need a specific role on specific physical
hardware; see the worked example in `vllm_omni/deploy/dreamzero_disaggregated_tp4denoise_cfgoff_inductor.yaml`.

## Step 5 — Verify

1. **Component isolation.** Start each stage and check its startup log line
   (`Diffusion stage startup: ... components=...` from `diffusion_model_runner.py::_log_stage_startup`).
   Confirm encode does NOT log `dit`, denoise does NOT log `text_encoder`/`vae_encoder`, decode logs only
   `vae_decoder`.
2. **Memory.** Sample per-card memory through a full request. Denoise cards should be close to what a
   DiT-only (no-encoder) monolithic-minus-encoder baseline would use, not the full monolithic figure.
   Encode/decode cards should be small.
3. **Correctness.** Compare output (video/image/actions) between the monolithic and disaggregated runs on
   identical input — they should match, since disaggregation only changes *where* computation happens, not
   the computation itself. DreamZero's phase methods (`_run_encode_phase` / `_run_denoise_phase` /
   `_run_decode_phase`) are shared between the monolithic `forward()` and the disaggregated atoms
   specifically so this holds by construction — don't fork the math when disaggregating; factor the
   monolithic path into phases and call the same phases from both.
4. **Foundation tests.** `tests/diffusion/disaggregated/` has torch-free tests for the generic machinery
   (`stage_roles`, `stage_payload`, the transition processor, topology validation) that don't need your
   model at all — run them first (`pytest tests/diffusion/disaggregated/ -m "not needs_runtime"`) to catch
   wiring mistakes before touching hardware.

## Common pitfalls

- **Forgetting a component leaks onto the wrong stage.** If `required_components_for_stage` for `denoise`
  accidentally returns `text_encoder=True`, the encoder gets built (and its weights loaded) on every TP
  rank again — silently reintroducing the exact problem disaggregation solves. Log the component list at
  stage startup and check it.
- **Putting live/session state in the payload.** It won't serialize (the payload validator rejects modules,
  generators, streams, scheduler objects) or, if you bypass validation, it will silently break across a
  process boundary. Session state is re-acquired by id, not transported.
- **Code paths that touch weights outside the normal forward.** If your denoise stage does anything eager
  outside the block's own `forward()` (precomputing a cache, warming something up) and you later want to
  combine disaggregation with CPU weight offloading, that code must onload/offload weights itself — offload
  hooks only fire around `forward()`. See the `_kv_populate_cross` fix in
  [dreamzero-tp4-disaggregation-results.md](dreamzero-tp4-disaggregation-results.md) for a concrete example
  of this failure mode and its fix.
