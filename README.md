[README.md](https://github.com/user-attachments/files/30066007/README.md)
# DreamZero RFC 4590 Disaggregated Diffusion

This repository contains an initial implementation of
[vLLM-Omni RFC 4590](https://github.com/vllm-project/vllm-omni/issues/4590)
for DreamZero.

The current implementation separates the DreamZero diffusion pipeline into
independently deployed Encode, Denoise, and Decode stages while preserving the
existing monolithic execution path.

```text
Encode  →  Denoise / DiT  →  Decode
```

## Current scope

The implementation currently provides:

- independent Encode, Denoise, and Decode diffusion stages;
- stage-specific component construction and weight loading;
- removal of text/image encoder replication from DiT tensor-parallel ranks;
- generic stage dispatch in `DiffusionModelRunner`;
- a typed and versioned `DiffusionStagePayload`;
- a model-agnostic diffusion stage transition processor;
- DreamZero-specific payload packing and restoration;
- denoise-stage ownership of AR-Diffusion KV and session state;
- compatibility with the original monolithic DreamZero topology;
- TP=4 XPU end-to-end validation.

This is a working first-stage implementation of RFC 4590. It is not yet the
complete production runtime described by the RFC.

## Architecture

### Monolithic execution

```text
DreamZero worker
├── Tokenizer
├── UMT5 text encoder
├── Image encoder
├── VAE
├── DiT
├── Scheduler
└── Action modules
```

Every DiT tensor-parallel rank constructs the full pipeline.

### Disaggregated execution

```text
Encode stage
├── Tokenizer
├── UMT5 text encoder
├── Image encoder
└── VAE encoder
        │
        ▼
Denoise stage
├── CausalWan DiT
├── Scheduler
├── Action modules
└── AR-Diffusion session KV
        │
        ▼
Decode stage
├── Action/output postprocessing
└── Video decode/output handling
```

The Denoise tensor-parallel ranks no longer construct or load the encoder
stack.

## Main implementation changes

### Pipeline decomposition

DreamZero inference is decomposed into shared mathematical phases:

```text
_run_encode_phase()
_run_denoise_phase()
_run_decode_phase()
```

Both monolithic and disaggregated execution use these same phase
implementations. Disaggregation changes where the computation runs, not the
model mathematics.

### Stage-specific component ownership

The DreamZero pipeline declares the components required by each stage:

```text
encode
→ tokenizer + text encoder + image encoder + VAE encoder

denoise
→ DiT + scheduler + action modules

decode
→ decode/output postprocessing components

diffusion
→ complete monolithic pipeline
```

Unused components are not constructed or weight-loaded on that stage.

### Generic runner support

`DiffusionModelRunner` dispatches execution according to `model_stage`:

```text
diffusion → monolithic execution
encode    → encode-stage execution
denoise   → denoise-stage execution
decode    → decode-stage execution
```

The runner manages generic request and stage lifecycle behavior. DreamZero
tensor layouts and model-private fields remain inside the DreamZero pipeline.

### Stage payload contract

Stage boundaries use a typed payload:

```text
DiffusionStagePayload
├── schema_version
├── request_id
├── source_stage
├── target_stage
├── payload_type
├── tensors
└── metadata
```

The generic runner and transition processor treat payload contents as opaque.
Only the model pipeline interprets model-specific tensor and metadata fields.

Live process-local objects are not transferred, including:

- model modules;
- schedulers;
- device streams;
- random generators;
- AR-Diffusion KV objects.

### Session and KV ownership

AR-Diffusion KV state is owned only by the Denoise stage.

```text
Encode stage  → no AR-Diffusion KV
Denoise stage → owns session-scoped KV
Decode stage  → no AR-Diffusion KV
```

Stage payloads carry stable data and `session_id`; the Denoise runner uses the
session identity to reacquire its local runtime state.

## Topologies

The repository preserves the original monolithic topology:

```text
pipeline: dreamzero
```

and adds the disaggregated topology:

```text
pipeline: dreamzero_disaggregated
```

The disaggregated topology is linear:

```text
Stage 0: Encode
Stage 1: Denoise
Stage 2: Decode
```

DreamZero's text encoder, image encoder, and VAE encoder currently run inside
one combined Encode stage. They are not yet represented as separate DAG nodes.

## Validated configuration

A validated deployment uses:

```text
Encode:  1 XPU
Denoise: TP=4 on 4 XPUs
Decode:  1 XPU
CFG:     disabled
Steps:   16
```

Example deployment configuration:

```text
vllm_omni/deploy/dreamzero_disaggregated_tp4denoise_cfgoff_inductor.yaml
```

Observed comparison on Intel Arc Pro B60 hardware:

| Metric | Monolithic TP=4 | Disaggregated TP=4 |
|---|---:|---:|
| Model load | 236.8 s | 80.1 s |
| Warm generation | approximately 98.0 s | 34.3 s |
| Warm completion | 94.9 s | 38.5 s |
| Peak memory per Denoise card | 22.71 GiB | 17.53 GiB |
| Layerwise CPU offload | Required | Removed |

The primary gain comes from removing the encoder and VAE stack from the DiT
tensor-parallel ranks, allowing the DiT shards to remain resident.

## RFC 4590 alignment

| RFC direction | Status |
|---|---|
| Encode / DiT / Decode separation | Implemented |
| Independent device placement | Implemented |
| Stage-specific component loading | Implemented for DreamZero |
| Pipeline owns model mathematics | Implemented |
| Generic runner owns stage dispatch | Implemented |
| Explicit stage payload contract | Implemented |
| Model-private state remains opaque | Implemented |
| Denoise-stage KV ownership | Implemented |
| Monolithic compatibility path | Implemented |
| Reuse by a conventional diffusion model | Not yet demonstrated |
| Runner-owned generic denoise step loop | Not yet complete |
| Stage-aware continuous batching | Not yet implemented |
| Connector-backed tensor transfer | Not yet implemented |
| Stage readiness and asynchronous prefetch | Not yet implemented |
| General DAG join and fan-out | Not yet implemented |
| Chunk-level streaming overlap | Not yet implemented |

## Current limitations

The current version focuses on correct stage isolation and partial component
loading.

It does not yet provide:

- generic runner-controlled denoise-step execution;
- independent stage batching and backpressure;
- device-to-device connector-backed artifact transfer;
- queue-time artifact prefetch;
- readiness-aware scheduling;
- multiple independent encoder or decoder DAG nodes;
- multi-consumer fan-out;
- chunk-level Decode/KV-update overlap;
- a completed automated monolithic-versus-disaggregated numerical-equivalence
  test.

## Next development phases

### Harden the current framework

- unify the formal pipeline contract with actual runner requirements;
- make the runner compose the existing step-execution protocol;
- make batch execution stage-aware;
- correct DreamZero decode/output component ownership;
- fix authoritative multi-chunk session progress;
- complete numerical-equivalence testing;
- enable one conventional diffusion model as a second reference.

### Add missing RFC runtime capabilities

- connector-backed artifact manifests;
- shared-memory reference transfer;
- stage readiness lifecycle;
- asynchronous prefetch during queue waiting;
- stage-local batching, replicas, and backpressure;
- DAG multi-input join and fan-out;
- streaming Decode/KV-update overlap;
- cross-node NIXL or UCX transport validation.

## Documentation

Detailed current implementation and RFC alignment:

```text
docs/rfc4590-current-implementation.md
```

Recommended agent instructions and implementation roadmap:

```text
RFC4590_AGENT_INSTRUCTION_HARDEN_EXISTING.md
RFC4590_AGENT_INSTRUCTION_IMPLEMENT_MISSING_RUNTIME.md
RFC4590_TODO.md
```

## Project status

The current repository demonstrates that RFC 4590-style DreamZero
disaggregation is operational and materially reduces memory pressure and
latency in the validated TP=4 deployment.

The remaining work is primarily runtime generalization and performance:

```text
generic step execution
→ stage-aware batching
→ connector transfer
→ readiness and prefetch
→ DAG and streaming overlap
```
