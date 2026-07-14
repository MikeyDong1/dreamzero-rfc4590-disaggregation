# DreamZero TP=4 Disaggregation: Changes and Per-Card Results

Node: gnr17409 (8x Intel Arc Pro B60, 22.71 GiB usable per card). Model: GEAR-Dreams/DreamZero-DROID.
All runs: CFG off (`cfg_scale=1.0`), `torch.compile`/inductor on for the denoise/DiT stage, 16 denoise
steps, raw/unencoded camera MP4 input, warm-only timing (first request is cold — session init + first-call
compile — and is excluded from the reported "time to completion").

## The problem

A monolithic DreamZero run at `tensor_parallel_size=4` builds the **entire pipeline** — tokenizer, UMT5-XXL
text encoder (~11 GB), image encoder, VAE encoder, DiT, scheduler, VAE decoder — on **every** TP rank. The
encoder/VAE stack is small compared to the model as a whole, but it is *replicated 4 times*, once per card,
on top of each card's DiT shard. This pins every card near its 22.71 GiB ceiling and forces layerwise CPU
weight offload just to fit, which then makes each denoise step slow (weights stream from host RAM every
forward instead of staying resident on-device).

## The fix

Run the existing RFC #4590 disaggregated pipeline (`pipeline: dreamzero_disaggregated`) instead of the
monolithic one (`pipeline: dreamzero`). Each stage calls
`DreamZeroPipeline.required_components_for_stage(model_stage)` and builds **only** the components that role
needs:

| Stage | model_stage | Components built | Devices |
|---|---|---|---|
| 0 | `encode` | tokenizer + text_encoder + image_encoder + vae_encoder | 1 card |
| 1 | `denoise` | dit + scheduler + action_modules | TP=4 (4 cards) |
| 2 | `decode` | vae_decoder | 1 card |

The text/image encoders and the VAE never touch the DiT's TP ranks. Denoise ranks hold only their DiT
shard, which fits **resident, with no offload**.

### Code changes required to get a full run working on this checkout revision

1. `vllm_omni/experimental/ar_diffusion/runner.py` — the cudagraph warm-up constructed
   `OmniDiffusionRequest(prompts=["warmup"], ...)`, but the dataclass field is the singular `prompt`. Fixed
   to `OmniDiffusionRequest(prompt="warmup", ...)`.
2. `vllm_omni/diffusion/models/dreamzero/pipeline_dreamzero.py::_kv_populate_cross` — this eager cross-attention
   KV precompute reads `block.cross_attn.{k,v,norm_k,k_img,...}` directly, without calling `block.forward()`.
   When layerwise CPU offload is enabled, a block's real weights only get materialized on-device inside the
   offload hook's `pre_forward`/`post_forward` — which this code path bypasses, so the weights were still
   0-element CPU placeholders (`RuntimeError: mat1 is on xpu:0, different from other tensors on cpu`). Added
   a `_layerwise_offload_hook()` helper and wrapped the per-block loop to onload each block before use and
   offload it after, mirroring the hook's own self-heal path. No-op when offload is disabled.
3. New deploy config `vllm_omni/deploy/dreamzero_disaggregated_tp4denoise_cfgoff_inductor.yaml` — the
   existing `_cfgoff` variant with `enforce_eager: false` on the denoise stage (inductor on) and
   `ar_diffusion_kv_config.warmup_cudagraph: false` (the synthetic warm-up rollout spiked memory past the
   ceiling and was already falling back to lazy capture on failure anyway).

Neither fix is disaggregation-specific in cause — #1 is a stale call site, #2 only matters when combining
disaggregation *or* the monolithic path with layerwise offload — but both were required to get *any* full
end-to-end DreamZero run working on this revision, disaggregated or not.

## Results: monolithic vs disaggregated (same node, same day)

| Metric | Monolithic TP=4 (offload required) | Disaggregated TP=4 (no offload) | Change |
|---|---|---|---|
| Model load | 236.8 s | 80.1 s | 3.0x faster |
| Cold first request | 116.8 s | 44.0 s | 2.7x faster |
| **Warm generation** | 94.4–101.6 s (mean 98.0 s) | 34.3 s | **2.9x faster** |
| Warm decode | 3.3 s (batched) | 4.2 s (single) | ~same |
| **Warm time to completion** | **94.9 s** | **38.5 s** | **2.5x faster** |
| **Peak memory, denoise/DiT card** | **23,255 MiB (22.71 GiB — at ceiling)** | **17,942 MiB (17.53 GiB)** | **−5.3 GiB/card** |
| Layerwise CPU offload | Required (won't fit without it) | Not used | eliminated |

## Per-card memory, by physical card number

### Monolithic (`pipeline: dreamzero`, all 4 cards run the full stack)

| Physical card | Role | Peak memory |
|---|---|---|
| 4 | full pipeline (encode+denoise+decode), TP rank 0 | 23,255 MiB |
| 5 | full pipeline, TP rank 1 | 23,255 MiB |
| 6 | full pipeline, TP rank 2 | 23,255 MiB |
| 7 | full pipeline, TP rank 3 | 23,255 MiB |

Every card is identical because every card builds every component. This is the signature to watch for:
when TP-N memory doesn't drop card-to-card and sits near the device ceiling, the encoder almost certainly
isn't separated from the DiT ranks.

### Disaggregated (`pipeline: dreamzero_disaggregated`, encode/denoise/decode on separate cards)

Container devices were mapped to physical cards via `ZE_AFFINITY_MASK=1,4,5,6,7,0` so the denoise (DiT) TP
group landed on the requested physical cards 4, 5, 6, 7:

| Physical card | Role | Peak memory |
|---|---|---|
| 1 | encode (tokenizer + text/image encoders + VAE encoder) | 4,127 MiB |
| 4 | denoise, TP rank 0 (DiT shard only) | 16,112 MiB |
| 5 | denoise, TP rank 1 (DiT shard only) | 17,942 MiB |
| 6 | denoise, TP rank 2 (DiT shard only) | 17,924 MiB |
| 7 | denoise, TP rank 3 (DiT shard only) | 17,924 MiB |
| 0 | decode (VAE decoder) | 17,923 MiB |

The encode card (physical 1) needs only 4.1 GiB — the entire text/image/VAE-encoder stack that used to be
duplicated on cards 4–7. The denoise cards (physical 4–7) drop by ~5.3 GiB each versus monolithic, enough
to fit resident without offload.

**Open item, not yet root-caused:** the decode card (physical 0) peaks at 17.9 GiB, higher than expected
for a VAE-decoder-only stage. Likely candidate is the full accumulated video-latent tensor plus decode
activations for the whole rollout being held at once; worth profiling separately if further memory headroom
is needed.

## How to reproduce

```bash
# On the node, inside a container with ZE_AFFINITY_MASK=1,4,5,6,7,0 (encode=phys1, denoise=phys4-7, decode=phys0)
# and PYTHONPATH pointing at this checkout:
python -c "
from vllm_omni import Omni
omni = Omni(
    model='GEAR-Dreams/DreamZero-DROID',
    deploy_config='vllm_omni/deploy/dreamzero_disaggregated_tp4denoise_cfgoff_inductor.yaml',
    enforce_eager=False,
)
"
```

See `examples/offline_inference/dreamzero/export_prediction_video.py` for a full working driver
(`_build_observations`, `_extract_latents`, `_decode_with_worker_disagg`).

## Related

- [disaggregation-instructions.md](disaggregation-instructions.md) — how to make a new model support this
  same encode/denoise/decode split.
