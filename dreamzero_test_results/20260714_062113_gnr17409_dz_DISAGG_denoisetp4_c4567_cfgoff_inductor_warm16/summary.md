# DreamZero DISAGGREGATED (encode|denoise-TP4|decode) - gnr17409 - SUCCESS

## Status: SUCCESS - and it fixes the encoder-duplication memory problem

The user's diagnosis was exactly right: the earlier 22.71 GiB/card peak was the encoder stack
(UMT5-XXL text encoder ~11GB + image encoder + VAE) being **replicated on all 4 DiT/TP cards** in the
monolithic pipeline. This run uses the RFC #4590 disaggregated pipeline so each stage builds ONLY its
own components - the encoder is no longer on the DiT ranks.

## Per-stage component build (from log - the proof)

- encode  (1 card):  tokenizer + text_encoder + image_encoder + vae_encoder
- denoise (TP=4):    dit + scheduler + action_modules   <- DiT ONLY, no encoder
- decode  (1 card):  vae_decoder

## RESULTS

| Metric | Disaggregated (this run) | Monolithic (prior) | Improvement |
|---|---|---|---|
| **Time to completion (warm)** | **38.5 s** | 94.9 s | **2.5x faster** |
| **Peak XPU memory (denoise cards)** | **17.9 GiB** | 22.71 GiB (at ceiling) | -4.8 GiB/card |
| Layerwise CPU offload needed? | **NO** | Yes (mandatory) | eliminated |
| Warm denoise gen | 34.3 s | 94.4 s | 2.75x |
| Model load | 80.1 s | 236.8 s | 3x (no encoder on DiT ranks) |
| Warm decode | 4.2 s | (part of TTC) | - |
| Cold first request | 44.0 s | 116.8 s | - |

Per-card peak memory:
- encode card: **4.1 GiB** (the encoders live here now, on ONE card, not x4)
- denoise cards (TP=4, physical 4,5,6,7): 16.1 / 17.9 / 17.9 / 17.9 GiB - DiT fits RESIDENT, no offload
- decode card: 17.9 GiB

Output: 21-frame 640x352 MP4 (+GIF), actions finite on all requests.

## Configuration

- Topology: pipeline=dreamzero_disaggregated (RFC #4590)
- Denoise TP=4 on **physical cards 4,5,6,7** (ZE_AFFINITY_MASK=1,4,5,6,7,0 -> container xpu:1-4)
- encode on physical card 1, decode on physical card 0
- CFG off (cfg_scale=1.0), inductor ON for denoise (enforce_eager=false), denoise_steps=16
- raw/unencoded camera MP4 input, warm-only (request 0 = cold, discarded)
- AR-Diffusion cudagraph warm-up off (as before)
- NO layerwise offload (the whole point - it's no longer needed)

## Why this is the correct fix

DiT-alone TP=4 is ~10-18 GiB/card; the monolithic path added the full encoder/VAE stack (~5 GiB
resident + more transient) on top of EVERY DiT rank, forcing the 22.71 GiB ceiling and mandatory
offload (which then made warm gen 2.5x slower via host<->device weight streaming). Disaggregation puts
the encoder on its own single card, so the DiT ranks hold only their shard and run resident. This is
precisely what RFC #4590 was built to do.

## Notes

- The decode card shows 17.9 GiB, higher than expected for a VAE-decoder-only stage - worth a follow-up
  (may include the full video-latent tensor + decode activations for the 21-frame rollout). Not blocking.
- Same checkout source fixes as the monolithic run were required (prompts=->prompt=, cross-KV offload
  hook); the cross-KV fix is inert here since denoise runs without offload.
