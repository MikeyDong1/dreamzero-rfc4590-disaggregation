# DreamZero DiT-only — TP=4 eager vs TP=4 inductor+CFG-off

**Node:** gnr17408, cards 4,5,6,7 (4× Intel Arc Pro B60) • **Image:** `vllm-omni-xpu:mikey-dev7` (torch 2.11+xpu)
**Component under test:** DiT ONLY (`action_head.model.*`, 40-block CausalWanModel). Text encoder (UMT5)
and Wan VAE were NEVER loaded to the XPU. Input: pre-encoded `dit_inputs.pt` (one request, N=16 denoise steps).

Two run bundles (raw artifacts, checksummed):
- [`20260710_141433_gnr17408_dit_TP4_cards4567_N16_dev7/`](20260710_141433_gnr17408_dit_TP4_cards4567_N16_dev7/summary.md) — TP=4, eager, CFG on
- [`20260711_014425_gnr17408_dit_TP4_inductor_cfgoff_N16_rep10_dev7/`](20260711_014425_gnr17408_dit_TP4_inductor_cfgoff_N16_rep10_dev7/summary.md) — TP=4, inductor, CFG off, +10x repeat

## Headline numbers (warm; clean timed run, no profiler)

| Metric | Eager, CFG-on | **Inductor, CFG-off** | Change |
|---|---|---|---|
| complete_output_time | 20.17 s | **11.68 s** | **1.73×** |
| per-step (mean) | 1.199 s | **0.688 s** | 1.74× |
| time_to_first_output | 2.15 s | **1.51 s** | 1.42× |
| peak XPU memory / card (process alloc) | 9.74 GiB | 10.5–10.8 GiB | +~1 GiB |
| model_load_time | 152.7 s | 166.1 s | +13 s (compile wrapping, not tracing) |
| decode_time | n/a (no VAE decode in a DiT-only test) | n/a | — |

One-time inductor first-call compile cost, paid inside the (discarded) warmup: **82.7 s**.

## 10x sequential repeat — completion-time distribution (inductor + CFG-off, same warm pipeline)

| n | mean | std | min | p50 | p90 | max |
|---|---|---|---|---|---|---|
| 10 | **11.764 s** | 0.296 s (2.5%) | 11.342 s | 11.697 s | 12.163 s | 12.213 s |

Individual completion times (s): 12.158, 12.213, 12.007, 12.013, 11.661, 11.453, 11.732, 11.500, 11.562, 11.342
Individual TTFO (s): 1.475, 1.428, 1.496, 1.526, 1.418, 1.416, 1.349, 1.353, 1.348, 1.365

Tight distribution, no warm-up drift or thermal creep across 10 back-to-back requests on the same
resident model.

## Attributing the 1.73× — CFG-off vs inductor (from op call counts, not guesswork)

Op call counts in the inductor+CFG-off trace are **exactly half** of the eager+CFG-on trace:

| Op | Eager+CFG-on calls | Inductor+CFG-off calls | Ratio |
|---|---|---|---|
| `c10d::allreduce_` | 6994 | 3497 | 2.00× |
| `gemm_kernel` | 9040 | 4520 | 2.00× |
| `aten::addmm` | 8752 | 4376 | 2.00× |

That exact 2× is CFG-off's signature: with CFG off, `predict_noise` runs only the positive
branch, so every step does 1 DiT forward instead of 2.

Per-call latency for the same three ops is **unchanged by inductor**:

| Op | Eager+CFG-on per-call | Inductor+CFG-off per-call |
|---|---|---|
| `gemm_kernel` | 476.0 µs | 476.6 µs |
| `aten::addmm` | 489.6 µs | 490.5 µs |
| `c10d::allreduce_` | 872.4 µs | 926.9 µs (noise, not a regression) |

**Conclusion: `torch.compile` does not speed up the matmuls or the collective dispatch** —
both already hit oneDNN/XeTLA gemm kernels and the CCL allreduce beneath `aten`, which Dynamo
does not rewrite. Inductor's real contribution is fusing the ~15 small elementwise/RMSNorm/RoPE
ops per block into fused `triton_poi_fused_*` / `triton_red_fused_*` kernels (absent in the eager
trace, including one that fuses the elementwise math *around* the TP all-reduce, not the
collective itself) — fewer kernel launches on the non-dominant ops, not faster big ops.

**Net: CFG-off is essentially the whole 2× on call count; inductor's slice is the gap between
that 2× and the observed 1.74× per-step** (kernel-launch-overhead reduction on the small ops).
This run mixes both variables (compile × cfg), so the exact CFG-off-alone number is inferred,
not measured — an eager+CFG-off-only run would isolate it precisely if that split matters for
a decision.

## Profiling — where the time goes now (rank0, torch.profiler CPU+XPU, warm N=16)

Total self-XPU time ≈ 14.85 s (down from 34.6 s profiled-with-overhead in the eager+CFG-on run).

| Op | Self-XPU % | Self-XPU | Calls | Note |
|---|---|---|---|---|
| `c10d::allreduce_` | **43.7%** | 3.24 s | 3497 | TP all-reduce — grew as a SHARE (was 35.2%) |
| `gemm_kernel` | 29.0% | 2.15 s | 4520 | matmul compute |
| `aten::addmm` | 28.9% | 2.15 s | 4376 | linear layers |
| `ReduceCopyKernel` (bf16) | 17.8% | 1.32 s | 6120 | CCL ring reduce copy |
| `allreduce_large_su_ring_write` (2 variants) | 11.4%+5.8% | 1.27 s | 6120 | CCL ring kernels |
| `_vllm_fa2_C::varlen_fwd` | 4.9% | 0.36 s | 2040 | FlashAttention-2 (stays eager; no fake kernel for Dynamo) |
| `triton_poi_fused_add_unsqueeze_0` (2 variants) | 4.8%+4.8% | 0.71 s | 1360 | inductor-fused elementwise |
| `triton_red_fused_add_mul_native_layer_norm...` (2 variants) | 4.5%+4.5% | 0.66 s | 2560 | inductor-fused RMSNorm |
| `triton_poi_fused__to_copy_add_all_reduce_div_mul_rsqrt...` | 0.31%+0.28% | 0.09 s | 1280 | inductor fused the pre/post-reduce elementwise around the all-reduce (not the collective) |

**Communication is now a LARGER share (43.7% vs 35.2% before) even though it halved in
absolute terms (6.10 s → 3.24 s)** — because CFG-off + inductor shrank compute and the small
elementwise ops faster than it shrank the collective.

## Recommended next steps, in order of expected payoff

1. **Overlap block N's all-reduce with block N+1's compute.** The trace suggests each block's
   forward currently blocks on its own reduce before the next block starts — pipelining this
   (compute-comm overlap) would hide most of the now-dominant 43.7% communication cost.
2. **Batch multiple blocks' reduces into fewer, larger collectives.** All-reduce here is
   latency-bound, not bandwidth-bound (per-call time didn't drop even though CFG-off cut call
   count in half, not per-call tensor size) — fewer, larger calls amortize the fixed per-call
   overhead.
3. **CFG-parallel instead of CFG-off**, if classifier-free guidance must come back for
   production quality: put the positive/negative branches on separate rank groups instead of
   running them sequentially through the same TP group (2× forwards). This recovers CFG without
   paying its current 2× DiT-forward cost.
4. **Isolate CFG-off from inductor** with a 3rd eager+CFG-off run, if the exact split between
   the two levers matters for a go/no-go decision on adopting `torch.compile`.
