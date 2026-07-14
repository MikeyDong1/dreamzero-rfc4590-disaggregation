# DreamZero DiT-only — TP=4 + inductor + CFG-off (+10x repeat)

**Status:** success (exit 0)  •  **Node:** gnr17408 (cards 4,5,6,7)  •  **Run ID:** 20260711_014425_gnr17408_dit_TP4_inductor_cfgoff_N16_rep10_dev7
**Config:** tensor_parallel_size=4, compile=inductor (all 40 blocks, fullgraph=False), cfg=off,
offload=full_resident, image vllm-omni-xpu:mikey-dev7 (torch 2.11+xpu). Same pre-encoded
dit_inputs.pt as the eager baseline. DiT ONLY: text encoder + VAE never on XPU.

## The 5 required metrics (warm; clean timed run, no profiler)
| Metric | Value |
|---|---|
| model_load_time | **166.1 s** (includes wrapping 40 blocks in torch.compile; +13s vs eager's 152.7s) |
| time_to_first_output | **1.51 s** (prefill 0.67 s + first denoise step) |
| decode_time | **n/a** — VAE decode not run in a DiT-only test |
| complete_output_time | **11.68 s** (prefill 0.67 s + 16-step denoise 11.01 s) |
| peak_xpu_memory | **10.5-10.8 GiB/card** process alloc, **15.9-19.1 GiB/card** whole-device peak |

Per-step denoise: mean **0.688 s** (min 0.626, max 0.838), 16 steps. One-time inductor
first-call compile cost, paid in the (discarded) warmup: **82.7 s**.

## 10x sequential repeat -- completion-time distribution (same warm pipeline, no profiler)
| n | mean | std | min | p50 | p90 | max |
|---|---|---|---|---|---|---|
| 10 | **11.764 s** | 0.296 s | 11.342 s | 11.697 s | 12.163 s | 12.213 s |

Individual values (s): 12.158, 12.213, 12.007, 12.013, 11.661, 11.453, 11.732, 11.500, 11.562, 11.342
TTFO per repeat (s): 1.475, 1.428, 1.496, 1.526, 1.418, 1.416, 1.349, 1.353, 1.348, 1.365
Tight distribution (std/mean = 2.5%), no warm-up drift or thermal creep across 10 back-to-back requests.

## Comparison vs the TP=4 eager+CFG-on baseline (same TP=4, same N=16, same dev7 image, same cards)
| Config | complete (s) | per-step (s) | peak alloc/card |
|---|---|---|---|
| TP=4 eager, cfg=on | 20.17 | 1.199 | 9.74 GiB |
| **TP=4 inductor, cfg=off** | **11.68** | **0.688** | **10.5-10.8 GiB** |
| **Speedup** | **1.73x** | **1.74x** | (+0.8-1.1 GiB; more live activations, no offload change) |

## Attributing the 1.73x -- CFG-off vs inductor (from op call counts, not guesswork)
Op call counts in the new trace are **exactly half** the eager+cfg=on trace:
`c10d::allreduce_` 6994 -> 3497, `gemm_kernel` 9040 -> 4520, `aten::addmm` 8752 -> 4376.
That is CFG-off's signature (1 DiT forward/step instead of 2) -- it is the dominant lever.

Per-call latency for the same three ops is **unchanged by inductor**:
`gemm_kernel` 476.0 -> 476.6 us/call, `aten::addmm` 489.6 -> 490.5 us/call, `c10d::allreduce_`
872 -> 927 us/call (noise, not a regression). **torch.compile does not speed up the matmul or
collective dispatch** -- those already hit oneDNN/XeTLA gemm kernels and the CCL allreduce
beneath aten, which Dynamo does not rewrite. Inductor's real contribution is fusing the
~15 small elementwise/RMSNorm/RoPE ops per block into fused `triton_poi_fused_*` /
`triton_red_fused_*` kernels (absent in the eager trace) -- fewer kernel launches around the
big ops, not faster big ops. Net: **CFG-off is approximately the whole 2x on call count;
inductor's slice is the gap between that 2x and the observed 1.74x per-step** (kernel-launch-
overhead reduction on the non-dominant ops). **This run does not isolate a clean eager+cfg-off
data point**, so the exact CFG-off-alone number is inferred, not measured -- a 3rd run
(eager, cfg=off) would close that gap if the split matters for a decision.

## Profiling (rank0, torch.profiler CPU+XPU, warm N=16, cfg=off, inductor)
Total self-XPU time approx 14.85 s (down from 34.6s profiled-with-overhead in the eager+cfg=on run).
| Op | Self-XPU % | Self-XPU | Calls | Note |
|---|---|---|---|---|
| c10d::allreduce_ | **43.7%** | 3.24 s | 3497 | TP all-reduce -- grew as a SHARE (was 35.2%) |
| gemm_kernel | 29.0% | 2.15 s | 4520 | matmul compute |
| aten::addmm | 28.9% | 2.15 s | 4376 | linear layers |
| ReduceCopyKernel (bf16) | 17.8% | 1.32 s | 6120 | CCL ring reduce copy |
| allreduce_large_su_ring_write (2 variants) | 11.4%+5.8% | 1.27 s | 6120 | CCL ring kernels |
| _vllm_fa2_C::varlen_fwd | 4.9% | 0.36 s | 2040 | FlashAttention-2 (stays eager; no fake kernel) |
| triton_poi_fused_add_unsqueeze_0 (2 variants) | 4.8%+4.8% | 0.71 s | 1360 | inductor-fused elementwise |
| triton_red_fused_add_mul_native_layer_norm... (2 variants) | 4.5%+4.5% | 0.66 s | 2560 | inductor-fused RMSNorm |
| triton_poi_fused__to_copy_add_all_reduce_div_mul_rsqrt... | 0.31%+0.28% | 0.09 s | 1280 | inductor fused the pre/post-reduce elementwise around the allreduce (NOT the collective itself) |

**Communication is now a LARGER share (43.7% vs 35.2% before) even though it halved in
absolute terms (6.10s to 3.24s)** -- because CFG-off+inductor shrank compute and the small
elementwise ops faster than it shrank the collective. **The allreduce is the clear next
target.** It's latency- not bandwidth-bound at this message size (per-call time didn't drop
even though CFG-off didn't change per-call tensor size, only call count) -- so the next lever
is fewer/larger collectives, not faster ones: batch multiple blocks' reduces into one call,
overlap allreduce of block N with compute of block N+1 (currently appears synchronous --
each block's forward blocks on its own reduce before the next block starts), or move off
plain TP toward a scheme with less communication per step (e.g. CFG-parallel across ranks
if CFG must come back, rather than 2x more forwards through the same TP group).

## Artifacts
- `metrics/metrics.json` -- machine-readable metrics + repeat distribution + top-ops
- `output/profile/chrome_trace.json.gz` -- perfetto/chrome timeline (30 MB)
- `output/profile/op_table_self_{xpu,cpu}.txt` -- full key_averages tables (56 rows)
- `metrics/xpu_memory.csv` -- 6080 whole-device samples across 4 cards @400 ms
- `config/effective_config.yaml`, `scripts/run_test.sh`, `environment.txt`, `logs/run.log`
