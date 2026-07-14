# DreamZero DiT-only — Tensor Parallel (TP=4) profiling run

**Status:** ✅ success (exit 0)  •  **Node:** gnr17408 (8× Intel Arc Pro B60, 22.7 GiB)
**Cards:** 4,5,6,7 (ZE_AFFINITY_MASK → in-container xpu:0..3)  •  **Run ID:** 20260710_141433_gnr17408_dit_TP4_cards4567_N16_dev7

## What was tested
Only the **DiT** (`action_head.model.*`, CausalWanModel, 40 blocks) under a **real 4-rank
Tensor-Parallel world** (torchrun --nproc_per_node=4, tensor_parallel_size=4). One request,
16 denoise steps. Inputs were the **pre-encoded** `dit_inputs.pt`; the UMT5 text encoder and
Wan VAE were **never loaded to the XPU** (streamed only DiT weights + `enable_layerwise_offload`
keeps the VAE off-device). Each rank holds its 1/4 weight shard **fully resident (no offload)**.

- **Image:** `vllm-omni-xpu:mikey-dev7` (torch 2.11+xpu). NOTE: `:latest` v0240 now fail-fasts
  requiring the AR-Diffusion engine for direct `_prefill_kv_cache`/`diffuse` drive, so dev7
  (the image the TP=1 baseline was measured on) was used for an apples-to-apples comparison.

## The 5 required metrics (warm; clean timed run, no profiler)
| Metric | Value |
|---|---|
| model_load_time | **152.67 s** (build pipeline + stream+shard DiT weights, 4 ranks) |
| time_to_first_output | **2.15 s** (prefill 0.98 s + first denoise step) |
| decode_time | **n/a** — VAE decode not run in a DiT-only test |
| complete_output_time | **20.17 s** (prefill 0.98 s + 16-step denoise 19.19 s) |
| peak_xpu_memory | **9.74 GiB/card** process alloc (9972 MiB) · **12.82 GiB/card** whole-device peak |

Per-step denoise: mean **1.199 s** (min 1.098, max 1.327), 16 steps.
Output shapes: video [1,2,16,44,80], action [1,24,32] — both finite. All 4 ranks step-locked
(identical per-step times → TP collectives healthy).

## TP=4 vs TP=1 baseline (same model, same N=16, same dev7 image)
| Config | complete (s) | per-step (s) | peak alloc/card | offload |
|---|---|---|---|---|
| TP=1 K10 (30 blk offloaded) | 25.93 | 1.558 | 13.7 GiB | sliding-window, H2D 15.2 s |
| TP=1 K20 (20 blk offloaded) | 21.50 | 1.293 | 21.5 GiB | sliding-window, H2D 10.1 s |
| **TP=4 full-resident** | **20.17** | **1.199** | **9.74 GiB** | **none** |

TP=4 is the fastest **and** lightest per card: the 1/4 shard (~7.7 GiB weights) fits fully
resident, eliminating the H2D weight-streaming that dominated TP=1 (10–15 s of the runtime).
Speedup is modest (1.07× vs the best TP=1 K20, 1.29× vs K10) because it trades the offload
bottleneck for TP all-reduce traffic — but it does so at <½ the per-card memory.

## Profiling (rank0, torch.profiler CPU+XPU, warm N=16)
Total self-XPU time ≈ 17.3 s. **TP communication is the dominant device cost:**
| Op | Self-XPU % | Self-XPU | Calls | Note |
|---|---|---|---|---|
| c10d::allreduce_ | **35.2%** | 6.10 s | 6994 | TP all-reduce (RowParallel + QK-norm fused) |
| gemm_kernel | 24.8% | 4.30 s | 9040 | matmul compute |
| aten::addmm | 24.7% | 4.29 s | 8752 | linear layers |
| ReduceCopyKernel (bf16) | 15.3% | 2.65 s | 12240 | CCL ring reduce copy |
| aten::add | 11.8% | 2.05 s | 16917 | elementwise |
| allreduce_large_su_ring_write | 9.8%+4.9% | 2.55 s | 12240 | CCL ring all-reduce kernels |
| aten::native_layer_norm | 5.4% | 0.94 s | 4182 | |
| _vllm_fa2_C::varlen_fwd | 4.2% | 0.73 s | 4080 | FlashAttention-2 |

**Takeaway:** ~60–65% of DiT self-XPU time at TP=4 is spent in collective communication
(allreduce + CCL ring copies), not compute. The gemm/addmm compute (~50% combined) is what
TP is meant to shrink per-card; the all-reduce is the price. On this 8×B60 box TP=4 mainly
buys **memory headroom** (removes offload) with a small latency win — scaling past the
communication overhead would need faster inter-card links or fewer/cheaper all-reduces
(e.g. sequence/CFG parallel, or a coarser TP-group reduction).

## Artifacts
- `metrics/metrics.json` — machine-readable metrics + top-ops
- `output/profile/chrome_trace.json.gz` — perfetto/chrome timeline (72 MB)
- `output/profile/op_table_self_{xpu,cpu}.txt` — full key_averages tables
- `metrics/xpu_memory.csv` — 6012 whole-device samples across 4 cards @400 ms
- `config/effective_config.yaml`, `scripts/run_test.sh`, `environment.txt`, `logs/run.log`
