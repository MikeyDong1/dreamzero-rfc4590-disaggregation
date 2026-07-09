# Partial-Residency Layerwise Offload — Strategy

**Target:** run the DreamZero-DROID DiT (`CausalWanModel`, 40 transformer blocks,
~32 GB in bf16) on a **single** Intel Arc Pro B60 while minimizing the per-denoise-step
PCIe transfer cost. Applies to the standalone harness
`dreamzero_run/standalone_dreamzero_layerwise.py` (TP=1, bypasses the Omni serving stack).

## Background: the stock 1-block sliding window

`vllm_omni/diffusion/offloader/layerwise_backend.py` (`LayerWiseOffloadBackend`)
keeps **only 1 of 40 blocks resident** on the XPU at a time. Each block's forward
prefetches the *next* block CPU→XPU on a side stream (overlap), then frees itself
afterward. This was tuned for the 24.5 GB B60, where the ~32 GB DiT cannot fit.

**Cost:** every denoise step re-copies **all 40 blocks** CPU→XPU. The Arc Pro
B-series has **no XeLink** — every prefetch crosses **PCIe** — so on a PCIe-bound
card this transfer dominates the step time. On the 32 GB card the stock window
peaked at only **8.1 GiB**, wasting ~24 GB of capacity.

## Strategy: keep the first K blocks permanently resident

Use the spare memory. Keep the **first K** transformer blocks **permanently
resident** on the XPU (no offload hooks at all), and apply the sliding-window
offload **only to the trailing `N−K`** blocks. The per-step CPU→XPU traffic drops
from `N` blocks to `N−K` blocks — a `K/N` reduction — while peak memory rises by
roughly `K × per_block_size`.

```
blocks[0 : K]   -> .to(device), NO hook         (resident, copied once at load)
blocks[K : N]   -> apply_block_hook sliding window over this sub-list
```

The sliding window over the trailing sub-list is otherwise identical to the stock
backend: `last_offloaded_block` prefetches `first_offloaded_block`, each block
prefetches its successor (wrapping within the sub-list), and `_prev_hook`
back-links are wired for cache-dit skip-safety.

### Auto-sizing K from a memory budget

K is derived from a budget so it adapts to the card and the measured block size:

```
per_block_bytes = sum(p.numel*p.element_size for p in block.parameters()) + buffers
K = floor(resident_mem_gib * 2^30 / per_block_bytes)
K = clamp(K, 0, N-2)          # always leave >= 2 blocks for a valid sliding window
```

`--resident-blocks N` overrides K explicitly; `--resident-mem-gib G` sets the
budget used when `--resident-blocks=-1` (auto).

### Peak-memory model (measured on this B60)

Per-block ≈ **0.75 GiB** (770 MiB). Empirically:

```
peak_GiB ≈ K × 0.75 + base_overhead,   base_overhead ≈ 8.2 GiB
```

`base_overhead` = resident non-block modules + the sliding window's transient
double-buffer + activations + KV/cross-attn caches + allocator reserve. So to
target a peak budget `P`: `K ≈ (P − 8.2) / 0.75`.

| K (resident) | offloaded | predicted peak | per-step H2D cut |
|---|---|---|---|
| 1 (stock)    | 40 | ~8.1 GiB  | 0%  |
| 22 (mem=17)  | 18 | ~24.7 GiB | 55% |
| 28           | 12 | ~29.2 GiB | 70% |
| 29           | 11 | ~30.0 GiB | 72% |

## Measured results (card0, TP=1, num_steps=4, OFFLINE no-proxy)

| Metric | Stock (K=1) | K=22 (mem=17) | **K=29 (~30GB)** |
|---|---|---|---|
| Time to first output | 3.81 s | 2.76 s (−28%) | **2.31 s (−39%)** |
| Time to complete output | 12.28 s | 7.71 s (−37%) | **6.14 s (−50%)** |
| Prefill | 4.11 s | 2.93 s | **2.45 s** |
| Peak XPU memory | 8.10 GiB | 24.70 GiB | **29.97 GiB** |
| Per-step H2D cut | 0% | 55% | **72%** |
| Output finite | yes | yes | **yes** |

**K=29 is the practical max for a 30 GB budget on this 32 GB card** — peak 29.97 GiB
landed exactly on the `K×0.75 + 8.2` model. K=30 would predict ~30.7 GiB and risks
OOM once allocator fragmentation is included, so 29 is the safe ceiling. The
remaining 6.14 s is near the ~4 s compute floor + 11 offloaded blocks × ~0.2 s.

Loop-time model (4-step run): `TTC ≈ 3.97 s compute_floor + 0.208 s × offloaded_blocks`.
The compute floor (~4 s) is the hard lower bound for this config — reachable only
if all 40 blocks were resident (~38 GiB, exceeds the 32 GB card), so the 30 GB
budget caps useful K at ≈29.

## How to run

```bash
# auto-size K to a peak budget (e.g. ~30 GB -> resident-mem-gib ~= 21.8)
python dreamzero_run/standalone_dreamzero_layerwise.py \
    --model-path "$SNAP" --num-steps 4 \
    --resident-mem-gib 21.8 \
    --out-dir dreamzero_run/output_opt
# or pin K explicitly
python ... --resident-blocks 29 ...
```

Run env (offline, no proxy): `ZE_AFFINITY_MASK=0`, `SYCL_UR_USE_LEVEL_ZERO_V2=0`,
`VLLM_WORKER_MULTIPROC_METHOD=spawn`, `HF_HUB_OFFLINE=1`, empty `http(s)_proxy`,
`LD_LIBRARY_PATH=/opt/intel/oneapi/ccl/2021.15/lib:...` first.

## Tradeoffs / notes

- **Pure win up to the memory ceiling**: residency only removes redundant copies;
  it never changes numerics (output stays bit-for-bit the sliding-window result).
- Diminishing returns: each resident block saves the same ~0.21 s/block but costs
  ~0.75 GiB; once peak approaches the card size you must stop.
- TP>1 / Omni-serving path uses `LayerWiseOffloadBackend.enable()` directly — the
  same K-resident split could be added there (split `blocks` before hooking), but
  this doc covers the standalone TP=1 harness only.
