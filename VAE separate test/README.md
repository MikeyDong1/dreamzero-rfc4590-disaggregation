# VAE separate test — isolated Wan-VAE encode timing across cards

Isolated Wan2.1-VAE `_encode` benchmark for the DreamZero preprocessing stage,
run standalone on **one card** (no UMT5, no DiT, no TP). This is the VAE that
dominates DreamZero preprocessing (~70% of the ~2.6 s warm encode window).

**Input (identical across runs):** the real obs#1 VAE input reconstructed exactly
as `DreamZeroPipeline._encode_image` builds it —
`concat([first_stitched_frame, zeros(32 frames)])` = **`(1, 3, 33, 352, 640)`** →
latent `(1, 16, 9, 44, 80)`. The 33-frame video volume is what makes it costly.

**Two dtype modes:**
- `autocast_bf16` — **faithful pipeline path** (`_encode_image` wraps the VAE in
  `torch.amp.autocast(bf16)`, so fp32 weights run bf16 convs). **Use this number.**
- `pure_fp32` — reference (no autocast).

## Results (warm, mean of 8 reps)

| Card | Node | Device ID | Mem | torch | **VAE bf16 (faithful)** | VAE fp32 | Cold bf16 |
|---|---|---|---|---|---|---|---|
| **B60** | gnr17409 | 0xe211 | 24.5 GB | 2.11.0+xpu | **1831.8 ms** | 4370.8 ms | 3252.6 ms |
| **"B70" (newer)** | srf797635 | 0xe223 | 32 GB (PCIe Gen5) | 2.12.0+xpu | **1084.0 ms** | 2762.7 ms | 4281.4 ms |

**The newer 0xe223 card is ~1.69× faster on the faithful bf16 VAE encode**
(1084 ms vs 1832 ms) and ~1.58× faster in fp32 (2763 ms vs 4371 ms). Peak XPU
~3.9 GiB either way.

## Caveats / faithfulness notes
- **B60** used vllm_omni's `DistributedAutoencoderKLWan` (default ctor, real
  `action_head.vae.*` weights streamed from the checkpoint).
- **0xe223** node had been wiped since the last visit (my workspace + image + the
  46 GB model download were gone). VAE encode wall-clock depends only on input
  shape + architecture, **not weight values**, so this run used **stock diffusers
  `AutoencoderKLWan`** with the default Wan2.1 config (random init). This is
  timing-equivalent: `OmniAutoencoderKLWan`/`DistributedAutoencoderKLWan` subclass
  diffusers `AutoencoderKLWan` with **no `__init__`/`_encode`/forward override**
  (encode just wraps `super().encode()`; the tiling executor is inactive on the
  non-distributed single-card path). Default configs verified identical:
  z_dim=16, base_dim=96, dim_mult=[1,2,4,4], temperal_downsample=[F,T,T], fp32.
- Different torch (2.11 vs 2.12+xpu) is a second variable between the two nodes,
  so treat the ~1.7× as card+stack combined, not card-silicon alone.
- The `0xe223` reported name is "Intel(R) Graphics [0xe223]" (stepping A0, PCIe
  Gen5, 32656 MiB). The user refers to it as B70; the memory note had it as a
  32 GB B60. Either way it is a distinct, newer part than gnr17409's 0xe211.

## Profiling (torch.profiler, CPU+XPU)
See **[`PROFILE_REPORT.md`](PROFILE_REPORT.md)** for the full op-level analysis of the
bf16 encode on the B60. Headline: only **38 % is convolution**; ~60 % is
memory-movement + elementwise (`copy_` 26 %, `cat` 10.5 %, norm/act 23 %) from Wan-VAE's
chunked causal feature-cache. bf16 beats fp32 2.4× and that entire win is a 4.95× conv
speedup (the memory-bound tail doesn't accelerate). Peak XPU 3.95 GiB. ~97 % of the
convolved volume is zero-padding frames (Wan I2V first-frame conditioning).

## Files
| File | What |
|---|---|
| `PROFILE_REPORT.md` | **Profiler analysis + recommendations** |
| `vae_profile.py` (in `dreamzero_xpu_run/`) | Profiling harness (torch.profiler, chrome trace + key_averages) |
| `profile/vae_profile_summary.json` | Machine-readable top-ops summary (bf16 + fp32) |
| `profile/vae_keyavg_*_{by_device,by_host,by_shape}.txt` | key_averages tables |
| `profile/vae_trace_*.json.gz` | chrome traces (gunzip → chrome://tracing / Perfetto) |
| `profile/run.log` | full run log |
| `vae_only_bench.py` | vllm_omni-based bench used on the B60 (real VAE weights) |
| `vae_only_bench_stock.py` | stock-diffusers bench used on 0xe223 (timing-equivalent) |
| `vae_only_results_B60_gnr17409.json` / `.log` | B60 timing results |
| `vae_only_results_797635_e223_32GB.json` | 0xe223 timing results |
