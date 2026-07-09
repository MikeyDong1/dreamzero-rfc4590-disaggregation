# Wan-VAE encode — torch.compile(inductor) vs eager, kernel-launch count

**Question:** how many XPU kernels are launched per VAE encode with **torch-inductor on**
(`torch.compile(backend="inductor")`) vs **enforce-eager** — run twice each.

**What was profiled:** the exact obs#1 VAE encode `DreamZeroPipeline._encode_image` performs —
stitched first frame + 32 zero frames → `(1,3,33,352,640)` bf16 → `vae._encode` → chunk →
normalize → latent `(1,16,9,44,80)`, faithful path = `autocast(bf16)`. Stock diffusers
`AutoencoderKLWan` (default Wan2.1 config) on one XPU card (`ZE_AFFINITY_MASK=0`); kernel
count is structural (weight-value-independent), and stock diffusers avoids `import vllm_omni`
which otherwise disables triton-xpu and breaks the inductor backend.

**How counted:** `torch.profiler` (CPU+XPU). The headline **launches/encode = the
`urEnqueueKernelLaunch` count** (host-side Level-Zero/UR dispatch, fires once per compute
kernel; `zeCommandListAppendLaunchKernel` matches it exactly). `memcpy/encode` =
`urEnqueueUSMMemcpy`. 4 warm-up encodes (compile + JIT + allocator reach steady state, NOT
counted) then **5 profiled reps**; **each setting run twice** — both passes identical.

| Env | Value |
|---|---|
| Device | Intel Graphics `0xe223`, 32 GB (node srf797635, sdp@10.23.14.76) |
| Container | `vllm-omni-xpu:latest`, torch 2.11.0+xpu, triton-xpu 3.7.0 |
| Input | `(1,3,33,352,640)` bf16 (real obs#1 stitched npz) → latent `(1,16,9,44,80)`, finite ✓ |
| Peak XPU | 3.59 GiB |

---

## Headline — kernels launched per encode

| Setting | **launches / encode** | memcpy / encode | distinct kernel names | wall ms/encode |
|---|---:|---:|---:|---:|
| **eager (enforce_eager)** — pass 1 | **4020** | 177 | 26 | 1171.6 |
| **eager (enforce_eager)** — pass 2 | **4020** | 177 | 26 | 1162.9 |
| **inductor on** — pass 1 | **1655** | 0 | 144 | 631.5 |
| **inductor on** — pass 2 | **1655** | 0 | 144 | 631.5 |

> **Eager launches 4020 compute kernels/encode (+177 memcpy). Inductor launches 1655
> (0 memcpy).** That is **2.43× fewer launches (−58.8%)**, and the wall-clock drops
> **1172 → 632 ms/encode (1.86× faster)**. Both passes are bit-for-bit identical in count.

(4020 launches + 177 memcpy = 4197 total XPU device events in eager — matches the ~4030
"kernel launches per encode" from the earlier eager-only profile report.)

## Why the count drops — and why not further

Inductor does **not** touch the convolutions. `gen_conv` fires **1390×** and `conv_reorder`
~2400× in *both* settings (these are oneDNN library calls, not fusible); in the inductor run
convolution is still 61% of device time. The entire reduction comes from **fusing the
memory-bound elementwise/normalization/cat/pad/copy tail** — in eager these are ~2600
separate `at::native::xpu::*ElementwiseKernel` / `CatArrayBatchedCopy` / `copy_` launches;
inductor collapses them into ~130 generated `triton_poi_fused_*` / `triton_red_fused_*`
kernels (each fusing many ops: e.g.
`triton_poi_fused__to_copy_add_cat_clamp_min_clone_constant_pad_nd_convolution_div_...`).
That is also why the **distinct-name count goes UP** (26 → 144): eager reuses a few generic
elementwise templates thousands of times; inductor emits many bespoke fused kernels, each
called far fewer times. The 1655 residual launches are the un-fusible conv library calls plus
the fused-kernel invocations around them.

The 1.86× wall-clock (< the 2.43× launch cut) is Amdahl: convolution — unchanged — is still
~60% of the time, so halving the memory-bound tail can't more than roughly halve total.

## Artifacts

- `vae_inductor_vs_eager_summary.json` — machine-readable, all 4 passes + per-kernel tallies
- `keyavg_{eager,inductor}_pass{1,2}.txt` — full `key_averages` device-time tables
- Harness: [`vae_inductor_vs_eager.py`](../../dreamzero_xpu_run/vae_inductor_vs_eager.py)
  (on node `/home/sdp/mikey_vae_test/`)
