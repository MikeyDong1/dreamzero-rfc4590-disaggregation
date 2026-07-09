# DreamZero Wan-VAE encode — profiler report (B60 / gnr17409)

**What was profiled:** the exact VAE encode `DreamZeroPipeline._encode_image` runs for
observation #1 — the saved stitched first frame + 32 zero frames →
`(1, 3, 33, 352, 640)` bf16 → `vae._encode` → chunk → normalize → latent
`(1, 16, 9, 44, 80)`. One XPU card (`ZE_AFFINITY_MASK=0`), real
`action_head.vae.*` weights (194 tensors), single process, no UMT5/DiT/TP.

**How:** `torch.profiler` with `ProfilerActivity.CPU + XPU`, `record_shapes`,
`profile_memory`. 3 warm-up encodes (kernel JIT + allocator reach steady state),
then **5 profiled reps**. Faithful path = `autocast(bf16)` (what the pipeline does);
`pure_fp32` profiled too as the reference. Harness:
[`vae_profile.py`](../dreamzero_xpu_run/vae_profile.py).

| Env | Value |
|---|---|
| Device | Intel Arc Pro B60, device `0xe211`, 24.5 GB |
| Node | gnr17409, container `vllm-omni-dev-dreamzero` |
| torch | 2.11.0+xpu |
| Steady-state wall-clock | **1831 ms/encode (bf16)**, 4363 ms/encode (fp32) |
| Device (Self XPU) time | 1826 ms/encode (bf16) — device-bound, host keeps up |
| Peak XPU memory | **3.95 GiB** (of 24.5 GB — not memory-capacity bound) |
| Latent out | `(1, 16, 9, 44, 80)`, finite ✓ |

Artifacts in [`profile/`](profile/): `vae_profile_summary.json`, the six
`vae_keyavg_*` tables (by device / host / input-shape, ×2 dtypes), gzipped chrome
traces (`vae_trace_autocast_bf16.json.gz` 7 MB, `vae_trace_pure_fp32.json.gz` 6 MB
— `gunzip` then open in `chrome://tracing` / Perfetto), and `run.log`.

---

## 1. Where the 1831 ms goes (bf16, the faithful path)

Device-time budget per encode. The `aten::*` rows below **partition the 1826 ms
device total exactly** (they sum to 1826 ms); the named GPU kernels
(`gen_conv`, `CatArrayBatchedCopy…`, `at::native::xpu::*Kernel`) are the *same*
time re-attributed by kernel — e.g. `gen_conv 612 ms + conv_reorder 79 ms = 691 ms
= the convolution op`.

| Operation | ms/encode | % device | calls/encode | What it is |
|---|---:|---:|---:|---|
| **Convolution** (3D + a few 2D) | **691** | **37.8 %** | 278 | The actual conv math. `gen_conv` 612 ms compute + `conv_reorder` 79 ms layout |
| **`copy_`** | **474** | **26.0 %** | 1360 | Feature-cache / causal-pad tensor copies (see §3) |
| **`mul`** | 221 | 12.1 % | 397 | Norm scale + activation gating |
| **`cat`** | 191 | 10.5 % | 272 | Causal temporal-cache concatenation (see §3) |
| `add` | 73 | 4.0 % | 297 | Norm bias / residual adds |
| `div` | 53 | 2.9 % | 198 | Normalization divide |
| `silu` | 53 | 2.9 % | 189 | SiLU activations |
| `fill_` | 36 | 2.0 % | 225 | Zero-init of pad/cache buffers |
| `linalg_vector_norm` | 28 | 1.5 % | 198 | RMS/group-norm reduction |
| **attention** (`sdpa`/`micro_sdpa`) | **4.5** | **0.25 %** | 9 | The VAE's single mid-block attention over the `44×80` bottleneck — negligible |
| clamp_min, sub, memcpy | ~1 | <0.1 % | — | |

### The one-line takeaway

> **Only 38 % of the VAE encode is convolution. ~60 % is memory-movement and
> elementwise work** — `copy_` (26 %) + `cat` (10.5 %) + `mul`/`add`/`div`/`silu`/`fill`
> (23 %). The VAE is **half compute-bound, half bandwidth/overhead-bound.**

---

## 2. Which convolutions dominate (bf16, by input shape)

Grouped by conv input shape (`vae_keyavg_autocast_bf16_by_shape.txt`), share of the
691 ms conv budget:

| Conv input shape (`[N,C,T,H,W]`) | kernel | ms/enc | % of conv | Stage |
|---|---|---:|---:|---|
| `[1, 96, 6, 354, 642]` | `[96,96,3,3,3]` | 253 | **36.7 %** | Encoder stage-1, 96 ch, ~full res |
| `[1, 192, 6, 178, 322]` | `[192,192,3,3,3]` | 219 | **31.8 %** | Encoder stage-2, 192 ch, half res |
| `[1, 384, 4, 90, 162]` | `[384,384,3,3,3]` | 76 | 11.0 % | Stage-3, 384 ch |
| `[1, 384, 3, 46, 82]` | `[384,384,3,3,3]` | 38 | 5.5 % | Stage-4 |
| `[1, 96, 6, 178, 322]` | `[192,96,3,3,3]` | 28 | 4.0 % | Stage-1→2 downsample |
| `[4, 96, 353, 641]` | `[96,96,3,3]` (2D) | 15 | 2.2 % | Per-frame 2D conv |

**The top two shapes = 68 % of all convolution.** These are the *shallow,
high-resolution* stages (96 ch and 192 ch, near-full spatial). As the encoder goes
deeper, channels grow 96→192→384 but spatial shrinks faster (÷2 per axis per
downsample, plus temporal ÷2), so FLOPs concentrate in the early high-res stages.
The `354×642` / `178×322` sizes are the `352×640` / `176×320` maps + causal/spatial
padding of 1 per side.

---

## 3. Why so much `copy_` / `cat` / `pad`? — the chunked causal design

Op counts for a *single* encode are enormous: **278 convs, 272 `cat`, 260 `pad`,
1360 `copy_`, and ~4030 XPU kernel launches per encode.** That is the fingerprint of
Wan-VAE's **causal temporal processing with a rolling feature cache** (`feat_cache`):
the 33-frame volume is processed in small temporal chunks (the `T=6/4/3/2` you see in
the conv shapes), and between chunks the model **concatenates** the cached boundary
frames (`cat`), **copies** them into padded buffers (`copy_`), and applies causal
**temporal padding** (`constant_pad_nd`). This machinery — not the math — is what the
26 % `copy_` + 10.5 % `cat` + 2 % `fill_` (≈ **38 % of device time**) is spent on.

This is the single biggest *structural* lever: a larger temporal chunk (fewer
cache/pad boundaries) would cut op count and this ~38 % overhead. It is an
architecture/impl property of diffusers `AutoencoderKLWan`, not something the pipeline
tunes today.

---

## 4. bf16 vs fp32 — the speedup is entirely convolution

All numbers ms/encode, device (Self XPU) time; % = share of that dtype's device total.

| | conv | copy_ | cat | norm/act (mul+add+div+silu) | attention | **total/enc** |
|---|---:|---:|---:|---:|---:|---:|
| **fp32** device | **3419 ms (76 %)** | 185 | 185 | ~330 | 182 | **4363 ms** |
| **bf16** device | **691 ms (38 %)** | 474 | 191 | ~400 | 4.5 | **1826 ms** |
| speedup | **4.95×** | 0.39× (slower) | 1.0× | ~0.8× (slower) | ~40× | **2.39×** |

- **Convolution is 4.95× faster in bf16** — the XMX/DPAS matrix engines accelerate
  bf16 conv ~5×. That is the whole story of the 2.4× overall win.
- **The memory-bound tail does *not* speed up.** `cat`, `mul`, `add`, `div`, `silu`
  cost the same in both dtypes because (a) they are bandwidth-limited and (b)
  `autocast` keeps the weights fp32 and runs normalizations in fp32, casting only conv
  inputs to bf16. bf16 even *adds* device `copy_` time (474 vs 185 ms) from the
  autocast cast tensors.
- Classic Amdahl: once conv is 5× faster, the un-accelerated 60 % memory/elementwise
  tail becomes the co-bottleneck — which is exactly why overall is 2.4×, not 5×.

(Note: the fp32 table shows a 20 s host_total under `sdpa` — that is just where the
per-step `zeEventHostSynchronize` blocking-wait happened to land in the async queue,
not real attention cost. The real device attention cost is 182 ms fp32 / 4.5 ms bf16.)

---

## 5. Host vs device — is it launch-bound?

No. Self CPU (9.27 s) ≈ Self XPU (9.13 s) over the 5 reps, and `zeCommandListHostSync`
= 4.89 s is the host *waiting* on the device (our explicit per-step sync). Kernel-launch
overhead (`urEnqueueKernelLaunch`, 4030 launches/encode) is 124 ms/encode — real, but
« the 1826 ms device time, so the encode is **device-bound; the host keeps the queue
fed.** The 4030 launches/encode is a symptom of the chunked design (§3), not a
bottleneck in itself here.

---

## 6. The highest-level observation

The pipeline pays the **full 33-frame video-VAE cost to encode 1 real image + 32 zero
frames.** `_encode_image` builds `concat([first_frame, zeros(1,3,32,352,640)])`; dense
3D convolution does identical FLOPs whether the frames are real or zero, so **~97 % of
the convolved volume is padding.** This is Wan I2V's first-frame conditioning format
(the DiT expects the 33-frame latent `(1,16,9,44,80)`), so it is not trivially
removable — but it means the ~1.8 s is spent almost entirely convolving zeros.

---

## 7. Recommendations (in leverage order)

1. **Reduce the encoded temporal length / exploit the zero frames.** The biggest
   theoretical waste (§6): 32 of 33 frames are zeros. If the conditioning latent for
   the zero region could be produced analytically (a single image-encode + known
   zero-input latent) instead of a full 33-frame 3D conv, most of the 1.8 s
   disappears. Requires validating the DiT accepts an equivalently-constructed latent —
   a model-level change, highest payoff.
2. **Larger causal temporal chunk in the Wan VAE** (§3) — cut the 272 `cat` / 260 `pad`
   / 1360 `copy_` per encode; targets the ~38 % memory-movement tail. Impl-level.
3. **Full-bf16 VAE weights (not just autocast)** — let the normalization/elementwise
   ops (§1, ~23 %) run bf16 and drop the extra autocast cast-copies (474→~185 ms).
   Risk: Wan VAE normalizations are often kept fp32 for stability; validate latent
   drift before adopting.
4. **Newer silicon** — the `0xe223` ("B70") card already does this same encode in
   **1084 ms vs 1832 ms (1.69×)** (see [`README.md`](README.md)); the conv-heavy 38 %
   is what scales with the better matrix engine + memory bandwidth.
5. Convolution itself (38 %) is already bf16-accelerated ~5× and is near the
   irreducible floor for a `352×640×33` volume — only smaller input resolution reduces
   it. Not a software lever.

Memory is a non-issue: 3.95 GiB peak on a 24.5 GB card.
