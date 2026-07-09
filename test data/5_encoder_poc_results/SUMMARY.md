# DreamZero parallel-encoder POC — test results & profile analysis

**Date:** 2026-07-08 · **Node:** gnr17408 (`sdp@10.54.109.211`)
**Hardware:** 2× Intel Arc Pro B60 Graphics (22.7 GiB each), cards 0+1
**Container image:** `vllm-omni-xpu:latest` = **v0240** (id `9a1d6f2e…`, built 2026-07-07)
**Stack:** torch 2.12.0+xpu · vLLM 0.24.0 · vllm-omni 0.1.dev6 · Python 3.12
**Scripts:** `dreamzero_parallel_encoders.py` + `parallel_encoders.py` (benchmark), `pe_profile.py` (profiler), `pe_probe.py` (probes)

> **Update (2026-07-08, part 2):** two follow-ups were added after the initial eager
> run below — (A) a root-cause investigation of *why* `one_card_stream` gave no
> overlap, and (B) enabling **torch.compile(inductor)** for the VAE (instead of
> enforce-eager), which roughly **halves** the encode-to-DiT time. See the two new
> sections **"Why one_card_stream shows no overlap"** and **"Inductor VAE results"**
> at the end. The original eager numbers are unchanged and kept for reference.

## What was tested

The **two DreamZero encoders only** — the UMT5-xxl text encoder and the Wan VAE
encoder — run either **serially** (default, the production path) or **in parallel**.
**The DiT (CausalWanModel, ~28 GB) is never built or loaded**; this measures purely
the time to produce the tensors that would be handed to the DiT boundary.

- **Input data:** `1_huggingface_plain` / `2_vae_input_3videos` (the real HF-plain
  DROID observation).
  - VAE input: real stitched 3-view first frame `model_input_stitched.npz`
    `images (1,352,640,3) uint8` → conditioning window `(1,3,33,352,640)` bf16.
  - Text input: the real prompt *"Move the pan forward and use the brush in the
    middle of the plates to brush the inside of the pan"* → UMT5 tokens `(1,512)`.
- **Model:** `GEAR-Dreams/DreamZero-DROID` (snapshot `96ad3441…`), encoder weights
  only (242 text params + 194 VAE params loaded from the root checkpoint).
- **Timing method:** 3 warm-up + 10 measured runs; device synchronized before and
  after each measured region; **model load excluded** from all timings; the measured
  region contains no host syncs (`.cpu()`/`.item()`/print). Correctness is checked
  against the serial reference and against the exact pipeline `_encode_text` math.

## Modes

| Mode | Meaning |
|---|---|
| `serial` | text encode **then** VAE encode, one card, one stream (**the default / current behavior**) |
| `one_card_stream` | text ‖ VAE on **two SYCL streams of one card** |
| `two_card` | text on `xpu:0`, VAE on `xpu:1` (upper bound; +d2d gather of the latent) |

---

## Results — time to produce the DiT-input tensors

Benchmark: 3 warm-up + 10 timed runs, mean ms. Speedup is vs the serial baseline.

| Mode | total ms | text ms | VAE ms | overlap ms | peak mem GiB | speedup | correct vs serial |
|---|---:|---:|---:|---:|---:|---:|:--:|
| **serial** (default) | **2352.6** | 180.1 | 2172.2 | 0.0 | 14.17 | 1.000× | — (reference) |
| **one_card_stream** | 2351.6 | 179.2 | 2178.7 | 0.7 | 14.17 | 1.000× | ✅ bit-identical |
| **two_card** | **2174.4** | 180.1 | 2172.2 | **177.9** | 10.85 | **1.082×** | ✅ bit-identical |

- **Isolated component times** (single stream): text = **180 ms**, VAE = **2172 ms**.
  The VAE encode is **~12× longer** than the text encode.
- **two_card d2d transfer** (VAE latent `xpu:1 → xpu:0` for the DiT boundary): **1.34 ms** — negligible.
- **Correctness:** `ALL_CORRECT=True` in every mode. The sync-free text-padding mask
  used in the parallel path is **bit-identical** to the pipeline's device-scalar slice
  (`max_abs_diff = 0.0`, verified via `--verify-against-pipeline`).
- **Serial run-to-run spread was tiny** (min 2350.9 / max 2354.6 ms over 10 runs) — the numbers are stable, not noise.

### Headline

- **Serial (default) time to DiT input ≈ 2.35 s**, of which the text encode is 180 ms (7.7%).
- **One-card two-stream does NOT help** (overlap ≈ 0.7 ms → 0% speedup).
- **Two-card hides the whole text encode** (overlap ≈ 178 ms ≈ the isolated text time) →
  total drops to **2.17 s, a 1.08× speedup**, at the cost of a second card.

---

## Profile analysis (torch.profiler, CPU+XPU activities)

Profiled 5 reps per mode after warm-up. Steady-state wall-clock reproduced the
benchmark exactly (serial 2353 ms, one_card 2349 ms, two_card 2175 ms), so the
profiled window is representative. Op breakdown below is **serial**, summed over the
5 profiled encodes (`Self XPU time total: 11.721 s` ≈ 2.34 s/encode).

### Where the device time goes (self-XPU %, serial)

| Op (aten / kernel) | Self XPU % | Belongs to | Note |
|---|---:|---|---|
| `aten::copy_` | **34.1%** | VAE (+dtype casts) | fp32↔bf16 casts + layout copies; 12,270 calls |
| `aten::convolution_overrideable` / `gen_conv` | **30.6% / 26.3%** | VAE | the 3D conv stack — the real compute core |
| `ElementwiseGlobalRangeKernel` | 16.4% | VAE | conv epilogue / broadcasting |
| `aten::mul` | 10.7% | VAE | normalization / gating |
| `aten::add`, `conv_reorder`, `aten::silu`, `aten::div`, `fill_` | 3–6% each | VAE | resnet blocks + activations |
| `gemm_kernel` / `aten::mm` | **2.6% / 2.4%** | **text (UMT5)** | the entire text encoder's matmuls |
| `aten::_softmax`, `bmm`, `micro_sdpa` (SDPA) | <0.4% each | text/VAE attention | attention is not the bottleneck |

*(The `aten::convolution_overrideable` XPU % overlaps with its child `gen_conv`/`conv_reorder`
kernels — they're the same conv work counted at op vs kernel granularity, not additive.)*

### Interpretation

1. **The VAE is ~98% of the encoder device time; the text encoder is ~2–3%.**
   The convolution stack (`convolution_overrideable`+`gen_conv`+`conv_reorder` ≈ 61% of
   self-XPU) plus its elementwise epilogues (`copy_`, `mul`, `add`, `silu`, `div` ≈ 35%)
   dominate. UMT5's `gemm`/`mm` is a rounding error by comparison. This is exactly why
   the *isolated* text (180 ms) is ~12× cheaper than the VAE (2172 ms).

2. **Why one-card two-stream gives ~0 overlap.** The VAE conv kernels already saturate
   the B60's compute (EUs) and, notably, its memory system — `aten::copy_` alone is 34%
   of device time, i.e. the VAE is heavily **memory-bandwidth bound**, not latency bound.
   A second SYCL stream carrying the tiny text GEMMs finds no idle execution resource to
   slot into, so the text work serializes behind the VAE anyway. The profiler confirms
   the op mix is unchanged between serial and one_card_stream (identical per-op XPU %),
   consistent with no genuine concurrency on one card.

3. **Why two-card works — and its ceiling.** Putting the text encoder on a *separate*
   card lets its 180 ms run fully concurrently with the VAE's 2172 ms. Because the two
   `run_*` helpers are sync-free, the host launches both cards back-to-back and the text
   finishes "for free" inside the VAE window. The observed overlap (178 ms) ≈ the full
   isolated text time, so the hiding is essentially perfect. But the **ceiling is low**:
   since text is only 7.7% of the serial total, hiding *all* of it caps the speedup at
   ~1.08×. The d2d latent gather (1.34 ms) does not erode it.

4. **Memory.** Peak drops from 14.17 GiB (serial/one-card, both encoders + activations
   on one card) to 10.85 GiB on the busier card in two-card (VAE-only card holds the
   large conv activations; text card holds UMT5). Both fit comfortably in 22.7 GiB.

### Takeaways for the parallelization feature

- **Do not expect a win from one-card multi-stream** for this encoder pair on B60 — the
  VAE is bandwidth-bound and leaves no slack. This is a hardware-utilization result, not
  a scheduling bug (correctness and op-mix both confirm it).
- **Two-card overlap is real but small** (~8%), because the text encoder is a tiny slice
  of encoder time. It only pays off if a spare card is otherwise idle at that moment.
- **The lever that matters is the VAE itself** — the 3D conv stack and its fp32↔bf16
  `copy_` traffic. Speeding up encoder-stage latency means attacking VAE conv/cast cost
  (e.g. reduce redundant dtype copies, fuse conv epilogues, or shard the VAE), not
  parallelizing text against it.
- **The parallel path is safe to adopt**: every mode reproduced the serial tensors
  bit-for-bit, so enabling two-card overlap is a pure latency optimization with no
  numerical change to what the DiT receives.

---

# Follow-up A — Why `one_card_stream` shows no overlap (root cause)

**Question:** the parallel POC's one-card two-stream mode produced ~0 ms overlap.
Is that a bug in `encode_parallel_one_card`, or a property of the hardware?

**Method:** two direct probes on the same B60 (`pe_probe.py`, results in `probe/`).

### Evidence 1 — the per-stream timings are a perfect *sum*, not a bug

From the eager `parallel_all_results.json` `stream_diag`:

```
vae_stream = 2178.7 ms,  text_stream = 179.2 ms,  wall total = 2351.6 ms
2178.7 + 179.2 = 2357.9  ≈  2351.6  →  the two streams run back-to-back, not together
```

If the streams overlapped, wall would approach `max(2178.7, 179.2) ≈ 2179 ms`. Instead
it equals their **sum**. That is textbook serialization.

### Evidence 2 — even two *independent* streams don't overlap on one B60

`pe_probe.py` PROBE B ran two **independent** matmul chains (no shared data, pure
device compute) and compared same-stream vs two-stream wall-clock:

| test | wall ms |
|---|---:|
| 1 chain (isolated) | 702.6 |
| 2 chains, **same** (default) stream | 1405.1 |
| 2 chains, **two separate** streams | 1405.5 |

Gain from using two streams: **−0.3 ms (none).** Repeated with immediate-command-lists
**off** (`SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0`, `UR_L0_USE_IMMEDIATE_COMMANDLISTS=0`):
still 1405.3 ms, gain −0.7 ms. Both env configs → identical serialization.

### Conclusion (CORRECTED — see note below)

**`encode_parallel_one_card` is implemented correctly** — it enqueues both streams
sync-free. Measured fact: **one-card two-stream yields no measurable speedup for this
encoder pair** under the tested settings (default, and immediate-command-lists off).

> ⚠️ **Correction (2026-07-08):** An earlier version of this section over-claimed that
> this *proves* the B60 "cannot co-execute kernels from multiple `torch.xpu` streams —
> a hardware limitation." **That claim is withdrawn as not proven.** The probe it rested
> on (two identical 4096-fp32 matmul chains) was confounded by **saturation**: a single
> such chain already maxes out the GPU, so two serialize on *any* GPU, revealing nothing
> about stream scheduling. A corrected experiment matrix (occupancy sweep + non-saturating
> workloads + a two-card positive control) was run to settle it, but its measurement
> harness **failed its own positive control** (two separate GPUs also measured as
> "no overlap," due to the XPU event-timeline returning zeros and Python host-launch/GIL
> overhead). So we could **not** cleanly determine whether the cause is hardware
> (no stream co-scheduling) or saturation. What remains defensible: *for this workload,
> one-card streaming does not help.* Full analysis: `../../one-card-parallel-paradox/README.md`.

Either way, the productive change for the single-card path is **not** stream tricks but
making the VAE itself faster — which is Follow-up B.

---

# Follow-up B — Inductor VAE results (torch.compile instead of enforce_eager)

Per request, the VAE encode was switched from **enforce-eager** to
**`torch.compile(backend="inductor")`**. A new `--compile-vae` flag threads a compiled
encode through all three modes (`compile_vae_encode()` in `parallel_encoders.py`). The
one-time compile (~56 s first graph, cached ~2 s after) happens in warm-up and is
**excluded from all timings**, per the POC rules.

**First, viability** (`pe_probe.py` PROBE A) — the older VAE-only report claimed importing
`vllm_omni` breaks triton-xpu/inductor. **That does NOT hold on the v0240 image:** the
real vllm_omni VAE encode compiled and ran correctly (torch 2.12+xpu, triton-xpu working;
XPU platform reports `supports_torch_inductor=True`).

## Results — inductor VAE, time to DiT-input tensors

Benchmark: 3 warm-up + 10 timed runs. **Speedup column is vs the eager serial baseline (2352.6 ms).**

| Mode (inductor VAE) | total ms | text ms | VAE ms | peak GiB | speedup vs eager-serial | correct |
|---|---:|---:|---:|---:|---:|:--:|
| **serial** | **1164.3** | 178.7 | **987.5** | 15.43 | **2.02×** | ✅ bit-identical to eager serial |
| one_card_stream | 1164.7 | 178.9 | — † | 15.43 | 2.02× | ✅ bit-identical |
| **two_card** | **989.2** | 178.7 | 987.5 | 10.85 | **2.38×** | ✅ within tol (abs 3.1e-2) |

- **Isolated VAE encode: 2172 ms → 987.5 ms = 2.20× faster** with inductor (matches the
  standalone probe: eager 2173 → inductor 985 ms). This is the whole story of the win.
- **Serial encode-to-DiT: 2352.6 → 1164.3 ms (2.02×)** just from compiling the VAE.
- **two_card + inductor: 989.2 ms — 2.38× over the original eager-serial baseline** — the
  best of both: inductor-fast VAE *and* the text encode hidden on the second card
  (overlap 177 ms). d2d gather 0.99 ms.
- Peak memory rises slightly (14.17 → 15.43 GiB serial) — inductor's fused kernels hold a
  bit more scratch; still well within 22.7 GiB.

† **one_card `vae_stream` diagnostic reads 3546 ms — this is an artifact, not the result.**
The authoritative 10-run `total_ms` is **1164.7 ms** and its VAE output is **bit-identical
to serial** (`max_abs_diff = 0.0`). The inflated per-stream number comes from the separate
event-instrumented diagnostic run, where the compiled graph is re-traced/guard-checked the
first time it executes inside a non-default stream context (a one-off recompile caught
inside the timed event). It does not affect the benchmarked total. Either way, one-card
streaming did not help this pair (see the corrected Follow-up A note above).

## Profile analysis — eager vs inductor (serial, self-XPU %)

From `inductor_profile/pe_keyavg_serial_by_device.txt` (5 profiled encodes,
`Self XPU total 5.78 s` ≈ 1.16 s/encode — half the eager 2.34 s):

| Op class | **eager** self-XPU % | **inductor** self-XPU % | what changed |
|---|---:|---:|---|
| Convolutions (`convolution_overrideable`/`gen_conv`/`conv_reorder`) | ~61% | **~67%** (59.7 + 52.7 + 7.0 overlap-counted) | **unchanged in absolute time** — oneDNN library calls, not fusible |
| `aten::copy_` (dtype/layout) | **34.1%** | **1.75%** | collapsed into fused kernels |
| loose elementwise (`mul`,`add`,`silu`,`div`,`fill_`, Elementwise*Kernel) | ~35% (many kernels) | folded into `triton_poi_fused_*` / `triton_red_fused_*` | fused |
| text `gemm`/`mm` | ~2.4% | ~5% (larger share only because total shrank) | unchanged |

**Interpretation:**

1. **Inductor did exactly what the earlier kernel-count study predicted.** It does not
   touch the convolutions (still `gen_conv` ×1390, `conv_reorder` ×2380 — oneDNN calls).
   The entire ~2.2× VAE win comes from **fusing the memory-bound elementwise/pad/cat/copy
   tail** into ~130 generated `triton_poi_fused_*` / `triton_red_fused_*` kernels. The
   giant eager `aten::copy_` cost (34% → 1.75%) is the clearest signature: the redundant
   fp32↔bf16 casts and layout copies are now fused into the producing/consuming kernels.
2. **Convolution is now the hard floor.** With the tail fused away, `gen_conv` +
   `convolution_overrideable` are ~57–67% of a now-halved device budget. Going faster than
   ~987 ms/encode would require attacking the conv stack itself (different conv algo,
   lower precision, or VAE spatial sharding), which inductor cannot do.
3. **This is Amdahl-consistent:** the standalone study saw kernel launches drop 2.43× but
   wall-clock only 1.86× because convolution (unchanged) dominates. Here we see 2.20× on
   the full normalized encode — same mechanism.

## Recommendation

- **Use inductor for the VAE (drop enforce_eager).** It is a **2.0–2.2× single-card win**
  on the encode-to-DiT latency, verified numerically correct, with only a one-time compile
  cost that is amortized immediately. This is far larger than anything encoder-parallelism
  can offer.
- **Combine with two-card** for the maximum measured result (**2.38×** vs eager-serial):
  inductor-fast VAE on one card + the text encode hidden on a second.
- **Do not pursue one-card multi-stream** — it gave no measurable speedup for this pair
  under tested settings (Follow-up A). *(Note: this is a workload result; we did not prove
  it's a hardware impossibility — see the corrected Follow-up A note.)*
- The remaining floor is the **oneDNN 3D-convolution stack**; that is the next lever if
  further VAE speedup is needed.

---

## Files in this folder

```
5_encoder_poc_results/
├── SUMMARY.md                          ← this file
├── serial/serial_results.json          ← Test 1: EAGER serial baseline (10-run timings)
├── parallel/parallel_all_results.json  ← Test 2: EAGER serial + one_card + two_card + correctness
├── profile/                            ← Test 3: EAGER profile (keyavg tables + summary + traces)
│   ├── pe_profile_summary.json
│   ├── pe_keyavg_<mode>_by_device.txt / _by_host.txt
│   └── pe_trace_<mode>.json.gz         ← chrome traces (open in perfetto / chrome://tracing)
├── inductor/inductor_all_results.json  ← Follow-up B: INDUCTOR serial + one_card + two_card + correctness
├── inductor_profile/                   ← Follow-up B: INDUCTOR profile
│   ├── pe_profile_summary.json
│   ├── pe_keyavg_<mode>_by_device.txt
│   └── pe_trace_<mode>.json.gz
└── probe/                              ← Follow-up A/B probes
    ├── probe_default_env.json          ← A: inductor viability + B: stream concurrency (default env)
    └── probe_no_immediate_cmdlist.json ← B: stream concurrency (immediate-cmdlists off)
```

### How to reproduce (on gnr17408, container `vllm-omni-pe-mikey-gnr17408`)

```bash
# EAGER (default / enforce_eager): encoders only, all modes, 10 timed runs, pipeline-parity check:
bash /workspace/pe/scripts/pe_run_encoders.sh all /workspace/pe/results/parallel/parallel_all_results.json
# INDUCTOR VAE: same, add --compile-vae (compile excluded from timing):
bash /workspace/pe/scripts/pe_run_encoders.sh all /workspace/pe/results/inductor/inductor_all_results.json --compile-vae
# profiled (add --compile-vae for the inductor profile):
python pe_profile.py --model-path <snapshot> --stitched-npz <stitched.npz> \
   --modes serial,one_card_stream,two_card --warmup 3 --active 5 --out-dir <out>
```

*Model load (~175–184 s) is excluded from all reported timings, per the POC timing rules.*
