# The One-Card Parallel "Paradox" — what we measured, and what we could NOT prove

**Question that started this:** In the DreamZero encoder POC, the `one_card_stream` mode
(run the UMT5 text encoder and the Wan VAE encoder on two separate `torch.xpu` streams of
the *same* Arc Pro B60 card) produced **~0 ms overlap** — no speedup over running them
serially. Why?

**Honest bottom line (read this first):**

1. **Measured fact (trustworthy):** For the actual DreamZero text+VAE encoder pair on
   this B60, putting the two encoders on two `torch.xpu` streams of one card gave **no
   measurable speedup** — wall-clock ≈ the serial sum — under every configuration we
   tried (default, and immediate-command-lists off). This is a real, reproducible result
   about *this workload on this stack*.
2. **What we could NOT prove:** *Why*. We could not cleanly establish whether that is
   because (**H_hw**) the hardware/runtime genuinely will not co-schedule two streams on
   one card, or (**H_sat**) it can, but these workloads left no idle resource. **A first
   "proof" of H_hw in an earlier version of this folder was wrong** (see below), and the
   corrected experiments designed to settle it **failed their own positive control** —
   our measurement harness could not even detect the overlap of two *separate GPUs*, so it
   was not trustworthy enough to rule H_hw in or out.

So the correctly-scoped claim is: **"one-card two-stream overlap does not help this
encoder pair under the settings tested"** — NOT "multi-streaming is impossible on this
hardware." The productive single-card lever remains **compiling the VAE with inductor**
(~2.2×, verified — see `../test data/5_encoder_poc_results/`).

3. **What we CAN now rule out — a POC-code bug (added 2026-07-08).** A separate concern
   was: maybe the zero-overlap is simply because the POC/VAE code contains a hidden
   host<->device sync, so the CPU blocks inside the VAE encode and never reaches the
   text-enqueue line (a *code* cause, not hardware). We tested this directly with an
   **async-launch check** (`pe_async_launch_check.py`) that times how long the *real*
   `run_vae_encoder` takes to **return** (enqueue) vs to **finish** (device idle):
   - VAE enqueue returns in **182 ms** but the device runs for **2174 ms** → ratio **0.08**.
   - The exact POC one-card enqueue sequence returns in **245 ms** vs **2350 ms** → **0.10**.
   The enqueue returns ~10–25× sooner than the compute completes, i.e. the VAE launch is
   **asynchronous** — the host is *free* during the VAE compute and the POC *does* reach
   and enqueue the text stream while the VAE is still running. **So the zero-overlap is
   NOT a POC host-sync bug.** The code does exactly what it claims. (Raw:
   `evidence/async_launch_check.json`.) This narrows the cause to device-side scheduling
   or saturation — the very thing we could not instrument — and leaves the practical
   conclusion unchanged.

---

## Why the original "impossible" conclusion was wrong

The first version of this folder claimed to *prove* H_hw with a probe that ran **two
identical 4096×4096 fp32 matmul chains** on two streams and showed wall ≈ 2× one chain.
That probe was **fatally confounded**:

- **Saturation confound.** A single 4096-fp32 matmul chain already saturates the B60's
  compute units. Two workloads that each individually saturate the GPU serialize on
  **any** GPU — including ones that fully support concurrent streams — simply because
  there is no idle resource for the second to use. So "wall ≈ 2×" is equally consistent
  with H_hw *and* H_sat. The probe could not tell them apart.
- **Invalid control.** It claimed an NVIDIA run would overlap; in fact two saturating
  cuBLAS chains serialize on NVIDIA too, so that "control" would not have discriminated.

`demonstrate_one_card_serialization.py` in this folder is that **original flawed probe,
kept deliberately as a documented example of the mistake** (its header now says so). Do
not cite its output as proof of anything.

---

## What we did to try to settle it — and why it was inconclusive

We designed a proper experiment matrix (saturation sweep + non-saturating workloads +
complementary compute-vs-bandwidth pair + a two-card positive control that must show
overlap if the instrument works). Scripts:
`onecard_concurrency_matrix.py` and `onecard_calib_threaded.py`.

The experiments ran, and produced two genuinely useful sub-results **plus** the reason we
stopped:

### Useful result 1 — the occupancy sweep is valid (E1)

We *did* find provably **non-saturating** workloads (single-stream, well below peak
throughput), so the "workload was too big" objection could be removed:

| size | bf16 throughput | % of peak | saturated? |
|---|---:|---:|:--:|
| N=512 | 7.9 TFLOP/s | **10%** | no (S*) |
| N=1024 | 53 TFLOP/s | 66% | approaching |
| N=2048 | 73 TFLOP/s | 90% | **yes (N_sat)** |
| N=4096 | 81 TFLOP/s | 100% | yes |

(bf16 peak ≈ 81 TFLOP/s; fp32 ≈ 11 TFLOP/s. Full data in `evidence/matrix_default_env.json`.)

### Useful result 2 — at a non-saturating size, one-card two-stream *still* showed R≈2 (E2/E3)

Running two independent **non-saturating** chains (N=512, ~10% of peak) on two streams:
wall ≈ 2× one chain (R≈1.99–2.04), and the complementary compute-vs-bandwidth pair also
showed no hiding. **Taken alone this looks like support for H_hw** — but see the next part.

### Why we could NOT conclude — the instrument failed its own positive control

Both corrected harnesses **failed the mandatory calibration**: the **two-card positive
control** (two *separate* GPUs, which obviously *can* run concurrently) also measured
R≈1.9–2.1 — i.e. our harness reported "no overlap" even where overlap must exist. Causes:

- The cross-stream `torch.xpu.Event.elapsed_time` overlap metric returned **0 on this
  build** (device-timeline instrumentation unavailable — the same all-zeros issue seen in
  the POC profiler).
- Driving two streams/devices from Python is **host-launch / GIL bound**: the negative
  control (two threads onto the *same* stream) measured R≈3.8, i.e. *worse* than 2×,
  proving Python kernel-issue overhead — not device scheduling — dominated the pair timing.

**Per our own decision rule, a failed positive control ⇒ INSTRUMENT-INVALID ⇒ no verdict.**
Reporting H_hw off a blind instrument would repeat the original sin in a new form, so we
did not. Settling H_hw vs H_sat rigorously needs device-side concurrency instrumentation
(a working XPU profiler timeline, or an on-device graph capture that removes the host from
the loop) that this stack did not give us — a measurement-engineering project beyond the
POC's scope.

---

## Practical conclusion (correctly scoped)

- **For the DreamZero text+VAE pair on this B60 + torch-xpu 2.12 stack:** one-card
  two-stream parallelism yields **no measurable speedup** under the tested settings. Don't
  ship it expecting a win. *(measured, reproducible)*
- **We do NOT claim** two `torch.xpu` streams *can never* overlap on this hardware. We
  could not prove that; our attempts to were instrument-limited. *(explicitly unproven)*
- **Do this instead:** compile the VAE with `torch.compile(inductor)` — a verified ~2.0–2.2×
  single-card speedup on the encode-to-DiT latency (`../test data/5_encoder_poc_results/`),
  optionally combined with the **two-card** split (text on card 0, VAE on card 1), which
  *does* overlap because they are genuinely separate devices.

---

## Files

| file | what it is |
|---|---|
| `README.md` | this writeup (the correctly-scoped conclusion) |
| `pe_async_launch_check.py` | **decisive POC-code check**: times the *real* VAE encode's enqueue-return vs device-finish. Shows the launch is async (ratio 0.08) ⇒ the zero-overlap is **not** a POC host-sync bug |
| `demonstrate_one_card_serialization.py` | the **original FLAWED** probe, kept as a documented example of the saturation confound — not proof |
| `onecard_concurrency_matrix.py` | corrected experiment matrix (occupancy sweep E1 is valid; E0 calibration failed on this build) |
| `onecard_calib_threaded.py` | threaded two-card-control attempt; documents the host/GIL measurement limit |
| `evidence/matrix_default_env.json` | raw output: occupancy sweep + E2/E3 (default env) |
| `evidence/calib_threaded_default.json` | raw output: failed positive control (R_A≈1.9) — why we stopped |
| `evidence/probe_default_env.json` | original POC probe (default env) + inductor-viability probe A |
| `evidence/probe_no_immediate_cmdlist.json` | original POC probe with immediate-command-lists off |
| `evidence/async_launch_check.json` | raw output of the async-launch check: VAE launch 182 ms vs full 2174 ms (ratio 0.08) — proves the enqueue is async, not host-blocking |

### Lesson

A "no speedup" measurement is only evidence about the *workload*. Turning it into a claim
about the *hardware* requires (a) proving the workload wasn't saturating **and** (b) a
measurement instrument you've calibrated against a known-overlap positive control. We had
neither for the strong claim, so the strong claim is withdrawn.
