#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Async-launch check — is the POC's one-card serialization a CODE bug or not?

THE QUESTION
------------
The POC's `one_card_stream` mode enqueues the VAE encode on stream 1 and the text
encode on stream 2, then does ONE join at the end. For the two streams to have any
chance of overlapping, the CPU must be able to enqueue the VAE work and *return
immediately* (async), so it can go on to enqueue the text work while the VAE runs.

If instead `run_vae_encoder(...)` BLOCKS THE HOST until the VAE finishes (a hidden
host<->device sync somewhere inside `vae._encode` — a `.item()`, a shape/`.cpu()`
check, feat-cache bookkeeping, a python-scalar branch, etc.), then the CPU never
reaches the text-enqueue line until the VAE is already done. That would make the
one-card design SERIALIZE *by construction of the POC/model code* — a code-level
cause, entirely distinct from any hardware stream-scheduling limit.

THE TEST (decisive, needs no profiler)
--------------------------------------
Time the VAE encode TWICE against the same t0:
  * launch_ms : wall until `run_vae_encoder(...)` RETURNS      (NO sync after)
  * full_ms   : wall until the device sync completes           (real compute time)

  launch_ms << full_ms  (launch a few ms, full ~1-2 s)
        => the enqueue is ASYNC. The host is free during the VAE compute, so the
           text stream CAN be enqueued concurrently. The zero-overlap is NOT a
           POC host-sync bug. (Points to device scheduling / saturation instead.)

  launch_ms ~= full_ms  (both ~1-2 s)
        => there IS a hidden host sync inside the VAE encode. The CPU is blocked
           for the whole VAE, so text can't be enqueued until VAE is done. The
           one-card serialization is then a CODE/MODEL issue, and the POC's
           one-card design cannot work until that sync is removed.

We run this for BOTH encoders (VAE is the one under suspicion; text is a control),
in eager and (optionally) inductor, and also time the exact POC one-card enqueue
sequence to confirm the launch-level behavior matches the benchmark.

This imports the benchmark's REAL build/load/encode code, so it exercises the
actual Wan VAE and the actual `parallel_encoders.run_vae_encoder`, not a stand-in.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Reuse the benchmark's model build/load + the POC encode helpers verbatim.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dreamzero_parallel_encoders as bench  # noqa: E402  (also triggers vllm_omni init)
import parallel_encoders as pe  # noqa: E402
import torch  # noqa: E402


def log(m):
    print(m, flush=True)


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def launch_vs_full(call, device, warmup, reps, label):
    """Measure (median launch_ms, median full_ms) for a no-sync-inside `call`.

    `call()` must enqueue device work and return a tensor WITHOUT synchronizing.
    launch_ms = wall until call() returns; full_ms = wall until device idle.

    NOTE: the ENTIRE routine (warm-up included) runs under torch.inference_mode()
    — otherwise the VAE forward builds an autograd graph and retains every 3D-conv
    activation for backward, which exhausts device memory (UR_OUT_OF_RESOURCES).
    """
    launches, fulls = [], []
    with torch.inference_mode():
        for _ in range(warmup):
            call()
            pe.sync_device(device)
        for _ in range(reps):
            pe.sync_device(device)
            t0 = time.perf_counter()
            out = call()                       # enqueue only (no sync inside)
            t_launch = time.perf_counter()
            pe.sync_device(device)             # wait for the device to finish
            t_full = time.perf_counter()
            launches.append((t_launch - t0) * 1000.0)
            fulls.append((t_full - t0) * 1000.0)
            del out
    lm, fm = _median(launches), _median(fulls)
    ratio = lm / fm if fm else float("nan")
    verdict = ("ASYNC (host free during compute -> NOT a POC host-sync bug)"
               if ratio < 0.25 else
               "HOST-BLOCKING (hidden sync in encode -> one-card serializes by CODE)")
    log(f"[{label}] launch={lm:8.1f} ms   full={fm:8.1f} ms   launch/full={ratio:5.2f}  => {verdict}")
    return {"label": label, "launch_ms": lm, "full_ms": fm, "launch_over_full": ratio,
            "verdict": verdict, "launches_ms": launches, "fulls_ms": fulls}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--stitched-npz", default=None)
    ap.add_argument("--prompt", default=bench.DEFAULT_PROMPT)
    ap.add_argument("--device", default="xpu:0")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--compile-vae", action="store_true",
                    help="Also test the inductor-compiled VAE encode (in addition to eager).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    pe.set_accel_device(device)
    log(f"[env] device={device} torch={torch.__version__}")

    # ---- Build + load the two encoders exactly as the benchmark does ----
    log("[load] building UMT5 text encoder + Wan VAE ...")
    text_encoder = bench.build_text_encoder(torch.bfloat16)
    vae = bench.build_vae(args.model_path)
    n_text, n_vae = bench.load_encoder_weights(text_encoder, vae, args.model_path)
    vae_mean, vae_inv_std = bench.make_vae_buffers(vae)
    log(f"[load] text params={n_text} vae params={n_vae}")

    tokenizer_source = (args.model_path
                        if os.path.isdir(os.path.join(args.model_path, "tokenizer"))
                        else "google/umt5-xxl")
    text_tokens, attention_mask, vae_input, num_frames = bench.build_inputs(
        args.model_path, args.prompt, tokenizer_source, 352, 640, args.stitched_npz, device)

    # place on device
    text_encoder.to(device)
    vae.to(device=device, dtype=torch.float32)
    vmean = vae_mean.to(device); vinv = vae_inv_std.to(device)
    vin = vae_input.to(device)
    tok = text_tokens.to(device); am = attention_mask.to(device)

    results = {"device": str(device), "torch": torch.__version__,
               "n_text_params": n_text, "n_vae_params": n_vae,
               "vae_input_shape": list(vin.shape), "tests": {}}

    log("\n=== async-launch check: does an encode enqueue return before the device finishes? ===")

    # ---- VAE eager (the encoder under suspicion) ----
    def vae_eager():
        return pe.run_vae_encoder(vin, vae, vmean, vinv, autocast=True, compiled_encode=None)
    results["tests"]["vae_eager"] = launch_vs_full(vae_eager, device, args.warmup, args.reps, "VAE  eager  ")

    # ---- Text eager (control: tiny, should also be async) ----
    def text_eager():
        return pe.run_text_encoder(tok, am, text_encoder)
    results["tests"]["text_eager"] = launch_vs_full(text_eager, device, args.warmup, args.reps, "TEXT eager  ")

    # ---- The exact POC one-card enqueue sequence, measured at launch level ----
    # Enqueue VAE on stream 1 then text on stream 2 (as encode_parallel_one_card does),
    # but time when the CPU RETURNS from both enqueues vs when the device is idle.
    s_vae = pe.make_stream(device); s_txt = pe.make_stream(device)

    def poc_enqueue_both():
        cur = pe.current_stream(device)
        pe._stream_wait_stream(s_vae, cur)
        pe._stream_wait_stream(s_txt, cur)
        with pe.stream_context(device, s_vae):
            v = pe.run_vae_encoder(vin, vae, vmean, vinv, autocast=True, compiled_encode=None)
        with pe.stream_context(device, s_txt):
            t = pe.run_text_encoder(tok, am, text_encoder)
        pe._stream_wait_stream(cur, s_vae)
        pe._stream_wait_stream(cur, s_txt)
        return (v, t)
    results["tests"]["poc_one_card_enqueue"] = launch_vs_full(
        poc_enqueue_both, device, args.warmup, args.reps, "POC 1card enq")

    # ---- Optional: inductor-compiled VAE ----
    if args.compile_vae:
        log("[compile] building inductor VAE (excluded from timing) ...")
        tc = time.perf_counter()
        fn = pe.compile_vae_encode(vae, vmean, vinv, autocast=True, device_type=device.type)
        with torch.inference_mode():
            fn(vin); pe.sync_device(device); fn(vin); pe.sync_device(device)
        log(f"[compile] done in {time.perf_counter()-tc:.1f}s")

        def vae_inductor():
            return fn(vin)
        results["tests"]["vae_inductor"] = launch_vs_full(
            vae_inductor, device, args.warmup, args.reps, "VAE  inductor")

    # ---- Summary interpretation ----
    vae_ratio = results["tests"]["vae_eager"]["launch_over_full"]
    log("\n================= INTERPRETATION =================")
    if vae_ratio < 0.25:
        log("  VAE enqueue is ASYNC: the host returns long before the VAE finishes.")
        log("  => The POC's one-card code CAN enqueue the text stream during the VAE compute.")
        log("  => Zero-overlap is therefore NOT caused by a host-sync bug in the POC/VAE code.")
        log("     (Cause lies in device stream scheduling / saturation, not the POC code.)")
    else:
        log("  VAE enqueue BLOCKS THE HOST for ~the whole compute.")
        log("  => There is a hidden host<->device sync inside the VAE encode.")
        log("  => The one-card serialization is a CODE/MODEL issue: text cannot be")
        log("     enqueued until the VAE is already done. The POC design needs that")
        log("     sync removed before one-card streaming could ever help.")
    log("==================================================")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2, default=str))
        log(f"SAVED={args.out}")
    log("DONE")


if __name__ == "__main__":
    main()
