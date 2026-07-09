#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Profile the DreamZero parallel-encoder POC (encoders ONLY, no DiT).

Reuses the *exact* build/load/input/encode code from
``dreamzero_parallel_encoders.py`` + ``parallel_encoders.py`` so the profiled
math is byte-identical to the benchmark that produced the timing JSONs. Wraps
each mode's encode in ``torch.profiler`` with XPU activity and dumps, per mode:
  * chrome trace  (pe_trace_<mode>.json)
  * key_averages sorted by device time   (pe_keyavg_<mode>_by_device.txt)
  * key_averages sorted by host time     (pe_keyavg_<mode>_by_host.txt)
and a combined machine-readable summary (pe_profile_summary.json).

Timing rules honored: warmup before the profiled window; synchronize around
each step; model load excluded (it happens before any profiling).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity

# Import the POC modules (same dir).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parallel_encoders as pe  # noqa: E402
import dreamzero_parallel_encoders as dz  # noqa: E402


def log(m):
    print(m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--stitched-npz", required=True)
    ap.add_argument("--prompt", default=dz.DEFAULT_PROMPT)
    ap.add_argument("--modes", default="serial,one_card_stream,two_card")
    ap.add_argument("--encoder-device", default="xpu:0")
    ap.add_argument("--vae-encoder-device", default="xpu:1")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--active", type=int, default=5)
    ap.add_argument("--compile-vae", action="store_true",
                    help="Compile the VAE encode with inductor (compile happens before the profiled window).")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    enc_device = torch.device(args.encoder_device)
    vae_device = torch.device(args.vae_encoder_device)
    pe.set_accel_device(enc_device)

    xpu_activity = getattr(ProfilerActivity, "XPU", None)
    activities = [ProfilerActivity.CPU] + ([xpu_activity] if xpu_activity else [])
    if xpu_activity is not None:
        device_sort = "self_xpu_time_total"
    else:
        device_sort = "self_cpu_time_total"
    log(f"[prof] activities={[str(a) for a in activities]} device_sort={device_sort}")

    # ---- Build + load encoders (NOT profiled) ----
    t0 = time.perf_counter()
    log("[load] building UMT5 text encoder (bf16) + Wan VAE (fp32) ...")
    text_encoder = dz.build_text_encoder(torch.bfloat16)
    vae = dz.build_vae(args.model_path)
    n_text, n_vae = dz.load_encoder_weights(text_encoder, vae, args.model_path)
    vae_mean, vae_inv_std = dz.make_vae_buffers(vae)
    log(f"[load] text params={n_text} vae params={n_vae} MODEL_LOAD_S={time.perf_counter()-t0:.1f} (excluded)")

    tok_src = args.model_path if os.path.isdir(os.path.join(args.model_path, "tokenizer")) else "google/umt5-xxl"
    text_tokens, attention_mask, vae_input, num_frames = dz.build_inputs(
        args.model_path, args.prompt, tok_src, 352, 640, args.stitched_npz, enc_device)

    dev_name = torch.xpu.get_device_properties(0).name if torch.xpu.is_available() else "cpu"

    def compiled_for(device, vmean, vinv, vin):
        """Build + warm up (compile) the inductor VAE for ``device`` if requested."""
        if not args.compile_vae:
            return None
        log(f"[compile] building inductor VAE on {device} (excluded from profiled window)...")
        t0c = time.perf_counter()
        fn = pe.compile_vae_encode(vae, vmean, vinv, autocast=True, device_type=device.type)
        with torch.inference_mode():
            fn(vin); pe.sync_device(device)
            fn(vin); pe.sync_device(device)
        log(f"[compile] done in {time.perf_counter()-t0c:.1f}s on {device}")
        return fn

    def place(device):
        text_encoder.to(device)
        vae.to(device=device, dtype=torch.float32)
        vmean, vinv, vin = vae_mean.to(device), vae_inv_std.to(device), vae_input.to(device)
        models = pe.EncoderModels(text_encoder=text_encoder, vae=vae,
                                  vae_latents_mean=vmean, vae_latents_inv_std=vinv,
                                  vae_compiled_encode=compiled_for(device, vmean, vinv, vin))
        inputs = pe.EncoderInputs(text_tokens=text_tokens.to(device),
                                  attention_mask=attention_mask.to(device),
                                  vae_input=vin)
        return models, inputs

    def make_encode_fn(mode):
        if mode == pe.MODE_SERIAL:
            models, inputs = place(enc_device)
            return lambda: pe.encode_serial(inputs, models, enc_device, vae_autocast=True), (enc_device,)
        if mode == pe.MODE_ONE_CARD:
            models, inputs = place(enc_device)
            streams = (pe.make_stream(enc_device), pe.make_stream(enc_device))
            return lambda: pe.encode_parallel_one_card(inputs, models, enc_device, vae_autocast=True, streams=streams), (enc_device,)
        if mode == pe.MODE_TWO_CARD:
            text_encoder.to(enc_device)
            vae.to(device=vae_device, dtype=torch.float32)
            vmean, vinv, vin = vae_mean.to(vae_device), vae_inv_std.to(vae_device), vae_input.to(vae_device)
            models = pe.EncoderModels(text_encoder=text_encoder, vae=vae,
                                      vae_latents_mean=vmean, vae_latents_inv_std=vinv,
                                      vae_compiled_encode=compiled_for(vae_device, vmean, vinv, vin))
            inputs = pe.EncoderInputs(text_tokens=text_tokens.to(enc_device),
                                      attention_mask=attention_mask.to(enc_device),
                                      vae_input=vin)
            return (lambda: pe.encode_parallel_two_card(inputs, models, enc_device, vae_device,
                                                        gather_device=enc_device, vae_autocast=True,
                                                        time_transfer=False),
                    (enc_device, vae_device))
        raise ValueError(mode)

    def sync(devs):
        for d in devs:
            pe.sync_device(d)

    summaries = {}
    for mode in modes:
        log(f"\n[prof] ===== mode={mode} =====")
        fn, devs = make_encode_fn(mode)
        with torch.inference_mode():
            # warmup
            for _ in range(args.warmup):
                fn()
            sync(devs)
            # steady-state wall clock
            t = time.perf_counter()
            for _ in range(max(3, args.active)):
                fn()
            sync(devs)
            warm_ms = (time.perf_counter() - t) / max(3, args.active) * 1000.0
            log(f"[{mode}] steady-state wall-clock per-encode = {warm_ms:.1f} ms")

            trace_path = outdir / f"pe_trace_{mode}.json"
            with profile(activities=activities, record_shapes=True,
                         profile_memory=True, with_stack=False) as prof:
                for _ in range(args.active):
                    fn()
                    sync(devs)

        prof.export_chrome_trace(str(trace_path))
        ka = prof.key_averages(group_by_input_shape=False)
        (outdir / f"pe_keyavg_{mode}_by_device.txt").write_text(ka.table(sort_by=device_sort, row_limit=40))
        (outdir / f"pe_keyavg_{mode}_by_host.txt").write_text(ka.table(sort_by="self_cpu_time_total", row_limit=40))
        log(f"[{mode}] key_averages (top 25 by device) =====")
        log(ka.table(sort_by=device_sort, row_limit=25))

        def evt_row(e):
            dev_self = getattr(e, "self_xpu_time_total", 0.0) or 0.0
            dev_total = getattr(e, "xpu_time_total", 0.0) or 0.0
            if xpu_activity is None:
                dev_self, dev_total = e.self_cpu_time_total, e.cpu_time_total
            return {"name": e.key, "count": int(e.count),
                    "self_device_us": float(dev_self), "device_total_us": float(dev_total),
                    "self_host_us": float(e.self_cpu_time_total), "host_total_us": float(e.cpu_time_total)}

        rows = [evt_row(e) for e in ka]
        summaries[mode] = {
            "steady_state_wall_ms_per_encode": warm_ms,
            "profiled_reps": args.active,
            "total_self_device_us": sum(r["self_device_us"] for r in rows),
            "total_self_host_us": sum(r["self_host_us"] for r in rows),
            "device_time_measured": xpu_activity is not None,
            "top_by_device": sorted(rows, key=lambda r: r["self_device_us"], reverse=True)[:20],
            "top_by_host": sorted(rows, key=lambda r: r["self_host_us"], reverse=True)[:20],
            "trace": trace_path.name,
        }

    summary = {
        "device_name": dev_name,
        "torch_version": torch.__version__,
        "vae_input_shape": list(vae_input.shape),
        "text_tokens_shape": list(text_tokens.shape),
        "num_frames": num_frames,
        "modes": summaries,
    }
    (outdir / "pe_profile_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"\nSAVED={outdir/'pe_profile_summary.json'}")
    log("DONE")


if __name__ == "__main__":
    main()
