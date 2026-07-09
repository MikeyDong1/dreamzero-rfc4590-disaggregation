#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Standalone parallel-encoder benchmark for DreamZero on Intel XPU (Feature A POC).

Runs DreamZero's **UMT5 text encoder** and **Wan VAE encoder** either serially or
concurrently and measures whether stream/device parallelism hides the small text
encode behind the long VAE encode. It loads ONLY the two encoders (skips the ~28 GB
CausalWanModel DiT), so it is light on memory and fast to start.

The heavy DreamZero serving path this mirrors lives in
``vllm_omni/diffusion/models/dreamzero/pipeline_dreamzero.py``:
  * text  : ``DreamZeroPipeline._encode_text``        (:650)
  * VAE   : ``DreamZeroPipeline._encode_vae_latents``  (:847) inside the
            ``autocast(bf16)`` of ``_encode_image``     (:667)

Modes (``--parallel-encoder-mode``):
  * ``serial``          : text then VAE, one device, one stream (baseline).
  * ``one_card_stream`` : text ‖ VAE on two streams of one device.
  * ``two_card``        : text on one card, VAE on another (upper-bound).

Example
-------
    python benchmarks/dreamzero_parallel_encoders.py \
        --model-path /models/DreamZero-DROID \
        --parallel-encoder-mode all \
        --num-warmup-runs 3 --num-benchmark-runs 10

Timing rules honored: synchronize before starting the timer and after each measured
section; warm-up runs precede benchmark runs; model load time is excluded. The
measured region contains no ``.cpu()`` / ``.item()`` / ``.numpy()`` / print.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Import vllm_omni FIRST: its package init disables the broken triton-xpu import
# (triton.tools.disasm.get_spvdis) before any `vllm.config` import.
import vllm_omni  # noqa: F401,E402  (side effect: triton-xpu disable)

from vllm.utils.torch_utils import set_default_torch_dtype  # noqa: E402
from transformers import AutoTokenizer, UMT5Config, UMT5EncoderModel  # noqa: E402

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (  # noqa: E402
    DistributedAutoencoderKLWan,
)
from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline  # noqa: E402

# The parallel-encoder helper lives next to this script (self-contained POC
# bundle), not inside the installed vllm_omni package. Import it from here.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parallel_encoders as pe  # noqa: E402,F401

DEFAULT_PROMPT = (
    "Move the pan forward and use the brush in the middle of the plates "
    "to brush the inside of the pan"
)


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Model building (encoders only)
# ---------------------------------------------------------------------------
def _load_root_config(model_path: str) -> dict:
    with open(os.path.join(model_path, "config.json")) as f:
        return json.load(f)


def build_text_encoder(dtype: torch.dtype) -> UMT5EncoderModel:
    """Build the UMT5-xxl encoder exactly as ``DreamZeroPipeline.__init__`` does."""
    umt5_config = UMT5Config(
        d_model=4096,
        d_ff=10240,
        num_heads=64,
        num_layers=24,
        vocab_size=256384,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        dense_act_fn="gelu_new",
        feed_forward_proj="gated-gelu",
        is_encoder_decoder=False,
    )
    # Serving builds the pipeline under set_default_torch_dtype(od_config.dtype),
    # so UMT5 params are bf16. Match that here for numerical parity.
    with set_default_torch_dtype(dtype):
        text_encoder = UMT5EncoderModel(umt5_config)
    return text_encoder.eval()


def build_vae(model_path: str) -> DistributedAutoencoderKLWan:
    """Build the Wan VAE (fp32) as the pipeline does when no explicit vae/ source.

    We intentionally do NOT call ``init_distributed()``: the distributed tiling
    executor only activates with tiling AND parallel_size>1, which never happens
    on this single-card encode path — ``_encode`` runs plain/replicated, identical
    to each rank in the real run (see dreamzero_xpu_run/vae_only_bench.py)."""
    vae_dir = os.path.join(model_path, "vae")
    if os.path.isdir(vae_dir):
        vae = DistributedAutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
    else:
        vae = DistributedAutoencoderKLWan()
    return vae.eval()


def _iter_shard_weights(model_path: str, prefixes: tuple[str, ...]):
    """Yield (name, tensor) for checkpoint keys under any of ``prefixes``."""
    from safetensors import safe_open

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    shard_to_keys: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        if key.startswith(prefixes):
            shard_to_keys.setdefault(shard, []).append(key)
    for shard, keys in shard_to_keys.items():
        with safe_open(os.path.join(model_path, shard), framework="pt", device="cpu") as f:
            for name in keys:
                yield name, f.get_tensor(name)


def load_encoder_weights(
    text_encoder: UMT5EncoderModel,
    vae: DistributedAutoencoderKLWan,
    model_path: str,
) -> tuple[int, int]:
    """Fill the two encoders from the root checkpoint using the pipeline's remaps.

    Uses ``DreamZeroPipeline._remap_text_encoder_key`` / ``._remap_vae_key`` so the
    loaded weights are byte-identical to what serving loads."""
    text_params = dict(text_encoder.named_parameters())
    vae_params = dict(vae.named_parameters())
    n_text = n_vae = 0
    for name, tensor in _iter_shard_weights(
        model_path, ("action_head.text_encoder.", "action_head.vae.")
    ):
        if name.startswith("action_head.text_encoder."):
            mapped = DreamZeroPipeline._remap_text_encoder_key(name)
            if mapped is None:
                continue
            for new_name in mapped if isinstance(mapped, list) else [mapped]:
                if new_name in text_params:
                    text_params[new_name].data.copy_(tensor.to(text_params[new_name].dtype))
                    n_text += 1
        elif name.startswith("action_head.vae."):
            mapped = DreamZeroPipeline._remap_vae_key(name)
            if mapped is None:
                continue
            if mapped in vae_params:
                vae_params[mapped].data.copy_(tensor.to(vae_params[mapped].dtype))
                n_vae += 1
    return n_text, n_vae


def make_vae_buffers(vae: DistributedAutoencoderKLWan) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the pipeline's ``vae_latents_mean`` / ``vae_latents_inv_std`` (fp32)."""
    mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1)
    inv_std = (1.0 / torch.tensor(vae.config.latents_std, dtype=torch.float32)).view(1, -1, 1, 1, 1)
    return mean, inv_std


# ---------------------------------------------------------------------------
# Input building (mirrors _encode_image obs#1)
# ---------------------------------------------------------------------------
def _preprocess_video(videos: torch.Tensor) -> torch.Tensor:
    """uint8 [B,T,H,W,C] -> bf16 [B,C,T,H,W] in [-1,1] (copy of pipeline:625)."""
    videos = videos.permute(0, 4, 1, 2, 3)
    if videos.dtype == torch.uint8:
        videos = videos.float() / 255.0
        videos = videos.to(dtype=torch.bfloat16)
        b, c, t, h, w = videos.shape
        videos = videos.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        videos = videos * 2.0 - 1.0
        videos = videos.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    return videos.to(dtype=torch.bfloat16)


def build_inputs(
    model_path: str,
    prompt: str,
    tokenizer_source: str,
    height: int,
    width: int,
    stitched_npz: str | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Return (text_tokens, attention_mask, vae_input, num_frames) on CPU.

    Text: tokenize ``prompt`` with the UMT5 tokenizer (falls back to synthetic
    token ids if the tokenizer can't be loaded offline). VAE: build the obs#1 I2V
    conditioning window ``(1,3,num_frames,H,W)`` = first frame + zero frames."""
    root_cfg = _load_root_config(model_path)
    num_frames = root_cfg["action_head_cfg"]["config"]["num_frames"]

    # ---- Text tokens ----
    text_tokens = attention_mask = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        enc = tokenizer(
            prompt,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        text_tokens = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        log(f"[input] tokenized prompt via {tokenizer_source!r} -> tokens {tuple(text_tokens.shape)}")
    except Exception as exc:  # noqa: BLE001 - offline fallback
        log(f"[input] tokenizer unavailable ({exc}); using synthetic tokens (latency-only).")
        seq = 512
        real_len = 24
        text_tokens = torch.zeros(1, seq, dtype=torch.long)
        text_tokens[:, :real_len] = torch.randint(1, 256000, (1, real_len))
        attention_mask = torch.zeros(1, seq, dtype=torch.long)
        attention_mask[:, :real_len] = 1

    # ---- VAE input (obs#1 window) ----
    if stitched_npz:
        z = np.load(stitched_npz)
        stitched = z["images"]
        if stitched.ndim == 3:
            stitched = stitched[None]  # (1,H,W,C)
        videos = torch.from_numpy(stitched).unsqueeze(0)  # (B=1,T=1,H,W,C)
        videos = _preprocess_video(videos)  # (1,3,1,H,W) bf16
        height, width = videos.shape[-2], videos.shape[-1]
        image_input = videos[:, :, :1]  # (1,3,1,H,W)
        log(f"[input] loaded real stitched frame from {stitched_npz} -> {tuple(image_input.shape)}")
    else:
        # Synthetic first frame at the real DreamZero DROID resolution.
        image_input = (torch.rand(1, 3, 1, height, width) * 2.0 - 1.0).to(torch.bfloat16)
        log(f"[input] synthetic first frame {tuple(image_input.shape)} ({height}x{width})")

    image_zeros = torch.zeros(1, 3, num_frames - 1, height, width, dtype=image_input.dtype)
    vae_input = torch.concat([image_input, image_zeros], dim=2)  # (1,3,num_frames,H,W)
    log(f"[input] vae_input {tuple(vae_input.shape)} dtype={vae_input.dtype}")
    return text_tokens, attention_mask, vae_input, num_frames


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------
def measure(fn, device: torch.device, warmup: int, runs: int) -> dict:
    """Warm up ``warmup`` times, then time ``runs`` calls of ``fn``.

    Synchronizes before starting the timer and after the call. Resets and reads
    peak memory around the timed runs. Returns mean/min/max ms + peak_mem_gb."""
    for _ in range(warmup):
        fn()
    pe.sync_device(device)
    pe.reset_peak_memory(device)

    times_ms: list[float] = []
    for _ in range(runs):
        pe.sync_device(device)
        t0 = time.perf_counter()
        fn()
        pe.sync_device(device)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    peak = pe.peak_memory_gb(device)
    return {
        "mean_ms": sum(times_ms) / len(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "runs_ms": times_ms,
        "peak_mem_gb": peak,
    }


def _pipeline_text_reference(text_tokens, attention_mask, text_encoder) -> torch.Tensor:
    """EXACT copy of ``DreamZeroPipeline._encode_text`` (device-scalar padding slice).

    Used only by ``--verify-against-pipeline`` to prove the sync-free mask in
    ``pe.run_text_encoder`` matches the production math for B=1."""
    seq_lens = attention_mask.gt(0).sum(dim=1).long()
    prompt_emb = text_encoder(text_tokens, attention_mask).last_hidden_state
    prompt_emb = prompt_emb.clone().to(dtype=torch.bfloat16)
    for i, v in enumerate(seq_lens):
        prompt_emb[:, v:] = 0
    return prompt_emb


# ---------------------------------------------------------------------------
# Correctness helpers
# ---------------------------------------------------------------------------
def compare_tensors(name: str, a: torch.Tensor, b: torch.Tensor, rtol: float, atol: float) -> dict:
    a_c = a.detach().float().cpu()
    b_c = b.detach().float().cpu()
    max_abs = (a_c - b_c).abs().max().item()
    denom = b_c.abs().clamp_min(1e-6)
    max_rel = ((a_c - b_c).abs() / denom).max().item()
    try:
        torch.testing.assert_close(a_c, b_c, rtol=rtol, atol=atol)
        ok = True
    except AssertionError:
        ok = False
    log(f"[correctness] {name}: max_abs_diff={max_abs:.3e} max_rel_diff={max_rel:.3e} within_tol={ok}")
    return {"name": name, "max_abs_diff": max_abs, "max_rel_diff": max_rel, "within_tol": ok}


# ---------------------------------------------------------------------------
# Per-stream diagnostic timing (one_card, event-based)
# ---------------------------------------------------------------------------
def instrumented_one_card(inputs: pe.EncoderInputs, models: pe.EncoderModels, device: torch.device,
                          vae_autocast: bool) -> dict:
    """One measured parallel encode with per-stream device events (best-effort)."""
    accel = pe._accel(device)
    if accel is None or not hasattr(accel, "Event"):
        return {}
    try:
        vae_stream = pe.make_stream(device)
        text_stream = pe.make_stream(device)
        if vae_stream is None or text_stream is None:
            return {}
        ev = lambda: accel.Event(enable_timing=True)  # noqa: E731
        v0, v1, t0, t1 = ev(), ev(), ev(), ev()
        pe.sync_device(device)
        with torch.inference_mode():
            with pe.stream_context(device, vae_stream):
                v0.record(vae_stream)
                pe.run_vae_encoder(inputs.vae_input, models.vae, models.vae_latents_mean,
                                   models.vae_latents_inv_std, autocast=vae_autocast)
                v1.record(vae_stream)
            with pe.stream_context(device, text_stream):
                t0.record(text_stream)
                pe.run_text_encoder(inputs.text_tokens, inputs.attention_mask, models.text_encoder)
                t1.record(text_stream)
            pe.sync_device(device)
        return {
            "vae_stream_ms": v0.elapsed_time(v1),
            "text_stream_ms": t0.elapsed_time(t1),
        }
    except Exception as exc:  # noqa: BLE001 - events optional on some xpu builds
        log(f"[diag] per-stream event timing unavailable: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_device(s: str) -> torch.device:
    return torch.device(s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True, help="DreamZero-DROID checkpoint dir")
    ap.add_argument("--enable-parallel-encoders", action="store_true",
                    help="Convenience flag: ensure the serial baseline AND one_card_stream mode are both run "
                         "(so a speedup can be reported). Ignored if --parallel-encoder-mode names an explicit mode.")
    ap.add_argument("--parallel-encoder-mode", default="all",
                    choices=["serial", "one_card_stream", "two_card", "all"],
                    help="Which mode(s) to benchmark. 'all' runs serial + one_card_stream (+ two_card if 2 devices). "
                         "An explicit choice takes precedence over --enable-parallel-encoders.")
    ap.add_argument("--encoder-device", default="xpu:0", help="Device for serial / one_card_stream.")
    ap.add_argument("--text-encoder-device", default=None, help="two_card: device for the text encoder (default = encoder-device).")
    ap.add_argument("--vae-encoder-device", default=None, help="two_card: device for the VAE (default = xpu:1).")
    ap.add_argument("--dit-device", default=None, help="two_card: gather device for the DiT boundary (default = encoder-device).")
    ap.add_argument("--num-warmup-runs", type=int, default=3)
    ap.add_argument("--num-benchmark-runs", type=int, default=10)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--tokenizer", default=None, help="Tokenizer source (default: <model-path> then google/umt5-xxl).")
    ap.add_argument("--height", type=int, default=352, help="Frame height for synthetic input (DROID stitched = 352).")
    ap.add_argument("--width", type=int, default=640, help="Frame width for synthetic input (DROID stitched = 640).")
    ap.add_argument("--stitched-npz", default=None, help="Optional real stitched first-frame npz (images:(1,H,W,C) uint8).")
    ap.add_argument("--no-vae-autocast", action="store_true", help="Disable the bf16 autocast around the VAE encode.")
    ap.add_argument("--compile-vae", action="store_true",
                    help="Compile the VAE encode with torch.compile(inductor) instead of running it eager "
                         "(enforce_eager). Fuses the VAE's memory-bound elementwise tail (~2x faster VAE on XPU). "
                         "The one-time compile cost is absorbed in warm-up and excluded from timing.")
    ap.add_argument("--rtol", type=float, default=2e-2, help="Relative tolerance for the correctness check (bf16).")
    ap.add_argument("--atol", type=float, default=2e-2, help="Absolute tolerance for the correctness check (bf16).")
    ap.add_argument("--verify-against-pipeline", action="store_true",
                    help="Additionally verify the helper's sync-free text math matches the pipeline's exact _encode_text.")
    ap.add_argument("--out", default=None, help="Optional path to write the results JSON.")
    args = ap.parse_args()

    vae_autocast = not args.no_vae_autocast
    enc_device = parse_device(args.encoder_device)
    accel = pe._accel(enc_device)
    pe.set_accel_device(enc_device)

    device_count = 0
    if accel is not None and hasattr(accel, "device_count"):
        with contextlib.suppress(Exception):
            device_count = accel.device_count()

    # ---- Resolve which modes to run ----
    # An explicit --parallel-encoder-mode takes precedence; "all" (the default)
    # runs serial + one_card_stream (+ two_card when 2 devices are visible).
    # --enable-parallel-encoders is only a convenience alias for the default:
    # run the serial baseline AND one_card_stream so a speedup can be reported.
    if args.parallel_encoder_mode == "all":
        modes = [pe.MODE_SERIAL, pe.MODE_ONE_CARD]
        if device_count >= 2:
            modes.append(pe.MODE_TWO_CARD)
        else:
            log(f"[setup] only {device_count} device(s) visible; skipping two_card mode.")
    else:
        modes = [args.parallel_encoder_mode]
        if args.enable_parallel_encoders:
            # Ensure both baseline and one_card are present for a meaningful speedup.
            for m in (pe.MODE_SERIAL, pe.MODE_ONE_CARD):
                if m not in modes:
                    modes.append(m)

    two_card = pe.MODE_TWO_CARD in modes
    text_dev = parse_device(args.text_encoder_device) if args.text_encoder_device else enc_device
    vae_dev = parse_device(args.vae_encoder_device) if args.vae_encoder_device else (
        torch.device("xpu:1") if two_card else enc_device
    )
    dit_dev = parse_device(args.dit_device) if args.dit_device else enc_device

    log(f"[setup] modes={modes} enc_device={enc_device} device_count={device_count} "
        f"vae_autocast={vae_autocast}")

    # ---- Build + load the two encoders (NOT timed) ----
    t_load0 = time.perf_counter()
    log("[load] building text encoder (UMT5-xxl bf16) + Wan VAE (fp32) ...")
    text_encoder = build_text_encoder(torch.bfloat16)
    vae = build_vae(args.model_path)
    n_text, n_vae = load_encoder_weights(text_encoder, vae, args.model_path)
    log(f"[load] loaded text params={n_text} vae params={n_vae}")
    vae_mean, vae_inv_std = make_vae_buffers(vae)
    model_load_s = time.perf_counter() - t_load0
    log(f"[load] MODEL_LOAD_S={model_load_s:.3f} (excluded from timing)")

    tokenizer_source = args.tokenizer or (
        args.model_path if os.path.isdir(os.path.join(args.model_path, "tokenizer")) else "google/umt5-xxl"
    )
    text_tokens, attention_mask, vae_input, num_frames = build_inputs(
        args.model_path, args.prompt, tokenizer_source, args.height, args.width, args.stitched_npz,
        enc_device,
    )

    results: dict = {
        "model_load_s": model_load_s,
        "n_text_params": n_text,
        "n_vae_params": n_vae,
        "num_frames": num_frames,
        "vae_input_shape": list(vae_input.shape),
        "text_tokens_shape": list(text_tokens.shape),
        "config": {
            "warmup": args.num_warmup_runs,
            "runs": args.num_benchmark_runs,
            "vae_autocast": vae_autocast,
            "compile_vae": args.compile_vae,
            "rtol": args.rtol,
            "atol": args.atol,
            "device_count": device_count,
        },
        "modes": {},
    }

    # =====================================================================
    # Helper to place models + inputs on a device and build the containers.
    # =====================================================================
    def _maybe_compile_vae(device: torch.device, vmean: torch.Tensor, vinv: torch.Tensor,
                           inp: torch.Tensor):
        """Build + warm up (compile) the inductor VAE encode for ``device``.

        Compilation happens HERE (before any measured region), so the first-call
        graph-build cost never lands in a timed run. Returns the compiled callable
        or ``None`` when --compile-vae is off."""
        if not args.compile_vae:
            return None
        log(f"[compile] building inductor VAE encode on {device} (first call compiles; excluded from timing)...")
        tC = time.perf_counter()
        fn = pe.compile_vae_encode(vae, vmean, vinv, autocast=vae_autocast, device_type=device.type)
        with torch.inference_mode():
            fn(inp)                 # triggers compilation
            pe.sync_device(device)
            fn(inp)                 # second call = steady state
            pe.sync_device(device)
        log(f"[compile] VAE inductor compile done in {time.perf_counter()-tC:.1f}s on {device}")
        return fn

    def place(device: torch.device) -> tuple[pe.EncoderModels, pe.EncoderInputs]:
        text_encoder.to(device)
        vae.to(device=device, dtype=torch.float32)
        vmean = vae_mean.to(device)
        vinv = vae_inv_std.to(device)
        vin = vae_input.to(device)
        models = pe.EncoderModels(
            text_encoder=text_encoder,
            vae=vae,
            vae_latents_mean=vmean,
            vae_latents_inv_std=vinv,
            vae_compiled_encode=_maybe_compile_vae(device, vmean, vinv, vin),
        )
        inputs = pe.EncoderInputs(
            text_tokens=text_tokens.to(device),
            attention_mask=attention_mask.to(device),
            vae_input=vin,
        )
        return models, inputs

    # =====================================================================
    # Isolated component timings on the encoder device (for the table + overlap).
    # =====================================================================
    log("\n[bench] === isolated component timings (encoder device) ===")
    models, inputs = place(enc_device)
    mem_after_load = pe.memory_allocated_gb(enc_device)
    log(f"[mem] model memory after load on {enc_device}: {mem_after_load:.3f} GiB")
    results["model_memory_after_load_gb"] = mem_after_load

    def _text_only():
        with torch.inference_mode():
            pe.run_text_encoder(inputs.text_tokens, inputs.attention_mask, models.text_encoder)

    def _vae_only():
        with torch.inference_mode():
            pe.run_vae_encoder(inputs.vae_input, models.vae, models.vae_latents_mean,
                               models.vae_latents_inv_std, autocast=vae_autocast,
                               compiled_encode=models.vae_compiled_encode)

    text_iso = measure(_text_only, enc_device, args.num_warmup_runs, args.num_benchmark_runs)
    vae_iso = measure(_vae_only, enc_device, args.num_warmup_runs, args.num_benchmark_runs)
    log(f"[bench] isolated text_ms={text_iso['mean_ms']:.1f}  vae_ms={vae_iso['mean_ms']:.1f}")
    results["isolated"] = {"text_ms": text_iso["mean_ms"], "vae_ms": vae_iso["mean_ms"]}

    # Reference serial outputs (for correctness comparison across modes).
    serial_ref = pe.encode_serial(inputs, models, enc_device, vae_autocast=vae_autocast)
    pe.sync_device(enc_device)

    # Optional: prove the sync-free text mask matches the pipeline's exact math.
    if args.verify_against_pipeline:
        with torch.inference_mode():
            pipe_text = _pipeline_text_reference(inputs.text_tokens, inputs.attention_mask, models.text_encoder)
        results["verify_against_pipeline"] = compare_tensors(
            "text (helper vs pipeline._encode_text)", serial_ref.text_embedding, pipe_text,
            rtol=0.0, atol=0.0,
        )

    # =====================================================================
    # Per-mode benchmark
    # =====================================================================
    for mode in modes:
        log(f"\n[bench] === mode={mode} ===")
        mode_res: dict = {}

        if mode == pe.MODE_SERIAL:
            models, inputs = place(enc_device)
            fn = lambda: pe.encode_serial(inputs, models, enc_device, vae_autocast=vae_autocast)  # noqa: E731
            m = measure(fn, enc_device, args.num_warmup_runs, args.num_benchmark_runs)
            mode_res.update(total_ms=m["mean_ms"], text_ms=text_iso["mean_ms"], vae_ms=vae_iso["mean_ms"],
                            overlap_ms=0.0, peak_mem_gb=m["peak_mem_gb"], detail=m)

        elif mode == pe.MODE_ONE_CARD:
            models, inputs = place(enc_device)
            # Create the two streams once and reuse them across all timed calls,
            # exactly as a serving EncoderStage would (avoids per-call stream churn).
            reuse_streams = (pe.make_stream(enc_device), pe.make_stream(enc_device))
            fn = lambda: pe.encode_parallel_one_card(  # noqa: E731
                inputs, models, enc_device, vae_autocast=vae_autocast, streams=reuse_streams)
            m = measure(fn, enc_device, args.num_warmup_runs, args.num_benchmark_runs)
            diag = instrumented_one_card(inputs, models, enc_device, vae_autocast)
            overlap = text_iso["mean_ms"] + vae_iso["mean_ms"] - m["mean_ms"]
            # Prefer per-stream device-event timings when available and finite;
            # otherwise fall back to the isolated single-stream timings and flag it.
            diag_text = diag.get("text_stream_ms")
            diag_vae = diag.get("vae_stream_ms")
            stream_timing_ok = (
                diag_text is not None and diag_vae is not None
                and math.isfinite(diag_text) and math.isfinite(diag_vae)
            )
            mode_res.update(
                total_ms=m["mean_ms"],
                text_ms=diag_text if stream_timing_ok else text_iso["mean_ms"],
                vae_ms=diag_vae if stream_timing_ok else vae_iso["mean_ms"],
                overlap_ms=overlap,
                peak_mem_gb=m["peak_mem_gb"],
                detail=m,
                stream_diag=diag,
                stream_timing_available=stream_timing_ok,
            )
            if not stream_timing_ok:
                log("[bench] per-stream event timing unavailable/invalid; "
                    "text_ms/vae_ms columns show isolated single-stream timings.")
            # correctness vs serial
            out = pe.encode_parallel_one_card(
                inputs, models, enc_device, vae_autocast=vae_autocast, streams=reuse_streams)
            pe.sync_device(enc_device)
            mode_res["correctness"] = [
                compare_tensors("text", out.text_embedding, serial_ref.text_embedding, args.rtol, args.atol),
                compare_tensors("vae", out.vae_latent, serial_ref.vae_latent, args.rtol, args.atol),
            ]

        elif mode == pe.MODE_TWO_CARD:
            # Place text + VAE on separate cards.
            text_encoder.to(text_dev)
            vae.to(device=vae_dev, dtype=torch.float32)
            vmean2 = vae_mean.to(vae_dev)
            vinv2 = vae_inv_std.to(vae_dev)
            vin2 = vae_input.to(vae_dev)
            models2 = pe.EncoderModels(
                text_encoder=text_encoder, vae=vae,
                vae_latents_mean=vmean2, vae_latents_inv_std=vinv2,
                # Compile the VAE on its own card (vae_dev) before timing.
                vae_compiled_encode=_maybe_compile_vae(vae_dev, vmean2, vinv2, vin2),
            )
            inputs2 = pe.EncoderInputs(
                text_tokens=text_tokens.to(text_dev), attention_mask=attention_mask.to(text_dev),
                vae_input=vin2,
            )

            def _two():
                pe.encode_parallel_two_card(inputs2, models2, text_dev, vae_dev,
                                            gather_device=dit_dev, vae_autocast=vae_autocast, time_transfer=False)

            for _ in range(args.num_warmup_runs):
                _two()
            pe.sync_device(text_dev)
            pe.sync_device(vae_dev)
            pe.reset_peak_memory(text_dev)
            pe.reset_peak_memory(vae_dev)
            times_ms, d2d_ms_list = [], []
            for _ in range(args.num_benchmark_runs):
                pe.sync_device(text_dev)
                pe.sync_device(vae_dev)
                t0 = time.perf_counter()
                out2 = pe.encode_parallel_two_card(inputs2, models2, text_dev, vae_dev,
                                                    gather_device=dit_dev, vae_autocast=vae_autocast,
                                                    time_transfer=True)
                times_ms.append((time.perf_counter() - t0) * 1000.0)
                if out2.d2d_transfer_ms is not None:
                    d2d_ms_list.append(out2.d2d_transfer_ms)
            total = sum(times_ms) / len(times_ms)
            overlap = text_iso["mean_ms"] + vae_iso["mean_ms"] - total
            # Keep out2 for the correctness compare below, but read peak first on a
            # scratch copy so we can drop the timed-loop's last output. (The serial /
            # one_card modes free their outputs every iteration inside measure(), so
            # this keeps the peak comparison apples-to-apples.)
            pe.sync_device(text_dev)
            pe.sync_device(vae_dev)
            peak = max(pe.peak_memory_gb(text_dev), pe.peak_memory_gb(vae_dev))
            mode_res.update(
                total_ms=total, text_ms=text_iso["mean_ms"], vae_ms=vae_iso["mean_ms"],
                overlap_ms=overlap, peak_mem_gb=peak,
                d2d_transfer_ms=(sum(d2d_ms_list) / len(d2d_ms_list)) if d2d_ms_list else None,
                text_device=str(text_dev), vae_device=str(vae_dev), dit_device=str(dit_dev),
                detail={"runs_ms": times_ms},
            )
            # correctness vs serial (move both back to a common device for comparison)
            mode_res["correctness"] = [
                compare_tensors("text", out2.text_embedding.to("cpu"), serial_ref.text_embedding.to("cpu"),
                                args.rtol, args.atol),
                compare_tensors("vae", out2.vae_latent.to("cpu"), serial_ref.vae_latent.to("cpu"),
                                args.rtol, args.atol),
            ]

        results["modes"][mode] = mode_res

    # =====================================================================
    # Report
    # =====================================================================
    serial_total = results["modes"].get(pe.MODE_SERIAL, {}).get("total_ms")
    log("\n=================== RESULTS ===================")
    log(f"{'mode':<20}{'total_ms':>12}{'text_ms':>12}{'vae_ms':>12}{'overlap_ms':>13}{'peak_mem_gb':>13}{'speedup':>10}")
    for mode in modes:
        r = results["modes"][mode]
        # Speedup is only meaningful relative to the serial baseline; show '--'
        # when serial was not run (e.g. an explicit --parallel-encoder-mode).
        if serial_total and r.get("total_ms"):
            speedup_str = f"{serial_total / r['total_ms']:>10.3f}"
        else:
            speedup_str = f"{'--':>10}"
        log(f"{mode:<20}{r['total_ms']:>12.1f}{r['text_ms']:>12.1f}{r['vae_ms']:>12.1f}"
            f"{r['overlap_ms']:>13.1f}{r['peak_mem_gb']:>13.3f}{speedup_str}")
        if r.get("d2d_transfer_ms") is not None:
            log(f"{'':>20}d2d_transfer_ms={r['d2d_transfer_ms']:.3f}")
    if serial_total is None:
        log("(speedup omitted: serial baseline not run — pass --parallel-encoder-mode all or --enable-parallel-encoders)")
    log("===============================================")

    # Correctness summary
    all_ok = True
    for mode in modes:
        for c in results["modes"][mode].get("correctness", []):
            all_ok = all_ok and c["within_tol"]
    if "verify_against_pipeline" in results:
        all_ok = all_ok and results["verify_against_pipeline"]["within_tol"]
    results["all_correct"] = all_ok
    log(f"ALL_CORRECT={all_ok}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        log(f"SAVED={args.out}")
    log("DONE")


if __name__ == "__main__":
    main()
