# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Parallel encoder execution for DreamZero (Feature A proof-of-concept).

Standalone, reusable helper that runs DreamZero's **text encoder** (UMT5-xxl)
and **VAE encoder** (Wan VAE) either serially or **concurrently**, so the small
text encode (~0.2-0.4 s) can be hidden behind the long VAE encode (~1.0-1.8 s).

This module mirrors the exact encode math of
``vllm_omni/diffusion/models/dreamzero/pipeline_dreamzero.py``:

* ``run_text_encoder``  == ``DreamZeroPipeline._encode_text``   (:650)
* ``run_vae_encoder``   == ``DreamZeroPipeline._encode_vae_latents`` (:847)
  (with the ``autocast(bf16)`` that ``_encode_image`` (:667) applies for obs#1)

The only intentional deviation from the pipeline is in the text-padding zero-out:
the pipeline does ``for i, v in enumerate(seq_lens): prompt_emb[:, v:] = 0`` which
indexes with a **device scalar** ``v`` and therefore forces a host<->device sync
mid-encode. For a single stream that is harmless, but it would *serialize* the two
streams in :func:`encode_parallel_one_card` (the CPU blocks on the text kernels
before it can enqueue the VAE kernels). We instead zero the padding with a
sync-free boolean mask, which is **bit-identical for batch size 1** (DreamZero is
always B=1) and mathematically identical in general.

Design rules honored (see ``docs/feature_parallelization_study.md`` Feature A):
  * inference-only: everything runs under ``torch.inference_mode``.
  * no hidden sync inside the timed/streamed region (no ``.cpu()`` / ``.item()`` /
    ``.numpy()`` / python-scalar indexing / prints).
  * the parallel path produces byte-for-byte the same tensors as the serial path.

The three ``encode_*`` entry points share the :class:`EncoderModels` /
:class:`EncoderInputs` interface so a future vLLM-Omni ``EncoderStage`` can call
:func:`encode` directly (see ``__init__``-style dispatcher :func:`encode`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Encoder-parallelism modes exposed through the ``--parallel-encoder-mode`` flag.
MODE_SERIAL = "serial"
MODE_ONE_CARD = "one_card_stream"
MODE_TWO_CARD = "two_card"
VALID_MODES = (MODE_SERIAL, MODE_ONE_CARD, MODE_TWO_CARD)


# ---------------------------------------------------------------------------
# Data containers (the EncoderStage-facing interface)
# ---------------------------------------------------------------------------
@dataclass
class EncoderModels:
    """The two encoders + the VAE normalization buffers.

    ``text_encoder`` is a ``transformers.UMT5EncoderModel`` (bf16). ``vae`` is a
    ``DistributedAutoencoderKLWan`` (fp32 weights, bf16 autocast compute). The
    two buffers are the pipeline's ``vae_latents_mean`` / ``vae_latents_inv_std``
    (shape ``(1, C, 1, 1, 1)``, fp32).
    """

    text_encoder: nn.Module
    vae: nn.Module
    vae_latents_mean: torch.Tensor
    vae_latents_inv_std: torch.Tensor
    # Optional torch.compile(inductor) callable ``f(vae_input) -> mu`` built by
    # :func:`compile_vae_encode`. When set, :func:`run_vae_encoder` dispatches to
    # it instead of the eager path (inductor fuses the VAE's memory-bound
    # elementwise tail; the un-fusible oneDNN convs are unchanged). ``None`` =
    # eager (the default / enforce_eager-equivalent path).
    vae_compiled_encode: object = None


@dataclass
class EncoderInputs:
    """Already-preprocessed encoder inputs (tokenization / video preprocessing
    is done *before* timing so it never pollutes the measured region).

    * ``text_tokens`` / ``attention_mask``: ``(B, L)`` from the UMT5 tokenizer.
    * ``vae_input``: ``(B, C, T, H, W)`` float, the I2V conditioning window
      (first frame + zero frames) exactly as built by ``_encode_image``.
    """

    text_tokens: torch.Tensor
    attention_mask: torch.Tensor
    vae_input: torch.Tensor


@dataclass
class EncoderOutputs:
    """The DiT-boundary tensors both encoders feed into ``_prefill_kv_cache``."""

    text_embedding: torch.Tensor
    vae_latent: torch.Tensor
    # Populated for two_card mode: cost of moving vae_latent to the DiT device.
    d2d_transfer_ms: float | None = None


# ---------------------------------------------------------------------------
# XPU stream / synchronization / memory utilities (with CUDA + CPU fallbacks)
# ---------------------------------------------------------------------------
def _accel(device: torch.device):
    """Return the ``torch.xpu`` / ``torch.cuda`` module for ``device`` (or None)."""
    if device.type == "xpu" and hasattr(torch, "xpu"):
        return torch.xpu
    if device.type == "cuda" and hasattr(torch, "cuda"):
        return torch.cuda
    return None


def set_accel_device(device: torch.device) -> None:
    """Make ``device`` the current device for its accelerator (best effort).

    ``torch.xpu`` memory / stream APIs on torch-xpu 2.12 are **current-device
    scoped** (they do not accept a ``device`` argument), so callers that want a
    specific card's stream or memory stats must set it current first."""
    accel = _accel(device)
    if accel is None or device.index is None or not hasattr(accel, "set_device"):
        return
    try:
        accel.set_device(device)
    except (RuntimeError, ValueError, TypeError):
        pass


def make_stream(device: torch.device):
    """Create a new stream on ``device``.

    ``torch.xpu.Stream`` (2.12) has no ``device=`` kwarg and creates the stream
    on the *current* device, so we set the device current first. Returns ``None``
    on CPU / unsupported builds so callers can degrade to serial execution.
    """
    accel = _accel(device)
    if accel is None or not hasattr(accel, "Stream"):
        return None
    set_accel_device(device)
    try:
        return accel.Stream()
    except (RuntimeError, TypeError):
        return None


def current_stream(device: torch.device):
    """The current stream on ``device`` (or ``None`` if unavailable)."""
    accel = _accel(device)
    if accel is None or not hasattr(accel, "current_stream"):
        return None
    try:
        return accel.current_stream(device)
    except (TypeError, RuntimeError):
        try:
            return accel.current_stream()
        except Exception:  # noqa: BLE001
            return None


def stream_context(device: torch.device, stream):
    """Context manager that makes ``stream`` current on ``device``.

    Falls back to a null context when streams are unavailable (CPU / old build).
    """
    accel = _accel(device)
    if accel is None or stream is None or not hasattr(accel, "stream"):
        import contextlib

        return contextlib.nullcontext()
    return accel.stream(stream)


def sync_device(device: torch.device) -> None:
    """Block until all work on ``device`` completes.

    Prefers the device-scoped ``synchronize(device)`` signature and falls back
    to the argument-less form (per the POC spec's fallback rule).
    """
    accel = _accel(device)
    if accel is None:
        return
    try:
        accel.synchronize(device)
    except (TypeError, RuntimeError):
        accel.synchronize()


def reset_peak_memory(device: torch.device) -> None:
    accel = _accel(device)
    if accel is None or not hasattr(accel, "reset_peak_memory_stats"):
        return
    # xpu stats are current-device scoped; select the device then call no-arg.
    set_accel_device(device)
    try:
        accel.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001 - best-effort telemetry
        pass


def peak_memory_gb(device: torch.device) -> float:
    """Peak allocated memory on ``device`` in GiB (0.0 if unavailable)."""
    accel = _accel(device)
    if accel is None or not hasattr(accel, "max_memory_allocated"):
        return 0.0
    set_accel_device(device)
    try:
        return accel.max_memory_allocated() / (1024**3)
    except Exception:  # noqa: BLE001
        return 0.0


def memory_allocated_gb(device: torch.device) -> float:
    """Currently allocated memory on ``device`` in GiB (0.0 if unavailable)."""
    accel = _accel(device)
    if accel is None or not hasattr(accel, "memory_allocated"):
        return 0.0
    set_accel_device(device)
    try:
        return accel.memory_allocated() / (1024**3)
    except Exception:  # noqa: BLE001
        return 0.0


# ---------------------------------------------------------------------------
# The two encoders (faithful to pipeline_dreamzero.py)
# ---------------------------------------------------------------------------
def run_text_encoder(
    text_tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    text_encoder: nn.Module,
) -> torch.Tensor:
    """UMT5 text encode -> ``[B, L, 4096]`` bf16 with padding positions zeroed.

    Equivalent to ``DreamZeroPipeline._encode_text`` (pipeline_dreamzero.py:650),
    but zeros padding with a **sync-free** mask instead of a device-scalar slice
    so it does not force a host sync inside a parallel-stream region. For B=1 the
    result is bit-identical to the pipeline.
    """
    seq_lens = attention_mask.gt(0).sum(dim=1).long()  # (B,)
    prompt_emb = text_encoder(text_tokens, attention_mask).last_hidden_state
    prompt_emb = prompt_emb.clone().to(dtype=torch.bfloat16)

    # Sync-free equivalent of ``for i, v in enumerate(seq_lens): prompt_emb[:, v:] = 0``.
    # keep_mask[b, l] == True while l < seq_lens[b]; zero everything else.
    seq_len_dim = prompt_emb.shape[1]
    positions = torch.arange(seq_len_dim, device=prompt_emb.device)  # (L,)
    keep_mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)  # (B, L)
    prompt_emb = prompt_emb.masked_fill(~keep_mask.unsqueeze(-1), 0.0)
    return prompt_emb


def _vae_encode_core(
    vae_input: torch.Tensor,
    vae: nn.Module,
    vae_latents_mean: torch.Tensor,
    vae_latents_inv_std: torch.Tensor,
) -> torch.Tensor:
    """The normalized-latent math, autocast-agnostic (used eager + compiled).

    ``vae._encode`` -> chunk -> ``(mu - mean) * inv_std`` -> cast back. Identical
    to ``DreamZeroPipeline._encode_vae_latents``. Kept as a free function (not a
    closure) so ``torch.compile`` can trace it once and reuse the graph.
    """
    input_dtype = vae_input.dtype
    hidden = vae._encode(vae_input.to(dtype=vae.dtype))
    mu, _ = hidden.chunk(2, dim=1)
    mean = vae_latents_mean.to(device=mu.device, dtype=mu.dtype)
    inv_std = vae_latents_inv_std.to(device=mu.device, dtype=mu.dtype)
    mu = (mu - mean) * inv_std
    return mu.to(dtype=input_dtype)


def compile_vae_encode(
    vae: nn.Module,
    vae_latents_mean: torch.Tensor,
    vae_latents_inv_std: torch.Tensor,
    *,
    autocast: bool = True,
    device_type: str = "xpu",
):
    """Return an inductor-compiled ``f(vae_input) -> mu`` for the VAE encode.

    On Intel XPU this fuses the VAE's memory-bound elementwise/normalization/pad/
    cat tail into ~130 triton kernels (the oneDNN convs are untouched), roughly
    halving VAE encode latency. The autocast context is baked into the compiled
    region so the result is numerically the same path as the eager encode.

    NOTE: the FIRST call triggers compilation (tens of seconds to minutes on XPU);
    callers must warm it up OUTSIDE any timed region. Requires a working
    triton-xpu (present in the v0240 image).
    """
    def _fn(vae_input: torch.Tensor) -> torch.Tensor:
        if autocast and vae_input.device.type in ("xpu", "cuda"):
            with torch.amp.autocast(device_type=vae_input.device.type, dtype=torch.bfloat16):
                return _vae_encode_core(vae_input, vae, vae_latents_mean, vae_latents_inv_std)
        return _vae_encode_core(vae_input, vae, vae_latents_mean, vae_latents_inv_std)

    return torch.compile(_fn, backend="inductor")


def run_vae_encoder(
    vae_input: torch.Tensor,
    vae: nn.Module,
    vae_latents_mean: torch.Tensor,
    vae_latents_inv_std: torch.Tensor,
    *,
    autocast: bool = True,
    compiled_encode=None,
) -> torch.Tensor:
    """Wan VAE encode -> normalized latent ``mu``.

    Equivalent to ``DreamZeroPipeline._encode_vae_latents`` (pipeline_dreamzero.py:847).
    ``autocast=True`` reproduces the ``torch.amp.autocast(bf16)`` that
    ``_encode_image`` wraps around the obs#1 VAE encode (the heavy path). The
    returned tensor is cast back to ``vae_input.dtype`` exactly as the pipeline does.

    ``compiled_encode`` (from :func:`compile_vae_encode`) short-circuits to the
    inductor-compiled graph, which already bakes in the autocast context; the
    eager path is otherwise byte-for-byte the original.
    """
    if compiled_encode is not None:
        return compiled_encode(vae_input)

    input_dtype = vae_input.dtype
    device = vae_input.device

    def _encode() -> torch.Tensor:
        hidden = vae._encode(vae_input.to(dtype=vae.dtype))
        mu, _ = hidden.chunk(2, dim=1)
        mean = vae_latents_mean.to(device=mu.device, dtype=mu.dtype)
        inv_std = vae_latents_inv_std.to(device=mu.device, dtype=mu.dtype)
        mu = (mu - mean) * inv_std
        return mu.to(dtype=input_dtype)

    if autocast and device.type in ("xpu", "cuda"):
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            return _encode()
    return _encode()


# ---------------------------------------------------------------------------
# Serial baseline
# ---------------------------------------------------------------------------
def encode_serial(
    inputs: EncoderInputs,
    models: EncoderModels,
    device: torch.device,
    *,
    vae_autocast: bool = True,
) -> EncoderOutputs:
    """Reference path: text encode, then VAE encode, on one device / one stream."""
    with torch.inference_mode():
        text_out = run_text_encoder(inputs.text_tokens, inputs.attention_mask, models.text_encoder)
        vae_out = run_vae_encoder(
            inputs.vae_input,
            models.vae,
            models.vae_latents_mean,
            models.vae_latents_inv_std,
            autocast=vae_autocast,
            compiled_encode=models.vae_compiled_encode,
        )
    return EncoderOutputs(text_embedding=text_out, vae_latent=vae_out)


# ---------------------------------------------------------------------------
# One-card, two-stream parallel
# ---------------------------------------------------------------------------
def _stream_wait_stream(waiter, waitee) -> None:
    """Make ``waiter`` wait for all work currently queued on ``waitee``.

    Used so a side stream does not start before the default stream's input
    copies/allocations are visible. Best-effort: silently no-op if the build
    lacks ``Stream.wait_stream``."""
    if waiter is None or waitee is None or not hasattr(waiter, "wait_stream"):
        return
    try:
        waiter.wait_stream(waitee)
    except Exception:  # noqa: BLE001
        pass


def encode_parallel_one_card(
    inputs: EncoderInputs,
    models: EncoderModels,
    device: torch.device,
    *,
    vae_autocast: bool = True,
    streams: tuple | None = None,
) -> EncoderOutputs:
    """Run text and VAE encoders on two streams of a single device.

    Both encoders and all inputs must already be resident on ``device`` before
    this is called. The two ``run_*`` helpers are sync-free, so the CPU enqueues
    both stream's kernels back-to-back and the device scheduler is free to
    interleave them. A single full-device sync joins both before returning.

    ``streams`` optionally supplies a reused ``(vae_stream, text_stream)`` pair
    (created once via :func:`make_stream`) so a serving ``EncoderStage`` called
    repeatedly does not allocate fresh stream objects on every call. When
    ``None``, two streams are created for this call.

    Falls back to :func:`encode_serial` if the platform has no stream support.
    """
    if streams is not None:
        vae_stream, text_stream = streams
    else:
        vae_stream = make_stream(device)
        text_stream = make_stream(device)
    if vae_stream is None or text_stream is None:
        logger.warning("Streams unavailable on %s; falling back to serial encode.", device)
        return encode_serial(inputs, models, device, vae_autocast=vae_autocast)

    # The inputs and model weights were placed on `device` via the default stream.
    # Make both side streams wait for that work so they cannot read the tensors
    # before their allocations/copies are visible (cross-stream data hazard). This
    # is a device-side dependency edge, not a host sync — it does not block the CPU.
    default_stream = current_stream(device)
    _stream_wait_stream(vae_stream, default_stream)
    _stream_wait_stream(text_stream, default_stream)

    with torch.inference_mode():
        # Enqueue the long pole (VAE) first so its kernels are in flight while the
        # short text encode is queued onto the other stream and slots into idle
        # compute. Neither helper syncs, so both queues fill without a barrier.
        with stream_context(device, vae_stream):
            vae_out = run_vae_encoder(
                inputs.vae_input,
                models.vae,
                models.vae_latents_mean,
                models.vae_latents_inv_std,
                autocast=vae_autocast,
                compiled_encode=models.vae_compiled_encode,
            )
        with stream_context(device, text_stream):
            text_out = run_text_encoder(inputs.text_tokens, inputs.attention_mask, models.text_encoder)
        # Join: the default stream waits for both side streams, then a full-device
        # sync guarantees the outputs are readable on the host / default stream.
        _stream_wait_stream(default_stream, vae_stream)
        _stream_wait_stream(default_stream, text_stream)
        sync_device(device)

    return EncoderOutputs(text_embedding=text_out, vae_latent=vae_out)


# ---------------------------------------------------------------------------
# Two-card parallel (upper-bound comparison)
# ---------------------------------------------------------------------------
def encode_parallel_two_card(
    inputs: EncoderInputs,
    models: EncoderModels,
    text_device: torch.device,
    vae_device: torch.device,
    *,
    gather_device: torch.device | None = None,
    vae_autocast: bool = True,
    time_transfer: bool = True,
) -> EncoderOutputs:
    """Run text encode on ``text_device`` and VAE encode on ``vae_device``.

    The encoders must already be placed on their respective devices and the
    inputs must be resident there (done by the caller before timing). Because the
    two ``run_*`` helpers do not sync, launching them back-to-back from one thread
    lets both cards run concurrently. Both devices are then synchronized.

    If ``gather_device`` is given (the DiT device, typically ``vae_device`` or
    ``text_device``), the VAE latent is copied there and — when
    ``time_transfer`` — the device-to-device copy cost is measured and reported
    in :attr:`EncoderOutputs.d2d_transfer_ms`.
    """
    with torch.inference_mode():
        # Launch both; neither call blocks the host, so the cards overlap.
        vae_out = run_vae_encoder(
            inputs.vae_input,
            models.vae,
            models.vae_latents_mean,
            models.vae_latents_inv_std,
            autocast=vae_autocast,
            compiled_encode=models.vae_compiled_encode,
        )
        text_out = run_text_encoder(inputs.text_tokens, inputs.attention_mask, models.text_encoder)
        sync_device(vae_device)
        sync_device(text_device)

    d2d_ms: float | None = None
    if gather_device is not None:
        if time_transfer:
            import time as _time

            sync_device(vae_device)
            sync_device(gather_device)
            t0 = _time.perf_counter()
            vae_out = vae_out.to(gather_device, non_blocking=True)
            if text_out.device != gather_device:
                text_out = text_out.to(gather_device, non_blocking=True)
            sync_device(gather_device)
            d2d_ms = (_time.perf_counter() - t0) * 1000.0
        else:
            vae_out = vae_out.to(gather_device)
            if text_out.device != gather_device:
                text_out = text_out.to(gather_device)

    return EncoderOutputs(text_embedding=text_out, vae_latent=vae_out, d2d_transfer_ms=d2d_ms)


# ---------------------------------------------------------------------------
# Dispatcher (EncoderStage-facing single entry point)
# ---------------------------------------------------------------------------
def encode(
    inputs: EncoderInputs,
    models: EncoderModels,
    *,
    mode: str = MODE_SERIAL,
    device: torch.device | None = None,
    text_device: torch.device | None = None,
    vae_device: torch.device | None = None,
    gather_device: torch.device | None = None,
    vae_autocast: bool = True,
) -> EncoderOutputs:
    """Single entry point that a future Omni ``EncoderStage`` can call.

    ``mode`` selects the execution strategy. ``serial`` / ``one_card_stream`` use
    ``device``; ``two_card`` uses ``text_device`` + ``vae_device`` (+ optional
    ``gather_device`` for the DiT). Output format is identical across modes.
    """
    if mode == MODE_SERIAL:
        assert device is not None, "serial mode requires device"
        return encode_serial(inputs, models, device, vae_autocast=vae_autocast)
    if mode == MODE_ONE_CARD:
        assert device is not None, "one_card_stream mode requires device"
        return encode_parallel_one_card(inputs, models, device, vae_autocast=vae_autocast)
    if mode == MODE_TWO_CARD:
        assert text_device is not None and vae_device is not None, "two_card mode requires text_device and vae_device"
        return encode_parallel_two_card(
            inputs,
            models,
            text_device,
            vae_device,
            gather_device=gather_device,
            vae_autocast=vae_autocast,
        )
    raise ValueError(f"Unknown parallel-encoder mode {mode!r}; expected one of {VALID_MODES}.")
