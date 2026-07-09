#!/usr/bin/env python3
"""Standalone DreamZero CausalWanModel — TP=1 + layerwise (block-wise) CPU<->XPU offload.

This bypasses the vLLM-Omni serving stack (no Omni / AsyncOmniEngine / DiffusionWorker
/ orchestrator). It:
  1. Bootstraps vLLM's distributed + model-parallel state at world_size=1, tp=1
     (the CausalWanModel uses QKVParallelLinear/ColumnParallelLinear/RowParallelLinear
     and the diffusion Attention layer, all of which require this state even at TP=1).
  2. Builds CausalWanModel (i2v, 40 layers, dim=5120) directly on CPU.
  3. Loads ONLY the DiT weight subset `action_head.model.*` from the root safetensors
     (with the same QKV-fusion / img_emb remap as DreamZeroPipeline.load_weights).
  4. Applies the repo's layerwise-offload hooks to model.blocks (1 of 40 blocks resident
     on XPU at a time, prefetch overlap), keeping the small non-block modules resident.
  5. Runs a prefill + N-step denoise loop with correctly-shaped *synthetic* conditioning
     (raw DiT forward — output is video-latent noise prediction + action noise prediction,
     NOT VAE-decoded RGB).

Reports MODEL_LOAD_S, TIME_TO_FIRST_OUTPUT_S, TIME_TO_OUTPUT_FINISHED_S and peak XPU
memory (to demonstrate the 28GB DiT fits on a single 24.5GB B60 via offload, where TP=4
OOMs). Saves output tensors as .npy.
"""

from __future__ import annotations

import argparse
import json
import os
import re as re_module
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

# ---- vLLM-Omni bootstrap imports -----------------------------------------
from vllm.config import CompilationConfig, DeviceConfig, VllmConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.dreamzero.causal_wan_model import CausalWanModel
from vllm_omni.diffusion.offloader.layerwise_backend import apply_block_hook
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.platforms import current_omni_platform

DEVICE = "xpu"


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# 1. Distributed / model-parallel bootstrap (world_size=1, tp=1)
# ---------------------------------------------------------------------------
def bootstrap_tp1(device: torch.device, od_config: OmniDiffusionConfig) -> VllmConfig:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29555")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    current_omni_platform.set_device(device)

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(),
        device_config=DeviceConfig(device=device),
    )
    vllm_config.parallel_config.tensor_parallel_size = 1
    vllm_config.parallel_config.data_parallel_size = 1

    init_distributed_environment(world_size=1, rank=0, local_rank=0)
    initialize_model_parallel(
        data_parallel_size=1,
        cfg_parallel_size=1,
        sequence_parallel_size=1,
        ulysses_degree=1,
        ring_degree=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )
    log("[bootstrap] distributed + model-parallel initialized (world=1, tp=1)")
    return vllm_config


# ---------------------------------------------------------------------------
# 2. Build CausalWanModel from the root config (DiT only)
# ---------------------------------------------------------------------------
def build_dit_on_cpu(model_path: str) -> tuple[CausalWanModel, dict]:
    with open(os.path.join(model_path, "config.json")) as f:
        root_cfg = json.load(f)
    ah_config = root_cfg["action_head_cfg"]["config"]
    diffusion_model_cfg = ah_config["diffusion_model_cfg"]

    transformer_kwargs = {
        k: v for k, v in diffusion_model_cfg.items() if k not in ("_convert_", "_target_")
    }
    transformer_kwargs["action_dim"] = ah_config["action_dim"]
    transformer_kwargs["max_state_dim"] = ah_config["max_state_dim"]
    transformer_kwargs["num_frame_per_block"] = ah_config["num_frame_per_block"]

    log(f"[build] CausalWanModel kwargs: {json.dumps(transformer_kwargs, default=str)}")
    # Build on CPU in bf16 (defer device placement to the offload step).
    with torch.device("cpu"):
        model = CausalWanModel(**transformer_kwargs)
    model = model.to(dtype=torch.bfloat16)
    model.eval()
    return model, ah_config


# ---------------------------------------------------------------------------
# 3. Load ONLY action_head.model.* (the DiT) with QKV-fusion + img_emb remap
#    (mirrors DreamZeroPipeline.load_weights, action_head.model branch)
# ---------------------------------------------------------------------------
def load_dit_weights(model: CausalWanModel, model_path: str) -> int:
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    loaded: set[str] = set()

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    # group keys by shard file to open each shard once
    shard_to_keys: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        if key.startswith("action_head.model."):
            shard_to_keys.setdefault(shard, []).append(key)

    for shard, keys in shard_to_keys.items():
        with safe_open(os.path.join(model_path, shard), framework="pt", device="cpu") as f:
            for name in keys:
                tensor = f.get_tensor(name)
                new_name = "transformer." + name[len("action_head.model.") :]
                new_name = (
                    new_name.replace("img_emb.proj.0.", "img_emb.norm1.")
                    .replace("img_emb.proj.1.", "img_emb.fc1.")
                    .replace("img_emb.proj.3.", "img_emb.fc2.")
                    .replace("img_emb.proj.4.", "img_emb.norm2.")
                )
                # This CausalWanModel version uses SEPARATE q/k/v
                # ColumnParallelLinear layers in every attention module (no
                # fused qkv), so q/k/v load directly by name like every other
                # weight — no shard fusion remap needed.

                # strip leading "transformer." since our root IS the transformer
                local_name = new_name[len("transformer.") :]

                if local_name in params:
                    param = params[local_name]
                    wl = getattr(param, "weight_loader", default_weight_loader)
                    wl(param, tensor.to(param.dtype))
                    loaded.add(local_name)
                elif local_name in buffers:
                    buffers[local_name].data.copy_(tensor.to(buffers[local_name].dtype))
                    loaded.add(local_name)

    total_params = len(params)
    log(f"[weights] loaded {len(loaded)} / {total_params} DiT params+buffers")
    missing = [n for n in params if n not in loaded]
    if missing:
        log(f"[weights] WARNING: {len(missing)} params NOT loaded (first 10): {missing[:10]}")
    return len(loaded)


# ---------------------------------------------------------------------------
# 4. Apply layerwise offload to model.blocks (repo hooks)
#    Mirrors LayerWiseOffloadBackend.enable() for a bare DiT module.
# ---------------------------------------------------------------------------
def _block_bytes(block: torch.nn.Module) -> int:
    return (
        sum(p.numel() * p.element_size() for p in block.parameters())
        + sum(b.numel() * b.element_size() for b in block.buffers())
    )


def enable_layerwise_offload(
    model: CausalWanModel,
    device: torch.device,
    pin_memory: bool = True,
    resident_blocks: int = -1,
    resident_mem_gib: float = 17.0,
):
    """Partial-residency layerwise offload.

    OPTIMIZATION over the stock 1-block sliding window: on a card with spare
    memory (this B60 has 32 GB but the stock window peaked at only ~8 GiB),
    keep the first K transformer blocks PERMANENTLY resident on the XPU and only
    sliding-window-offload the remaining (num_blocks - K). Each denoise step then
    pays the CPU->XPU copy for (num_blocks - K) blocks instead of all num_blocks
    -- removing K/num_blocks of the PCIe-bound per-step transfer (B-series has no
    XeLink, so every prefetch crosses PCIe). K is auto-sized from a memory budget
    using the measured per-block size, clamped to leave >=2 blocks offloaded so
    the sliding window stays valid.
    """
    blocks = list(model.blocks)
    num_blocks = len(blocks)

    # Move all NON-block children to device (resident).
    for name, child in model.named_children():
        if name == "blocks":
            continue
        child.to(device)
    # Move top-level params/buffers (rope freqs etc. are python lists/tensors, handled in fwd)
    for p in model._parameters.values():
        if p is not None:
            p.data = p.data.to(device)
    for b in model._buffers.values():
        if b is not None:
            b.data = b.data.to(device)

    # Decide how many leading blocks stay permanently resident (K).
    blk_bytes = _block_bytes(blocks[0])
    if resident_blocks is not None and resident_blocks >= 0:
        K = resident_blocks
    else:
        K = int((resident_mem_gib * (1024**3)) // blk_bytes)
    K = max(0, min(K, num_blocks - 2))  # keep >=2 blocks offloaded
    log(
        f"[offload] per-block ~{blk_bytes / 1024**2:.0f} MiB; keeping K={K}/{num_blocks} "
        f"blocks resident, {num_blocks - K} offloaded (sliding window). "
        f"Per-step H2D cut by ~{100 * K / num_blocks:.0f}%."
    )

    # Leading K blocks: permanently resident on device, NO hooks.
    for blk in blocks[:K]:
        blk.to(device)

    # Trailing (num_blocks - K) blocks: sliding-window offload over the sub-list.
    off_blocks = blocks[K:]
    n_off = len(off_blocks)
    copy_stream = current_omni_platform.Stream()

    # Pre-fetch the first offloaded block by hooking the last offloaded block to it.
    last_block, first_off = off_blocks[-1], off_blocks[0]
    last_hook = apply_block_hook(last_block, first_off, device, copy_stream, pin_memory)
    last_hook.prefetch_layer(non_blocking=False)

    block_hooks = [last_hook]
    for i, block in enumerate(off_blocks[:-1]):
        next_block = off_blocks[(i + 1) % n_off]
        hook = apply_block_hook(block, next_block, device, copy_stream, pin_memory)
        block_hooks.append(hook)
    for i in range(len(block_hooks)):
        block_hooks[i]._prev_hook = block_hooks[i - 1]

    return block_hooks


# ---------------------------------------------------------------------------
# 5. Build synthetic conditioning + run prefill + denoise loop
# ---------------------------------------------------------------------------
def make_kv_caches(num_layers, batch, num_heads, head_dim, dtype, device):
    kv = [torch.zeros(2, batch, 0, num_heads, head_dim, dtype=dtype, device=device) for _ in range(num_layers)]
    cross = [{"is_init": False, "k": None, "v": None, "k_img": None, "v_img": None} for _ in range(num_layers)]
    return kv, cross


def run(model: CausalWanModel, ah_config: dict, device: torch.device, num_steps: int, out_dir: Path):
    B = 1
    dim = model.dim
    num_heads = model.num_heads
    head_dim = dim // num_heads
    num_layers = model.num_layers
    nfpb = model.num_frame_per_block          # 2
    action_horizon = ah_config["action_horizon"]   # 24
    action_dim = model.action_dim             # 32
    max_state_dim = ah_config["max_state_dim"]      # 64
    text_len = model.text_len                 # 512

    # latent spatial dims for H=352,W=640 video -> /8 VAE -> 44 x 80
    H_lat, W_lat = 44, 80
    frame_seqlen = (H_lat // 2) * (W_lat // 2)  # patch (1,2,2): 22*40 = 880
    text_dim = 4096
    clip_tokens, clip_dim = 257, 1280

    gen = torch.Generator(device="cpu").manual_seed(1140)

    def rb(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32).to(device=device, dtype=torch.bfloat16)

    context = rb(B, text_len, text_dim)
    clip_feature = rb(B, clip_tokens, clip_dim)

    kv_cache, crossattn_cache = make_kv_caches(num_layers, B, num_heads, head_dim, torch.bfloat16, device)

    # ---- PREFILL (current_start_frame=0, single frame, no action) ----
    x_pre = rb(B, 16, 1, H_lat, W_lat)
    y_pre = rb(B, 20, 1, H_lat, W_lat)
    t_pre = torch.zeros(B, 1, dtype=torch.long, device=device)

    log("[run] starting prefill ...")
    t0 = time.perf_counter()
    with torch.no_grad():
        vp, ap, updated = model(
            x=x_pre, timestep=t_pre, context=context, seq_len=frame_seqlen,
            kv_cache=kv_cache, crossattn_cache=crossattn_cache, current_start_frame=0,
            y=y_pre, clip_feature=clip_feature, action=None, timestep_action=None,
            state=None, embodiment_id=None,
        )
    for i, kv in enumerate(updated):
        kv_cache[i] = kv.clone()
    current_omni_platform.synchronize()
    t_prefill = time.perf_counter() - t0
    log(f"[run] PREFILL done in {t_prefill:.3f}s, video_pred {tuple(vp.shape)}")

    # ---- DENOISE LOOP (current_start_frame=1, nfpb frames, with action) ----
    seq_len = frame_seqlen * nfpb
    x = rb(B, 16, nfpb, H_lat, W_lat)
    y = rb(B, 20, nfpb, H_lat, W_lat)
    action = rb(B, action_horizon, action_dim)
    state = rb(B, 1, max_state_dim)

    time_to_first = None
    last_vp = last_ap = None
    t_loop0 = time.perf_counter()
    with torch.no_grad():
        for step in range(num_steps):
            ts = torch.full((B, nfpb), int(1000 * (num_steps - step) / num_steps), dtype=torch.long, device=device)
            ts_a = torch.full((B, action_horizon), int(1000 * (num_steps - step) / num_steps), dtype=torch.long, device=device)
            # fresh KV caches per step for the moving window (denoise reuses prefill KV;
            # we pass update_kv_cache=False equivalent by not persisting updated caches)
            vp, ap, _ = model(
                x=x, timestep=ts, context=context, seq_len=seq_len,
                kv_cache=kv_cache, crossattn_cache=crossattn_cache, current_start_frame=1,
                y=y, clip_feature=clip_feature, action=action, timestep_action=ts_a,
                state=state, embodiment_id=None,
            )
            current_omni_platform.synchronize()
            if step == 0:
                time_to_first = time.perf_counter() - t_loop0
                log(f"[run] TIME_TO_FIRST_OUTPUT={time_to_first:.3f}s")
            last_vp, last_ap = vp, ap
            log(f"[run] denoise step {step+1}/{num_steps} done")
    t_loop = time.perf_counter() - t_loop0
    log(f"[run] DENOISE LOOP done in {t_loop:.3f}s")

    # ---- Save outputs ----
    out_dir.mkdir(parents=True, exist_ok=True)
    vp_np = last_vp.float().cpu().numpy()
    ap_np = last_ap.float().cpu().numpy() if last_ap is not None and last_ap.numel() else None
    np.save(out_dir / "video_noise_pred_latent.npy", vp_np)
    if ap_np is not None:
        np.save(out_dir / "action_noise_pred.npy", ap_np)

    peak = current_omni_platform.max_memory_allocated() / (1024**3)
    return {
        "prefill_s": t_prefill,
        "time_to_first_output_s": time_to_first,
        "time_to_output_finished_s": t_loop,
        "num_steps": num_steps,
        "video_pred_shape": list(vp_np.shape),
        "action_pred_shape": list(ap_np.shape) if ap_np is not None else None,
        "video_pred_finite": bool(np.isfinite(vp_np).all()),
        "peak_xpu_gib": peak,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--out-dir", default="/workspace/vllm-omni/outputs/dreamzero/standalone_layerwise")
    parser.add_argument(
        "--resident-blocks", type=int, default=-1,
        help="Number of leading DiT blocks to keep permanently resident on XPU "
        "(skip offload). -1 = auto-size from --resident-mem-gib.",
    )
    parser.add_argument(
        "--resident-mem-gib", type=float, default=17.0,
        help="Memory budget (GiB) for resident blocks when --resident-blocks=-1.",
    )
    args = parser.parse_args()

    device = torch.device(DEVICE, 0)
    out_dir = Path(args.out_dir)

    od_config = OmniDiffusionConfig(model=args.model_path, dtype=torch.bfloat16)
    vllm_config = bootstrap_tp1(device, od_config)

    results = {}
    with set_forward_context(vllm_config=vllm_config, omni_diffusion_config=od_config), \
         set_current_diffusion_config(od_config):
        t_load0 = time.perf_counter()
        model, ah_config = build_dit_on_cpu(args.model_path)
        load_dit_weights(model, args.model_path)
        enable_layerwise_offload(
            model, device,
            resident_blocks=args.resident_blocks,
            resident_mem_gib=args.resident_mem_gib,
        )
        model_load_s = time.perf_counter() - t_load0
        log(f"[main] MODEL_LOAD_S={model_load_s:.3f}")

        results = run(model, ah_config, device, args.num_steps, out_dir)
        results["model_load_s"] = model_load_s
        results["num_blocks"] = len(model.blocks)
        results["resident_blocks_arg"] = args.resident_blocks
        results["resident_mem_gib_arg"] = args.resident_mem_gib

    log("=========== RESULTS ===========")
    for k, v in results.items():
        log(f"{k} = {v}")
    log(f"MODEL_LOAD_S={results['model_load_s']:.3f}")
    log(f"TIME_TO_FIRST_OUTPUT_S={results['time_to_first_output_s']:.3f}")
    log(f"TIME_TO_OUTPUT_FINISHED_S={results['time_to_output_finished_s']:.3f}")
    log(f"PEAK_XPU_GIB={results['peak_xpu_gib']:.3f}")
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    log("DONE")


if __name__ == "__main__":
    main()
