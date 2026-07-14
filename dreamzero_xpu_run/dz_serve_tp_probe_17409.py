#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real-serving-path DreamZero test: raw camera MP4 -> Omni -> real forward.

gnr17409 variant: this node's on-disk /data/vllm-omni checkout has an older,
self-contained export_prediction_video.py (no client_schedule.py module split),
so this reuses ITS OWN helpers (_build_observations / _extract_latents /
_decode_with_worker) exactly like the maintainer's own timed_export.py does,
instead of importing client_schedule (which doesn't exist at this revision).

Goes through the actual vLLM-Omni serving engine (tokenizer -> UMT5 -> image
encoder -> VAE encode -> DiT -> VAE decode) with RAW camera frames -- not a
direct-drive DiT-only bypass harness.

Collects:
- wall-clock timing (model load, first-output, total generate, decode)
- per-rank module report via a custom worker extension RPC (tp_replication_report)
  to check whether text_encoder/image_encoder/vae are replicated (identical shapes
  on every rank, no TP-sharding classes) vs the transformer (TP-sharded via
  QKVParallelLinear/ColumnParallelLinear/RowParallelLinear)
- gpu_mem_stats() (process allocator peak) per rank, cross-checked against the
  external whole-device sampler (xpu_memory.csv, started by the caller shell script)
- action + video output sanity (dtype/shape/nan/zero checks, saved PNG frame + npz)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="GEAR-Dreams/DreamZero-DROID")
    p.add_argument("--deploy-config", type=Path, required=True)
    p.add_argument("--video-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--prompt",
        default="Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan",
    )
    p.add_argument("--fps", type=int, default=5)
    p.add_argument(
        "--report-dir",
        type=Path,
        default=Path("/workspace/probe_reports"),
        help="Shared dir each worker rank writes its own JSON report into (must match TP_PROBE_REPORT_DIR).",
    )
    p.add_argument("--tp-size", type=int, default=4, help="Expected worker world size, for report-file polling.")
    p.add_argument(
        "--dreamzero-example-dir",
        type=Path,
        default=Path("/workspace/vllm-omni/examples/offline_inference/dreamzero"),
        help="Dir containing export_prediction_video.py (this node's on-disk checkout).",
    )
    return p.parse_args()


def _read_rank_reports(
    report_dir: Path, prefix: str, expected_count: int | None = None, timeout_s: float = 10.0
) -> list[dict]:
    """Read per-rank JSON files, retrying briefly since the RPC returns to the
    caller as soon as rank 0 replies -- other ranks may still be flushing their
    file to the (possibly network/overlay) shared mount for a few hundred ms."""
    deadline = time.perf_counter() + timeout_s
    reports = []
    while True:
        reports = []
        for f in sorted(report_dir.glob(f"{prefix}_rank*.json")):
            try:
                reports.append(json.loads(f.read_text()))
            except (json.JSONDecodeError, OSError):
                continue  # file mid-write; retry
        if expected_count is None or len(reports) >= expected_count or time.perf_counter() >= deadline:
            break
        time.sleep(0.2)
    return sorted(reports, key=lambda r: r.get("rank", 0))


def main():
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.dreamzero_example_dir))

    import export_prediction_video as E

    from vllm_omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.outputs import OmniRequestOutput

    session_id = f"dreamzero-tp4probe-{uuid.uuid4()}"

    camera_frames, observations = E._build_observations(args.video_dir, args.prompt, session_id)
    print(f"Loaded raw camera frames: {[(k, v.shape) for k, v in camera_frames.items()]}", flush=True)
    print(f"Built {len(observations)} AR observations (this revision hardcodes 1 initial + 1 chunk)", flush=True)

    timings = {}
    t0 = time.perf_counter()
    omni = Omni(
        model=args.model,
        deploy_config=str(args.deploy_config),
        enforce_eager=True,
        worker_extension_cls="tp_replication_probe.TPReplicationProbeExtension",
    )
    if torch.accelerator.is_available():
        torch.accelerator.synchronize()
    timings["model_load_time_ms"] = (time.perf_counter() - t0) * 1000.0
    print(f"model_load_time_ms={timings['model_load_time_ms']:.1f}", flush=True)

    # ---- TP replication probe: run BEFORE any generation so it reflects the
    # loaded-but-idle state (weights only, no transient activation memory) ----
    stage_client = omni.engine.stage_clients[0]
    engine = getattr(stage_client, "_engine", None)
    if engine is None:
        raise RuntimeError("Requires inline diffusion stage access.")

    # NOTE: collective_rpc's Python return value only surfaces rank 0's result
    # (see tp_replication_probe.py docstring) even with exec_all_ranks=True; the
    # method still runs on every rank, and each rank writes its own file to the
    # shared --report-dir mount, which we read directly below.
    t_probe0 = time.perf_counter()
    engine.executor.collective_rpc("tp_replication_report", args=(), unique_reply_rank=0, exec_all_ranks=True)
    timings["tp_probe_time_ms"] = (time.perf_counter() - t_probe0) * 1000.0
    tp_reports = _read_rank_reports(args.report_dir, "tp_report", expected_count=args.tp_size)
    print(f"TP replication reports from {len(tp_reports)} rank file(s) collected.", flush=True)
    got_ranks = {r.get("rank") for r in tp_reports}
    if got_ranks != set(range(args.tp_size)):
        raise RuntimeError(
            f"tp_replication_report: expected ranks 0..{args.tp_size - 1}, "
            f"only got {sorted(got_ranks)} -- a worker rank may have failed silently "
            "(unique_reply_rank=0 does not surface non-zero-rank errors)."
        )

    outputs: list[OmniRequestOutput] = []
    per_request_times = []
    t_gen0 = time.perf_counter()
    first_output_t = None
    for index, obs in enumerate(observations):
        sampling_params = OmniDiffusionSamplingParams(
            extra_args={"reset": index == 0, "session_id": obs["session_id"], "robot_obs": obs}
        )
        t_req0 = time.perf_counter()
        result = omni.generate(obs["prompt"], sampling_params_list=[sampling_params])
        t_req1 = time.perf_counter()
        if first_output_t is None:
            first_output_t = t_req1
        per_request_times.append((t_req1 - t_req0) * 1000.0)
        if not result:
            raise RuntimeError(f"No output returned for DreamZero request {index}")
        outputs.append(result[0])
        print(f"  request {index}: {(t_req1 - t_req0) * 1000.0:.1f} ms", flush=True)

    timings["time_to_first_output_ms"] = (first_output_t - t_gen0) * 1000.0
    timings["total_generation_time_ms"] = (time.perf_counter() - t_gen0) * 1000.0
    timings["per_request_ms"] = per_request_times

    # ---- sanity-check action outputs ----
    action_reports = []
    for i, out in enumerate(outputs):
        actions = out.multimodal_output.get("actions") if out.multimodal_output else None
        if actions is None:
            action_reports.append({"index": i, "present": False})
            continue
        arr = np.asarray(actions)
        action_reports.append(
            {
                "index": i,
                "present": True,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "has_nan": bool(np.isnan(arr).any()),
                "all_zero": bool(np.all(arr == 0)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
            }
        )
    print(f"Action sanity: {json.dumps(action_reports, indent=2)}", flush=True)

    # ---- decode video (reuse this revision's own extract/decode helpers) ----
    t_decode0 = time.perf_counter()
    latent_steps = [E._extract_latents(o) for o in outputs]
    full_latents = torch.cat(latent_steps, dim=2)
    decoded = engine.executor.collective_rpc(
        "decode_video_latents_to_uint8", args=(full_latents,), unique_reply_rank=0, exec_all_ranks=True
    )
    if isinstance(decoded, torch.Tensor):
        decoded = decoded.numpy()
    timings["decode_time_ms"] = (time.perf_counter() - t_decode0) * 1000.0

    frames = decoded
    video_report = {
        "shape": list(frames.shape),
        "dtype": str(frames.dtype),
        "has_nan": bool(np.isnan(frames.astype(np.float32)).any()) if frames.dtype != np.uint8 else False,
        "all_zero": bool(np.all(frames == 0)),
        "min": int(frames.min()),
        "max": int(frames.max()),
        "mean": float(frames.mean()),
        "std": float(frames.std()),
    }
    print(f"Video sanity: {json.dumps(video_report, indent=2)}", flush=True)

    mp4_path = args.output_dir / "dreamzero_tp4_probe.mp4"
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), float(args.fps), (width, height))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"SAVED_MP4={mp4_path} frames={len(frames)}", flush=True)

    cv2.imwrite(str(args.output_dir / "frame_first.png"), cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(args.output_dir / "frame_last.png"), cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR))

    # ---- gpu mem stats (process allocator peak, per rank) ----
    engine.executor.collective_rpc("dump_gpu_mem_stats", args=(), unique_reply_rank=0, exec_all_ranks=True)
    mem_stats = _read_rank_reports(args.report_dir, "mem", expected_count=args.tp_size)
    if {m.get("rank") for m in mem_stats} != set(range(args.tp_size)):
        print(f"WARNING: dump_gpu_mem_stats only got ranks {[m.get('rank') for m in mem_stats]}", flush=True)

    timings["total_e2e_ms"] = (time.perf_counter() - t0) * 1000.0

    omni.close()

    result = {
        "timings_ms": timings,
        "tp_replication_reports": tp_reports,
        "gpu_mem_stats_per_rank": mem_stats,
        "action_sanity": action_reports,
        "video_sanity": video_report,
        "num_observations": len(observations),
        "output_mp4": str(mp4_path),
    }
    out_json = args.output_dir / "probe_result.json"
    out_json.write_text(json.dumps(result, indent=2, default=str))
    print(f"SAVED_JSON={out_json}", flush=True)
    print("RUN_OK", flush=True)


if __name__ == "__main__":
    main()
