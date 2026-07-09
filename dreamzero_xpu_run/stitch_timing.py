#!/usr/bin/env python3
"""Time the CPU-side 3-camera -> 1-image stitching in DreamZero preprocessing.

This is the transform step that runs at the START of DreamZeroPipeline.forward()
(via _transform_robot_obs -> transform.transform_input), BEFORE any encoder. It
turns the 3 camera frames into ONE 352x640 stitched image. Pure CPU (numpy +
torchvision), independent of the XPU card.

Decomposes:
  transform_input (total)  = per-view crop+resize (x3) + 2x2 mosaic + template + state
    _stitch_views          = 3x _preprocess_view + mosaic assembly
      _preprocess_view (x3) = center-crop 95% + bilinear resize to 176x320
Reports cold + warm (mean of N reps) in milliseconds.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import vllm_omni  # noqa: F401
from vllm_omni.diffusion.models.dreamzero.transform import ensure_transforms_loaded
from vllm_omni.diffusion.models.dreamzero.transform.base import get_transform

CAMERA_FILES = {
    "observation/exterior_image_0_left": "exterior_image_1_left.mp4",
    "observation/exterior_image_1_left": "exterior_image_2_left.mp4",
    "observation/wrist_image_left": "wrist_image_left.mp4",
}
PROMPT = ("Move the pan forward and use the brush in the middle of the plates "
          "to brush the inside of the pan")


def first_frame(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"no frame from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def build_obs(video_dir: Path):
    obs = {}
    for cam_key, fname in CAMERA_FILES.items():
        obs[cam_key] = first_frame(video_dir / fname)  # (180,320,3) single frame
    obs["observation/joint_position"] = np.zeros(7, dtype=np.float32)
    obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
    obs["observation/gripper_position"] = np.zeros(1, dtype=np.float32)
    obs["prompt"] = PROMPT
    obs["embodiment"] = "roboarena"
    return obs


def timeit(fn, reps):
    fn()  # cold
    t0 = time.perf_counter(); fn(); cold = time.perf_counter() - t0
    warm = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); warm.append(time.perf_counter() - t)
    return cold, warm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args()

    ensure_transforms_loaded()
    transform = get_transform("roboarena")
    obs = build_obs(Path(args.video_dir))

    # Map raw obs -> transform's internal images dict (images/exterior_0, ...)
    images = {transform.IMAGE_KEY_MAP[k]: obs[k] for k in CAMERA_FILES if k in transform.IMAGE_KEY_MAP}

    print(f"[input] 3 camera first-frames, each {obs['observation/exterior_image_0_left'].shape}", flush=True)

    # (a) full transform_input
    c1, w1 = timeit(lambda: transform.transform_input(obs), args.reps)
    # (b) just the 2x2 stitch (3x preprocess_view + mosaic)
    c2, w2 = timeit(lambda: transform._stitch_views(dict(images)), args.reps)
    # (c) a single per-view crop+resize
    one_view = images["images/exterior_0"]
    c3, w3 = timeit(lambda: transform._preprocess_view(one_view[np.newaxis]), args.reps)

    def ms(x): return x * 1000
    def mean(w): return sum(w) / len(w) * 1000

    out = transform.transform_input(obs)
    stitched = np.asarray(out["images"])
    print(f"[output] stitched images shape={stitched.shape} dtype={stitched.dtype}", flush=True)
    print("", flush=True)
    print("=========== STITCH TIMING (CPU, ms) ===========", flush=True)
    print(f"transform_input (total)  : cold {ms(c1):7.2f} | warm mean {mean(w1):6.3f}  (min {min(w1)*1000:.3f} max {max(w1)*1000:.3f})", flush=True)
    print(f"  _stitch_views          : cold {ms(c2):7.2f} | warm mean {mean(w2):6.3f}  (min {min(w2)*1000:.3f} max {max(w2)*1000:.3f})", flush=True)
    print(f"    _preprocess_view (1) : cold {ms(c3):7.2f} | warm mean {mean(w3):6.3f}  (x3 views = {mean(w3)*3:.3f} ms)", flush=True)
    print(f"  (template+state remain : ~{mean(w1)-mean(w2):.3f} ms)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
