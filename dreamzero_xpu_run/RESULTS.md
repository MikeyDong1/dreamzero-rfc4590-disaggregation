# DreamZero-DROID XPU Run — Results

**Config:** TP=8 across 8× Intel Arc Pro B60 on `gnr17409`
**Model:** `GEAR-Dreams/DreamZero-DROID` (61 GB, BF16)
**Run date:** 2026-06-24 22:26–22:28 UTC
**Status:** ✅ `WRAPPER_EXIT=0` — completed end to end, clean shutdown.

## Timing

| Metric | Value |
|---|---|
| Model load | 40.7 s |
| Time to first output | 66.1 s |
| Time to output finished | 132.3 s |
| Per-generation times | 66.1 s, 60.9 s |
| Decode | 2.4 s |

## Output (behaves as expected: video + robot instructions)

```
FRAMES_SHAPE=(17, 352, 640, 3) dtype=uint8 min=0 max=255
ACTION[0] shape=(24, 8) dtype=float32 min=-0.0931 max=0.5634 finite=True
ACTION[1] shape=(24, 8) dtype=float32 min=-0.0140 max=0.5406 finite=True
MP4=.../timed_prediction.mp4 exists=True bytes=224163
GIF=.../timed_prediction.gif exists=True bytes=2366228
```

- **Video:** 17 frames @ 352×640 RGB → `output/timed_prediction.mp4` (219 KB) and `output/timed_prediction.gif` (2.3 MB).
- **Robot actions:** 2 generations, each (24 horizon steps × 8 DoF) float32, all values finite and in a sane range.

## Prompt used

> "Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan"

## Files in this folder

- `output/timed_prediction.mp4` — generated prediction video
- `output/timed_prediction.gif` — same, as GIF
- `output/run_tp8.log` — full stdout/stderr of the successful run
- `../FINDINGS.md` — root-cause analysis and how to reproduce

## Environment that produced this

```
ZE_AFFINITY_MASK=0,1,2,3,4,5,6,7
SYCL_UR_USE_LEVEL_ZERO_V2=0
VLLM_WORKER_MULTIPROC_METHOD=spawn
HF_HOME=/mnt/data
SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0
UR_L0_USE_IMMEDIATE_COMMANDLISTS=0
deploy_config = vllm_omni/deploy/dreamzero_tp8.yaml  (tensor_parallel_size: 8)
torch 2.11.0+xpu, vLLM 0.23.0, vLLM-Omni 0.1.dev5+gd1bad6c57.xpu
```
