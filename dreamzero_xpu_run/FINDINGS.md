# DreamZero-DROID on Intel Arc Pro B60 (XPU) — How to Run It Smoothly

**Node:** `gnr17409` — 8× Intel Arc Pro B60 (24.5 GB each), dual-socket (cards 0–3 = NUMA0, 4–7 = NUMA1).
**Model:** `GEAR-Dreams/DreamZero-DROID` — 61 GB BF16 video-diffusion + action policy.
**Date verified:** 2026-06-24.

---

## TL;DR — the one thing that matters

**Run with Tensor Parallel = 8 (all 8 cards), not TP=4.** TP=4 OOMs in the
attention forward pass and *hangs* (looks like "GPU util 0"). TP=8 fits and
completes cleanly.

Also export these env vars so any residual OOM fails *fast* instead of hanging:

```bash
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0
export UR_L0_USE_IMMEDIATE_COMMANDLISTS=0
```

---

## Why TP=4 "showed GPU util 0" (root cause)

The symptom was misleading. The cards were **not idle** — the run was **hung**:

- All 4 `DiffusionWorker` processes pinned cards 0–3 at **22.22 % engine / 99.99 % memory**, busy-spinning forever in `urEventWait` (Intel Level-Zero driver).
- The processes were effectively **unkillable** — they spin inside the L0 driver ioctl, so `SIGKILL` is deferred. Only `docker restart` (cgroup teardown) clears them.
- A login that happened to catch the moment between a crash and the next launch saw "0 % memory" → hence the "GPU util 0" report.

**Actual root cause:** the forward pass **runs out of device memory** at
`vllm_omni/diffusion/models/dreamzero/causal_wan_model.py:587` (self-attention,
first prefill) → `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`.

The trap is XPU-specific: **with Level-Zero immediate command lists ON (the
default), an out-of-memory failure deep in the in-order queue surfaces as an
infinite `urEventWait` spin instead of a Python exception.** That spin *is* the
"hang." Disabling immediate command lists converts the silent hang into an
immediate, debuggable OOM error.

Under TP=4 each rank holds ~15 GB of weights plus the attention activations for
the full sequence — that overflows the 24.5 GB B60. TP=8 halves the per-rank
shard (model `num_heads = 40`, divisible by 8 → 5 heads/rank), so it fits.

## What was ruled out (so we don't chase these again)

Every one of these was tested in isolation on the same cards and **passed**, so
none is the cause:

| Hypothesis | Verdict | Evidence |
|---|---|---|
| fp64 → int64 device cast hangs | ❌ not it | exact `float64→int64 .to(xpu)` ran in 0.64 s |
| XCCL/oneCCL collective backend broken | ❌ not it | raw `all_reduce`, `all_to_all_single`, `batch_isend_irecv` P2P ring all < 1 s |
| The specific `[1,1785,5120]` bf16 all_reduce | ❌ not it | replayed ×100 in 1.2 s; ×30 under 96 % mem pressure in 0.46 s |
| CCL PCIe-topology env tuning | ❌ no effect | `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0`, `CCL_ATL_TRANSPORT`, etc. — still hung |
| Rank divergence / shape mismatch | ❌ not it | `TORCH_DISTRIBUTED_DEBUG=DETAIL`: all 4 ranks lockstep to identical SequenceNumber 8199 |
| TP=1 as a baseline | ❌ impossible | 61 GB model ≫ 24.5 GB/card → dies at weight alloc (`DEVICE_LOST`) |

The cards talk over **PCIe only** — Arc Pro B-series has **no XeLink** (that's a
data-center Max-series feature). The recurring `CCL_WARN ... PCIe connection`
message is correct and harmless, not a misconfiguration.

---

## Step-by-step: run it

All work happens inside the dev container `vllm-omni-dev-..` on the node
(image `vllm-omni-xpu`). Repo and HF cache are host mounts:
`/home/sdp/workspace/vllm-omni -> /workspace/vllm-omni`, `/data -> /mnt/data`.

1. **Deploy config** — `vllm_omni/deploy/dreamzero_tp8.yaml`:

   ```yaml
   pipeline: dreamzero
   distributed_executor_backend: mp
   dtype: bfloat16
   stages:
     - stage_id: 0
       devices: "0,1,2,3,4,5,6,7"
       max_num_seqs: 1
       enforce_eager: true
       model_class_name: DreamZeroPipeline
       parallel_config:
         tensor_parallel_size: 8
         cfg_parallel_size: 1
       model_config:
         default_robot_embodiment: roboarena
         policy_server_config:
           image_resolution: [180, 320]
           n_external_cameras: 2
           needs_wrist_camera: true
           needs_stereo_camera: false
           needs_session_id: true
           action_space: joint_position
   ```

2. **Launch env** (key lines):

   ```bash
   cd /workspace/vllm-omni/examples/offline_inference/dreamzero
   export ZE_AFFINITY_MASK=0,1,2,3,4,5,6,7
   export SYCL_UR_USE_LEVEL_ZERO_V2=0
   export VLLM_WORKER_MULTIPROC_METHOD=spawn
   export HF_HOME=/mnt/data
   export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0   # OOM fails fast, not hang
   export UR_L0_USE_IMMEDIATE_COMMANDLISTS=0
   python -u timed_export_tp8.py        # deploy_config -> dreamzero_tp8.yaml
   ```

3. **Expect:** model load ~40 s, first output ~66 s, full finish ~132 s, exit 0.
   Output = an MP4/GIF video + per-step robot action tensors.

## Operational gotchas

- **If a run ever hangs again** (cards stuck at 22.22 %/99.99 %, workers in
  `urEventWait`), `kill -9` will not work. Reap by killing the parent
  `python -u timed_export*` process; if processes stay in state `Rl`/`Z`, do
  `docker restart <container>` — the repo and model survive (host mounts).
- **Always confirm a clean start** (`xpu-smi dump -m 5` shows ~0.1 % on all
  cards, no `DiffusionWorker` procs) before launching. Launching over wedged
  workers produces a misleading transient `UR_RESULT_ERROR_DEVICE_LOST`.
- **TP must divide `num_heads` (40):** valid TP ∈ {1,2,4,5,8}, but only TP=8
  has enough aggregate memory for this model + activations on B60s.
