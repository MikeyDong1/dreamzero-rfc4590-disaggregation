# DreamZero XPU Session — Handoff Summary (2026-06-27)

## Session Intent
Build the vLLM-Omni XPU Docker image on node **gnr17409**, download the DreamZero-DROID
model, and run DreamZero **offline serving** (proxy disabled). Through debugging, the
focus became fixing a **TP=8 `tensor_model_parallel_all_reduce` hang** on the Arc Pro
B60 (Battlemage) cards. **Root cause found and proven.** Current task:
**(Option 2) rebuild the image from the updated `Dockerfile.xpu`, then (Option 1) run TP=8 DreamZero with the BMG oneCCL fix.**

## Node / Access (see memory `gnr17409-node-access`)
- SSH: `sdp@10.54.109.214`, password `sdpIntel` (askpass pattern; non-interactive).
- 8× Intel Arc Pro B60 (Battlemage/BMG), 24.5 GB each, PCIe-only (no XeLink).
- `sudo`: `echo sdpIntel | sudo -S ...`. `xpu-smi` is host-side (sudo).
- Repo on node: `/home/sdp/workspace/vllm-omni` → mounts to `/workspace/vllm-omni`.
- HF cache: host `/data` → container `/mnt/data`, `HF_HOME=/mnt/data`.
- Docker root is on `/data` (1.1 TB free); host `/` is 100% full (~163 MB) — not a blocker.

## ROOT CAUSE (PROVEN) — BMG oneCCL mismatch
The TP all_reduce hang = the **default non-BMG oneCCL 2021.17 loads instead of the
BMG-capable oneCCL 2021.15.9**. The node `docker/Dockerfile.xpu` installs BMG oneCCL
2021.15.9 but only exposes it via `/root/.bashrc`; `.bashrc` line 6 `[ -z "$PS1" ] && return`
means that source NEVER runs under non-interactive `bash -c` (how runs execute), so the
baked `ENV CCL_ROOT=.../ccl/2021.17` wins. The BMG lib's version string says:
`Gold-2021.15.9 ... To run on BMG, CCL_SYCL_ALLGATHERV_TMP_BUF must be set to 0`.
Minimal probe (`_tmp_allreduce_probe.py`, torchrun, bf16 [1,1785,5120]):
- CCL 2021.17, 2 ranks → **HANG** (90s watchdog, exit 7) — reproduces the DreamZero hang.
- BMG 2021.15 (lib swap alone), 2 ranks → ✅ DONE ~1.4s, value=2.0.
- BMG 2021.15 + tmp_buf=0, **8 ranks** → ✅ all 8 DONE ~2.8s, value=8.0 (full TP=8 scale).
See memory `dreamzero-tp-allreduce-bmg-oneccl-rootcause` + local
`tp8 no-offload proxy0/ALLREDUCE_HANG_ROOT_CAUSE.md`.

## Key Decisions
- "Offline serving" = `examples/offline_inference/dreamzero/timed_export_tp8.py` driving
  the `Omni` engine with deploy yaml `vllm_omni/deploy/dreamzero_tp8.yaml` (TP=8, devices
  "0,1,2,3,4,5,6,7", no layerwise offload). proxy=0 = empty proxy envs + `HF_HUB_OFFLINE=1`.
- oneCCL **2021.13 is impossible** to pin (memory `dreamzero-oneccl-2021.13-incompatible`):
  needs `libsycl.so.7` (image has `.so.8`) AND lacks symbol `reduction_create_pre_mul_sum`
  that torch-xpu 2.11 requires. Wrong lever; abandoned.
- **TP=2 + layerwise offload** is the proven-correct fallback (finite actions + real video):
  yaml `vllm_omni/deploy/dreamzero_tp2_offload.yaml`, wrapper `timed_export_tp2_offload.py`.

## Artifact Trail
### Created (local, under `dreamzero_xpu_run/tp8 no-offload proxy0/`)
- `RESULTS.md` — TP=8 no-offload run results (both hung attempts).
- `ALLREDUCE_HANG_ROOT_CAUSE.md` — full root-cause analysis + probe table + fix options.
- `_tmp_allreduce_probe.py` — temporary minimal all_reduce probe (kept as documentation).
- `dreamzero_tp8_noffload_proxy0.log`, `dreamzero_tp8_fresh_proxy0.log` — the two hung runs.
- `dreamzero_tp8.yaml`, `timed_export_tp8.py` — copies of node config/wrapper.
### Created (local)
- `dreamzero_xpu_run/SESSION_HANDOFF.md` — this file.
### Memory files written/updated (C:\Users\xianzhed\.claude\projects\...\memory\)
- `dreamzero-tp8-noffload-allreduce-hang.md` (hang observation; fresh-container retry also hung)
- `dreamzero-oneccl-2021.13-incompatible.md` (why 2021.13 can't be pinned)
- `dreamzero-tp-allreduce-bmg-oneccl-rootcause.md` (PROVEN root cause + fix) + MEMORY.md index
### Node state
- Image `vllm-omni-xpu:latest` (id 92a207eade15, 26.5 GB) built from node `docker/Dockerfile.xpu`.
- DreamZero-DROID cached: `/data/hub/models--GEAR-Dreams--DreamZero-DROID/snapshots/96ad344138c66e82536422432ad742f015784942` (61 GB).
- Containers: `vllm-omni-dev-dreamzero` (left intact). Throwaway containers removed
  (`...-ccl2113`, `...-tp8fresh`, `vllm-omni-probe`). Temp `_tmp_allreduce_probe.py` removed from node.
- Updated Dockerfile to build from: **`C:\Users\xianzhed\Downloads\Dockerfile.xpu`** — its
  **line 104** bakes `/opt/intel/oneapi/ccl/2021.15/lib` to FRONT of `ENV LD_LIBRARY_PATH`
  (node version only adds `/usr/local/lib`). Also pins oneAPI to 2025.3 (apt prefs, lines 32–45),
  base `intel/deep-learning-essentials:2025.3.2-0-devel-ubuntu24.04`.

## Current State / Next Steps (THIS TASK)
1. **Option 2 — rebuild image** from `~/Downloads/Dockerfile.xpu`: copy it to the node repo,
   build `vllm-omni-xpu` (tag e.g. `vllm-omni-xpu:bmg` or overwrite `:latest`). It bakes
   BMG oneCCL 2021.15 onto LD_LIBRARY_PATH so non-interactive runs auto-load it.
   Recommend also `ENV CCL_SYCL_ALLGATHERV_TMP_BUF=0`.
2. **Option 1 — run TP=8 DreamZero** offline serving on a fresh container from the new image,
   proxy=0 + `HF_HUB_OFFLINE=1`, cards 0–7. Expect MODEL_LOAD ~45s, first output ~66s,
   finish ~132s; outputs = finite robot actions + real mp4/gif. Save log + mp4/gif locally.

## Run knobs (env every docker exec uses)
`ZE_AFFINITY_MASK=0,1,2,3,4,5,6,7`, `SYCL_UR_USE_LEVEL_ZERO_V2=0`,
`VLLM_WORKER_MULTIPROC_METHOD=spawn`, `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0`,
`UR_L0_USE_IMMEDIATE_COMMANDLISTS=0`, `HF_HOME=/mnt/data`, `HF_HUB_OFFLINE=1`, empty proxies.
Build proxy = `http://proxy-dmz.intel.com:912`.
