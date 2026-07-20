# DreamZero disaggregated — per-process profiling demo

End-to-end command-line recipe to reproduce the deep per-process profiling run
that decomposes the disaggregated DreamZero pipeline's per-request wall time into
encode / denoise / decode compute + inter-stage transport, and copies the result
bundle back to the local PC.

This is the exact flow used for run
`20260719_085915_gnr17405_disagg_tp4_decode_card3_profiled` (decode moved to a
separate card). Result of that run: **denoise DiT loop is ~95% of the ~22.6 s
steady-state wall time; disaggregation/transport overhead is <1%.**

Harness scripts referenced below live in [`dreamzero_xpu_run/`](dreamzero_xpu_run/):

| file | role |
|---|---|
| `prof_sitecustomize.py` | profiler + XPU shims; deployed as `sitecustomize.py` on `PYTHONPATH` so Python auto-imports it in **every** process (orchestrator + all spawned stage/rank workers) |
| `timed_disagg_prof.py` | driver: builds observations, runs multi-request, gathers per-stage peak-mem + server-perf RPCs |
| `deploy_disagg_decode_card3.yaml` | 3-stage deploy: encode / denoise TP4 / decode-on-separate-card |
| `run_disagg_prof.sh` | container launch with the device layout + all env wiring |
| `analyze_disagg_prof.py` | offline: reconstructs the cross-process timeline from `events.<pid>.jsonl` |

---

## 0. Node & prerequisites

Approved node (key auth, no password needed):

```bash
ssh sdp@10.54.109.207          # gnr17405
```

Fixed assets already on the node (do not re-download):

- Model: `/data/sdp_dreamzero/hf_home/hub/models--GEAR-Dreams--DreamZero-DROID`
- Camera MP4s: `/data/sdp_dreamzero/assets/{exterior_image_1_left,exterior_image_2_left,wrist_image_left}.mp4`
- Base checkout (has the `examples/` harness but NOT the disagg code): `/tmp/vllm-omni-fresh`
- Diffusion-ready image: `vllm-omni-xpu:latest` (vLLM 0.23.0)

Sanity-check the node is free before launching:

```bash
ssh sdp@10.54.109.207 'for d in 0 3 4 5 6 7; do echo -n "dev$d: "; \
  xpu-smi stats -d $d 2>/dev/null | grep -i "GPU Memory Used" | head -1; done'
```

All six target cards should read `0` MiB used.

---

## 1. Stage a disagg-capable checkout on the node

The node's base checkout and the baked image are upstream-main **without** the
RFC4590 disaggregation code. Overlay the local (this-repo) `vllm_omni` onto a
copy of the node checkout so you keep the node's `examples/` harness but run the
disaggregation source.

```bash
# --- pick a run id (node clock) ---
RUN_ID="$(ssh sdp@10.54.109.207 date +%Y%m%d_%H%M%S)_gnr17405_disagg_tp4_decode_card3_profiled"
RD="/home/sdp/dreamzero-vllm-omni-runs/$RUN_ID"

# --- create the run tree + copy the base checkout (keeps examples/ harness) ---
ssh sdp@10.54.109.207 "mkdir -p $RD/{logs,metrics,output,config,scripts,prof} $RD/metrics/prof && \
  cp -a /tmp/vllm-omni-fresh $RD/vllm-omni"

# --- overlay THIS repo's vllm_omni (disagg code) onto the staged checkout ---
#     (Windows has no rsync/sshpass: tar + scp + untar)
tar --exclude='__pycache__' --exclude='*.pyc' -czf /tmp/vllm_omni_overlay.tgz vllm_omni
scp /tmp/vllm_omni_overlay.tgz sdp@10.54.109.207:/tmp/vllm_omni_overlay.tgz
ssh sdp@10.54.109.207 "rm -rf $RD/vllm-omni/vllm_omni && \
  tar xzf /tmp/vllm_omni_overlay.tgz -C $RD/vllm-omni && \
  ls $RD/vllm-omni/vllm_omni/diffusion/stage_payload.py"   # exists => overlay ok
```

---

## 2. Deploy the harness + profiler onto the node

The profiler must be named `sitecustomize.py` and sit on a directory that is
**prepended** to `PYTHONPATH` — that is what makes Python's `site` module
auto-import it at interpreter startup in every spawned worker (workers use
`multiprocessing` **spawn**, so parent-process monkeypatches do not propagate).

```bash
# from dreamzero_xpu_run/ in this repo:
scp timed_disagg_prof.py        sdp@10.54.109.207:$RD/vllm-omni/examples/offline_inference/dreamzero/
scp prof_sitecustomize.py       sdp@10.54.109.207:$RD/prof/sitecustomize.py
scp deploy_disagg_decode_card3.yaml sdp@10.54.109.207:$RD/config/deploy_config_used.yaml
scp run_disagg_prof.sh analyze_disagg_prof.py sdp@10.54.109.207:$RD/scripts/

# strip Windows CRLF (breaks `set -o pipefail` under bash) + chmod
ssh sdp@10.54.109.207 "sed -i 's/\r\$//' \
  $RD/scripts/run_disagg_prof.sh $RD/scripts/analyze_disagg_prof.py \
  $RD/prof/sitecustomize.py $RD/config/deploy_config_used.yaml \
  $RD/vllm-omni/examples/offline_inference/dreamzero/timed_disagg_prof.py && \
  chmod +x $RD/scripts/run_disagg_prof.sh"
```

### Device layout (decode on a separate card)

`run_disagg_prof.sh` sets `ZE_AFFINITY_MASK=0,3,4,5,6,7`, which renumbers the
physical cards to container-relative indices:

| container xpu | physical card | stage |
|---|---|---|
| xpu:0 | card 0 | encode |
| xpu:1 | card 3 | **decode (separate)** |
| xpu:2–5 | cards 4,5,6,7 | denoise TP4 |

and the YAML uses those container indices: encode `devices:"0"`, denoise
`devices:"2,3,4,5"`, decode `devices:"1"`.

---

## 3. Run the profiled multi-request test

```bash
ssh sdp@10.54.109.207 "cd $RD/scripts && \
  RUN_ID='$RUN_ID' DZ_MODE=multi DZ_NUM_CHUNKS=12 DZ_PROFILE=1 \
  nohup bash run_disagg_prof.sh > $RD/logs/run_multi.log 2>&1 &"
```

Knobs (env vars consumed by the run script / driver):

| var | meaning | value used |
|---|---|---|
| `DZ_MODE` | `multi` (N-chunk session) or `single` (1 warm request) | `multi` |
| `DZ_NUM_CHUNKS` | chunk requests after the warmup request | `12` (=> 13 total) |
| `DZ_PROFILE` | `1` also starts the worker torch.profiler | `1` |

The disaggregated topology needs **one sampling_params per stage** (the driver
builds `[sp_encode, sp_denoise, sp_decode]`; stage-0 carries the DreamZero
`robot_obs`/`session_id`/`reset` extra_args). Attention backend is forced to
`TORCH_SDPA` (FLASH_ATTN has no meta kernel, breaks inductor tracing).

### Watch progress

```bash
# model load (~110s) then per-request lines
ssh sdp@10.54.109.207 "grep -E 'MODEL_LOAD_S|num_stages=|\[gen\] req=|\[SUMMARY\]|WRAPPER_EXIT' $RD/logs/run_multi.log | tail -20"
```

`WRAPPER_EXIT=0` and a `[SUMMARY] ... collapse=False` line mean success. Six
`events.<pid>.jsonl` files (1 encode + 4 denoise ranks + 1 decode) will be in
`$RD/metrics/prof/`.

> **Profiler caveat:** do **not** add timers *inside* the compiled DiT graph
> (`predict_noise`, the TP `tensor_model_parallel_all_reduce`). Wrapping in-graph
> code trips `torch._dynamo.exc.FailOnRecompileLimitHit: Hard failure due to
> fullgraph=True` and kills the worker. Only eager phase boundaries are wrapped.

---

## 4. Analyze — reconstruct the cross-process timeline

```bash
ssh sdp@10.54.109.207 "cd $RD/scripts && \
  python3 analyze_disagg_prof.py $RD/metrics/prof $RD/metrics/result_multi.json"
```

Prints a per-request table (wall / encode / gapE→D / denoise / gapD→De / decode /
unaccounted / csf) and a steady-state mean, and writes
`$RD/metrics/prof/timeline_analysis.json`. The inter-stage **gaps** are computed
as `downstream.t_start − upstream.t_end` on the shared host wall clock, so they
capture transport + queue-wait directly.

### Per-stage peak XPU memory (host `xpu-smi` reads 0 under `ZE_AFFINITY_MASK`)

Peak memory is gathered inside the workers via the `gpu_mem_stats` RPC (already
called by `timed_disagg_prof.py`); read it from `metrics/result_multi.json`
(`peak_xpu_mem_per_stage`) or the `[peakmem]` log line.

---

## 5. Build the result bundle + import to the local PC

```bash
# --- assemble metrics.json + summary.md + checksums on the node ---
ssh sdp@10.54.109.207 "cd $RD && \
  sha256sum metrics/metrics.json metrics/result_multi.json \
    metrics/prof/timeline_analysis.json config/deploy_config_used.yaml \
    logs/run_multi.log summary.md metrics/prof/events.*.jsonl > checksums.sha256"

# --- tar the bundle (EXCLUDE the big staged checkout) + pull it home ---
ssh sdp@10.54.109.207 "cd ~/dreamzero-vllm-omni-runs && \
  tar --exclude='$RUN_ID/vllm-omni' -czf /tmp/tp_profile_bundle.tgz $RUN_ID"
scp sdp@10.54.109.207:/tmp/tp_profile_bundle.tgz /tmp/tp_profile_bundle.tgz

# --- extract into the local TP-profile folder + verify checksums ---
mkdir -p TP-profile
tar xzf /tmp/tp_profile_bundle.tgz -C TP-profile
cd "TP-profile/$RUN_ID" && sha256sum -c checksums.sha256
```

All lines should print `OK`. The local `TP-profile/<run_id>/` is the
authoritative bundle:

```
TP-profile/<run_id>/
├── summary.md                     # human-readable findings
├── metrics/
│   ├── metrics.json               # curated metrics + breakdown
│   ├── result_multi.json          # raw per-request marks, peak mem, layout
│   └── prof/
│       ├── timeline_analysis.json # cross-process timeline + subphases + transport
│       └── events.<pid>.jsonl     # raw per-process phase events (6 files)
├── config/deploy_config_used.yaml
├── logs/run_multi.log
├── scripts/                       # run + driver + profiler + analyzer (as run)
└── checksums.sha256
```

---

## 6. Cleanup

```bash
ssh sdp@10.54.109.207 "rm -f /tmp/tp_profile_bundle.tgz /tmp/vllm_omni_overlay.tgz"
# container is --rm (auto-removed on exit); cards free automatically.
# Leave the image and the model cache in place.
```

---

## Result headline (this run)

Steady-state (requests ≥2, n=11) of the ~22.6 s/request wall time:

| component | seconds | share |
|---|---:|---:|
| denoise DiT loop | 21.40 | 94.5% |
| encode | 1.05 | 4.7% |
| gap encode→denoise (transport+queue) | 0.099 | 0.4% |
| gap denoise→decode (transport+queue) | 0.051 | 0.2% |
| decode | 0.003 | 0.01% |
| orchestrator dispatch/collect | 0.029 | 0.1% |

The drag is the denoise DiT loop, which grows with the AR window (16.4→23.0 s)
then plateaus at ~23 s once the window cap (`window_chunks=9`) fills. It is
**not** disaggregation tax and **not** decode placement. Optimize the 16-step
DiT loop, not the stage plumbing.
