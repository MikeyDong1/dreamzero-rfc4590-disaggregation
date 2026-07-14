# DreamZero TP=4 raw-input serving test — summary

**Status: PARTIAL FAILURE.** Model construction/loading succeeded and the TP
replication question was answered definitively; end-to-end generation crashed
with a reproducible upstream device fault, so no video/action output, peak
memory, or wall-clock timing could be collected.

## What was attempted

- **Goal**: run DreamZero through the real vLLM-Omni serving path (`Omni` +
  offline example, NOT the direct-drive DiT-only bypass harness), TP=4 on
  cards 4,5,6,7, fed **raw** (non pre-encoded) three-camera MP4 input, and
  profile (1) text-encoder/VAE replication vs sharding, (2) peak memory,
  (3) time-to-complete, (4) output sanity.
- **Nodes tried**: `gnr17408` (`sdp@10.54.109.211`), then `gnr17409`
  (`sdp@10.54.109.214`) after gnr17408 became unavailable mid-session (its
  containers were removed, cards left untouched).
- **Image / revision**: `vllm-omni-xpu:latest` (gnr17408) / `vllm-omni-xpu:v0240`
  (gnr17409) — both at vllm-omni commit `0807dda9648f3805dc946158cbfc87486fca3bef`.
- **Model**: `GEAR-Dreams/DreamZero-DROID`, snapshot `96ad344138c66e82536422432ad742f015784942`.
- **Assets**: `YangshenDeng/vllm-omni-dreamzero-assets` (`exterior_image_1_left.mp4`,
  `exterior_image_2_left.mp4`, `wrist_image_left.mp4`; 24 frames/camera, 180x320).

## Result 1 (answered): text encoder / VAE replication vs DiT sharding — CONFIRMED

Collected via a custom worker RPC extension (`tp_replication_probe.py`) that
had every one of the 4 TP ranks report its own module parameter counts/shapes
to a shared file (`probe_reports/tp_report_rank{0..3}.json`), run right after
model load completed (before generation, which is where the crash below
happened).

| Module | rank0 | rank1 | rank2 | rank3 | Verdict |
|---|---|---|---|---|---|
| text_encoder (UMT5) | 10.582 GiB / 5,680,910,336 params | identical | identical | identical | **Fully replicated** — every rank holds the complete text encoder |
| image_encoder (CLIP) | 1.177 GiB / 632,076,801 params | identical | identical | identical | **Fully replicated** |
| vae (Wan VAE) | 0.236 GiB / 126,892,531 params | identical | identical | identical | **Fully replicated** |
| transformer (CausalWan DiT) | 8.130 GiB / 4,364,683,168 params | identical | identical | identical | **TP-sharded** (see below) — each rank holds only its 1/4 shard, so the *shard* size is identical across ranks by construction, but it is 1/4 of the model, not the full model |

DiT sharding proof (block-0 layer shapes, identical on all 4 ranks):

```
self_attn.qkv: [3840, 5120]   # full model: dim=5120, num_heads=40, head_dim=128
self_attn.o:   [5120, 1280]   # 3*(5120/4 heads-worth) = 3*1280 = 3840; TP=4 -> 10 heads/rank
ffn.0:         [3456, 5120]   # full ffn_dim=13824 (config.json); 13824/4 = 3456
ffn.2:         [5120, 3456]
```

`config.json`'s `diffusion_model_cfg`: `dim=5120, ffn_dim=13824, num_heads=40`.
`ffn.0` output width 3456 = 13824/4 and `qkv` output width 3840 = 3×(5120/4) —
**exactly 1/4 of the full unsharded dimensions**, confirming `QKVParallelLinear`
/`ColumnParallelLinear`/`RowParallelLinear` are correctly tensor-sharding the
DiT across all 4 ranks (10 attention heads/rank of 40 total).

**Answer to "are text encoder and VAE still duplicated over 4 cards?": YES —
confirmed duplicated (replicated) on this revision.** Every rank loads and
holds a full, byte-identical copy of the UMT5 text encoder, the CLIP image
encoder, and the Wan VAE. Only the CausalWan DiT transformer is TP-sharded.
This means TP=4 buys attention/FFN compute+memory sharding for the DiT only;
the ~12 GiB combined encoder+VAE footprint is paid on *every* one of the 4
cards, not divided among them.

## Result 2, 3, 4 (blocked): peak memory, completion time, output sanity — NOT COLLECTED

Generation crashed identically and reproducibly on **both nodes**, at TP=4
**and** at an isolated TP=1 control (single process, single card, no
distributed collectives), always at the same point:

```
RuntimeError: level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)
```

At TP=4 the crash surfaced during the first `omni.generate()` call
(`pipeline_dreamzero.py forward()`, at `embodiment_id = torch.tensor(...)`,
immediately after the AR-Diffusion KV-cache preallocation log lines). At an
isolated TP=1 control run (fresh container, single card, no TP/collectives at
all) the SAME error surfaced even earlier, during model **construction**
(`CausalWanModel.__init__` → `QKVParallelLinear.create_weights` →
`torch.empty(..., device=xpu:0)`).

Because the fault reproduces at TP=1 with no distributed collectives, on two
independent physical nodes/hardware instances, using two different Docker
image tags, it is not a TP-sharding bug, a node/hardware fault, or a
container/session staleness issue (fresh containers were created and cards
were independently verified healthy — `torch.randn`/`matmul`/large
`torch.empty` all succeeded outside the DreamZero worker process both before
and after each crash).

**Diagnosis: DreamZero's mandatory `ARDiffusionEngine` backend
(`vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine`, required
unconditionally by `DreamZeroPipeline.__init__`) is broken on Intel XPU at
this exact vllm-omni revision (`0807dda9648f`).** This is corroborated by
independent evidence found on gnr17409: prior maintainer runs
(`/data/vllm-omni/logs/dreamzero_tp8_bmg_proxy0.log`, June 27) succeeded with
**finite, sane actions** using the **standard diffusion engine** (KV-connector
path, no `engine_backend` override) at TP=8 — proving the non-AR-engine path
works correctly on this hardware/image family — while every attempt through
`ARDiffusionEngine` in this session failed identically. `DreamZeroPipeline`
hard-requires the AR engine (raises `ValueError` otherwise), so there is no
config-only way to route around this and still exercise the real serving path.

Two additional real upstream bugs were found and worked around (via a
`sitecustomize.py` import-time patch, not an installed-package edit) while
chasing this, in case they're useful for a fix:
1. `ARDiffusionModelRunner._preallocate_kv_cache` (`experimental/ar_diffusion/runner.py:199`)
   calls `torch.cuda.mem_get_info(self.device)` unconditionally on an XPU
   device, which raises `ValueError: Expected a cuda device, but got: xpu:N`.
2. `ARDiffusionModelRunner.execute_model(self, req)` doesn't accept the
   `kv_prefetch_jobs` kwarg that `DiffusionWorker.execute_model` always passes,
   raising `TypeError: unexpected keyword argument 'kv_prefetch_jobs'`.

Neither of these caused the DEVICE_LOST fault (both are Python-level errors,
patched before generation was reached); they were just the first two
blockers encountered before the real, unfixable-from-here device fault.

## Nodes / cleanup

- **gnr17408**: became unavailable mid-session per user instruction ("the node
  is done for some reason"). All containers created by this session
  (`vllm-omni-dz-serve-tp4-gnr17408`, `dz-tp1-isolated`) were removed. Nothing
  else on that node was touched. Downloaded assets/run directories left in
  place under `/data/sdp/mikey_dreamzero/runs/` in case still useful.
- **gnr17409**: test container (`xianzhed-dz-serve-tp4-gnr17409`) removed after
  the run. Cards 4-7 verified free and healthy (0 GiB used, `torch.xpu`
  compute succeeds) both before this session started and after every crash.
  No other user's containers/images touched.

## Artifacts in this bundle

- `probe_reports/tp_report_rank{0,1,2,3}.json` — the per-rank module
  parameter-count/shape reports (source of the replication-vs-sharding answer
  above).
- `driver.log` — full run log from the final (AR-engine, HF_HOME-fixed)
  attempt, including the DEVICE_LOST crash traceback.

## Recommended next step

To get real completion-time/peak-memory/output-sanity numbers, either:
(a) get an upstream fix/patch for `ARDiffusionEngine` on XPU (file a bug with
the two Python-level issues above plus this DEVICE_LOST repro as a starting
point), or (b) if session/KV-continuity across AR chunks is not required for
this particular measurement, ask whether a temporary code-level bypass of the
`engine_backend` guard in `DreamZeroPipeline.__init__` (using the standard
engine, matching the proven `dreamzero_tp8.yaml` pattern) is acceptable —
that path is proven to work on this hardware/image family.
