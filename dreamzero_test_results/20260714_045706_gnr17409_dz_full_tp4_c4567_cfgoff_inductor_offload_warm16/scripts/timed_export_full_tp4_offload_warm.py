#!/usr/bin/env python3
"""FULL/monolithic DreamZero: TP=4 cards 4,5,6,7, CFG off, inductor ON,
layerwise CPU offload, denoise_steps=16, raw camera MP4 input. WARM only:
request 0 discarded (cold load + first-call compile); requests 1,2 = warm.
"""
import time, uuid, sys, json
from pathlib import Path
import numpy as np, torch

sys.argv = ["x"]
sys.path.insert(0, "/workspace/vllm-omni/examples/offline_inference/dreamzero")
import export_prediction_video as E
import vllm_omni
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

print("[env] vllm_omni loaded from: %s" % vllm_omni.__file__, flush=True)

MODEL = "GEAR-Dreams/DreamZero-DROID"
DEPLOY = Path("/workspace/config_run/dreamzero_tp4_c4567_cfgoff_inductor_offload.yaml")
VIDEO_DIR = Path("/workspace/assets_run")
OUT = Path("/workspace/output_run")
METRICS_OUT = Path("/workspace/metrics_run/metrics.json")
PROMPT = "Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan"
NUM_REQUESTS = 3


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sid = "full-tp4-offload-" + str(uuid.uuid4())
    T = {}
    camera_frames, base_obs = E._build_observations(video_dir=VIDEO_DIR, prompt=PROMPT, session_id=sid)
    observations = base_obs + [base_obs[-1]]
    print("[timing] obs_built n=%d" % len(observations), flush=True)

    t0 = time.perf_counter()
    # inductor ON: enforce_eager=False (standing setup). Deploy YAML also sets it.
    omni = Omni(model=MODEL, deploy_config=str(DEPLOY), enforce_eager=False, worker_extension_cls=E.WORKER_EXTENSION)
    t1 = time.perf_counter()
    T["model_load_s"] = t1 - t0
    print("[timing] MODEL_LOADED_S=%.3f" % T["model_load_s"], flush=True)

    outputs, gen_times = [], []
    for i, obs in enumerate(observations[:NUM_REQUESTS]):
        sp = OmniDiffusionSamplingParams(extra_args={"reset": i == 0, "session_id": obs["session_id"], "robot_obs": obs})
        g0 = time.perf_counter()
        result = omni.generate(obs["prompt"], sampling_params_list=[sp])
        g1 = time.perf_counter()
        gen_times.append(g1 - g0)
        if not result:
            raise RuntimeError("No output for request %d" % i)
        outputs.append(result[0])
        print("[timing] %s_gen%d_s=%.3f" % ("COLD" if i == 0 else "WARM", i, g1 - g0), flush=True)

    T["cold_gen0_s"] = gen_times[0]
    T["warm_gen_times_s"] = gen_times[1:]
    T["warm_gen_mean_s"] = sum(gen_times[1:]) / len(gen_times[1:])

    td0 = time.perf_counter()
    warm_latents = torch.cat([E._extract_latents(o) for o in outputs[1:]], dim=2)
    frames = E._decode_with_worker(omni, warm_latents)
    td1 = time.perf_counter()
    T["warm_decode_s"] = td1 - td0
    # single warm request time-to-completion: gen[1] + decode of just that one latent
    sd0 = time.perf_counter()
    _ = E._decode_with_worker(omni, E._extract_latents(outputs[1]))
    sd1 = time.perf_counter()
    T["warm_single_request_decode_s"] = sd1 - sd0
    T["warm_single_request_time_to_completion_s"] = T["warm_gen_times_s"][0] + T["warm_single_request_decode_s"]

    mp4 = OUT / "dreamzero_full_tp4_c4567_cfgoff_inductor_offload_warm.mp4"
    E._write_mp4(mp4, frames, fps=5)

    acts = [np.asarray(o.multimodal_output.get("actions")) if getattr(o, "multimodal_output", None) and o.multimodal_output.get("actions") is not None else None for o in outputs]
    print("=========== RESULTS ===========", flush=True)
    print("FRAMES_SHAPE=%s dtype=%s min=%d max=%d" % (tuple(frames.shape), frames.dtype, int(frames.min()), int(frames.max())), flush=True)
    for i, a in enumerate(acts):
        if a is not None:
            print("ACTION[%d] shape=%s min=%.4f max=%.4f mean=%.4f finite=%s" % (i, tuple(a.shape), float(a.min()), float(a.max()), float(a.mean()), bool(np.isfinite(a).all())), flush=True)
        else:
            print("ACTION[%d]=None" % i, flush=True)
    print("MP4=%s exists=%s bytes=%d" % (mp4, mp4.exists(), mp4.stat().st_size if mp4.exists() else 0), flush=True)
    print("--- TIMING SUMMARY ---", flush=True)
    print("MODEL_LOAD_S=%.3f" % T["model_load_s"], flush=True)
    print("COLD_GEN0_S=%.3f" % T["cold_gen0_s"], flush=True)
    print("WARM_GEN_TIMES_S=%s" % [round(x,3) for x in T["warm_gen_times_s"]], flush=True)
    print("WARM_GEN_MEAN_S=%.3f" % T["warm_gen_mean_s"], flush=True)
    print("WARM_DECODE_S=%.3f" % T["warm_decode_s"], flush=True)
    print("WARM_SINGLE_REQUEST_TIME_TO_COMPLETION_S=%.3f" % T["warm_single_request_time_to_completion_s"], flush=True)

    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_OUT, "w") as f:
        json.dump(T, f, indent=2)
    omni.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
