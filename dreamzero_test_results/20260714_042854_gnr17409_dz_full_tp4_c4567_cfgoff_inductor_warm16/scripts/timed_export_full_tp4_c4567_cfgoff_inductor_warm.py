#!/usr/bin/env python3
"""FULL/monolithic DreamZero: TP=4 on cards 4,5,6,7, CFG off, inductor ON,
denoise_steps=16, raw/unencoded camera MP4 input. WARM measurement only:
request 0 is discarded (cold load + first-call inductor compile); requests
1 and 2 are the reported warm steady-state window.
"""
import time, uuid, sys, json
from pathlib import Path
import numpy as np, torch

sys.argv = ["x"]
sys.path.insert(0, "/workspace/vllm-omni/examples/offline_inference/dreamzero")
import export_prediction_video as E
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = "GEAR-Dreams/DreamZero-DROID"
DEPLOY = Path("/workspace/config_run/dreamzero_tp4_c4567_cfgoff_inductor.yaml")
VIDEO_DIR = Path("/workspace/assets_run")
OUT = Path("/workspace/output_run")
METRICS_OUT = Path("/workspace/metrics_run/metrics.json")
PROMPT = "Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan"
NUM_REQUESTS = 3  # 0=cold(load+compile), 1 & 2 = warm steady-state


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sid = "full-tp4-c4567-cfgoff-inductor-" + str(uuid.uuid4())
    T = {}

    camera_frames, base_observations = E._build_observations(video_dir=VIDEO_DIR, prompt=PROMPT, session_id=sid)
    # _build_observations returns exactly 2 distinct frame-window observations;
    # replay the second one again for a 3rd request to get 2 warm samples.
    observations = base_observations + [base_observations[-1]]
    print("[timing] obs_built n=%d" % len(observations), flush=True)

    t_load0 = time.perf_counter()
    omni = Omni(model=MODEL, deploy_config=str(DEPLOY), enforce_eager=False, worker_extension_cls=E.WORKER_EXTENSION)
    t_load1 = time.perf_counter()
    T["model_load_s"] = t_load1 - t_load0
    print("[timing] MODEL_LOADED_S=%.3f" % T["model_load_s"], flush=True)

    outputs = []
    gen_times = []
    for index, obs in enumerate(observations[:NUM_REQUESTS]):
        sp = OmniDiffusionSamplingParams(extra_args={"reset": index == 0, "session_id": obs["session_id"], "robot_obs": obs})
        g0 = time.perf_counter()
        result = omni.generate(obs["prompt"], sampling_params_list=[sp])
        g1 = time.perf_counter()
        gen_times.append(g1 - g0)
        if not result:
            raise RuntimeError("No output for request %d" % index)
        outputs.append(result[0])
        tag = "COLD" if index == 0 else "WARM"
        print("[timing] %s_gen%d_s=%.3f" % (tag, index, g1 - g0), flush=True)

    T["cold_gen0_s"] = gen_times[0]
    T["warm_gen_times_s"] = gen_times[1:]
    T["warm_gen_mean_s"] = sum(gen_times[1:]) / len(gen_times[1:])

    t_decode0 = time.perf_counter()
    # Decode only the warm requests' latents for the reported warm completion time.
    warm_latent_steps = [E._extract_latents(o) for o in outputs[1:]]
    warm_full_latents = torch.cat(warm_latent_steps, dim=2)
    frames = E._decode_with_worker(omni, warm_full_latents)
    t_decode1 = time.perf_counter()
    T["warm_decode_s"] = t_decode1 - t_decode0
    T["warm_time_to_completion_s"] = T["warm_gen_mean_s"] * len(gen_times[1:]) + T["warm_decode_s"]
    # Also report a strict single-request warm time-to-completion (gen[1] + full decode of just that latent)
    single_warm_latents = E._extract_latents(outputs[1])
    t_sd0 = time.perf_counter()
    _ = E._decode_with_worker(omni, single_warm_latents)
    t_sd1 = time.perf_counter()
    T["warm_single_request_decode_s"] = t_sd1 - t_sd0
    T["warm_single_request_time_to_completion_s"] = T["warm_gen_times_s"][0] + T["warm_single_request_decode_s"]

    mp4 = OUT / "dreamzero_full_tp4_c4567_cfgoff_inductor_warm.mp4"
    E._write_mp4(mp4, frames, fps=5)

    acts = []
    for o in outputs:
        a = o.multimodal_output.get("actions") if getattr(o, "multimodal_output", None) else None
        acts.append(np.asarray(a) if a is not None else None)

    print("=========== RESULTS ===========", flush=True)
    print("FRAMES_SHAPE=%s dtype=%s min=%d max=%d" % (tuple(frames.shape), frames.dtype, int(frames.min()), int(frames.max())), flush=True)
    for i, a in enumerate(acts):
        if a is not None:
            print("ACTION[%d] shape=%s dtype=%s min=%.4f max=%.4f mean=%.4f std=%.4f finite=%s" % (
                i, tuple(a.shape), a.dtype, float(a.min()), float(a.max()), float(a.mean()), float(a.std()), bool(np.isfinite(a).all())
            ), flush=True)
        else:
            print("ACTION[%d]=None" % i, flush=True)
    print("MP4=%s exists=%s bytes=%d" % (mp4, mp4.exists(), mp4.stat().st_size if mp4.exists() else 0), flush=True)
    print("--- TIMING SUMMARY ---", flush=True)
    print("MODEL_LOAD_S=%.3f" % T["model_load_s"], flush=True)
    print("COLD_GEN0_S=%.3f" % T["cold_gen0_s"], flush=True)
    print("WARM_GEN_TIMES_S=%s" % [round(x, 3) for x in T["warm_gen_times_s"]], flush=True)
    print("WARM_GEN_MEAN_S=%.3f" % T["warm_gen_mean_s"], flush=True)
    print("WARM_DECODE_S=%.3f" % T["warm_decode_s"], flush=True)
    print("WARM_TIME_TO_COMPLETION_S=%.3f" % T["warm_time_to_completion_s"], flush=True)
    print("WARM_SINGLE_REQUEST_TIME_TO_COMPLETION_S=%.3f" % T["warm_single_request_time_to_completion_s"], flush=True)

    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_OUT, "w") as f:
        json.dump(T, f, indent=2)

    omni.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
