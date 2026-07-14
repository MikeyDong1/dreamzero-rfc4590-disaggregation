#!/usr/bin/env python3
import asyncio
import time, uuid, sys
from pathlib import Path
import numpy as np, torch

sys.argv = ["x"]
import export_prediction_video as E
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = "GEAR-Dreams/DreamZero-DROID"
DEPLOY = Path("/workspace/vllm-omni/vllm_omni/deploy/dreamzero_disaggregated_tp4denoise_cfgoff.yaml")
VIDEO_DIR = Path("/workspace/vllm-omni/outputs/dreamzero/assets")
OUT = Path("/workspace/vllm-omni/outputs/dreamzero/generated_predictions_disagg_tp4denoise_cfgoff")
PROMPT = "Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan"

DECODE_STAGE_ID = 2


def _decode_with_worker_disagg(omni, full_latents):
    stage_client = omni.engine.stage_clients[DECODE_STAGE_ID]
    engine = getattr(stage_client, "_engine", None)
    if engine is not None:
        decoded = engine.executor.collective_rpc(
            "decode_video_latents_to_uint8",
            args=(full_latents,),
            unique_reply_rank=0,
            exec_all_ranks=True,
        )
    else:
        decoded = asyncio.run(
            stage_client.collective_rpc_async(
                "decode_video_latents_to_uint8",
                args=(full_latents,),
            )
        )
        if isinstance(decoded, list):
            decoded = decoded[0]
    if isinstance(decoded, torch.Tensor):
        decoded = decoded.numpy()
    if not isinstance(decoded, np.ndarray):
        raise TypeError(f"Unexpected decoded output type: {type(decoded)!r}")
    return decoded


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sid = "timed-disagg-cfgoff-" + str(uuid.uuid4())
    T = {}

    camera_frames, observations = E._build_observations(video_dir=VIDEO_DIR, prompt=PROMPT, session_id=sid)
    print("[timing] obs_built n=%d" % len(observations), flush=True)

    t_load0 = time.perf_counter()
    omni = Omni(model=MODEL, deploy_config=str(DEPLOY), enforce_eager=True, worker_extension_cls=E.WORKER_EXTENSION)
    t_load1 = time.perf_counter()
    T["model_load_s"] = t_load1 - t_load0
    print("[timing] MODEL_LOADED_S=%.3f" % T["model_load_s"], flush=True)
    print("[info] num_stages=%d" % omni.num_stages, flush=True)

    outputs = []
    gen_times = []
    for index, obs in enumerate(observations):
        encode_sp = OmniDiffusionSamplingParams(
            extra_args={"reset": index == 0, "session_id": obs["session_id"], "robot_obs": obs}
        )
        stage_params = [encode_sp] + [OmniDiffusionSamplingParams() for _ in range(omni.num_stages - 1)]

        if index == 1:
            print("[timing] WARM_WINDOW_START_UNIX=%.3f" % time.time(), flush=True)

        g0 = time.perf_counter()
        result = omni.generate(obs["prompt"], sampling_params_list=stage_params)
        g1 = time.perf_counter()
        gen_times.append(g1 - g0)
        if index == 0:
            T["time_to_first_output_s"] = g1 - t_load1
            print("[timing] COLD_gen0_s=%.3f" % (g1 - g0), flush=True)
        else:
            T["warm_gen_s"] = g1 - g0
            print("[timing] WARM_gen%d_s=%.3f" % (index, g1 - g0), flush=True)
            print("[timing] WARM_WINDOW_END_UNIX=%.3f" % time.time(), flush=True)
        if not result:
            raise RuntimeError("No output for request %d" % index)
        out0 = result[0]
        print(
            "[debug] request %d output: images_len=%s multimodal_output_keys=%s"
            % (
                index,
                len(out0.images) if getattr(out0, "images", None) is not None else None,
                list(out0.multimodal_output.keys()) if getattr(out0, "multimodal_output", None) else None,
            ),
            flush=True,
        )
        outputs.append(out0)

    t_decode0 = time.perf_counter()
    t_decode0_unix = time.time()
    latent_steps = [E._extract_latents(o) for o in outputs]
    full_latents = torch.cat(latent_steps, dim=2)
    frames = _decode_with_worker_disagg(omni, full_latents)
    t_decode1 = time.perf_counter()
    t_decode1_unix = time.time()
    T["decode_s"] = t_decode1 - t_decode0
    T["warm_time_to_completion_s"] = T["warm_gen_s"] + T["decode_s"]
    print("[timing] DECODE_WINDOW_START_UNIX=%.3f" % t_decode0_unix, flush=True)
    print("[timing] DECODE_WINDOW_END_UNIX=%.3f" % t_decode1_unix, flush=True)

    mp4 = OUT / "timed_prediction_disagg_cfgoff.mp4"
    E._write_mp4(mp4, frames, fps=5)
    gif = OUT / "timed_prediction_disagg_cfgoff.gif"
    E._write_gif(gif, frames, fps=5)

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
    print("GIF=%s exists=%s bytes=%d" % (gif, gif.exists(), gif.stat().st_size if gif.exists() else 0), flush=True)
    print("--- TIMING SUMMARY ---", flush=True)
    print("MODEL_LOAD_S=%.3f" % T["model_load_s"], flush=True)
    print("COLD_GEN0_S=%.3f" % gen_times[0], flush=True)
    print("WARM_GEN1_S=%.3f" % T["warm_gen_s"], flush=True)
    print("DECODE_S=%.3f" % T["decode_s"], flush=True)
    print("WARM_TIME_TO_COMPLETION_S=%.3f" % T["warm_time_to_completion_s"], flush=True)
    omni.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
