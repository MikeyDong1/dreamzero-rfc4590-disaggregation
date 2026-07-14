# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavioral runner tests for disaggregated stages (RFC #4590 §14.3).

These import the real ``DiffusionModelRunner`` and drive it with fake pipelines
and lightweight tensors, so they require torch + the vllm_omni runtime. They are
marked ``needs_runtime`` and skipped where those deps are unavailable (e.g. the
Windows dev host); run them on the XPU node inside the vllm-omni-xpu container:

    pytest tests/diffusion/disaggregated/test_runner_stages.py -m needs_runtime
"""

from __future__ import annotations

import types

import pytest

# The full runner path needs torch + the vllm_omni runtime. Any failure to
# import them (absent, or a broken partial install as on the Windows dev host)
# skips the whole module rather than erroring collection.
try:
    import torch

    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.stage_payload import DiffusionStagePayload
    from vllm_omni.diffusion.stage_roles import DECODE, DENOISE, ENCODE, StageComponentSpec
    from vllm_omni.diffusion.worker.diffusion_model_runner import (
        STAGE_PAYLOAD_OUTPUT_KEY,
        STAGE_PAYLOAD_PROMPT_KEY,
        DiffusionModelRunner,
    )
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
except Exception as exc:  # ImportError, or torch DLL load failure
    pytest.skip(f"vllm_omni runtime unavailable: {exc}", allow_module_level=True)

pytestmark = pytest.mark.needs_runtime


class FakeDisaggregatedPipeline:
    """Minimal pipeline implementing the disaggregated + atom protocols.

    Records which hooks ran so tests can assert a stage never crosses into
    another stage's work (no DiT on encode/decode, no encode on denoise, etc.).
    """

    supports_disaggregated_execution = True
    supports_diffusion_atoms = True

    def __init__(self):
        self.calls: list[str] = []
        self.enable_diffusion_pipeline_profiler = False

    # --- atom protocol (encode) ---
    def check_inputs(self, state, **kw):
        self.calls.append("check_inputs")
        return state

    def encode_conditions(self, state, **kw):
        self.calls.append("encode_conditions")
        state.prompt_embeds = torch.zeros(1, 4)
        return state

    def prepare_latents_and_timesteps(self, state, **kw):
        self.calls.append("prepare_latents_and_timesteps")
        state.latents = torch.zeros(1, 16)
        state.timesteps = torch.arange(4)
        return state

    # --- atom protocol (decode) ---
    def decode_latents(self, state, **kw):
        self.calls.append("decode_latents")
        return state

    def postprocess_outputs(self, state, **kw):
        self.calls.append("postprocess_outputs")
        return DiffusionOutput(output={"image": torch.zeros(3, 8, 8)})

    # --- denoise ---
    def run_denoise(self, state, **kw):
        self.calls.append("run_denoise")
        state.latents = torch.ones(1, 16)

    # --- disaggregated payload hooks ---
    def export_stage_payload(self, state, *, source_stage, target_stage):
        self.calls.append(f"export:{source_stage}->{target_stage}")
        tensors = {}
        if state.latents is not None:
            tensors["latents"] = state.latents
        if state.prompt_embeds is not None:
            tensors["prompt_embeds"] = state.prompt_embeds
        return DiffusionStagePayload.create(
            request_id=state.request_id,
            source_stage=source_stage,
            target_stage=target_stage,
            payload_type=f"{source_stage}_to_{target_stage}",
            tensors=tensors,
            metadata={"session_id": "sess"},
        )

    def import_stage_payload(self, payload, *, target_stage, request=None):
        self.calls.append(f"import:{target_stage}")
        from vllm_omni.diffusion.worker.utils import DiffusionRequestState

        state = DiffusionRequestState(
            request_id=payload.request_id,
            sampling=OmniDiffusionSamplingParams(),
        )
        if "latents" in payload.tensors:
            state.latents = payload.tensors["latents"]
        return state

    @classmethod
    def required_components_for_stage(cls, model_stage):
        if model_stage == ENCODE:
            return StageComponentSpec(tokenizer=True, text_encoder=True, vae_encoder=True)
        if model_stage == DENOISE:
            return StageComponentSpec(dit=True, scheduler=True)
        if model_stage == DECODE:
            return StageComponentSpec(vae_decoder=True)
        return StageComponentSpec()


def _make_runner(model_stage):
    runner = object.__new__(DiffusionModelRunner)
    runner.od_config = types.SimpleNamespace(
        model_stage=model_stage,
        model_class_name="FakeDisaggregatedPipeline",
        parallel_config=types.SimpleNamespace(use_hsdp=False),
        stage_id=0,
    )
    runner.device = torch.device("cpu")
    runner.vllm_config = None
    runner.pipeline = FakeDisaggregatedPipeline()
    runner.state_cache = {}
    return runner


def _request(prompt):
    return OmniDiffusionRequest(
        prompt=prompt,
        sampling_params=OmniDiffusionSamplingParams(seed=0),
        request_id="req-1",
    )


def test_encode_stage_runs_only_encode_atoms():
    runner = _make_runner(ENCODE)
    out = runner.execute_encode_stage(_request({"prompt": "hi"}))
    calls = runner.pipeline.calls
    assert "check_inputs" in calls and "encode_conditions" in calls
    assert "run_denoise" not in calls and "decode_latents" not in calls
    payload = out.custom_output[STAGE_PAYLOAD_OUTPUT_KEY]
    assert payload.source_stage == ENCODE and payload.target_stage == DENOISE
    # state released
    assert "req-1" not in runner.state_cache


def test_denoise_stage_imports_payload_and_does_not_encode():
    runner = _make_runner(DENOISE)
    enc_payload = DiffusionStagePayload.create(
        request_id="req-1",
        source_stage=ENCODE,
        target_stage=DENOISE,
        payload_type="encode_to_denoise",
        tensors={"latents": torch.zeros(1, 16)},
        metadata={"session_id": "sess"},
    )
    req = _request({"prompt": "", "extra": {STAGE_PAYLOAD_PROMPT_KEY: enc_payload}})
    out = runner.execute_denoise_stage(req)
    calls = runner.pipeline.calls
    assert "import:denoise" in calls and "run_denoise" in calls
    assert "check_inputs" not in calls and "encode_conditions" not in calls
    payload = out.custom_output[STAGE_PAYLOAD_OUTPUT_KEY]
    assert payload.source_stage == DENOISE and payload.target_stage == DECODE


def test_decode_stage_runs_only_decode():
    runner = _make_runner(DECODE)
    den_payload = DiffusionStagePayload.create(
        request_id="req-1",
        source_stage=DENOISE,
        target_stage=DECODE,
        payload_type="denoise_to_decode",
        tensors={"latents": torch.ones(1, 16)},
        metadata={"session_id": "sess"},
    )
    req = _request({"prompt": "", "extra": {STAGE_PAYLOAD_PROMPT_KEY: den_payload}})
    out = runner.execute_decode_stage(req)
    calls = runner.pipeline.calls
    assert "decode_latents" in calls and "postprocess_outputs" in calls
    assert "run_denoise" not in calls
    assert out.output is not None and "image" in out.output


def test_denoise_rejects_wrong_transition():
    runner = _make_runner(DENOISE)
    wrong = DiffusionStagePayload.create(
        request_id="req-1",
        source_stage=DENOISE,
        target_stage=DECODE,
        payload_type="denoise_to_decode",
        tensors={"latents": torch.ones(1, 16)},
    )
    req = _request({"prompt": "", "extra": {STAGE_PAYLOAD_PROMPT_KEY: wrong}})
    with pytest.raises(Exception, match="transition"):
        runner.execute_denoise_stage(req)


def test_missing_payload_raises_actionable_error():
    runner = _make_runner(DENOISE)
    req = _request({"prompt": "no payload here"})
    with pytest.raises(Exception, match="without a DiffusionStagePayload"):
        runner.execute_denoise_stage(req)
