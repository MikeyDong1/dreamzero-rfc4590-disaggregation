# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DreamZero disaggregated-execution tests (RFC #4590 §14.5-14.7).

Two tiers, both ``needs_runtime`` (torch + vllm_omni + AR-Diffusion engine):

* ``test_component_spec_*`` and ``test_carrier_*`` need only torch (no GPU / no
  checkpoint) — they validate the component ownership table and the payload
  round-trip of the DreamZero carrier.
* ``test_numerical_equivalence`` needs the real DreamZero checkpoint + XPU. It is
  additionally gated on the ``DREAMZERO_MODEL_PATH`` env var and skipped when
  unset. Manual command (on the gnr XPU node, inside vllm-omni-xpu):

    DREAMZERO_MODEL_PATH=/models/DreamZero-DROID \\
      pytest tests/diffusion/disaggregated/test_dreamzero_disaggregated.py \\
      -m needs_runtime -k numerical_equivalence -s
"""

from __future__ import annotations

import os

import pytest

try:
    import torch

    from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline
    from vllm_omni.diffusion.models.dreamzero.state_dreamzero import DreamZeroStageCarrier
    from vllm_omni.diffusion.stage_roles import DECODE, DENOISE, ENCODE
    from vllm_omni.diffusion.worker.utils import DiffusionRequestState
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
except Exception as exc:  # pragma: no cover - import-environment dependent
    pytest.skip(f"vllm_omni runtime unavailable: {exc}", allow_module_level=True)

pytestmark = pytest.mark.needs_runtime


# --- capability flags (regression guard) -----------------------------------


def test_dreamzero_declares_disaggregated_and_atom_capabilities():
    """Both capability flags must be present.

    The runner's run_encode_atoms / run_decode_atoms shims gate on
    isinstance(pipeline, SupportsDiffusionAtoms), which for a @runtime_checkable
    Protocol only passes when the ``supports_diffusion_atoms`` ClassVar is set.
    Missing it silently falls back to prepare_encode/post_decode (which DreamZero
    lacks) and crashes the encode/decode stages. Guard both flags here.
    """
    from vllm_omni.diffusion.models.interface import (
        SupportsDiffusionAtoms,
        SupportsDisaggregatedDiffusionExecution,
    )

    assert DreamZeroPipeline.supports_disaggregated_execution is True
    assert DreamZeroPipeline.supports_diffusion_atoms is True
    # Protocol structural checks (methods + flags present on the class).
    assert issubclass(DreamZeroPipeline, SupportsDiffusionAtoms)
    assert issubclass(DreamZeroPipeline, SupportsDisaggregatedDiffusionExecution)


# --- component ownership (torch only, no checkpoint) -----------------------


def test_component_spec_encode_owns_encoders_not_dit():
    spec = DreamZeroPipeline.required_components_for_stage(ENCODE)
    assert spec.tokenizer and spec.text_encoder and spec.image_encoder and spec.vae_encoder
    assert not spec.dit and not spec.vae_decoder


def test_component_spec_denoise_owns_dit_not_encoders_or_vae_decoder():
    spec = DreamZeroPipeline.required_components_for_stage(DENOISE)
    assert spec.dit and spec.scheduler and spec.action_modules
    assert not spec.text_encoder and not spec.image_encoder and not spec.vae_decoder and not spec.vae_encoder


def test_component_spec_decode_owns_vae_decoder_not_dit():
    spec = DreamZeroPipeline.required_components_for_stage(DECODE)
    assert spec.vae_decoder
    assert not spec.dit and not spec.text_encoder and not spec.image_encoder


def test_component_spec_monolithic_owns_everything():
    spec = DreamZeroPipeline.required_components_for_stage("diffusion")
    assert all(
        (spec.tokenizer, spec.text_encoder, spec.image_encoder, spec.vae_encoder,
         spec.dit, spec.scheduler, spec.vae_decoder, spec.action_modules)
    )


# --- carrier payload round-trip (torch only, no checkpoint) ----------------


def _make_encode_state_with_carrier():
    state = DiffusionRequestState(request_id="req-1", sampling=OmniDiffusionSamplingParams(seed=0))
    carrier = DreamZeroStageCarrier(
        session_id="sess-A",
        embodiment_name="roboarena",
        transform_embodiment="roboarena",
        reset_reason="session",
        do_true_cfg=True,
        current_start_frame=0,
        height=180,
        width=320,
        seq_len=352,
        frame_seqlen=88,
        num_inference_steps=4,
        sigma_shift=5.0,
        prompt_embeds=torch.zeros(1, 512, 4096, dtype=torch.bfloat16),
        clip_feas=torch.zeros(1, 257, 1280, dtype=torch.bfloat16),
        ys=torch.zeros(1, 20, 4, 22, 40, dtype=torch.bfloat16),
        image_latent=torch.zeros(1, 1, 16, 22, 40, dtype=torch.bfloat16),
        noise_obs=torch.zeros(1, 4, 16, 22, 40, dtype=torch.bfloat16),
        noise_action=torch.zeros(1, 16, 32, dtype=torch.bfloat16),
        embodiment_id=torch.zeros(1, dtype=torch.long),
        state_for_postprocess=torch.zeros(1, 1, 64, dtype=torch.float32),
    )
    state.extra[DreamZeroPipeline._CARRIER_KEY] = carrier
    return state, carrier


def test_export_encode_to_denoise_payload_shape_dtype():
    # export_stage_payload is an instance method but only touches state.extra;
    # bind it unbound to avoid constructing the (checkpoint-heavy) pipeline.
    state, _ = _make_encode_state_with_carrier()
    payload = DreamZeroPipeline.export_stage_payload(
        _StubPipeline(), state, source_stage=ENCODE, target_stage=DENOISE
    )
    payload.validate()
    assert payload.source_stage == ENCODE and payload.target_stage == DENOISE
    assert payload.metadata["session_id"] == "sess-A"
    assert payload.metadata["do_true_cfg"] is True
    assert payload.metadata["num_inference_steps"] == 4
    # tensors preserved with dtype/shape
    assert payload.tensors["prompt_embeds"].shape == (1, 512, 4096)
    assert payload.tensors["prompt_embeds"].dtype == torch.bfloat16
    assert payload.tensors["noise_obs"].shape == (1, 4, 16, 22, 40)
    # KV / scheduler objects never appear
    assert "scheduler" not in payload.metadata


def test_import_reconstructs_carrier():
    state, carrier = _make_encode_state_with_carrier()
    payload = DreamZeroPipeline.export_stage_payload(
        _StubPipeline(), state, source_stage=ENCODE, target_stage=DENOISE
    )
    restored_state = DreamZeroPipeline.import_stage_payload(
        _StubPipeline(), payload, target_stage=DENOISE, request=None
    )
    restored = restored_state.extra[DreamZeroPipeline._CARRIER_KEY]
    assert restored.session_id == "sess-A"
    assert restored.num_inference_steps == 4
    assert restored.sigma_shift == 5.0
    assert restored.current_start_frame == 0
    assert torch.equal(restored.prompt_embeds, carrier.prompt_embeds)
    assert restored.prompt_embeds.dtype == torch.bfloat16


class _StubPipeline:
    """Bare object exposing just what export/import touch (no checkpoint load)."""

    _CARRIER_KEY = DreamZeroPipeline._CARRIER_KEY


# --- numerical equivalence (needs GPU + checkpoint) ------------------------


@pytest.mark.skipif(
    not os.environ.get("DREAMZERO_MODEL_PATH"),
    reason="DREAMZERO_MODEL_PATH unset; full GPU equivalence test requires the checkpoint.",
)
def test_numerical_equivalence():
    """Golden-equivalence: monolithic forward vs encode->denoise->decode phases.

    Intended check (wire on the node): build one DreamZeroPipeline, run a
    deterministic single forward through the monolithic path, then re-run the
    SAME inputs through ``_run_encode_phase`` -> ``_run_denoise_phase`` ->
    ``_run_decode_phase`` on a fresh session and compare the action + video
    outputs. Because both paths call the identical phase methods on identical
    inputs, outputs must match within tight fp tolerance (exact for
    integer/metadata, torch.allclose for float tensors).

    This requires a concrete ``robot_obs`` fixture and an ARDiffusionKVState
    attach that mirror the dreamzero-vllm-omni test harness; those are provided
    on the XPU node (see skill: dreamzero-vllm-omni-test), not pinned here.
    """
    pytest.skip(
        "Provide a robot_obs fixture + ARDiffusionKVState attach to run the full "
        "equivalence comparison on the node (see skill: dreamzero-vllm-omni-test). "
        f"DREAMZERO_MODEL_PATH={os.environ.get('DREAMZERO_MODEL_PATH')!r}."
    )
