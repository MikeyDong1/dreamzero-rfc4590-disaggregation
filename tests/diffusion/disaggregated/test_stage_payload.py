# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the DiffusionStagePayload transport envelope (RFC #4590 §14.1)."""

from __future__ import annotations

import pytest


def test_create_valid_payload(stage_payload):
    p = stage_payload.DiffusionStagePayload.create(
        request_id="req-1",
        source_stage="encode",
        target_stage="denoise",
        payload_type="encode_to_denoise",
        metadata={"session_id": "s0", "num_steps": 4, "cfg": 1.5, "shape": [1, 16, 4]},
    )
    assert p.schema_version == stage_payload.DIFFUSION_STAGE_PAYLOAD_SCHEMA_VERSION
    assert p.request_id == "req-1"
    assert p.source_stage == "encode"
    assert p.target_stage == "denoise"
    assert p.metadata["session_id"] == "s0"
    # frozen dataclass
    with pytest.raises(Exception):
        p.request_id = "other"  # type: ignore[misc]


def test_schema_version_validation(stage_payload):
    p = stage_payload.DiffusionStagePayload(
        schema_version=999,
        request_id="r",
        source_stage="encode",
        target_stage="denoise",
        payload_type="x",
    )
    with pytest.raises(stage_payload.StagePayloadError, match="schema_version"):
        p.validate()


def test_empty_request_id_rejected(stage_payload):
    with pytest.raises(stage_payload.StagePayloadError, match="request_id"):
        stage_payload.DiffusionStagePayload.create(
            request_id="",
            source_stage="encode",
            target_stage="denoise",
            payload_type="x",
        )


@pytest.mark.parametrize("field_name", ["source_stage", "target_stage", "payload_type"])
def test_empty_role_fields_rejected(stage_payload, field_name):
    kwargs = dict(
        request_id="r",
        source_stage="encode",
        target_stage="denoise",
        payload_type="x",
    )
    kwargs[field_name] = ""
    with pytest.raises(stage_payload.StagePayloadError, match=field_name):
        stage_payload.DiffusionStagePayload.create(**kwargs)


def test_request_id_preserved_through_transition_helpers(stage_payload):
    p = stage_payload.DiffusionStagePayload.create(
        request_id="keep-me",
        source_stage="denoise",
        target_stage="decode",
        payload_type="denoise_to_decode",
    )
    p.expect_transition(source="denoise", target="decode")
    assert p.request_id == "keep-me"
    with pytest.raises(stage_payload.StagePayloadError, match="expected"):
        p.expect_transition(source="encode", target="denoise")


def test_tensor_dtype_shape_preserved(stage_payload, fake_tensor_cls):
    t = fake_tensor_cls(shape=(1, 16, 4, 22, 40), dtype="bfloat16")
    p = stage_payload.DiffusionStagePayload.create(
        request_id="r",
        source_stage="encode",
        target_stage="denoise",
        payload_type="encode_to_denoise",
        tensors={"latents": t},
    )
    out = p.tensors["latents"]
    assert out.shape == (1, 16, 4, 22, 40)
    assert out.dtype == "bfloat16"


def test_sanitize_detaches_and_moves_to_host(stage_payload, fake_tensor_cls):
    t = fake_tensor_cls(data_ptr=500)
    # sanitize path: detach -> cpu -> contiguous; clone only if ptr unchanged.
    out = stage_payload.sanitize_transport_tensor(t)
    assert "detach" in t.calls and "cpu" in t.calls and "contiguous" in t.calls
    # data_ptr is unchanged in the fake, so a clone must have been forced.
    assert out is not t


def test_non_tensor_in_tensors_rejected(stage_payload):
    with pytest.raises(stage_payload.NonTransportableValueError, match="tensors"):
        stage_payload.DiffusionStagePayload.create(
            request_id="r",
            source_stage="encode",
            target_stage="denoise",
            payload_type="x",
            tensors={"bad": [1, 2, 3]},
        )


def test_tensor_in_metadata_rejected(stage_payload, fake_tensor_cls):
    with pytest.raises(stage_payload.NonTransportableValueError, match="tensors' dict"):
        stage_payload.DiffusionStagePayload.create(
            request_id="r",
            source_stage="encode",
            target_stage="denoise",
            payload_type="x",
            metadata={"latents": fake_tensor_cls()},
        )


def test_scheduler_object_in_metadata_rejected(stage_payload):
    class FlowUniPCMultistepScheduler:  # name matches the non-transportable guard
        pass

    with pytest.raises(stage_payload.NonTransportableValueError, match="scheduler"):
        stage_payload.DiffusionStagePayload.create(
            request_id="r",
            source_stage="encode",
            target_stage="denoise",
            payload_type="x",
            metadata={"sched": FlowUniPCMultistepScheduler()},
        )


def test_callable_in_metadata_rejected(stage_payload):
    with pytest.raises(stage_payload.NonTransportableValueError):
        stage_payload.DiffusionStagePayload.create(
            request_id="r",
            source_stage="encode",
            target_stage="denoise",
            payload_type="x",
            metadata={"fn": lambda x: x},
        )


def test_generator_like_object_rejected(stage_payload):
    class _FakeTorchGenerator:
        pass

    _FakeTorchGenerator.__module__ = "torch"
    _FakeTorchGenerator.__name__ = "Generator"

    with pytest.raises(stage_payload.NonTransportableValueError, match="Generator"):
        stage_payload.DiffusionStagePayload.create(
            request_id="r",
            source_stage="encode",
            target_stage="denoise",
            payload_type="x",
            metadata={"gen": _FakeTorchGenerator()},
        )


def test_process_local_state_object_rejected(stage_payload):
    class DreamZeroState:
        pass

    with pytest.raises(stage_payload.NonTransportableValueError, match="process-local"):
        stage_payload.DiffusionStagePayload.create(
            request_id="r",
            source_stage="encode",
            target_stage="denoise",
            payload_type="x",
            metadata={"state": DreamZeroState()},
        )


def test_nested_metadata_allowed(stage_payload):
    p = stage_payload.DiffusionStagePayload.create(
        request_id="r",
        source_stage="encode",
        target_stage="denoise",
        payload_type="x",
        metadata={"nested": {"a": [1, 2, {"b": None, "c": b"bytes"}], "d": True}},
    )
    assert p.metadata["nested"]["d"] is True


def test_require_tensors(stage_payload, fake_tensor_cls):
    p = stage_payload.DiffusionStagePayload.create(
        request_id="r",
        source_stage="encode",
        target_stage="denoise",
        payload_type="x",
        tensors={"latents": fake_tensor_cls()},
    )
    p.require_tensors("latents")
    with pytest.raises(stage_payload.StagePayloadError, match="missing required tensors"):
        p.require_tensors("latents", "prompt_embeds")


def test_summary_has_no_tensor_contents(stage_payload, fake_tensor_cls):
    p = stage_payload.DiffusionStagePayload.create(
        request_id="r",
        source_stage="encode",
        target_stage="denoise",
        payload_type="encode_to_denoise",
        tensors={"latents": fake_tensor_cls(shape=(1, 16), dtype="bfloat16")},
        metadata={"session_id": "s0"},
    )
    s = p.summary()
    assert "latents" in s and "(1, 16)" in s and "session_id" in s
