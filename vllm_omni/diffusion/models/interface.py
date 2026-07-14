# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    import torch

    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.stage_payload import DiffusionStagePayload
    from vllm_omni.diffusion.stage_roles import StageComponentSpec
    from vllm_omni.diffusion.worker.input_batch import InputBatch
    from vllm_omni.diffusion.worker.utils import DiffusionRequestState


@runtime_checkable
class SupportImageInput(Protocol):
    support_image_input: ClassVar[bool] = True
    color_format: ClassVar[str] = "RGB"  # Default color format


@dataclass(frozen=True)
class ReferenceVideoDecodeSpec:
    max_frames: int | None = None
    keep: Literal["first", "last"] = "first"


@runtime_checkable
class SupportAudioInput(Protocol):
    support_audio_input: ClassVar[bool] = True


@runtime_checkable
class SupportAudioOutput(Protocol):
    support_audio_output: ClassVar[bool] = True


@runtime_checkable
class SupportsStepExecution(Protocol):
    """State-driven step-level execution protocol for diffusion pipelines.

    Pipelines should split request-level ``forward()`` into:
    ``prepare_encode()`` (one-time request setup), ``denoise_step()``
    (one denoise forward), ``step_scheduler()`` (one scheduler update),
    and ``post_decode()`` (final decode).
    """

    supports_step_execution: ClassVar[bool] = True

    def prepare_encode(self, state: DiffusionRequestState, **kwargs: Any) -> DiffusionRequestState:
        """Prepare request-level inputs and return initialized state."""
        ...

    def denoise_step(self, input_batch: InputBatch, **kwargs: Any) -> torch.Tensor | None:
        """Run one denoise forward on the runner-assembled batch."""
        ...

    def step_scheduler(self, state: DiffusionRequestState, noise_pred: torch.Tensor, **kwargs: Any) -> None:
        """Run one scheduler step."""
        ...

    def post_decode(self, state: DiffusionRequestState, **kwargs: Any) -> DiffusionOutput:
        """Decode output after denoise loop or at a partial chunk boundary."""
        ...


@runtime_checkable
class SupportsComponentDiscovery(Protocol):
    """Declares which submodules serve as pipeline components.

    Used by the framework to locate DiT, encoder, and VAE modules for
    CPU offload, HSDP sharding, and other operations that need to know
    the pipeline's internal structure.

    All attribute names support dotted paths for nested submodules
    (e.g. ``"pipe.transformer"``).

    Attributes:
        _dit_modules: Denoising submodules (on GPU during diffusion).
        _encoder_modules: Encoder submodules (offloaded during diffusion).
        _vae_modules: VAE(s) (always on GPU).
        _resident_modules: Extra modules pinned on GPU during layerwise
            offloading.  Optional, defaults to ``[]``.
    """

    _dit_modules: ClassVar[list[str]]
    _encoder_modules: ClassVar[list[str]]
    _vae_modules: ClassVar[list[str]]
    _resident_modules: ClassVar[list[str]] = []


def supports_step_execution(pipeline: object) -> bool:
    """Return whether `pipeline` implements :class:`SupportsStepExecution`."""

    return isinstance(pipeline, SupportsStepExecution)


@runtime_checkable
class SupportsDiffusionAtoms(Protocol):
    """Finer-grained pipeline atoms for disaggregated diffusion execution.

    Additive to :class:`SupportsStepExecution`: a pipeline may implement these
    finer atoms so the encode and decode stages can run only their portion of
    the work. The generic runner prefers these atoms when present and falls back
    to the four-method step contract (``prepare_encode`` / ``post_decode``)
    otherwise, so existing step pipelines keep working unchanged.

    * ``check_inputs`` — validate request inputs; return the initialized state.
    * ``encode_conditions`` — run text/image/observation encoders.
    * ``prepare_latents_and_timesteps`` — initial latents, timestep schedule, CFG.
    * ``decode_latents`` — VAE/audio/video decode of final latents.
    * ``postprocess_outputs`` — final output construction (``DiffusionOutput``).
    """

    supports_diffusion_atoms: ClassVar[bool] = True

    def check_inputs(self, state: DiffusionRequestState, **kwargs: Any) -> DiffusionRequestState: ...

    def encode_conditions(self, state: DiffusionRequestState, **kwargs: Any) -> DiffusionRequestState: ...

    def prepare_latents_and_timesteps(self, state: DiffusionRequestState, **kwargs: Any) -> DiffusionRequestState: ...

    def decode_latents(self, state: DiffusionRequestState, **kwargs: Any) -> DiffusionRequestState: ...

    def postprocess_outputs(self, state: DiffusionRequestState, **kwargs: Any) -> DiffusionOutput: ...


@runtime_checkable
class SupportsDisaggregatedDiffusionExecution(Protocol):
    """Pipeline capability for the three-stage disaggregated diffusion runtime.

    A pipeline that declares this capability can be driven as an ``encode``,
    ``denoise``, or ``decode`` stage (RFC #4590). The runner calls these hooks
    instead of reaching into model-private state:

    * ``export_stage_payload`` converts runner-local
      :class:`DiffusionRequestState` into a transportable
      :class:`DiffusionStagePayload` at a stage boundary.
    * ``import_stage_payload`` rebuilds a runner-local state from a payload the
      upstream stage produced (attaching any live session state the target
      stage owns — e.g. AR-Diffusion KV — through the normal engine mechanism,
      *not* via the payload).
    * ``required_components_for_stage`` declares which components a given role
      must construct/load, queried before module construction (see
      :class:`~vllm_omni.diffusion.stage_roles.StageComponentSpec`).

    The runner never interprets the tensors or metadata a pipeline packs; those
    are model-private and only round-trip through this pipeline's own hooks.
    """

    supports_disaggregated_execution: ClassVar[bool] = True

    def export_stage_payload(
        self,
        state: DiffusionRequestState,
        *,
        source_stage: str,
        target_stage: str,
    ) -> DiffusionStagePayload: ...

    def import_stage_payload(
        self,
        payload: DiffusionStagePayload,
        *,
        target_stage: str,
        request: OmniDiffusionRequest | None = None,
    ) -> DiffusionRequestState: ...

    @classmethod
    def required_components_for_stage(cls, model_stage: str) -> StageComponentSpec: ...


def supports_diffusion_atoms(pipeline: object) -> bool:
    """Return whether `pipeline` implements :class:`SupportsDiffusionAtoms`."""

    return isinstance(pipeline, SupportsDiffusionAtoms)


def supports_disaggregated_execution(pipeline: object) -> bool:
    """Return whether `pipeline` implements the disaggregated capability.

    Uses an explicit ``supports_disaggregated_execution`` flag plus the required
    hooks. ``isinstance`` against the runtime-checkable Protocol validates hook
    presence; the flag lets a pipeline opt out even if the method names happen
    to collide.
    """

    if not bool(getattr(pipeline, "supports_disaggregated_execution", False)):
        return False
    return isinstance(pipeline, SupportsDisaggregatedDiffusionExecution)
