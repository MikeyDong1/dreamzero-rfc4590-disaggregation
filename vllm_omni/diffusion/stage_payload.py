# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Typed, versioned transport envelope for disaggregated diffusion stages.

RFC #4590 §2.3 requires a small typed envelope that carries the *data* one
diffusion stage produces for the next, kept strictly separate from the mutable
runner-local ``DiffusionRequestState``. A ``DiffusionRequestState`` must never
cross a process boundary; a :class:`DiffusionStagePayload` may.

Design invariants (enforced by :meth:`DiffusionStagePayload.validate`):

* request identity is explicit (``request_id``);
* source and target stages are explicit and named;
* the schema is versioned (``schema_version``);
* tensors live in ``tensors`` (name -> tensor); small metadata lives in
  ``metadata``; the common envelope never grows a field per model;
* non-transportable values (modules, generators, CUDA streams, schedulers,
  process-local cache objects, device pointers) are rejected early;
* model-specific keys are interpreted only by the owning pipeline; the runner
  and the generic transition processor treat them opaquely;
* tensors can later be backed by connector handles without changing the
  envelope's shape or the runner's semantics.

The first implementation uses the existing host-based (msgpack over ZMQ, or
in-process) stage transport. To stay compatible with both, tensors are moved to
host memory, detached from autograd, and made contiguous at the *export*
boundary (see :func:`sanitize_transport_tensor`). Host-to-device movement
happens once at the *restore* boundary inside the owning pipeline.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


#: Bump when the envelope's wire shape changes incompatibly. Consumers reject
#: payloads whose ``schema_version`` they do not understand.
DIFFUSION_STAGE_PAYLOAD_SCHEMA_VERSION = 1

#: The two channels a :class:`DiffusionStagePayload` travels between stages, kept
#: here (the torch-free transport leaf every stage layer already imports) as the
#: single source of truth so the runner, the generic transition processor, and
#: the AR-Diffusion runner cannot drift:
#:
#: * :data:`STAGE_PAYLOAD_OUTPUT_KEY` — key under which the *producer* stage's
#:   ``DiffusionOutput.custom_output`` carries the outgoing payload.
#: * :data:`STAGE_PAYLOAD_PROMPT_KEY` — key under which the *consumer* stage's
#:   request prompt ``extra`` dict carries the incoming payload.
STAGE_PAYLOAD_OUTPUT_KEY = "__diffusion_stage_payload__"
STAGE_PAYLOAD_PROMPT_KEY = "diffusion_stage_payload"

#: Metadata values are restricted to these plain, transportable Python types
#: (recursively for containers). Everything else is rejected at validation.
#: ``bytes`` is allowed for opaque model-private blobs; numpy scalars/arrays are
#: allowed because the existing msgpack transport already supports them.
_ALLOWED_METADATA_SCALARS = (str, bool, bytes, numbers.Number, type(None))


class StagePayloadError(ValueError):
    """Raised when a :class:`DiffusionStagePayload` fails validation."""


class NonTransportableValueError(StagePayloadError):
    """Raised when payload metadata/tensors contain a non-transportable value.

    The message names the offending key path and type so the failure points at
    the model code that packed it, not at the generic transport layer.
    """


def _is_tensor(value: Any) -> bool:
    """Duck-typed torch.Tensor check that does not import torch eagerly."""
    cls = type(value)
    # torch.Tensor instances report a module of "torch"; checking the MRO names
    # avoids importing torch in environments (config-only tools, tests) that do
    # not have it while still recognizing real tensors at runtime.
    return any(base.__module__ == "torch" and base.__name__ == "Tensor" for base in cls.__mro__)


def _looks_non_transportable(value: Any) -> str | None:
    """Return a reason string if ``value`` is clearly non-transportable.

    Catches the specific hazards RFC #4590 §2.3 enumerates: live modules,
    generators, CUDA streams, schedulers, and other process-local objects. This
    is a *positive* guard against obviously-unsafe types; the metadata validator
    additionally enforces an allow-list, so novel unsafe types are still caught.
    """
    cls = type(value)
    module = cls.__module__ or ""
    name = cls.__name__

    if callable(value) and (module or name):
        # Functions, methods, lambdas, and class objects are not data.
        if module == "builtins" and name in ("function", "builtin_function_or_method"):
            return "callable"
        if cls.__name__ in ("function", "method", "builtin_function_or_method", "type"):
            return "callable"

    # torch process-local objects.
    if module.startswith("torch"):
        if name in ("Generator", "Stream", "Event", "device", "dtype"):
            return f"torch.{name}"
        if "Module" in name or name.endswith("Module"):
            return f"torch module ({name})"

    # Diffusion scheduler instances and runner-local state objects.
    lowered = name.lower()
    if "scheduler" in lowered:
        return f"scheduler object ({name})"
    if name in ("DiffusionRequestState", "DreamZeroState", "ARDiffusionKVState", "ARDiffusionKVCache"):
        return f"process-local state object ({name})"
    if hasattr(value, "__next__") and hasattr(value, "__iter__") and name == "generator":
        return "generator"

    return None


def sanitize_transport_tensor(tensor: "torch.Tensor") -> "torch.Tensor":
    """Return a host-resident, detached, contiguous copy of ``tensor``.

    This is the single explicit device->host transport boundary (RFC #4590 §7):

    * ``detach()`` drops autograd linkage;
    * ``.cpu()`` performs the one device-to-host move;
    * ``.contiguous()`` makes the buffer serialization-safe;
    * ``.clone()`` guarantees the exported tensor never aliases a buffer the
      producer may reuse (a stage-boundary copy is acceptable for the first
      correctness implementation).

    dtype and shape are preserved exactly.
    """
    detached = tensor.detach()
    host = detached.cpu()
    contiguous = host.contiguous()
    # clone() only when .cpu()/.contiguous() returned a view onto live memory.
    if contiguous.data_ptr() == tensor.data_ptr():
        contiguous = contiguous.clone()
    return contiguous


@dataclass(frozen=True)
class DiffusionStagePayload:
    """Immutable transport envelope handed from one diffusion stage to the next.

    Attributes:
        schema_version: Envelope wire version (see module constant).
        request_id: The originating request identity, preserved end to end.
        source_stage: Role of the stage that produced this payload.
        target_stage: Role of the stage expected to consume it.
        payload_type: Coarse model-agnostic tag (e.g. ``"encode_to_denoise"``)
            for observability and processor routing; not a per-model field.
        tensors: name -> host tensor. The only place tensors live.
        metadata: small, transportable model-private scalars/containers. The
            runner and processor treat these opaquely; only the owning pipeline
            interprets them.
    """

    schema_version: int
    request_id: str
    source_stage: str
    target_stage: str
    payload_type: str
    tensors: dict[str, "torch.Tensor"] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- construction helpers ------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        request_id: str,
        source_stage: str,
        target_stage: str,
        payload_type: str,
        tensors: dict[str, "torch.Tensor"] | None = None,
        metadata: dict[str, Any] | None = None,
        sanitize: bool = True,
        validate: bool = True,
    ) -> DiffusionStagePayload:
        """Build a payload, optionally sanitizing tensors and validating.

        ``sanitize=True`` (default) routes every tensor through
        :func:`sanitize_transport_tensor` so the caller does not have to scatter
        ``.detach().cpu()`` through model code. ``validate=True`` runs the full
        transportability check before returning.
        """
        prepared_tensors: dict[str, "torch.Tensor"] = {}
        for name, tensor in (tensors or {}).items():
            if not _is_tensor(tensor):
                raise NonTransportableValueError(
                    f"tensors[{name!r}] is {type(tensor).__name__}, expected a torch.Tensor. "
                    "Small non-tensor values belong in metadata."
                )
            prepared_tensors[name] = sanitize_transport_tensor(tensor) if sanitize else tensor

        payload = cls(
            schema_version=DIFFUSION_STAGE_PAYLOAD_SCHEMA_VERSION,
            request_id=request_id,
            source_stage=source_stage,
            target_stage=target_stage,
            payload_type=payload_type,
            tensors=prepared_tensors,
            metadata=dict(metadata or {}),
        )
        if validate:
            payload.validate()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiffusionStagePayload:
        """Rehydrate a payload flattened to a plain dict by cross-process transport.

        The inter-stage IPC (msgpack) does not preserve dataclass identity, so a
        payload sent through ``custom_output`` arrives on the receiving stage as
        a plain ``dict`` with the same keys as this dataclass's fields. Two
        callers reconstruct the real type via this method before validation: the
        processor's ``_unwrap_stage_payload`` (whose caller then validates) and
        the runner's ``DiffusionModelRunner._extract_incoming_payload`` (which
        validates the result directly).
        """
        return cls(
            schema_version=data["schema_version"],
            request_id=data["request_id"],
            source_stage=data["source_stage"],
            target_stage=data["target_stage"],
            payload_type=data["payload_type"],
            tensors=dict(data.get("tensors") or {}),
            metadata=dict(data.get("metadata") or {}),
        )

    # -- validation ----------------------------------------------------------

    def validate(self) -> None:
        """Raise :class:`StagePayloadError` if the envelope is not transportable.

        Checks identity, versioning, tensor typing, and (recursively) that
        metadata contains only allow-listed transportable values with no
        obviously process-local object anywhere in the tree.
        """
        if not isinstance(self.request_id, str) or not self.request_id:
            raise StagePayloadError("DiffusionStagePayload.request_id must be a non-empty string.")
        if self.schema_version != DIFFUSION_STAGE_PAYLOAD_SCHEMA_VERSION:
            raise StagePayloadError(
                f"Unsupported DiffusionStagePayload schema_version {self.schema_version}; "
                f"this build understands version {DIFFUSION_STAGE_PAYLOAD_SCHEMA_VERSION}."
            )
        for role_field in ("source_stage", "target_stage", "payload_type"):
            value = getattr(self, role_field)
            if not isinstance(value, str) or not value:
                raise StagePayloadError(f"DiffusionStagePayload.{role_field} must be a non-empty string.")

        if not isinstance(self.tensors, dict):
            raise StagePayloadError("DiffusionStagePayload.tensors must be a dict[str, Tensor].")
        for name, tensor in self.tensors.items():
            if not isinstance(name, str):
                raise StagePayloadError(f"tensor key {name!r} must be a string.")
            if not _is_tensor(tensor):
                raise NonTransportableValueError(
                    f"tensors[{name!r}] is {type(tensor).__name__}, expected a torch.Tensor."
                )

        if not isinstance(self.metadata, dict):
            raise StagePayloadError("DiffusionStagePayload.metadata must be a dict.")
        for key, value in self.metadata.items():
            if not isinstance(key, str):
                raise StagePayloadError(f"metadata key {key!r} must be a string.")
            self._validate_metadata_value(value, path=f"metadata[{key!r}]")

    @classmethod
    def _validate_metadata_value(cls, value: Any, *, path: str) -> None:
        # Tensors do not belong in metadata — they must be transported via the
        # dedicated ``tensors`` channel so a future connector can back them with
        # handles without rewriting metadata packing.
        if _is_tensor(value):
            raise NonTransportableValueError(
                f"{path} is a torch.Tensor; tensors must be placed in the payload's 'tensors' dict, not metadata."
            )
        reason = _looks_non_transportable(value)
        if reason is not None:
            raise NonTransportableValueError(f"{path} is non-transportable ({reason}).")

        if isinstance(value, _ALLOWED_METADATA_SCALARS):
            return
        if isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                cls._validate_metadata_value(item, path=f"{path}[{i}]")
            return
        if isinstance(value, dict):
            for k, v in value.items():
                if not isinstance(k, str):
                    raise StagePayloadError(f"{path} has non-string key {k!r}.")
                cls._validate_metadata_value(v, path=f"{path}[{k!r}]")
            return
        # numpy arrays/scalars are transportable via the existing msgpack codec.
        module = type(value).__module__ or ""
        if module.startswith("numpy"):
            return
        raise NonTransportableValueError(
            f"{path} has non-transportable type {type(value).__name__}. Allowed: "
            "str/bool/bytes/number/None, nested list/tuple/dict of those, numpy arrays, "
            "or a torch.Tensor placed in the 'tensors' dict."
        )

    # -- convenience ---------------------------------------------------------

    def require_tensors(self, *names: str) -> None:
        """Raise if any of ``names`` is missing from ``tensors``."""
        missing = [n for n in names if n not in self.tensors]
        if missing:
            raise StagePayloadError(
                f"Payload {self.payload_type!r} for request {self.request_id!r} is missing "
                f"required tensors: {missing}. Present: {sorted(self.tensors)}."
            )

    def expect_transition(self, *, source: str, target: str) -> None:
        """Raise if the payload's source/target do not match expectations."""
        if self.source_stage != source or self.target_stage != target:
            raise StagePayloadError(
                f"Payload for request {self.request_id!r} has transition "
                f"{self.source_stage!r}->{self.target_stage!r}, expected {source!r}->{target!r}."
            )

    def summary(self) -> str:
        """One-line, tensor-content-free description for debug logging."""
        tensor_desc = ", ".join(
            f"{name}{tuple(t.shape)}:{str(t.dtype).replace('torch.', '')}" for name, t in self.tensors.items()
        )
        return (
            f"DiffusionStagePayload(req={self.request_id} {self.source_stage}->{self.target_stage} "
            f"type={self.payload_type} v{self.schema_version} tensors=[{tensor_desc}] "
            f"meta_keys={sorted(self.metadata)})"
        )
