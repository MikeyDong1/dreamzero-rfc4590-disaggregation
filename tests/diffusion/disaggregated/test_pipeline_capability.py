# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline capability-detection tests (RFC #4590 §14.2).

Torch-free: interface.py imports torch only under TYPE_CHECKING, so the
capability helpers can be exercised directly. These lock the contract that a
pipeline must carry the ``supports_diffusion_atoms`` flag (not just the methods)
to be treated as atom-capable — the exact regression that crashes the
encode/decode stages if a pipeline defines the atoms but omits the flag.
"""

from __future__ import annotations


class _AtomMethods:
    def check_inputs(self, s, **k): ...
    def encode_conditions(self, s, **k): ...
    def prepare_latents_and_timesteps(self, s, **k): ...
    def decode_latents(self, s, **k): ...
    def postprocess_outputs(self, s, **k): ...


class _DisaggMethods:
    def export_stage_payload(self, s, *, source_stage, target_stage): ...
    def import_stage_payload(self, p, *, target_stage, request=None): ...

    @classmethod
    def required_components_for_stage(cls, model_stage): ...


def test_atoms_require_flag_presence(interface_mod):
    class WithFlag(_AtomMethods):
        supports_diffusion_atoms = True

    class MissingFlag(_AtomMethods):
        pass  # methods present, flag absent -> must NOT be treated as atom-capable

    assert interface_mod.supports_diffusion_atoms(WithFlag()) is True
    # The critical regression: methods without the flag must fail detection, so
    # the runner does not fall back to prepare_encode/post_decode.
    assert interface_mod.supports_diffusion_atoms(MissingFlag()) is False


def test_step_only_pipeline_is_not_disaggregated(interface_mod):
    class StepOnly:
        supports_step_execution = True

        def prepare_encode(self, s, **k): ...
        def denoise_step(self, b, **k): ...
        def step_scheduler(self, s, n, **k): ...
        def post_decode(self, s, **k): ...

    assert interface_mod.supports_step_execution(StepOnly()) is True
    assert interface_mod.supports_disaggregated_execution(StepOnly()) is False


def test_disaggregated_requires_flag_and_hooks(interface_mod):
    class WithFlag(_DisaggMethods):
        supports_disaggregated_execution = True

    class FlagOff(_DisaggMethods):
        supports_disaggregated_execution = False  # explicit opt-out honored

    class NoHooks:
        supports_disaggregated_execution = True

    assert interface_mod.supports_disaggregated_execution(WithFlag()) is True
    assert interface_mod.supports_disaggregated_execution(FlagOff()) is False
    assert interface_mod.supports_disaggregated_execution(NoHooks()) is False


def test_monolithic_pipeline_needs_no_capability(interface_mod):
    class Monolithic:
        def forward(self, batch): ...

    assert interface_mod.supports_disaggregated_execution(Monolithic()) is False
    assert interface_mod.supports_diffusion_atoms(Monolithic()) is False
    assert interface_mod.supports_step_execution(Monolithic()) is False
