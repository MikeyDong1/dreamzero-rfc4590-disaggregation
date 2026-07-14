# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard: the runner and the transition processor must agree on the payload keys.

The runner (worker package) and the generic transition processor (model_executor
package) each define STAGE_PAYLOAD_OUTPUT_KEY / STAGE_PAYLOAD_PROMPT_KEY locally
to avoid a cross-package import at the worker layer. This test parses both source
files (no torch import) and asserts the string literals are identical, so the two
copies cannot drift.
"""

from __future__ import annotations

import ast
from pathlib import Path

VLLM_OMNI_ROOT = Path(__file__).resolve().parents[3] / "vllm_omni"

_PROCESSOR = VLLM_OMNI_ROOT / "model_executor" / "stage_input_processors" / "diffusion.py"
_RUNNER = VLLM_OMNI_ROOT / "diffusion" / "worker" / "diffusion_model_runner.py"


def _module_string_constants(path: Path, names: set[str]) -> dict[str, str]:
    """Extract top-level ``NAME = "literal"`` assignments for the given names."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names and isinstance(node.value, ast.Constant):
                found[target.id] = node.value.value
    return found


def test_payload_keys_match_between_runner_and_processor():
    names = {"STAGE_PAYLOAD_OUTPUT_KEY", "STAGE_PAYLOAD_PROMPT_KEY"}
    proc = _module_string_constants(_PROCESSOR, names)
    runner = _module_string_constants(_RUNNER, names)
    assert proc == {
        "STAGE_PAYLOAD_OUTPUT_KEY": "__diffusion_stage_payload__",
        "STAGE_PAYLOAD_PROMPT_KEY": "diffusion_stage_payload",
    }
    assert runner == proc, f"payload key drift: runner={runner} processor={proc}"
