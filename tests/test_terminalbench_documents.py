"""Verify candidate identity, runtime field preservation, and optimizer parity."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from gepa.adapters.terminal_bench_adapter.documents import (
    COMPONENT_KINDS,
    CONTEXT_FIELDS,
    document_digest,
    seed_documents,
    validate_documents,
    write_document_bundle,
)


@pytest.mark.parametrize("component", COMPONENT_KINDS)
def test_every_document_changes_candidate_identity(component: str) -> None:
    """Make skill and auxiliary-prompt edits distinct from the parent.

    Args:
        component: Each component exposed to both optimizers.
    """
    parent = seed_documents("generic")
    child = {**parent, component: parent[component] + "\nChanged guidance."}
    assert document_digest(child) != document_digest(parent)
    assert document_digest(dict(reversed(list(parent.items())))) == document_digest(parent)


def test_all_documents_are_materialized_without_evaluating_candidate_braces(tmp_path: Path) -> None:
    """Preserve literal text, including Hebrew, while inserting real runtime inputs.

    Args:
        tmp_path: Isolated evaluation directory.
    """
    candidate = {name: f"{name}: שלום {{literal}} {{instruction}}" for name in COMPONENT_KINDS}
    path = write_document_bundle(tmp_path, candidate)
    bundle = json.loads(path.read_text())
    fields = {
        key: f"OBSERVED_{key}"
        for key in (
            "instruction",
            "original_instruction",
            "terminal_state",
            "command",
            "timeout_sec",
            "summary",
            "questions",
            "answers",
            "limit_str",
            "warnings_text",
        )
    }
    initial = (tmp_path / "terminus-prompt.txt").read_text().format(**fields)
    assert "OBSERVED_instruction" in initial
    assert "OBSERVED_terminal_state" in initial
    for name in ("instruction_prompt", "terminal_tool", "skill_discovery", "command_format"):
        assert candidate[name] in initial
    for name in CONTEXT_FIELDS:
        assert candidate[name] in bundle["prompts"][name].format(**fields)
    for skill in bundle["skills"]:
        assert candidate[skill["component"]] in (tmp_path / "skills" / skill["component"] / "SKILL.md").read_text()
    assert bundle["documents"] == candidate


def test_prompt_only_candidates_cannot_resume_as_complete_bundles() -> None:
    """Reject the old candidate shape before it can reach Harbor."""
    with pytest.raises(ValueError, match="complete document bundle"):
        validate_documents({"instruction_prompt": "old experiment"})
