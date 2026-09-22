"""Verify identical runtime seeds and enforce the system-prompt edit boundary."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from gepa.adapters.terminal_bench_adapter import (
    HarborEvaluation,
    HarborTrialResult,
    TerminusAdapter,
    load_terminalbench_manifest,
)
from gepa.adapters.terminal_bench_adapter.documents import (
    COMPONENT_KINDS,
    INITIAL_COMPONENTS,
    seed_documents,
    write_document_bundle,
)
from gepa.adapters.terminal_bench_adapter.text_scope import OPTIMIZATION_SCOPES, TerminalBenchTextScope
from gepa.core.adapter import EvaluationBatch
from gepa.strategies.document_template import TEMPLATE_FAMILIES

MANIFEST_PATH = Path(__file__).parents[1] / "examples/terminalbench/terminalbench-v2.1-manifest.json"


def test_adapter_defaults_to_the_unified_initial_prompt() -> None:
    """Make default API evaluations enforce the same prompt-only scope as the CLI."""
    manifest = load_terminalbench_manifest(MANIFEST_PATH)
    adapter = TerminusAdapter(manifest, Mock(manifest=manifest))
    assert adapter.text_scope == TerminalBenchTextScope("system_prompt")
    assert set(adapter.text_scope.seed_candidate()) == {"instruction_prompt"}


@pytest.mark.parametrize("family", TEMPLATE_FAMILIES)
def test_scopes_start_with_identical_model_text_and_skill_files(tmp_path: Path, family: str) -> None:
    """Keep runtime inputs equal while exposing sixteen artifacts or one valid prompt."""
    bundles = []
    for scope_name in OPTIMIZATION_SCOPES:
        scope = TerminalBenchTextScope(scope_name, family)
        candidate = scope.seed_candidate()
        assert len(candidate) == (16 if scope_name == "all_text" else 1)
        for name, text in candidate.items():
            TEMPLATE_FAMILIES[family][scope.component_kinds[name]].parse(text)
        directory = tmp_path / scope_name
        directory.mkdir()
        bundles.append(json.loads(write_document_bundle(directory, scope.materialize(candidate)).read_text()))
    first, second = (tmp_path / scope for scope in OPTIMIZATION_SCOPES)
    for filename in ("terminus-prompt.txt", "timeout.txt"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    assert bundles[0]["prompts"] == bundles[1]["prompts"]
    assert bundles[0]["skills"] == bundles[1]["skills"]
    for skill in bundles[0]["skills"]:
        relative = Path("skills") / skill["component"] / "SKILL.md"
        assert (first / relative).read_bytes() == (second / relative).read_bytes()


def test_prompt_edit_preserves_all_later_prompts_and_skills(tmp_path: Path) -> None:
    """Materialize only the edited initial block without duplicating fixed tool guidance."""
    scope = TerminalBenchTextScope("system_prompt", "generic")
    edit = "## Task\nRevised instructions; שלום {literal} {instruction}."
    documents = scope.materialize({"instruction_prompt": edit})
    baseline = seed_documents("generic")
    assert all(documents[name] == baseline[name] for name in baseline if name not in INITIAL_COMPONENTS)
    assert all(documents[name] == "" for name in INITIAL_COMPONENTS[1:])
    write_document_bundle(tmp_path, documents)
    prompt = (tmp_path / "terminus-prompt.txt").read_text().format(instruction="TASK", terminal_state="TERMINAL")
    assert prompt.count(edit) == 1
    assert "TASK" in prompt and "TERMINAL" in prompt
    assert "Use the tmux terminal" not in prompt


@pytest.mark.parametrize("component", [name for name in COMPONENT_KINDS if name != "instruction_prompt"])
def test_prompt_scope_rejects_extra_artifacts_before_harbor(component: str) -> None:
    """Block tool, auxiliary-prompt, and skill edits even if a proposer returns extra keys."""
    manifest = load_terminalbench_manifest(MANIFEST_PATH)
    harbor = Mock(manifest=manifest)
    scope = TerminalBenchTextScope("system_prompt")
    adapter = TerminusAdapter(manifest, harbor, text_scope=scope)
    candidate = {**scope.seed_candidate(), component: "unauthorized edit"}
    with pytest.raises(ValueError, match="editable system_prompt components"):
        adapter.evaluate(manifest.tasks("train", 1), candidate)
    harbor.run.assert_not_called()
    with pytest.raises(ValueError, match="Unknown Terminal Bench document selection"):
        adapter.make_reflective_dataset(scope.seed_candidate(), EvaluationBatch([], [], []), [component])


@pytest.mark.parametrize("scope_name", OPTIMIZATION_SCOPES)
def test_adapter_materializes_scope_and_reflects_only_editable_text(tmp_path: Path, scope_name: str) -> None:
    """Keep official rewards and training evidence while restricting reflected components."""
    manifest = load_terminalbench_manifest(MANIFEST_PATH)
    scope = TerminalBenchTextScope(scope_name)
    candidate = scope.seed_candidate()
    task = manifest.tasks("train", 1)[0]
    documents = scope.materialize(candidate)
    trial = HarborTrialResult(
        task_id=task.task_id,
        reward=0.0,
        rewards={"reward": 0.0},
        errors=[],
        atif_trajectories=[],
        raw_result={},
        trial_dir=tmp_path,
        verifier_logs={"test-stdout.txt": "The requested file is missing."},
    )
    harbor = Mock(manifest=manifest)
    harbor.run.return_value = HarborEvaluation(
        evaluation_id="scope-evaluation",
        candidate_digest=manifest.candidate_digest(documents),
        config_path=tmp_path / "config.json",
        job_dir=tmp_path,
        returncode=0,
        stdout_path=tmp_path / "stdout.txt",
        stderr_path=tmp_path / "stderr.txt",
        trials={task.task_id: trial},
    )
    adapter = TerminusAdapter(manifest, harbor, text_scope=scope)
    evaluated = adapter.evaluate([task], candidate, capture_traces=True)
    harbor.run.assert_called_once_with([task.task_id], documents)
    assert evaluated.scores == [0.0] and evaluated.num_metric_calls == 1
    feedback = adapter.make_reflective_dataset(candidate, evaluated, list(scope.component_kinds))
    assert set(feedback) == set(candidate)
    assert all("requested file is missing" in rows[0]["Feedback"] for rows in feedback.values())
