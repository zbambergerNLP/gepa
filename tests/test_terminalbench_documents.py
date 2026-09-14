"""Verify candidate identity, runtime field preservation, and optimizer parity."""

import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from terminalbench_pilot_helpers import offline_runtime as offline_runtime
from terminalbench_pilot_helpers import write_pilot_fixture

sys.path.insert(0, str(Path(__file__).parents[1]))

from examples.terminalbench import main as cli
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


@pytest.mark.parametrize("experiment", cli.EXPERIMENT_MANIFESTS)
def test_cli_gives_both_methods_the_same_documents_and_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, experiment: str
) -> None:
    """Exercise the real CLI wiring without starting model calls or Docker.

    Args:
        tmp_path: Separate run directories for the two conditions.
        monkeypatch: Fixture replacing only external execution boundaries.
        experiment: Each independently selectable benchmark and optimization target.
    """
    optimize = Mock()
    evaluate = Mock()
    monkeypatch.setattr(cli, "optimize", optimize)
    monkeypatch.setattr("examples.terminalbench.evaluate.main", evaluate)
    monkeypatch.setattr(cli.HarborCLI, "check_requirements", Mock())
    for condition in ("vanilla", "react_v2"):
        monkeypatch.setattr(
            "sys.argv",
            [
                "terminalbench",
                "--experiment",
                experiment,
                "--condition",
                condition,
                "--max-metric-calls",
                "50",
                "--run-dir",
                str(tmp_path / condition),
                "--harbor-work-dir",
                str(tmp_path / "harbor"),
            ],
        )
        args = cli.build_parser().parse_args()
        manifest = cli.load_terminalbench_manifest(cli.EXPERIMENT_MANIFESTS[experiment])
        _, family = cli.seed_candidate(args.student_model, "auto", experiment)
        contract = cli.build_run_contract(
            args, manifest, manifest.tasks("train"), manifest.tasks("val"), condition, family
        )
        pilot_dir = write_pilot_fixture(tmp_path / "pilot", contract, manifest)
        monkeypatch.setattr(sys, "argv", [*sys.argv, "--reviewed-pilot", str(pilot_dir)])
        cli.main()
    vanilla, forest = [call.kwargs for call in optimize.call_args_list]
    assert evaluate.call_count == 2
    for condition, call in zip(("vanilla", "react_v2"), evaluate.call_args_list, strict=True):
        assert call.args[0][:2] == ["--run-dir", f"system_prompt__{condition}={tmp_path / condition}"]
    for key in (
        "seed_candidate",
        "component_kinds",
        "module_selector",
        "trainset",
        "valset",
        "max_metric_calls",
        "template_family",
    ):
        assert vanilla[key] == forest[key]
    assert vanilla["reflection_lm"].model == forest["reflection_lm"].model
    for field in ("reflection_lm_kwargs", "completion_kwargs"):
        settings = [item[field] if field == "reflection_lm_kwargs" else item["reflection_lm"].completion_kwargs for item in (vanilla, forest)]
        assert {key: value for key, value in settings[0].items() if key != "_gepa_provider_retry"} == {
            key: value for key, value in settings[1].items() if key != "_gepa_provider_retry"
        }
        for condition, kwargs in zip(("vanilla", "react_v2"), settings, strict=True):
            assert kwargs["_gepa_provider_retry"]["log_path"] == str(tmp_path / condition / "provider-attempts.jsonl")
    assert forest["reflection_strategy"].react_max_iterations is None
    assert forest["reflection_strategy"].react_max_tool_calls is None
    assert vanilla["reflection_level"] == 0
    assert forest["reflection_level"] == 2
    expected = {"instruction_prompt": "user_prompt"}
    assert vanilla["module_selector"] == "all"
    assert vanilla["component_kinds"] == expected
    assert set(vanilla["seed_candidate"]) == set(expected)
    for invocation in (vanilla, forest):
        assert invocation["adapter"].manifest.experiment == experiment
        assert invocation["adapter"].harbor.manifest.experiment == experiment
        test_ids = set(invocation["adapter"].manifest.splits["test"])
        assert not test_ids.intersection(task.task_id for task in invocation["trainset"] + invocation["valset"])
    old_contract = json.loads((tmp_path / "vanilla" / cli.RUN_CONTRACT_FILENAME).read_text())
    assert old_contract["experiment"] == experiment
    old_contract["schema_version"] = 4
    with pytest.raises(ValueError, match="different Terminal-Bench configuration"):
        cli.ensure_run_contract(tmp_path / "vanilla", old_contract)
