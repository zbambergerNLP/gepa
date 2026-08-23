"""Offline tests for the Terminal-Bench experiment CLI contract."""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

import examples.terminalbench.main as terminalbench_main
from examples.terminalbench.main import build_parser, build_run_contract, ensure_run_contract, seed_candidate
from gepa.adapters.terminal_bench_adapter import load_terminalbench_manifest
from gepa.proposer.reflective_mutation.three_role import ThreeRoleReflectionLM
from gepa.strategies.document_template import TEMPLATE_FAMILIES
from gepa.strategies.intervention import controller_policy_contract, semantic_action_catalog


def test_qwen_student_uses_alibaba_system_prompt_template() -> None:
    candidate, family = seed_candidate("hosted_vllm/Qwen3.8", "auto")
    bodies = TEMPLATE_FAMILIES[family]["prompt"].parse(candidate["instruction_prompt"])

    assert family == "alibaba"
    assert "autonomous terminal agent" in bodies["Objective"]
    assert all(bodies.values())


def test_parser_exposes_react_v2_condition_and_ablation_axes() -> None:
    help_text = build_parser().format_help()

    assert "react_v2" in help_text
    assert "rlm" in help_text
    assert "--reflection-level" in help_text
    assert "--edit-tool-set" in help_text
    assert "--template-family" in help_text


def test_run_contract_allows_exact_resume_and_rejects_drift(tmp_path: Path) -> None:
    contract = {"condition": "react_v2", "student_model": "Qwen3.8", "edit_tool_set": "broad"}
    path = ensure_run_contract(tmp_path, contract)

    assert ensure_run_contract(tmp_path, contract) == path
    with pytest.raises(ValueError, match="different Terminal-Bench configuration"):
        ensure_run_contract(tmp_path, {**contract, "edit_tool_set": "minimal"})


def test_generated_run_contract_records_metric_call_budget(tmp_path: Path) -> None:
    manifest_path = Path(__file__).parents[1] / "examples" / "terminalbench" / "terminalbench-v3-manifest.json"
    args = build_parser().parse_args(
        [
            "--condition",
            "react_v2",
            "--student-model",
            "provider/qwen3-8b",
            "--proposer-model",
            "provider/deepseek-v4-flash",
            "--max-metric-calls",
            "400",
            "--manifest",
            str(manifest_path),
            "--run-dir",
            str(tmp_path / "run"),
            "--harbor-work-dir",
            str(tmp_path / "harbor"),
        ]
    )
    manifest = load_terminalbench_manifest(manifest_path)

    contract = build_run_contract(
        args,
        manifest,
        manifest.tasks("train", 1),
        manifest.tasks("val", 1),
        "react_v2",
        "alibaba",
    )

    assert contract["max_metric_calls"] == 400
    assert contract["semantic_action_space"] == semantic_action_catalog("prompt")
    assert contract["semantic_controller_policy"] == controller_policy_contract()


def test_rlm_run_contract_records_full_budget_and_eight_call_cap(tmp_path: Path) -> None:
    """Persist the backend and its matched tree-wide budget for safe resume."""
    manifest_path = Path(__file__).parents[1] / "examples" / "terminalbench" / "terminalbench-v3-manifest.json"
    args = build_parser().parse_args(
        [
            "--condition",
            "rlm",
            "--student-model",
            "provider/qwen3-8b",
            "--proposer-model",
            "provider/deepseek-v4-flash",
            "--max-metric-calls",
            "400",
            "--manifest",
            str(manifest_path),
            "--run-dir",
            str(tmp_path / "run"),
            "--harbor-work-dir",
            str(tmp_path / "harbor"),
        ]
    )
    manifest = load_terminalbench_manifest(manifest_path)

    contract = build_run_contract(
        args,
        manifest,
        manifest.tasks("train", 1),
        manifest.tasks("val", 1),
        "rlm",
        "alibaba",
    )

    assert contract["proposer_backend"] == "rlm"
    assert contract["max_proposer_model_calls"] == 8
    assert contract["rlm_budget"] == {
        "max_root_iterations": 4,
        "max_child_iterations": 2,
        "max_repl_calls": 6,
        "max_llm_queries": 2,
        "max_rlm_queries": 1,
        "max_recursion_depth": 1,
        "max_exec_seconds": 5,
        "max_output_chars": 4000,
    }


def test_rlm_main_wires_explicit_strategy_without_launching_an_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reach the optimize boundary and inspect the RLM strategy without executing it."""
    manifest_path = Path(__file__).parents[1] / "examples" / "terminalbench" / "terminalbench-v3-manifest.json"
    captured: dict[str, Any] = {}

    class FakeHarbor:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def check_requirements(self) -> None:
            pass

    monkeypatch.setattr(terminalbench_main, "HarborCLI", FakeHarbor)
    monkeypatch.setattr(terminalbench_main, "TerminalBenchAdapter", lambda _manifest, _harbor: object())
    monkeypatch.setattr(terminalbench_main, "optimize", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "terminalbench",
            "--condition",
            "rlm",
            "--student-model",
            "provider/qwen3-8b",
            "--proposer-model",
            "provider/deepseek-v4-flash",
            "--max-metric-calls",
            "20",
            "--train-limit",
            "1",
            "--val-limit",
            "1",
            "--manifest",
            str(manifest_path),
            "--run-dir",
            str(tmp_path / "run"),
            "--harbor-work-dir",
            str(tmp_path / "harbor"),
        ],
    )

    terminalbench_main.main()

    strategy = captured["reflection_strategy"]
    assert isinstance(strategy, ThreeRoleReflectionLM)
    assert strategy.proposer_backend == "rlm"
    assert strategy.level == 2
    assert strategy.edit_tool_set == "broad"
    assert strategy.rlm_budget.max_model_calls == 8
    assert captured["reflection_level"] == 2
    assert captured["max_metric_calls"] == 20


@pytest.mark.parametrize(
    "invalid_axis",
    [
        pytest.param(["--reflection-level", "1"], id="level_one"),
        pytest.param(["--edit-tool-set", "minimal"], id="minimal_basis"),
    ],
)
def test_rlm_cli_rejects_unsupported_axes_before_harbor(
    invalid_axis: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail before external setup when RLM cannot honor the requested action/operator contract."""
    manifest_path = Path(__file__).parents[1] / "examples" / "terminalbench" / "terminalbench-v3-manifest.json"

    class UnexpectedHarbor:
        def __init__(self, **_kwargs: Any) -> None:
            raise AssertionError("Harbor must not be initialized")

    monkeypatch.setattr(terminalbench_main, "HarborCLI", UnexpectedHarbor)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "terminalbench",
            "--condition",
            "rlm",
            "--student-model",
            "provider/qwen3-8b",
            "--proposer-model",
            "provider/deepseek-v4-flash",
            "--max-metric-calls",
            "20",
            "--train-limit",
            "1",
            "--val-limit",
            "1",
            "--manifest",
            str(manifest_path),
            "--run-dir",
            str(tmp_path / "run"),
            "--harbor-work-dir",
            str(tmp_path / "harbor"),
            *invalid_axis,
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        terminalbench_main.main()


def test_legacy_state_without_contract_is_not_resumed(tmp_path: Path) -> None:
    (tmp_path / "gepa_state.bin").write_bytes(b"old-state")

    with pytest.raises(ValueError, match="no terminalbench-run-contract.json"):
        ensure_run_contract(tmp_path, {"condition": "react_v2"})
