"""Offline tests for the Terminal-Bench experiment CLI contract."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from examples.terminalbench.main import build_parser, build_run_contract, ensure_run_contract, seed_candidate
from gepa.adapters.terminal_bench_adapter import load_terminalbench_manifest
from gepa.strategies.document_template import TEMPLATE_FAMILIES


def test_qwen_student_uses_alibaba_system_prompt_template() -> None:
    candidate, family = seed_candidate("hosted_vllm/Qwen3.8", "auto")
    bodies = TEMPLATE_FAMILIES[family]["prompt"].parse(candidate["instruction_prompt"])

    assert family == "alibaba"
    assert "autonomous terminal agent" in bodies["Objective"]
    assert all(bodies.values())


def test_parser_exposes_react_v2_condition_and_ablation_axes() -> None:
    help_text = build_parser().format_help()

    assert "react_v2" in help_text
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


def test_legacy_state_without_contract_is_not_resumed(tmp_path: Path) -> None:
    (tmp_path / "gepa_state.bin").write_bytes(b"old-state")

    with pytest.raises(ValueError, match="no terminalbench-run-contract.json"):
        ensure_run_contract(tmp_path, {"condition": "react_v2"})
