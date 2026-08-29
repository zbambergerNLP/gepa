"""Offline tests for the Terminal-Bench experiment CLI contract."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from examples.common.experiment_models import (
    EXPERIMENT_NUM_RETRIES,
    GLM_5_3_FLASH_MODEL,
    GLM_5_3_FLASH_OPENROUTER_MODEL,
    QWEN3_8_27B_MODEL,
    QWEN3_8_27B_MODEL_INFO,
    QWEN3_8_27B_OPENROUTER_MODEL,
    experiment_decoding,
    experiment_request_overrides,
)
from examples.terminalbench import main as terminalbench_main
from examples.terminalbench.main import build_parser, build_run_contract, ensure_run_contract, seed_candidate
from gepa.adapters.terminal_bench_adapter import load_terminalbench_manifest
from gepa.strategies.document_template import TEMPLATE_FAMILIES
from gepa.strategies.intervention import CONTROLLER_POLICY_CONTRACT, SEMANTIC_ACTION_CATALOGS

MANIFEST_PATH = Path(__file__).parents[1] / "examples" / "terminalbench" / "terminalbench-v3-manifest.json"


def _model_args(tmp_path: Path, student_model: str, proposer_model: str) -> argparse.Namespace:
    """Parse a complete Terminal-Bench invocation for one model pair.

    Args:
        tmp_path: Pytest directory used for required output paths.
        student_model: Model assigned to Terminus.
        proposer_model: Model assigned to GEPA reflection.

    Returns:
        Parsed arguments ready for the run-contract builder.
    """
    return build_parser().parse_args(
        [
            "--condition",
            "react_v2",
            "--student-model",
            student_model,
            "--proposer-model",
            proposer_model,
            "--max-metric-calls",
            "400",
            "--manifest",
            str(MANIFEST_PATH),
            "--run-dir",
            str(tmp_path / "run"),
            "--harbor-work-dir",
            str(tmp_path / "harbor"),
        ]
    )


def test_qwen_student_uses_alibaba_user_prompt_template() -> None:
    """Render the Qwen seed as a sparse Alibaba user prompt."""
    candidate, family = seed_candidate(QWEN3_8_27B_MODEL, "auto")
    prompt = candidate["instruction_prompt"]
    bodies = TEMPLATE_FAMILIES[family]["user_prompt"].parse(prompt)

    assert family == "alibaba"
    assert "assigned command-line task" in bodies["Objective"]
    assert bodies["Context"] == ""
    assert [line for line in prompt.splitlines() if line.startswith("## ")] == ["## Objective"]


def test_glm_student_uses_generic_user_prompt_template() -> None:
    """Render the GLM seed as a sparse generic user prompt."""
    candidate, family = seed_candidate(GLM_5_3_FLASH_MODEL, "auto")
    prompt = candidate["instruction_prompt"]
    bodies = TEMPLATE_FAMILIES[family]["user_prompt"].parse(prompt)

    assert family == "generic"
    assert "assigned command-line task" in bodies["Task"]
    assert all(not body for section, body in bodies.items() if section != "Task")
    assert [line for line in prompt.splitlines() if line.startswith("## ")] == ["## Task"]


def test_parser_exposes_react_v2_condition_and_ablation_axes() -> None:
    """Expose the ReAct V2 condition, reflection level, tools, and templates."""
    help_text = build_parser().format_help()

    assert "react_v2" in help_text
    assert "--reflection-level" in help_text
    assert "--edit-tool-set" in help_text
    assert "--template-family" in help_text
    assert "--api-profile" in help_text


def test_parser_defaults_both_roles_to_qwen3_8_27b(tmp_path: Path) -> None:
    """Use the homogeneous Qwen condition when model flags are omitted.

    Args:
        tmp_path: Pytest directory used for required CLI paths.
    """
    args = build_parser().parse_args(
        [
            "--condition",
            "react_v2",
            "--max-metric-calls",
            "400",
            "--run-dir",
            str(tmp_path / "run"),
            "--harbor-work-dir",
            str(tmp_path / "harbor"),
        ]
    )

    assert args.student_model == QWEN3_8_27B_MODEL
    assert args.proposer_model == QWEN3_8_27B_MODEL


def test_run_contract_allows_exact_resume_and_rejects_drift(tmp_path: Path) -> None:
    """Accept an exact run contract while rejecting changed resume settings.

    Args:
        tmp_path: Pytest directory used for the isolated run contract.
    """
    contract = {"condition": "react_v2", "student_model": QWEN3_8_27B_MODEL, "edit_tool_set": "broad"}
    path = ensure_run_contract(tmp_path, contract)

    assert ensure_run_contract(tmp_path, contract) == path
    with pytest.raises(ValueError, match="different Terminal-Bench configuration"):
        ensure_run_contract(tmp_path, {**contract, "edit_tool_set": "minimal"})


def test_generated_run_contract_records_metric_call_budget(tmp_path: Path) -> None:
    """Record metric budget and semantic Controller policy in generated state.

    Args:
        tmp_path: Pytest directory used for parsed output paths.
    """
    args = build_parser().parse_args(
        [
            "--condition",
            "react_v2",
            "--max-metric-calls",
            "400",
            "--manifest",
            str(MANIFEST_PATH),
            "--run-dir",
            str(tmp_path / "run"),
            "--harbor-work-dir",
            str(tmp_path / "harbor"),
        ]
    )
    manifest = load_terminalbench_manifest(MANIFEST_PATH)

    contract = build_run_contract(
        args,
        manifest,
        manifest.tasks("train", 1),
        manifest.tasks("val", 1),
        "react_v2",
        "alibaba",
    )

    assert contract["max_metric_calls"] == 400
    assert contract["schema_version"] == 5
    assert contract["api_profile"] == "direct"
    assert contract["component_kinds"] == {"instruction_prompt": "user_prompt"}
    assert contract["student_model"] == QWEN3_8_27B_MODEL
    assert contract["student_runtime_model"] == QWEN3_8_27B_MODEL
    assert contract["student_request_overrides"] == {}
    assert contract["proposer_model"] == QWEN3_8_27B_MODEL
    assert contract["proposer_runtime_model"] == QWEN3_8_27B_MODEL
    assert contract["proposer_request_overrides"] == {}
    assert contract["student_decoding"] == experiment_decoding(QWEN3_8_27B_MODEL)
    assert contract["student_model_info"] == QWEN3_8_27B_MODEL_INFO
    assert contract["proposer_decoding"] == experiment_decoding(QWEN3_8_27B_MODEL)
    assert contract["student_num_retries"] == EXPERIMENT_NUM_RETRIES
    assert contract["proposer_num_retries"] == EXPERIMENT_NUM_RETRIES
    assert contract["semantic_action_space"] == SEMANTIC_ACTION_CATALOGS["prompt"]
    assert contract["semantic_controller_policy"] == CONTROLLER_POLICY_CONTRACT


def test_glm_run_contract_uses_the_separate_same_model_condition(tmp_path: Path) -> None:
    """Record GLM-5.3-Flash in both roles with its fixed decoding.

    Args:
        tmp_path: Pytest directory used for parsed output paths.
    """
    args = _model_args(tmp_path, GLM_5_3_FLASH_MODEL, GLM_5_3_FLASH_MODEL)
    manifest = load_terminalbench_manifest(MANIFEST_PATH)

    contract = build_run_contract(
        args,
        manifest,
        manifest.tasks("train", 1),
        manifest.tasks("val", 1),
        "react_v2",
        "generic",
    )

    assert contract["student_model"] == GLM_5_3_FLASH_MODEL
    assert contract["proposer_model"] == GLM_5_3_FLASH_MODEL
    assert contract["student_decoding"] == experiment_decoding(GLM_5_3_FLASH_MODEL)
    assert contract["student_model_info"] is None
    assert contract["proposer_decoding"] == experiment_decoding(GLM_5_3_FLASH_MODEL)


@pytest.mark.parametrize(
    ("canonical_model", "runtime_model"),
    [
        (QWEN3_8_27B_MODEL, QWEN3_8_27B_OPENROUTER_MODEL),
        (GLM_5_3_FLASH_MODEL, GLM_5_3_FLASH_OPENROUTER_MODEL),
    ],
)
def test_openrouter_profile_reaches_terminalbench_student_and_proposer(
    monkeypatch,
    tmp_path: Path,
    canonical_model: str,
    runtime_model: str,
) -> None:
    """Route Harbor and GEPA through the same pinned OpenRouter runtime.

    Args:
        monkeypatch: Pytest fixture used to replace Harbor and optimization.
        tmp_path: Pytest directory used for isolated run artifacts.
        canonical_model: Scientific model identity assigned to both roles.
        runtime_model: Exact OpenRouter model slug used for completions.
    """
    harbor = Mock()
    harbor_constructor = Mock(return_value=harbor)
    optimize = Mock()
    run_dir = tmp_path / "run"
    monkeypatch.setattr(terminalbench_main, "HarborCLI", harbor_constructor)
    monkeypatch.setattr(terminalbench_main, "optimize", optimize)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "terminalbench",
            "--condition",
            "react_v2",
            "--api-profile",
            "openrouter",
            "--student-model",
            canonical_model,
            "--proposer-model",
            canonical_model,
            "--max-metric-calls",
            "6",
            "--train-limit",
            "1",
            "--val-limit",
            "1",
            "--manifest",
            str(MANIFEST_PATH),
            "--run-dir",
            str(run_dir),
            "--harbor-work-dir",
            str(tmp_path / "harbor"),
        ],
    )

    terminalbench_main.main()

    request_overrides = experiment_request_overrides(runtime_model)
    student_kwargs = harbor_constructor.call_args.kwargs
    assert student_kwargs["student_model"] == runtime_model
    assert student_kwargs["student_agent_kwargs"]["llm_kwargs"] == {
        "num_retries": EXPERIMENT_NUM_RETRIES,
        **experiment_decoding(runtime_model),
        **request_overrides,
    }
    assert "model_info" not in student_kwargs["student_agent_kwargs"]
    harbor.check_requirements.assert_called_once_with()

    proposer_kwargs = optimize.call_args.kwargs
    assert proposer_kwargs["reflection_lm"] == runtime_model
    assert proposer_kwargs["reflection_lm_kwargs"] == {
        "num_retries": EXPERIMENT_NUM_RETRIES,
        **experiment_decoding(runtime_model),
        **request_overrides,
    }
    assert proposer_kwargs["template_model"] == canonical_model

    contract = json.loads((run_dir / terminalbench_main.RUN_CONTRACT_FILENAME).read_text())
    assert contract["api_profile"] == "openrouter"
    assert contract["student_model"] == canonical_model
    assert contract["student_runtime_model"] == runtime_model
    assert contract["student_request_overrides"] == request_overrides
    assert contract["proposer_model"] == canonical_model
    assert contract["proposer_runtime_model"] == runtime_model
    assert contract["proposer_request_overrides"] == request_overrides


def test_run_contract_rejects_a_cross_model_pair(tmp_path: Path) -> None:
    """Reject a Qwen student paired with the GLM proposer.

    Args:
        tmp_path: Pytest directory used for parsed output paths.
    """
    args = _model_args(tmp_path, QWEN3_8_27B_MODEL, GLM_5_3_FLASH_MODEL)
    manifest = load_terminalbench_manifest(MANIFEST_PATH)

    with pytest.raises(ValueError, match="same model"):
        build_run_contract(
            args,
            manifest,
            manifest.tasks("train", 1),
            manifest.tasks("val", 1),
            "react_v2",
            "alibaba",
        )


def test_run_contract_rejects_an_unknown_model_pair(tmp_path: Path) -> None:
    """Reject homogeneous models outside the two configured experiment arms.

    Args:
        tmp_path: Pytest directory used for parsed output paths.
    """
    args = _model_args(tmp_path, "provider/unknown", "provider/unknown")
    manifest = load_terminalbench_manifest(MANIFEST_PATH)

    with pytest.raises(ValueError, match="Unsupported experiment model"):
        build_run_contract(
            args,
            manifest,
            manifest.tasks("train", 1),
            manifest.tasks("val", 1),
            "react_v2",
            "generic",
        )


def test_legacy_state_without_contract_is_not_resumed(tmp_path: Path) -> None:
    """Reject legacy GEPA state that lacks a Terminal-Bench contract.

    Args:
        tmp_path: Pytest directory containing simulated legacy state.
    """
    (tmp_path / "gepa_state.bin").write_bytes(b"old-state")

    with pytest.raises(ValueError, match="no terminalbench-run-contract.json"):
        ensure_run_contract(tmp_path, {"condition": "react_v2"})
