"""Offline checks for frozen Terminal-Bench comparisons and repeated Pass@1."""

from __future__ import annotations

import json
import sys
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
from terminalbench_pilot_helpers import offline_runtime as offline_runtime
from terminalbench_pilot_helpers import write_pilot_fixture

sys.path.insert(0, str(Path(__file__).parents[1]))

from examples.common.experiment_models import DEEPSEEK_V4_1_FLASH_MODEL, QWEN3_8_27B_MODEL
from examples.terminalbench import evaluate
from examples.terminalbench.main import (
    EXPERIMENT_MANIFESTS,
    RUN_CONTRACT_FILENAME,
    SCOPE_CAMPAIGN_CELLS,
    build_parser,
    build_run_contract,
    ensure_run_contract,
    seed_candidate,
)
from examples.terminalbench.pilot import review_pilot
from examples.terminalbench.runtime import _digest
from gepa.adapters.terminal_bench_adapter import (
    HarborEvaluation,
    HarborExecutionError,
    HarborTrialResult,
    load_terminalbench_manifest,
)
from gepa.adapters.terminal_bench_adapter.documents import render_instruction
from gepa.adapters.terminal_bench_adapter.text_scope import TerminalBenchTextScope
from gepa.core.state import GEPAState, ValsetEvaluation
from gepa.strategies.text_limits import TextLimits


def _write_run(
    root: Path,
    experiment: str,
    condition: str,
    model: str = QWEN3_8_27B_MODEL,
    budget: str = "standard",
    text_limits: TextLimits | None = None,
    n_concurrent: int = 1,
    optimization_scope: str = "all_text",
) -> Path:
    """Create a real checkpoint whose validation winner is not the last candidate."""
    label = f"{optimization_scope}__{condition}{'_2x' if budget == 'double' else ''}"
    run_dir = root / label
    args = build_parser().parse_args(
        [
            "--experiment",
            experiment,
            "--optimization-scope",
            optimization_scope,
            "--condition",
            condition,
            "--budget",
            budget,
            "--student-model",
            model,
            "--proposer-model",
            model,
            "--student-api-base",
            "http://localhost:8000/v1",
            "--proposer-api-base",
            "http://localhost:8000/v1",
            "--run-dir",
            str(run_dir),
            "--harbor-work-dir",
            str(run_dir / "harbor"),
            "--text-limits",
            json.dumps(text_limits.to_dict() if text_limits else None),
            "--n-concurrent",
            str(n_concurrent),
        ]
    )
    manifest = load_terminalbench_manifest(EXPERIMENT_MANIFESTS[experiment])
    initial, family = seed_candidate(model, "auto", experiment, optimization_scope)
    contract = build_run_contract(args, manifest, manifest.tasks("train"), manifest.tasks("val"), condition, family)
    pilot_dir = write_pilot_fixture(root / "pilot", contract, manifest)
    contract["pilot_review"] = review_pilot(pilot_dir, contract, manifest)
    ensure_run_contract(run_dir, contract)
    count = len(manifest.splits["val"])
    state = GEPAState(initial, ValsetEvaluation({}, dict.fromkeys(range(count), 0.0)))
    state.i = contract["optimization_budget"]["max_iterations"] - 1
    state.total_num_evals = 1000
    state.num_full_ds_evals = 3
    component = next(iter(initial))
    for name, score in [("winner", 1.0), ("last", 0.0)]:
        candidate = {**initial, component: initial[component] + f"\n{label}-{name}"}
        state.update_state_with_new_program(
            [0],
            candidate,
            ValsetEvaluation({}, dict.fromkeys(range(count), score)),
            None,
            100,
            iteration_id=f"{condition}-{name}",
        )
    state.save(str(run_dir))
    return run_dir


def _write_comparison(
    root: Path,
    experiment: str,
    model: str = QWEN3_8_27B_MODEL,
    text_limits: TextLimits | None = None,
    n_concurrent: int = 1,
) -> dict[str, Path]:
    """Create all twelve scope/method/budget source checkpoints."""
    return {
        label: _write_run(root, experiment, condition, model, budget, text_limits, n_concurrent, scope)
        for label, (scope, condition, budget) in SCOPE_CAMPAIGN_CELLS.items()
    }


@pytest.mark.parametrize("damage", ["missing", "changed_proposer"])
def test_final_matrix_requires_matching_material_runtimes(tmp_path: Path, damage: str) -> None:
    """Reject old runtime-free runs and different optimizer-server configurations."""
    run_dirs = _write_comparison(tmp_path, "tb2.1")
    path = run_dirs["all_text__react_v2"] / RUN_CONTRACT_FILENAME
    contract = json.loads(path.read_text())
    if damage == "missing":
        contract.pop("execution_runtime")
    else:
        identity = contract["execution_runtime"]["proposer"]
        identity["parallelism"]["data"] = 2
        identity["sha256"] = _digest({key: value for key, value in identity.items() if key != "sha256"})
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError):
        evaluate.freeze_comparison(run_dirs)


def _fake_runner(manifest, comparison, output_dir: Path, *, fail_on_call: int | None = None) -> Mock:
    """Return distinct Harbor jobs with repetition scores zero, one-half, and one."""
    completed = Counter()
    attempts = 0
    by_digest = {harness["candidate_digest"]: label for label, harness in comparison["harnesses"].items()}

    def run(task_ids, candidate):
        """Require freezing first and emit one binary reward per held-out task."""
        nonlocal attempts
        attempts += 1
        frozen = json.loads((output_dir / evaluate.FROZEN_COMPARISON_FILENAME).read_text())
        assert frozen["shared_configuration"] == comparison["shared_configuration"]
        assert all(value == comparison["harnesses"][label] for label, value in frozen["harnesses"].items())
        assert task_ids == manifest.splits["test"]
        if attempts == fail_on_call:
            raise HarborExecutionError("simulated interrupted Harbor job")
        digest = manifest.candidate_digest(candidate)
        assert digest in {harness["candidate_digest"] for harness in frozen["harnesses"].values()}
        label = by_digest[digest]
        completed[label] += 1
        passes = (completed[label] - 1) * len(task_ids) // 2
        trials = {
            task_id: HarborTrialResult(
                task_id=task_id,
                reward=float(index < passes),
                rewards={"reward": float(index < passes)},
                errors=[],
                atif_trajectories=[],
                raw_result={},
                trial_dir=output_dir / f"trial-{index}",
            )
            for index, task_id in enumerate(task_ids)
        }
        return HarborEvaluation(
            evaluation_id=f"evaluation-{attempts}",
            candidate_digest=digest,
            config_path=output_dir / f"job-{attempts}.json",
            job_dir=output_dir / f"job-{attempts}",
            returncode=0,
            stdout_path=output_dir / "stdout.log",
            stderr_path=output_dir / "stderr.log",
            trials=trials,
        )

    return Mock(manifest=manifest, run=Mock(side_effect=run))


@pytest.mark.parametrize("experiment", EXPERIMENT_MANIFESTS)
@pytest.mark.parametrize("model", [QWEN3_8_27B_MODEL, DEEPSEEK_V4_1_FLASH_MODEL])
@pytest.mark.parametrize("configured", [False, True])
def test_evaluation_cli_freezes_validation_winners_and_repeats_test_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, experiment: str, model: str, configured: bool
) -> None:
    """Test all benchmark/model arms through the CLI without any optimization or model call."""
    text_limits = TextLimits(verifier_log_chars=5000, manifestor_steering_chars=3000) if configured else TextLimits()
    n_concurrent = 2 if configured else 1
    run_dirs = _write_comparison(tmp_path, experiment, model, text_limits, n_concurrent)
    manifest, comparison = evaluate.freeze_comparison(run_dirs)
    for condition in SCOPE_CAMPAIGN_CELLS:
        assert comparison["source_runs"][condition]["selected_candidate_index"] == 1
        epochs = 8 if condition.endswith("_2x") else 4
        assert comparison["source_runs"][condition]["contract"]["optimization_budget"]["training_epochs"] == epochs
        assert any(f"{condition}-winner" in text for text in comparison["harnesses"][condition]["documents"].values())
    output_dir = tmp_path / "test"
    runner = _fake_runner(manifest, comparison, output_dir)
    adapter = evaluate.TerminusAdapter(manifest, runner, text_scope=TerminalBenchTextScope("all_text"))
    adapter_evaluate = Mock(wraps=adapter.evaluate)
    monkeypatch.setattr(adapter, "evaluate", adapter_evaluate)
    monkeypatch.setattr(evaluate, "TerminusAdapter", Mock(return_value=adapter))
    factory = Mock(return_value=runner)
    monkeypatch.setattr(evaluate, "HarborCLI", factory)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            *[argument for label, path in run_dirs.items() for argument in ("--run-dir", f"{label}={path}")],
            "--output-dir",
            str(output_dir),
        ],
    )

    evaluate.main()
    assert runner.run.call_count == 39
    assert adapter_evaluate.call_count == 39
    assert all(call.args[0] == manifest.tasks("test") for call in adapter_evaluate.call_args_list)
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["complete"] is True
    assert summary["protocol"]["optimization_runs_per_configuration"] == 1
    assert summary["protocol"]["attempts_per_task_per_repetition"] == 1
    for scores in summary["harnesses"].values():
        assert scores["repetition_pass_at_1"] == [0.0, 0.5, 1.0]
        assert scores["mean_pass_at_1"] == scores["std_pass_at_1"] == 0.5
        assert scores["task_attempts"] == 120
    assert len(summary["harnesses"]) == 13
    assert {row["optimization_scope"] for row in summary["harnesses"].values()} == {
        "all_text",
        "system_prompt",
        "reference",
    }
    kwargs = factory.call_args.kwargs
    contract = comparison["shared_configuration"]
    assert contract["text_limits"] == text_limits.to_dict()
    assert kwargs["text_limits"] == text_limits
    assert kwargs["n_concurrent"] == contract["n_concurrent"] == n_concurrent
    assert kwargs["student_model"] == model
    assert kwargs["student_api_base"] == contract["student_api_base"]
    assert kwargs["student_agent_kwargs"]["model_info"] == contract["student_model_info"]
    assert kwargs["student_agent_kwargs"]["token_limits"] == contract["token_limits"]
    assert kwargs["student_agent_kwargs"]["llm_kwargs"]["max_tokens"] == 32_768
    assert kwargs["student_agent_kwargs"]["llm_kwargs"] == {
        "num_retries": contract["student_num_retries"],
        **contract["student_decoding"],
        **contract["student_request_overrides"],
    }
    assert kwargs["student_agent_kwargs"]["llm_kwargs"]["extra_body"]["chat_template_kwargs"] == (
        {"enable_thinking": True, "reasoning_effort": "medium"}
        if model == QWEN3_8_27B_MODEL
        else {"thinking": True, "reasoning_effort": 75}
    )
    evaluate.main()
    assert runner.run.call_count == 39
    assert adapter_evaluate.call_count == 39


@pytest.mark.parametrize(
    "specifications",
    [
        ["vanilla"],
        ["vanilla="],
        ["unknown=/tmp/run"],
        ["all_text__vanilla=/tmp/first", "all_text__vanilla=/tmp/second"],
    ],
)
def test_evaluation_cli_rejects_invalid_or_duplicate_cell_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, specifications: list[str]
) -> None:
    """Reject ambiguous source mappings before reading runs or constructing Harbor."""
    freeze = Mock()
    harbor = Mock()
    monkeypatch.setattr(evaluate, "freeze_comparison", freeze)
    monkeypatch.setattr(evaluate, "HarborCLI", harbor)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            *[argument for specification in specifications for argument in ("--run-dir", specification)],
            "--output-dir",
            str(tmp_path / "test"),
        ],
    )
    with pytest.raises(SystemExit):
        evaluate.main()
    freeze.assert_not_called()
    harbor.assert_not_called()


def test_interrupted_evaluation_resumes_only_missing_repetitions(tmp_path: Path) -> None:
    """Keep completed attempts unchanged and withhold a summary until all 39 jobs finish."""
    run_dirs = _write_comparison(tmp_path, "tb2.1")
    manifest, comparison = evaluate.freeze_comparison(run_dirs)
    output_dir = tmp_path / "test"
    runner = _fake_runner(manifest, comparison, output_dir, fail_on_call=5)

    with pytest.raises(HarborExecutionError, match="interrupted"):
        evaluate.evaluate_comparison(manifest, comparison, output_dir, runner)
    saved = {path.name: path.read_bytes() for path in output_dir.glob("*-repetition-*.json")}
    assert len(saved) == 4
    assert not (output_dir / "summary.json").exists()
    summary = evaluate.evaluate_comparison(manifest, comparison, output_dir, runner)
    assert runner.run.call_count == 40
    assert summary["complete"] is True
    assert all((output_dir / name).read_bytes() == content for name, content in saved.items())
    assert all(row["mean_pass_at_1"] == 0.5 for row in summary["harnesses"].values())


def test_unchanged_winners_still_receive_separate_test_repetitions(tmp_path: Path) -> None:
    """Preserve fresh attempts when validation selects the initial harness in all twelve runs."""
    run_dirs = _write_comparison(tmp_path, "tb2.1")
    for run_dir in run_dirs.values():
        state = GEPAState.load(str(run_dir))
        state.prog_candidate_val_subscores[1] = dict.fromkeys(state.prog_candidate_val_subscores[1], 0.0)
        state.save(str(run_dir))
    manifest, comparison = evaluate.freeze_comparison(run_dirs)
    assert all(run["selected_candidate_index"] == 0 for run in comparison["source_runs"].values())
    assert len({render_instruction(harness["documents"]) for harness in comparison["harnesses"].values()}) == 1
    output_dir = tmp_path / "test"
    runner = _fake_runner(manifest, comparison, output_dir)
    summary = evaluate.evaluate_comparison(manifest, comparison, output_dir, runner)
    assert runner.run.call_count == 39
    assert len(list(output_dir.glob("*-repetition-*.json"))) == 39
    assert all(row["task_attempts"] == 120 for row in summary["harnesses"].values())


@pytest.mark.parametrize(
    "damage",
    [
        "incomplete",
        "partial_validation",
        "partial_train",
        "different_seed",
        "different_model",
        "different_selector",
        "wrong_budget",
        "incomplete_double",
        "different_condition",
        "different_policy",
        "different_feedback",
        "different_failure_policy",
        "different_context_policy",
        "manifestor_trace_cap",
        "manifestor_temperature",
        "controller_top_p",
        "manifestor_top_p",
        "react_v2_proposer_top_p",
        "missing_failure_policy",
    ],
)
def test_invalid_source_runs_are_rejected_before_test_execution(tmp_path: Path, damage: str) -> None:
    """Reject unfinished optimization, pilot splits, and unmatched experimental settings."""
    run_dirs = _write_comparison(tmp_path, "tb2.1")
    forest = run_dirs["all_text__react_v2"]
    if damage == "different_model":
        forest = _write_run(tmp_path / "other-model", "tb2.1", "react_v2", DEEPSEEK_V4_1_FLASH_MODEL)
        run_dirs["all_text__react_v2"] = forest
    if damage == "wrong_budget":
        run_dirs["all_text__react_v2_2x"] = forest
    elif damage == "incomplete_double":
        state = GEPAState.load(str(run_dirs["all_text__react_v2_2x"]))
        state.i = 31
        state.save(str(run_dirs["all_text__react_v2_2x"]))
    if damage in {"incomplete", "partial_validation"}:
        state = GEPAState.load(str(forest))
        if damage == "incomplete":
            state.i -= 1
        else:
            del state.prog_candidate_val_subscores[1][0]
        state.save(str(forest))
    elif damage in {
        "partial_train",
        "different_seed",
        "different_selector",
        "different_condition",
        "different_policy",
        "different_feedback",
        "different_failure_policy",
        "different_context_policy",
        "manifestor_trace_cap",
        "manifestor_temperature",
        "controller_top_p",
        "manifestor_top_p",
        "react_v2_proposer_top_p",
        "missing_failure_policy",
    }:
        path = forest / RUN_CONTRACT_FILENAME
        contract = json.loads(path.read_text())
        if damage == "partial_train":
            contract["train_task_ids"].pop()
        elif damage == "different_selector":
            contract["module_selector"] = "round_robin"
        elif damage == "different_condition":
            contract["condition"] = "action"
        elif damage == "different_policy":
            contract["controller_selection"] = "uniform_random"
        elif damage == "different_feedback":
            contract["reflection_feedback"]["max_chars_per_verifier_log"] = 2048
        elif damage == "different_failure_policy":
            contract["failure_policy"]["harbor_max_retries"] = 1
        elif damage == "different_context_policy":
            contract["reflection_context"]["duplicates"] = "exact_text_and_paragraph_references_within_each_prompt"
        elif damage == "manifestor_trace_cap":
            contract["manifestor_traces_chars"] = 8000
        elif damage == "manifestor_temperature":
            contract["manifestor_temperature"] = 0.0
        elif damage.endswith("_top_p"):
            contract["reflection_role_decoding"][damage.removesuffix("_top_p")]["requested"]["top_p"] = 0.5
        elif damage == "missing_failure_policy":
            del contract["failure_policy"]
        else:
            contract["seed"] = 19
        path.write_text(json.dumps(contract))
    with pytest.raises(ValueError):
        evaluate.freeze_comparison(run_dirs)


@pytest.mark.parametrize("experiment", EXPERIMENT_MANIFESTS)
@pytest.mark.parametrize("damage", ["disabled", "changed_threshold", "missing"])
def test_task_context_drift_cannot_resume_or_enter_final_test(tmp_path: Path, experiment: str, damage: str) -> None:
    """Reject disabled, changed, or unrecorded task summarization across methods."""
    run_dirs = _write_comparison(tmp_path, experiment)
    forest = run_dirs["all_text__react_v2"]
    path = forest / RUN_CONTRACT_FILENAME
    original = json.loads(path.read_text())
    changed = json.loads(path.read_text())
    if damage == "missing":
        del changed["task_context_settings"]
    elif damage == "disabled":
        changed["task_context_settings"]["enable_summarize"] = False
    else:
        changed["task_context_settings"]["proactive_summarization_threshold"] = 0
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="different Terminal-Bench configuration"):
        ensure_run_contract(forest, original)
    with pytest.raises(ValueError, match="expected a matching react_v2 run"):
        evaluate.freeze_comparison(run_dirs)


@pytest.mark.parametrize("missing", [False, True])
def test_provider_retry_drift_cannot_resume_or_enter_final_test(tmp_path: Path, missing: bool) -> None:
    """Reject absent or changed provider-attempt policies before any task runs."""
    run_dirs = _write_comparison(tmp_path, "tb2.1")
    forest = run_dirs["all_text__react_v2"]
    path = forest / RUN_CONTRACT_FILENAME
    original = json.loads(path.read_text())
    changed = json.loads(path.read_text())
    assert original["provider_retry_policy"]["max_attempts"] == 3
    if missing:
        del changed["provider_retry_policy"]
    else:
        changed["provider_retry_policy"]["max_attempts"] = 9
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="different Terminal-Bench configuration"):
        ensure_run_contract(forest, original)
    with pytest.raises(ValueError, match="expected a matching react_v2 run"):
        evaluate.freeze_comparison(run_dirs)


@pytest.mark.parametrize("damage", ["missing", "entry_point", "implementation", "upstream_commit"])
def test_adapter_drift_cannot_resume_or_enter_final_test(tmp_path: Path, damage: str) -> None:
    """Reject runs that omit or change the adapter implementation or upstream provenance."""
    run_dirs = _write_comparison(tmp_path, "tb2.1")
    forest = run_dirs["all_text__react_v2"]
    path = forest / RUN_CONTRACT_FILENAME
    original = json.loads(path.read_text())
    changed = json.loads(path.read_text())
    if damage == "missing":
        del changed["adapter"]
    else:
        changed["adapter"][damage] = "different-adapter"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="different Terminal-Bench configuration"):
        ensure_run_contract(forest, original)
    with pytest.raises(ValueError, match="expected a matching react_v2 run"):
        evaluate.freeze_comparison(run_dirs)


@pytest.mark.parametrize("damage", ["missing_scope", "changed_scope", "swapped_label", "extra_component"])
def test_final_comparison_enforces_both_scope_boundaries(tmp_path: Path, damage: str) -> None:
    """Reject mislabeled scope arms and expanded prompt-only edits."""
    run_dirs = _write_comparison(tmp_path, "tb2.1")
    run_dir = run_dirs["system_prompt__react_v2"]
    if damage == "swapped_label":
        run_dirs["all_text__react_v2"] = run_dir
    elif damage == "extra_component":
        state = GEPAState.load(str(run_dir))
        state.program_candidates[1]["skill_debugging"] = "Changed fixed skill."
        state.save(str(run_dir))
    else:
        path = run_dir / RUN_CONTRACT_FILENAME
        original = json.loads(path.read_text())
        changed = dict(original)
        if damage == "missing_scope":
            del changed["optimization_scope"]
        else:
            changed["optimization_scope"] = "all_text"
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="different Terminal-Bench configuration"):
            ensure_run_contract(run_dir, original)
    with pytest.raises(ValueError):
        evaluate.freeze_comparison(run_dirs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("cache_evaluation", True),
        ("candidate_selection_strategy", "current_best"),
        ("frontier_type", "objective"),
        ("acceptance_criterion", "improvement_or_equal"),
        ("skip_perfect_score", False),
        ("perfect_score", 0.5),
        ("validation_evaluation", "partial"),
        ("training_batch_order", {"rng_stream": "shared"}),
        ("pilot_protocol", {}),
    ],
)
@pytest.mark.parametrize("missing", [False, True])
def test_optimizer_policy_drift_cannot_resume_or_enter_final_test(
    tmp_path: Path, field: str, value: str | bool | float | dict, missing: bool
) -> None:
    """Reject changed optimizer policies or training order before held-out work."""
    run_dirs = _write_comparison(tmp_path, "tb2.1")
    forest = run_dirs["all_text__react_v2"]
    path = forest / RUN_CONTRACT_FILENAME
    original = json.loads(path.read_text())
    changed = json.loads(path.read_text())
    if missing:
        del changed[field]
    else:
        changed[field] = value
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="different Terminal-Bench configuration"):
        ensure_run_contract(forest, original)
    with pytest.raises(ValueError, match="expected a matching react_v2 run"):
        evaluate.freeze_comparison(run_dirs)


@pytest.mark.parametrize("experiment", ["tb2", "tb4"])
def test_removed_benchmark_cannot_enter_final_comparison(tmp_path: Path, experiment: str) -> None:
    """Reject obsolete dataset identities before loading any selected harness."""
    run_dirs = _write_comparison(tmp_path, "tb2.1")
    path = run_dirs["all_text__react_v2"] / RUN_CONTRACT_FILENAME
    contract = json.loads(path.read_text())
    contract["experiment"] = experiment
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match=r"only Terminal-Bench 2\.1 runs"):
        evaluate.freeze_comparison(run_dirs)


@pytest.mark.parametrize("damage", ["missing", "unreviewed", "partial", "missing_smoke"])
def test_final_comparison_requires_both_reviewed_pilot_stages(tmp_path: Path, damage: str) -> None:
    """Reject completed optimization checkpoints without the agreed training pilot evidence."""
    run_dirs = _write_comparison(tmp_path, "tb2.1")
    path = run_dirs["system_prompt__vanilla"] / RUN_CONTRACT_FILENAME
    contract = json.loads(path.read_text())
    if damage == "missing":
        contract["pilot_review"] = None
    elif damage == "unreviewed":
        contract["pilot_review"]["reviewed_metrics"] = []
    elif damage == "partial":
        contract["pilot_review"]["full_pilot"]["config"]["task_ids"].pop()
    else:
        contract["pilot_review"]["full_pilot"]["config"]["smoke_evidence"] = None
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError):
        evaluate.freeze_comparison(run_dirs)


@pytest.mark.parametrize("role", ["student", "proposer"])
@pytest.mark.parametrize("damage", ["implicit", "lower_effort", "thinking_disabled"])
def test_reasoning_changes_are_rejected_before_final_test(tmp_path: Path, role: str, damage: str) -> None:
    """Reject implicit or changed Qwen reasoning settings before freezing test candidates."""
    run_dirs = _write_comparison(tmp_path, "tb2.1")
    path = run_dirs["all_text__react_v2"] / RUN_CONTRACT_FILENAME
    contract = json.loads(path.read_text())
    if damage == "implicit":
        contract[f"{role}_request_overrides"] = {}
    else:
        template_kwargs = contract[f"{role}_request_overrides"]["extra_body"]["chat_template_kwargs"]
        if damage == "lower_effort":
            template_kwargs["reasoning_effort"] = "low"
        else:
            template_kwargs["enable_thinking"] = False
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="expected a matching react_v2 run"):
        evaluate.freeze_comparison(run_dirs)


@pytest.mark.parametrize(
    "field", ["student_decoding", "proposer_decoding", "student_model_info", "token_limits", "token_usage_policy"]
)
def test_changed_output_budget_or_usage_policy_cannot_resume_or_enter_final_test(tmp_path: Path, field: str) -> None:
    """Reject old caps or missing telemetry policy before any held-out evaluation."""
    run_dirs = _write_comparison(tmp_path, "tb2.1")
    forest = run_dirs["all_text__react_v2"]
    path = forest / RUN_CONTRACT_FILENAME
    original = json.loads(path.read_text())
    changed = json.loads(path.read_text())
    if field.endswith("decoding"):
        changed[field]["max_tokens"] = 16_384
    elif field == "token_usage_policy":
        del changed[field]
    else:
        changed[field]["max_output_tokens"] = 16_384
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="different Terminal-Bench configuration"):
        ensure_run_contract(forest, original)
    with pytest.raises(ValueError, match="expected a matching react_v2 run"):
        evaluate.freeze_comparison(run_dirs)


@pytest.mark.parametrize("experiment", EXPERIMENT_MANIFESTS)
@pytest.mark.parametrize(
    "policy,field,value",
    [
        ("react_execution", "max_iterations", 8),
        ("react_execution", "max_tool_calls", 4),
        ("react_execution", "completion", "first_successful_edit"),
        ("react_execution", "scope", "whole_document"),
        ("document_length", "max_component_chars", 10000),
        ("document_length", "selector_target_chars", 8000),
    ],
)
def test_old_editor_limits_or_scope_cannot_enter_final_comparison(
    tmp_path: Path, experiment: str, policy: str, field: str, value: object
) -> None:
    """Reject changed editor limits, length targets, completion, or section scope."""
    run_dirs = _write_comparison(tmp_path, experiment)
    path = run_dirs["all_text__react_v2"] / RUN_CONTRACT_FILENAME
    contract = json.loads(path.read_text())
    contract[policy][field] = value
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError):
        evaluate.freeze_comparison(run_dirs)


@pytest.mark.parametrize("experiment", EXPERIMENT_MANIFESTS)
@pytest.mark.parametrize("field", TextLimits.__dataclass_fields__)
def test_changed_character_limits_cannot_resume_or_enter_final_comparison(
    tmp_path: Path, experiment: str, field: str
) -> None:
    """Reject changes to any character setting before using a saved run."""
    run_dirs = _write_comparison(tmp_path, experiment)
    forest = run_dirs["all_text__react_v2"]
    path = forest / RUN_CONTRACT_FILENAME
    original = json.loads(path.read_text())
    changed = json.loads(path.read_text())
    changed["text_limits"][field] = 1234
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="different Terminal-Bench configuration"):
        ensure_run_contract(forest, original)
    with pytest.raises(ValueError):
        evaluate.freeze_comparison(run_dirs)


@pytest.mark.parametrize("experiment", EXPERIMENT_MANIFESTS)
def test_changed_concurrency_cannot_resume_or_enter_final_comparison(tmp_path: Path, experiment: str) -> None:
    """Keep task concurrency fixed within every benchmark/model comparison."""
    run_dirs = _write_comparison(tmp_path, experiment)
    forest = run_dirs["all_text__react_v2"]
    path = forest / RUN_CONTRACT_FILENAME
    original = json.loads(path.read_text())
    changed = {**original, "n_concurrent": original["n_concurrent"] + 1}
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="different Terminal-Bench configuration"):
        ensure_run_contract(forest, original)
    with pytest.raises(ValueError):
        evaluate.freeze_comparison(run_dirs)


def test_frozen_output_rejects_a_changed_validation_winner(tmp_path: Path) -> None:
    """Prevent replacing an optimized harness after test feedback has been observed."""
    run_dirs = _write_comparison(tmp_path, "tb2.1")
    forest = run_dirs["all_text__react_v2_2x"]
    manifest, comparison = evaluate.freeze_comparison(run_dirs)
    output_dir = tmp_path / "test"
    runner = _fake_runner(manifest, comparison, output_dir, fail_on_call=2)
    with pytest.raises(HarborExecutionError):
        evaluate.evaluate_comparison(manifest, comparison, output_dir, runner)
    state = GEPAState.load(str(forest))
    state.program_candidates[1]["instruction_prompt"] += "\nChanged after testing"
    state.save(str(forest))
    _, changed = evaluate.freeze_comparison(run_dirs)
    with pytest.raises(ValueError, match="different frozen comparison"):
        evaluate.evaluate_comparison(manifest, changed, output_dir, runner)
    assert runner.run.call_count == 2


@pytest.mark.parametrize("cell", SCOPE_CAMPAIGN_CELLS)
def test_one_completed_ablation_can_test_before_the_other_runs_exist(tmp_path: Path, cell: str) -> None:
    """Evaluate the common baseline and one validation winner on exactly the same test tasks."""
    scope, condition, budget = SCOPE_CAMPAIGN_CELLS[cell]
    run_dir = _write_run(tmp_path, "tb2.1", condition, budget=budget, optimization_scope=scope)
    manifest, comparison = evaluate.freeze_comparison({cell: run_dir})
    assert set(comparison["source_runs"]) == {cell}
    output_dir = tmp_path / "test"
    runner = _fake_runner(manifest, comparison, output_dir)
    summary = evaluate.evaluate_comparison(manifest, comparison, output_dir, runner)
    assert runner.run.call_count == 6
    assert set(summary["harnesses"]) == {"initial", cell}
    assert summary["complete"] is True
    assert summary["campaign_complete"] is False
    assert summary["completed_cells"] == [cell]
    assert set(summary["pending_cells"]) == set(SCOPE_CAMPAIGN_CELLS) - {cell}
    assert all(row["task_attempts"] == 120 for row in summary["harnesses"].values())


def test_incremental_campaign_reuses_baseline_and_keeps_every_cell_on_identical_data(tmp_path: Path) -> None:
    """Add twelve cells one by one without repeating, replacing, or dropping completed test evidence."""
    run_dirs = _write_comparison(tmp_path, "tb2.1")
    manifest, full_comparison = evaluate.freeze_comparison(run_dirs)
    output_dir = tmp_path / "test"
    runner = _fake_runner(manifest, full_comparison, output_dir)
    saved = {}
    for index, (cell, run_dir) in enumerate(run_dirs.items(), start=1):
        _, comparison = evaluate.freeze_comparison({cell: run_dir})
        summary = evaluate.evaluate_comparison(manifest, comparison, output_dir, runner)
        assert runner.run.call_count == 3 * (index + 1)
        assert len(summary["completed_cells"]) == index
        assert summary["campaign_complete"] is (index == len(run_dirs))
        assert len(summary["pending_cells"]) == len(run_dirs) - index
        assert all((output_dir / name).read_bytes() == value for name, value in saved.items())
        saved = {path.name: path.read_bytes() for path in output_dir.glob("*-repetition-*.json")}
        assert evaluate.evaluate_comparison(manifest, comparison, output_dir, runner) == summary
        assert runner.run.call_count == 3 * (index + 1)
    assert json.loads((output_dir / evaluate.FROZEN_COMPARISON_FILENAME).read_text()) == full_comparison
    assert len(saved) == 39


def test_new_ablation_cannot_change_shared_settings_after_earlier_testing(tmp_path: Path) -> None:
    """Keep cross-ablation matching checks when cells arrive in separate invocations."""
    first = _write_run(tmp_path, "tb2.1", "vanilla", optimization_scope="system_prompt")
    manifest, comparison = evaluate.freeze_comparison({"system_prompt__vanilla": first})
    output_dir = tmp_path / "test"
    runner = _fake_runner(manifest, comparison, output_dir)
    evaluate.evaluate_comparison(manifest, comparison, output_dir, runner)
    frozen = (output_dir / evaluate.FROZEN_COMPARISON_FILENAME).read_bytes()
    changed = _write_run(tmp_path, "tb2.1", "react_v2", n_concurrent=2, optimization_scope="all_text")
    _, incoming = evaluate.freeze_comparison({"all_text__react_v2": changed})
    with pytest.raises(ValueError, match="shared configuration changed"):
        evaluate.evaluate_comparison(manifest, incoming, output_dir, runner)
    assert runner.run.call_count == 6
    assert (output_dir / evaluate.FROZEN_COMPARISON_FILENAME).read_bytes() == frozen


def test_interrupted_new_cell_invalidates_previous_summary_and_resumes(tmp_path: Path) -> None:
    """Never report the expanded comparison complete while the new cell's tests are unfinished."""
    run_dirs = _write_comparison(tmp_path, "tb2.1")
    manifest, full_comparison = evaluate.freeze_comparison(run_dirs)
    output_dir = tmp_path / "test"
    runner = _fake_runner(manifest, full_comparison, output_dir, fail_on_call=8)
    first, second = list(run_dirs)[:2]
    _, comparison = evaluate.freeze_comparison({first: run_dirs[first]})
    evaluate.evaluate_comparison(manifest, comparison, output_dir, runner)
    saved = {path.name: path.read_bytes() for path in output_dir.glob("*-repetition-*.json")}
    _, incoming = evaluate.freeze_comparison({second: run_dirs[second]})
    with pytest.raises(HarborExecutionError):
        evaluate.evaluate_comparison(manifest, incoming, output_dir, runner)
    assert not (output_dir / "summary.json").exists()
    summary = evaluate.evaluate_comparison(manifest, incoming, output_dir, runner)
    assert runner.run.call_count == 10
    assert summary["completed_cells"] == [first, second]
    assert all((output_dir / name).read_bytes() == value for name, value in saved.items())


@pytest.mark.parametrize("field", ["dataset", "task_refs", "train", "val", "test"])
def test_testing_rejects_changed_data_or_split_order_before_running_tasks(tmp_path: Path, field: str) -> None:
    """Equal split sizes cannot hide changed task content, membership, or ordering."""
    run_dir = _write_run(tmp_path, "tb2.1", "vanilla", optimization_scope="system_prompt")
    manifest, comparison = evaluate.freeze_comparison({"system_prompt__vanilla": run_dir})
    if field in ("train", "val", "test"):
        splits = deepcopy(manifest.splits)
        splits[field].reverse()
        changed = replace(manifest, splits=splits)
    elif field == "task_refs":
        refs = dict(manifest.task_refs)
        refs[manifest.splits["test"][0]] = "changed-task-content"
        changed = replace(manifest, task_refs=refs)
    else:
        changed = replace(manifest, dataset={**manifest.dataset, "registry_content_hash": "changed-dataset"})
    runner = Mock()
    with pytest.raises(ValueError, match="identical train/validation/test splits"):
        evaluate.evaluate_comparison(changed, comparison, tmp_path / "test", runner)
    runner.run.assert_not_called()
    assert not (tmp_path / "test").exists()


@pytest.mark.parametrize("run_dirs", [{}, {"unknown": Path("unused")}])
def test_freezing_requires_at_least_one_known_cell(run_dirs: dict[str, Path]) -> None:
    """Reject empty and mislabeled requests before loading any checkpoints."""
    with pytest.raises(ValueError, match="one or more supported campaign cells"):
        evaluate.freeze_comparison(run_dirs)


@pytest.mark.parametrize(
    "damage", ["missing_task", "nonbinary", "wrong_candidate", "wrong_repetition", "duplicate_job"]
)
def test_corrupt_saved_repetition_is_not_silently_reused(tmp_path: Path, damage: str) -> None:
    """Reject incomplete, mismatched, or duplicate test evidence during resume."""
    run_dirs = _write_comparison(tmp_path, "tb2.1")
    manifest, comparison = evaluate.freeze_comparison(run_dirs)
    output_dir = tmp_path / "test"
    runner = _fake_runner(manifest, comparison, output_dir)
    evaluate.evaluate_comparison(manifest, comparison, output_dir, runner)
    path = output_dir / "initial-repetition-1.json"
    record = json.loads(path.read_text())
    if damage == "missing_task":
        record["scores"].pop(manifest.splits["test"][0])
    elif damage == "nonbinary":
        record["scores"][manifest.splits["test"][0]] = 0.5
    elif damage == "wrong_candidate":
        record["candidate_digest"] = "another-candidate"
    elif damage == "wrong_repetition":
        record["repetition"] = 3
    else:
        other = json.loads((output_dir / "initial-repetition-2.json").read_text())
        record["evaluation_id"] = other["evaluation_id"]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        evaluate.evaluate_comparison(manifest, comparison, output_dir, runner)
    assert runner.run.call_count == 39
