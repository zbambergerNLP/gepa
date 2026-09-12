"""Offline tests for the Terminal-Bench experiment CLI contract."""

import argparse
import json
import random
import sys
from pathlib import Path
from unittest.mock import Mock

import litellm
import pytest
from terminalbench_pilot_helpers import offline_runtime as offline_runtime
from terminalbench_pilot_helpers import write_pilot_fixture

sys.path.insert(0, str(Path(__file__).parents[1]))

from examples.common.experiment_models import (
    DEEPSEEK_V4_1_FLASH_MODEL,
    EXPERIMENT_NUM_RETRIES,
    QWEN3_8_27B_MODEL,
    experiment_model_version,
    experiment_request_overrides,
)
from examples.common.provider_retries import install_provider_retries
from examples.terminalbench import evaluate as terminalbench_evaluate
from examples.terminalbench import main as terminalbench_main
from examples.terminalbench.main import (
    EXPERIMENT_MANIFESTS,
    build_parser,
    build_run_contract,
    ensure_run_contract,
    seed_candidate,
)
from examples.terminalbench.model_settings import terminalbench_decoding, terminalbench_limits, terminalbench_model_info
from gepa import optimize
from gepa.adapters.terminal_bench_adapter import (
    TERMINUS_ADAPTER_CONTRACT,
    TerminusAdapter,
    load_terminalbench_manifest,
)
from gepa.adapters.terminal_bench_adapter.documents import COMPONENT_KINDS
from gepa.adapters.terminal_bench_adapter.terminal_bench_adapter import derive_terminalbench_splits
from gepa.adapters.terminal_bench_adapter.text_scope import OPTIMIZATION_SCOPES, TerminalBenchTextScope
from gepa.core.adapter import EvaluationBatch
from gepa.strategies.batch_sampler import IndependentEpochShuffledBatchSampler
from gepa.strategies.document_template import TEMPLATE_FAMILIES
from gepa.strategies.intervention import CONTROLLER_POLICY_CONTRACT, SEMANTIC_ACTION_CATALOGS
from gepa.strategies.text_limits import TextLimits

MANIFEST_PATH = Path(__file__).parents[1] / "examples" / "terminalbench" / "terminalbench-v2.1-manifest.json"


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
            "--experiment",
            "tb2.1",
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
    candidate, family = seed_candidate(QWEN3_8_27B_MODEL, "auto", "tb2.1")
    prompt = candidate["instruction_prompt"]
    bodies = TEMPLATE_FAMILIES[family]["user_prompt"].parse(prompt)

    assert family == "alibaba"
    assert "assigned command-line task" in bodies["Objective"]
    assert bodies["Context"] == ""
    assert [line for line in prompt.splitlines() if line.startswith("## ")] == ["## Objective"]


def test_deepseek_student_uses_generic_user_prompt_template() -> None:
    """Render the DeepSeek seed as a sparse generic user prompt."""
    candidate, family = seed_candidate(DEEPSEEK_V4_1_FLASH_MODEL, "auto", "tb2.1")
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


def test_parser_defaults_both_roles_to_qwen3_8_27b(tmp_path: Path) -> None:
    """Use the homogeneous Qwen condition when model flags are omitted.

    Args:
        tmp_path: Pytest directory used for required CLI paths.
    """
    args = build_parser().parse_args(
        [
            "--experiment",
            "tb2.1",
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
    assert args.optimization_scope == "system_prompt"


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


def test_campaign_rejects_a_valid_but_different_manifest_split_before_optimization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Do not permit resplitting the same pinned tasks while retaining the 30/19/40 counts."""
    payload = json.loads(MANIFEST_PATH.read_text())
    payload["split_policy"]["seed"] = "different-ablation-seed"
    payload["splits"] = derive_terminalbench_splits(
        list(payload["task_refs"]), payload["split_policy"]["seed"], payload["split_policy"]["counts"]
    )
    path = tmp_path / "different-manifest.json"
    path.write_text(json.dumps(payload))
    assert load_terminalbench_manifest(path).splits != load_terminalbench_manifest(MANIFEST_PATH).splits
    optimizer, harbor = Mock(), Mock()
    monkeypatch.setattr(terminalbench_main, "optimize", optimizer)
    monkeypatch.setattr(terminalbench_main, "HarborCLI", harbor)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "terminalbench",
            "--condition",
            "vanilla",
            "--manifest",
            str(path),
            "--run-dir",
            str(tmp_path / "run"),
            "--harbor-work-dir",
            str(tmp_path / "harbor"),
        ],
    )
    with pytest.raises(SystemExit):
        terminalbench_main.main()
    optimizer.assert_not_called()
    harbor.assert_not_called()
    assert not (tmp_path / "run").exists()


def test_generated_run_contract_records_metric_call_budget(tmp_path: Path) -> None:
    """Record metric budget and semantic Controller policy in generated state.

    Args:
        tmp_path: Pytest directory used for parsed output paths.
    """
    args = build_parser().parse_args(
        [
            "--experiment",
            "tb2.1",
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
    assert contract["schema_version"] == 32
    assert contract["evaluation_protocol"]["test_timing"] == "after_each_completed_ablation"
    assert contract["skip_perfect_score"] is True
    assert contract["perfect_score"] == 1.0
    assert contract["adapter"] == TERMINUS_ADAPTER_CONTRACT
    assert contract["task_context_settings"] == {
        "enable_summarize": True,
        "proactive_summarization_threshold": 8_000,
    }
    assert contract["max_proposer_model_calls"] is None
    assert contract["react_execution"]["completion"] == "explicit_finish"
    assert contract["react_execution"]["max_iterations"] is None
    assert contract["react_execution"]["max_tool_calls"] is None
    assert contract["manifestor_traces_chars"] is None
    assert contract["manifestor_temperature"] == 1.0
    assert contract["reflection_context"]["version"] == 1
    assert contract["failure_policy"]["accepted_trial_exceptions"] == ["AgentTimeoutError"]
    assert contract["failure_policy"]["harbor_max_retries"] == 0
    assert contract["reflection_feedback"]["reflection_split"] == "train"
    assert contract["reflection_feedback"]["max_chars_per_verifier_log"] is None
    assert contract["optimization_scope"] == "system_prompt"
    assert contract["component_kinds"] == {"instruction_prompt": "user_prompt"}
    assert contract["runtime_component_kinds"] == COMPONENT_KINDS
    assert contract["student_model"] == QWEN3_8_27B_MODEL
    assert contract["proposer_model"] == QWEN3_8_27B_MODEL
    assert contract["student_decoding"] == terminalbench_decoding(QWEN3_8_27B_MODEL)
    assert contract["student_model_info"] == terminalbench_model_info(QWEN3_8_27B_MODEL)
    assert contract["proposer_decoding"] == terminalbench_decoding(QWEN3_8_27B_MODEL, agentic=False)
    assert contract["student_num_retries"] == EXPERIMENT_NUM_RETRIES
    assert contract["proposer_num_retries"] == EXPERIMENT_NUM_RETRIES
    assert contract["semantic_action_space"] == SEMANTIC_ACTION_CATALOGS
    assert contract["semantic_controller_policy"] == CONTROLLER_POLICY_CONTRACT


def test_deepseek_run_contract_uses_the_separate_same_model_condition(tmp_path: Path) -> None:
    """Record DeepSeek V4 Flash in both roles with its fixed decoding.

    Args:
        tmp_path: Pytest directory used for parsed output paths.
    """
    args = _model_args(tmp_path, DEEPSEEK_V4_1_FLASH_MODEL, DEEPSEEK_V4_1_FLASH_MODEL)
    manifest = load_terminalbench_manifest(MANIFEST_PATH)

    contract = build_run_contract(
        args,
        manifest,
        manifest.tasks("train", 1),
        manifest.tasks("val", 1),
        "react_v2",
        "generic",
    )

    assert contract["student_model"] == DEEPSEEK_V4_1_FLASH_MODEL
    assert contract["proposer_model"] == DEEPSEEK_V4_1_FLASH_MODEL
    assert contract["student_decoding"] == terminalbench_decoding(DEEPSEEK_V4_1_FLASH_MODEL)
    assert contract["student_model_info"] == terminalbench_model_info(DEEPSEEK_V4_1_FLASH_MODEL)
    assert contract["proposer_decoding"] == terminalbench_decoding(DEEPSEEK_V4_1_FLASH_MODEL, agentic=False)


@pytest.mark.parametrize("experiment", EXPERIMENT_MANIFESTS)
@pytest.mark.parametrize("condition,budget", list(terminalbench_main.CAMPAIGN_CELLS.values()))
@pytest.mark.parametrize("model", [QWEN3_8_27B_MODEL, DEEPSEEK_V4_1_FLASH_MODEL])
@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("optimization_scope", OPTIMIZATION_SCOPES)
def test_provider_settings_reach_all_runtime_roles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    experiment: str,
    condition: str,
    budget: str,
    model: str,
    configured: bool,
    optimization_scope: str,
) -> None:
    """Forward each provider's sampling and thinking settings through every campaign cell."""
    limits = (
        TextLimits(
            max_component_chars=25000,
            max_candidate_chars=250000,
            selector_target_chars=20000,
            max_prompt_chars=200000,
            controller_feedback_chars=17000,
            stateless_feedback_chars=15000,
            manifestor_trace_chars=80000,
            manifestor_steering_chars=3000,
            history_text_chars=7000,
            verifier_log_chars=5000,
        )
        if configured
        else TextLimits()
    )
    requirements = Mock()
    monkeypatch.setattr(terminalbench_main.HarborCLI, "check_requirements", requirements)
    harbor_factory = Mock(wraps=terminalbench_main.HarborCLI)
    optimizer = Mock()
    monkeypatch.setattr(terminalbench_main, "HarborCLI", harbor_factory)
    monkeypatch.setattr(terminalbench_main, "optimize", optimizer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "terminalbench",
            "--optimization-scope",
            optimization_scope,
            *(["--text-limits", json.dumps(limits.to_dict())] if configured else []),
            "--experiment",
            experiment,
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
            "--max-metric-calls",
            "4",
            "--train-limit",
            "1",
            "--val-limit",
            "1",
            "--run-dir",
            str(tmp_path / "run"),
            "--harbor-work-dir",
            str(tmp_path / "harbor"),
        ],
    )

    terminalbench_main.main()

    requirements.assert_called_once_with()
    harbor_kwargs = harbor_factory.call_args.kwargs
    optimize_kwargs = optimizer.call_args.kwargs
    assert type(optimize_kwargs["adapter"]) is TerminusAdapter
    assert optimize_kwargs["skip_perfect_score"] is True
    assert optimize_kwargs["perfect_score"] == 1.0
    assert harbor_kwargs["text_limits"] == optimize_kwargs["text_limits"] == limits
    saved = json.loads((tmp_path / "run" / terminalbench_main.RUN_CONTRACT_FILENAME).read_text())
    assert saved["adapter"] == TERMINUS_ADAPTER_CONTRACT
    assert saved["text_limits"] == limits.to_dict()
    assert saved["reflection_feedback"]["max_chars_per_verifier_log"] == limits.verifier_log_chars
    student_kwargs = harbor_kwargs["student_agent_kwargs"]
    expected_body = experiment_request_overrides(model, explicit_reasoning=True)["extra_body"]
    general = terminalbench_decoding(model, agentic=False)
    agentic = terminalbench_decoding(model, agentic=True)
    temperature = general["temperature"]
    assert temperature == 1.0
    assert harbor_kwargs["student_model"] == optimize_kwargs["reflection_lm"].model == model
    assert student_kwargs["model_info"] == terminalbench_model_info(model)
    assert student_kwargs["token_limits"] == terminalbench_limits(model)
    assert student_kwargs["model_info"]["max_output_tokens"] == 32_768
    assert student_kwargs["llm_kwargs"]["max_tokens"] == 32_768
    assert optimize_kwargs["reflection_lm"].completion_kwargs["max_tokens"] == 32_768
    assert student_kwargs["llm_kwargs"].get("extra_body") == expected_body
    assert optimize_kwargs["reflection_lm_kwargs"].get("extra_body") == expected_body
    assert student_kwargs["llm_kwargs"]["temperature"] == temperature
    assert optimize_kwargs["reflection_lm_kwargs"]["temperature"] == temperature
    assert student_kwargs["llm_kwargs"]["top_p"] == agentic["top_p"]
    assert optimize_kwargs["reflection_lm_kwargs"]["top_p"] == general["top_p"]
    assert harbor_kwargs["student_api_base"] == optimize_kwargs["reflection_lm_kwargs"]["api_base"]
    assert len(optimize_kwargs["trainset"]) == len(optimize_kwargs["valset"]) == 1
    manifest = load_terminalbench_manifest(EXPERIMENT_MANIFESTS[experiment])
    scope = TerminalBenchTextScope(optimization_scope, saved["template_family"])
    assert optimize_kwargs["seed_candidate"] == scope.seed_candidate()
    assert optimize_kwargs["component_kinds"] == scope.component_kinds
    assert optimize_kwargs["adapter"].text_scope == scope
    assert saved["text_scope"] == scope.contract()
    assert saved["runtime_component_kinds"] == manifest.component_kinds
    assert optimize_kwargs["reflection_level"] == (2 if condition in terminalbench_main.FOREST_CONDITIONS else 0)
    strategy = optimize_kwargs["reflection_strategy"]
    clients = [optimize_kwargs["reflection_lm"]]
    if condition in terminalbench_main.FOREST_CONDITIONS:
        assert strategy.controller_selection == ("uniform_random" if condition == "react_v2_random" else "verbalized")
        assert strategy.max_chars == limits.max_component_chars
        assert strategy.text_limits == limits
        clients.extend([strategy.base_lm, strategy.manifestor_lm])
        if condition == "react_v2":
            clients.append(strategy.controller_lm)
        assert strategy.base_lm.model == strategy.manifestor_lm.model == model
        assert strategy.base_lm.completion_kwargs.get("extra_body") == expected_body
        assert strategy.manifestor_lm.completion_kwargs.get("extra_body") == expected_body
        assert strategy.base_lm.completion_kwargs["api_base"] == "http://localhost:8000/v1"
        assert strategy.base_lm.completion_kwargs["temperature"] == temperature
        assert strategy.manifestor_lm.completion_kwargs["temperature"] == temperature
        assert strategy.base_lm.completion_kwargs["top_p"] == agentic["top_p"]
        assert strategy.manifestor_lm.completion_kwargs["top_p"] == general["top_p"]
        if condition == "react_v2":
            assert strategy.controller_lm.model == model
            assert strategy.controller_lm.completion_kwargs.get("extra_body") == expected_body
            assert strategy.controller_lm.completion_kwargs["temperature"] == temperature
            assert strategy.controller_lm.completion_kwargs["top_p"] == general["top_p"]
    elif condition == "action":
        assert isinstance(strategy, terminalbench_main.ComponentActionReflectionLM)
        for reflector in strategy.reflectors.values():
            assert reflector.text_limits == reflector.action_selector.text_limits == limits
            clients.extend([reflector.lm, reflector.action_selector.lm])
            assert reflector.lm.model == reflector.action_selector.lm.model == model
            assert (
                reflector.lm.completion_kwargs.get("extra_body")
                == reflector.action_selector.lm.completion_kwargs.get("extra_body")
                == expected_body
            )
            assert reflector.lm.completion_kwargs["temperature"] == temperature
            assert reflector.action_selector.lm.completion_kwargs["temperature"] == temperature
            assert reflector.lm.completion_kwargs["top_p"] == general["top_p"]
            assert reflector.action_selector.lm.completion_kwargs["top_p"] == general["top_p"]
            assert (
                reflector.lm.completion_kwargs["api_base"] == reflector.action_selector.lm.completion_kwargs["api_base"]
            )
    else:
        assert strategy is None
    unique_clients = {id(client): client for client in clients}
    for client in unique_clients.values():
        assert client.completion_kwargs["max_tokens"] == 32_768
        raw = litellm.ModelResponse(
            model=model, choices=[{"message": {"role": "assistant", "content": "done"}, "finish_reason": "length"}],
            usage={"prompt_tokens": 10, "completion_tokens": 32_768, "total_tokens": 32_778},
        )
        monkeypatch.setattr(litellm, "completion", Mock(return_value=raw))
        monkeypatch.setattr(litellm, "completion_cost", Mock(return_value=0.0))
        install_provider_retries()
        assert client("offline input") == "done"
    records = [json.loads(line) for line in (tmp_path / "run" / "token-usage.jsonl").read_text().splitlines()]
    assert len(records) == len(unique_clients)
    assert all(record["length_finish"] and record["output_cap_reached"] for record in records)
    assert optimize_kwargs["stop_callbacks"].max_proposals == (8 if budget == "double" else 4)
    assert optimize_kwargs["max_metric_calls"] == 4
    sampler = optimize_kwargs["batch_sampler"]
    assert isinstance(sampler, IndependentEpochShuffledBatchSampler)
    assert optimize_kwargs["reflection_minibatch_size"] is None
    assert optimize_kwargs["module_selector"] == "all"
    assert optimize_kwargs["use_merge"] is False
    contract = json.loads((tmp_path / "run" / "terminalbench-run-contract.json").read_text())
    assert contract["training_batch_order"] == sampler.contract()
    assert sampler.seed == contract["seed"]
    assert sampler.minibatch_size == contract["reflection_minibatch_size"] == 3
    assert contract["experiment"] == experiment
    assert contract["module_selector"] == optimize_kwargs["module_selector"]
    assert contract["cache_evaluation"] is optimize_kwargs["cache_evaluation"] is False
    assert contract["candidate_selection_strategy"] == optimize_kwargs["candidate_selection_strategy"] == "pareto"
    assert contract["frontier_type"] == optimize_kwargs["frontier_type"] == "instance"
    assert contract["acceptance_criterion"] == optimize_kwargs["acceptance_criterion"] == "strict_improvement"
    assert contract["validation_evaluation"] == optimize_kwargs["val_evaluation_policy"] == "full_eval"
    assert contract["student_model_version"] == contract["proposer_model_version"] == experiment_model_version(model)
    assert (
        contract["student_request_overrides"]
        == contract["proposer_request_overrides"]
        == experiment_request_overrides(model, explicit_reasoning=True)
    )
    assert contract["manifestor_temperature"] == temperature
    assert contract["student_decoding"] == agentic
    assert contract["proposer_decoding"] == general
    if condition in terminalbench_main.FOREST_CONDITIONS:
        assert contract["reflection_role_decoding"] == {
            "controller": ({"requested": general, "provider_ignored_fields": []} if condition == "react_v2" else None),
            "manifestor": {"requested": general, "provider_ignored_fields": []},
            "react_v2_proposer": {"requested": agentic, "provider_ignored_fields": []},
        }
    else:
        assert contract["reflection_role_decoding"] is None


@pytest.mark.parametrize("experiment,iterations,padding", [("tb2.1", 40, 0)])
@pytest.mark.parametrize("outcome", ["accepted", "tied", "worse", "perfect"])
@pytest.mark.parametrize("budget_name,epochs", [("standard", 4), ("double", 8)])
@pytest.mark.parametrize("optimization_scope", OPTIMIZATION_SCOPES)
def test_epoch_cli_budget_stops_and_resumes_with_real_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    experiment: str,
    iterations: int,
    padding: int,
    outcome: str,
    budget_name: str,
    epochs: int,
    optimization_scope: str,
) -> None:
    """Complete either budget across resume regardless of proposal success."""
    iterations *= epochs // 4
    run_dir = tmp_path / "run"
    stop_file = run_dir / "gepa.stop"
    parent_batches = []
    evaluations = []
    selected_components = []
    trainset = load_terminalbench_manifest(EXPERIMENT_MANIFESTS[experiment]).tasks("train")

    class SamplingRecorder:
        """Record each sampling step once and pause partway through an epoch."""

        def on_minibatch_sampled(self, event):
            """Pause after the fifth minibatch while letting its evaluations finish."""
            parent_batches.append([trainset[index].task_id for index in event["minibatch_ids"]])
            if len(parent_batches) == 5:
                stop_file.touch()

    class BudgetAdapter:
        """Exercise real sampling, stopping, validation, and checkpoints offline."""

        def evaluate(self, batch, candidate, capture_traces=False):
            """Return controlled rewards for training and validation tasks."""
            evaluations.append([task.task_id for task in batch])
            score = sum(text.count("budget_step") for text in candidate.values()) / (100 * len(candidate))
            if outcome == "worse":
                score = 0.0 if score else 0.5
            elif outcome != "accepted":
                score = 1.0 if outcome == "perfect" else 0.0
            return EvaluationBatch(
                outputs=[{} for _ in batch],
                scores=[score for _ in batch],
                trajectories=[{} for _ in batch] if capture_traces else None,
                num_metric_calls=len(batch),
            )

        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            """Supply deterministic feedback without invoking a model."""
            return {key: [{"feedback": "Try the next revision"}] for key in components_to_update}

        def propose_new_texts(self, candidate, reflective_dataset, components_to_update):
            """Append a revision marker inside each selected document."""
            selected_components.append(set(components_to_update))
            return {key: candidate[key] + "\nbudget_step" for key in components_to_update}

    results = []

    def optimize_offline(**kwargs):
        """Keep CLI budget wiring while replacing model work with the adapter."""
        kwargs["reflection_lm"] = None
        kwargs["callbacks"] = [SamplingRecorder()]
        assert kwargs["raise_on_exception"] is True
        results.append(optimize(**kwargs, display_progress_bar=False))

    monkeypatch.setattr(terminalbench_main.HarborCLI, "check_requirements", Mock())
    monkeypatch.setattr(terminalbench_main, "TerminusAdapter", lambda *args, **kwargs: BudgetAdapter())
    monkeypatch.setattr(terminalbench_main, "optimize", optimize_offline)
    heldout = Mock(return_value={"harnesses": {}})
    monkeypatch.setattr(terminalbench_evaluate, "evaluate_comparison", heldout)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "terminalbench",
            "--optimization-scope",
            optimization_scope,
            "--experiment",
            experiment,
            "--condition",
            "vanilla",
            "--budget",
            budget_name,
            "--run-dir",
            str(run_dir),
            "--harbor-work-dir",
            str(tmp_path / "harbor"),
        ],
    )

    args = build_parser().parse_args()
    manifest = load_terminalbench_manifest(EXPERIMENT_MANIFESTS[experiment])
    _, family = seed_candidate(args.student_model, "auto", experiment, optimization_scope)
    initial_contract = build_run_contract(args, manifest, trainset, manifest.tasks("val"), "vanilla", family)
    pilot_dir = write_pilot_fixture(tmp_path / "pilot", initial_contract, manifest)
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--reviewed-pilot", str(pilot_dir)])

    with pytest.raises(ValueError, match="optimization has not completed"):
        terminalbench_main.main()
    assert len(parent_batches) == 5
    heldout.assert_not_called()
    monkeypatch.setattr(sys, "argv", sys.argv[:-2])
    stop_file.unlink()
    terminalbench_main.main()
    assert len(parent_batches) == iterations
    heldout.assert_called_once()
    frozen_manifest, comparison, test_dir, _ = heldout.call_args.args
    assert frozen_manifest.splits == manifest.splits
    cell = f"{optimization_scope}__vanilla{'_2x' if budget_name == 'double' else ''}"
    assert comparison["source_runs"][cell]["selected_candidate_index"] == results[-1].best_idx
    assert test_dir == run_dir / "heldout"
    terminalbench_main.main()
    assert len(parent_batches) == iterations
    assert heldout.call_count == 2

    contract = json.loads((run_dir / "terminalbench-run-contract.json").read_text())
    budget = contract["optimization_budget"]
    assert contract["max_metric_calls"] is None
    assert budget["training_epochs"] == epochs
    assert budget["max_iterations"] == iterations
    assert budget["padding_tasks_per_epoch"] == padding
    assert budget["sampled_training_tasks"] == sum(map(len, parent_batches)) == iterations * 3
    for start in range(0, iterations, iterations // epochs):
        epoch_ids = [task_id for batch in parent_batches[start : start + iterations // epochs] for task_id in batch]
        assert set(epoch_ids) == set(contract["train_task_ids"])
        assert len(epoch_ids) == len(contract["train_task_ids"]) + padding
    rng = random.Random(contract["seed"])
    expected_batches = []
    for _ in range(epochs):
        task_ids = [task.task_id for task in trainset]
        rng.shuffle(task_ids)
        expected_batches.extend(task_ids[start : start + 3] for start in range(0, len(task_ids), 3))
    assert parent_batches == expected_batches
    assert not set(contract["test_task_ids"]).intersection(task_id for batch in evaluations for task_id in batch)
    per_iteration = 3 if outcome == "perfect" else 6
    if outcome == "accepted":
        per_iteration += len(contract["val_task_ids"])
    assert results[-1].total_metric_calls == len(contract["val_task_ids"]) + iterations * per_iteration
    assert selected_components == ([] if outcome == "perfect" else [set(contract["component_kinds"])] * iterations)
    validation_batches = [batch for batch in evaluations if set(batch) == set(contract["val_task_ids"])]
    assert len(validation_batches) == (iterations + 1 if outcome == "accepted" else 1)
    training_batches = [batch for batch in evaluations if set(batch).issubset(contract["train_task_ids"])]
    if outcome != "perfect":
        assert training_batches[::2] == training_batches[1::2]
    if outcome == "accepted":
        assert len(results[-1].candidates) == iterations + 1
        assert all(text.count("budget_step") == iterations for text in results[-1].candidates[-1].values())
    else:
        assert len(results[-1].candidates) == 1


@pytest.mark.parametrize("field,value", [("reflection_minibatch_size", 0), ("max_metric_calls", -1)])
def test_run_contract_rejects_invalid_budget_inputs(tmp_path: Path, field: str, value: int) -> None:
    """Reject invalid budget arithmetic before starting any Harbor work."""
    args = _model_args(tmp_path, QWEN3_8_27B_MODEL, QWEN3_8_27B_MODEL)
    setattr(args, field, value)
    manifest = load_terminalbench_manifest(MANIFEST_PATH)
    with pytest.raises(ValueError, match="must be positive"):
        build_run_contract(args, manifest, manifest.tasks("train"), manifest.tasks("val"), "vanilla", "alibaba")


def test_six_cell_matrix_pins_methods_budgets_and_resume_identity(tmp_path: Path) -> None:
    """Expose four standard methods and two double-budget methods with distinct contracts."""
    assert terminalbench_main.CAMPAIGN_CELLS == {
        "vanilla": ("vanilla", "standard"),
        "react_v2": ("react_v2", "standard"),
        "react_v2_random": ("react_v2_random", "standard"),
        "action": ("action", "standard"),
        "vanilla_2x": ("vanilla", "double"),
        "react_v2_2x": ("react_v2", "double"),
    }
    args = _model_args(tmp_path, QWEN3_8_27B_MODEL, QWEN3_8_27B_MODEL)
    manifest = load_terminalbench_manifest(MANIFEST_PATH)
    contracts = []
    for condition, budget in terminalbench_main.CAMPAIGN_CELLS.values():
        args.budget = budget
        contract = build_run_contract(
            args, manifest, manifest.tasks("train"), manifest.tasks("val"), condition, "alibaba"
        )
        contracts.append(contract)
        assert contract["condition"] == condition
        assert contract["optimization_budget"]["max_iterations"] == (80 if budget == "double" else 40)
        assert contract["module_selector"] == "all"
        assert contract["cache_evaluation"] is False
        assert contract["candidate_selection_strategy"] == "pareto"
        assert contract["frontier_type"] == "instance"
        assert contract["acceptance_criterion"] == "strict_improvement"
        assert contract["validation_evaluation"] == "full_eval"
        assert contract["max_proposer_model_calls"] is None
        assert contract["document_length"] == {
            "version": 2,
            "max_component_chars": None,
            "max_candidate_chars": None,
            "selector_target_chars": None,
        }
        assert contract["text_limits"] == TextLimits().to_dict()
        if condition in terminalbench_main.FOREST_CONDITIONS:
            assert contract["react_execution"]["completion"] == "explicit_finish"
            assert contract["react_execution"]["max_iterations"] is None
            assert contract["react_execution"]["max_tool_calls"] is None
        else:
            assert contract["react_execution"] is None
    ensure_run_contract(tmp_path / "resume", contracts[1])
    for contract in contracts[:1] + contracts[2:]:
        with pytest.raises(ValueError, match="different Terminal-Bench configuration"):
            ensure_run_contract(tmp_path / "resume", contract)


@pytest.mark.parametrize("condition", ["react_v2_random", "action"])
def test_double_budget_rejects_extra_ablation_cells_before_harbor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, condition: str
) -> None:
    """Keep the larger budget exclusive to the two headline methods."""
    harbor_factory = Mock()
    monkeypatch.setattr(terminalbench_main, "HarborCLI", harbor_factory)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "terminalbench",
            "--experiment",
            "tb2.1",
            "--condition",
            condition,
            "--budget",
            "double",
            "--run-dir",
            str(tmp_path / "run"),
            "--harbor-work-dir",
            str(tmp_path / "harbor"),
        ],
    )
    with pytest.raises(SystemExit):
        terminalbench_main.main()
    harbor_factory.assert_not_called()
    assert not (tmp_path / "run").exists()


def test_run_contract_rejects_a_cross_model_pair(tmp_path: Path) -> None:
    """Reject a Qwen student paired with the DeepSeek proposer.

    Args:
        tmp_path: Pytest directory used for parsed output paths.
    """
    args = _model_args(tmp_path, QWEN3_8_27B_MODEL, DEEPSEEK_V4_1_FLASH_MODEL)
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

    with pytest.raises(ValueError, match=r"no terminalbench-run-contract\.json"):
        ensure_run_contract(tmp_path, {"condition": "react_v2"})


def test_experiment_defaults_to_tb21(tmp_path: Path) -> None:
    """Select the only supported benchmark without requiring an experiment flag."""
    args = build_parser().parse_args(
        [
            "--condition",
            "vanilla",
            "--run-dir",
            str(tmp_path),
            "--harbor-work-dir",
            str(tmp_path / "harbor"),
        ]
    )
    assert args.experiment == "tb2.1"
    assert set(EXPERIMENT_MANIFESTS) == {"tb2.1"}


@pytest.mark.parametrize("experiment", ["tb2", "tb2.0", "tb4"])
def test_removed_experiments_are_rejected(tmp_path: Path, experiment: str) -> None:
    """Reject old experiment flags instead of relabeling their dataset contents."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--experiment",
                experiment,
                "--condition",
                "vanilla",
                "--run-dir",
                str(tmp_path),
                "--harbor-work-dir",
                str(tmp_path / "harbor"),
            ]
        )


@pytest.mark.parametrize("family", TEMPLATE_FAMILIES)
def test_tb21_starts_from_the_approved_full_text_and_skills(family: str) -> None:
    """Preserve all 16 editable components when migrating the task dataset."""
    candidate, resolved_family = seed_candidate(QWEN3_8_27B_MODEL, family, "tb2.1", "all_text")
    assert resolved_family == family
    assert set(candidate) == set(COMPONENT_KINDS)
    assert len(candidate) == 16
    assert {"command_format", "skill_debugging", "skill_verification"}.issubset(candidate)
    assert "{instruction}" not in "".join(candidate.values())
    assert "{terminal_state}" not in "".join(candidate.values())


@pytest.mark.parametrize("experiment", ["tb2", "tb4"])
def test_old_experiment_contracts_cannot_resume_as_tb21(tmp_path: Path, experiment: str) -> None:
    """Require a fresh run directory when migrating from a removed benchmark."""
    args = _model_args(tmp_path, QWEN3_8_27B_MODEL, QWEN3_8_27B_MODEL)
    manifest = load_terminalbench_manifest(EXPERIMENT_MANIFESTS["tb2.1"])
    contract = build_run_contract(args, manifest, manifest.tasks("train"), manifest.tasks("val"), "vanilla", "generic")
    ensure_run_contract(tmp_path / "resume", {**contract, "experiment": experiment})
    with pytest.raises(ValueError, match="different Terminal-Bench configuration"):
        ensure_run_contract(tmp_path / "resume", contract)
    args.experiment = experiment
    with pytest.raises(ValueError, match="must match"):
        build_run_contract(args, manifest, manifest.tasks("train"), manifest.tasks("val"), "vanilla", "generic")
