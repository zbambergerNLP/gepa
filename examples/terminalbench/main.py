"""Configure system-prompt and full-text experiments on Terminal-Bench 2.1.

Each completed ablation freezes its validation winner and then evaluates test.

* ``vanilla`` uses stock free-form GEPA reflection.
* ``react_v2`` uses the Controller -> Manifestor -> ReAct V2 workflow.
* ``react_v2_random`` replaces only the Controller with uniform selection.
* ``action`` uses semantic action selection and a stateless section rewrite.

Within each model arm, both scopes use the same official Harbor rewards,
manifest, student/proposer model, task splits, and initial runtime text. Methods
within a scope share editable components. All four methods run for four epochs;
vanilla and full FOREST also run for eight epochs.
Terminal-Bench 2.1 is the sole supported benchmark for this campaign.
"""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

from examples.common.experiment_models import (
    EXPERIMENT_NUM_RETRIES,
    QWEN3_8_27B_MODEL,
    experiment_model_version,
    experiment_request_overrides,
    validate_experiment_model_pair,
)
from examples.common.pilot_checks import OPTIMIZER_PILOT_PROTOCOL, CycleEvidence
from examples.common.provider_retries import PROVIDER_RETRY_POLICY, provider_retry_kwargs
from examples.common.react_v2 import build_react_v2_strategy, resolve_template_family
from examples.common.recovery import RecoveryCallback, run_guarded
from examples.terminalbench.model_settings import (
    terminalbench_decoding,
    terminalbench_limits,
    terminalbench_model_info,
)
from examples.terminalbench.pilot import (
    PILOT_PROTOCOL,
    load_completed_pilot,
    review_pilot,
    run_runtime,
    validate_review,
    validate_runtime,
)
from examples.terminalbench.reflection import ComponentActionReflectionLM
from examples.terminalbench.runtime import load_role_runtimes
from examples.terminalbench.token_usage import TOKEN_USAGE_POLICY, observe_optimizer
from gepa import optimize
from gepa.adapters.terminal_bench_adapter import (
    TERMINUS_ADAPTER_CONTRACT,
    HarborCLI,
    TerminalBenchManifest,
    TerminalBenchTask,
    TerminusAdapter,
    load_terminalbench_manifest,
)
from gepa.adapters.terminal_bench_adapter.documents import (
    BUNDLE_VERSION,
    seed_documents,
)
from gepa.adapters.terminal_bench_adapter.terminal_bench_adapter import (
    FAILURE_POLICY_CONTRACT,
    REFLECTION_FEEDBACK_CONTRACT,
    TASK_CONTEXT_SETTINGS,
)
from gepa.adapters.terminal_bench_adapter.text_scope import (
    DEFAULT_OPTIMIZATION_SCOPE,
    OPTIMIZATION_SCOPES,
    TerminalBenchTextScope,
)
from gepa.lm import LM
from gepa.proposer.reflective_mutation.react_v2_proposer import REACT_V2_EXECUTION_CONTRACT
from gepa.strategies.action_space import stateless_selector_policy_contract
from gepa.strategies.batch_sampler import IndependentEpochShuffledBatchSampler
from gepa.strategies.intervention import (
    CONTROLLER_POLICY_CONTRACT,
    SEMANTIC_ACTION_CATALOGS,
    UNIFORM_RANDOM_CONTROLLER_POLICY_CONTRACT,
)
from gepa.strategies.proposal_sampling import SingleMutationSampling
from gepa.strategies.reflection_context import REFLECTION_CONTEXT_CONTRACT
from gepa.strategies.text_limits import parse_text_limits, resolve_text_limits
from gepa.utils.stop_condition import MaxCandidateProposalsStopper

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_MANIFESTS = {
    "tb2.1": Path(__file__).with_name("terminalbench-v2.1-manifest.json"),
}
RUN_CONTRACT_FILENAME = "terminalbench-run-contract.json"
CONDITIONS_BY_BUDGET = {
    "standard": ("vanilla", "react_v2", "react_v2_random", "action"),
    "double": ("vanilla", "react_v2"),
}
TRAINING_EPOCHS_BY_BUDGET = {"standard": 4, "double": 8}
CAMPAIGN_CELLS = {
    f"{condition}{'_2x' if budget == 'double' else ''}": (condition, budget)
    for budget, conditions in CONDITIONS_BY_BUDGET.items()
    for condition in conditions
}
SCOPE_CAMPAIGN_CELLS = {
    f"{scope}__{cell}": (scope, condition, budget)
    for scope in OPTIMIZATION_SCOPES
    for cell, (condition, budget) in CAMPAIGN_CELLS.items()
}
FOREST_CONDITIONS = {"react_v2", "react_v2_random"}
TEST_REPETITIONS = 3
EVALUATION_PROTOCOL = {
    "optimization_runs_per_configuration": 1,
    "test_repetitions": TEST_REPETITIONS,
    "attempts_per_task_per_repetition": 1,
    "selection_metric": "mean_validation_reward",
    "test_metric": "pass_at_1",
    "standard_deviation_ddof": 1,
    "test_timing": "after_each_completed_ablation",
}
TemplateFamily = Literal["generic", "openai", "anthropic", "google", "alibaba"]


def seed_candidate(
    student_model: str, template_family: str, experiment: str, optimization_scope: str = DEFAULT_OPTIMIZATION_SCOPE
) -> tuple[dict[str, str], TemplateFamily]:
    """Build the experiment's seed with the selected provider template.

    Args:
        student_model: Task model used for automatic provider inference.
        template_family: Explicit provider family or ``"auto"``.
        experiment: Benchmark receiving the shared full agent text and skills.
        optimization_scope: Full text or the unified initial instruction block.

    Returns:
        Editable components and their resolved template family.
    """
    resolved_family = cast(TemplateFamily, resolve_template_family(template_family, student_model))
    if experiment not in EXPERIMENT_MANIFESTS:
        raise ValueError(f"Unknown Terminal-Bench experiment: {experiment!r}")
    return TerminalBenchTextScope(optimization_scope, resolved_family).seed_candidate(), resolved_family


def ensure_run_contract(run_dir: Path, contract: dict[str, Any]) -> Path:
    """Write the run contract or reject an incompatible resumable directory.

    Args:
        run_dir: Experiment directory that owns the resumable state.
        contract: Complete material configuration for the requested run.

    Returns:
        Path to the existing or newly written contract file.

    Raises:
        ValueError: Existing state has a different contract, or legacy GEPA
            state has no contract to validate.
    """
    path = run_dir / RUN_CONTRACT_FILENAME
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != contract:
            raise ValueError(f"Run directory {run_dir} contains a different Terminal-Bench configuration.")
        return path
    if (run_dir / "gepa_state.bin").exists():
        raise ValueError(
            f"Run directory {run_dir} has GEPA state but no {RUN_CONTRACT_FILENAME}; choose a clean directory."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return path


def build_parser() -> argparse.ArgumentParser:
    """Build the experiment CLI without launching any evaluation.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="GEPA on pinned Terminal-Bench 2.1 through Harbor")
    parser.add_argument(
        "--optimizer-pilot", action="store_true", help="One real optimizer cycle using three training tasks only"
    )
    parser.add_argument(
        "--optimizer-pilot-calibration", type=Path, help="Completed full training pilot required before optimizer checks"
    )
    parser.add_argument(
        "--text-limits",
        type=parse_text_limits,
        default=None,
        help="JSON object of optional character limits; omitted or null fields are unlimited",
    )
    parser.add_argument(
        "--experiment",
        choices=tuple(EXPERIMENT_MANIFESTS),
        default="tb2.1",
        help="Pinned Terminal-Bench 2.1 dataset",
    )
    parser.add_argument(
        "--optimization-scope",
        choices=OPTIMIZATION_SCOPES,
        default=DEFAULT_OPTIMIZATION_SCOPE,
        help="Edit the unified initial system prompt (default), or all 16 text artifacts",
    )
    parser.add_argument(
        "--condition",
        choices=CONDITIONS_BY_BUDGET["standard"],
        required=True,
        help="Optimization condition to run",
    )
    parser.add_argument(
        "--budget",
        choices=tuple(CONDITIONS_BY_BUDGET),
        default="standard",
        help="Four epochs for all methods; double gives vanilla and full FOREST eight epochs",
    )
    parser.add_argument(
        "--student-model",
        default=QWEN3_8_27B_MODEL,
        help="Terminus model; use the same supported model as --proposer-model",
    )
    parser.add_argument(
        "--proposer-model",
        default=QWEN3_8_27B_MODEL,
        help="GEPA proposer; use the same supported model as --student-model",
    )
    parser.add_argument("--student-api-base", default=None)
    parser.add_argument("--proposer-api-base", default=None)
    parser.add_argument(
        "--runtime-record", type=Path, help="Current local task-server record from the runtime launcher"
    )
    parser.add_argument(
        "--proposer-runtime-record",
        type=Path,
        help="Current optimizer-server record, if it uses a different server; otherwise reuse --runtime-record",
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=None,
        help="Optional early-stop cap on task evaluations, in addition to the selected epoch budget",
    )
    parser.add_argument("--reflection-minibatch-size", type=int, default=3)
    parser.add_argument("--n-concurrent", type=int, default=1)
    parser.add_argument(
        "--reviewed-pilot",
        type=Path,
        help="Completed full pilot directory; supplying it attests review of usage, cutoffs, timeouts, and throughput",
    )
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--edit-tool-set",
        choices=("minimal", "broad"),
        default="broad",
        help="Edit tools used by ReAct V2",
    )
    parser.add_argument(
        "--reflection-level",
        type=int,
        choices=(1, 2),
        default=2,
        help="Reflection level: region only, or region plus an applied semantic action",
    )
    parser.add_argument(
        "--template-family",
        choices=("auto", "generic", "openai", "anthropic", "google", "alibaba"),
        default="auto",
    )
    parser.add_argument("--manifest", type=Path, default=None, help="Optional manifest path; must match --experiment")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--test-output-dir",
        type=Path,
        help="Shared comparison directory for this model's ablations; defaults to RUN_DIR/heldout",
    )
    parser.add_argument("--harbor-work-dir", type=Path, required=True)
    parser.add_argument("--harbor-executable", default="harbor")
    parser.add_argument("--docker-executable", default="docker")
    parser.add_argument(
        "--harbor-process-timeout-sec",
        type=float,
        default=None,
        help="Optional whole-job timeout; default leaves long-horizon runs to task-level Harbor timeouts",
    )
    return parser


def build_run_contract(
    args: argparse.Namespace,
    manifest: TerminalBenchManifest,
    trainset: list[TerminalBenchTask],
    valset: list[TerminalBenchTask],
    condition: str,
    resolved_family: str,
) -> dict[str, Any]:
    """Record every material axis needed for safe resume and comparison.

    Args:
        args: Parsed Terminal-Bench CLI arguments.
        manifest: Validated pinned benchmark manifest.
        trainset: Selected training tasks in manifest order.
        valset: Selected validation tasks in manifest order.
        condition: Canonical optimization condition.
        resolved_family: Provider template family used by the student prompt.

    Returns:
        JSON-serializable run contract including exact task identities.
    """
    validate_experiment_model_pair(args.student_model, args.proposer_model)
    if manifest.experiment != args.experiment:
        raise ValueError("--manifest must match the selected --experiment")
    pinned_manifest = load_terminalbench_manifest(EXPERIMENT_MANIFESTS[args.experiment])
    if replace(manifest, path=pinned_manifest.path) != pinned_manifest:
        raise ValueError("All ablations must use the pinned benchmark data and identical train/validation/test splits")
    if not trainset or not valset:
        raise ValueError("train and validation selections must both be non-empty")
    if args.reflection_minibatch_size <= 0:
        raise ValueError("--reflection-minibatch-size must be positive")
    if args.max_metric_calls is not None and args.max_metric_calls <= 0:
        raise ValueError("--max-metric-calls must be positive")
    if args.budget not in CONDITIONS_BY_BUDGET or condition not in CONDITIONS_BY_BUDGET[args.budget]:
        raise ValueError("The double budget supports only vanilla GEPA and full FOREST (react_v2)")
    training_epochs = TRAINING_EPOCHS_BY_BUDGET[args.budget]
    iterations_per_epoch = (len(trainset) + args.reflection_minibatch_size - 1) // args.reflection_minibatch_size
    sampled_tasks_per_epoch = iterations_per_epoch * args.reflection_minibatch_size
    scope = TerminalBenchTextScope(getattr(args, "optimization_scope", DEFAULT_OPTIMIZATION_SCOPE), resolved_family)
    candidate = scope.seed_candidate()
    operated = condition in FOREST_CONDITIONS
    reflection_level = args.reflection_level if operated else 0
    controller_selection = (
        "uniform_random"
        if condition == "react_v2_random"
        else "verbalized"
        if operated or condition == "action"
        else None
    )
    controller_policy = (
        UNIFORM_RANDOM_CONTROLLER_POLICY_CONTRACT if condition == "react_v2_random" else CONTROLLER_POLICY_CONTRACT
    )
    text_limits = resolve_text_limits(getattr(args, "text_limits", None))
    proposer_decoding = terminalbench_decoding(args.proposer_model, agentic=False)
    react_decoding = terminalbench_decoding(args.proposer_model, agentic=True)
    reflection_role_decoding = None
    if operated:
        reflection_role_decoding = {
            "controller": (
                {"requested": dict(proposer_decoding), "provider_ignored_fields": []}
                if controller_selection == "verbalized"
                else None
            ),
            "manifestor": (
                {"requested": dict(proposer_decoding), "provider_ignored_fields": []} if reflection_level >= 2 else None
            ),
            "react_v2_proposer": {"requested": react_decoding, "provider_ignored_fields": []},
        }
    return {
        "schema_version": 32,
        "execution_runtime": deepcopy(getattr(args, "execution_runtime", None)),
        "pilot_protocol": deepcopy(PILOT_PROTOCOL),
        "pilot_review": deepcopy(getattr(args, "pilot_review", None)),
        "adapter": deepcopy(TERMINUS_ADAPTER_CONTRACT),
        "provider_retry_policy": deepcopy(PROVIDER_RETRY_POLICY),
        "task_context_settings": dict(TASK_CONTEXT_SETTINGS),
        "token_limits": terminalbench_limits(args.student_model),
        "token_usage_policy": deepcopy(TOKEN_USAGE_POLICY),
        "experiment": manifest.experiment,
        "optimization_target": "agent_text",
        "optimization_scope": scope.name,
        "text_scope": scope.contract(),
        "condition": condition,
        "budget": args.budget,
        "controller_selection": controller_selection,
        "component_kinds": scope.component_kinds,
        "runtime_component_kinds": manifest.component_kinds,
        "module_selector": "all",
        "training_batch_order": IndependentEpochShuffledBatchSampler(
            args.reflection_minibatch_size, args.seed
        ).contract(),
        "cache_evaluation": False,
        "candidate_selection_strategy": "pareto",
        "frontier_type": "instance",
        "acceptance_criterion": "strict_improvement",
        "skip_perfect_score": True,
        "perfect_score": 1.0,
        "validation_evaluation": "full_eval",
        "document_bundle_version": BUNDLE_VERSION,
        "seed_document_digest": manifest.candidate_digest(scope.materialize(candidate)),
        "reference_seed_digest": manifest.candidate_digest(seed_documents(resolved_family)),
        "dataset": manifest.dataset,
        "split_policy": manifest.split_policy,
        "task_refs": manifest.task_refs,
        "test_task_ids": [task.task_id for task in manifest.tasks("test")],
        "edit_tool_set": args.edit_tool_set,
        "harbor_process_timeout_sec": args.harbor_process_timeout_sec,
        "manifest": str(manifest.path),
        "max_metric_calls": args.max_metric_calls,
        "evaluation_protocol": dict(EVALUATION_PROTOCOL),
        "reflection_feedback": {
            **deepcopy(REFLECTION_FEEDBACK_CONTRACT),
            "max_chars_per_verifier_log": text_limits.verifier_log_chars,
        },
        "reflection_context": deepcopy(REFLECTION_CONTEXT_CONTRACT),
        "manifestor_traces_chars": text_limits.manifestor_trace_chars,
        "document_length": text_limits.document_contract(),
        "text_limits": text_limits.to_dict(),
        "manifestor_temperature": float(proposer_decoding["temperature"]),
        "failure_policy": deepcopy(FAILURE_POLICY_CONTRACT),
        "optimization_budget": {
            "unit": "training_epochs",
            "reference": "https://arxiv.org/html/2608.23041v1#A2",
            "training_epochs": training_epochs,
            "iterations_per_epoch": iterations_per_epoch,
            "max_iterations": training_epochs * iterations_per_epoch,
            "sampled_training_tasks": training_epochs * sampled_tasks_per_epoch,
            "padding_tasks_per_epoch": sampled_tasks_per_epoch - len(trainset),
            "batch_sampler": "epoch_shuffled",
            "sampling_strategy": "single_mutation",
            "use_merge": False,
        },
        "n_concurrent": args.n_concurrent,
        "proposer_api_base": args.proposer_api_base,
        "proposer_backend": "react_v2" if operated else "stateless",
        "proposer_decoding": proposer_decoding,
        "reflection_role_decoding": reflection_role_decoding,
        "proposer_model": args.proposer_model,
        "proposer_model_version": experiment_model_version(args.proposer_model),
        "proposer_request_overrides": experiment_request_overrides(args.proposer_model, explicit_reasoning=True),
        "proposer_num_retries": EXPERIMENT_NUM_RETRIES,
        "reflection_level": reflection_level,
        "reflection_minibatch_size": args.reflection_minibatch_size,
        "max_proposer_model_calls": None,
        "react_execution": deepcopy(REACT_V2_EXECUTION_CONTRACT) if operated else None,
        "semantic_action_space": (
            deepcopy(SEMANTIC_ACTION_CATALOGS) if reflection_level == 2 or condition == "action" else None
        ),
        "semantic_controller_policy": deepcopy(controller_policy) if reflection_level == 2 else None,
        "stateless_selector_policy": (
            {
                **stateless_selector_policy_contract("verbalized", text_limits=text_limits),
                "component_schedule": "per_component",
            }
            if condition == "action"
            else None
        ),
        "seed": args.seed,
        "student_api_base": args.student_api_base,
        "student_decoding": terminalbench_decoding(args.student_model, agentic=True),
        "student_model": args.student_model,
        "student_model_version": experiment_model_version(args.student_model),
        "student_request_overrides": experiment_request_overrides(args.student_model, explicit_reasoning=True),
        "student_model_info": terminalbench_model_info(args.student_model),
        "student_num_retries": EXPERIMENT_NUM_RETRIES,
        "template_family": resolved_family,
        "train_task_ids": [task.task_id for task in trainset],
        "val_task_ids": [task.task_id for task in valset],
    }


def main(argv: list[str] | None = None) -> None:
    """Validate the pinned harness and start the requested GEPA condition.

    Raises:
        ValueError: Training or validation selection is empty.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_experiment_model_pair(args.student_model, args.proposer_model)
    except ValueError as exc:
        parser.error(str(exc))
    manifest_path = args.manifest or EXPERIMENT_MANIFESTS[args.experiment]
    manifest = load_terminalbench_manifest(manifest_path)
    if manifest.experiment != args.experiment:
        parser.error("--manifest must match the selected --experiment")
    if args.optimizer_pilot:
        if args.optimizer_pilot_calibration is None:
            parser.error("Optimizer checks require --optimizer-pilot-calibration with the completed full training pilot")
        if args.budget != "standard" or args.reflection_minibatch_size != 3 or args.max_metric_calls is not None:
            parser.error("Optimizer pilots use one cycle on three training tasks and no separate metric cap")
        if args.train_limit is not None or args.val_limit is not None or args.reviewed_pilot is not None:
            parser.error("Optimizer pilots use their fixed training-only selection without validation or review overrides")
        trainset = manifest.tasks("train", 3)
        valset = trainset
    else:
        if args.optimizer_pilot_calibration is not None:
            parser.error("--optimizer-pilot-calibration is only for --optimizer-pilot checks")
        trainset = manifest.tasks("train", args.train_limit)
        valset = manifest.tasks("val", args.val_limit)
    if not trainset or not valset:
        raise ValueError("train and validation selections must both be non-empty")

    candidate, resolved_family = seed_candidate(
        args.student_model, args.template_family, args.experiment, args.optimization_scope
    )
    scope = TerminalBenchTextScope(args.optimization_scope, resolved_family)
    condition = args.condition
    try:
        args.execution_runtime = load_role_runtimes(args)
        contract = build_run_contract(args, manifest, trainset, valset, condition, resolved_family)
        if args.optimizer_pilot:
            contract["optimizer_pilot"] = OPTIMIZER_PILOT_PROTOCOL
            contract["pilot_evaluation_split"] = "train"
            contract["optimization_budget"].update(training_epochs=1, max_iterations=1, sampled_training_tasks=3)
            calibration = load_completed_pilot(args.optimizer_pilot_calibration, manifest, "full")
            validate_runtime(calibration["config"], run_runtime(contract))
            contract["optimizer_pilot_calibration"] = calibration
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    try:
        if args.reviewed_pilot is not None:
            contract["pilot_review"] = review_pilot(args.reviewed_pilot, contract, manifest)
        elif (args.run_dir / RUN_CONTRACT_FILENAME).exists():
            saved_contract = json.loads((args.run_dir / RUN_CONTRACT_FILENAME).read_text())
            contract["pilot_review"] = saved_contract.get("pilot_review")
        if len(trainset) == len(manifest.splits["train"]):
            validate_review(contract["pilot_review"], contract, manifest)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    ensure_run_contract(args.run_dir, contract)
    text_limits = resolve_text_limits(contract["text_limits"])

    student_agent_kwargs: dict[str, Any] = {
        "token_limits": contract["token_limits"],
        "model_info": contract["student_model_info"],
        "llm_kwargs": {
            "num_retries": EXPERIMENT_NUM_RETRIES,
            **contract["student_decoding"],
            **contract["student_request_overrides"],
        },
    }

    harbor = HarborCLI(
        manifest=manifest,
        student_model=args.student_model,
        student_api_base=args.student_api_base,
        work_dir=args.harbor_work_dir,
        agent_python_path=REPO_ROOT,
        n_concurrent=args.n_concurrent,
        harbor_executable=args.harbor_executable,
        docker_executable=args.docker_executable,
        student_agent_kwargs=student_agent_kwargs,
        process_timeout_sec=args.harbor_process_timeout_sec,
        text_limits=text_limits,
    )
    harbor.check_requirements()
    adapter = TerminusAdapter(manifest, harbor, text_scope=scope)

    reflection_lm_kwargs: dict[str, Any] = {
        "num_retries": EXPERIMENT_NUM_RETRIES,
        **provider_retry_kwargs(args.run_dir / "provider-attempts.jsonl", "optimizer"),
        **terminalbench_decoding(args.proposer_model, agentic=False),
        **experiment_request_overrides(args.proposer_model, explicit_reasoning=True),
    }
    if args.proposer_api_base is not None:
        reflection_lm_kwargs["api_base"] = args.proposer_api_base

    usage_path = args.run_dir / "token-usage.jsonl"
    reflection_lm = observe_optimizer(
        LM(args.proposer_model, **reflection_lm_kwargs), usage_path, "stateless_proposer", contract["token_limits"]
    )
    reflection_strategy = None
    if condition in FOREST_CONDITIONS:
        reflection_strategy, _ = build_react_v2_strategy(
            reflection_model=args.proposer_model,
            task_model=args.student_model,
            lm_kwargs=reflection_lm_kwargs,
            level=args.reflection_level,
            edit_tool_set=args.edit_tool_set,
            template_family=resolved_family,
            component_kinds=scope.component_kinds,
            controller_selection=contract["controller_selection"],
            rng=random.Random(args.seed),
            text_limits=text_limits,
            manifestor_temperature=contract["manifestor_temperature"],
            react_top_p=float(contract["reflection_role_decoding"]["react_v2_proposer"]["requested"]["top_p"]),
        )
        shared_controller = reflection_strategy.controller_lm is reflection_strategy.base_lm
        observe_optimizer(
            reflection_strategy.base_lm,
            usage_path,
            "controller-proposer" if shared_controller and condition == "react_v2" else "react_v2_proposer",
            contract["token_limits"],
        )
        if not shared_controller:
            observe_optimizer(reflection_strategy.controller_lm, usage_path, "controller", contract["token_limits"])
        observe_optimizer(reflection_strategy.manifestor_lm, usage_path, "manifestor", contract["token_limits"])
    elif condition == "action":
        reflection_strategy = ComponentActionReflectionLM(
            lm=reflection_lm,
            selector_lm=observe_optimizer(
                LM(args.proposer_model, **reflection_lm_kwargs), usage_path, "action_selector", contract["token_limits"]
            ),
            component_kinds=scope.component_kinds,
            template_family=resolved_family,
            rng=random.Random(args.seed),
            text_limits=text_limits,
        )
    cycle = CycleEvidence(args.run_dir) if args.optimizer_pilot else None
    optimize(
        seed_candidate=candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        reflection_lm_kwargs=reflection_lm_kwargs,
        reflection_strategy=reflection_strategy,
        max_metric_calls=args.max_metric_calls,
        stop_callbacks=MaxCandidateProposalsStopper(contract["optimization_budget"]["max_iterations"]),
        callbacks=[RecoveryCallback(args.run_dir), *([cycle] if cycle else [])],
        batch_sampler=IndependentEpochShuffledBatchSampler(args.reflection_minibatch_size, args.seed),
        reflection_minibatch_size=None,
        sampling_strategy=SingleMutationSampling(),
        module_selector=contract["module_selector"],
        cache_evaluation=contract["cache_evaluation"],
        candidate_selection_strategy=contract["candidate_selection_strategy"],
        frontier_type=contract["frontier_type"],
        acceptance_criterion=contract["acceptance_criterion"],
        skip_perfect_score=contract["skip_perfect_score"],
        perfect_score=contract["perfect_score"],
        val_evaluation_policy=contract["validation_evaluation"],
        use_merge=False,
        raise_on_exception=True,
        run_dir=str(args.run_dir),
        seed=args.seed,
        reflection_level=contract["reflection_level"],
        edit_tool_set=args.edit_tool_set,
        component_kinds=scope.component_kinds,
        template_family=resolved_family,
        template_model=args.student_model,
        text_limits=text_limits,
    )

    if cycle is not None:
        cycle.verify()
        print(f"Training-only optimizer pilot completed: {args.run_dir}")
        return
    if trainset != manifest.tasks("train") or valset != manifest.tasks("val"):
        print("Partial-split diagnostic finished; held-out testing requires the complete campaign splits.")
        return

    # Import after configuration is defined: the evaluator also validates these contracts.
    from examples.terminalbench.evaluate import main as evaluate_main

    cell = f"{args.optimization_scope}__{condition}{'_2x' if args.budget == 'double' else ''}"
    evaluate_main(
        [
            "--run-dir",
            f"{cell}={args.run_dir}",
            "--output-dir",
            str(args.test_output_dir or args.run_dir / "heldout"),
            "--harbor-executable",
            args.harbor_executable,
            "--docker-executable",
            args.docker_executable,
            *(["--runtime-record", str(args.runtime_record)] if args.runtime_record else []),
        ]
    )


if __name__ == "__main__":
    run_guarded(main)
