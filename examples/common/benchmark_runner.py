"""Run comparable vanilla GEPA and FOREST experiments on benchmark adapters."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import random
import statistics
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from examples.common.benchmark_types import BenchmarkBuilder, BenchmarkDefinition, BenchmarkModels
from examples.common.experiment_models import (
    DEFAULT_PROPOSER_MODEL,
    DEFAULT_SOLVER_MODEL,
    experiment_model_version,
)
from examples.common.model_settings import resolve_benchmark_lm_kwargs, validate_benchmark_model_pair
from examples.common.pilot_checks import atomic_json, digest
from examples.common.provider_retries import PROVIDER_RETRY_POLICY, provider_retry_kwargs
from examples.common.react_v2 import benchmark_data_identity, build_react_v2_strategy, resolve_template_family
from gepa import optimize
from gepa.core.adapter import EvaluationBatch
from gepa.lm import LM
from gepa.lm_constants import PROVIDER_ATTEMPT_LOG
from gepa.strategies.batch_sampler import IndependentEpochShuffledBatchSampler
from gepa.strategies.forest_constants import (
    BROAD_EDIT_TOOL_SET,
    DEFAULT_REFLECTION_LEVEL,
    DEFAULT_REFLECTION_MINIBATCH_SIZE,
    OPTIMIZER_ROLE,
    SOLVER_ROLE,
)
from gepa.strategies.proposal_sampling import SingleMutationSampling
from gepa.strategies.proposal_selection import AllImprovements
from gepa.utils.stop_condition import MaxCandidateProposalsStopper

DEFAULT_MAX_METRIC_CALLS = 6_871
DEFAULT_MAX_WORKERS = 1
DEFAULT_SEED = 0
DEFAULT_PILOT_SIZE = 3
RUN_CONTRACT_FILENAME = "benchmark-run-contract.json"


def build_parser(
    benchmark_name: str, add_arguments: Callable[[argparse.ArgumentParser], None]
) -> argparse.ArgumentParser:
    """Build the shared CLI and let the benchmark add only its own settings.

    Args:
        benchmark_name: Stable benchmark identifier.
        add_arguments: Callback adding data and harness configuration flags.

    Returns:
        A parser with consistent solver, proposer and evaluation conventions.
    """
    parser = argparse.ArgumentParser(description=f"Vanilla GEPA and FOREST on {benchmark_name}")
    parser.add_argument("--model", "--solver-model", dest="model", default=DEFAULT_SOLVER_MODEL)
    parser.add_argument(
        "--reflection-model", "--proposer-model", dest="reflection_model", default=DEFAULT_PROPOSER_MODEL
    )
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--solver-api-base", default=None)
    parser.add_argument("--reflection-api-base", default=None)
    parser.add_argument("--run-dir", type=Path, default=Path("outputs") / benchmark_name)
    parser.add_argument("--condition", choices=("vanilla", "react_v2", "both"), default="both")
    parser.add_argument("--mode", choices=("optimize", "pilot", "baseline"), default="optimize")
    parser.add_argument("--max-metric-calls", type=int, default=DEFAULT_MAX_METRIC_CALLS)
    parser.add_argument("--reflection-minibatch-size", type=int, default=DEFAULT_REFLECTION_MINIBATCH_SIZE)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--pilot-size", type=int, default=DEFAULT_PILOT_SIZE)
    parser.add_argument(
        "--template-family", default="auto", choices=("auto", "generic", "openai", "anthropic", "google", "alibaba")
    )
    parser.add_argument("--wandb-project", default=None, help="Write offline W&B optimization logs")
    parser.add_argument("--wandb-entity", default=None)
    add_arguments(parser)
    return parser


def resolve_models(args: argparse.Namespace) -> BenchmarkModels:
    """Resolve the shared model pair with independently selectable endpoints.

    Args:
        args: Parsed shared command-line arguments.

    Returns:
        Pinned model identities and independent request dictionaries.
    """
    validate_benchmark_model_pair(args.model, args.reflection_model)
    solver_base = args.solver_api_base or args.api_base
    proposer_base = args.reflection_api_base or args.api_base
    return BenchmarkModels(
        solver_model=args.model,
        proposer_model=args.reflection_model,
        solver_api_base=solver_base,
        proposer_api_base=proposer_base,
        solver_kwargs=resolve_benchmark_lm_kwargs(args.model, solver_base, role=SOLVER_ROLE),
        proposer_kwargs=resolve_benchmark_lm_kwargs(args.reflection_model, proposer_base, role=OPTIMIZER_ROLE),
    )


def validate_definition(definition: BenchmarkDefinition) -> None:
    """Reject missing data, duplicate IDs and overlap before any model work.

    Args:
        definition: Full benchmark configuration before optional prefix limits.

    Raises:
        ValueError: The benchmark cannot support a disjoint, identified evaluation.
    """
    if not definition.name or not definition.source or not definition.runtime or not definition.metric_name:
        raise ValueError("Benchmark name, immutable data source, runtime and metric must be recorded.")
    if not definition.seed_candidate or any(
        not isinstance(v, str) or not v.strip() for v in definition.seed_candidate.values()
    ):
        raise ValueError("Each editable seed component must contain its actual harness instructions.")
    if not isinstance(definition.test_repetitions, int) or definition.test_repetitions < 1:
        raise ValueError("Test repetitions must be a positive integer.")
    if definition.max_candidate_proposals is not None and (
        type(definition.max_candidate_proposals) is not int or definition.max_candidate_proposals < 1
    ):
        raise ValueError("The candidate proposal budget must be a positive integer.")
    seen: set[str] = set()
    for name, records in (("train", definition.trainset), ("val", definition.valset), ("test", definition.testset)):
        if not records:
            raise ValueError(f"The {name} split must not be empty.")
        ids = [str(record.get("id", "")) for record in records]
        if any(not item for item in ids) or len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError(f"The {name} split has missing, duplicate or overlapping example IDs.")
        seen.update(ids)
    if definition.component_kinds and set(definition.component_kinds) != set(definition.seed_candidate):
        raise ValueError("Component kinds must cover the exact editable prompt components.")
    json.dumps({"source": definition.source, "runtime": definition.runtime}, allow_nan=False)


def ensure_contract(directory: Path, filename: str, contract: dict[str, Any]) -> None:
    """Persist exact settings or refuse to reuse results from a different run.

    Args:
        directory: Directory owning the configuration and its artifacts.
        filename: Contract filename.
        contract: Material settings, including ordered data content fingerprints.

    Raises:
        ValueError: Existing settings or untracked execution artifacts conflict.
    """
    normalized = json.loads(json.dumps(contract, sort_keys=True, allow_nan=False))
    path = directory / filename
    if path.exists():
        if json.loads(path.read_text()) != normalized:
            raise ValueError(f"Benchmark configuration or data changed in {directory}; use a new run directory.")
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f"Existing artifacts in {directory} have no matching {filename}.")
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(path, normalized)


def validate_evaluation(evaluation: EvaluationBatch, count: int, capture_traces: bool = False) -> None:
    """Reject partial or unscored evaluations and invalid per-example timings.

    Args:
        evaluation: Adapter result with one entry per requested example.
        count: Exact expected number of examples.
        capture_traces: Whether optimizer reflection requested trajectories.

    Raises:
        ValueError: Results are incomplete, non-finite or lack episode timing.
    """
    if len(evaluation.outputs) != count or len(evaluation.scores) != count:
        raise ValueError("Benchmark evaluation returned incomplete outputs or scores.")
    if capture_traces and (evaluation.trajectories is None or len(evaluation.trajectories) != count):
        raise ValueError("Reflective evaluation requires one trajectory per example.")
    for output, score in zip(evaluation.outputs, evaluation.scores, strict=True):
        if not isinstance(score, int | float) or not math.isfinite(score):
            raise ValueError("Every benchmark example needs a finite official score.")
        seconds = output.get("elapsed_seconds") if isinstance(output, dict) else None
        if not isinstance(seconds, int | float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Every benchmark output needs a finite, nonnegative elapsed_seconds duration.")
    json.dumps(evaluation.outputs, allow_nan=False)


class RecordedAdapter:
    """Observe task execution without changing the benchmark's solver or reward."""

    def __init__(self, definition: BenchmarkDefinition, directory: Path, seed: int):
        """Keep the official adapter and the exact split membership for logging."""
        self.definition = definition
        self.directory = directory
        self.seed = seed
        self.repetition = 0
        self.phase: str | None = None
        self.propose_new_texts = None
        self._train_ids = {str(row["id"]) for row in definition.trainset}
        self._val_ids = {str(row["id"]) for row in definition.valset}

    def evaluate(self, batch: list[dict], candidate: dict[str, str], capture_traces: bool = False) -> EvaluationBatch:
        """Record episode latency separately from concurrent batch throughput."""
        ids = {str(row["id"]) for row in batch}
        phase = self.phase
        if phase is None:
            if ids <= self._train_ids:
                phase = "train"
            elif ids <= self._val_ids:
                phase = "val"
            else:
                raise ValueError("Optimization tried to evaluate records outside the training/validation splits.")
        set_context = getattr(self.definition.adapter, "set_evaluation_context", None)
        if set_context is not None:
            set_context(split=phase, repetition=self.repetition, seed=self.seed + self.repetition)
        started = time.perf_counter()
        evaluated = self.definition.adapter.evaluate(batch, deepcopy(candidate), capture_traces=capture_traces)
        elapsed = time.perf_counter() - started
        validate_evaluation(evaluated, len(batch), capture_traces)
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / "task-timings.jsonl").open("a") as stream:
            for record, output, score in zip(batch, evaluated.outputs, evaluated.scores, strict=True):
                stream.write(
                    json.dumps(
                        {
                            "id": str(record["id"]),
                            "split": phase,
                            "repetition": self.repetition,
                            "candidate_sha256": digest(candidate),
                            "score": score,
                            "elapsed_seconds": output["elapsed_seconds"],
                            "error": output.get("error"),
                            "batch_elapsed_seconds": elapsed,
                            "batch_count": len(batch),
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
        return evaluated

    def make_reflective_dataset(
        self, candidate: dict, eval_batch: EvaluationBatch, components_to_update: list[str]
    ) -> Any:
        """Forward the benchmark's real trace-to-feedback mapping unchanged."""
        return self.definition.adapter.make_reflective_dataset(candidate, eval_batch, components_to_update)

    def get_adapter_state(self) -> dict[str, Any]:
        """Preserve optional upstream state in GEPA checkpoints."""
        getter = getattr(self.definition.adapter, "get_adapter_state", None)
        return deepcopy(getter()) if getter is not None else {}

    def set_adapter_state(self, state: dict[str, Any]) -> None:
        """Restore upstream state when the official adapter exposes that hook."""
        setter = getattr(self.definition.adapter, "set_adapter_state", None)
        if setter is not None:
            setter(deepcopy(state))
        elif state:
            raise ValueError("Stored benchmark state cannot be restored by this adapter.")


def evaluate_candidate(
    definition: BenchmarkDefinition,
    candidate: dict[str, str],
    records: list[dict],
    directory: Path,
    identity: dict[str, Any],
    *,
    split: str,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate a fixed candidate with reusable, verified per-repetition evidence.

    Args:
        definition: Benchmark and official adapter.
        candidate: Frozen harness prompts.
        records: Exact ordered examples for this evaluation.
        directory: Evaluation evidence directory.
        identity: Model, data and harness settings common to compared methods.
        split: Training pilot or held-out test label.
        repetitions: Number of independent benchmark-protocol trials.
        seed: Fixed seed shared across method arms.

    Returns:
        Scores, benchmark-specific aggregate metrics, latency and throughput.
    """
    contract = {
        "schema_version": 1,
        "identity": identity,
        "candidate": candidate,
        "records_sha256": digest(records),
        "ids": [str(row["id"]) for row in records],
        "split": split,
        "repetitions": repetitions,
        "seed": seed,
    }
    ensure_contract(directory, "evaluation-contract.json", contract)
    batches: list[EvaluationBatch] = []
    total_elapsed = 0.0
    observer = RecordedAdapter(definition, directory, seed)
    observer.phase = split
    for repetition in range(repetitions):
        path = directory / f"repetition-{repetition:03d}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            payload = saved["payload"]
            if saved.get("sha256") != digest(payload) or payload.get("contract_sha256") != digest(contract):
                raise ValueError(f"Stored evaluation evidence changed: {path}")
            if payload.get("repetition") != repetition:
                raise ValueError("Stored evaluation has a different repetition index.")
            evaluated = EvaluationBatch(
                outputs=payload["outputs"],
                scores=payload["scores"],
                objective_scores=payload.get("objective_scores"),
            )
            validate_evaluation(evaluated, len(records))
            elapsed = payload["elapsed_seconds"]
            if not isinstance(elapsed, int | float) or not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("Stored evaluation batch duration is invalid.")
        else:
            observer.repetition = repetition
            started = time.perf_counter()
            evaluated = observer.evaluate(records, candidate)
            elapsed = time.perf_counter() - started
            payload = {
                "contract_sha256": digest(contract),
                "repetition": repetition,
                "outputs": evaluated.outputs,
                "scores": evaluated.scores,
                "objective_scores": evaluated.objective_scores,
                "elapsed_seconds": elapsed,
            }
            atomic_json(path, {"sha256": digest(payload), "payload": payload})
        batches.append(evaluated)
        total_elapsed += elapsed
    durations = [output["elapsed_seconds"] for batch in batches for output in batch.outputs]
    scores = [float(score) for batch in batches for score in batch.scores]
    summarize = getattr(definition.adapter, "summarize_evaluation", None)
    metrics = (
        summarize(records, batches) if summarize is not None else {definition.metric_name: statistics.fmean(scores)}
    )
    summary = {
        "schema_version": 1,
        "candidate_sha256": digest(candidate),
        "example_count": len(records),
        "repetitions": repetitions,
        "mean_score": statistics.fmean(scores),
        "metrics": metrics,
        "repetition_mean_scores": [statistics.fmean(batch.scores) for batch in batches],
        "timing": {
            "attempt_count": len(durations),
            "mean_seconds": statistics.fmean(durations),
            "median_seconds": statistics.median(durations),
            "p95_seconds": sorted(durations)[math.ceil(0.95 * len(durations)) - 1],
            "failed_attempts": sum(bool(output.get("error")) for batch in batches for output in batch.outputs),
            "recorded_batch_seconds": total_elapsed,
            "tasks_per_hour": len(durations) * 3600 / total_elapsed,
        },
    }
    json.dumps(summary, allow_nan=False)
    atomic_json(directory / "summary.json", summary)
    return summary


def _starting_baseline(
    definition: BenchmarkDefinition,
    root: Path,
    identity: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Share the initial harness result across matched methods and budgets."""
    key = digest({"identity": identity, "candidate": definition.seed_candidate, "seed": seed})
    directory = root.parent / "benchmark-baselines" / definition.name / key
    directory.parent.mkdir(parents=True, exist_ok=True)
    with (directory.parent / f".{key}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process is evaluating this shared starting baseline.") from exc
        return evaluate_candidate(
            definition,
            definition.seed_candidate,
            definition.testset,
            directory,
            identity,
            split="test",
            repetitions=definition.test_repetitions,
            seed=seed,
        )


def _run_condition(
    definition: BenchmarkDefinition,
    models: BenchmarkModels,
    args: argparse.Namespace,
    identity: dict[str, Any],
    condition: str,
) -> dict[str, Any]:
    """Freeze a validation winner before evaluating it and its shared baseline."""
    directory = args.run_dir / condition
    component_kinds = definition.component_kinds or dict.fromkeys(definition.seed_candidate, "system_prompt")
    template_family = resolve_template_family(args.template_family, models.solver_model)
    batch_size = min(args.reflection_minibatch_size, len(definition.trainset))
    contract = {
        "schema_version": 1,
        "identity": identity,
        "condition": condition,
        "proposer": {
            "model": models.proposer_model,
            "revision": experiment_model_version(models.proposer_model),
            "kwargs": models.proposer_kwargs,
        },
        "optimizer": {
            "max_metric_calls": args.max_metric_calls,
            "max_candidate_proposals": definition.max_candidate_proposals,
            "reflection_minibatch_size": batch_size,
            "module_selector": "round_robin",
            "candidate_selection": "pareto",
            "acceptance": "strict_improvement",
            "proposal_selection": "all_improvements",
            "merge": False,
            "seed": args.seed,
            "template_family": template_family,
            "reflection_level": DEFAULT_REFLECTION_LEVEL if condition == "react_v2" else 0,
            "edit_tool_set": BROAD_EDIT_TOOL_SET,
            "component_kinds": component_kinds,
        },
    }
    ensure_contract(directory, RUN_CONTRACT_FILENAME, contract)
    winner_path = directory / "frozen-winner.json"
    if winner_path.exists():
        winner = json.loads(winner_path.read_text())
        if winner.get("contract_sha256") != digest(contract) or winner.get("candidate_sha256") != digest(
            winner.get("candidate")
        ):
            raise ValueError("Frozen validation winner no longer matches its run contract.")
        candidate = winner["candidate"]
    else:
        adapter = RecordedAdapter(definition, directory, args.seed)
        proposer_kwargs = {
            **deepcopy(models.proposer_kwargs),
            **provider_retry_kwargs(directory / PROVIDER_ATTEMPT_LOG, OPTIMIZER_ROLE),
            "response_journal_path": str(directory / "responses.sqlite"),
            "response_journal_namespace": OPTIMIZER_ROLE,
        }
        strategy = None
        if condition == "react_v2":
            strategy, _ = build_react_v2_strategy(
                reflection_model=models.proposer_model,
                task_model=models.solver_model,
                lm_kwargs=proposer_kwargs,
                level=DEFAULT_REFLECTION_LEVEL,
                edit_tool_set=BROAD_EDIT_TOOL_SET,
                template_family=template_family,
                component_kinds=component_kinds,
                rng=random.Random(args.seed),
                manifestor_temperature=float(proposer_kwargs["temperature"]),
            )
        result = optimize(
            seed_candidate=deepcopy(definition.seed_candidate),
            trainset=definition.trainset,
            valset=definition.valset,
            adapter=adapter,
            reflection_lm=LM(models.proposer_model, **proposer_kwargs),
            reflection_strategy=strategy,
            candidate_selection_strategy="pareto",
            frontier_type="instance",
            module_selector="round_robin",
            batch_sampler=IndependentEpochShuffledBatchSampler(minibatch_size=batch_size, seed=args.seed),
            sampling_strategy=SingleMutationSampling(),
            selection_strategy=AllImprovements(),
            acceptance_criterion="strict_improvement",
            use_merge=False,
            max_metric_calls=args.max_metric_calls,
            stop_callbacks=(
                MaxCandidateProposalsStopper(definition.max_candidate_proposals)
                if definition.max_candidate_proposals is not None
                else None
            ),
            run_dir=str(directory),
            seed=args.seed,
            raise_on_exception=True,
            val_evaluation_policy="full_eval",
            write_agent_state=True,
            track_best_outputs=True,
            cache_evaluation=False,
            component_kinds=component_kinds,
            template_family=template_family,
            use_wandb=bool(args.wandb_project),
            wandb_init_kwargs={
                "project": args.wandb_project,
                "entity": args.wandb_entity,
                "name": f"{definition.name}-{condition}",
                "mode": "offline",
                "config": contract,
            },
        )
        candidate = result.best_candidate
        if not isinstance(candidate, dict) or set(candidate) != set(definition.seed_candidate):
            raise ValueError("GEPA returned an invalid benchmark harness candidate.")
        winner = {
            "contract_sha256": digest(contract),
            "candidate": candidate,
            "candidate_sha256": digest(candidate),
            "validation_score": result.best_score,
            "best_idx": result.best_idx,
            "total_metric_calls": result.total_metric_calls,
        }
        atomic_json(directory / "candidates.json", result.to_dict())
        atomic_json(winner_path, winner)
    test = evaluate_candidate(
        definition,
        candidate,
        definition.testset,
        directory / "heldout",
        identity,
        split="test",
        repetitions=definition.test_repetitions,
        seed=args.seed,
    )
    baseline = _starting_baseline(definition, args.run_dir, identity, args.seed)
    summary = {
        "condition": condition,
        "winner": winner,
        "test": test,
        "baseline": baseline,
        "improvement": test["mean_score"] - baseline["mean_score"],
    }
    atomic_json(directory / "summary.json", summary)
    return summary


def run_cli(
    *,
    benchmark_name: str,
    build_benchmark: BenchmarkBuilder,
    add_arguments: Callable[[argparse.ArgumentParser], None],
    argv: list[str] | None = None,
    configure_models: Callable[[argparse.Namespace, BenchmarkModels], BenchmarkModels] | None = None,
) -> int:
    """Execute a benchmark using shared model, split and held-out conventions.

    Args:
        benchmark_name: Stable CLI and output-directory name.
        build_benchmark: Builder for the official adapter and pinned data.
        add_arguments: Callback adding benchmark-specific CLI flags.
        argv: Explicit arguments for tests; None reads the process arguments.
        configure_models: Optional benchmark-specific role budget configuration.

    Returns:
        Zero after all requested stages finish and verified results are saved.
    """
    parser = build_parser(benchmark_name, add_arguments)
    args = parser.parse_args(argv)
    for key in (
        "max_metric_calls",
        "reflection_minibatch_size",
        "max_workers",
        "pilot_size",
        "train_limit",
        "val_limit",
        "test_limit",
    ):
        value = getattr(args, key)
        if value is not None and value < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    models = resolve_models(args)
    if configure_models is not None:
        models = configure_models(args, models)
        validate_benchmark_model_pair(models.solver_model, models.proposer_model)
    definition = build_benchmark(args, models)
    validate_definition(definition)
    if args.mode == "optimize" and args.max_metric_calls is None and definition.max_candidate_proposals is None:
        parser.error("Optimization requires a metric-call or candidate-proposal budget")
    full_identity = benchmark_data_identity(
        source=definition.source, trainset=definition.trainset, valset=definition.valset, testset=definition.testset
    )
    definition = replace(
        definition,
        trainset=definition.trainset[: args.train_limit],
        valset=definition.valset[: args.val_limit],
        testset=definition.testset[: args.test_limit],
    )
    identity = {
        "benchmark": definition.name,
        "seed_candidate": definition.seed_candidate,
        "full_data": full_identity,
        "data": benchmark_data_identity(
            source=definition.source, trainset=definition.trainset, valset=definition.valset, testset=definition.testset
        ),
        "runtime": definition.runtime,
        "max_workers": args.max_workers,
        "test_repetitions": definition.test_repetitions,
        "solver": {
            "model": models.solver_model,
            "revision": experiment_model_version(models.solver_model),
            "kwargs": models.solver_kwargs,
        },
        "provider_retry_policy": PROVIDER_RETRY_POLICY,
    }
    if args.mode == "pilot":
        summary = evaluate_candidate(
            definition,
            definition.seed_candidate,
            definition.trainset[: args.pilot_size],
            args.run_dir / "pilot",
            identity,
            split="train",
            repetitions=1,
            seed=args.seed,
        )
    elif args.mode == "baseline":
        summary = _starting_baseline(definition, args.run_dir, identity, args.seed)
    else:
        conditions = ("vanilla", "react_v2") if args.condition == "both" else (args.condition,)
        summary = {condition: _run_condition(definition, models, args, identity, condition) for condition in conditions}
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0
