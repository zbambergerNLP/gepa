"""Run comparable vanilla GEPA and FOREST experiments on benchmark adapters."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import platform
import random
import statistics
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, cast

from examples.common.artifacts import atomic_json, digest
from examples.common.benchmark_pilot import OptimizerPilotEvidence
from examples.common.benchmark_settings import (
    DEFAULT_MAX_METRIC_CALLS,
    DEFAULT_MAX_WORKERS,
    DEFAULT_PILOT_SIZE,
    DEFAULT_SEED,
)
from examples.common.benchmark_types import BenchmarkBuilder, BenchmarkDefinition, BenchmarkModels
from examples.common.benchmark_variants import (
    FOREST_CONDITIONS,
    add_variant_arguments,
    optimizer_budget,
    proposal_strategies,
    selected_conditions,
    variant_settings,
)
from examples.common.experiment_models import (
    DEFAULT_PROPOSER_MODEL,
    DEFAULT_SOLVER_MODEL,
    experiment_model_version,
)
from examples.common.model_settings import resolve_benchmark_lm_kwargs, validate_benchmark_model_pair
from examples.common.provider_retries import PROVIDER_RETRY_POLICY, provider_retry_kwargs
from examples.common.react_v2 import (
    benchmark_data_identity,
    build_react_v2_strategy,
    file_sha256,
    resolve_template_family,
)
from gepa import optimize
from gepa.core.adapter import EvaluationBatch, ProposalFn
from gepa.core.callbacks import GEPACallback
from gepa.lm import LM
from gepa.lm_constants import PROVIDER_ATTEMPT_LOG, PROVIDER_RETRY_KEY
from gepa.strategies.action_space import (
    ActionSelector,
    RandomActionSelector,
    VerbalizedActionSelector,
    stateless_selector_policy_contract,
)
from gepa.strategies.batch_sampler import IndependentEpochShuffledBatchSampler
from gepa.strategies.document_template import TEMPLATE_FAMILIES
from gepa.strategies.forest_constants import (
    BROAD_EDIT_TOOL_SET,
    DEFAULT_REFLECTION_MINIBATCH_SIZE,
    OPTIMIZER_ROLE,
    SOLVER_ROLE,
)
from gepa.strategies.intervention import SEMANTIC_ACTIONS, StatelessActionConstraint
from gepa.utils.stop_condition import MaxCandidateProposalsStopper

RUN_CONTRACT_FILENAME = "benchmark-run-contract.json"
TemplateFamily = Literal["generic", "openai", "anthropic", "google", "alibaba"]


def implementation_identity(benchmark_name: str) -> dict[str, Any]:
    """Fingerprint the shared engine, benchmark harness and installed model transport."""
    root = Path(__file__).resolve().parents[2]
    files = (
        sorted((root / "src" / "gepa").rglob("*.py"))
        + sorted((root / "examples" / "common").glob("*.py"))
        + sorted((root / "examples" / benchmark_name).glob("*.py"))
    )
    return {
        "source_sha256": digest([[str(path.relative_to(root)), file_sha256(path)] for path in files]),
        "lock_sha256": file_sha256(root / "uv.lock"),
        "python": platform.python_version(),
        "litellm": version("litellm"),
    }


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
    add_variant_arguments(parser)
    parser.add_argument("--mode", choices=("optimize", "pilot", "optimizer-pilot", "baseline"), default="optimize")
    parser.add_argument("--max-metric-calls", type=int, default=DEFAULT_MAX_METRIC_CALLS)
    parser.add_argument("--reflection-minibatch-size", type=int, default=DEFAULT_REFLECTION_MINIBATCH_SIZE)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--pilot-size", type=int, default=DEFAULT_PILOT_SIZE)
    parser.add_argument("--pilot-proposals", type=int, default=1, help="Training-only optimizer-pilot iteration limit")
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
        solver_kwargs={
            **resolve_benchmark_lm_kwargs(args.model, solver_base, role=SOLVER_ROLE),
            **provider_retry_kwargs(args.run_dir / PROVIDER_ATTEMPT_LOG, SOLVER_ROLE),
        },
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
    json.dumps(
        {"source": definition.source, "runtime": definition.runtime, "optimizer_runtime": definition.optimizer_runtime},
        allow_nan=False,
    )


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
        self.propose_new_texts: ProposalFn | None = None
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
    """Hold one exclusive writer lock for each resumable optimization directory."""
    root = args.run_dir / "optimizer-pilot" if args.mode == "optimizer-pilot" else args.run_dir
    root.mkdir(parents=True, exist_ok=True)
    with (root / f".{condition}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is running {condition} in {args.run_dir}.") from exc
        return _run_condition_locked(definition, models, args, identity, condition)


def _run_condition_locked(
    definition: BenchmarkDefinition,
    models: BenchmarkModels,
    args: argparse.Namespace,
    identity: dict[str, Any],
    condition: str,
) -> dict[str, Any]:
    """Freeze the selected winner, withholding all held-out work from optimizer pilots."""
    pilot = args.mode == "optimizer-pilot"
    directory = args.run_dir / "optimizer-pilot" / condition if pilot else args.run_dir / condition
    component_kinds = definition.component_kinds or dict.fromkeys(definition.seed_candidate, "system_prompt")
    template_family = cast(TemplateFamily, resolve_template_family(args.template_family, models.solver_model))
    settings = variant_settings(args, condition)
    trainset = definition.trainset[: args.pilot_size] if pilot else definition.trainset
    selection_set = trainset if pilot else definition.valset
    batch_size = min(args.reflection_minibatch_size, len(trainset))
    budget = optimizer_budget(args, definition.max_candidate_proposals)
    contract = {
        "schema_version": 2,
        "identity": identity,
        "condition": condition,
        "mode": args.mode,
        "selection_split": "train" if pilot else "val",
        "optimization_data": {
            "train_ids": [str(row["id"]) for row in trainset],
            "selection_ids": [str(row["id"]) for row in selection_set],
        },
        "proposer": {
            "model": models.proposer_model,
            "revision": experiment_model_version(models.proposer_model),
            "kwargs": models.proposer_kwargs,
            "runtime": definition.optimizer_runtime,
        },
        "optimizer": {
            **settings,
            **budget,
            "reflection_minibatch_size": batch_size,
            "skip_perfect_score": not pilot,
            "seed": args.seed,
            "template_family": template_family,
            "component_kinds": component_kinds,
            "stateless_selector_policy": (
                stateless_selector_policy_contract(settings["stateless_action_selection"])
                if settings["stateless_action_selection"]
                else None
            ),
        },
    }
    ensure_contract(directory, RUN_CONTRACT_FILENAME, contract)
    pilot_evidence_path = directory / "optimizer-pilot-evidence.json"
    pilot_observer = OptimizerPilotEvidence(digest(contract)) if pilot else None
    winner_path = directory / ("pilot-winner.json" if pilot else "frozen-winner.json")
    if winner_path.exists():
        winner = json.loads(winner_path.read_text())
        if winner.get("contract_sha256") != digest(contract) or winner.get("candidate_sha256") != digest(
            winner.get("candidate")
        ):
            raise ValueError("Frozen validation winner no longer matches its run contract.")
        if pilot:
            evidence = json.loads(pilot_evidence_path.read_text())
            if (
                evidence.get("contract_sha256") != digest(contract)
                or not evidence.get("completed_cycles")
                or winner.get("pilot_evidence_sha256") != digest(evidence)
            ):
                raise ValueError("Optimizer pilot completion evidence no longer matches its winner and run contract")
        candidate = winner["candidate"]
    else:
        adapter = RecordedAdapter(definition, directory, args.seed)
        proposer_kwargs = {
            **deepcopy(models.proposer_kwargs),
            **provider_retry_kwargs(directory / PROVIDER_ATTEMPT_LOG, OPTIMIZER_ROLE),
            "response_journal_path": str(directory / "responses.sqlite"),
            "response_journal_namespace": OPTIMIZER_ROLE,
        }
        strategy, action_selector = None, None
        if condition in FOREST_CONDITIONS:
            strategy, _ = build_react_v2_strategy(
                reflection_model=models.proposer_model,
                task_model=models.solver_model,
                lm_kwargs=proposer_kwargs,
                level=settings["reflection_level"],
                edit_tool_set=settings["edit_tool_set"] or BROAD_EDIT_TOOL_SET,
                controller_selection=settings["controller_selection"] or "verbalized",
                editor_mode=settings["editor_mode"] or "react",
                proposal_policy=settings["proposal_policy"] or "independent",
                react_max_iterations=settings["react_max_iterations"],
                react_max_tool_calls=settings["react_max_tool_calls"],
                template_family=template_family,
                component_kinds=component_kinds,
                rng=random.Random(args.seed),
                manifestor_temperature=float(proposer_kwargs["temperature"]),
            )
        if settings["stateless_action_selection"]:
            kinds = set(component_kinds.values())
            if len(kinds) != 1:
                raise ValueError("Stateless action conditions require one shared component template kind")
            template = TEMPLATE_FAMILIES[template_family][next(iter(kinds))]
            actions = [
                StatelessActionConstraint(spec, section, template)
                for section in template.sections
                for spec in SEMANTIC_ACTIONS
            ]
            if condition == "random":
                action_selector = RandomActionSelector(actions, rng=random.Random(args.seed))
            else:
                action_selector = VerbalizedActionSelector(
                    actions,
                    lm=LM(
                        models.proposer_model,
                        **{**proposer_kwargs, "response_journal_namespace": "stateless-controller"},
                    ),
                    rng=random.Random(args.seed),
                )
        sampling, selection = proposal_strategies(args)
        result = optimize(
            seed_candidate=deepcopy(definition.seed_candidate),
            trainset=trainset,
            valset=selection_set,
            adapter=adapter,
            callbacks=[cast(GEPACallback, pilot_observer)] if pilot_observer is not None else None,
            reflection_lm=LM(models.proposer_model, **proposer_kwargs),
            reflection_strategy=strategy,
            action_selector=cast(ActionSelector[StatelessActionConstraint] | None, action_selector),
            candidate_selection_strategy=settings["candidate_selection"],
            frontier_type=settings["frontier_type"],
            module_selector=settings["module_selector"],
            batch_sampler=IndependentEpochShuffledBatchSampler(minibatch_size=batch_size, seed=args.seed),
            sampling_strategy=sampling,
            selection_strategy=selection,
            acceptance_criterion=settings["acceptance"],
            use_merge=settings["merge"],
            max_merge_invocations=settings["max_merge_invocations"],
            skip_perfect_score=not pilot,
            max_metric_calls=budget["max_metric_calls"],
            stop_callbacks=(
                MaxCandidateProposalsStopper(budget["max_optimizer_iterations"])
                if budget["max_optimizer_iterations"] is not None
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
        evidence = pilot_observer.completion_evidence() if pilot_observer is not None else None
        winner = {
            "contract_sha256": digest(contract),
            "candidate": candidate,
            "candidate_sha256": digest(candidate),
            "training_score" if pilot else "validation_score": result.best_score,
            "selection_split": "train" if pilot else "val",
            "best_idx": result.best_idx,
            "total_metric_calls": result.total_metric_calls,
        }
        if evidence is not None:
            atomic_json(pilot_evidence_path, evidence)
            winner["pilot_evidence_sha256"] = digest(evidence)
        atomic_json(directory / "candidates.json", result.to_dict())
        atomic_json(winner_path, winner)
    if pilot:
        summary = {"condition": condition, "mode": "optimizer-pilot", "winner": winner}
        atomic_json(directory / "summary.json", summary)
        return summary
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
        "pilot_proposals",
        "proposal_count",
        "parent_count",
        "proposal_top_k",
        "react_max_iterations",
        "react_max_tool_calls",
        "train_limit",
        "val_limit",
        "test_limit",
    ):
        value = getattr(args, key)
        if value is not None and value < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    conditions = selected_conditions(args)
    if args.mode in {"optimize", "optimizer-pilot"}:
        try:
            for condition in conditions:
                variant_settings(args, condition)
        except ValueError as exc:
            parser.error(str(exc))
    models = resolve_models(args)
    if configure_models is not None:
        models = configure_models(args, models)
        validate_benchmark_model_pair(models.solver_model, models.proposer_model)
    definition = build_benchmark(args, models)
    validate_definition(definition)
    if args.mode in {"optimize", "optimizer-pilot"} and {"random", "action"}.intersection(conditions):
        if args.module_selector == "all" and len(definition.seed_candidate) > 1:
            parser.error(
                "Stateless random/action conditions require one module per proposal; use --module-selector round_robin"
            )
        if len(set(definition.component_kinds.values())) > 1:
            parser.error("Stateless random/action conditions require one shared component template kind")
    if args.mode in {"optimize", "optimizer-pilot"}:
        try:
            optimizer_budget(args, definition.max_candidate_proposals)
        except ValueError as exc:
            parser.error(str(exc))
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
    solver_identity_kwargs = deepcopy(models.solver_kwargs)
    if PROVIDER_RETRY_KEY in solver_identity_kwargs:
        solver_identity_kwargs[PROVIDER_RETRY_KEY]["log_path"] = None
    identity = {
        "benchmark": definition.name,
        "implementation": implementation_identity(benchmark_name),
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
            "kwargs": solver_identity_kwargs,
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
        summary = {condition: _run_condition(definition, models, args, identity, condition) for condition in conditions}
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0
