"""Adapt the pinned Terminus harness to the shared benchmark record protocol."""

from __future__ import annotations

import math
import statistics
from datetime import datetime
from typing import Any

from gepa.adapters.terminal_bench_adapter import HarborExecutionError, TerminalBenchTask, TerminusAdapter
from gepa.core.adapter import EvaluationBatch, ProposalFn


def trial_elapsed_seconds(result: dict[str, Any]) -> float:
    """Use Harbor's complete trial lifetime, including environment teardown."""
    try:
        start = datetime.fromisoformat(result["started_at"].replace("Z", "+00:00"))
        finish = datetime.fromisoformat(result["finished_at"].replace("Z", "+00:00"))
        if start.utcoffset() is None or finish.utcoffset() is None:
            raise ValueError("Harbor timestamps must include their timezone.")
        elapsed = (finish - start).total_seconds()
    except (KeyError, AttributeError, TypeError, ValueError) as error:
        raise HarborExecutionError("Missing or malformed official Harbor per-task timing.") from error
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise HarborExecutionError("Official Harbor per-task duration must be positive and finite.")
    return elapsed


class SharedTerminusAdapter:
    """Preserve official rewards and prompt execution while adding record IDs and latency."""

    propose_new_texts: ProposalFn | None = None

    def __init__(self, adapter: TerminusAdapter, records: list[dict[str, Any]]):
        self.adapter = adapter
        self.records = {record["id"]: dict(record) for record in records}
        self.split: str | None = None

    def set_evaluation_context(self, *, split: str, repetition: int, seed: int) -> None:
        """Use deterministic distinct request seeds across held-out repetitions."""
        if split not in {"train", "val", "test"} or repetition < 0:
            raise ValueError("Invalid Terminal-Bench evaluation context.")
        self.split = split
        self.adapter.harbor.student_agent_kwargs["llm_kwargs"]["seed"] = seed

    def evaluate(
        self, batch: list[dict[str, Any]], candidate: dict[str, str], capture_traces: bool = False
    ) -> EvaluationBatch:
        """Run the existing adapter and obtain each trial's own authoritative timestamps."""
        if any(self.records.get(record.get("id")) != record for record in batch):
            raise ValueError("Terminal-Bench task or immutable ref drift.")
        if self.split is not None and any(record["split"] != self.split for record in batch):
            raise ValueError("Terminal-Bench task does not belong to the requested evaluation split.")
        result = self.adapter.evaluate(
            [TerminalBenchTask(record["task_id"]) for record in batch], candidate, capture_traces=True
        )
        if result.trajectories is None or not (
            len(result.outputs) == len(result.scores) == len(result.trajectories) == len(batch)
        ):
            raise HarborExecutionError("Incomplete Terminal-Bench trial evidence.")
        outputs = []
        for record, output, trajectory, score in zip(
            batch, result.outputs, result.trajectories, result.scores, strict=True
        ):
            if output["task_id"] != record["task_id"] or trajectory["task_id"] != record["task_id"]:
                raise HarborExecutionError("Terminal-Bench result ordering changed.")
            if type(score) not in (int, float) or not math.isfinite(score) or output["reward"] != score:
                raise HarborExecutionError("Invalid or inconsistent official Harbor reward.")
            outputs.append(
                {
                    **output,
                    **record,
                    "elapsed_seconds": trial_elapsed_seconds(trajectory["trial_result"]),
                    "error": "; ".join(output["errors"]) if output["errors"] else None,
                }
            )
        return EvaluationBatch(
            outputs=outputs,
            scores=result.scores,
            trajectories=result.trajectories if capture_traces else None,
            objective_scores=result.objective_scores,
            num_metric_calls=result.num_metric_calls,
        )

    def make_reflective_dataset(self, candidate: dict, eval_batch: EvaluationBatch, components_to_update: list[str]):
        """Retain the existing training-only ATIF and verifier reflection contract."""
        return self.adapter.make_reflective_dataset(candidate, eval_batch, components_to_update)

    def summarize_evaluation(self, records: list[dict[str, Any]], evaluations: list[EvaluationBatch]) -> dict[str, Any]:
        """Report pass-at-one across fresh repetitions using official verifier rewards."""
        if not records or not evaluations:
            raise ValueError("Cannot summarize an empty Terminal-Bench evaluation.")
        if any(self.records.get(record.get("id")) != record for record in records):
            raise ValueError("Terminal-Bench summary records differ from the pinned manifest.")
        ids = [record["id"] for record in records]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate Terminal-Bench summary task IDs.")
        means = []
        seen_evaluations: set[str] = set()
        for evaluation in evaluations:
            if [output.get("id") for output in evaluation.outputs] != ids or len(evaluation.scores) != len(ids):
                raise ValueError("Incomplete or reordered Terminal-Bench evaluation.")
            evaluation_ids = {output["evaluation_id"] for output in evaluation.outputs}
            if (
                len(evaluation_ids) != 1
                or not next(iter(evaluation_ids))
                or seen_evaluations.intersection(evaluation_ids)
            ):
                raise ValueError("Each Terminal-Bench repetition requires a distinct official Harbor job.")
            seen_evaluations.update(evaluation_ids)
            for output, score in zip(evaluation.outputs, evaluation.scores, strict=True):
                if type(score) not in (int, float) or not math.isfinite(score) or output["reward"] != score:
                    raise ValueError("Terminal-Bench score differs from the official reward.")
            means.append(statistics.fmean(evaluation.scores))
        return {
            "pass_at_1": statistics.fmean(means),
            "repetition_pass_at_1": means,
            "sample_standard_deviation": statistics.stdev(means) if len(means) > 1 else None,
            "repetitions": len(means),
            "tasks_per_repetition": len(records),
            "score_source": "official_harbor_verifier_reward",
        }
