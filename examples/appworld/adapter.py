"""Run upstream-style ReAct episodes and score their actual AppWorld state."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

from examples.appworld.benchmark_settings import COMPONENT, DEFAULT_MAX_STEPS
from examples.appworld.prompts import extract_code, initial_messages
from examples.appworld.runtime import AppWorldRuntimeError
from examples.appworld.utils import scenario_members, tree_digest
from gepa.core.adapter import EvaluationBatch


def validate_evaluation(result: dict[str, Any], task_id: str) -> bool:
    """Require complete official evidence; never coerce text or missing fields to success."""
    counts = [result.get(key) for key in ("num_tests", "passed", "failed")]
    if any(type(value) is not int or value < 0 for value in counts):
        raise AppWorldRuntimeError("Malformed official evaluator counts.")
    total, passed, failed = counts
    if total == 0 or total != passed + failed or type(result.get("success")) is not bool:
        raise AppWorldRuntimeError("Incomplete official evaluator result.")
    if result["success"] != (total == passed) or result.get("task_id") != task_id:
        raise AppWorldRuntimeError("Official evaluator identity or score mismatch.")
    if not isinstance(result.get("evaluation_path"), str) or not result["evaluation_path"]:
        raise AppWorldRuntimeError("Missing saved official evaluation evidence.")
    if not isinstance(result.get("evaluation_sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", result["evaluation_sha256"]
    ):
        raise AppWorldRuntimeError("Missing saved official evaluation fingerprint.")
    return result["success"]


class AppWorldAdapter:
    """Expose one real system prompt to both GEPA and FOREST with identical execution."""

    def __init__(
        self,
        task_model: Callable[[list[dict[str, str]]], str],
        world_factory: Callable,
        records: list[dict[str, Any]],
        data_root: Path,
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
    ):
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("AppWorld max_steps must be a positive integer.")
        self.task_model = task_model
        self.world_factory = world_factory
        self.records = {record["id"]: dict(record) for record in records}
        if len(self.records) != len(records):
            raise ValueError("Duplicate AppWorld record IDs.")
        self.scenarios = scenario_members(records)
        self.data_root = data_root
        self.max_steps = max_steps

    def evaluate(
        self, batch: list[dict[str, Any]], candidate: dict[str, str], capture_traces: bool = False
    ) -> EvaluationBatch:
        """Execute fresh task worlds sequentially, measuring end-to-end episode latency."""
        if (
            set(candidate) != {COMPONENT}
            or not isinstance(candidate[COMPONENT], str)
            or not candidate[COMPONENT].strip()
        ):
            raise ValueError("AppWorld requires exactly one nonempty system_prompt component.")
        for record in batch:
            if self.records.get(record.get("id")) != record:
                raise ValueError("AppWorld record or official split identity drift.")
            if tree_digest(self.data_root / "data" / "tasks" / record["task_id"]) != record["record_sha256"]:
                raise ValueError("AppWorld task content drift before rollout.")
        outputs, trajectories, scores = [], [], []
        for record in batch:
            started = time.perf_counter()
            error = None
            termination = "max_steps"
            completed = False
            steps = 0
            with self.world_factory() as world:
                context = world.initialize(record)
                messages = initial_messages(candidate, context)
                for _ in range(self.max_steps):
                    steps += 1
                    # Exceptions from the shared provider policy abort the run;
                    # an unavailable model/environment is not a benchmark miss.
                    response = self.task_model(deepcopy(messages))
                    try:
                        if not isinstance(response, str):
                            raise ValueError("Model returned non-text output.")
                        code, action = extract_code(response)
                    except ValueError:
                        error = "malformed_action"
                        termination = error
                        messages.append({"role": "assistant", "content": response if isinstance(response, str) else ""})
                        break
                    messages.append({"role": "assistant", "content": action})
                    result = world.request("execute", code=code)
                    if not isinstance(result.get("observation"), str) or type(result.get("task_completed")) is not bool:
                        raise AppWorldRuntimeError("Malformed AppWorld execution result.")
                    messages.append({"role": "user", "content": "Output:\n```\n" + result["observation"] + "\n```"})
                    completed = result["task_completed"]
                    if completed:
                        termination = "supervisor_complete_task"
                        break
                evaluation = world.request("evaluate")
                success = validate_evaluation(evaluation, record["task_id"])
            elapsed = time.perf_counter() - started
            if not math.isfinite(elapsed) or elapsed < 0:
                raise AppWorldRuntimeError("Invalid AppWorld wall-clock measurement.")
            output = {
                **record,
                "elapsed_seconds": elapsed,
                "error": error,
                "termination": termination,
                "steps": steps,
                "task_completed": completed,
                "task_goal_completion": success,
                "evaluation": evaluation,
            }
            outputs.append(output)
            scores.append(float(success))
            if capture_traces:
                trajectories.append(
                    {"record": dict(record), "instruction": context["instruction"], "messages": messages}
                )
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories if capture_traces else None)

    def make_reflective_dataset(
        self, candidate: dict[str, str], eval_batch: EvaluationBatch, components_to_update: list[str]
    ) -> dict[str, list[dict[str, Any]]]:
        """Use training traces and scalar state feedback without answers or evaluator tests."""
        if components_to_update != [COMPONENT] or set(candidate) != {COMPONENT}:
            raise ValueError("Only the AppWorld system prompt can be optimized.")
        if eval_batch.trajectories is None:
            raise ValueError("AppWorld reflection requires captured training trajectories.")
        examples = []
        for trace, output, score in zip(eval_batch.trajectories, eval_batch.outputs, eval_batch.scores, strict=True):
            if trace["record"]["official_split"] != "train" or output["official_split"] != "train":
                raise ValueError("Validation and test tasks must never enter AppWorld reflection.")
            if trace["record"]["id"] != output["id"]:
                raise ValueError("AppWorld reflection task identity mismatch.")
            examples.append(
                {
                    "Inputs": {"task": trace["instruction"]},
                    "Generated Outputs": trace["messages"],
                    "Feedback": {
                        "task_goal_completion": score,
                        "termination": output["termination"],
                        "error": output["error"],
                        "passed_requirements": output["evaluation"]["passed"],
                        "total_requirements": output["evaluation"]["num_tests"],
                    },
                }
            )
        return {COMPONENT: examples}

    def summarize_evaluation(self, records: list[dict[str, Any]], evaluations: list[EvaluationBatch]) -> dict[str, Any]:
        """Call official Metric aggregation separately for each held-out subset/repetition."""
        if not records or not evaluations:
            raise ValueError("Cannot summarize an empty AppWorld evaluation.")
        if any(self.records.get(record.get("id")) != record for record in records):
            raise ValueError("Unknown AppWorld summary records.")
        expected = [record["id"] for record in records]
        if len(set(expected)) != len(expected):
            raise ValueError("Duplicate AppWorld summary records.")
        repetitions = []
        for evaluation in evaluations:
            if [output.get("id") for output in evaluation.outputs] != expected or len(evaluation.scores) != len(
                expected
            ):
                raise ValueError("Incomplete or reordered AppWorld evaluation batch.")
            for record, output, score in zip(records, evaluation.outputs, evaluation.scores, strict=True):
                if float(validate_evaluation(output["evaluation"], record["task_id"])) != score:
                    raise ValueError("AppWorld batch score differs from official evaluator evidence.")
            groups = {"all": records}
            groups.update(
                {
                    split: [record for record in records if record["official_split"] == split]
                    for split in dict.fromkeys(record["official_split"] for record in records)
                }
            )
            evidence = {
                output["task_id"]: {
                    "path": output["evaluation"]["evaluation_path"],
                    "sha256": output["evaluation"]["evaluation_sha256"],
                }
                for output in evaluation.outputs
            }
            metrics = {}
            with self.world_factory() as world:
                for name, group in groups.items():
                    aggregate = world.request(
                        "aggregate", evaluations={record["task_id"]: evidence[record["task_id"]] for record in group}
                    )
                    for key in ("task_goal_completion", "scenario_goal_completion"):
                        if (
                            type(aggregate.get(key)) not in (int, float)
                            or not math.isfinite(aggregate[key])
                            or not 0 <= aggregate[key] <= 100
                        ):
                            raise AppWorldRuntimeError("Malformed official aggregate metrics.")
                    members = scenario_members(group)
                    complete = all(ids == self.scenarios[scenario] for scenario, ids in members.items())
                    metrics[name] = {
                        **aggregate,
                        "scenario_goal_completion": aggregate["scenario_goal_completion"] if complete else None,
                        "scenario_goal_completion_available": complete,
                        "task_count": len(group),
                        "scenario_count": len(members),
                    }
            repetitions.append(metrics)
        return {
            "unit": "percent",
            "metric_implementation": "appworld.evaluator.Metric.compute_metrics",
            "repetitions": repetitions,
        }
