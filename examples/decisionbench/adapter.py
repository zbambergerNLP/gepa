"""Execute the official structured-decision contract with an editable system prompt."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import litellm
from decision_bench.evaluate import ECE_BINS, expected_calibration_error
from decision_bench.prompt import render_user_prompt, response_format
from decision_bench.schemas import DecisionExample, DecisionPrediction
from decision_bench.scoring import negative_log_likelihood, score_prediction
from litellm.exceptions import ContextWindowExceededError

from examples.decisionbench.benchmark_settings import COMPONENT, PROBABILITY_SOURCE
from examples.decisionbench.utils import digest
from gepa.core.adapter import EvaluationBatch, ProposalFn

if TYPE_CHECKING:
    from examples.common.benchmark_types import BenchmarkModels


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read either an SDK response object or its serialized mapping."""
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys instead of letting the parser discard a probability vector."""
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON keys in prediction")
    return result


def parse_prediction(response: Any, candidate_count: int) -> DecisionPrediction:
    """Reject unfinished or malformed output and apply the official normalization."""
    choices = _field(response, "choices", [])
    if len(choices) != 1 or _field(choices[0], "finish_reason") != "stop":
        raise ValueError("A decision requires one complete stop-finished response")
    message = _field(choices[0], "message")
    content = _field(message, "content")
    if _field(message, "refusal") or _field(message, "tool_calls") or not isinstance(content, str):
        raise ValueError("A decision requires a JSON probability vector")
    payload = json.loads(content, object_pairs_hook=_strict_object)
    # Enforce the upstream JSON schema's numeric type before Pydantic coercion.
    if not isinstance(payload, dict) or set(payload) != {"probabilities"}:
        raise ValueError("Prediction must contain only probabilities")
    values = payload["probabilities"]
    if not isinstance(values, list) or any(type(value) not in (int, float) for value in values):
        raise ValueError("Probabilities must be JSON numbers")
    prediction = DecisionPrediction.model_validate(payload)
    if len(prediction.probabilities) != candidate_count:
        raise ValueError("Prediction length does not match candidate count")
    return prediction


class DecisionBenchAdapter:
    """Score generated distributions with Hanno Labs' pinned official evaluator."""

    propose_new_texts: ProposalFn | None = None

    def __init__(
        self,
        models: BenchmarkModels,
        *,
        training_ids: set[str],
        max_workers: int = 1,
        completion: Callable[..., Any] | None = None,
    ) -> None:
        self.models = models
        self.training_ids = frozenset(training_ids)
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self.max_workers = max_workers
        self.completion = completion

    def _evaluate_one(self, record: dict, candidate: dict[str, str]) -> dict[str, Any]:
        started = time.perf_counter()
        example = DecisionExample.model_validate(record["example"])
        request = {
            **deepcopy(self.models.solver_kwargs),
            "model": self.models.solver_model,
            "messages": [
                {"role": "system", "content": candidate[COMPONENT]},
                {"role": "user", "content": render_user_prompt(example)},
            ],
            "response_format": response_format(len(example.candidates)),
            "drop_params": False,
        }
        if self.models.solver_api_base is not None:
            request["api_base"] = self.models.solver_api_base
        result: dict[str, Any] = {
            "id": record["id"],
            "row_id": example.row_id,
            "task_id": record["task_id"],
            "task_name": example.task_name,
            "family": example.family,
            "domain": example.domain,
            "primitive": example.primitive.value,
            "candidate_count": len(example.candidates),
            "prompt_sha256": digest(candidate),
            "probability_source": PROBABILITY_SOURCE,
            "request": {key: request[key] for key in ("model", "messages", "response_format")},
            "error": None,
        }
        try:
            response = (self.completion or litellm.completion)(**request)
        except ContextWindowExceededError:
            result.update(status="error", error="context_limit_exceeded_no_truncation")
        else:
            result["response"] = {
                "id": _field(response, "id"),
                "model": _field(response, "model"),
                "choices": [
                    {
                        "finish_reason": _field(choice, "finish_reason"),
                        "content": _field(_field(choice, "message"), "content"),
                    }
                    for choice in _field(response, "choices", [])
                ],
            }
            try:
                prediction = parse_prediction(response, len(example.candidates))
            except (ValueError, TypeError) as error:
                result.update(status="error", error=f"invalid_prediction: {error}")
            else:
                result.update(
                    status="ok",
                    scored=score_prediction(example, prediction).model_dump(mode="json"),
                    negative_log_likelihood=negative_log_likelihood(example, prediction),
                )
        result["elapsed_seconds"] = time.perf_counter() - started
        result["latency_seconds"] = result["elapsed_seconds"]
        return result

    def evaluate(self, batch: list[dict], candidate: dict[str, str], capture_traces: bool = False) -> EvaluationBatch:
        """Run every row with the current editable prompt, preserving batch order."""
        if (
            set(candidate) != {COMPONENT}
            or not isinstance(candidate[COMPONENT], str)
            or not candidate[COMPONENT].strip()
        ):
            raise ValueError("DecisionBench requires exactly one nonempty system_prompt component")
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            outputs = list(pool.map(lambda record: self._evaluate_one(record, candidate), batch))
        scores = [float(output.get("scored", {}).get("correct", False)) for output in outputs]
        trajectories = (
            [{"record": record, "output": output} for record, output in zip(batch, outputs, strict=True)]
            if capture_traces
            else None
        )
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Supply training-only decision feedback; held-out targets cannot enter reflection."""
        if components_to_update != [COMPONENT] or eval_batch.trajectories is None:
            raise ValueError("Reflection requires system_prompt training traces")
        feedback = []
        for trajectory in eval_batch.trajectories:
            record, output = trajectory["record"], trajectory["output"]
            if record["id"] not in self.training_ids or record["split"] != "train":
                raise ValueError("Validation or test rows cannot be used for DecisionBench reflection")
            if output["prompt_sha256"] != digest(candidate):
                raise ValueError("Reflection trace belongs to a different prompt")
            example = DecisionExample.model_validate(record["example"])
            feedback.append(
                {
                    "Inputs": render_user_prompt(example),
                    "Generated Outputs": output.get("scored", output.get("response")),
                    "Feedback": {
                        "correct": output.get("scored", {}).get("correct", False),
                        "error": output["error"],
                        "gold_candidate_index": next(
                            index
                            for index, item in enumerate(example.candidates)
                            if item.id == example.gold_candidate_id
                        ),
                        "gold_probabilities": example.gold_probabilities,
                        "negative_log_likelihood": output.get("negative_log_likelihood"),
                    },
                }
            )
        return {COMPONENT: feedback}

    def summarize_evaluation(self, records: list[dict], evaluations: list[EvaluationBatch]) -> dict[str, Any]:
        """Report all-row accuracy and official successful-row probability metrics."""
        if len(evaluations) != 1 or not records:
            raise ValueError("DecisionBench uses exactly one attempt per row")
        evaluation = evaluations[0]
        if len(evaluation.outputs) != len(records) or len(evaluation.scores) != len(records):
            raise ValueError("Incomplete DecisionBench evaluation")
        groups: dict[str, list[dict]] = defaultdict(list)
        for record, output, score in zip(records, evaluation.outputs, evaluation.scores, strict=True):
            if output["row_id"] != record["id"]:
                raise ValueError("DecisionBench evaluation row order changed")
            elapsed = output["elapsed_seconds"]
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("Missing valid episode timing")
            if output.get("status") == "ok":
                example = DecisionExample.model_validate(record["example"])
                values = output["scored"]["probabilities"]
                prediction = DecisionPrediction(probabilities=values)
                if not math.isclose(sum(values), 1.0, abs_tol=1e-12):
                    raise ValueError("Saved probabilities are not normalized")
                # Re-normalizing an already normalized vector can alter its last float bit.
                prediction.probabilities = values
                scored = score_prediction(example, prediction).model_dump(mode="json")
                if output["error"] is not None or output["scored"] != scored:
                    raise ValueError("Saved score differs from the official evaluator")
                if not math.isclose(output["negative_log_likelihood"], negative_log_likelihood(example, prediction)):
                    raise ValueError("Saved probability metric changed")
                expected = float(scored["correct"])
            else:
                if output.get("status") != "error" or not output.get("error") or "scored" in output:
                    raise ValueError("Incomplete DecisionBench attempt")
                expected = 0.0
            if score != expected:
                raise ValueError("Saved scalar score differs from the official evaluator")
            groups["overall"].append(output)
            for dimension in ("task_id", "family", "domain", "primitive", "candidate_count"):
                groups[f"{dimension}:{record[dimension]}"].append(output)
        summaries = {}
        for name, outputs in groups.items():
            successful = [output for output in outputs if output["status"] == "ok"]
            correct = sum(output["scored"]["correct"] for output in successful)
            summaries[name] = {
                "rows": len(outputs),
                "successful_rows": len(successful),
                "error_rows": len(outputs) - len(successful),
                "accuracy": correct / len(outputs),
                "coverage": len(successful) / len(outputs),
                "supported_row_accuracy": correct / len(successful) if successful else None,
                "mean_negative_log_likelihood": (
                    sum(output["negative_log_likelihood"] for output in successful) / len(successful)
                    if successful
                    else None
                ),
                "expected_calibration_error": expected_calibration_error(successful) if successful else None,
                "ece_bins": ECE_BINS,
            }
        return {"probability_source": PROBABILITY_SOURCE, "metrics": summaries}
