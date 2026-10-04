"""Rewrite once with the actual candidate prompt and score ranked documents with trec_eval."""

from __future__ import annotations

import importlib
import json
import math
import time
from collections import defaultdict
from typing import Any

from examples.obliqbench.benchmark_settings import METRIC_NAME, RETRIEVAL_K, TASK_DESCRIPTIONS
from examples.obliqbench.retrieval import Retriever, require_version
from gepa.core.adapter import EvaluationBatch

COMPONENT = "query_rewriter_system_prompt"


def score_ranking(record: dict[str, Any], ranking: list[tuple[str, float]]) -> dict[str, float]:
    """Compute released gold/pooled NDCG and recall using pinned trec_eval semantics."""
    require_version("pytrec-eval-terrier")
    pytrec_eval = importlib.import_module("pytrec_eval")
    results = {doc: float(len(ranking) - index) for index, (doc, _) in enumerate(ranking)}
    metrics = {}
    for label, qrels in (("gold", record["gold_qrels"]), ("pooled", record["pooled_qrels"])):
        if qrels is None:
            continue
        evaluator = pytrec_eval.RelevanceEvaluator({record["id"]: qrels}, {"ndcg_cut.10,50", "recall.10,50,100"})
        values = evaluator.evaluate({record["id"]: results})[record["id"]]
        for name, value in values.items():
            metrics[f"{label}_{name.replace('ndcg_cut_', 'ndcg_at_').replace('recall_', 'recall_at_')}"] = value
    return metrics


def parse_rewrite(content: str) -> str:
    """Reject malformed responses rather than retrieving a partial or inferred rewrite."""

    def unique_object(pairs):
        if len(dict(pairs)) != len(pairs):
            raise ValueError("Duplicate JSON query field")
        return dict(pairs)

    value = json.loads(content, object_pairs_hook=unique_object)
    if (
        not isinstance(value, dict)
        or set(value) != {"query"}
        or not isinstance(value["query"], str)
        or not value["query"].strip()
    ):
        raise ValueError("Expected exactly one nonempty JSON query field")
    return value["query"].strip()


class ObliqAdapter:
    """Execute a single optimized system prompt followed by a fixed retriever."""

    propose_new_texts = None

    def __init__(self, solver: Any, retrievers: dict[str, Retriever], *, original_query: bool = False):
        self.solver = solver
        self.retrievers = retrievers
        self.original_query = original_query
        self.known_ids = {name: frozenset(retriever.ids) for name, retriever in retrievers.items()}

    def evaluate(
        self, batch: list[dict[str, Any]], candidate: dict[str, str], capture_traces: bool = False
    ) -> EvaluationBatch:
        """Time each complete rewrite/search/metric episode, returning zero on malformed task output."""
        if (
            set(candidate) != {COMPONENT}
            or not isinstance(candidate[COMPONENT], str)
            or not candidate[COMPONENT].strip()
        ):
            raise ValueError("OBLIQ optimizes exactly one nonempty query-rewriter system prompt")
        outputs, traces, scores = [], [], []
        for record in batch:
            retriever = self.retrievers[record["subset"]]
            started = time.perf_counter()
            output = {"id": record["id"], "subset": record["subset"], "error": None}
            try:
                if self.original_query:
                    query = record["query"]
                else:
                    messages = [
                        {"role": "system", "content": candidate[COMPONENT]},
                        {
                            "role": "user",
                            "content": json.dumps(
                                {"task": TASK_DESCRIPTIONS[record["subset"]], "query": record["query"]},
                                ensure_ascii=False,
                            ),
                        },
                    ]
                    output["raw_response"] = self.solver(messages)
                    query = parse_rewrite(output["raw_response"])
                output["rewritten_query"] = query
                excluded = set(record["excluded_ids"])
                ranking = retriever.search(query, excluded, RETRIEVAL_K)
                expected = min(RETRIEVAL_K, len(retriever.ids) - len(excluded))
                ids = [doc for doc, _ in ranking]
                if (
                    len(ranking) != expected
                    or len(set(ids)) != expected
                    or not set(ids) <= self.known_ids[record["subset"]]
                    or set(ids) & excluded
                ):
                    raise ValueError("Incomplete, duplicate, unknown, or excluded retrieval results")
                if any(not math.isfinite(score) for _, score in ranking):
                    raise ValueError("Non-finite retrieval scores")
                output["ranking"] = ids
                output["retrieval_scores"] = [score for _, score in ranking]
                output["metrics"] = score_ranking(record, ranking)
                score = output["metrics"][METRIC_NAME]
            except (ValueError, TypeError) as exc:
                output["error"] = f"{type(exc).__name__}: {exc}"
                score = 0.0
            output["elapsed_seconds"] = time.perf_counter() - started
            outputs.append(output)
            scores.append(score)
            traces.append(
                {
                    "id": record["id"],
                    "query": record["query"],
                    "subset": record["subset"],
                    "split": record["split"],
                    "output": output,
                }
            )
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=traces if capture_traces else None)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        """Expose training/validation execution feedback without gold documents or judgments."""
        if any(component != COMPONENT for component in components_to_update) or eval_batch.trajectories is None:
            raise ValueError("OBLIQ reflection requires captured query-rewriter traces")
        if len(eval_batch.trajectories) != len(eval_batch.scores):
            raise ValueError("Incomplete reflection batch")
        examples = []
        for trace, score in zip(eval_batch.trajectories, eval_batch.scores, strict=True):
            if trace["split"] == "test":
                raise ValueError("Held-out OBLIQ test traces cannot enter reflection")
            output = trace["output"]
            examples.append(
                {
                    "Inputs": {"task": TASK_DESCRIPTIONS[trace["subset"]], "query": trace["query"]},
                    "Generated Outputs": output.get("rewritten_query", output.get("raw_response", "")),
                    "Feedback": {METRIC_NAME: score, "metrics": output.get("metrics", {}), "error": output["error"]},
                }
            )
        return dict.fromkeys(components_to_update, examples)

    def summarize_evaluation(self, records, evaluations):
        """Report query means and per-subset official metrics only for complete runs."""
        expected = [record["id"] for record in records]
        if not evaluations or not expected or len(set(expected)) != len(expected):
            raise ValueError("Empty or duplicate OBLIQ evaluation records")
        collected = defaultdict(lambda: defaultdict(list))
        for evaluation in evaluations:
            if [row["id"] for row in evaluation.outputs] != expected or len(evaluation.scores) != len(records):
                raise ValueError("Incomplete or reordered OBLIQ evaluation")
            for record, output, score in zip(records, evaluation.outputs, evaluation.scores, strict=True):
                if output["error"] is not None or not output.get("metrics"):
                    raise ValueError("Failed OBLIQ episodes cannot be reported as a completed benchmark")
                labels = ("gold", "pooled") if record["pooled_qrels"] is not None else ("gold",)
                expected_metrics = {
                    f"{label}_{metric}_at_{k}"
                    for label in labels
                    for metric, cutoffs in (("ndcg", (10, 50)), ("recall", (10, 50, 100)))
                    for k in cutoffs
                }
                if set(output["metrics"]) != expected_metrics or score != output["metrics"][METRIC_NAME]:
                    raise ValueError("Missing or inconsistent OBLIQ metrics")
                for name, value in output["metrics"].items():
                    if not isinstance(value, int | float) or not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError("Invalid OBLIQ metric value")
                    collected[record["subset"]][name].append(value)
        subsets = {
            subset: {metric: sum(values) / len(values) for metric, values in metrics.items()}
            for subset, metrics in collected.items()
        }
        gold = [output["metrics"][METRIC_NAME] for evaluation in evaluations for output in evaluation.outputs]
        return {
            "per_subset": subsets,
            "query_mean_gold_ndcg_at_10": sum(gold) / len(gold),
            "subset_macro_gold_ndcg_at_10": sum(metrics[METRIC_NAME] for metrics in subsets.values()) / len(subsets),
            "query_count": len(records),
            "repetitions": len(evaluations),
        }
