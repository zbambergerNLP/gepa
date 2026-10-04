"""Expose the existing four-predictor HotPotQA program to the shared GEPA runner."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any, cast

from examples.common.artifacts import digest
from examples.common.benchmark_types import BenchmarkModels
from examples.common.wikipedia import WikipediaPassage, WikipediaRetriever
from examples.hotpotqa.benchmark_settings import RETRIEVAL_K, SEED_CANDIDATE
from examples.hotpotqa.utils import artifact_component_records, build_hotpotqa_task_lm, hotpotqa_metric, run_two_stage
from gepa.core.adapter import EvaluationBatch, ProposalFn


class HotPotQAAdapter:
    """Run both retrieval hops and all four actual DSPy prompt components."""

    propose_new_texts: ProposalFn | None = None

    def __init__(
        self,
        models: BenchmarkModels,
        retriever: WikipediaRetriever,
        *,
        training_ids: set[str],
        max_workers: int = 1,
    ) -> None:
        self.models = models
        self.retriever = retriever
        self.training_ids = frozenset(training_ids)
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self.max_workers = max_workers
        self.task_lm = build_hotpotqa_task_lm(models.solver_model, models.solver_api_base, models.solver_kwargs)

    def _evaluate_one(self, record: dict, candidate: dict[str, str]) -> tuple[dict, float]:
        started = time.perf_counter()
        output: dict[str, Any] = {"id": record["id"], "prompt_sha256": digest(candidate), "error": None}
        try:
            query, prediction, trace = run_two_stage(
                candidate["summarize1"],
                candidate["create_query_hop2"],
                candidate["summarize2"],
                candidate["final_answer"],
                record["question"],
                self.retriever,
                model=self.models.solver_model,
                api_base=self.models.solver_api_base,
                retrieval_k=RETRIEVAL_K,
                task_lm=self.task_lm,
                lm_kwargs=self.models.solver_kwargs,
            )
        except ValueError as error:
            if not error.args or not str(error.args[0]).startswith("Failed to parse response as per signature"):
                raise
            output.update(prediction=None, error="task_output_parse_error", trace=None)
            score = 0.0
        else:
            score, _ = hotpotqa_metric(prediction, record["answer"])
            serialized = {
                name: [asdict(document) for document in cast(list[WikipediaPassage], value)]
                if name in {"hop1_documents", "hop2_documents"}
                else value
                for name, value in trace.items()
            }
            output.update(prediction=prediction, query=query, trace=serialized)
            if not prediction.strip():
                output["error"] = "empty_final_answer"
                score = 0.0
        output["elapsed_seconds"] = time.perf_counter() - started
        return output, score

    def evaluate(self, batch: list[dict], candidate: dict[str, str], capture_traces: bool = False) -> EvaluationBatch:
        """Evaluate the current four prompts without passing gold context to the solver."""
        if set(candidate) != set(SEED_CANDIDATE) or any(
            not isinstance(v, str) or not v.strip() for v in candidate.values()
        ):
            raise ValueError("HotPotQA requires the four nonempty artifact prompt components")
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            evaluated = list(pool.map(lambda record: self._evaluate_one(record, candidate), batch))
        outputs = [output for output, _ in evaluated]
        scores = [score for _, score in evaluated]
        trajectories = (
            [
                {"record": record, "output": output, "score": score}
                for record, (output, score) in zip(batch, evaluated, strict=True)
            ]
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
        """Reuse artifact component feedback exclusively for the selected training rows."""
        if (
            not components_to_update
            or not set(components_to_update) <= set(SEED_CANDIDATE)
            or eval_batch.trajectories is None
        ):
            raise ValueError("HotPotQA reflection requires known components and training trajectories")
        feedback = {component: [] for component in components_to_update}
        for trajectory in eval_batch.trajectories:
            record, output = trajectory["record"], trajectory["output"]
            if record["id"] not in self.training_ids:
                raise ValueError("HotPotQA validation and test records cannot enter reflection")
            if output["prompt_sha256"] != digest(candidate):
                raise ValueError("HotPotQA reflection trace belongs to a different candidate")
            if output["error"]:
                component_records = {
                    component: {
                        "Inputs": {"question": record["question"]},
                        "Generated Outputs": {"answer": output["prediction"]},
                        "Feedback": {"error": output["error"], "exact_match": 0.0},
                    }
                    for component in components_to_update
                }
            else:
                trace: dict[str, object] = {
                    name: [WikipediaPassage(**document) for document in value]
                    if name in {"hop1_documents", "hop2_documents"}
                    else value
                    for name, value in output["trace"].items()
                }
                component_records = artifact_component_records(
                    record, trace, trajectory["score"], include_diagnostics=True
                )
            for component in components_to_update:
                feedback[component].append(component_records[component])
        return feedback
