"""Score actual tau conversations with the official end-state evaluator."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from examples.taubench.benchmark_settings import TEST_REPETITIONS, TRIAL_SEED
from examples.taubench.utils import digest
from gepa.core.adapter import EvaluationBatch, ProposalFn

NORMAL_ENDINGS = {"agent_stop", "user_stop"}
FAILED_ENDINGS = {"max_steps", "timeout", "too_many_errors", "agent_error", "user_error", "context_window_exceeded"}


def trial_seed(trial: int) -> int:
    """Use upstream's Random(300) trial-seed schedule for paired repetitions."""
    rng = random.Random(TRIAL_SEED)
    return [rng.randint(0, 1000000) for _ in range(TEST_REPETITIONS)][trial]


def validate_outputs(records: list[dict], outputs: Any, candidate_sha256: str, trial: int) -> None:
    """Fail closed on partial worker results, invalid rewards, and task substitution."""
    if not isinstance(outputs, list) or len(records) != len(outputs):
        raise ValueError("Incomplete tau batch")
    for record, output in zip(records, outputs, strict=True):
        if not isinstance(output, dict) or (
            output.get("id") != record["id"]
            or output.get("task_id") != record["task_id"]
            or output.get("trial") != trial
            or output.get("seed") != trial_seed(trial)
            or output.get("candidate_sha256") != candidate_sha256
        ):
            raise ValueError("tau output identity or repetition mismatch")
        elapsed, reward = output.get("elapsed_seconds"), output.get("reward")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, int | float)
            or not math.isfinite(elapsed)
            or elapsed <= 0
        ):
            raise ValueError("Invalid episode wall time")
        if (
            isinstance(reward, bool)
            or not isinstance(reward, int | float)
            or not math.isfinite(reward)
            or reward not in (0, 1)
        ):
            raise ValueError("Invalid official reward")
        reason = output.get("termination_reason")
        if reason not in NORMAL_ENDINGS | FAILED_ENDINGS:
            raise ValueError("Missing or invalid termination reason")
        if reason in FAILED_ENDINGS and (reward != 0 or output.get("error") != reason):
            raise ValueError("Incomplete tau attempt must be an explicit failure")
        if reason in NORMAL_ENDINGS and output.get("error") is not None:
            raise ValueError("Successful execution cannot conceal an error")
        if not isinstance(output.get("messages"), list) or not output["messages"]:
            raise ValueError("Missing actual conversation trace")


class TauBankingAdapter:
    """Expose one agent system prompt while the simulator and grading stay fixed."""

    propose_new_texts: ProposalFn | None = None

    def __init__(self, runtime, manifest: dict, max_workers: int = 1):
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self.runtime = runtime
        self.manifest = manifest
        self.manifest_sha256 = digest(manifest)
        self.records = {record["id"]: record for record in manifest["records"]}
        self.max_workers = max_workers
        self._split: str | None = None
        self._trial = 0

    def set_evaluation_context(self, *, split: str, repetition: int, seed: int) -> None:
        """Take an explicit repetition index so partial resume cannot repeat trial zero."""
        if split not in {"train", "val", "test"} or type(repetition) is not int:
            raise ValueError("Invalid evaluation context")
        if not 0 <= repetition < (TEST_REPETITIONS if split == "test" else 1):
            raise ValueError("Invalid repetition index")
        self._split, self._trial = split, repetition

    def evaluate(self, batch: list[dict], candidate: dict[str, str], capture_traces: bool = False) -> EvaluationBatch:
        """Return official binary rewards; never infer success from dialogue or actions."""
        if (
            set(candidate) != {"system_prompt"}
            or not isinstance(candidate["system_prompt"], str)
            or not candidate["system_prompt"].strip()
        ):
            raise ValueError("Exactly one nonempty system_prompt is editable")
        if not batch:
            return EvaluationBatch(outputs=[], scores=[], trajectories=[] if capture_traces else None)
        if any(self.records.get(record.get("id")) != record for record in batch):
            raise ValueError("Unknown or changed tau record")
        if len({record["split"] for record in batch}) != 1:
            raise ValueError("Cannot evaluate mixed data splits")
        if batch[0]["split"] != "train" and len({record["id"] for record in batch}) != len(batch):
            raise ValueError("Duplicate validation or test tasks")
        if capture_traces and batch[0]["split"] == "test":
            raise ValueError("Held-out traces cannot be requested for optimization")
        if self._split is not None and self._split != batch[0]["split"]:
            raise ValueError("Evaluation context and record split disagree")
        trial = self._trial if batch[0]["split"] == "test" else 0
        chunks = [batch[i :: self.max_workers] for i in range(min(self.max_workers, len(batch)))]
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(lambda rows: self.runtime.run(rows, candidate, trial), chunks))
        # GEPA pads training minibatches with repeated IDs; each position owns a rollout.
        outputs: list[dict] = [{} for _ in batch]
        for chunk_index, (records, result) in enumerate(zip(chunks, results, strict=True)):
            if result.get("manifest_sha256") != self.manifest_sha256:
                raise ValueError("Worker ran another tau source or data revision")
            validate_outputs(records, result.get("outputs"), digest(candidate), trial)
            outputs[chunk_index :: self.max_workers] = result["outputs"]
        trajectories = (
            [
                {
                    "id": row["id"],
                    "messages": row["messages"],
                    "reward": row["reward"],
                    "termination_reason": row["termination_reason"],
                }
                for row in outputs
            ]
            if capture_traces
            else None
        )
        return EvaluationBatch(outputs=outputs, scores=[row["reward"] for row in outputs], trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Give the proposer actual training dialogue and reward, with no reference actions."""
        if components_to_update != ["system_prompt"] or not eval_batch.trajectories:
            raise ValueError("Reflection requires the single system prompt and training traces")
        rows = []
        for trace in eval_batch.trajectories:
            if self.records.get(trace["id"], {}).get("split") != "train":
                raise ValueError("Held-out or validation traces cannot enter reflection")
            rows.append(
                {
                    "Inputs": {"domain": "banking_knowledge", "task_id": trace["id"]},
                    "Generated Outputs": trace["messages"],
                    "Feedback": {"official_reward": trace["reward"], "termination_reason": trace["termination_reason"]},
                }
            )
        return {"system_prompt": rows}

    def summarize_evaluation(self, records: list[dict], evaluations: list[EvaluationBatch]) -> dict:
        """Compute tau pass^k = C(successes,k)/C(trials,k), never pass-at-k."""
        n = len(evaluations)
        expected = TEST_REPETITIONS if records and records[0]["split"] == "test" else 1
        if not records or n != expected:
            raise ValueError(f"Expected {expected} complete tau repetitions")
        candidate_sha256 = evaluations[0].outputs[0].get("candidate_sha256") if evaluations[0].outputs else None
        if not isinstance(candidate_sha256, str) or len(candidate_sha256) != 64:
            raise ValueError("Missing frozen candidate identity")
        success_counts = [0] * len(records)
        for trial, evaluation in enumerate(evaluations):
            if len(evaluation.outputs) != len(records) or len(evaluation.scores) != len(records):
                raise ValueError("Incomplete repetition matrix")
            validate_outputs(records, evaluation.outputs, candidate_sha256, trial)
            for index, (record, output, score) in enumerate(
                zip(records, evaluation.outputs, evaluation.scores, strict=True)
            ):
                if (
                    output.get("id") != record["id"]
                    or output.get("trial") != trial
                    or output.get("seed") != trial_seed(trial)
                ):
                    raise ValueError("Repetition IDs or seeds changed")
                if type(score) not in (int, float) or score not in (0, 1) or score != output.get("reward"):
                    raise ValueError("Invalid score in repetition matrix")
                success_counts[index] += score == 1
        return {
            "pass_hat_k": {
                str(k): sum(math.comb(s, k) / math.comb(n, k) for s in success_counts) / len(records)
                for k in range(1, n + 1)
            },
            "pass_hat_1": sum(success_counts) / (n * len(records)),
            "scored_failures": n * len(records) - sum(success_counts),
            "num_trials": n,
            "num_tasks": len(records),
        }
