"""Run resumable HotPotQA training calibration and real optimizer checks."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from examples.common.pilot_checks import (
    METHODS,
    OPTIMIZER_PILOT_PROTOCOL,
    CycleEvidence,
    atomic_json,
    digest,
    load_cycle,
    require_contract,
)
from examples.common.provider_retries import PROVIDER_RETRY_KEY, provider_retry_kwargs
from examples.common.react_v2 import benchmark_data_identity, resolve_template_family
from examples.common.recovery import RecoveryCallback, run_guarded, seal_progress
from examples.common.wiki17_bm25 import Wiki17BM25Retriever
from examples.hotpotqa.main import (
    _validate_scientific_data_identity,
    _verify_scientific_retriever_integrity,
    build_config,
    build_parser,
    build_run_contract,
    make_evaluator,
    run_condition,
    run_program,
    seed_candidate,
)
from examples.hotpotqa.tracking import HotpotqaWandb
from examples.hotpotqa.utils import (
    HOTPOTQA_HF_REVISION,
    build_hotpotqa_task_lm,
    f1_score,
    load_hotpotqa_dataset,
    normalize_answer,
    resolve_hotpotqa_lm_kwargs,
)
from examples.terminalbench.token_usage import summarize_usage

PILOT_PROTOCOL = {
    "version": 2,
    "split": "train",
    "smoke": 3,
    "throughput": 12,
    "full": 150,
    "optimizer": OPTIMIZER_PILOT_PROTOCOL,
}
LIMITS = {"max_output_tokens": 16384, "context_tokens": 262144}


def observed_kwargs(model: str, api_base: str, directory: Path, role: str) -> dict:
    """Apply the production decoding/retry policy and retain raw usage."""
    kwargs = resolve_hotpotqa_lm_kwargs(model, api_base, role="solver" if role == "solver" else "optimizer")
    kwargs.update(provider_retry_kwargs(directory / "provider-attempts.jsonl", role))
    retry_settings = kwargs[PROVIDER_RETRY_KEY]
    assert isinstance(retry_settings, dict)
    retry_settings.update(
        token_usage_log=str(directory / "token-usage.jsonl"),
        token_limits={**LIMITS, "max_output_tokens": kwargs["max_tokens"]},
    )
    return kwargs


def validate_calibration(directory: Path, count: int) -> dict:
    """Verify complete, unchanged calibration evidence before the next stage."""
    marker = json.loads((directory / "pilot-complete.json").read_text())
    summary = json.loads((directory / "pilot-summary.json").read_text())
    contract = json.loads((directory / "pilot-contract.json").read_text())
    usage = json.loads((directory / "token-usage-summary.json").read_text())
    records = {path.name: digest(json.loads(path.read_text())) for path in (directory / "records").glob("*.json")}
    if (
        marker.get("summary_sha256") != digest(summary)
        or marker.get("contract_sha256") != digest(contract)
        or summary.get("usage_sha256") != digest(usage)
        or summary.get("protocol") != PILOT_PROTOCOL
        or summary.get("record_hashes") != records
        or summary.get("completed_questions") != count
        or len(records) != count
        or not summary.get("qualified")
    ):
        raise ValueError(f"Incomplete or changed calibration evidence: {directory}")
    return summary


def strict_evaluator(evaluate):
    """Retain wrong answers but surface the task parser's scored-error case."""

    def checked(candidate, example):
        """Execute the production evaluator and require usable task output."""
        score, feedback = evaluate(candidate, example)
        if feedback.get("evaluation_error"):
            raise RuntimeError("Optimizer pilot encountered malformed task output")
        return score, feedback

    return checked


def run_calibration(
    directory: Path, examples: list[dict], candidate: dict, execute, *, contract: dict, workers: int
) -> dict:
    """Evaluate initial prompts and resume only verified completed questions."""
    require_contract(directory, {**contract, "examples": examples, "candidate": candidate, "workers": workers})
    if (directory / "pilot-complete.json").exists():
        return validate_calibration(directory, len(examples))
    records_dir = directory / "records"
    records_dir.mkdir(exist_ok=True)
    lock = threading.Lock()
    started = time.time()
    window_path = directory / "allocations" / f"{os.environ.get('SLURM_JOB_ID', 'local')}-{time.time_ns()}.json"
    window = {"job_id": os.environ.get("SLURM_JOB_ID"), "started_at": started, "ended_at": None}
    atomic_json(window_path, window)

    def score_one(item: tuple[int, dict]) -> dict:
        """Persist a usable prediction without requiring a correct answer."""
        index, example = item
        path = records_dir / f"{index:04d}.json"
        identity = digest({"example": example, "candidate": candidate})
        if path.exists():
            record = json.loads(path.read_text())
            if (
                record.get("identity") != identity
                or not isinstance(record.get("prediction"), str)
                or not record["prediction"].strip()
                or record.get("id") != example["id"]
                or record.get("exact_match")
                != float(normalize_answer(record["prediction"]) == normalize_answer(example["answer"]))
                or record.get("f1") != f1_score(record["prediction"], example["answer"])
            ):
                raise ValueError(f"Invalid pilot recovery record: {path}")
            return record
        call_started = time.time()
        prediction, trace = execute(candidate, example)
        if not isinstance(prediction, str) or not prediction.strip():
            raise RuntimeError(f"Pilot question {example['id']} returned no usable prediction")
        record = {
            "identity": identity,
            "id": example["id"],
            "prediction": prediction,
            "trace": trace,
            "exact_match": float(normalize_answer(prediction) == normalize_answer(example["answer"])),
            "f1": f1_score(prediction, example["answer"]),
            "started_at": call_started,
            "ended_at": time.time(),
            "job_id": os.environ.get("SLURM_JOB_ID"),
        }
        with lock:
            atomic_json(path, record)
            artifacts = sorted(records_dir.glob("*.json"))
            seal_progress(directory, len(artifacts), artifacts)
        return record

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(score_one, enumerate(examples)))
    finally:
        window["ended_at"] = time.time()
        atomic_json(window_path, window)
        atomic_json(directory / "token-usage-summary.json", summarize_usage([directory]))
    usage = json.loads((directory / "token-usage-summary.json").read_text())
    cutoffs = sum(
        role.get("length_finish", 0) + role.get("output_cap_reached", 0)
        for roles in usage["models"].values()
        for role in roles.values()
    )
    windows = [json.loads(path.read_text()) for path in (directory / "allocations").glob("*.json")]
    elapsed = sum(row["ended_at"] - row["started_at"] for row in windows if row["ended_at"] is not None)
    summary = {
        "protocol": PILOT_PROTOCOL,
        "stage": contract["stage"],
        "completed_questions": len(records),
        "exact_match": sum(row["exact_match"] for row in records) / len(records),
        "f1": sum(row["f1"] for row in records) / len(records),
        "elapsed_seconds": elapsed,
        "questions_per_hour": len(records) * 3600 / max(elapsed, 1e-9),
        "unfinished_allocation_windows": sum(row["ended_at"] is None for row in windows),
        "cutoff_review_required": bool(cutoffs),
        "qualified": not cutoffs,
        "record_hashes": {path.name: digest(json.loads(path.read_text())) for path in records_dir.glob("*.json")},
        "usage_sha256": digest(usage),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(directory / "pilot-summary.json", summary)
    if cutoffs:
        raise RuntimeError("Training pilot completed with output cutoffs; review them before continuing")
    atomic_json(
        directory / "pilot-complete.json",
        {
            "summary_sha256": digest(summary),
            "contract_sha256": digest(json.loads((directory / "pilot-contract.json").read_text())),
        },
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    """Run only the approved training stages on the verified Della runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--reflection-model", help="Teacher model; defaults to the task model")
    parser.add_argument("--reflection-api-base", help="Teacher endpoint; defaults to the task endpoint")
    parser.add_argument("--throughput-questions", type=int, default=12)
    parser.add_argument("--wiki17-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("all", "preliminary", "smoke", "throughput", "full", "optimizer"), default="all"
    )
    parser.add_argument(
        "--method", choices=METHODS, help="Run one optimizer check before resuming the remaining methods"
    )
    parser.add_argument("--text-limits", default="null")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--editor-mode", choices=("react", "single_call"), default="react")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not 12 <= args.throughput_questions <= 150:
        parser.error("--throughput-questions must be between 12 and 150 training questions")
    reflection_model = args.reflection_model or args.model
    reflection_api_base = args.reflection_api_base or args.api_base
    if args.method and args.stage != "optimizer":
        parser.error("--method is only valid with --stage optimizer")
    settings = build_parser().parse_args(
        [
            "--solver-model",
            args.model,
            "--reflection-model",
            reflection_model,
            "--solver-api-base",
            args.api_base,
            "--reflection-api-base",
            reflection_api_base,
            "--wiki17-dir",
            str(args.wiki17_dir),
            "--max-workers",
            str(args.workers),
            "--condition",
            "vanilla",
            "--enforce-scientific-contract",
            "--text-limits",
            args.text_limits,
            "--editor-mode",
            args.editor_mode,
        ]
    )
    train, validation, test = load_hotpotqa_dataset(seed=0)
    settings.data_identity = benchmark_data_identity(
        source={
            "type": "huggingface",
            "dataset": "hotpot_qa",
            "config": "fullwiki",
            "revision": HOTPOTQA_HF_REVISION,
            "source_split": "train",
            "split_policy": "ordered-40-40-20-then-independent-seed1-sampling",
            "experiment_seed": 0,
        },
        trainset=train,
        valset=validation,
        testset=test,
    )
    _validate_scientific_data_identity(settings)
    del validation, test
    retriever = Wiki17BM25Retriever(args.wiki17_dir)
    _verify_scientific_retriever_integrity(retriever)
    settings.retrieval_provenance = retriever.provenance()
    runtime_contract = build_run_contract("vanilla", settings)
    require_contract(args.output_dir, {"protocol": PILOT_PROTOCOL, "runtime": runtime_contract})
    family = resolve_template_family("auto", args.model)
    candidate = seed_candidate("2stage", "structured", family)

    def calibrate(stage: str, count: int) -> None:
        """Measure the unchanged seed on the same ordered training examples."""
        directory = args.output_dir / stage
        kwargs = observed_kwargs(args.model, args.api_base, directory, "solver")
        lm = build_hotpotqa_task_lm(args.model, args.api_base, kwargs)

        def execute(seed: dict, example: dict, task_lm=lm, call_kwargs=kwargs) -> tuple[str, dict]:
            """Use the unchanged four-call production task program."""
            _, answer, trace = run_program(
                seed, example["question"], "2stage", args.model, args.api_base, retriever, 7, task_lm, call_kwargs
            )
            return answer, trace

        run_calibration(
            directory,
            train[:count],
            candidate,
            execute,
            contract={"runtime": runtime_contract, "stage": stage},
            workers=args.workers,
        )

    if args.stage in ("all", "preliminary", "smoke", "throughput"):
        calibrate("smoke", 3)
    if args.stage in ("all", "preliminary", "optimizer"):
        validate_calibration(args.output_dir / "smoke", 3)
        for method in (args.method,) if args.method else METHODS:
            directory = args.output_dir / "optimizer" / method
            contract = {
                "protocol": OPTIMIZER_PILOT_PROTOCOL,
                "runtime": build_run_contract(method, settings),
                "examples": train[:3],
                "candidate": candidate,
            }
            contract["runtime"]["optimizer"].update(max_metric_calls=None, max_candidate_proposals=None)
            require_contract(directory, contract)
            if (directory / "optimizer-pilot-complete.json").exists():
                load_cycle(directory)
                continue
            evidence = CycleEvidence(directory)
            callbacks = [RecoveryCallback(directory), evidence]
            if args.wandb_project:
                callbacks.append(HotpotqaWandb(directory, contract["runtime"], args.wandb_project, args.wandb_entity, kind="qualification"))
            kwargs = observed_kwargs(args.model, args.api_base, directory, "solver")
            evaluator = strict_evaluator(make_evaluator(args.model, retriever, args.api_base, solver_lm_kwargs=kwargs))
            config, _ = build_config(
                method,
                settings,
                observed_kwargs(reflection_model, reflection_api_base, directory, "optimizer"),
                str(directory),
            )
            config.engine.max_metric_calls = None
            config.engine.max_candidate_proposals = None
            config.stop_callbacks = evidence.completed_cycle
            try:
                run_condition(
                    method,
                    candidate,
                    train[:3],
                    train[:3],
                    config,
                    evaluator,
                    callbacks=callbacks,
                )
                evidence.verify()
            finally:
                atomic_json(
                    directory / "token-usage-summary.json", summarize_usage([directory / "provider-attempts.jsonl"])
                )
    if args.stage in ("all", "preliminary", "throughput"):
        validate_calibration(args.output_dir / "smoke", 3)
        calibrate("throughput", args.throughput_questions)
    if args.stage in ("all", "full"):
        validate_calibration(args.output_dir / "smoke", 3)
        for method in METHODS:
            load_cycle(args.output_dir / "optimizer" / method)
        calibrate("full", 150)
    print(f"Pilot evidence ready for review: {args.output_dir}")


if __name__ == "__main__":
    run_guarded(main)
