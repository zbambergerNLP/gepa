"""Measure the initial harness on training tasks before freezing runtime settings."""

import argparse
import json
import time
from pathlib import Path

from examples.common.experiment_models import (
    EXPERIMENT_MODELS,
    EXPERIMENT_NUM_RETRIES,
    QWEN3_8_27B_MODEL,
    experiment_model_version,
    experiment_request_overrides,
)
from examples.common.provider_retries import PROVIDER_RETRY_POLICY
from examples.terminalbench.main import EXPERIMENT_MANIFESTS, REPO_ROOT, seed_candidate
from examples.terminalbench.model_settings import (
    terminalbench_decoding,
    terminalbench_limits,
    terminalbench_model_info,
)
from examples.terminalbench.pilot import (
    PILOT_PROTOCOL,
    PILOT_SCHEMA_VERSION,
    complete_pilot,
    load_completed_pilot,
    validate_runtime,
)
from examples.terminalbench.runtime import load_runtime_record
from examples.terminalbench.token_usage import TOKEN_USAGE_POLICY, summarize_usage
from gepa.adapters.terminal_bench_adapter import (
    TERMINUS_ADAPTER_CONTRACT,
    HarborCLI,
    TerminusAdapter,
    load_terminalbench_manifest,
)
from gepa.adapters.terminal_bench_adapter.documents import seed_documents
from gepa.adapters.terminal_bench_adapter.terminal_bench_adapter import TASK_CONTEXT_SETTINGS
from gepa.adapters.terminal_bench_adapter.text_scope import (
    DEFAULT_OPTIMIZATION_SCOPE,
    OPTIMIZATION_SCOPES,
    TerminalBenchTextScope,
)
from gepa.strategies.text_limits import parse_text_limits, resolve_text_limits


def main(argv: list[str] | None = None) -> None:
    """Run a separate training-only pilot and retain usage even if its job fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=EXPERIMENT_MANIFESTS, default="tb2.1")
    parser.add_argument("--optimization-scope", choices=OPTIMIZATION_SCOPES, default=DEFAULT_OPTIMIZATION_SCOPE)
    parser.add_argument("--model", choices=EXPERIMENT_MODELS, default=QWEN3_8_27B_MODEL)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--runtime-record", type=Path, help="Current local server record from the runtime launcher")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "full", "calibration"), default="smoke")
    parser.add_argument("--train-limit", type=int, default=None, help="Custom training coverage for calibration only")
    parser.add_argument("--smoke-dir", type=Path, help="Completed three-task smoke pilot required by the full stage")
    parser.add_argument("--n-concurrent", type=int, default=1, help="Maximum simultaneous training-task trials")
    parser.add_argument("--harbor-executable", default="harbor")
    parser.add_argument("--docker-executable", default="docker")
    parser.add_argument("--harbor-process-timeout-sec", type=float, default=None)
    parser.add_argument("--text-limits", type=parse_text_limits, default=None)
    args = parser.parse_args(argv)
    text_limits = resolve_text_limits(args.text_limits)
    if args.train_limit is not None and args.train_limit <= 0:
        parser.error("--train-limit must be positive")
    if args.n_concurrent <= 0:
        parser.error("--n-concurrent must be positive")
    manifest = load_terminalbench_manifest(EXPERIMENT_MANIFESTS[args.experiment])
    stage_limit = 30 if args.stage == "full" else 3
    if args.stage != "calibration" and args.train_limit not in (None, stage_limit):
        parser.error(f"The {args.stage} stage requires exactly {stage_limit} training tasks")
    train_limit = args.train_limit if args.stage == "calibration" and args.train_limit is not None else stage_limit
    if train_limit > len(manifest.splits["train"]) or args.n_concurrent > train_limit:
        parser.error("Calibration needs at least n-concurrent tasks and cannot exceed the training split")
    if args.stage != "full" and args.smoke_dir is not None:
        parser.error("--smoke-dir is only used by the full stage")
    tasks = manifest.tasks("train", train_limit)
    try:
        execution_runtime = load_runtime_record(args.runtime_record, args.model, args.api_base)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    candidate, family = seed_candidate(args.model, "auto", args.experiment, args.optimization_scope)
    scope = TerminalBenchTextScope(args.optimization_scope, family)
    limits = terminalbench_limits(args.model)
    agent_kwargs = {
        "model_info": terminalbench_model_info(args.model),
        "token_limits": limits,
        "llm_kwargs": {
            "num_retries": EXPERIMENT_NUM_RETRIES,
            **terminalbench_decoding(args.model),
            **experiment_request_overrides(args.model, explicit_reasoning=True),
        },
    }
    config = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "execution_runtime": execution_runtime,
        "pilot_protocol": PILOT_PROTOCOL,
        "stage": args.stage,
        "adapter": TERMINUS_ADAPTER_CONTRACT,
        "provider_retry_policy": PROVIDER_RETRY_POLICY,
        "experiment": args.experiment,
        "dataset": manifest.dataset,
        "optimization_scope": scope.name,
        "text_scope": scope.contract(),
        "split": "train",
        "n_concurrent": args.n_concurrent,
        "task_context_settings": dict(TASK_CONTEXT_SETTINGS),
        "task_ids": [task.task_id for task in tasks],
        "task_refs": {task.task_id: manifest.task_refs[task.task_id] for task in tasks},
        "model": args.model,
        "model_version": experiment_model_version(args.model),
        "api_base": args.api_base,
        "template_family": family,
        "candidate_digest": manifest.candidate_digest(scope.materialize(candidate)),
        "reference_seed_digest": manifest.candidate_digest(seed_documents(family)),
        "student_agent_kwargs": agent_kwargs,
        "token_usage_policy": TOKEN_USAGE_POLICY,
        "text_limits": text_limits.to_dict(),
        "harbor_process_timeout_sec": args.harbor_process_timeout_sec,
        "smoke_evidence": None,
    }
    if args.stage == "full":
        if args.smoke_dir is None:
            parser.error("The full stage requires --smoke-dir from a completed three-task smoke pilot")
        try:
            smoke = load_completed_pilot(args.smoke_dir, manifest, "smoke")
            validate_runtime(smoke["config"], config, allow_concurrency_change=True)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        config["smoke_evidence"] = smoke
    harbor = HarborCLI(
        manifest=manifest,
        student_model=args.model,
        student_api_base=args.api_base,
        work_dir=args.output_dir / "harbor",
        agent_python_path=REPO_ROOT,
        n_concurrent=args.n_concurrent,
        harbor_executable=args.harbor_executable,
        docker_executable=args.docker_executable,
        student_agent_kwargs=agent_kwargs,
        process_timeout_sec=args.harbor_process_timeout_sec,
        text_limits=text_limits,
    )
    harbor.check_requirements()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "canary-config.json").write_text(json.dumps(config, indent=2) + "\n")
    started = time.monotonic()
    try:
        batch = TerminusAdapter(manifest, harbor, text_scope=scope).evaluate(tasks, candidate)
        (args.output_dir / "task-results.json").write_text(json.dumps(batch.outputs, indent=2) + "\n")
    finally:
        report = summarize_usage([args.output_dir / "harbor"])
        (args.output_dir / "token-usage-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    complete_pilot(args.output_dir, time.monotonic() - started)
    print(f"Review token usage, cutoffs, timeouts, and throughput: {args.output_dir / 'pilot-summary.json'}")
    if args.stage == "smoke":
        print("Smoke check completed. The full 30-task training pilot is still required before a campaign.")


if __name__ == "__main__":
    main()
