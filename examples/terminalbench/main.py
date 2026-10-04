"""Run Terminal-Bench 2.1 through the shared GEPA/FOREST benchmark lifecycle."""

from __future__ import annotations

import argparse
import hashlib
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from examples.common.benchmark_runner import run_cli
from examples.common.benchmark_types import BenchmarkDefinition, BenchmarkModels
from examples.common.react_v2 import resolve_template_family
from examples.terminalbench.benchmark_settings import MANIFEST_PATH, TEST_REPETITIONS, TRAINING_EPOCHS_BY_BUDGET
from examples.terminalbench.model_settings import terminalbench_decoding, terminalbench_limits, terminalbench_model_info
from examples.terminalbench.runtime import load_role_runtimes
from examples.terminalbench.shared_adapter import SharedTerminusAdapter
from examples.terminalbench.token_usage import TOKEN_USAGE_POLICY
from gepa.adapters.terminal_bench_adapter import (
    TERMINUS_ADAPTER_CONTRACT,
    HarborCLI,
    TerminusAdapter,
    load_terminalbench_manifest,
)
from gepa.adapters.terminal_bench_adapter.documents import BUNDLE_VERSION
from gepa.adapters.terminal_bench_adapter.terminal_bench_adapter import FAILURE_POLICY_CONTRACT, TASK_CONTEXT_SETTINGS
from gepa.adapters.terminal_bench_adapter.text_scope import TerminalBenchTextScope

REPO_ROOT = Path(__file__).resolve().parents[2]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Retain Terminal-Bench's epoch budget and required serving/runtime preflights."""
    parser.set_defaults(max_metric_calls=None)
    parser.add_argument("--budget", choices=tuple(TRAINING_EPOCHS_BY_BUDGET), default="standard")
    parser.add_argument("--runtime-record", type=Path, required=True, help="Current local solver-server runtime record")
    parser.add_argument("--proposer-runtime-record", type=Path, help="Current local proposer-server runtime record")
    parser.add_argument("--harbor-work-dir", type=Path, help="Defaults to RUN_DIR/harbor")
    parser.add_argument("--harbor-executable", default="harbor")
    parser.add_argument("--docker-executable", default="docker")
    parser.add_argument("--harbor-process-timeout-sec", type=float, default=None)


def configure_models(args: argparse.Namespace, models: BenchmarkModels) -> BenchmarkModels:
    """Apply the established combined 32768-token ceiling to both model roles."""

    def role_kwargs(model: str, kwargs: dict[str, Any], *, agentic: bool) -> dict[str, Any]:
        result = {**deepcopy(kwargs), **terminalbench_decoding(model, agentic=agentic)}
        # The historical Terminal-Bench profile uses the checkpoint's thinking
        # template with a combined output cap, not HotPotQA's separate allowance.
        result["extra_body"].pop("thinking_token_budget", None)
        return result

    return replace(
        models,
        solver_kwargs=role_kwargs(models.solver_model, models.solver_kwargs, agentic=True),
        proposer_kwargs=role_kwargs(models.proposer_model, models.proposer_kwargs, agentic=False),
    )


def build_benchmark(args: argparse.Namespace, models: BenchmarkModels) -> BenchmarkDefinition:
    """Bind fixed data, actual prompt editing, official Harbor rewards, and runtime identity."""
    manifest = load_terminalbench_manifest(MANIFEST_PATH)
    records = {
        split: [
            {
                "id": "terminalbench:" + task_id,
                "task_id": task_id,
                "task_ref": manifest.task_refs[task_id],
                "split": split,
                "dataset": manifest.dataset["reference"],
            }
            for task_id in task_ids
        ]
        for split, task_ids in manifest.splits.items()
    }
    runtime = load_role_runtimes(
        argparse.Namespace(
            runtime_record=args.runtime_record,
            proposer_runtime_record=args.proposer_runtime_record,
            student_model=models.solver_model,
            proposer_model=models.proposer_model,
            student_api_base=models.solver_api_base,
            proposer_api_base=models.proposer_api_base,
        ),
        include_proposer=args.mode in {"optimize", "optimizer-pilot"},
    )
    scope = TerminalBenchTextScope("system_prompt", resolve_template_family(args.template_family, models.solver_model))
    solver_kwargs = deepcopy(models.solver_kwargs)
    # Harbor carries the endpoint as an agent constructor parameter.
    solver_kwargs.pop("api_base", None)
    harbor = HarborCLI(
        manifest=manifest,
        student_model=models.solver_model,
        student_api_base=models.solver_api_base,
        work_dir=args.harbor_work_dir or args.run_dir / "harbor",
        agent_python_path=REPO_ROOT,
        n_concurrent=args.max_workers,
        harbor_executable=args.harbor_executable,
        docker_executable=args.docker_executable,
        student_agent_kwargs={
            "token_limits": terminalbench_limits(models.solver_model),
            "model_info": terminalbench_model_info(models.solver_model),
            "llm_kwargs": solver_kwargs,
        },
        process_timeout_sec=args.harbor_process_timeout_sec,
    )
    harbor.check_requirements()
    selected_train_count = min(args.train_limit or len(records["train"]), len(records["train"]))
    minibatch_size = min(args.reflection_minibatch_size, selected_train_count)
    iterations_per_epoch = (selected_train_count + minibatch_size - 1) // minibatch_size
    return BenchmarkDefinition(
        name="terminalbench",
        adapter=SharedTerminusAdapter(
            TerminusAdapter(manifest, harbor, text_scope=scope),
            [record for split in records.values() for record in split],
        ),
        seed_candidate=scope.seed_candidate(),
        trainset=records["train"],
        valset=records["val"],
        testset=records["test"],
        source={
            "manifest_sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
            "dataset": manifest.dataset,
            "split_policy": manifest.split_policy,
            "task_refs": manifest.task_refs,
        },
        runtime={
            "harness": "terminalbench-shared-v1",
            "adapter": TERMINUS_ADAPTER_CONTRACT,
            "execution_runtime": {"student": runtime["student"]},
            "optimization_scope": scope.contract(),
            "document_bundle_version": BUNDLE_VERSION,
            "seed_document_digest": manifest.candidate_digest(scope.materialize(scope.seed_candidate())),
            "token_limits": terminalbench_limits(models.solver_model),
            "token_usage_policy": TOKEN_USAGE_POLICY,
            "task_context_settings": TASK_CONTEXT_SETTINGS,
            "failure_policy": FAILURE_POLICY_CONTRACT,
            "n_concurrent_trials": args.max_workers,
            "harbor_process_timeout_sec": args.harbor_process_timeout_sec,
            "attempts_per_task_per_repetition": 1,
            "repeat_seed_policy": "shared seed + repetition index; fresh Harbor job per repetition",
            "timing": "Harbor TrialResult.finished_at minus started_at; includes setup and teardown",
        },
        metric_name="pass_at_1",
        test_repetitions=TEST_REPETITIONS,
        component_kinds=scope.component_kinds,
        max_candidate_proposals=TRAINING_EPOCHS_BY_BUDGET[args.budget] * iterations_per_epoch,
        optimizer_runtime=runtime.get("proposer", {}),
    )


def main(argv: list[str] | None = None) -> int:
    """Route the primary entrypoint directly into the shared runner."""
    return run_cli(
        benchmark_name="terminalbench",
        build_benchmark=build_benchmark,
        add_arguments=add_arguments,
        configure_models=configure_models,
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
