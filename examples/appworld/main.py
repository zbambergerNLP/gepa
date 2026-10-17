"""Run the shared GEPA/FOREST protocol on the official AppWorld benchmark."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from examples.common.benchmark_types import BenchmarkDefinition, BenchmarkModels

from examples.appworld.adapter import AppWorldAdapter
from examples.appworld.benchmark_settings import (
    APPWORLD_REVISION,
    CODE_TIMEOUT_SECONDS,
    COMPONENT,
    DEFAULT_MAX_STEPS,
    ENVIRONMENT_SEED,
    HARNESS_VERSION,
    RPC_TIMEOUT_SECONDS,
)
from examples.appworld.prompts import UPSTREAM_PROMPT, seed_candidate
from examples.appworld.runtime import OfficialAppWorld, inspect_runtime
from examples.appworld.utils import file_digest, load_dataset
from examples.common.react_v2 import resolve_template_family
from gepa.lm import LM

LOCAL_RUNTIME = Path(__file__).with_name(".runtime")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add only AppWorld's isolated runtime location and episode step cap."""
    parser.add_argument("--appworld-root", type=Path, default=LOCAL_RUNTIME / "world")
    parser.add_argument("--appworld-python", type=Path, default=LOCAL_RUNTIME / "venv" / "bin" / "python")
    parser.add_argument("--appworld-max-steps", type=int, default=DEFAULT_MAX_STEPS)


def build_benchmark(args: argparse.Namespace, models: BenchmarkModels) -> BenchmarkDefinition:
    """Verify data/runtime pins before constructing the real interactive agent."""
    root = args.appworld_root.resolve()
    records, source = load_dataset(root)
    runtime_identity = inspect_runtime(root, args.appworld_python)
    all_records = [record for split in records.values() for record in split]
    solver_kwargs = dict(models.solver_kwargs)
    if models.solver_api_base is not None:
        solver_kwargs["api_base"] = models.solver_api_base
    adapter = AppWorldAdapter(
        LM(models.solver_model, **solver_kwargs),
        partial(OfficialAppWorld, root, args.appworld_python),
        all_records,
        root,
        max_steps=args.appworld_max_steps,
    )
    return BenchmarkDefinition(
        name="appworld",
        adapter=adapter,
        seed_candidate=seed_candidate(resolve_template_family(args.template_family, models.solver_model)),
        trainset=records["train"],
        valset=records["dev"],
        testset=records["test_normal"] + records["test_challenge"],
        source=source,
        runtime={
            "harness": HARNESS_VERSION,
            "official_engine_revision": APPWORLD_REVISION,
            "runtime_identity": runtime_identity,
            "upstream_prompt_sha256": file_digest(UPSTREAM_PROMPT),
            "solver_model": models.solver_model,
            "proposer_model": models.proposer_model,
            "max_steps": args.appworld_max_steps,
            "environment_seed": ENVIRONMENT_SEED,
            "code_timeout_seconds": CODE_TIMEOUT_SECONDS,
            "rpc_timeout_seconds": RPC_TIMEOUT_SECONDS,
            "environment_max_interactions": 1000,
            "max_api_calls_per_interaction": 1000,
            "execution": "one isolated subprocess per episode; sequential adapter batches",
            "repeat_seed_policy": "one official attempt per task; fixed environment seed; shared model request seed",
            "ground_truth_mode": "minimal; evaluator only",
            "action_parser": "first closed python fence; no partial-code repair or stop sequence",
            "history": "full public demonstration and conversation; no history truncation",
            "prompt_scope": "upstream general and key instructions moved into one editable system prompt",
            "success": "official TestTracker.success; complete_task only controls termination",
            "scenario_aggregation": "official Metric; omitted for any incomplete scenario group",
        },
        metric_name="task_goal_completion",
        test_repetitions=1,
        component_kinds={COMPONENT: "system_prompt"},
    )


def main(argv: list[str] | None = None) -> None:
    """Delegate optimization, selection, baselines, timing, and resume to the shared runner."""
    from examples.common.benchmark_runner import run_cli

    run_cli(benchmark_name="appworld", build_benchmark=build_benchmark, add_arguments=add_arguments, argv=argv)


if __name__ == "__main__":
    main()
