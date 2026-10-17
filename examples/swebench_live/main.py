"""Run the shared experiment lifecycle with the pinned local SWE-bench Live grader."""

from pathlib import Path

from examples.common.benchmark_types import BenchmarkDefinition
from examples.common.benchmark_variants import selected_conditions
from examples.swebench_live.adapter import SWEBenchLiveAdapter
from examples.swebench_live.apptainer import runtime_identity as apptainer_runtime
from examples.swebench_live.apptainer_images import verify_images as verify_apptainer_images
from examples.swebench_live.data import load_data
from examples.swebench_live.runtime import verify_runtime
from examples.swebench_live.settings import COMPONENT, ROOT
from examples.swebench_live.swe_agent import runtime_identity as agent_runtime
from examples.swebench_live.swe_agent import seed_prompt


def add_arguments(parser):
    """Expose local asset paths and record explicit native execution limits."""
    parser.add_argument("--swebench-live-root", type=Path, default=ROOT)
    parser.add_argument("--container-backend", choices=("apptainer",), default="apptainer")
    parser.add_argument("--swe-agent-root", type=Path)
    parser.add_argument("--task-memory-gib", type=int, default=16)
    parser.add_argument("--test-timeout", type=int, default=1800)
    parser.set_defaults(train_limit=60, val_limit=120, test_limit=120)


def build_benchmark(args, models):
    """Bind explicit hosted models to disjoint data and verified native task images."""
    if args.api_profile is None:
        raise ValueError("SWE-bench Live requires an explicit --api-profile; no implicit model defaults")
    if args.mode in {"optimize", "optimizer-pilot"} and selected_conditions(args) != ("vanilla",):
        raise ValueError(
            "The standard SWE-agent prompt supports vanilla GEPA only. Structured-prompt conditions need a "
            "separately reviewed integration; the upstream prompt will not be reformatted automatically."
        )
    if args.max_workers != 1:
        raise ValueError("Local SWE-bench Live executes one container at a time")
    if min(args.task_memory_gib, args.test_timeout) < 1:
        raise ValueError("SWE-bench Live execution bounds must be positive")
    root = args.swebench_live_root
    splits, source = load_data(root)
    runtime = verify_runtime(root)
    selected = [record for name, rows in splits.items() for record in rows[: getattr(args, f"{name}_limit")]]
    container_runtime = apptainer_runtime(args.task_memory_gib)
    images = verify_apptainer_images(root, selected)
    agent_root = args.swe_agent_root or root / "swe-agent"
    agent_identity = agent_runtime(agent_root, models)
    return BenchmarkDefinition(
        name="swebench_live",
        adapter=SWEBenchLiveAdapter(
            models,
            root,
            args.run_dir,
            [row for rows in splits.values() for row in rows],
            images,
            agent_root=agent_root,
            memory_gib=args.task_memory_gib,
            test_timeout=args.test_timeout,
        ),
        seed_candidate={COMPONENT: seed_prompt(agent_root)},
        trainset=splits["train"],
        valset=splits["val"],
        testset=splits["test"],
        source=source,
        runtime={
            "container": container_runtime,
            "official_grader": runtime,
            "images": images,
            "agent": agent_identity,
            "memory_gib": args.task_memory_gib,
            "cpus": 4,
            "network": "none",
            "test_timeout": args.test_timeout,
            "test_repetitions": 1,
        },
        metric_name="resolved",
        component_kinds={COMPONENT: "system_prompt"},
    )


def main(argv=None):
    """Use the existing authorization, journaling, optimization and frozen-winner gates."""
    from examples.common.benchmark_runner import run_cli

    run_cli(benchmark_name="swebench_live", build_benchmark=build_benchmark, add_arguments=add_arguments, argv=argv)


if __name__ == "__main__":
    main()
