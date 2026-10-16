"""Run the shared experiment lifecycle with the pinned local SWE-bench Live grader."""

from pathlib import Path

from examples.common.benchmark_types import BenchmarkDefinition
from examples.common.react_v2 import resolve_template_family, structured_prompt
from examples.swebench_live.adapter import SWEBenchLiveAdapter
from examples.swebench_live.data import load_data
from examples.swebench_live.images import verify_images
from examples.swebench_live.runtime import verify_runtime
from examples.swebench_live.settings import COMPONENT, ROOT, SEED_PROMPT


def add_arguments(parser):
    """Expose local asset paths and record explicit native execution limits."""
    parser.add_argument("--swebench-live-root", type=Path, default=ROOT)
    parser.add_argument("--agent-max-steps", type=int, default=100)
    parser.add_argument("--task-memory-gib", type=int, default=16)
    parser.add_argument("--test-timeout", type=int, default=1800)
    parser.set_defaults(train_limit=150, val_limit=300, test_limit=None)


def build_benchmark(args, models):
    """Bind explicit hosted models to disjoint data and verified native task images."""
    if args.api_profile is None:
        raise ValueError("SWE-bench Live requires an explicit --api-profile; no implicit model defaults")
    if args.max_workers != 1:
        raise ValueError("Local SWE-bench Live executes one container at a time")
    if min(args.agent_max_steps, args.task_memory_gib, args.test_timeout) < 1:
        raise ValueError("SWE-bench Live execution bounds must be positive")
    root = args.swebench_live_root
    splits, source = load_data(root)
    runtime = verify_runtime(root)
    selected = [record for name, rows in splits.items() for record in rows[: getattr(args, f"{name}_limit")]]
    images = verify_images(root, selected, args.task_memory_gib)
    family = resolve_template_family(args.template_family, models.solver_model)
    return BenchmarkDefinition(
        name="swebench_live",
        adapter=SWEBenchLiveAdapter(
            models,
            root,
            args.run_dir,
            [row for rows in splits.values() for row in rows],
            images,
            max_steps=args.agent_max_steps,
            memory_gib=args.task_memory_gib,
            test_timeout=args.test_timeout,
        ),
        seed_candidate={COMPONENT: structured_prompt(SEED_PROMPT, family)},
        trainset=splits["train"],
        valset=splits["val"],
        testset=splits["test"],
        source=source,
        runtime={
            "official_grader": runtime,
            "images": images,
            "agent": "native-shell-v1",
            "max_steps": args.agent_max_steps,
            "memory_gib": args.task_memory_gib,
            "cpus": 4,
            "network": "none",
            "test_timeout": args.test_timeout,
            "command_timeout": 120,
            "template_family": family,
            "truncation": "none",
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
