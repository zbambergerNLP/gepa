"""Run comparable GEPA and FOREST optimization of a single DecisionBench prompt."""

from __future__ import annotations

import argparse
from pathlib import Path

from decision_bench.prompt import PROMPT_VERSION, SYSTEM_PROMPT

from examples.common.benchmark_types import BenchmarkDefinition, BenchmarkModels
from examples.common.react_v2 import resolve_template_family, structured_prompt
from examples.decisionbench.adapter import DecisionBenchAdapter
from examples.decisionbench.benchmark_settings import ADAPTER_VERSION, COMPONENT, PROBABILITY_SOURCE
from examples.decisionbench.upstream import validate_upstream_runtime
from examples.decisionbench.utils import file_digest, load_decisionbench


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add only dataset-location arguments; keep experiment settings in the shared CLI."""
    parser.add_argument("--data-file", type=Path, help="Local copy of the exact pinned canonical Parquet")
    parser.add_argument("--dataset-cache", type=Path, help="Optional Hugging Face dataset download cache")
    parser.set_defaults(train_limit=150, val_limit=300, test_limit=300)


def build_benchmark(args: argparse.Namespace, models: BenchmarkModels) -> BenchmarkDefinition:
    """Bind official scoring to the fixed data, seed prompt, and shared model transport."""
    upstream = validate_upstream_runtime()
    splits, source = load_decisionbench(args.data_file, args.dataset_cache)
    family = resolve_template_family(args.template_family, models.solver_model)
    return BenchmarkDefinition(
        name="decisionbench",
        adapter=DecisionBenchAdapter(
            models,
            training_ids={record["id"] for record in splits["train"]},
            max_workers=args.max_workers,
        ),
        seed_candidate={COMPONENT: structured_prompt(SYSTEM_PROMPT, family)},
        trainset=splits["train"],
        valset=splits["val"],
        testset=splits["test"],
        source=source,
        runtime={
            "adapter": ADAPTER_VERSION,
            "official_runtime": upstream,
            "implementation_sha256": {
                name: file_digest(Path(__file__).with_name(name))
                for name in ("adapter.py", "utils.py", "main.py", "benchmark_settings.py", "upstream.py")
            },
            "prompt_contract": PROMPT_VERSION,
            "probability_source": PROBABILITY_SOURCE,
            "prediction_normalization": "divide_positive_finite_values_by_sum",
            "truncation_policy": "none_context_overflow_counts_as_miss",
            "input_fields": "standard",
            "repetition_policy": "one_attempt_per_row_shared_request_seed",
            "optimization_metric": "all_row_accuracy_errors_count_as_misses",
            "template_family": family,
            "transport": "shared_litellm_provider_retries",
        },
        metric_name="accuracy",
        test_repetitions=1,
        component_kinds={COMPONENT: "system_prompt"},
    )


def main(argv: list[str] | None = None) -> None:
    """Delegate experiment lifecycle, model settings, and tracking to the shared runner."""
    from examples.common.benchmark_runner import run_cli

    run_cli(benchmark_name="decisionbench", build_benchmark=build_benchmark, add_arguments=add_arguments, argv=argv)


if __name__ == "__main__":
    main()
