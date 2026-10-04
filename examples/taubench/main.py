"""Run matched GEPA/FOREST optimization of the tau banking agent system prompt."""

from __future__ import annotations

import argparse
from pathlib import Path

from examples.common.benchmark_types import BenchmarkDefinition, BenchmarkModels
from examples.common.react_v2 import resolve_template_family, structured_prompt
from examples.taubench.adapter import TauBankingAdapter
from examples.taubench.benchmark_settings import (
    DEFAULT_SOURCE,
    DOMAIN,
    MAX_ERRORS,
    MAX_STEPS,
    PYTHON_VERSION,
    RETRIEVAL_CONFIG,
    RETRIEVAL_TOP_K,
    RUNTIME_SUPPLEMENTS,
    TEST_REPETITIONS,
    TRIAL_SEED,
    UPSTREAM_REVISION,
)
from examples.taubench.model_settings import JUDGE_MODEL, USER_KWARGS, USER_MODEL
from examples.taubench.runtime import TauRuntime
from examples.taubench.utils import digest, load_data, upstream_system_prompt


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the one benchmark-specific path; scientific settings remain pinned."""
    parser.add_argument(
        "--tau-source", type=Path, default=DEFAULT_SOURCE, help=f"Official tau checkout at {UPSTREAM_REVISION}"
    )


def build_benchmark(args: argparse.Namespace, models: BenchmarkModels) -> BenchmarkDefinition:
    """Verify data before binding the upstream worker to the shared experiment runner."""
    source = args.tau_source.resolve()
    splits, manifest = load_data(source)
    runtime = TauRuntime(
        source, Path(args.run_dir) / "tau-episodes", models.solver_model, models.solver_api_base, models.solver_kwargs
    )
    family = resolve_template_family(args.template_family, models.solver_model)
    seed = {"system_prompt": structured_prompt(upstream_system_prompt(source), family)}
    return BenchmarkDefinition(
        name="taubench",
        adapter=TauBankingAdapter(runtime, manifest, args.max_workers),
        seed_candidate=seed,
        trainset=splits["train"],
        valset=splits["val"],
        testset=splits["test"],
        source=manifest,
        runtime={
            "harness": "upstream_tau_text_llm_agent",
            "source_revision": UPSTREAM_REVISION,
            "python": PYTHON_VERSION,
            "supplements": list(RUNTIME_SUPPLEMENTS),
            "domain": DOMAIN,
            "retrieval": RETRIEVAL_CONFIG,
            "top_k": RETRIEVAL_TOP_K,
            "corpus_order": "document_id_ascending",
            "user_model": USER_MODEL,
            "user_kwargs": USER_KWARGS,
            "judge_model": JUDGE_MODEL,
            "judge_kwargs": USER_KWARGS,
            "evaluator": "EvaluationType.ALL; required reward_basis product",
            "max_steps": MAX_STEPS,
            "max_errors": MAX_ERRORS,
            "enforce_communication_protocol": False,
            "trial_seed": TRIAL_SEED,
            "trial_seed_policy": "random.Random(300).randint(0,1000000) per trial",
            "optimization_trial": 0,
            "test_repetitions": TEST_REPETITIONS,
            "trajectory_retries": 0,
            "upstream_prompt_sha256": digest(upstream_system_prompt(source)),
            "template_family": family,
            "manifest_sha256": digest(manifest),
            "max_workers": args.max_workers,
        },
        metric_name="pass_hat_1",
        test_repetitions=TEST_REPETITIONS,
        component_kinds={"system_prompt": "system_prompt"},
    )


def main(argv: list[str] | None = None):
    """Delegate optimizer, baseline, pilot, tracking, and held-out lifecycle to the shared runner."""
    from examples.common.benchmark_runner import run_cli

    return run_cli(benchmark_name="taubench", build_benchmark=build_benchmark, add_arguments=add_arguments, argv=argv)


if __name__ == "__main__":
    main()
