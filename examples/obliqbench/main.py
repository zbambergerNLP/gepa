"""Run the shared GEPA/FOREST experiment protocol on OBLIQ retrieval."""

from __future__ import annotations

import argparse
from pathlib import Path

from examples.common.benchmark_runner import run_cli
from examples.common.benchmark_types import BenchmarkDefinition, BenchmarkModels
from examples.common.provider_retries import provider_retry_kwargs
from examples.common.react_v2 import resolve_template_family, structured_prompt
from examples.obliqbench.adapter import COMPONENT, ObliqAdapter
from examples.obliqbench.benchmark_settings import HARNESS_VERSION, METRIC_NAME, RETRIEVAL_K, SEED_INSTRUCTION, SUBSETS
from examples.obliqbench.retrieval import BM25Retriever, DenseRetriever, QwenEncoder, file_sha256, retrieval_contract
from examples.obliqbench.utils import load_data
from gepa.lm import LM
from gepa.lm_constants import PROVIDER_ATTEMPT_LOG
from gepa.strategies.forest_constants import SOLVER_ROLE


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add only the benchmark's data and fixed retrieval runtime settings."""
    parser.add_argument("--subsets", nargs="+", choices=tuple(SUBSETS), default=list(SUBSETS))
    parser.add_argument("--data-dir", type=Path, default=Path(".cache/obliqbench/data"))
    parser.add_argument("--index-dir", type=Path, default=Path(".cache/obliqbench/index"))
    parser.add_argument("--retriever", choices=("qwen", "bm25"), default="qwen")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument(
        "--original-query-reference",
        action="store_true",
        help="Baseline-only reference using original queries without an LLM call",
    )


def build_benchmark(args: argparse.Namespace, models: BenchmarkModels) -> BenchmarkDefinition:
    """Validate full immutable data before constructing the fixed retrieval program."""
    if args.original_query_reference and args.mode != "baseline":
        raise ValueError("Original-query reference is available only with --mode baseline")
    if args.max_workers != 1:
        raise ValueError("OBLIQ currently runs sequential episodes; use --max-workers 1")
    data = load_data(args.data_dir, args.subsets)
    retrieval = retrieval_contract(args.retriever, args.embedding_device, args.embedding_batch_size)
    encoder = QwenEncoder(args.embedding_device) if args.retriever == "qwen" else None
    retrievers = {
        name: DenseRetriever(corpus, encoder, args.index_dir, retrieval, args.embedding_batch_size)
        if encoder is not None
        else BM25Retriever(corpus)
        for name, corpus in data.corpora.items()
    }
    solver = LM(
        models.solver_model,
        **{
            **models.solver_kwargs,
            **provider_retry_kwargs(args.run_dir / PROVIDER_ATTEMPT_LOG, role=SOLVER_ROLE),
        },
    )
    family = resolve_template_family(args.template_family, models.solver_model)
    return BenchmarkDefinition(
        name="obliqbench",
        adapter=ObliqAdapter(solver, retrievers, original_query=args.original_query_reference),
        seed_candidate={COMPONENT: structured_prompt(SEED_INSTRUCTION, family)},
        trainset=data.splits["train"],
        valset=data.splits["val"],
        testset=data.splits["test"],
        source=data.source,
        runtime={
            "harness": HARNESS_VERSION,
            "harness_source_sha256": {
                path.name: file_sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))
            },
            "retriever": retrieval,
            "top_k": RETRIEVAL_K,
            "program": "original_query" if args.original_query_reference else "single_query_rewrite",
            "prompt_scope": COMPONENT,
            "metric": "trec_eval_graded_ndcg_and_positive_recall",
            "selection_metric": METRIC_NAME,
            "query_weighting": "equal_query",
            "repetition_seed_policy": "one_attempt_shared_solver_seed",
            "index_build_timing": "excluded_offline_preprocessing",
            "episode_timing": "rewrite_plus_query_embedding_plus_search_plus_metrics",
            "max_workers": 1,
        },
        metric_name=METRIC_NAME,
        test_repetitions=1,
        component_kinds={COMPONENT: "system_prompt"},
    )


def main(argv: list[str] | None = None) -> int:
    """Delegate optimizer, pilot, baseline, frozen selection, and tracking to the shared runner."""
    return run_cli(benchmark_name="obliqbench", build_benchmark=build_benchmark, add_arguments=add_arguments, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
