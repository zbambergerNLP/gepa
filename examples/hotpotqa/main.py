"""Run the pinned HotPotQA program through the shared benchmark lifecycle."""

from __future__ import annotations

import argparse
from pathlib import Path

from examples.common.benchmark_runner import run_cli
from examples.common.benchmark_types import BenchmarkDefinition, BenchmarkModels
from examples.common.react_v2 import benchmark_data_identity, file_sha256, resolve_template_family, structured_prompt
from examples.common.wiki17_bm25 import DEFAULT_WIKI17_ROOT, GEPA_ARTIFACT_COMMIT, Wiki17BM25Retriever
from examples.hotpotqa.adapter import HotPotQAAdapter
from examples.hotpotqa.benchmark_settings import (
    DATASET_SAMPLE_SEED,
    HOTPOTQA_DSPY_COMMIT,
    HOTPOTQA_DSPY_VERSION,
    HOTPOTQA_HF_REVISION,
    HOTPOTQA_SCIENTIFIC_SPLIT_SHA256,
    HOTPOTQA_SHARED_HARNESS,
    RETRIEVAL_K,
    SEED_CANDIDATE,
    SPLIT_COUNTS,
    TEST_REPETITIONS,
)
from examples.hotpotqa.utils import load_hotpotqa_dataset, validate_hotpotqa_dspy_runtime


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the existing dataset smoke override and frozen Wiki-2017 location."""
    parser.add_argument(
        "--wiki17-dir", type=Path, default=DEFAULT_WIKI17_ROOT, help="Prepared, verified Wiki-2017 BM25 index"
    )
    parser.add_argument("--data-path", type=Path, help="Explicit JSONL smoke data; default is pinned HotPotQA fullwiki")


def load_benchmark_data(data_path: Path | None) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Freeze the artifact split independently of optimization seed and prefix limits."""
    train, val, test = load_hotpotqa_dataset(data_path=str(data_path) if data_path else None)
    splits = {"train": train, "val": val, "test": test}
    seen_ids, seen_questions = set(), set()
    for name, rows in splits.items():
        if not rows:
            raise ValueError(f"HotPotQA {name} split is empty")
        for row in rows:
            if any(not isinstance(row.get(key), str) or not row[key].strip() for key in ("id", "question", "answer")):
                raise ValueError("HotPotQA requires identified questions with nonempty reference answers")
            question_key = " ".join(row["question"].casefold().split())
            if row["id"] in seen_ids or question_key in seen_questions:
                raise ValueError("HotPotQA splits contain repeated question identities")
            seen_ids.add(row["id"])
            seen_questions.add(question_key)
    if data_path is None:
        source = {
            "dataset": "hotpot_qa",
            "config": "fullwiki",
            "revision": HOTPOTQA_HF_REVISION,
            "source_split": "train",
            "split_policy": "ordered-40-40-20-then-independent-seed1-sampling",
            "sample_seed": DATASET_SAMPLE_SEED,
            "gepa_artifact_commit": GEPA_ARTIFACT_COMMIT,
        }
        identity = benchmark_data_identity(source=source, trainset=train, valset=val, testset=test)
        for name, rows in splits.items():
            if (
                len(rows) != SPLIT_COUNTS[name]
                or identity["splits"][name]["sha256"] != HOTPOTQA_SCIENTIFIC_SPLIT_SHA256[name]
            ):
                raise ValueError(f"HotPotQA pinned ordered {name} split changed")
    else:
        source = {"type": "explicit_jsonl_smoke", "sha256": file_sha256(data_path.expanduser().resolve())}
    return train, val, test, source


def build_benchmark(args: argparse.Namespace, models: BenchmarkModels) -> BenchmarkDefinition:
    """Bind the existing DSPy program, pinned BM25 corpus, EM and four editable prompts."""
    validate_hotpotqa_dspy_runtime()
    train, val, test, source = load_benchmark_data(args.data_path)
    retriever = Wiki17BM25Retriever(args.wiki17_dir)
    retriever.verify_integrity()
    provenance = retriever.provenance()
    if not provenance.get("integrity_manifest_sha256"):
        raise ValueError("HotPotQA requires verified Wiki-2017 corpus and index integrity")
    if len(retriever.search(train[0]["question"], RETRIEVAL_K)) != RETRIEVAL_K:
        raise ValueError("HotPotQA training preflight did not retrieve the required passages")
    family = resolve_template_family(args.template_family, models.solver_model)
    return BenchmarkDefinition(
        name="hotpotqa",
        adapter=HotPotQAAdapter(
            models, retriever, training_ids={row["id"] for row in train}, max_workers=args.max_workers
        ),
        seed_candidate={component: structured_prompt(text, family) for component, text in SEED_CANDIDATE.items()},
        trainset=train,
        valset=val,
        testset=test,
        source=source,
        runtime={
            "harness": HOTPOTQA_SHARED_HARNESS,
            "program": "2stage",
            "components": list(SEED_CANDIDATE),
            "dspy_version": HOTPOTQA_DSPY_VERSION,
            "dspy_commit": HOTPOTQA_DSPY_COMMIT,
            "retrieval": provenance,
            "retrieval_k": RETRIEVAL_K,
            "metric": "normalized_answer_exact_match",
            "template_family": family,
            "reflection": "training_only_artifact_component_feedback_with_outcome_diagnostics",
            "repetition_policy": "one_attempt_per_question_fixed_shared_request_seed",
            "implementation_sha256": {
                name: file_sha256(Path(__file__).with_name(name))
                for name in ("main.py", "adapter.py", "utils.py", "benchmark_settings.py")
            },
        },
        metric_name="exact_match",
        test_repetitions=TEST_REPETITIONS,
        component_kinds=dict.fromkeys(SEED_CANDIDATE, "system_prompt"),
    )


def main(argv: list[str] | None = None) -> int:
    """Use the shared optimizer, winner freeze, held-out baseline and timing lifecycle."""
    return run_cli(benchmark_name="hotpotqa", build_benchmark=build_benchmark, add_arguments=add_arguments, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
