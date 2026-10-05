"""Prepare pinned OBLIQ files and optional corpus indexes without solver calls."""

import argparse
import json
from pathlib import Path

from examples.obliqbench.benchmark_settings import SUBSETS
from examples.obliqbench.utils import frozen_records, load_data, prepare_data


def main(argv: list[str] | None = None) -> None:
    """Prepare all five subsets or an explicitly selected subset for a smoke check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(".cache/obliqbench/data"))
    parser.add_argument("--subsets", nargs="+", choices=tuple(SUBSETS), default=list(SUBSETS))
    parser.add_argument("--build-index", action="store_true", help="Build or verify all selected Qwen corpus indexes")
    parser.add_argument("--index-dir", type=Path, default=Path(".cache/obliqbench/index"))
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--embedding-attention", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--embedding-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Verify queries, graded judgments, and exclusions without downloading large corpora",
    )
    args = parser.parse_args(argv)
    if args.build_index and args.metadata_only:
        parser.error("--build-index requires complete corpora; it cannot be combined with --metadata-only")
    if args.embedding_batch_size < 1:
        parser.error("--embedding-batch-size must be positive")
    prepare_data(args.data_dir, args.subsets, metadata_only=args.metadata_only)
    if args.metadata_only:
        print({subset: len(frozen_records(args.data_dir, subset)[0]) for subset in args.subsets})
    else:
        data = load_data(args.data_dir, args.subsets)
        if args.build_index:
            from examples.obliqbench.embedding_runtime import QwenEncoder, retrieval_contract
            from examples.obliqbench.retrieval import DenseRetriever

            encoder_settings = {"attention": args.embedding_attention, "dtype": args.embedding_dtype}
            contract = retrieval_contract("qwen", args.embedding_device, args.embedding_batch_size, **encoder_settings)
            encoder = QwenEncoder(args.embedding_device, **encoder_settings)
            indexes = {}
            for name, corpus in data.corpora.items():
                retriever = DenseRetriever(corpus, encoder, args.index_dir, contract, args.embedding_batch_size)
                indexes[name] = {
                    "documents": len(corpus.ids),
                    "index_path": str(Path(retriever.embeddings.filename).resolve()),
                }
            print(
                json.dumps(
                    {
                        "subsets": list(data.corpora),
                        "split_counts": {split: len(rows) for split, rows in data.splits.items()},
                        "retriever": contract,
                        "indexes": indexes,
                    },
                    sort_keys=True,
                )
            )
            return
        print(
            {"subsets": list(data.corpora), "split_counts": {split: len(rows) for split, rows in data.splits.items()}}
        )


if __name__ == "__main__":
    main()
