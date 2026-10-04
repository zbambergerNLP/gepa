"""Download and verify pinned OBLIQ files without calling a model."""

import argparse
from pathlib import Path

from examples.obliqbench.benchmark_settings import SUBSETS
from examples.obliqbench.utils import frozen_records, load_data, prepare_data


def main(argv: list[str] | None = None) -> None:
    """Prepare all five subsets or an explicitly selected subset for a smoke check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(".cache/obliqbench/data"))
    parser.add_argument("--subsets", nargs="+", choices=tuple(SUBSETS), default=list(SUBSETS))
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Verify queries, graded judgments, and exclusions without downloading large corpora",
    )
    args = parser.parse_args(argv)
    prepare_data(args.data_dir, args.subsets, metadata_only=args.metadata_only)
    if args.metadata_only:
        print({subset: len(frozen_records(args.data_dir, subset)[0]) for subset in args.subsets})
    else:
        data = load_data(args.data_dir, args.subsets)
        print(
            {"subsets": list(data.corpora), "split_counts": {split: len(rows) for split, rows in data.splits.items()}}
        )


if __name__ == "__main__":
    main()
