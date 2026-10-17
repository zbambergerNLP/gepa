"""Prepare SWE-bench Live assets without model calls or task execution."""

import argparse
import json

from examples.swebench_live.apptainer_images import stage_images as stage_apptainer_images
from examples.swebench_live.data import load_data, prepare_data
from examples.swebench_live.images import stage_images
from examples.swebench_live.runtime import prepare_runtime
from examples.swebench_live.settings import ROOT


def main(argv=None) -> None:
    """Stage pinned data and the official grader, optionally downloading task images."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=type(ROOT), default=ROOT)
    parser.add_argument("--images", action="store_true", help="Download only; never start a container")
    parser.add_argument("--image-split", choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--backend", choices=("docker", "apptainer"), default="docker")
    parser.add_argument(
        "--instance-id", action="append", help="Stage explicit Apptainer task IDs without changing run limits"
    )
    args = parser.parse_args(argv)
    if args.instance_id and (args.backend != "apptainer" or not args.images):
        parser.error("--instance-id requires --backend apptainer --images")
    prepare_data(args.root)
    splits, manifest = load_data(args.root)
    result = {
        "data": {name: len(rows) for name, rows in splits.items()},
        "partition": manifest["partition"],
        "runtime": prepare_runtime(args.root),
        "model_calls": 0,
        "containers_started": 0,
    }
    if args.images:
        result["images"] = (
            stage_apptainer_images(args.root, args.image_split, args.instance_id)
            if args.backend == "apptainer"
            else stage_images(args.root, args.image_split)
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
