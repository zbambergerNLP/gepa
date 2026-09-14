"""Exercise all eight method/scope paths for one Terminal-Bench model."""

import argparse
from pathlib import Path

from examples.common.pilot_checks import METHODS
from examples.common.recovery import run_guarded
from examples.terminalbench.main import main as run_check


def main(argv: list[str] | None = None) -> None:
    """Run one cycle per path, keeping system-prompt checks first."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args, remaining = parser.parse_known_args(argv)
    forbidden = (
        "--optimizer-pilot",
        "--run-dir",
        "--harbor-work-dir",
        "--condition",
        "--optimization-scope",
        "--budget",
    )
    if any(value.split("=")[0] in forbidden for value in remaining):
        parser.error("The pilot owns scope, method, budget, and output directories")
    for scope in ("system_prompt", "all_text"):
        for method in METHODS:
            directory = args.output_dir / scope / method
            run_check(
                [
                    *remaining,
                    "--optimizer-pilot",
                    "--condition",
                    method,
                    "--optimization-scope",
                    scope,
                    "--run-dir",
                    str(directory),
                    "--harbor-work-dir",
                    str(directory / "harbor"),
                ]
            )


if __name__ == "__main__":
    run_guarded(main)
