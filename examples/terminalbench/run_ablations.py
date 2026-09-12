"""Run one model's Terminal-Bench ablations with system-prompt optimization first."""

import argparse
import shlex
import subprocess
from pathlib import Path

from examples.terminalbench.main import REPO_ROOT, SCOPE_CAMPAIGN_CELLS, build_parser


def main(argv: list[str] | None = None) -> None:
    """Optimize and test each cell before advancing, with shared data and runtime options.

    Args:
        argv: Optional arguments; omitted uses the process command line.

    Raises:
        subprocess.CalledProcessError: A cell failed; later cells are not launched.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Other options are forwarded to examples.terminalbench.main for every cell.",
        allow_abbrev=False,
    )
    parser.add_argument("--run-root", type=Path, required=True, help="One model's campaign directory")
    parser.add_argument("--dry-run", action="store_true", help="Print the ordered commands without executing them")
    args, shared_options = parser.parse_known_args(argv)
    managed = {"--optimization-scope", "--condition", "--budget", "--run-dir", "--harbor-work-dir", "--test-output-dir"}
    if any(option.split("=", 1)[0] in managed for option in shared_options):
        parser.error("Scope, condition, budget, and per-run directories are set by the campaign matrix")
    run_root = args.run_root.resolve()
    commands = []
    for label, (scope, condition, budget) in SCOPE_CAMPAIGN_CELLS.items():
        cell = label.split("__", 1)[1]
        run_dir = run_root / scope / cell
        options = [
            *shared_options,
            "--optimization-scope",
            scope,
            "--condition",
            condition,
            "--budget",
            budget,
            "--run-dir",
            str(run_dir),
            "--harbor-work-dir",
            str(run_dir / "harbor"),
            "--test-output-dir",
            str(run_root / "test"),
        ]
        cell_args = build_parser().parse_args(options)
        if cell_args.train_limit is not None or cell_args.val_limit is not None:
            parser.error("Campaign ablations must use the complete, identical pinned splits; no split limits")
        commands.append(["uv", "run", "--no-sync", "python", "-m", "examples.terminalbench.main", *options])
    for command in commands:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
