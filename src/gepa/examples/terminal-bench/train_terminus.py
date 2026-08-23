"""Deprecated entry point for the pre-Harbor Terminal-Bench adapter."""


def main() -> None:
    """Exit with the maintained Harbor harness location."""
    raise SystemExit(
        "This legacy `tb`/terminal_bench entry point is retired. Use "
        "`uv run python examples/terminalbench/main.py --help`; the maintained "
        "harness pins Harbor 0.22.0 and terminal-bench/terminal-bench@3.0.0."
    )


if __name__ == "__main__":
    main()
