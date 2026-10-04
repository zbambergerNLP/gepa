"""Deprecated entry point for the pre-Harbor Terminal-Bench adapter."""

if __name__ == "__main__":
    raise SystemExit(
        "This legacy `tb`/terminal_bench entry point is retired. Use "
        "`uv run python -m examples.terminalbench.main --help`; the maintained "
        "harness pins Harbor 0.22.0 and Terminal-Bench 2.1."
    )
