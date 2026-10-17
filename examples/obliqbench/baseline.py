"""Run the shared unoptimized rewrite baseline, or the explicit original-query reference."""

import sys

from examples.obliqbench.main import main

if __name__ == "__main__":
    raise SystemExit(main([*sys.argv[1:], "--mode", "baseline"]))
