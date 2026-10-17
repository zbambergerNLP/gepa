"""Calibrate complete episodes using only the frozen training split."""

import sys

from examples.obliqbench.main import main

if __name__ == "__main__":
    raise SystemExit(main([*sys.argv[1:], "--mode", "pilot"]))
