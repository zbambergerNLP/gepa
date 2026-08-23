# Terminal-Bench adapter

The adapter invokes the official Harbor CLI through an isolated subprocess boundary and parses its rewards, errors, and complete ATIF trajectories. See [`examples/terminalbench/README.md`](../../../../examples/terminalbench/README.md) for the pinned versions, manifest, setup, and run configuration.

The old `terminal_bench` Python-package integration under `src/gepa/examples/terminal-bench` is deprecated because it used the retired `tb` CLI, mutable `head` dataset version, shared prompt file, and partial legacy trajectory parsing.
