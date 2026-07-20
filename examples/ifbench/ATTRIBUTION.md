# Attribution

This directory vendors the IFBench evaluation setup used in the GEPA paper
(arXiv:2507.19457), taken from the paper's artifact repository:

- Source: https://github.com/gepa-ai/gepa-artifact
  (`gepa_artifact/benchmarks/IFBench/`), Copyright 2025 Lakshya A Agrawal.

Vendored contents:

- `utils_ifbench/` — the IFBench instruction checkers and registry, originally
  from the Allen Institute for AI's IFBench (https://github.com/allenai/IFBench),
  Copyright 2025 Allen Institute for AI, licensed under the Apache License 2.0
  (see the license headers in each file). The only local modification is
  rewriting relative imports to absolute imports; the files are otherwise
  verbatim and excluded from ruff. Two local modifications: relative imports
  rewritten to absolute, and the module-level spacy `download('en_core_web_sm')`
  in `instructions.py` guarded to run only when the model is missing (so the
  import works on offline compute nodes).
- `data/IFBench_test.jsonl` (294 examples) and `data/IFBench_train.jsonl`
  (14,971 examples) — the exact data files bundled with the artifact. They are
  not committed here (the train file is ~16 MB); `utils.py` downloads them from
  the artifact repository on first use. Splits used by the paper (replicated in
  `utils.py`): test = all of `IFBench_test.jsonl`; val =
  `IFBench_train.jsonl[:300]`; train = `IFBench_train.jsonl[300:600]`.

`utils.py` ports the artifact's `ifbench_metric.py` (`metric_with_feedback`)
without the DSPy wrapper, and `main.py` replicates the artifact's 2-stage
program (`ifbench_program.py`) with plain LM calls.
