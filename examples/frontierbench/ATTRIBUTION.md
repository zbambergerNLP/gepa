# Attribution

This directory implements the FrontierBench evaluation setup for GEPA, mirroring
`examples/ifbench`, `examples/pupa`, and `examples/frontiercs`. It is an
**infra-only stretch** benchmark — no runs are included.

- **Frontier-Bench**: https://github.com/laude-institute/frontier-bench,
  HuggingFace `laude/frontier-bench`. Frontier-Bench is the harder agentic
  research benchmark from the Terminal-Bench authors (Laude Institute). Tasks
  are end-to-end research assignments (literature + code + analysis) that
  require an agent to plan and execute, traditionally scored by a Terminal-Bench
  test harness in Docker. This infra-only stretch replicates that as a 2-stage
  GEPA program (`research_plan` → `execute_task`) with a 1-stage ablation, and
  scores outputs with an LLM-judge task-success rate (0-1) plus a heuristic
  fallback — the same metric style as FrontierCS/IFBench.

- **Dataset loading**: tries HuggingFace `laude/frontier-bench` via
  `datasets.load_dataset` (any split: `train`/`test`/`validation`/`default`);
  if that fails, falls back to a local `data/frontierbench.jsonl` (one JSON
  record per line, see `utils._normalize_record` for accepted fields), and
  finally to a deterministic synthetic 90-example pool (10 stems cycling) so
  that `load_frontierbench_dataset()` always returns 30/30/30 for smoke/tests
  without network. Splits are a deterministic shuffle (seed 0) then 30/30/30
  (train 0:30, val 30:60, test 60:90), noting the Terminal-Bench / Frontier-
  Bench scale — the same slicing convention as IFBench adapted to this
  benchmark.

- **Metric**: `frontierbench_metric` in `utils.py` ports the LLM-judge pattern
  from FrontierCS/IFBench/PUPA (per-test PASS/FAIL plus a final `SCORE: <0-1>`,
  mean pass rate 0-1, blended 70/30 with the holistic score, feedback = the
  judge's trace). Offline fallback is heuristic keyword overlap per test plus a
  structure/length signal, still 0-1 with per-test feedback so GEPA reflection
  has a learning signal. The real FrontierBench test-suite pass (Docker harness)
  is intentionally approximated here so the benchmark is runnable without Docker;
  for a full-harness integration, replace the judge path with the harness call
  (the infra-only choice is documented in `README.md`).

- **Decoding**: `_call_lm` is identical to `examples/ifbench/utils.py`,
  `examples/pupa/utils.py`, and `examples/frontiercs/utils.py`:
  `temperature=0.6, top_p=0.95, top_k=20, max_tokens=16384,
  enable_thinking=False` (via `chat_template_kwargs.enable_thinking`),
  `COT_FORMAT_INSTRUCTION` with `Final Response:` marker, `<think>` stripping,
  and truncation retries 16384→4096→1024→256 with a `reasoning_content` fallback
  — verbatim.

- **GEPA wiring**: `main.py` mirrors `examples/ifbench/main.py` /
  `examples/frontiercs/main.py` / `examples/hotpotqa/main.py`: `SEED_CANDIDATE`
  (2-stage + 1-stage), `_structured_seed` markdown skeleton, `condition_run_dir`,
  `seed_candidate`, `run_program`, `make_evaluator`, `evaluate_on_set`,
  `prompt_diversity`, `dump_candidates`, `dump_action_summary`, `build_config`
  (EngineConfig/GEPAConfig/ReflectionConfig with `action_selector` branching
  for `vanilla`/`random`/`action`), `run_condition`, and `main()` with
  `--data-path`, `--train-limit`/`--val-limit`/`--test-limit`, `--program`,
  `--seed-style`, `--actions`, `--condition`, `--tag`, `--solver-model` /
  `--reflection-model` / `--judge-model` / `--api-base`. Outputs are
  `outputs/<condition>[_{program}][_{tag}]/candidates.json` etc., identical
  to IFBench/FrontierCS.

- **Code provenance**: `utils.py` decoding, program, and fallback structure are
  copied from `examples/ifbench/utils.py` / `examples/pupa/utils.py` /
  `examples/frontiercs/utils.py` to keep solver behavior identical across
  benchmarks. No vendored harness is committed (unlike IFBench's
  `utils_ifbench/`); the test-suite approximation is inline so the example
  remains infra-only and does not modify other examples.

- **Splits & budget**: 30/30/30 splits with Terminal-Bench-scale note, budget
  default 4000 metric calls (within the 3000-5000 stretch range, scaled Wave B
  15000 like IFBench/FrontierCS). `--data-path` and limit flags work the same
  as `examples/frontiercs` / `examples/hotpotqa`.

- **SLURM**: `run_frontierbench.sbatch` is a 48-hour SLURM job mirroring
  `examples/frontiercs/run_frontiercs.sbatch` and
  `examples/ifbench/run_ifbench.sbatch` (1 GPU, 8 CPUs, 64 GB, vLLM serve via
  the POSIT venv, scratch cache, health check, `LIMIT_ARGS` including
  `--data-path`).
