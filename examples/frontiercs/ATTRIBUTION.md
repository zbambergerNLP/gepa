# Attribution

This directory implements the FrontierCS evaluation setup for GEPA, mirroring
`examples/ifbench` and `examples/pupa`. It is an **infra-only stretch**
benchmark — no runs are included.

- **Frontier-CS**: https://github.com/FrontierCS/Frontier-CS, HuggingFace
  `FrontierCS/Frontier-CS`. FrontierCS is an open-ended CS research problems
  benchmark (the paper reports ~100 problems across ML, Systems, Theory,
  Security, HCI) designed to evaluate auto-research frameworks. Each task is a
  research problem statement plus a rubric (criteria such as technical
  soundness, novelty, and evaluation plan). The reference implementation is an
  auto-research agent that surveys literature and drafts proposals. This
  example replicates that program structure as a 2-stage GEPA program
  (`literature_review` → `draft_proposal`) with a 1-stage ablation, and
  scores outputs with an LLM-judge rubric pass rate (0-1) plus a heuristic
  fallback — the same metric style as IFBench's instruction-level accuracy.

- **Dataset loading**: tries HuggingFace `FrontierCS/Frontier-CS` via
  `datasets.load_dataset` (any split: `train`/`test`/`validation`/`default`);
  if that fails, falls back to a local `data/frontiercs.jsonl` (one JSON
  record per line, see `utils._normalize_record` for accepted fields), and
  finally to a deterministic synthetic 90-example pool (10 stems cycling) so
  that `load_frontiercs_dataset()` always returns 30/30/30 for smoke/tests
  without network. Splits are a deterministic shuffle (seed 0) then
  30/30/30 (train 0:30, val 30:60, test 60:90), noting the paper's ~100-problem
  pool — the same slicing convention as IFBench (`IFBench_train.jsonl[300:600]`
  etc.) adapted to the smaller FrontierCS scale.

- **Metric**: `frontiercs_metric` in `utils.py` ports the rubric-based
  LLM-judge pattern from IFBench/PUPA (per-criterion PASS/FAIL plus a final
  `SCORE: <0-1>`, mean pass rate 0-1, blended 70/30 with the holistic score,
  feedback = the judge's trace). Offline fallback is heuristic keyword overlap
  per rubric item plus a length signal, still 0-1 with per-item feedback so
  GEPA reflection has a learning signal.

- **Decoding**: `_call_lm` is identical to `examples/ifbench/utils.py` and
  `examples/pupa/utils.py`: `temperature=0.6, top_p=0.95, top_k=20,
  max_tokens=16384, enable_thinking=False` (via
  `chat_template_kwargs.enable_thinking`), `COT_FORMAT_INSTRUCTION` with
  `Final Response:` marker, `<think>` stripping, and truncation retries
  16384→4096→1024→256 with a `reasoning_content` fallback — verbatim.

- **GEPA wiring**: `main.py` mirrors `examples/ifbench/main.py` /
  `examples/pupa/main.py` / `examples/hotpotqa/main.py`: `SEED_CANDIDATE`
  (2-stage + 1-stage), `_structured_seed` markdown skeleton, `condition_run_dir`,
  `seed_candidate`, `run_program`, `make_evaluator`, `evaluate_on_set`,
  `prompt_diversity`, `dump_candidates`, `dump_action_summary`, `build_config`
  (EngineConfig/GEPAConfig/ReflectionConfig with `action_selector` branching
  for `vanilla`/`random`/`action`), `run_condition`, and `main()` with
  `--data-path`, `--train-limit`/`--val-limit`/`--test-limit`, `--program`,
  `--seed-style`, `--actions`, `--condition`, `--tag`, `--solver-model` /
  `--reflection-model` / `--judge-model` / `--api-base`. Outputs are
  `outputs/<condition>[_{program}][_{tag}]/candidates.json` etc., identical
  to IFBench.

- **Code provenance**: `utils.py` decoding, program, and fallback structure
  are copied from `examples/ifbench/utils.py` and `examples/pupa/utils.py`
  to keep solver behavior identical across benchmarks. No vendored checker is
  needed (unlike IFBench's `utils_ifbench/`); the rubric is inline.

- **Splits & budget**: 30/30/30 splits with paper-size note (paper ~100),
  budget default 4000 metric calls (within the 3000-5000 stretch range,
  scaled Wave B 15000 like IFBench). `--data-path` and limit flags work the
  same as `examples/hotpotqa`.

- **SLURM**: `run_frontiercs.sbatch` is a 48-hour SLURM job mirroring
  `examples/ifbench/run_ifbench.sbatch` (1 GPU, 8 CPUs, 64 GB, vLLM serve via
  the POSIT venv, scratch cache, health check, `LIMIT_ARGS` including
  `--data-path`).
