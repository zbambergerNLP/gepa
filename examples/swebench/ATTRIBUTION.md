# Attribution

This directory implements the SWE-Bench evaluation setup in the GEPA
paper's style (arXiv:2507.19457), for SWE-Bench Verified
(Jimenez et al. 2024, https://www.swebench.com, ~2294 Python GitHub issues,
Verified 500 via `princeton-nlp/SWE-bench_Verified`).

- Dataset:
  https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified
  (SWE-Bench Verified, 500 curated instances sampled from the full SWE-Bench
  2294; source https://github.com/princeton-nlp/SWE-bench, paper
  Jimenez et al. 2024). Each instance has `instance_id`, `problem_statement`
  (issue text), `repo`, `base_commit`, `patch` (gold diff), `hints_text`,
  and test metadata. Used here via `datasets.load_dataset(
  "princeton-nlp/SWE-bench_Verified")` with shuffle seed 0 and splits
  30/30/30 (90 total) by default, auto-scaling to 100/100/100 when ≥300
  instances are available; for the full Verified test held-out set, pass
  `--test-limit 500`. The loader mirrors `examples/ifbench` (300/300/294) and
  `examples/pupa` (shuffled mid-split). No data files are committed; first
  load tries HF, then falls back to `data/swebench_verified.jsonl` (one issue
  per line: at least `instance_id`, `problem_statement`, `patch`), then
  synthetic issues so the pipeline never crashes offline. Place a local jsonl
  at `examples/swebench/data/swebench_verified.jsonl` or pass `--data-path`
  to override.

- Paper program: GEPA paper (Table 1, Sec 4, App E) describes code tasks with
  1- and 2-stage programs; this example adapts that pattern to SWE-Bench:
  1-stage code patch generation (`generate_patch`) and 2-stage locate-then-fix
  (`locate` -> `fix`: identify files/lines, then emit the unified diff),
  analogous to `examples/ifbench` (`generate_response` -> `ensure_correct_response`)
  and `examples/terminalbench` (plan-then-execute). Seed handling (plain
  sentences vs. markdown skeleton with Role/Task/Rules/Output Format/Examples)
  is copied from `examples/ifbench/main.py`.

- Metric: `swebench_metric` in `utils.py` is a proxy for patch-applies +
  tests-pass (offline-friendly): unified-diff structure checks (`diff --git`,
  `---`, `+++`, `@@`), then when a gold patch is available, file overlap and
  hunk line overlap scoring (1.0 exact match, 0.85 near match, 0.5 partial,
  0.25 correct file wrong hunk, 0.1 well-formed wrong file, 0.0 malformed),
  with ``` fencing warning (which would break `git apply`). Returns a score
  in [0,1] with per-check feedback for reflection — mirroring `examples/ifbench`
  `ifbench_metric` and `examples/pupa` `pupa_metric`. For publication runs,
  replace the proxy with real `git apply` + repository test execution
  (see https://www.swebench.com and `SWE-bench` harness):
  `python -m swebench.harness.run_evaluation --dataset_name
  princeton-nlp/SWE-bench_Verified --predictions_path <patches.json>`.

- Code provenance: `utils.py` `_call_lm` and decoding config (temp 0.6,
  top_p 0.95, top_k 20, max 16384, `enable_thinking: False`) are copied
  verbatim from `examples/ifbench/utils.py` to keep solver behavior identical
  across benchmarks. Truncation handling (stepping `max_tokens`
  16384→4096→1024→256 on `ContextWindowExceededError`, capping issue text at
  ~24k chars and location at ~8k) and `<think>` stripping are likewise
  identical, with long-code-context truncation added for SWE-Bench's
  typically larger inputs. `main.py` (EngineConfig/GEPAConfig/ReflectionConfig
  wiring, action_selector branching, seed_style/actions/condition handling,
  dumps, diversity report) is patterned on `examples/ifbench/main.py` and
  `examples/pupa/main.py`.

- Infrastructure: `run_swebench.sbatch` (48h, 1 GPU, 8 CPUs, 64G) mirrors
  `examples/ifbench/run_ifbench.sbatch` (vLLM via POSIT venv, unique per-job
  port, scratch caches, health check, `hosted_vllm/` + `api_base`).

License: MIT (GEPA). SWE-Bench datasets are licensed per their repositories
(see https://github.com/princeton-nlp/SWE-bench and HuggingFace dataset cards).
SWE-Bench Verified is a curated subset of https://github.com/princeton-nlp/SWE-bench.
