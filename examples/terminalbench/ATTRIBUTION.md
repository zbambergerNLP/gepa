# Attribution

This directory implements the TerminalBench evaluation setup in the GEPA
paper's style (arXiv:2507.19457), for TerminalBench (T-Bench,
https://terminal-bench.github.io, 2024, 50+ terminal agent tasks, Docker-based).

- Dataset:
  https://huggingface.co/datasets/laude/terminal-bench (T-Bench, 50+ tasks;
  source https://github.com/laude-institute/terminal-bench). Tasks are
  natural-language instructions whose solutions are shell commands/scripts,
  validated inside Docker containers via unit tests and exit-code checks
  (see https://terminal-bench.github.io). Used here via
  `datasets.load_dataset("laude/terminal-bench")` with shuffle seed 0 and
  splits 20/15/15 (50 total) for optimization, mirroring `examples/ifbench`
  (300/300/294). No data files are committed; first load tries HF, then
  falls back to `data/terminalbench.jsonl` (one task per line: at least
  `prompt`/`task_id`, optionally `expected_commands`/`tests`), then synthetic
  tasks so the pipeline never crashes offline. Place a local jsonl at
  `examples/terminalbench/data/terminalbench.jsonl` or pass
  `--data-path` to override.

- Paper program: GEPA paper (Table 1, Sec 4, App E) describes instruction
  following and code generation with 1- and 2-stage programs; this example
  adapts that pattern to TerminalBench: 1-stage shell command generation
  (`generate_command`) and 2-stage plan-then-execute (`plan` -> `execute`),
  analogous to `examples/ifbench` (`generate_response` -> `ensure_correct_response`)
  and `examples/pupa` (1-stage). The program structure and seed handling
  (plain sentences vs. markdown skeleton with Role/Task/Rules/Output Format/Examples)
  are copied from `examples/ifbench/main.py`.

- Metric: `terminalbench_metric` in `utils.py` is a proxy for the Docker
  unit-test / exit-code evaluation (offline-friendly): shell-validity heuristics
  (plausible tokens like `|`, `>`, `&&`, `ls`, `grep`), token overlap with
  `expected_commands`/`tests` when available, task keyword relevance, and
  markdown-fencing warning. Returns a score in [0,1] (0/0.25/0.5/1.0) with
  per-check feedback for reflection — mirroring `examples/ifbench`
  `ifbench_metric` (instruction-level accuracy + per-constraint feedback) and
  `examples/pupa` `pupa_metric` (quality+leakage aggregate). For publication
  runs, replace the proxy with real Docker execution (see T-Bench docs):
  `terminal-bench run --dataset laude/terminal-bench --agent <cmd>`.

- Code provenance: `utils.py` `_call_lm` and decoding config (temp 0.6,
  top_p 0.95, top_k 20, max 16384, `enable_thinking: False`) are copied
  verbatim from `examples/ifbench/utils.py` to keep solver behavior identical
  across benchmarks. Truncation handling (stepping `max_tokens` 16384→4096→1024→256
  on `ContextWindowExceededError`, capping stage-1 output at ~24k chars) and
  `<think>` stripping are likewise identical. `main.py` (EngineConfig/GEPAConfig/
  ReflectionConfig wiring, action_selector branching, seed_style/actions/condition
  handling, dumps, diversity report) is patterned on `examples/ifbench/main.py`
  and `examples/pupa/main.py`.

- Infrastructure: `run_terminalbench.sbatch` (48h, 1 GPU, 8 CPUs, 64G) mirrors
  `examples/ifbench/run_ifbench.sbatch` (vLLM via POSIT venv, unique per-job
  port, scratch caches, health check, `hosted_vllm/` + `api_base`).

License: MIT (GEPA). TerminalBench is licensed per its repository (see
https://github.com/laude-institute/terminal-bench).
