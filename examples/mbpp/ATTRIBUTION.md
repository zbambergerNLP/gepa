# Attribution

## Dataset: MBPP

- **MBPP (Mostly Basic Python Problems)** — Austin et al., 2021. 974 Python programming problems requiring single-function solutions. Paper: [Program Synthesis with Large Language Models](https://arxiv.org/abs/2108.07732).
- **HuggingFace**: [`mbpp`](https://huggingface.co/datasets/mbpp) (sanitized, 974; canonical splits vary by config — this example pools train/test/prompt to 150/300/300 seed 0), and [`google-research/mbpp`](https://huggingface.co/datasets/google-research/mbpp) mirror. Each example has `task_id`, `text` (problem description), `code` (reference solution), `test_list` (asserts), `challenge_test_list` (hidden tests), `test_setup_code`. Sanitized version is recommended (cleaner asserts).
- **HumanEval (paired)** — Chen et al., 2021. 164 hand-written Python problems (used alongside MBPP in CANTANTE/FlowBot). HF: [`openai/openai_humaneval`](https://huggingface.co/datasets/openai/openai_humaneval) / [`openai_humaneval`](https://huggingface.co/datasets/openai_humaneval). Not separately scaffolded here — MBPP covers the code generation cluster; add via `--data-path` with `humaneval.jsonl` if desired.
- **License**: CC BY 4.0 (MBPP via HF; see original MBPP release). Code generation with MBPP is for research evaluation.
- **Usage here**: loaded via `datasets.load_dataset("mbpp", "sanitized")` with fallbacks to `mbpp` plain and local `examples/mbpp/data/mbpp.jsonl`. Deterministic shuffle seed 0 → 150 train / 300 val / 300 test (cycled to 750 if needed, like `frontiercs`/`gsm8k`). No data files are committed. First HF load caches to `$HF_HOME` (on della, scratch via sbatch). For offline runs, place JSONL at `examples/mbpp/data/mbpp.jsonl` or pass `--data-path`.

## Papers referenced by this benchmark

- **GEPA** — Agrawal et al., 2025. *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning.* arXiv:2507.19457 (ICLR 2026 Oral). Tables 1–2 report GEPA on MBPP-aligned code suites indirectly via LiveBench/AIME; this scaffold mirrors GEPA's optimization loop as in `examples/ifbench` and `examples/gsm8k` (seed candidate, evaluator with `SideInfo`/`execution_feedback`, `EngineConfig`/`GEPAConfig`/`ReflectionConfig`, `ActionDiversityCallback`, parallel evaluation, candidate dumps).
- **CANTANTE** — *Optimizing Agentic Systems via Contrastive Credit Attribution*, arXiv:2605.13295 (May 2026). Reports MBPP 22.96 → **41.89** (**+18.93 pp** vs GEPA) and GSM8K 61.27 → **82.33** (+21 pp vs GEPA) — the largest code delta over GEPA in `docs/BENCHMARKS_GEPA_CITATIONS.md`. This is the primary motivator for the MBPP scaffold (GEPA's biggest code defeat).
- **FlowBot** — *Inducing LLM Workflows with Bilevel Optimization and Textual Gradients*, Wu et al., arXiv:2604.26258 (Apr 2026). Suite A includes HumanEval, MBPP, GSM8K, MATH, DROP, HotPotQA2; reports competitive vs human-crafted workflows at 145K calls vs AFlow 803K, beating GEPA's single-prompt optimization via workflow induction.
- **VISTA / Feedback Descent / PCO / Reward-Free** — See `docs/BENCHMARKS_GEPA_CITATIONS.md` for the full 10-paper beating-GEPA ranking; MBPP and GSM8K are the two benchmarks where GEPA is most flatly beaten outside the multi-step HotpotQA/IFBench/HoVer cluster.

## Code provenance

- `utils.py` `_call_lm`, `_strip_think`, `_extract_final_response`, `_extract_code`, `run_mbpp_single_stage`, `_exec_in_sandbox` (subprocess with 2s timeout, tempfile, capture, heuristic fallback), and `_normalize_code` are adapted from `examples/gsm8k/utils.py` and `examples/ifbench/utils.py` to keep solver behaviour identical across benchmarks (temp 0.6, top_p 0.95, top_k 20, max 16384, `enable_thinking: False`, truncation, `ContextWindowExceededError` retries).
- `main.py` mirrors `examples/gsm8k/main.py` and `examples/ifbench/main.py`: conditions `vanilla`/`random`/`action`/`all`, `seed_style` plain/structured, `actions` default/structured (`DEFAULT_ACTIONS` vs `build_structured_actions`), `ActionDiversityCallback`, `RandomActionSelector`/`VerbalizedActionSelector`, `--data-path` + `--train`/`--val`/`--test-limit`, dumps `candidates.json`/`action_summary.json`, final report with test accuracy and diversity. The only domain difference is code extraction (` ```python `) and execution-based scoring vs GSM8K's numeric normalization.
- `run_mbpp.sbatch` mirrors `examples/gsm8k/run_gsm8k.sbatch` and `examples/ifbench/run_ifbench.sbatch` (48h, vLLM serve via POSIT, env vars `MODEL`/`CONDITION`/`SEED_STYLE`/`ACTIONS`/`MAX_METRIC_CALLS`/`TAG`, HF cache on scratch).
- No vendored test harness is committed; sandbox is self-contained via `subprocess` + `tempfile` so the benchmark is runnable without Docker (unlike TerminalBench's full harness). For strict replication, swap `mbpp_metric`'s sandbox with the official MBPP evaluation script or HumanEval's `human_eval` package.

## License notes

- MBPP code problems are for research use under CC BY 4.0. Reference solutions in HF `mbpp` remain under the original MBPP authors' license. This scaffold uses only the problem statements and tests for optimization, not the reference solutions as candidates.
