# Attribution

This directory evaluates GEPA on **LiveBench-Math** (White et al. 2025,
https://livebench.ai, arXiv:2408.14596), the contamination-limited
math benchmark used in the GEPA paper (arXiv:2507.19457) and the
GEPA parallel-proposals release (2026-07-30).

- Dataset: **LiveBench-Math**, n=368 math problems (AMC/AIME, symbolic
  algebra, olympiad), contamination-limited (problems post-date most
  model cutoffs, released with timed lock). Primary source is
  HuggingFace `livebench/livebench` (filter `category == "math"` or
  config `math`) and the LiveBench site https://livebench.ai.
  Alternative mirror tried by `utils.py`: `livebench/livebench_math`,
  `livebench/math`. Local artifact fallback:
  `examples/livebench_math/data/livebench_math.jsonl` (each line
  `{"question": ..., "answer": ...}`) or env `LIVEBENCH_DATA`.
  This example does not vendor the data; it downloads on first use
  via `datasets.load_dataset` (like `examples/pupa`) and shuffles
  seed 0 -> 122/123/123 (paper-faithful; Terrarium split
  100/100/168 from the GEPA blog is available via `--splits terrarium`).
  No data files are committed.

- Paper program: GEPA paper (Table 1, Sec 4, App E.1) lists
  LiveBench-Math as a math-reasoning benchmark with exact-match
  scoring via LiveBench scorers. The GEPA parallel-proposals blog
  (2026-07-30) describes the Terrarium harness: single-step CoT
  with `gpt-4.1-mini` solver, 100/100/168 splits, 5,000 metric-call
  budget (this example reproduces the same single-step program and
  exact-match metric with the paper's 1839 budget and the new
  122/123/123 splits; use `--splits terrarium --max-metric-calls 5000`
  to match the blog).

- Metric: `livebench_metric` in `utils.py` implements exact-match
  accuracy after answer normalization (strip `<think>`, extract
  `\boxed{}`, remove `Answer:` prefix, lowercase, whitespace collapse,
  numeric epsilon 1e-6 and fraction handling `1/2` vs `0.5`) with
  feedback `"Your answer is correct/incorrect..."` for reflection.
  This ports the LiveBench scorer intent without requiring the
  official LiveBench scorer package; for strict replication, swap in
  `livebench`'s own `math_scorer`.

- Code provenance: `utils.py` `_call_lm` and decoding config
  (temp 0.6, top_p 0.95, top_k 20, max 16384,
  `enable_thinking: False`, context-window stepping) are copied from
  `examples/ifbench/utils.py` to keep solver behaviour identical
  across benchmarks. `main.py` reuses the same `GEPAConfig` /
  `ActionDiversityCallback` / `DEFAULT_ACTIONS` /
  `VerbalizedActionSelector` pattern as `examples/ifbench/main.py`
  and `examples/pupa/main.py`, on the scaled base (IFBench 15k,
  upstream 8a2bed96 parallel proposals + OA refactor).

- LiveBench citation: White et al., "LiveBench: A Challenging,
  Contamination-Limited LLM Benchmark", 2024 (updated 2025).
  Copyright for LiveBench problems remains with the original
  contest sources; this example uses them under the LiveBench
  distribution for research evaluation.
