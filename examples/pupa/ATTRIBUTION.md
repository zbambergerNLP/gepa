# Attribution

This directory vendors the PUPA evaluation setup used in the GEPA paper
(arXiv:2507.19457), taken from the paper's artifact and the public
HuggingFace dataset.

- Dataset: https://huggingface.co/datasets/Columbia-NLP/PUPA (`pupa_tnb`
  237 examples, `pupa_new` 664 examples), configs `pupa_tnb` / `pupa_new`.
  Each item has `user_query`, `redacted_query`, `pii_units` (`||`-separated),
  `predicted_category`, `target_response`, `conversation_hash`.
  Used here via `datasets.load_dataset` with shuffle seed 0; splits mirror
  `tests/test_pareto_frontier_types/test_pareto_frontier_types.py`
  (`init_pupa_dataset`, mid-split, 20-example held-out test when data < 443,
  else paper splits 111/111/221). No data files are committed.

- Paper program: GEPA paper (Table 1, Sec 4, App E.1) describes PUPA/PAPILLON
  as a privacy-conscious delegation task (quality + leakage aggregate,
  (quality+leakage)/2). The paper's compound PAPILLON system is 2-stage
  (query rewriter -> untrusted LLM -> response rewriter); this example
  implements the validated 1-stage ablation from
  `tests/test_evaluation_cache.py` and `tests/test_pareto_frontier_types`
  (single `system_prompt`, LLM judge vs gold + leakage check), which is
  the same harness that validates `DefaultAdapter` caching and frontier types
  in this repo.

- Metric: `pupa_metric` in `utils.py` ports `tests/test_pareto_frontier_types`
  `evaluator` (judge prompt "You are a strict grader...", quality float
  parsed from judge output, leakage via substring `pii_units in response`,
  total `(quality+leakage)/2`, per-objective `quality`/`leakage` in
  `objective_scores`). The judge uses the solver/reflection model via
  `litellm.completion` (like `utils.py` in `ifbench`), with exact-match
  fallback.

- Code provenance: `utils.py` `_call_lm` and decoding config (temp 0.6,
  top_p 0.95, top_k 20, max 16384, `enable_thinking: False`) are copied
  from `examples/ifbench/utils.py` to keep solver behavior identical across
  benchmarks.
