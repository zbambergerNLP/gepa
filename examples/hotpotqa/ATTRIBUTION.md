# Attribution

This directory vendors the HotpotQA evaluation setup used in the GEPA paper
(arXiv:2507.19457, Table 1), taken from the public HuggingFace dataset and the
official HotpotQA evaluation script. The original 20-example smoke sample is kept
for offline/CI.

- Dataset: https://huggingface.co/datasets/hotpot_qa (`distractor`, 90,447 train /
  7,405 validation; `fullwiki` is 113K raw). Each item has `id`, `question`,
  `answer`, `type` (bridge/comparison), `level` (easy/medium/hard),
  `supporting_facts` (`title`, `sent_id`), and `context` (`title`: [...],
  `sentences`: [[...], ...] — 10 paragraphs, 2 gold + 8 distractors).
  Used here via `datasets.load_dataset("hotpot_qa", "distractor")` with
  deterministic shuffle (seed 0) and paper Table 1 splits: train =
  `train[:150]`, val = `train[150:450]`, test = `validation[:300]` (150/300/300).
  Limits (`--train-limit` etc.) slice further. Offline / missing `datasets`
  falls back to the bundled smoke sample `data/hotpotqa_distractor_sample.jsonl`
  (20 examples, cycled to 150/300/300 for len-check tests). No HF data files are
  committed.

- Smoke sample: `data/hotpotqa_distractor_sample.jsonl` (20 HotpotQA distractor
  examples, 14 train / 6 val originally; now 14/3/3 for the 3-way pipeline so
  both smoke and paper pipelines share the same loader). Kept for offline runs
  and CI; the paper-faithful run downloads from HF.

- Paper program: GEPA paper (Table 1, Sec 4) describes HotpotQA as a 2-stage
  query-generation task (first-hop retrieval with the question, second-hop query
  generation, then answering). `utils.py` `run_two_stage` / `run_hotpotqa_two_stage`
  implements this: stage 1 `generate_query` (concise second-hop query from the
  question) and stage 2 `generate_answer` (multi-hop answering from
  `Context + Question + Search query`). The 1-stage ablation (`answer_question`,
  single system prompt) mirrors the IFBench/PUPA 1-stage variants. Both stages
  use the same `_call_lm` decoding and thinking-disabled handling.

- Metric: `utils.py` ports the official HotpotQA evaluation
  (`hotpot_evaluate_v1.py`, https://hotpotqa.github.io/): `normalize_answer`
  (lowercase, remove punctuation/articles, whitespace normalize), `f1_score`
  (token-overlap F1, primary), `em_score` (normalized exact match), and
  `hotpotqa_metric` (F1 score + F1/EM feedback). This matches the paper's
  reported token-F1/EM and `examples/hotpotqa/utils.py`'s original `f1_score`.

- Decoding: `_call_lm` in `utils.py` uses the paper's Qwen3-8B config from the
  IFBench artifact (`gepa-ai/gepa-artifact` `experiment_configs.py`:
  temp=0.6, top_p=0.95, top_k=20, max_tokens=16384, `enable_thinking: False`),
  shared with `examples/ifbench/utils.py` and `examples/pupa/utils.py` so
  solver behavior is identical across benchmarks. Includes truncation and
  `ContextWindowExceededError` retries.

- Code provenance: `utils.py` `_call_lm`, `_strip_think`,
  `_extract_final_response`, `format_passages` truncation, and `main.py`
  condition/build_config/action-tracking/diversity/dump helpers are copied
  from `examples/ifbench/` and `examples/pupa/` to keep the three benchmarks
  consistent. `main.py` supports `--condition vanilla|random|action`
  (plus `all`/`both` aliases), `--actions default|structured`,
  `--seed-style plain|structured`, `--max-metric-calls 6871` (paper) and
  `ActionDiversityCallback`, like IFBench/PUPA.
