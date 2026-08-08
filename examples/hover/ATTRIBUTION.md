# Attribution

This directory implements the HoVer evaluation setup referenced in the GEPA paper,
based on HoVer (Jiang et al. 2020, many-hop fact extraction & claim verification).

- Paper: Jiang et al. 2020, "HoVer: A Dataset for Many-Hop Fact Extraction and Claim
  Verification", https://arxiv.org/abs/2009.07258, https://hover-nlp.github.io/
  (up to 3 hops over 2017 Wikipedia abstracts; claim + supporting facts + label).
  Copyright 2020 the authors; dataset CC BY-SA 4.0 / Wikipedia CC BY-SA.

- Dataset: https://huggingface.co/datasets/hover (HuggingFace `hover`) and raw GitHub
  artifact https://github.com/hover-nlp/hover (`data/hover_train.json` /
  `data/hover_dev.json` etc., `hover_train_release_v1.1.json`). Each item has
  `claim`, `supporting_facts` (list of [title, sentence_id]), `label`
  (SUPPORTED / NOT_SUPPORTED), `num_hops` (2-4, we use up to 3 hops). Used here via
  `datasets.load_dataset("hover")` with shuffle seed 0; fallback is raw GitHub
  download into `data/` (not committed). Splits replicate the paper's intent:
  150 train / 300 val / 300 test (750 total, up to 3 hops), shuffled seed 0.

- Program & metric: The paper's full HoVer system is a 4-hop `gpt-4.1-mini` retrieval
  agent over a BM25 index of 5.2M Wikipedia abstracts; optimization scores each
  rollout by top-5 recall over gold pages (cf. docs/blog/2026-07-30-parallel-proposals).
  This example provides a lightweight offline surrogate: a 2-stage LM program
  (`query_writer` -> `doc_summarizer`) whose final titles are scored by gold-doc
  retrieval F1/recall (precision/recall/F1 over supporting titles), with
  `hover_metric`'s substring fallback for robust scoring. The split sizes
  (150/300/300) and the 2-stage decomposition mirror the paper's HoVer setup;
  the BM25 index is not required for the prompt-optimization loop.

- Code provenance: `utils.py` `_call_lm` and decoding config (temp 0.6, top_p 0.95,
  top_k 20, max 16384, `enable_thinking: False`) and the `Final Response:` /
  `COT_FORMAT_INSTRUCTION` handling are copied from `examples/ifbench/utils.py` and
  `examples/pupa/utils.py` to keep solver behavior identical across benchmarks.
  `main.py` mirrors `examples/ifbench/main.py` / `examples/pupa/main.py` (vanilla /
  random / action, `build_structured_actions`, `ActionDiversityCallback`,
  `dump_candidates` / `dump_action_summary`, `max_metric_calls` 7051).
