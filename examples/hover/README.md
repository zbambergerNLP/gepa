# HoVer: Action-Conditioned Reflection Experiments

This example evaluates whether **action-conditioned reflection** (constraining each GEPA mutation to a typed edit action) and **verbalized sampling** (letting the reflection LM propose a probability distribution over actions, then sampling its tails) improve GEPA on **HoVer** (Jiang et al. 2020, many-hop fact extraction & claim verification, up to 3 hops), the multi-hop retrieval benchmark from the GEPA paper.

The setup mirrors the paper's HoVer configuration: a 2-stage program (`query_writer` -> `doc_summarizer`) whose two prompts are optimized on 150 train claims, with 300 val claims for Pareto selection and 300 test claims held out for final scoring. The metric is **gold-doc retrieval F1/recall** (precision/recall/F1 over supporting Wikipedia titles), and the default budget of **7,051 metric calls** matches the paper. See `ATTRIBUTION.md` for provenance.

## Layout

| File | Purpose |
|---|---|
| `main.py` | Experiment runner: conditions, programs, seeds, dumps, final report |
| `utils.py` | Dataset loading (HF `hover` or raw GitHub `hover-nlp/hover`, auto-download), 2-stage LM program, gold-doc F1/recall metric |
| `run_hover.sbatch` | SLURM job for della (serves Qwen via vLLM, runs one experiment) |
| `ATTRIBUTION.md` | Data/code provenance |
| `README.md` | This file |

## Setup

```bash
uv sync --extra dev
# HoVer uses datasets + litellm (already in full)
```

First dataset load tries HuggingFace `datasets` (`hover`); if offline it falls back to raw GitHub artifact (`https://raw.githubusercontent.com/hover-nlp/hover/main/data`). Splits are 150 train / 300 val / 300 test (shuffled seed 0), matching the paper's 150/300/300 intent (up to 3 hops). Synthetic fallback is used only if no data is found offline.

## Running

```bash
uv run python examples/hover/main.py \
    --condition all            # vanilla | random | action | all \
    --program 2stage           # 2stage (paper protocol) | 1stage (ablation) \
    --seed-style plain         # plain (paper seed sentences) | structured (markdown skeleton) \
    --actions default          # default (6 generic actions) | structured (16 section-scoped) \
    --max-metric-calls 7051    # paper budget \
    --solver-model hosted_vllm/Qwen3.5-9B --api-base http://localhost:8000/v1 \
    --tag hover_rev1           # suffix for output dirs
```

Conditions: `vanilla` is stock GEPA reflection; `random` picks actions uniformly; `action` uses `VerbalizedActionSelector`. `--actions structured` implies structured seeds. Mini runs: `--train-limit/--val-limit/--test-limit`.

On della, submit via the wrapper (like `ifbench`/`pupa`):

```bash
MODEL=Qwen3.5-9B CONDITION=action SEED_STYLE=plain ACTIONS=default TAG=hover_rev1 MAX_METRIC_CALLS=7051 \
    sbatch examples/hover/run_hover.sbatch
# Mini test (20/20/20, 150 calls):
# sbatch --export=ALL,TRAIN_LIMIT=20,VAL_LIMIT=20,TEST_LIMIT=20,MAX_METRIC_CALLS=150 --time=02:00:00 examples/hover/run_hover.sbatch
# Scaled 48h run (like ifbench 48h):
# sbatch --time=48:00:00 --export=ALL,MAX_METRIC_CALLS=7051 examples/hover/run_hover.sbatch
```

## Artifacts per run (`outputs/<run_dir>/`)

- `candidates.json`: every accepted candidate with lineage (`parents`), val scores, discovery eval counts
- `action_summary.json`: per-action proposal/accept counts, plus the verbalized selector's full distribution history (`probs`, `sampled`, `fallback` per call)
- `run_log.txt`: every proposal including rejected ones, with minibatch decisions
- `candidate_tree.html`: interactive candidate tree (open in a browser)

## Metric details

`hover_metric` in `utils.py` implements gold-doc retrieval F1/recall: predicted titles are extracted from the final response (JSON list, bullet list, or fallback substring match); precision/recall/F1 are computed over whitespace-normalized, case-insensitive title sets vs gold `supporting_facts` titles. Fallback substring recall is used when no parseable title list is found, so every rollout scores >0 when it mentions a gold title. Feedback lists retrieved / missing / extra and P/R/F1 for reflection. `hover_recall` is the recall component (also substring fallback).

## 2-stage program

- **Stage 1 `query_writer`**: given claim, write search queries to retrieve supporting Wikipedia pages.
- **Stage 2 `doc_summarizer`**: given claim + queries, list the Wikipedia titles that support verification (final retrieval prediction scored by `hover_metric`).

`utils.py` `_call_lm` uses the same decoding config as `ifbench`/`pupa` (temp 0.6, top_p 0.95, top_k 20, max_tokens 16384, `enable_thinking: False`) and the same `COT_FORMAT_INSTRUCTION` / `Final Response:` handling. Stage-2 input is capped so claim + queries + output fits context.

## Known pitfalls

- HF `datasets` caches to scratch on della; `run_hover.sbatch` sets `HF_HOME` to scratch (offline after first fetch, like `ifbench`).
- Title extraction is strict (JSON or bullet list) then substring fallback; to maximize F1, instruct the model to return a JSON list of titles.
- Qwen thinking models: hidden `<think>` blocks can consume the whole token budget; the runner disables thinking and falls back to `reasoning_content`.
