# HotpotQA: Action-Conditioned Reflection Experiments

This example evaluates whether **action-conditioned reflection** (constraining each GEPA mutation to a typed edit action) and **verbalized sampling** (letting the reflection LM propose a probability distribution over actions, then sampling its tails) improve GEPA on **HotpotQA** (hotpot_qa distractor, 113K multi-hop QA, Yang et al. 2018).

The setup replicates the GEPA paper's HotpotQA protocol (arXiv:2507.19457, Table 1): exact HF dataset (`hotpot_qa` distractor via `datasets`), splits **150 train / 300 val / 300 test** (paper Table 1), a **2-stage query-generation program** (`generate_query` then `generate_answer`), the official **token-F1 / EM** metrics with per-example feedback, and the paper's **6871-metric-call budget** (MIPROv2-Heavy). The original **20-example smoke** (`data/hotpotqa_distractor_sample.jsonl`, 14 train / 6 val split smoke; expanded to 14/3/3 for the 3-way pipeline) is kept as an offline fallback. See `ATTRIBUTION.md` for provenance.

Built on top of the scaled IFBench baseline (`rev1_action-conditioned_reflection` — IFBench default scaled 3593→15000). HotpotQA reuses the same action-conditioned machinery, `ActionDiversityCallback`, and decoding config (temp 0.6 / top_p 0.95 / top_k 20).

## Layout

| File | Purpose |
|---|---|
| `main.py` | Experiment runner: conditions, programs, seeds, dumps, final report |
| `utils.py` | Dataset loading (HF `hotpot_qa` distractor, auto-download; smoke fallback), 1- and 2-stage LM programs, HotpotQA metrics |
| `data/hotpotqa_distractor_sample.jsonl` | 20-example smoke sample (kept for offline / CI) |
| `run_hotpotqa.sbatch` | SLURM job for della (serves Qwen via vLLM, runs one experiment) |
| `ATTRIBUTION.md` | Data/code provenance |
| `README.md` | This file |

## Setup

```bash
uv sync --extra dev
# HotpotQA uses datasets + litellm (already in full)
```

No committed HF data files (113K via `datasets`). The smoke sample is bundled for offline runs.

## Running

```bash
# Paper-faithful (150/300/300, 6871 calls, 2-stage)
uv run python examples/hotpotqa/main.py \
    --condition all            # vanilla | random | action | all (both = all) \
    --program 2stage           # 2stage (query-generation paper protocol) | 1stage (ablation) \
    --seed-style plain         # plain (paper sentences) | structured (markdown skeleton) \
    --actions default          # default (6 generic actions) | structured (16 section-scoped) \
    --max-metric-calls 6871    # paper budget; 200 = smoke, 15000 = Wave B scale \
    --solver-model hosted_vllm/Qwen3.5-9B --api-base http://localhost:8000/v1 \
    --tag hotpot_rev1

# Smoke (20 ex, ~14/3/3, 200 calls, fast local test)
uv run python examples/hotpotqa/main.py \
    --data-path examples/hotpotqa/data/hotpotqa_distractor_sample.jsonl \
    --condition both --max-metric-calls 200 --program 2stage \
    --solver-model hosted_vllm/Qwen3.5-9B --api-base http://localhost:8000/v1

# Mini limits for debugging
uv run python examples/hotpotqa/main.py --train-limit 20 --val-limit 12 --test-limit 20 --max-metric-calls 64 --condition vanilla
```

Conditions: `vanilla` is stock GEPA reflection; `random` picks actions uniformly (baseline); `action` uses `VerbalizedActionSelector`. `--actions structured` implies `--seed-style structured`. The runner logs per-action proposal/acceptance stats via `ActionDiversityCallback` for `random`/`action`.

On della, submit via sbatch (mirrors `ifbench`/`pupa` runners):

```bash
MODEL=Qwen3.5-9B CONDITION=action PROGRAM=2stage SEED_STYLE=plain ACTIONS=default MAX_METRIC_CALLS=6871 TAG=hotpot_rev1 \
    sbatch examples/hotpotqa/run_hotpotqa.sbatch
# Smoke on della:
# MODEL=Qwen3.5-9B CONDITION=both MAX_METRIC_CALLS=200 DATA_PATH=examples/hotpotqa/data/hotpotqa_distractor_sample.jsonl sbatch examples/hotpotqa/run_hotpotqa.sbatch
```

## Artifacts per run (`outputs/<run_dir>/`)

- `candidates.json`: every accepted candidate with lineage (`parents`), val scores, discovery eval counts
- `action_summary.json`: per-action proposal/accept counts, plus the verbalized selector's full distribution history (`probs`, `sampled`, `fallback` per call)
- `run_log.txt`: every proposal including rejected ones, with minibatch decisions
- `candidate_tree.html`: interactive candidate tree (open in a browser)

## Metric details

`utils.py` implements the official HotpotQA metrics (ported from `hotpot_evaluate_v1.py`):
- `normalize_answer` → lowercasing, punctuation/ article stripping
- `f1_score` → token-overlap F1 (primary score, 0–1)
- `em_score` → exact match after normalization (0/1)
- `hotpotqa_metric` → F1 as score, feedback includes both F1 and EM, partial-credit message

`_call_lm` uses the paper's decoding config for Qwen3 (temp 0.6, top_p 0.95, top_k 20, max_tokens 16384, `enable_thinking: False`), shared with IFBench/PUPA, with truncation and `ContextWindowExceededError` retries.

## Program details

- **2-stage (paper protocol)**: `generate_query` produces a concise second-hop query; `generate_answer` answers the multi-hop question using `Context + Question + Search query`. Both prompts are optimized (round-robin).
- **1-stage (ablation)**: single `answer_question` prompt, one LM call.

Seed sentences preserve the original 20-ex smoke's initial prompt verbatim in the `Task` section when `--seed-style structured`.

## Known pitfalls

- HF `datasets` caches to scratch on della; `run_hotpotqa.sbatch` sets `HF_HOME` to scratch (needs net on first fetch; offline falls back to cycled smoke for len-check tests).
- Qwen thinking models: hidden `<think>` blocks can consume token budget. The runner disables thinking (`enable_thinking: false`) and falls back to `reasoning_content`; do not remove this.
- Stage 2's input includes stage 1's generated query; the runner caps query/context so input + `max_tokens` fits the model context (32k for Qwen3).
