# AIME Math: Action-Conditioned Reflection Experiments

This example evaluates whether **action-conditioned reflection** (constraining each GEPA mutation to a typed edit action) and **verbalized sampling** (letting the reflection LM propose a probability distribution over actions, then sampling its tails) improve GEPA on **AIME** (AI-MO/aimo-validation-aime 2022-2024 → MathArena/aime_2025), the competition-math benchmark from the GEPA paper.

The setup replicates the paper's AIME protocol with splits **45 train / 45 val** (AI-MO 90 problems shuffled seed 0, split half) and **30 test problems expanded 5x → 150 items** (MathArena/aime_2025, repeated 5 times to reduce stochastic decoding variance, reported as 45/45/30x5). The program is **single-step CoT** (one optimized `instruction`, one LM call with `Final Answer:` marker), the metric is **exact integer-match accuracy** with solution-aware feedback, and the default budget is **500 metric calls** (legacy AIME default; scale to 15000 for Wave B diversity study like IFBench, mirroring the scaled base at `cf502ad6` merged upstream `8a2bed96` parallel proposals).

Built on top of the scaled IFBench baseline (`rev cf502ad6` — 15k IFBench, merged upstream `8a2bed96` parallel proposals + OA refactor). AIME reuses the same `ActionDiversityCallback` and `GEPAConfig` pattern as `examples/ifbench` and `examples/pupa`.

## Layout

| File | Purpose |
|---|---|
| `main.py` | Experiment runner: vanilla/random/action conditions, seeds, dumps, final report |
| `utils.py` | Dataset loading (HF AI-MO + MathArena, seed-0 splits 45/45/30x5; local `data/*.jsonl` fallback), single-step LM program (`_call_lm` temp 0.6/top_p 0.95/top_k 20 like IFBench), integer-match metric |
| `README.md` | This file |

## Setup

From the repo root (`gepa/`):

```bash
uv sync --extra dev
# AIME uses datasets + litellm (already in full extra). First dataset load
# downloads AI-MO/aimo-validation-aime and MathArena/aime_2025 from HF.
# Offline: place local artifacts at examples/aime_math/data/aime_2022_2024.jsonl
# and examples/aime_math/data/aime_2025.jsonl
```

Legacy (pre-action) setup used `dspy`; the upgraded example uses plain `litellm` `_call_lm` like IFBench/PUPA (dspy no longer required).

## Running

```bash
uv run python examples/aime_math/main.py \
    --condition all            # vanilla | random | action | all \
    --seed-style plain         # plain | structured (markdown skeleton) \
    --actions default          # default (6 generic) | structured (16 section-scoped) \
    --max-metric-calls 500     # legacy AIME budget; use 15000 for Wave B scale \
    --solver-model hosted_vllm/Qwen3.5-9B --api-base http://localhost:8000/v1 \
    --tag aime_rev1
```

Conditions: `vanilla` is stock GEPA reflection; `random` picks actions uniformly; `action` uses `VerbalizedActionSelector`. `--actions structured` implies structured seeds. Mini runs: `--train-limit/--val-limit/--test-limit` (e.g. 20/20/20 with 32 calls).

On della, mirror `examples/ifbench/run_ifbench.sbatch` (set `MODEL`, `CONDITION`, `MAX_METRIC_CALLS`, `TAG` and point `--solver-model`/`--api-base` to the vLLM endpoint).

## Artifacts per run (`outputs/<run_dir>/`)

- `candidates.json`: every accepted candidate with lineage (`parents`), val scores, discovery eval counts
- `action_summary.json`: per-action proposal/accept counts, plus verbalized selector's full distribution history (`probs`, `sampled`, `fallback` per call)
- `run_log.txt`: every proposal including rejected ones, with minibatch decisions
- `candidate_tree.html`: interactive candidate tree (open in a browser)

Analyze fetched runs locally with `examples/ifbench/analyze_actions.py` (same schema):

```bash
uv run python examples/ifbench/analyze_actions.py outputs/<fetched_dir>
```

## Metric details

`math_metric` in `utils.py` parses the integer after `Final Answer:` (fallback: last integer in full output), scores 1 if it equals the gold integer (MathArena/AI-MO answers are integers), and returns feedback with the correct answer plus the full written solution when available (like the legacy AIME feedback). The LM program `run_math_single_stage` is a single system-prompt call with `Final Answer:` extraction and context-window stepping (16384→4096→1024→256) like IFBench/PUPA.

## Splits (45/45/30x5)

- **Train 45 / Val 45**: AI-MO/aimo-validation-aime covers AIME 2022-2024 (30 per year, 90 total). Shuffled seed 0, split 45/45 exactly when 90 present, else mid-split – matches `tests` style.
- **Test 30x5 = 150**: MathArena/aime_2025 (AIME 2025, 30 problems) expanded 5 times (repeat id 0-4) so each problem is evaluated 5 times stochastically; held-out accuracy is the mean over 150 items (30 unique correctness averaged with reduced variance).

## Known pitfalls

- HF `datasets` caches to scratch on della; set `HF_HOME` to scratch and leave `HF_HUB_OFFLINE=0` on first fetch.
- Like IFBench, `<think>` blocks are stripped and `enable_thinking: False` is forced so Qwen3 reasoning does not consume the budget.
- Integer parsing is strict: the model must emit a single integer after `Final Answer:`; non-parsable outputs score 0 with feedback to fix formatting.
