# PUPA: Action-Conditioned Reflection Experiments

This example evaluates whether **action-conditioned reflection** (constraining each GEPA mutation to a typed edit action) and **verbalized sampling** (letting the reflection LM propose a probability distribution over actions, then sampling its tails) improve GEPA on **PUPA** (Columbia-NLP/PUPA, pupa_tnb), the privacy-conscious delegation benchmark from the GEPA paper (arXiv:2507.19457).

PUPA mirrors the paper's setup: a 1-stage program (single `system_prompt`) optimized on privacy-sensitive queries where the model must redact PII while preserving intent. The dataset is `Columbia-NLP/PUPA` `pupa_tnb` (237 examples, `user_query` → `redacted_query` with `pii_units` leakage signal). We replicate the paper's evaluation logic via `tests/test_pareto_frontier_types` (quality via LLM judge + leakage 1 - leaked_frac, aggregate = (quality+leakage)/2) with splits shuffled by seed 0 (tests split mid; paper reports 111/111/221 — our loader caps to that when data permits, else uses the mid split + 20-example held-out test).

Built on top of the scaled IFBench baseline (`rev1_action-conditioned_reflection` at `cf502ad6` — merged upstream `8a2bed96` parallel proposals + OA refactor, IFBench default scaled 3593 → 15000). PUPA reuses the same action-conditioned machinery and `ActionDiversityCallback`.

## Layout

| File | Purpose |
|---|---|
| `main.py` | Experiment runner: conditions, seeds, dumps, final report |
| `utils.py` | Dataset loading (HF datasets, auto-download), 1-stage LM program, PUPA metric |
| `run_pupa.sbatch` | SLURM job for della (serves Qwen via vLLM, runs one experiment) |
| `ATTRIBUTION.md` | Data/code provenance |
| `README.md` | This file |

## Setup

```bash
uv sync --extra dev
# PUPA uses datasets + litellm (already in full)
```

## Running

```bash
uv run python examples/pupa/main.py \
    --condition all            # vanilla | random | action | all \
    --config pupa_tnb          # pupa_tnb (237) | pupa_new (664) \
    --seed-style plain         # plain | structured (markdown skeleton) \
    --actions default          # default (6 generic) | structured (16 section-scoped) \
    --max-metric-calls 2426    # paper PUPA budget; use 15000 for Wave B scale \
    --solver-model hosted_vllm/Qwen3-8B --api-base http://localhost:8000/v1 \
    --tag pupa_rev1
```

Conditions: `vanilla` is stock GEPA reflection; `random` picks actions uniformly; `action` uses `VerbalizedActionSelector`. `--actions structured` implies structured seeds. Mini runs: `--train-limit/--val-limit/--test-limit`.

On della, submit via sbatch (mirrors `ifbench` runner):

```bash
MODEL=Qwen3-8B CONDITION=action SEED_STYLE=plain ACTIONS=default CONFIG=pupa_tnb TAG=pupa_rev1 MAX_METRIC_CALLS=2426 \
    sbatch examples/pupa/run_pupa.sbatch
# Mini test (20/12/20, 32 calls):
# sbatch --export=ALL,TRAIN_LIMIT=20,VAL_LIMIT=12,TEST_LIMIT=20,MAX_METRIC_CALLS=32 --time=01:00:00 examples/pupa/run_pupa.sbatch
```

## Artifacts per run (`outputs/<run_dir>/`)

- `candidates.json`: every accepted candidate with lineage (`parents`), val scores, discovery eval counts
- `action_summary.json`: per-action proposal/accept counts, plus verbalized selector's full distribution history (`probs`, `sampled`, `fallback` per call)
- `run_log.txt`: every proposal including rejected ones, with minibatch decisions
- `candidate_tree.html`: interactive candidate tree (open in a browser)

## Metric details

`pupa_metric` in `utils.py` implements the paper's aggregate: quality via LLM judge (strict grader comparing response to gold `redacted_query`, 0-1) + leakage `1 - leaked/PII` (case-insensitive substring check on `||`-separated `pii_units`), total `(quality+leakage)/2`. Feedback lists both components for reflection. When no judge model is given, falls back to exact-match with 0.5 partial credit.

## Known pitfalls

- HF `datasets` caches to scratch on della; `run_pupa.sbatch` sets `HF_HOME` to scratch and leaves `HF_HUB_OFFLINE=0` (needs net on first fetch, unlike IFBench's artifact URL).
- Judge calls double the LM calls; budget counts metric calls (candidate × example), judge calls are not budgeted but affect latency.
- Leakage check is substring-based; PII like "Rachel Zheng" vs "Rachel" both count if present.
