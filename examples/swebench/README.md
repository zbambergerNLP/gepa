# SWE-Bench: Action-Conditioned Reflection Experiments

This example evaluates whether **action-conditioned reflection** (constraining each GEPA mutation to a typed edit action) and **verbalized sampling** (letting the reflection LM propose a probability distribution over actions, then sampling its tails) improve GEPA on **SWE-Bench Verified** (Jimenez et al. 2024, https://www.swebench.com, ~2294 Python GitHub issues, Verified 500 via `princeton-nlp/SWE-bench_Verified`).

SWE-Bench tasks require an agent to generate a code patch that resolves a GitHub issue, validated by checking that the patch applies (`git apply`) and the repository's tests pass. This example mirrors the GEPA paper's IFBench evaluation pattern (`examples/ifbench`) but with a code-patch program: a 2-stage **locate-then-fix** pipeline (stage 1: identify files/lines; stage 2: emit the unified diff) optimized with GEPA, plus a **1-stage** single-patch-generation ablation. The metric is a proxy for patch-applies + tests-pass — exact patch match and hunk/file overlap with feedback — since real evaluation requires cloning repos and running tests. See `ATTRIBUTION.md` for provenance.

## Layout

| File | Purpose |
|---|---|
| `main.py` | Experiment runner: conditions, programs, seeds, dumps, final report |
| `utils.py` | Dataset loading (HF `princeton-nlp/SWE-bench_Verified` or `data/swebench_verified.jsonl` fallback), 1- and 2-stage LM programs, proxy metric + feedback, _call_lm identical to ifbench, truncation for long code context |
| `run_swebench.sbatch` | SLURM job for della (serves Qwen via vLLM, runs one experiment, 48h) |
| `ATTRIBUTION.md` | Data/code provenance |
| `README.md` | This file |

## Setup

```bash
uv sync --extra dev
# SWE-Bench uses datasets + litellm (already in full).
```

First dataset load tries HF `princeton-nlp/SWE-bench_Verified`; on failure (offline) it falls back to `examples/swebench/data/swebench_verified.jsonl` if present, else synthetic issues (so the pipeline never crashes offline). Pre-download on a login node for della:

```bash
python -c "from datasets import load_dataset; load_dataset('princeton-nlp/SWE-bench_Verified')"
# Or place a jsonl at examples/swebench/data/swebench_verified.jsonl:
# one JSON per line with at least {"instance_id": "...", "problem_statement": "fix X", "patch": "diff --git ..."}
```

Note: **Verified 500** (`princeton-nlp/SWE-bench_Verified`, test split, 500 curated instances) is the recommended subset for publication. The full SWE-Bench is ~2294 instances across multiple splits. The loader shuffles and re-splits for optimization — hold out test is not the original Verified test unless you set `--test-limit` to cover the full split.

## Running

```bash
uv run python examples/swebench/main.py \
    --condition all            # vanilla | random | action | all \
    --program 2stage           # 2stage (locate-then-fix) | 1stage (single patch gen) \
    --seed-style plain         # plain | structured (markdown skeleton) \
    --actions default          # default (6 generic) | structured (16 section-scoped) \
    --max-metric-calls 5000    # budget per condition (SWE-Bench scale) \
    --solver-model hosted_vllm/Qwen3-8B --api-base http://localhost:8000/v1 \
    --data-path examples/swebench/data/swebench_verified.jsonl \
    --tag swe_rev1
```

Conditions: `vanilla` is stock GEPA reflection; `random` picks actions uniformly; `action` uses `VerbalizedActionSelector`. `--actions structured` implies structured seeds. Mini runs: `--train-limit/--val-limit/--test-limit` and `--data-path` + `--seed`.

Splits: default **30/30/30** (90 total) for quick iteration; the loader auto-scales to **100/100/100** when ≥300 instances are available (and notes the full Verified 500 for final reporting). If the source has fewer than 30 instances, splits scale proportionally; synthetic fallback generates 90. Examples:
- 90-task sweep: defaults (30/30/30) or `--train-limit 30 --val-limit 30 --test-limit 30`
- 100/100/100: load ≥300 instances and `--train-limit 100 --val-limit 100 --test-limit 100`
- Full Verified 500 test: `--test-limit 500` (test is held-out scoring; for true Verified test reporting run with `--data-path` pointing to the original test split and evaluate that split as test).

On della, submit via sbatch (mirrors `ifbench` runner):

```bash
MODEL=Qwen3-8B CONDITION=action PROGRAM=2stage SEED_STYLE=plain ACTIONS=default TAG=swe_rev1 MAX_METRIC_CALLS=5000 \
    sbatch examples/swebench/run_swebench.sbatch
# 100/100/100:
# TRAIN_LIMIT=100 VAL_LIMIT=100 TEST_LIMIT=100 MAX_METRIC_CALLS=5000 sbatch --export=ALL,TRAIN_LIMIT=100,VAL_LIMIT=100,TEST_LIMIT=100,MAX_METRIC_CALLS=5000 examples/swebench/run_swebench.sbatch
# Mini test (10/5/5, 150 calls):
# sbatch --export=ALL,TRAIN_LIMIT=10,VAL_LIMIT=5,TEST_LIMIT=5,MAX_METRIC_CALLS=150 --time=02:00:00 examples/swebench/run_swebench.sbatch
# With local data:
# sbatch --export=ALL,DATA_PATH=/path/to/swebench_verified.jsonl examples/swebench/run_swebench.sbatch
```

## Artifacts per run (`outputs/<run_dir>/`)

- `candidates.json`: every accepted candidate with lineage (`parents`), val scores, discovery eval counts
- `action_summary.json`: per-action proposal/accept counts, plus verbalized selector's full distribution history (`probs`, `sampled`, `fallback` per call)
- `run_log.txt`: every proposal including rejected ones, with minibatch decisions
- `candidate_tree.html`: interactive candidate tree (open in a browser)

## Metric details

`swebench_metric` in `utils.py` is a proxy for `patch applies + tests pass` (offline-friendly): checks for unified-diff structure (`diff --git`, `---`, `+++`, `@@`), then when a gold patch is available computes file overlap and hunk line overlap, awarding 1.0 exact match, 0.85 near match, 0.5 partial, 0.25 correct file but wrong hunk, 0.1 well-formed but wrong file, 0.0 malformed. When no gold is present, scores on format alone. Feedback includes all checks plus gold/pred excerpts and a ``` fencing warning (which would break `git apply`). Real SWE-Bench evaluation requires cloning the repo at `base_commit`, applying the patch, and running the issue's tests (see https://www.swebench.com and https://github.com/princeton-nlp/SWE-bench).

## Known pitfalls

- Qwen thinking models: hidden `<think>` blocks can consume the whole token budget, leaving empty `message.content`. The runner disables thinking (`enable_thinking: false`) and falls back to `reasoning_content`; do not remove this.
- Long code context: issue descriptions can include large code blocks. Stage 1/2 inputs are capped at ~24k chars and the location at ~8k before feeding stage 2; `_call_lm` also steps `max_tokens` down on `ContextWindowExceededError`. For production patch generation, consider adding retrieval/RAG over the repo.
- HF `datasets` caches to scratch on della; `run_swebench.sbatch` sets `HF_HOME` to scratch. First fetch needs internet; fallback is local `data/swebench_verified.jsonl` or synthetic issues.
- Patch format: the metric warns on markdown fencing (```) — instruct prompts to emit raw diffs only (the structured seed does this).
