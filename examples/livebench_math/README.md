# LiveBench-Math: Action-Conditioned Reflection Experiments

This example evaluates whether **action-conditioned reflection** (constraining each GEPA mutation to a typed edit action) and **verbalized sampling** (letting the reflection LM propose a probability distribution over actions, then sampling its tails) improve GEPA on **LiveBench-Math** (White et al. 2025, n=368 math, contamination-limited, https://livebench.ai), the fresh competition-math benchmark from the GEPA paper (arXiv:2507.19457) and the GEPA parallel-proposals release.

LiveBench-Math collects fresh problems (AMC/AIME, symbolic algebra, olympiad) beyond most model cutoffs, graded by LiveBench's own scorers. The optimization program is **single-step CoT** (one optimized `instruction`, one LM call), the metric is **exact-match accuracy** after answer normalization (handling `\boxed{}`, fractions, numeric tolerance), and the dataset is split **122 train / 123 val / 123 test** by seed-0 shuffle (368 / 3). The Terrarium split 100/100/168 from the GEPA parallel-proposals blog is available via `--splits terrarium`. Default budget **1839 metric calls** matches the paper's LiveBench-Math budget (the parallel-proposals release used 5000; Wave B scales to 15000 like IFBench).

Built on top of the scaled IFBench baseline (`rev cf502ad6` → 15k IFBench, merged upstream `8a2bed96` parallel proposals + OA refactor). LiveBench-Math reuses the same `ActionDiversityCallback` and `GEPAConfig` pattern as `examples/ifbench` and `examples/pupa`.

## Layout

| File | Purpose |
|---|---|
| `main.py` | Experiment runner: vanilla/random/action conditions, seeds, dumps, final report |
| `utils.py` | Dataset loading (HF `livebench/livebench` filtered to math or local `data/livebench_math.jsonl`, seed-0 splits 122/123/123), single-step LM program (`_call_lm` temp 0.6/top_p 0.95/top_k 20 like IFBench), exact-match metric |
| `run_livebench_math.sbatch` | SLURM job for della (serves Qwen via vLLM, runs one experiment) – mirrors `run_ifbench.sbatch` |
| `ATTRIBUTION.md` | Data/code provenance |
| `README.md` | This file |

## Setup

```bash
uv sync --extra dev
# LiveBench-Math uses datasets + litellm (already in full extra)
# First run downloads from HF (livebench/livebench) or use a local artifact:
#   mkdir -p examples/livebench_math/data
#   # place livebench_math.jsonl there (each line: {"question": ..., "answer": ...})
# Optionally set LIVEBENCH_DATA=/path/to/jsonl or HF_HOME to scratch on della
```

## Running

```bash
uv run python examples/livebench_math/main.py \
    --condition all            # vanilla | random | action | all \
    --splits paper             # paper 122/123/123 | terrarium 100/100/168 \
    --seed-style plain         # plain | structured (markdown skeleton) \
    --actions default          # default (6 generic) | structured (16 section-scoped) \
    --max-metric-calls 1839    # paper budget; 5000 for parallel-proposals; 15000 for Wave B \
    --solver-model hosted_vllm/Qwen3.5-9B --api-base http://localhost:8000/v1 \
    --tag livebench_rev1
```

Conditions: `vanilla` is stock GEPA reflection; `random` picks actions uniformly; `action` uses `VerbalizedActionSelector`. `--actions structured` implies structured seeds. Mini runs: `--train-limit/--val-limit/--test-limit`. `--splits terrarium` reproduces the GEPA blog split (100/100/168) instead of the paper 122/123/123.

On della, submit via sbatch (mirrors `ifbench` runner):

```bash
MODEL=Qwen3.5-9B CONDITION=action SEED_STYLE=plain ACTIONS=default SPLITS=paper TAG=livebench_rev1 MAX_METRIC_CALLS=1839 \
    sbatch examples/livebench_math/run_livebench_math.sbatch
# Mini test (20/20/20, 32 calls):
# sbatch --export=ALL,TRAIN_LIMIT=20,VAL_LIMIT=20,TEST_LIMIT=20,MAX_METRIC_CALLS=32 --time=01:00:00 examples/livebench_math/run_livebench_math.sbatch
```

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

`livebench_metric` in `utils.py` implements exact-match accuracy after normalization: strips `<think>` blocks, extracts `\boxed{}` if present, removes `Answer:` prefixes, collapses whitespace, lowercases, and handles numeric tolerance (float epsilon 1e-6, fraction `1/2` vs `0.5`). Feedback `"Your answer is correct/incorrect. The correct answer is '...'."` is surfaced to the reflection LM. The LM program `run_livebench_single_stage` is a single system-prompt call with `Final Answer:` marker extraction (like IFBench/PUPA).

## Known pitfalls

- LiveBench HF dataset location has shifted; `utils.py` tries `livebench/livebench` (filtered to math), `livebench/livebench_math`, and local `data/livebench_math.jsonl` in order, with a synthetic 368-item fallback for offline CI – real evaluation requires HF or the artifact.
- Decoding config is `temp 0.6 / top_p 0.95 / top_k 20 / max 16384 / enable_thinking False` (copied from IFBench) to keep solver behaviour identical across benchmarks.
- Like IFBench, long inputs are capped and the output budget steps down (16384→4096→1024→256) on context overflow.
