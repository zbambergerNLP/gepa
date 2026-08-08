# TerminalBench: Action-Conditioned Reflection Experiments

This example evaluates whether **action-conditioned reflection** (constraining each GEPA mutation to a typed edit action) and **verbalized sampling** (letting the reflection LM propose a probability distribution over actions, then sampling its tails) improve GEPA on **TerminalBench** (T-Bench, https://terminal-bench.github.io, 2024, 50+ terminal agent tasks, Docker-based).

TerminalBench tasks require an agent to synthesize shell commands/scripts that are validated inside Docker containers via unit tests and exit-code checks. This example mirrors the GEPA paper's IFBench evaluation pattern (`examples/ifbench`) but with a terminal-code-agent program: a 2-stage **plan-then-execute** pipeline (stage 1: plan the approach; stage 2: emit the shell command) optimized with GEPA, plus a **1-stage** single-command-generation ablation. The metric is a proxy for task success — offline-friendly, using shell-validity heuristics and expected-command overlap with feedback — since real Docker evaluation requires containers. See `ATTRIBUTION.md` for provenance.

## Layout

| File | Purpose |
|---|---|
| `main.py` | Experiment runner: conditions, programs, seeds, dumps, final report |
| `utils.py` | Dataset loading (HF `laude/terminal-bench` or `data/terminalbench.jsonl` fallback), 1- and 2-stage LM programs, proxy metric + feedback, _call_lm identical to ifbench |
| `run_terminalbench.sbatch` | SLURM job for della (serves Qwen via vLLM, runs one experiment, 48h) |
| `ATTRIBUTION.md` | Data/code provenance |
| `README.md` | This file |

## Setup

```bash
uv sync --extra dev
# TerminalBench uses datasets + litellm (already in full). No extra system deps.
# Optional: pip install datasets (if not in full)
```

First dataset load tries HF `laude/terminal-bench`; on failure (offline) it falls back to `examples/terminalbench/data/terminalbench.jsonl` if present, else synthetic tasks (so the pipeline never crashes offline). Pre-download on a login node for della:

```bash
python -c "from datasets import load_dataset; load_dataset('laude/terminal-bench')"
# Or place a jsonl at examples/terminalbench/data/terminalbench.jsonl:
# one JSON object per line with at least {"task_id": "...", "prompt": "do X", "expected_commands": "echo hi"}
```

## Running

```bash
uv run python examples/terminalbench/main.py \
    --condition all            # vanilla | random | action | all \
    --program 2stage           # 2stage (plan-then-execute) | 1stage (single command gen) \
    --seed-style plain         # plain | structured (markdown skeleton) \
    --actions default          # default (6 generic) | structured (16 section-scoped) \
    --max-metric-calls 3000    # budget per condition (TerminalBench 50-task scale) \
    --solver-model hosted_vllm/Qwen3-8B --api-base http://localhost:8000/v1 \
    --data-path examples/terminalbench/data/terminalbench.jsonl \
    --tag tb_rev1               # suffix for output dirs
```

Conditions: `vanilla` is stock GEPA reflection; `random` picks actions uniformly; `action` uses `VerbalizedActionSelector`. `--actions structured` implies structured seeds. Mini runs: `--train-limit/--val-limit/--test-limit` and `--data-path` + `--seed`.

Splits: 50 total by default (shuffled seed 0): **20 train / 15 val / 15 test**. If the source has fewer than 50 tasks, splits scale proportionally; synthetic fallback generates 50. Override with `--train-limit/--val-limit/--test-limit`.

On della, submit via sbatch (mirrors `ifbench` runner):

```bash
MODEL=Qwen3-8B CONDITION=action PROGRAM=2stage SEED_STYLE=plain ACTIONS=default TAG=tb_rev1 MAX_METRIC_CALLS=3000 \
    sbatch examples/terminalbench/run_terminalbench.sbatch
# Mini test (10/5/5, 150 calls):
# sbatch --export=ALL,TRAIN_LIMIT=10,VAL_LIMIT=5,TEST_LIMIT=5,MAX_METRIC_CALLS=150 --time=02:00:00 examples/terminalbench/run_terminalbench.sbatch
# With local data:
# sbatch --export=ALL,DATA_PATH=/path/to/terminalbench.jsonl examples/terminalbench/run_terminalbench.sbatch
```

## Artifacts per run (`outputs/<run_dir>/`)

- `candidates.json`: every accepted candidate with lineage (`parents`), val scores, discovery eval counts
- `action_summary.json`: per-action proposal/accept counts, plus verbalized selector's full distribution history (`probs`, `sampled`, `fallback` per call)
- `run_log.txt`: every proposal including rejected ones, with minibatch decisions
- `candidate_tree.html`: interactive candidate tree (open in a browser)

## Metric details

`terminalbench_metric` in `utils.py` is a proxy for Docker/unit-test success (offline-friendly): empty check, shell-validity heuristics (contains shell tokens like `|`, `>`, `&&`, `ls`, `grep`), token overlap with `expected_commands`/`tests` when available, task keyword relevance, and markdown-fencing warning. Returns 0/0.25/0.5/1.0 with feedback listing which checks passed/failed for reflection. Real TerminalBench evaluation requires Docker — replace the proxy with container execution for publication runs (see https://terminal-bench.github.io).

## Known pitfalls

- Qwen thinking models: hidden `<think>` blocks can consume the whole token budget, leaving empty `message.content`. The runner disables thinking (`enable_thinking: false`) and falls back to `reasoning_content`; do not remove this.
- Long plans/commands: stage 1 output is capped at ~24k chars before feeding stage 2 so input + `max_tokens` fits the model context (32k for Qwen3-8B). `_call_lm` also steps `max_tokens` down on `ContextWindowExceededError`.
- HF `datasets` caches to scratch on della; `run_terminalbench.sbatch` sets `HF_HOME` to scratch. First fetch needs internet; fallback is local `data/terminalbench.jsonl` or synthetic tasks.
