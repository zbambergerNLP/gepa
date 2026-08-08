# FrontierBench: Action-Conditioned Reflection Experiments

This example evaluates whether **action-conditioned reflection** (constraining each GEPA mutation to a typed edit action) and **verbalized sampling** (letting the reflection LM propose a probability distribution over actions, then sampling its tails) improve GEPA on **FrontierBench** (https://github.com/laude-institute/frontier-bench), the harder agentic research benchmark from the Terminal-Bench authors (Laude Institute).

FrontierBench tasks are end-to-end research assignments — literature, code, analysis — that require an agent to plan and execute in a sandbox (Terminal-Bench lineage). The setup here mirrors `examples/ifbench`, `examples/pupa`, and `examples/frontiercs`: a 2-stage program (research plan → execution) and a 1-stage ablation, with the same GEPA harness, action space, and decoding config. See `ATTRIBUTION.md` for provenance.

## Layout

| File | Purpose |
|---|---|
| `main.py` | Experiment runner: conditions, programs, seeds, dumps, final report |
| `utils.py` | Dataset loading (HF `laude/frontier-bench` or local `data/frontierbench.jsonl` fallback + synthetic offline), 1- and 2-stage LM programs, task-success metric |
| `run_frontierbench.sbatch` | SLURM job for della (serves Qwen via vLLM, runs one experiment) |
| `ATTRIBUTION.md` | Data/code provenance |
| `README.md` | This file |
| `data/` | Optional local JSONL fallback (`frontierbench.jsonl`, one task per line) |

## Dataset & Splits

- **Primary**: HuggingFace `laude/frontier-bench` via `datasets.load_dataset` (harder agentic research tasks, Terminal-Bench lineage; the repo reports tasks that require multi-step tool use and are scored by test harnesses).
- **Fallbacks**: local `data/frontierbench.jsonl` if present, otherwise a deterministic synthetic 90-example pool (10 stems × 9 cycles) so smoke/tests work offline.
- **Splits**: deterministic shuffle (seed 0) then **30 train / 30 val / 30 test** (train 0:30, val 30:60, test 60:90), noting the FrontierBench scale. Mirrors IFBench's slicing convention but at the smaller scale appropriate for this benchmark. Override with `--train-limit` / `--val-limit` / `--test-limit` or `--data-path`.

JSONL schema (one record per line, fields flexible — see `utils._normalize_record`):
```json
{"id": "fb_001", "task": "Reproduce the ...", "category": "Data Analysis", "difficulty": "hard", "tests": ["Output contains methodology", "Output includes concrete results"]}
```
`tests` / `success_criteria` may be a string or a list; missing fields get sensible defaults. A short smoke file (e.g., 20 lines) is automatically cycled to 90 so 30/30/30 holds offline.

## Setup

```bash
uv sync --extra dev
# FrontierBench uses datasets + litellm (already in full/dev)
```

No vendored harness is needed for infra-only (the real FrontierBench test suite runs in Docker/Terminal-Bench; here we use an LLM-judge approximation plus a heuristic fallback so the benchmark is runnable without the full harness). First HF load caches to `$HF_HOME` (on della, scratch via the sbatch). For offline runs, place a JSONL at `examples/frontierbench/data/frontierbench.jsonl` or pass `--data-path`.

## Running

```bash
uv run python examples/frontierbench/main.py \
    --condition all            # vanilla | random | action | all \
    --program 2stage           # 2stage (plan-then-execute) | 1stage (single execution) \
    --seed-style plain         # plain (seed sentences) | structured (markdown Role/Task/Rules/Format/Examples) \
    --actions default          # default (6 generic actions) | structured (16 section-scoped) \
    --max-metric-calls 4000    # stretch default (3000-5000); use 15000 for Wave B scale \
    --solver-model hosted_vllm/Qwen3.5-9B --api-base http://localhost:8000/v1 \
    --tag frontierbench_rev1   # suffix for output dirs
```

- `vanilla` is stock GEPA reflection; `random` picks actions uniformly; `action` uses `VerbalizedActionSelector`.
- `--actions structured` implies `--seed-style structured`.
- Mini runs: `--train-limit 15 --val-limit 15 --test-limit 15 --max-metric-calls 150`.
- With a local file: `--data-path examples/frontierbench/data/frontierbench.jsonl`.
- Judge: by default the solver model judges its own outputs via `frontierbench_metric` (per-test `PASS`/`FAIL` + `SCORE:`); pass `--judge-model hosted_vllm/Qwen3.5-9B` to use a different judge.

On della, submit via sbatch:

```bash
MODEL=Qwen3.5-9B CONDITION=action PROGRAM=2stage SEED_STYLE=plain ACTIONS=default MAX_METRIC_CALLS=4000 TAG=frontierbench_rev1 \
    sbatch examples/frontierbench/run_frontierbench.sbatch
# Mini test:
# sbatch --export=ALL,TRAIN_LIMIT=15,VAL_LIMIT=15,TEST_LIMIT=15,MAX_METRIC_CALLS=150 --time=02:00:00 examples/frontierbench/run_frontierbench.sbatch
# With local data:
# sbatch --export=ALL,DATA_PATH=examples/frontierbench/data/frontierbench.jsonl examples/frontierbench/run_frontierbench.sbatch
```

## Artifacts per run (`outputs/<run_dir>/`)

- `candidates.json`: every accepted candidate with lineage (`parents`), val scores, discovery eval counts
- `action_summary.json`: per-action proposal/accept counts, plus the verbalized selector's full distribution history (`probs`, `sampled`, `fallback` per call)
- `run_log.txt`: every proposal including rejected ones, with minibatch decisions
- `candidate_tree.html`: interactive candidate tree (open in a browser)

## Metric details

`frontierbench_metric` in `utils.py` implements task success (0-1):

- **With a judge model** (default: solver model): the judge is prompted with the task, success criteria/tests, and output and asked to emit per-criterion `PASS`/`FAIL` plus a final `SCORE: <0-1>` (fraction passing). The score is the mean pass rate, blended 70/30 with the holistic `SCORE:` when both are available. Feedback is the judge's full trace — this is what GEPA's reflection sees. This approximates the real Frontier-Bench test-suite pass without requiring the Docker harness (infra-only stretch).
- **Offline fallback** (no judge or judge call fails): heuristic keyword overlap per test plus a structure/length signal, still 0-1 with per-test `[PASS]/[FAIL]` feedback.

Decoding is identical to `examples/ifbench/utils.py`: `temperature=0.6, top_p=0.95, top_k=20, max_tokens=16384, enable_thinking=False` (with `COT_FORMAT_INSTRUCTION` and `<think>` stripping, plus truncation retries 16384→4096→1024→256).

## Programs

- **2-stage** (`research_plan` → `execute_task`): stage 1 produces a detailed research plan (methodology, tools, validation); stage 2 executes the task conditioned on the task plus the plan (capped to 12k chars). Returns `(plan, output)`; only the output is scored.
- **1-stage** (`execute_task`): single turn — the whole task in one call.

## Known pitfalls

- **Real harness vs. judge**: the true Frontier-Bench evaluation runs tasks in a Terminal-Bench Docker harness with test scripts; this infra-only version uses an LLM-judge approximation so it is runnable without Docker and without modifying other examples. For a full-harness integration, replace `frontierbench_metric`'s judge path with the harness call (see `ATTRIBUTION.md`).
- **Long prompts**: section-scoped structured prompts can accrete; `SOFT_PROMPT_CHAR_BUDGET` / `MAX_PROPOSAL_CHARS` and the per-stage caps keep the 32k context feasible. Keep `enable_thinking: False`.
- **HF cache on della**: `run_frontierbench.sbatch` sets `HF_HOME` to scratch and leaves `HF_HUB_OFFLINE=0`; subsequent runs hit cache. For offline, use `--data-path`.
