# AppWorld: Action-Conditioned Reflection Experiments

This example evaluates whether **action-conditioned reflection** (constraining each GEPA mutation to a typed edit action) and **verbalized sampling** (letting the reflection LM propose a probability distribution over actions, then sampling its tails) improve GEPA on **AppWorld** (Trivedi et al. 2024, https://appworld.dev — 9 everyday apps, 168 tool APIs, 750 tasks).

AppWorld is an agentic benchmark where an LLM must interact with simulated apps (email, calendar, banking, shopping, etc.) via tool calls to complete everyday tasks (e.g., scheduling, purchasing, messaging). Each of the 750 tasks has evaluation code that checks whether all subtasks pass; the task counts as solved only when every check succeeds (task goal completion, TGC). The setup mirrors other benchmarks in this repo (IFBench, PUPA) and GEPA paper conventions.

## Layout

| File | Purpose |
|---|---|
| `main.py` | Experiment runner: conditions, programs, seeds, dumps, final report |
| `utils.py` | Dataset loading (HF + local fallback), 1- and 2-stage LM programs, metric |
| `run_appworld.sbatch` | SLURM job for della (serves Qwen via vLLM, runs one experiment) |
| `ATTRIBUTION.md` | Data/code provenance and license |
| `README.md` | This file |

## Setup

```bash
uv sync --extra dev
# AppWorld uses datasets + litellm (already in the base env).
# Optional: pre-fetch HF data so compute nodes can run offline
python -c "from datasets import load_dataset; load_dataset('appworld/appworld')"
```

Offline fallback: place JSONL files under `examples/appworld/data/` (e.g. `appworld_train.jsonl`, or any `*.jsonl`/`*.json`); the loader will pool and split them. If neither HF nor local data is found, a small synthetic placeholder keeps the harness runnable for infra testing.

## Running

```bash
uv run python -m examples.appworld.main \
    --condition all            # vanilla | random | action | all \
    --program 1stage           # 1stage (skill-based agent) | 2stage (plan then execute) \
    --seed-style plain         # plain | structured (markdown skeleton) \
    --actions default          # default (6 generic actions) | structured (16 section-scoped) \
    --max-metric-calls 4000    # MIPRO-style budget: 3000 light / 4000 default / 5000 heavy \
    --solver-model hosted_vllm/Qwen3-8B --api-base http://localhost:8000/v1 \
    --data-path /path/to/data  # optional: file or dir of JSONL/JSON; else HF/local auto \
    --tag rev1
```

Conditions: `vanilla` is stock GEPA reflection; `random` picks actions uniformly (baseline); `action` uses `VerbalizedActionSelector`. `--actions structured` implies structured seeds. Mini runs: `--train-limit/--val-limit/--test-limit` (e.g. `20/12/20`) or `50/50/remaining` like the task spec's alternative split.

Budget: default 4000 metric calls mirrors the paper's MIPRO budget style (roughly between MIPRO-light 3000 and MIPRO-heavy ~5000/3593). Override with `--max-metric-calls`.

On della, submit via sbatch (mirrors `ifbench`/`pupa` runners):

```bash
MODEL=Qwen3-8B CONDITION=action PROGRAM=1stage SEED_STYLE=plain ACTIONS=default TAG=rev1 MAX_METRIC_CALLS=4000 \
    sbatch examples/appworld/run_appworld.sbatch
# Mini test (20/12/20, 64 calls):
# sbatch --export=ALL,TRAIN_LIMIT=20,VAL_LIMIT=12,TEST_LIMIT=20,MAX_METRIC_CALLS=64 --time=02:00:00 examples/appworld/run_appworld.sbatch
# With a custom data path:
# DATA_PATH=/scratch/gpfs/BSTEWART/$USER/appworld_data sbatch examples/appworld/run_appworld.sbatch
```

## Splits

AppWorld has 750 tasks total. The GEPA harness splits pooled data deterministically (seed 0 shuffle, then slice) into **60 train / 75 val / remaining test** (615 when the full 750 are available), mirroring the repo's ifbench (300/300/294) and pupa (111/111) conventions. The `50/50/remaining` alternative mentioned in the spec is available via `--train-limit 50 --val-limit 50`. When the pool is incomplete (e.g. synthetic fallback of 40), test is the remainder (at least 20 when pooled < 100 in edge cases handled in code).

## Programs

- **1stage** (`--program 1stage`, default): skill-based agent. One optimized `system_prompt` that instructs the model to use the 168 tool APIs to complete the task; one LM call per example.
- **2stage** (`--program 2stage`): plan-then-execute. `plan` prompt produces a high-level plan (subtasks + tool ordering); `execute` prompt carries it out against the task. Two LM calls; final execution output is scored. The planner output is capped before being fed to stage 2 so input + output fits context.

Both programs use the same decoding config as `ifbench`/`pupa`: `temperature=0.6, top_p=0.95, top_k=20, max_tokens=16384, enable_thinking=False`, with truncation retries on `ContextWindowExceededError` and `<think>` stripping.

## Artifacts per run (`outputs/<run_dir>/`)

- `candidates.json`: every accepted candidate with lineage (`parents`), val scores, discovery eval counts
- `action_summary.json`: per-action proposal/accept counts, plus the verbalized selector's full distribution history (`probs`, `sampled`, `fallback` per call)
- `run_log.txt`: every proposal including rejected ones, with minibatch decisions
- `candidate_tree.html`: interactive candidate tree (open in a browser)

## Metric details

`appworld_metric` in `utils.py` implements **task success rate**: `1.0` if all subtasks pass, `0.0` otherwise (AppWorld's task goal completion / TGC — a task is solved only when every evaluation check succeeds). Offline, subtasks are read from any `tests`/`eval`/`subtasks`/`checks`/`evaluation` list in the example, or from `supervisor` code presence, or approximated as non-empty response when no checks are declared. Feedback lists passed vs failed subtasks and the subtask pass rate for reflection. The evaluator closes over `solver_model`/`api_base`/`program` and exposes `task_id` in `SideInfo` when available.

## Known pitfalls

- HF `datasets` caches to scratch on della; `run_appworld.sbatch` sets `HF_HOME` to scratch and leaves `HF_HUB_OFFLINE=0`.
- The 750-task AppWorld evaluation is heavier than IFBench/PUPA per task (tool traces); keep `max_metric_calls` modest (4000) unless running at full scale.
- Action space and seed handling are shared with `ifbench`/`pupa` via `gepa.strategies.action_space` (`DEFAULT_ACTIONS` / `build_structured_actions`).
