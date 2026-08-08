# TauBench: Action-Conditioned Reflection Experiments

This example evaluates whether **action-conditioned reflection** (constraining each GEPA mutation to a typed edit action) and **verbalized sampling** (letting the reflection LM propose a probability distribution over actions, then sampling its tails) improve GEPA on **Tau-bench** (Yao et al. 2024, https://tau-bench.github.io, `tau-bench/tau-bench`), the tool-augmented agent benchmark with **800+ tasks over airline + retail domains**.

Tau-bench tests an LLM agent's ability to call domain tools (airline: `get_user_details`, `get_reservation_details`, `search_direct_flight`, `update_reservation`, …; retail: `get_user_details`, `get_order_details`, `modify_pending_order_items`, …) and produce a correct final response / database state. The setup mirrors the GEPA paper's other benchmarks (IFBench, PUPA, HotpotQA): the same decoding config (temp 0.6 / top_p 0.95 / top_k 20), the same `ActionDiversityCallback`, and the same `EngineConfig`/`GEPAConfig`/`ReflectionConfig` machinery.

Built on top of the scaled IFBench baseline (`rev1_action-conditioned_reflection`). TauBench reuses the same post-hoc analysis (`prompt_diversity`, `dump_candidates`, `dump_action_summary`) and is infra-only (no runs required for the scaffold; the metric and dataset loader are fully functional offline).

## Layout

| File | Purpose |
|---|---|
| `main.py` | Experiment runner: conditions, programs, seeds, dumps, final report |
| `utils.py` | Dataset loading (HF `tau-bench/tau-bench` or `data/taubench.jsonl`; synthetic offline fallback), 1-stage tool-calling + 2-stage plan-then-act LM programs, pass^k / task-success metric |
| `data/taubench.jsonl` | Optional local fallback / smoke sample (20–400 tasks; not committed by default — generated offline or downloaded from HF) |
| `run_taubench.sbatch` | SLURM job for della (serves Qwen via vLLM, runs one experiment; 48 h, mirrors `ifbench`/`pupa`) |
| `ATTRIBUTION.md` | Data/code provenance and local modifications |
| `README.md` | This file |

## Setup

```bash
uv sync --extra dev
# TauBench uses datasets + litellm (already in full). No spacy/nltk extra needed.
```

No committed HF data files (800+ via `datasets`). The loader falls back to `data/taubench.jsonl` if present, else to a synthetic 400-task offline fallback (200 airline + 200 retail) so that `py_compile` and len-check tests pass without network.

First HF fetch (online) downloads `tau-bench/tau-bench` via `datasets.load_dataset("tau-bench/tau-bench")`. On offline della compute nodes, either pre-fetch on a login node with `HF_HUB_OFFLINE=0` or bundle `data/taubench.jsonl`.

## Running

```bash
# Paper-faithful (80/80 airline+retail or 150/300/300 when dataset large, 3500 calls, 2-stage plan-then-act)
uv run python examples/taubench/main.py \
    --condition all            # vanilla | random | action | all (both = all) \
    --program 2stage           # 2stage (plan-then-act) | 1stage (single tool-calling agent) \
    --domain all               # all | airline | retail \
    --seed-style plain         # plain | structured (markdown skeleton) \
    --actions default          # default (6 generic) | structured (16 section-scoped) \
    --max-metric-calls 3500    # spec: 3000-5000; Wave B scale 5000-15000 \
    --solver-model hosted_vllm/Qwen3.5-9B --api-base http://localhost:8000/v1 \
    --tag taubench_rev1

# Offline smoke (20 tasks, fast local test; uses data/taubench.jsonl or synthetic)
uv run python examples/taubench/main.py \
    --data-path examples/taubench/data/taubench.jsonl \
    --condition both --max-metric-calls 200 --program 1stage \
    --solver-model hosted_vllm/Qwen3.5-9B --api-base http://localhost:8000/v1

# Mini limits for debugging
uv run python examples/taubench/main.py --train-limit 20 --val-limit 12 --test-limit 20 --max-metric-calls 64 --condition vanilla --domain airline

# Single domain
uv run python examples/taubench/main.py --domain retail --max-metric-calls 3500 --condition action
```

Conditions: `vanilla` is stock GEPA reflection; `random` picks actions uniformly (baseline); `action` uses `VerbalizedActionSelector`. `--actions structured` implies `--seed-style structured`. The runner logs per-action proposal/acceptance stats via `ActionDiversityCallback` for `random`/`action`.

On della, submit via sbatch (mirrors `ifbench`/`pupa`):

```bash
MODEL=Qwen3.5-9B CONDITION=action PROGRAM=2stage SEED_STYLE=plain ACTIONS=default MAX_METRIC_CALLS=3500 TAG=taubench_rev1 \
    sbatch examples/taubench/run_taubench.sbatch
# Smoke on della:
# MODEL=Qwen3.5-9B CONDITION=both PROGRAM=1stage MAX_METRIC_CALLS=200 DATA_PATH=examples/taubench/data/taubench.jsonl \
#     sbatch examples/taubench/run_taubench.sbatch
```

Env vars consumed by the sbatch (all overridable with `sbatch --export=ALL,VAR=...`):

| Var | Default | Purpose |
|---|---|---|
| `MODEL` | `Qwen3.5-9B` | Generator model (under `MODEL_STORAGE`) |
| `CONDITION` | `all` | `vanilla` / `random` / `action` / `all` |
| `PROGRAM` | `2stage` | `1stage` / `2stage` |
| `SEED_STYLE` | `plain` | `plain` / `structured` |
| `ACTIONS` | `default` | `default` / `structured` |
| `MAX_METRIC_CALLS` | `3500` | Budget per condition |
| `TAG` | *(empty)* | Suffix for `outputs/` run dirs |
| `TRAIN_LIMIT` / `VAL_LIMIT` / `TEST_LIMIT` | *(empty)* | Caps for debugging |
| `DATA_PATH` | *(empty)* | Local JSONL path (smoke) |

## Splits

- **Default (HF, ≥320 tasks)**: `80 train / 80 val / remainder test` — the spec's `80/80 airline + retail` balanced split (e.g. 80/80/640 for the full 800). This preserves domain balance after a seed-0 shuffle.
- **Large (≥750 tasks)**: `150 train / 300 val / 300 test` — the GEPA paper style (HotpotQA 150/300/300, IFBench 300/300/294) when a local merged file is that large.
- **Smoke (20 tasks via `--data-path`)**: `14 train / 3 val / 3 test` — mirrors HotpotQA smoke 14/3/3 so the 3-way pipeline is exercised.
- **Synthetic offline fallback (400 tasks, 200 per domain)**: `80/80/240` (the 80/80 case above).

`--domain airline|retail` filters before shuffling/splitting. `--train-limit` etc. slice after splitting (like IFBench/HotpotQA). All shuffles are deterministic (seed 0, overridable with `--seed`).

## Artifacts per run (`outputs/<run_dir>/`)

- `candidates.json`: every accepted candidate with lineage (`parents`), val scores, discovery eval counts
- `action_summary.json`: per-action proposal/accept counts, plus the verbalized selector's full distribution history (`probs`, `sampled`, `fallback` per call)
- `run_log.txt`: every proposal including rejected ones, with minibatch decisions
- `candidate_tree.html`: interactive candidate tree (open in a browser)

## Metric details

`utils.py:taubench_metric` implements the paper's **pass^k / task success** (binary, 0/1):

- Primary score: 1 if the agent's response achieves the expected outcome, else 0. `pass^k` for `k > 1` is `score^k` per sample aggregated over `k` independent rollouts; single-sample evaluation is `k=1` (task success), the standard Tau-bench reporting.
- Layered offline check (no live simulator needed): exact normalized match or substring containment against `expected_answer` → 1; otherwise token-overlap ≥ 60% → 0.5; expected tool-name containment → 1/0.5; empty → 0. With `success`/`reward` fields, that signal is used directly.
- Feedback for reflection: domain, task id, instruction, expected vs observed, reasons, and score (like IFBench's per-constraint feedback and PUPA's quality+leakage breakdown).

`_call_lm` uses the paper's Qwen3 decoding config (temp 0.6, top_p 0.95, top_k 20, max_tokens 16384, `enable_thinking: False`), shared with IFBench/PUPA/HotpotQA, with truncation and `ContextWindowExceededError` retries. Non-empty `reasoning_content` fallback is included for Qwen thinking models.

## Program details

- **1-stage**: single `agent_prompt` (tool-calling agent), one LM call. Tools are described in the system prompt (`AIRLINE_TOOLS_DESC` / `RETAIL_TOOLS_DESC`) so the LM can emit JSON tool calls.
- **2-stage (plan-then-act)**: `plan_prompt` generates a concise step-by-step plan (no tools); `act_prompt` executes with tools conditioned on the plan. Both prompts are optimized (round-robin), mirroring IFBench's `generate_response`→`ensure_correct_response` and HotpotQA's `generate_query`→`generate_answer`. The stage-1 plan is capped to 4000 chars before stage 2 so input + `max_tokens` fits the model context (32k for Qwen3).

Seed sentences preserve the task description verbatim in the `Task` section when `--seed-style structured`.

## Known pitfalls

- HF `datasets` caches to scratch on della; `run_taubench.sbatch` sets `HF_HOME` to scratch. It leaves `HF_HUB_OFFLINE=0` so the first fetch can hit HF; set `HF_HUB_OFFLINE=1` for fully offline runs (fallback will be used).
- Qwen thinking models: hidden `<think>` blocks can consume the whole token budget, leaving empty `message.content`. The runner disables thinking (`enable_thinking: false`) and falls back to `reasoning_content`; do not remove this.
- Stage 2's input includes stage 1's plan; the runner caps it so input plus `max_tokens` fits the model context.
- Budget counts **metric calls** (candidate × example), not LM calls. Stage-2 doubles LM calls per example; latency scales accordingly.
- Domain tools are **described** in the prompt for offline evaluation; a live Tau-bench simulator (stateful DB + tool executor) is not bundled — the metric checks textual / tool-name success without executing a database. Swap in the real simulator by replacing `taubench_metric`'s check with `task.check_success(state)`.
