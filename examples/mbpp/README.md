# MBPP: Action-Conditioned Reflection Experiments

This example evaluates whether **action-conditioned reflection** (constraining each GEPA mutation to a typed edit action) and **verbalized sampling** (letting the reflection LM propose a probability distribution over actions, then sampling its tails) improve GEPA on **MBPP** (Austin et al. 2021, HF `mbpp` sanitized, 974 problems), the code generation benchmark where GEPA's largest defeats are reported (CANTANTE +18.9 pp on MBPP, FlowBot on HumanEval/MBPP).

The setup mirrors `examples/gsm8k`, `examples/aime_math` and `examples/ifbench`: a **single-step code generation** program (one optimized `instruction`, one LM call with `Final Answer:` marker and ```python block extraction), **execution-based pass@1** via sandboxed Python execution (2s timeout, subprocess, fallback heuristic) with reflection-ready feedback, deterministic splits **150 train / 300 val / 300 test** (pooled 750, cycled; paper's 374 train / 500 test noted), and a **5000-metric-call** budget (paper heavy ~5K). It reuses the same `ActionDiversityCallback`, `GEPAConfig`/`ReflectionConfig`/`EngineConfig`, and solver-model `hosted_vllm` + `_call_lm` (temp 0.6 / top_p 0.95 / top_k 20 / max 16384 / `enable_thinking: False`) harness as `ifbench`/`pupa`/`gsm8k`.

Built on top of the scaled IFBench baseline (`rev cf502ad6` — 15K IFBench, merged upstream `8a2bed96` parallel proposals + OA refactor). MBPP complements GSM8K (math) with code — together they cover the two benchmarks where GEPA is most flatly beaten.

## Layout

| File | Purpose |
|---|---|
| `main.py` | Experiment runner: vanilla/random/action conditions, seeds, dumps, final report |
| `utils.py` | Dataset loading (HF `mbpp` sanitized, seed-0 splits 150/300/300; local `data/mbpp.jsonl` fallback), single-step code LM program (`_call_lm` identical to ifbench), execution-based pass@1 + sandbox |
| `run_mbpp.sbatch` | SLURM job for della (serves Qwen via vLLM, runs one experiment, 48h) |
| `ATTRIBUTION.md` | Data/code provenance (MBPP, HumanEval, CANTANTE, FlowBot) |
| `README.md` | This file |
| `data/` | Optional local `mbpp.jsonl` fallback (one JSON per line: `task_id`, `text`, `test_list`) |

## Setup

From the repo root (`gepa/`):

```bash
uv sync --extra dev
# MBPP uses datasets + litellm (already in full extra). First dataset load
# downloads mbpp sanitized from HF (974 problems).
# Offline: place local artifact at examples/mbpp/data/mbpp.jsonl
# or pass --data-path /path/to/mbpp.jsonl
```

No extra system deps beyond `datasets`/`litellm`. The sandbox uses `subprocess` with a 2s timeout and no network/file access; on platforms without subprocess it falls back to a heuristic (def + token overlap check).

## Running

```bash
uv run python -m examples.mbpp.main \
    --condition all            # vanilla | random | action | all \
    --seed-style plain         # plain | structured (markdown skeleton) \
    --actions default          # default (6 generic) | structured (16 section-scoped) \
    --max-metric-calls 5000    # paper heavy budget; mini runs use smaller \
    --solver-model hosted_vllm/Qwen3.5-9B --api-base http://localhost:8000/v1 \
    --tag mbpp_rev1

# With explicit data path:
uv run python -m examples.mbpp.main --data-path examples/mbpp/data/mbpp.jsonl --max-metric-calls 100

# Mini run:
uv run python -m examples.mbpp.main --train-limit 20 --val-limit 30 --test-limit 30 --max-metric-calls 100 --condition vanilla
```

Conditions: `vanilla` is stock GEPA reflection; `random` picks actions uniformly; `action` uses `VerbalizedActionSelector`. `--actions structured` implies structured seeds (markdown skeleton with `Role`/`Task`/`Rules`/`Output Format`/`Examples`).

On della, submit via sbatch (mirrors `ifbench`/`pupa`/`gsm8k` runners):

```bash
MODEL=Qwen3-8B CONDITION=action SEED_STYLE=plain ACTIONS=default TAG=mbpp_rev1 MAX_METRIC_CALLS=5000 \
    sbatch examples/mbpp/run_mbpp.sbatch
# Mini test (20/30/30, 100 calls):
# sbatch --export=ALL,TRAIN_LIMIT=20,VAL_LIMIT=30,TEST_LIMIT=30,MAX_METRIC_CALLS=100 --time=02:00:00 examples/mbpp/run_mbpp.sbatch
```

## Metric details

- **Metric:** `mbpp_metric` in `utils.py` extracts ```python code from the LM output (after `Final Answer:` marker, `<think>` stripping), then executes it against the example's `test_list` + `challenge_test_list` in a sandboxed subprocess (2s timeout). Score is 1 if all tests pass, else 0. Feedback enumerates passed/failed count plus execution stdout/stderr (first 800 chars) and a code preview for reflection. When no tests are present, the heuristic checks for a `def` statement.

- **Dataset:** HF `mbpp` sanitized (974 problems, each with `text` prompt, `test_list` asserts, `challenge_test_list`, `test_setup_code`), plus `mbpp` plain and local `data/mbpp.jsonl` fallback. Deterministic shuffle seed 0 → 150/300/300 (cycled to 750 if needed, like `frontiercs`). No data files are committed.

- **Program:** Single-stage code generation (one system `instruction` prompt, one `_call_lm` call with `Final Answer:` + ```python). Mirrors `gsm8k`'s single-stage math CoT but with code extraction and execution rather than numeric normalization.

- **Code provenance:** `utils.py` `_call_lm`, `_strip_think`, `_extract_code`, `run_mbpp_single_stage`, `_exec_in_sandbox` sandbox, `load_mbpp_dataset` shuffle logic, and `main.py` condition/build_config/dump/action-tracking are copied from `examples/gsm8k` and `examples/ifbench` to keep all benchmarks consistent. See `ATTRIBUTION.md` for paper references (MBPP Austin et al. 2021, CANTANTE 2605.13295, FlowBot 2604.26258, HumanEval Chen et al. 2021).

## Artifacts per run (`outputs/<run_dir>/`)

- `candidates.json`: every accepted candidate with lineage (`parents`), val scores, discovery eval counts
- `action_summary.json`: per-action proposal/accept counts plus verbalized selector history
- `run_log.txt`: every proposal including rejected ones (if enabled via callback)
- `candidate_tree.html`: interactive candidate tree (open in a browser)
