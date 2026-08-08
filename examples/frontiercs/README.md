# FrontierCS: Action-Conditioned Reflection Experiments

This example evaluates whether **action-conditioned reflection** (constraining each GEPA mutation to a typed edit action) and **verbalized sampling** (letting the reflection LM propose a probability distribution over actions, then sampling its tails) improve GEPA on **FrontierCS** (Frontier-CS, https://github.com/FrontierCS/Frontier-CS), the open-ended CS research problems benchmark.

FrontierCS tasks are open-ended research problems — e.g., designing a protocol, algorithm, or evaluation — that test an auto-research framework's ability to survey literature and draft a complete proposal. The setup here mirrors `examples/ifbench` and `examples/pupa`: a 2-stage program (literature review → proposal) and a 1-stage ablation, with the same GEPA harness, action space, and decoding config. See `ATTRIBUTION.md` for provenance.

## Layout

| File | Purpose |
|---|---|
| `main.py` | Experiment runner: conditions, programs, seeds, dumps, final report |
| `utils.py` | Dataset loading (HF `FrontierCS/Frontier-CS` or local `data/frontiercs.jsonl` fallback + synthetic offline), 1- and 2-stage LM programs, rubric metric |
| `run_frontiercs.sbatch` | SLURM job for della (serves Qwen via vLLM, runs one experiment) |
| `ATTRIBUTION.md` | Data/code provenance |
| `README.md` | This file |
| `data/` | Optional local JSONL fallback (`frontiercs.jsonl`, one task per line) |

## Dataset & Splits

- **Primary**: HuggingFace `FrontierCS/Frontier-CS` via `datasets.load_dataset` (the paper reports ~100 open-ended CS research problems across ML/Systems/Theory/Security/HCI).
- **Fallbacks**: local `data/frontiercs.jsonl` if present, otherwise a deterministic synthetic 90-example pool (10 stems × 9 cycles) so that smoke/tests work offline.
- **Splits**: deterministic shuffle (seed 0) then **30 train / 30 val / 30 test** (train 0:30, val 30:60, test 60:90), noting the paper's ~100-problem pool. Mirrors IFBench's slicing convention (`IFBench_train.jsonl[300:600]` etc.) but at the smaller scale appropriate for FrontierCS. Override with `--train-limit` / `--val-limit` / `--test-limit` or `--data-path`.

JSONL schema (one record per line, fields flexible — see `utils._normalize_record`):
```json
{"id": "cs_001", "problem": "Design a ...", "area": "Systems", "difficulty": "hard", "rubric": ["Technically sound", "Novel", "Evaluation plan"]}
```
`rubric` may be a list or a string; missing fields get sensible defaults. A short smoke file (e.g., 20 lines) is automatically cycled to 90 so the 30/30/30 invariant holds for offline tests.

## Setup

```bash
uv sync --extra dev
# FrontierCS uses datasets + litellm (already in full/dev)
```

No vendored checker is needed (unlike IFBench). First HF load caches to `$HF_HOME` (on della, scratch via the sbatch). For fully offline runs, place a JSONL at `examples/frontiercs/data/frontiercs.jsonl` or pass `--data-path`.

## Running

```bash
uv run python examples/frontiercs/main.py \
    --condition all            # vanilla | random | action | all \
    --program 2stage           # 2stage (literature-then-proposal) | 1stage (single proposal) \
    --seed-style plain         # plain (seed sentences) | structured (markdown Role/Task/Rules/Format/Examples) \
    --actions default          # default (6 generic actions) | structured (16 section-scoped) \
    --max-metric-calls 4000    # stretch default (3000-5000); use 15000 for Wave B scale \
    --solver-model hosted_vllm/Qwen3.5-9B --api-base http://localhost:8000/v1 \
    --tag frontiercs_rev1      # suffix for output dirs
```

- `vanilla` is stock GEPA reflection; `random` picks actions uniformly; `action` uses `VerbalizedActionSelector`.
- `--actions structured` implies `--seed-style structured` (markdown skeleton with explicit sections for section-scoped edits).
- Mini runs: `--train-limit 15 --val-limit 15 --test-limit 15 --max-metric-calls 150`.
- With a local file: `--data-path examples/frontiercs/data/frontiercs.jsonl`.
- Judge: by default the solver model judges its own outputs via `frontiercs_metric` (rubric PASS/FAIL + `SCORE:`); pass `--judge-model hosted_vllm/Qwen3.5-9B` to use a different judge.

On della, submit via sbatch (mirrors `ifbench` runner):

```bash
MODEL=Qwen3.5-9B CONDITION=action PROGRAM=2stage SEED_STYLE=plain ACTIONS=default MAX_METRIC_CALLS=4000 TAG=frontiercs_rev1 \
    sbatch examples/frontiercs/run_frontiercs.sbatch
# Mini test (15/15/15, 150 calls):
# sbatch --export=ALL,TRAIN_LIMIT=15,VAL_LIMIT=15,TEST_LIMIT=15,MAX_METRIC_CALLS=150 --time=02:00:00 examples/frontiercs/run_frontiercs.sbatch
# With local data:
# sbatch --export=ALL,DATA_PATH=examples/frontiercs/data/frontiercs.jsonl examples/frontiercs/run_frontiercs.sbatch
```

## Artifacts per run (`outputs/<run_dir>/`)

- `candidates.json`: every accepted candidate with lineage (`parents`), val scores, discovery eval counts
- `action_summary.json`: per-action proposal/accept counts, plus the verbalized selector's full distribution history (`probs`, `sampled`, `fallback` per call)
- `run_log.txt`: every proposal including rejected ones, with minibatch decisions (via GEPA's engine)
- `candidate_tree.html`: interactive candidate tree (open in a browser)

## Metric details

`frontiercs_metric` in `utils.py` implements a rubric-based score:

- **With a judge model** (default: solver model): the judge is prompted with the problem, rubric, and proposal and asked to emit per-criterion `PASS`/`FAIL` plus a final `SCORE: <0-1>` (fraction of criteria passing). The score is the mean pass rate (0-1), blended 70/30 with the holistic `SCORE:` when both are available. Feedback is the judge's full trace, truncated to 2000 chars — this is what GEPA's reflection sees.
- **Offline fallback** (no judge or judge call fails): heuristic keyword overlap per rubric item plus a length signal, still 0-1 with per-item `[PASS]/[FAIL]` feedback so GEPA has a learning signal.

Decoding is identical to `examples/ifbench/utils.py`: `temperature=0.6, top_p=0.95, top_k=20, max_tokens=16384, enable_thinking=False` (with `COT_FORMAT_INSTRUCTION` and `<think>` stripping, plus truncation retries 16384→4096→1024→256).

## Programs

- **2-stage** (`literature_review` → `draft_proposal`): stage 1 surveys prior work and baselines for the problem; stage 2 drafts the proposal conditioned on the problem plus the literature summary (capped to 12k chars so the 32k context always fits). Returns `(literature, proposal)`; only the proposal is scored.
- **1-stage** (`research_proposal`): single turn, single prompt — the whole task in one call.

## Known pitfalls

- **Long prompts**: candidate prompts can accrete text over many GEPA iterations; `SOFT_PROMPT_CHAR_BUDGET` / `MAX_PROPOSAL_CHARS` in the action space and the per-stage caps (12k chars) keep the 32k context feasible. Do not remove the thinking disable (`enable_thinking: False`) or the `<think>` stripping.
- **HF cache on della**: `run_frontiercs.sbatch` sets `HF_HOME` to scratch and leaves `HF_HUB_OFFLINE=0` for the first fetch; subsequent runs hit cache. For fully offline, use `--data-path`.
- **Rubric parsing**: the judge may not emit exactly one `PASS`/`FAIL` per criterion; the metric handles both the `SCORE:` line and fuzzy PASS/FAIL counting, falling back to the heuristic if needed.
