# GSM8K: Action-Conditioned Reflection Experiments

This example evaluates whether **action-conditioned reflection** (constraining each GEPA mutation to a typed edit action) and **verbalized sampling** (letting the reflection LM propose a probability distribution over actions, then sampling its tails) improve GEPA on **GSM8K** (Cobbe et al. 2021, HF `gsm8k` / `openai/gsm8k` main config), the grade-school math benchmark where GEPA's largest defeats are reported (CANTANTE +21 pp, VISTA +74 pp recovery).

The setup mirrors `examples/aime_math` and `examples/ifbench`: a **single-step CoT** program (one optimized `instruction`, one LM call with `Final Answer:` marker), **exact-match accuracy** after numeric/boxed normalization with solution-aware feedback, deterministic splits **150 train / 300 val / 300 test** (or 200/300/300 with headroom; paper notes GSM8K's typical 7.5K train / 1K test), and a **5000-metric-call** budget (paper heavy ~5K, legacy vs scaled). It reuses the same `ActionDiversityCallback`, `GEPAConfig`/`ReflectionConfig`/`EngineConfig`, and solver-model `hosted_vllm` + `_call_lm` (temp 0.6 / top_p 0.95 / top_k 20 / max 16384 / `enable_thinking: False`) harness as `ifbench`/`pupa`/`aime_math`.

Built on top of the scaled IFBench baseline (`rev cf502ad6` — 15K IFBench, merged upstream `8a2bed96` parallel proposals + OA refactor). GSM8K reuses the same action-conditioned machinery and supports a **defective-seed** variant for VISTA-style recovery tests.

## Layout

| File | Purpose |
|---|---|
| `main.py` | Experiment runner: vanilla/random/action conditions, seeds, dumps, final report |
| `utils.py` | Dataset loading (HF `gsm8k`/`openai/gsm8k` main config, seed-0 splits 150/300/300; local `data/gsm8k.jsonl` fallback), single-step LM program (`_call_lm` identical to ifbench), numeric/boxed metric + defective-seed support |
| `run_gsm8k.sbatch` | SLURM job for della (serves Qwen via vLLM, runs one experiment) |
| `ATTRIBUTION.md` | Data/code provenance (GSM8K, CANTANTE, VISTA) |
| `README.md` | This file |
| `data/` | Optional local `gsm8k.jsonl` fallback (one JSON per line: `question`, `answer`) |

## Setup

From the repo root (`gepa/`):

```bash
uv sync --extra dev
# GSM8K uses datasets + litellm (already in full extra). First dataset load
# downloads gsm8k / openai/gsm8k from HF (main config).
# Offline: place local artifact at examples/gsm8k/data/gsm8k.jsonl
# or pass --data-path /path/to/gsm8k.jsonl
```

## Running

```bash
uv run python -m examples.gsm8k.main \
    --condition all            # vanilla | random | action | all \
    --seed-style plain         # plain | structured (markdown skeleton) \
    --actions default          # default (6 generic) | structured (16 section-scoped) \
    --max-metric-calls 5000    # paper heavy budget; mini runs use smaller \
    --solver-model hosted_vllm/Qwen3.5-9B --api-base http://localhost:8000/v1 \
    --tag gsm8k_rev1

# With explicit data path:
uv run python -m examples.gsm8k.main --data-path examples/gsm8k/data/gsm8k.jsonl --max-metric-calls 100

# VISTA-style defective-seed recovery test:
uv run python -m examples.gsm8k.main --defective-seed --defective-variant default --condition action --max-metric-calls 5000

# Mini run:
uv run python -m examples.gsm8k.main --train-limit 20 --val-limit 30 --test-limit 30 --max-metric-calls 100 --condition vanilla
```

Conditions: `vanilla` is stock GEPA reflection; `random` picks actions uniformly; `action` uses `VerbalizedActionSelector`. `--actions structured` implies structured seeds (markdown skeleton with `Role`/`Task`/`Rules`/`Output Format`/`Examples`). `--defective-seed` swaps the seed to a weak instruction for VISTA-style recovery experiments.

On della, submit via sbatch (mirrors `ifbench`/`pupa` runners):

```bash
MODEL=Qwen3-8B CONDITION=action SEED_STYLE=plain ACTIONS=default TAG=gsm8k_rev1 MAX_METRIC_CALLS=5000 \
    sbatch examples/gsm8k/run_gsm8k.sbatch
# Mini test (20/30/30, 100 calls):
# sbatch --export=ALL,TRAIN_LIMIT=20,VAL_LIMIT=30,TEST_LIMIT=30,MAX_METRIC_CALLS=100 --time=02:00:00 examples/gsm8k/run_gsm8k.sbatch
# Defective-seed (VISTA recovery):
# sbatch --export=ALL,DEFECTIVE_SEED=1,CONDITION=action TAG=gsm8k_vista sbatch examples/gsm8k/run_gsm8k.sbatch
```

Environment knobs for `run_gsm8k.sbatch` (all override via `sbatch --export=ALL,VAR=...`): `MODEL`, `CONDITION`, `SEED_STYLE`, `ACTIONS`, `MAX_METRIC_CALLS`, `TAG`, `TRAIN_LIMIT`, `VAL_LIMIT`, `TEST_LIMIT`, `DATA_PATH`, `DEFECTIVE_SEED`, `DEFECTIVE_VARIANT`, `GEN_PORT`, `GEN_GMU`, `GEN_MAX_LEN`, `HEALTH_TIMEOUT`, `MODEL_STORAGE`, `POSIT_DIR`, `SCRATCH_BASE`.

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

`gsm8k_metric` in `utils.py` mirrors `aime_math`'s `math_metric`:

- Extracts the text after `Final Answer:` (fallback: full output), unwraps `\boxed{}` if present, and checks **numeric equality** after stripping commas/dollars, handling integers, decimals, and fractions (`3/4` → 0.75) with small floating tolerance.
- Scores 1 if the normalized prediction equals the gold numeric answer (GSM8K answers have `#### <number>` suffix; loader extracts `answer_number`), else 0.
- Feedback is reflection-ready: `"Your answer is correct/incorrect. The correct answer is '...'. Here's the full step-by-step solution: ..."` when a solution trace is available.

The LM program `run_gsm8k_single_stage` is a single system-prompt call with `Final Answer:` extraction, `\boxed{}` handling, and context-window stepping (16384→4096→1024→256) identical to `ifbench`/`aime_math`.

## Splits (GSM8K)

- **Source**: HF `gsm8k` / `openai/gsm8k` `main` config. Canonical sizes: train 7473, test 1319. Loader merges train+test then shuffles deterministically (seed 0) and slices **150 train / 300 val / 300 test** (750 held-out from the shuffled pool; alternative 200/300/300 documented for headroom). For small pools (<750) falls back to proportional thirds. Caller can further cap via `--train-limit`/`--val-limit`/`--test-limit`.
- Paper notes GSM8K's typical split is ~7.5K train / 1K test (Cobbe et al. 2021); the lightweight 150/300/300 split leaves headroom for larger val/test and faster iteration while remaining deterministic.
- Each example carries `prompt`/`problem`/`input`/`question`, `answer` (full trace with `####`), `answer_number` (numeric), `solution`, and stable `id` for GEPA caching.

## Known pitfalls

- HF `datasets` caches to scratch on della; `run_gsm8k.sbatch` sets `HF_HOME` to scratch (offline after first fetch, `HF_HUB_OFFLINE=0` so HF download works).
- Like IFBench/AIME, `<think>` blocks are stripped and `enable_thinking: False` is forced so Qwen3 reasoning does not consume the budget.
- Integer/boxed parsing: the model should emit its answer after `Final Answer:`; non-parsable outputs score 0 with feedback to fix formatting. `\boxed{}` is handled as an alternative.
- GSM8K answers are numeric (often integers, sometimes decimals/fractions); the metric normalizes both sides before comparison.

## Defective seed (VISTA)

`utils.py` exposes `DEFECTIVE_SEED_CANDIDATE` and `get_defective_seed(variant)`; `main.py` exposes `--defective-seed` / `--defective-variant`. The default defective seed suppresses chain-of-thought (`"always wrap your final answer in \\boxed{}. Do not show your work."`), reproducing the VISTA analysis where GEPA degraded from 23.81% → 13.50% on GSM8K with a defective seed and VISTA recovered to 87.57% (+74 pp). Use `--defective-seed` to run a recovery experiment (compare `vanilla` vs `action` from the same defective start).
