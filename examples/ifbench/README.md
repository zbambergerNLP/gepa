# IFBench: Action-Conditioned Reflection Experiments

This example evaluates whether **action-conditioned reflection** (constraining each GEPA mutation to a typed edit action) and **verbalized sampling** (letting the reflection LM propose a probability distribution over actions, then sampling its tails) improve GEPA on IFBench, the instruction-following benchmark from the GEPA paper (arXiv:2507.19457).

The setup replicates the paper's artifact (`gepa-ai/gepa-artifact`): exact data files and splits (300 train / 300 val from IF-RLVR-style constraints, 294 novel-constraint test), the 2-stage program (`generate_response` then `ensure_correct_response`), the instruction-level accuracy metric with per-constraint feedback, and the 3,593-metric-call budget. See `ATTRIBUTION.md` for provenance and local modifications.

## Results so far

- **`FIRST_EXPERIMENTAL_RESULTS.md`**: Rev 1 results (July 2026, Qwen3.5-9B, plain seed prompts, generic 6-action space). Headline: no condition beat baseline on the novel-constraint test set, but action-conditioning prevented the val-overfitting collapse that vanilla GEPA suffered in the 2-stage program.
- Rev 2 (Qwen3-8B, structured markdown prompts, 16 section-scoped actions) is in flight; see `docs/HANDOVER_action_conditioned_reflection.md` at the repo root for the full timeline and status.

## Layout

| File | Purpose |
|---|---|
| `main.py` | Experiment runner: conditions, programs, seeds, dumps, final report |
| `utils.py` | Dataset loading (auto-downloads data on first use), 1- and 2-stage LM programs, metric port |
| `utils_ifbench/` | Vendored allenai constraint checkers (Apache 2.0, via the paper artifact) |
| `analyze_actions.py` | Post-hoc analysis: action-choice entropy, tail-sampling rate, proposal diversity |
| `run_ifbench.sbatch` | SLURM job for della (serves Qwen via vLLM, runs one experiment) |
| `ATTRIBUTION.md` | Data/code provenance and the two local modifications to vendored code |
| `FIRST_EXPERIMENTAL_RESULTS.md` | Rev 1 experiment summary (do not edit; historical record) |

## Setup

```bash
uv sync --extra dev --extra ifbench
```

The `ifbench` extra installs the vendored checkers' dependencies (nltk, spacy, langdetect, emoji, syllapy, immutabledict). First dataset load downloads the two data files from the artifact repo and nltk's `punkt_tab`; the spacy model `en_core_web_sm` installs on demand (pre-install both on offline clusters).

## Running

```bash
uv run python examples/ifbench/main.py \
    --condition all            # vanilla | random | action | all \
    --program 2stage           # 2stage (paper protocol) | 1stage (ablation) \
    --seed-style structured    # plain (paper seed sentences) | structured (markdown skeleton) \
    --actions structured       # default (6 generic actions) | structured (16 section-scoped) \
    --max-metric-calls 3593    # paper budget \
    --solver-model hosted_vllm/Qwen3-8B --api-base http://localhost:8000/v1 \
    --tag rev2                 # suffix for output dirs
```

Conditions: `vanilla` is stock GEPA reflection; `random` picks actions uniformly (baseline); `action` uses `VerbalizedActionSelector`. `--actions structured` implies structured seeds. Mini runs: `--train-limit/--val-limit/--test-limit`.

On della, submit via the wrapper (syncs code, then sbatches with the env's partition/model storage):

```bash
MODEL=Qwen3-8B PROGRAM=2stage CONDITION=action SEED_STYLE=structured ACTIONS=structured \
    TAG=rev2 TIME=03:00:00 scripts/della/submit_ifbench.sh
# First time: SETUP=1 installs the ifbench extra + nltk/spacy data on the vis node.
```

## Artifacts per run (`outputs/<run_dir>/`)

- `candidates.json`: every accepted candidate with lineage (`parents`), val scores, discovery eval counts
- `action_summary.json`: per-action proposal/accept counts, plus the verbalized selector's full distribution history (`probs`, `sampled`, `fallback` per call)
- `run_log.txt`: every proposal including rejected ones, with minibatch decisions
- `candidate_tree.html`: interactive candidate tree (open in a browser)

Analyze fetched runs locally:

```bash
uv run python examples/ifbench/analyze_actions.py outputs/<fetched_dir>
```

## Known pitfalls (fixed in code, kept here for context)

- Qwen thinking models: hidden `<think>` blocks can consume the whole token budget, leaving empty `message.content`. The runner disables thinking (`enable_thinking: false`) and falls back to `reasoning_content`; do not remove this.
- The vendored checkers can crash on degenerate outputs (punctuation-only sentences); `ifbench_metric` treats a crashing check as "constraint not followed".
- Stage 2's input includes stage 1's output; the runner caps it so input plus `max_tokens` fits the model context.
