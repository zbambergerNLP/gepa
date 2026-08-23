# HotpotQA: Wikipedia Retrieval Harness

This example compares vanilla GEPA with the Controller -> Manifestor -> ReAct V2 workflow on HotpotQA using the production retrieval path agreed for the benchmark. It loads HuggingFace `hotpot_qa/fullwiki`, discards every bundled context chunk, and retrieves evidence through the English Wikipedia MediaWiki API.

The default program mirrors `gepa-ai/gepa-artifact`:

1. Retrieve 7 pages with the original question.
2. `summarize1` summarizes the first-hop pages.
3. `create_query_hop2` generates the bridge query.
4. Retrieve 7 pages with that query.
5. `summarize2` synthesizes the second-hop evidence.
6. `final_answer` answers from the two summaries.

All four text components are optimized. Normalized exact match is the primary metric, matching the artifact; token F1 is retained in feedback and final reporting. The production split is 150 train / 300 validation / 300 test with the 6,871-metric-call budget.

## Setup

```bash
uv sync --extra dev
```

Retrieval uses a persistent SQLite cache (by default `$XDG_CACHE_HOME/gepa/wikipedia.sqlite3`, or `~/.cache/gepa/wikipedia.sqlite3`). The cache avoids repeating MediaWiki requests across candidates and runs. The endpoint, timeout, cache path, and per-hop retrieval count are configurable.

## Run

```bash
uv run python -m examples.hotpotqa.main \
    --condition both \
    --program 2stage \
    --max-metric-calls 6871 \
    --solver-model hosted_vllm/Qwen3.5-9B \
    --reflection-model hosted_vllm/Qwen3.5-9B \
    --api-base http://localhost:8000/v1 \
    --wikipedia-cache /tmp/gepa-wikipedia.sqlite3
```

`both` runs vanilla GEPA and ReAct V2 from the same canonical seed. ReAct V2 defaults to reflection level 2 and the broad insert/delete/replace/move tool set. Use `--edit-tool-set minimal` for the atomic insert/delete basis, `--reflection-level 1` to omit semantic-action manifestation, and `--template-family auto` (the default) to select prompt sections from the student model prefix.

The planned FOREST experiment roles are configurable without changing code:

```bash
uv run python -m examples.hotpotqa.main \
    --condition both \
    --solver-model "$QWEN_3_8_MODEL" \
    --solver-api-base "$QWEN_API_BASE" \
    --reflection-model "$DEEPSEEK_V4_FLASH_MODEL" \
    --reflection-api-base "$DEEPSEEK_API_BASE"
```

The solver and reflection endpoints are separate because the student and proposer need not be served by the same provider. Those environment variables must be set to the exact provider/checkpoint identifiers chosen at launch. Confirming those identifiers and the H200 allocation belongs to the excluded experiment-launch step; this harness does not start a run merely by being imported or tested.

Each condition directory contains `wikipedia-run-contract.json`, including the exact model IDs/endpoints, optimizer budget and axes, Wikipedia endpoint/cache, and hashes plus ordered IDs for all selected data splits. An existing `gepa_state.bin` without that contract, or a contract mismatch, is rejected before optimization. `candidates.json` embeds the same contract for portable result provenance.

Useful retrieval settings:

```bash
--wikipedia-endpoint https://en.wikipedia.org/w/api.php
--wikipedia-cache /path/to/wikipedia.sqlite3
--wikipedia-timeout 20
--retrieval-k 7
```

The committed 20-record distractor sample is data-only smoke input and is never selected automatically. Its bundled passages are also discarded, so an end-to-end smoke still performs live Wikipedia retrieval:

```bash
uv run python -m examples.hotpotqa.main \
    --data-path examples/hotpotqa/data/hotpotqa_distractor_sample.jsonl \
    --train-limit 1 --val-limit 1 --test-limit 1 \
    --condition vanilla --max-metric-calls 4
```

If `hotpot_qa/fullwiki` cannot be loaded, the production command fails with an actionable error instead of silently cycling smoke records.

On della:

```bash
MODEL="$QWEN_3_8_LOCAL_CHECKPOINT" \
    REFLECTION_MODEL="$DEEPSEEK_V4_FLASH_MODEL" \
    REFLECTION_API_BASE="$DEEPSEEK_API_BASE" \
    CONDITION=both MAX_METRIC_CALLS=6871 \
    WIKIPEDIA_CACHE=/scratch/gpfs/BSTEWART/$USER/gepa/.cache/gepa/wikipedia.sqlite3 \
    sbatch examples/hotpotqa/run_hotpotqa.sbatch
```

`MODEL` names the solver checkpoint directory under `MODEL_STORAGE` (or set `SOLVER_MODEL_PATH` and `SOLVER_SERVED_NAME` separately). The solver is served locally; `REFLECTION_MODEL` and optional `REFLECTION_API_BASE` route the proposer independently. For a deliberate same-model smoke run, set `SAME_MODEL=1`; omitting both a proposer model and that flag is an error.

## Files

| File | Purpose |
|---|---|
| `main.py` | GEPA conditions, configuration, optimization, and reporting |
| `utils.py` | Fullwiki loading, two-hop retrieval program, exact-match/F1 metrics |
| `data/hotpotqa_distractor_sample.jsonl` | Explicit data-only smoke sample |
| `run_hotpotqa.sbatch` | Della/vLLM runner with persistent retrieval cache |
| `ATTRIBUTION.md` | Dataset, artifact, and metric provenance |

`examples/common/wikipedia.py` contains the injectable MediaWiki client used by both HotpotQA and HoVer. Unit tests inject deterministic transports and retrievers; they do not depend on network access.
