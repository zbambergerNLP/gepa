# HoVer: Wikipedia Retrieval Harness

This example compares vanilla GEPA with the Controller -> Manifestor -> ReAct V2 workflow on the official HoVer v1.1 release. It reads `hover_train_release_v1.1.json`, parses document names from `supporting_facts` (including HuggingFace's `key` representation), and retains only records with exactly three unique supporting documents, matching the GEPA artifact.

The default program performs genuine retrieval rather than asking the LM to invent a title list:

1. Retrieve 7 Wikipedia pages with the claim.
2. `summarize1` summarizes those pages.
3. `create_query_hop2` generates a second-hop query and retrieves 7 pages.
4. `summarize2` combines first- and second-hop evidence.
5. `create_query_hop3` generates a third-hop query and retrieves 10 pages.

All four text components are optimized. A rollout scores 1 only when the retrieved pages contain all three gold documents, matching the artifact's discrete retrieval metric. Gold-document recall is reported separately. The split is 150 train / 300 validation / 300 test, with the 7,051-metric-call budget.

## Setup

```bash
uv sync --extra dev
```

The first production load downloads the official v1.1 training JSON from `hover-nlp/hover` into `examples/hover/data/`. A failed download is an error; synthetic data is never selected implicitly. Wikipedia retrieval is cached in SQLite at `$XDG_CACHE_HOME/gepa/wikipedia.sqlite3` by default.

## Run

```bash
uv run python -m examples.hover.main \
    --condition both \
    --program 2stage \
    --max-metric-calls 7051 \
    --solver-model hosted_vllm/Qwen3.5-9B \
    --reflection-model hosted_vllm/Qwen3.5-9B \
    --api-base http://localhost:8000/v1 \
    --wikipedia-cache /tmp/gepa-wikipedia.sqlite3
```

`both` runs vanilla GEPA and ReAct V2 from the same canonical seed. ReAct V2 defaults to reflection level 2 and the broad insert/delete/replace/move tool set. Use `--edit-tool-set minimal` for the atomic insert/delete basis, `--reflection-level 1` to omit semantic-action manifestation, and `--template-family auto` (the default) to select prompt sections from the student model prefix.

The planned FOREST experiment roles are configurable without changing code:

```bash
uv run python -m examples.hover.main \
    --condition both \
    --solver-model "$QWEN_3_8_MODEL" \
    --solver-api-base "$QWEN_API_BASE" \
    --reflection-model "$DEEPSEEK_V4_FLASH_MODEL" \
    --reflection-api-base "$DEEPSEEK_API_BASE"
```

The solver and reflection endpoints are separate because the student and proposer need not be served by the same provider. Those environment variables must be set to the exact provider/checkpoint identifiers chosen at launch. Confirming those identifiers and the H200 allocation belongs to the excluded experiment-launch step; this harness does not start a run merely by being imported or tested.

Each condition directory contains `wikipedia-run-contract.json`, including the exact model IDs/endpoints, optimizer budget and axes, Wikipedia endpoint/cache, and hashes plus ordered IDs for all selected data splits. An existing `gepa_state.bin` without that contract, or a contract mismatch, is rejected before optimization. `candidates.json` embeds the same contract for portable result provenance.

Useful settings:

```bash
--data-dir /path/containing/hover_train_release_v1.1.json
--wikipedia-endpoint https://en.wikipedia.org/w/api.php
--wikipedia-cache /path/to/wikipedia.sqlite3
--retrieval-k 7
--final-retrieval-k 10
```

The built-in three-record smoke dataset is available only through the explicit flag below. It changes only the claims; it still uses live Wikipedia retrieval:

```bash
uv run python -m examples.hover.main \
    --smoke --condition vanilla --max-metric-calls 4
```

On della:

```bash
MODEL="$QWEN_3_8_LOCAL_CHECKPOINT" \
    REFLECTION_MODEL="$DEEPSEEK_V4_FLASH_MODEL" \
    REFLECTION_API_BASE="$DEEPSEEK_API_BASE" \
    CONDITION=both MAX_METRIC_CALLS=7051 \
    WIKIPEDIA_CACHE=/scratch/gpfs/BSTEWART/$USER/gepa/.cache/gepa/wikipedia.sqlite3 \
    sbatch examples/hover/run_hover.sbatch
```

`MODEL` names the solver checkpoint directory under `MODEL_STORAGE` (or set `SOLVER_MODEL_PATH` and `SOLVER_SERVED_NAME` separately). The solver is served locally; `REFLECTION_MODEL` and optional `REFLECTION_API_BASE` route the proposer independently. For a deliberate same-model smoke run, set `SAME_MODEL=1`; omitting both a proposer model and that flag is an error.

## Files

| File | Purpose |
|---|---|
| `main.py` | GEPA conditions, configuration, optimization, and reporting |
| `utils.py` | Official v1.1 loading, three-hop retrieval program, document metric |
| `run_hover.sbatch` | Della/vLLM runner with persistent retrieval cache |
| `ATTRIBUTION.md` | Dataset and artifact provenance |

`examples/common/wikipedia.py` provides the shared injectable MediaWiki client. Unit tests use fake retrievers and transports, so CI does not require Wikipedia or dataset network access.
