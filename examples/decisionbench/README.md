# Hanno Labs DecisionBench

This example optimizes one structured-decision system prompt with vanilla GEPA
and FOREST. It uses the shared benchmark runner for model defaults, provider
retries, FOREST no-edit recovery, validation selection, frozen held-out evaluation,
the reusable unoptimized baseline, timing and tracking.

The solver defaults to `QWEN3_8_27B_MODEL` and the proposer to
`DEEPSEEK_V4_1_FLASH_MODEL`, with their pinned revisions and shared HotPotQA role
budgets. No model weights are downloaded by this example.

## Baseline and official scoring

The baseline is Hanno Labs' published **structured chat** protocol
`generic-decision-distribution-v1`. We import its `SYSTEM_PROMPT`,
`render_user_prompt`, exact-length `response_format`, `DecisionPrediction`,
`score_prediction`, `negative_log_likelihood` and 15-bin
`expected_calibration_error` directly from the pinned official package. All
primitives use the same prompt and schema; candidate labels, order, descriptions
and ordinal values come from the runtime row. Candidate counts up to 255 work.

The model generates a probability vector. These are **elicited probabilities**,
not native logits, token log-probabilities, or calibrated probabilities inferred
from a generated label. We preserve the upstream probability-source distinction.
Each value must be finite and between 0 and 1, the vector must have positive sum,
and the official schema normalizes by that sum. Accuracy uses the first argmax in
candidate order, including ties. Ordered Score additionally records the probability
weighted expected ordinal value; NLL uses the complete gold distribution with the
official `1e-15` probability floor.

The optimization metric is **all-row accuracy**, including malformed output,
refusals and context overflow as misses. The summary reports coverage, successful
row accuracy, successful row NLL, and successful row top-label ECE, with views by
full `task_id`, family, domain, primitive and candidate count. Parsing alone earns
no credit. Exhausted provider failures abort the run instead of manufacturing a
complete benchmark result. Incomplete responses are subject to the shared bounded
provider retry policy; malformed probability vectors receive no semantic retry.

Deliberate changes from the upstream `run-openrouter` baseline:

- Qwen is served through the shared LiteLLM/vLLM transport using the fixed shared
  decoding and reasoning budgets. We retain the upstream user prompt and strict
  response schema, including array length and numeric bounds.
- The upstream system instruction is placed inside the shared model-family prompt
  template so FOREST can edit it. The same seed, template, schema and runtime are
  used for baseline, GEPA and FOREST. Every rollout sends the actual current
  `system_prompt`; neither row targets nor provenance are sent to the solver.
- We reject duplicate JSON keys, numeric strings, booleans, extra fields, missing
  choices and non-`stop` finishes. We do not truncate inputs or use compact fields.
- Episode time covers rendering, all provider attempts/backoff, and scoring.
  The shared runner reports count, failures, mean/median/p95 wall time separately
  from batch throughput. There are no tools or user simulator in this benchmark.

This is an adapter around official DecisionBench rendering and scoring, with a
derived optimization split. It is not an official full-eval leaderboard run.

## Frozen data and leakage boundary

Canonical source: `Hanno-Labs/decision-bench` at
`071b7b2d2e1504c89e1e5a811a3f82e1bfe3aedb`,
`data/eval-00000-of-00001.parquet`, SHA-256
`d9dd5be8e0944a7ceadfcac6cf8f9d17aab256093acff23c6de6a037ad6e55f6`.
The upstream top-level manifest records the older bytes before optional compact
columns were added; `provenance/compact/manifest.json` records the current file
hash. Our loader uses the unchanged standard fields.

The only official split is `eval`: 23,900 rows, 43 task IDs, 27 families,
28 domains and three primitives. It combines 20,000 canonical rows, 2,700 applied
task rows and 1,200 reasoning rows. Original `source_json` lineage and a fingerprint
of every complete storage row are retained. That lineage includes Qwen-generated
paraphrases and synthetic tasks; sharing a model family with the Qwen solver is
an upstream data-provenance limitation, not evidence of independent human data.

For prompt optimization, `source-connected-sha256-60-20-20-task-round-robin-v1`
constructs connected groups from identical states, original source IDs, raw row
hashes, paraphrase parents, seed hashes and generation request hashes. Original
text/state also links paraphrases with different targets. MSMARCO query and
passage IDs use separate namespaces. All members of a connected group stay in
one split, including related MuSiQue tasks. The group hash, fixed split seed 1,
and policy name assign hash buckets 0–5 to train, 6–7 to validation and 8–9 to test.
This is a reproducible partition of public eval data, **not an official train
split**. It protects recorded source overlap; it cannot prove absence of semantic
near-duplicates in the upstream corpus or model pretraining.

| Derived split | Complete rows | Source groups | Default selected prefix |
|---|---:|---:|---:|
| Train | 14,331 | 14,003 | 150 |
| Validation | 4,752 | 4,655 | 300 |
| Held-out test | 4,817 | 4,707 | 300 |

Rows are ordered deterministically within tasks, then interleaved by task. All
43 tasks are represented in each full split and in each default selected prefix.
The default prefixes are a task-balanced view, not a random estimate of the
full 23,900-row distribution. The loader pins ordered full-record fingerprints
and returns complete splits; only the shared runner applies prefix limits after
freezing their identity. Model, method, optimization budget and request seeds
never resample the data. Change prefix sizes explicitly to select another view.

The runtime verifies the upstream Git revision and relevant installed source
hashes. The run contract also fingerprints adapter source and dependency versions.
Resume refuses data, prompt, runtime or configuration drift. Reflection accepts
training records only. The shared runner selects by validation, freezes the winner,
then evaluates held-out test once per row and the same unoptimized seed baseline.
Pilot calibration uses training rows only.

## Run

Use Python 3.12 or 3.13. The optional DecisionBench dependency is pinned to
`decision-bench @ git+https://github.com/Hanno-Labs/decision-bench.git@9a6328ee066a3dec0ed94ea9e57ddd3f63c57aa9`.
Its upstream requirements include `pyarrow>=21,<22` and `transformers>=5.17,<6`;
Python 3.14 is not supported by this integration. For a standalone development
checkout before the suite dependency extra is merged:

```bash
uv sync --locked --extra dev --python 3.13
uv pip install --python .venv/bin/python \
  'decision-bench @ git+https://github.com/Hanno-Labs/decision-bench.git@9a6328ee066a3dec0ed94ea9e57ddd3f63c57aa9'
uv run --no-sync pytest -q tests/test_decisionbench.py
uv run --no-sync python -m examples.decisionbench.main --help
```

After the shared runner is present, a training-only latency pilot against the
configured solver server is:

```bash
uv run --no-sync python -m examples.decisionbench.main \
  --mode pilot --solver-api-base http://localhost:8000/v1 \
  --reflection-api-base http://localhost:8001/v1 \
  --run-dir outputs/decisionbench
```

Use `--mode optimize --condition both` for the paired methods or `--mode baseline`
for the shared seed baseline. The common optimizer budget is 6,871 metric calls;
it does not determine split membership. The local server must support the strict
JSON-schema response format without dropping it. Transport and actual model
performance require a live model pilot; the offline tests do not demonstrate them.

`--data-file /path/to/eval.parquet` accepts an offline copy only if its bytes match
the pinned hash. Otherwise the loader downloads only the approximately 54 MiB
canonical Parquet, never models. `--dataset-cache` selects an optional HF cache.

## Sources

- [Canonical dataset and release card](https://huggingface.co/datasets/Hanno-Labs/decision-bench/tree/071b7b2d2e1504c89e1e5a811a3f82e1bfe3aedb)
- [Hanno Labs introduction](https://hannolabs.ai/field-notes/decisionbench)
- [Official runtime at the pinned revision](https://github.com/Hanno-Labs/decision-bench/tree/9a6328ee066a3dec0ed94ea9e57ddd3f63c57aa9)
- [Official evaluation and comparability contract](https://github.com/Hanno-Labs/decision-bench/blob/9a6328ee066a3dec0ed94ea9e57ddd3f63c57aa9/docs/get_started/usage/running_the_evaluation.md)
- [Published model readout contracts](https://github.com/Hanno-Labs/decision-bench/blob/9a6328ee066a3dec0ed94ea9e57ddd3f63c57aa9/docs/overview/models.md)
