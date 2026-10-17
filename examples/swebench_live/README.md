# SWE-bench Live on the local machine

This is the seventh benchmark alongside HotPotQA, DecisionBench, AppWorld,
OBLIQ, Tau and Terminal-Bench. It uses Microsoft's **SWE-bench Live** Python
dataset and official Docker test grader. It does not use the older SWE-bench
Verified prototype or its patch-similarity metric.

## Prepare without running an experiment

From the repository root, with the [local API environment](../local_api/README.md)
installed:

```bash
UV_PROJECT_ENVIRONMENT=.venv-local-api uv run --no-sync python -m examples.swebench_live.prepare

# Download native task images only; no containers or model calls:
UV_PROJECT_ENVIRONMENT=.venv-local-api uv run --no-sync python -m examples.swebench_live.prepare --images
```

Preparation stores assets in `.cache/swebench-live`; `--root` selects another
location. The data revision is `b51a86422e10cfd403beb4773e5a2947953e36ec` and the
official Python harness is `ad79b850f15e33992e96f03f6e97f05ddf9aa0be`. All five
Parquet files and their ordered JSONL exports have fixed checksums. Updating to
a newer Live dataset requires a new reviewed manifest and campaign.

The upstream `full`, `lite`, `verified` and `test` names are overlapping subsets,
not optimization partitions. The official `test` has 1,000 unique tasks and is
preserved in its original order. Excluding every test ID from `full` leaves 895
unique tasks; sorting their IDs by SHA-256 and dividing in half gives 447 training
and 448 validation tasks. These are derived research splits, not official
training/validation splits. The shared CLI defaults to the first 150 training
and 300 validation tasks and the complete official test set. Limits, ordered
records, source hashes and runtime are saved in each immutable run contract.

The pinned release has one repeated ID in `full` and `verified`, with differing
test commands/hints, and eight official test IDs absent from `full`. Raw files
retain these rows. The repeated ID is held out; its authoritative `test` record
is used. No test ID enters training or validation. `data-manifest.json` records
these facts rather than silently treating the upstream subsets as disjoint.

`local-runtime.patch` removes eager Modal imports/dependency and rejects that
execution mode. It also selects the published AMD64 images on Apple Silicon.
The grading/test logic is unchanged. Dependencies are hash-locked in
`runtime-requirements.txt`, installed in a separate environment, and verified
with the source patch before execution. No Modal SDK or credentials are needed.
To regenerate that dependency lock after a deliberate upstream update, run
`uv pip compile <prepared-source>/pyproject.toml --no-sources --python-version 3.12
--generate-hashes -o examples/swebench_live/runtime-requirements.txt` outside the
upstream workspace (its pinned workspace references an absent `launch` package).

Image preparation resolves official DockerHub tags to immutable manifests,
requests `linux/amd64`, verifies the downloaded IDs/digests, and keeps a disk
reserve. Re-running the preparation command continues existing downloads and
never replaces a recorded manifest with a newer tag. `--image-split train`,
`val` or `test` stages one partition. All partitions together need 1,895 distinct
task images; the complete collection can far exceed a laptop's free storage.
Partial staging is reported explicitly and does not imply native readiness.

The default local execution contract requires four Docker CPUs and 16 GiB RAM
per sequential task. This follows the upstream resource guidance; passing the
resource check still does not establish native test compatibility. A 4 GiB
Docker VM cannot pass this default check. Preparation never changes Docker
settings or restarts it. Image downloads alone are not native execution tests.

## Execution contract

The solver uses one native `shell(command)` tool inside an isolated container,
with no host mounts, credentials or external network. It receives only a general
system prompt and `problem_statement`. Hints, gold patches and test metadata are
not exposed to the solver. The complete conversation and shell outputs are saved
without context truncation. The default limit is 100 model steps and 120 seconds
per shell command, with four CPUs and 16 GiB RAM.

After the rollout ends and its container is removed, the patch is frozen. A
separate container runs the official evaluator with the test patch, using the
same immutable image and resource/network settings. The scalar metric is the
official boolean `resolved`, once per task. Empty patches and context misses
score zero and remain visible; provider or infrastructure failures stop the run
without being mislabeled as failed tests. Native logs and incomplete intents
remain on disk. The shared API task journal blocks replay of an unresolved
started evaluation.

Both optimization methods edit the same general system prompt. Reflection uses
training rollouts only and never supplies a gold patch. Validation chooses the
winner; the shared runner freezes it before held-out testing and counts the
matching starting-prompt baseline once. All shared search variants remain
available. This is an experimental prompt-optimization protocol, not a claim
of official leaderboard submission eligibility.

After separate authorization to run paid model calls, use an explicitly selected
API profile and prepared assets:

```bash
UV_PROJECT_ENVIRONMENT=.venv-local-api uv run --no-sync python -m examples.swebench_live.main \
  --api-profile outputs/api-profile.json --env-file /absolute/path/to/.env \
  --allow-api-calls --condition both --run-dir outputs/swebench-live-new-campaign
```

The profile controls both models; there is no assumed model for this benchmark.
`--agent-max-steps`, `--task-memory-gib` and `--test-timeout` are explicit runtime
settings and changing them requires a new campaign. Unsupported tasks are not
silently omitted. Before a full experiment, native task compatibility still
needs a separately authorized check on the target hardware.

Sources: [official Python harness](https://github.com/microsoft/SWE-bench-Live/tree/python-only),
[dataset](https://huggingface.co/datasets/SWE-bench-Live/SWE-bench-Live),
[upstream evaluation protocol](https://github.com/microsoft/SWE-bench-Live/blob/main/evaluation/README.md).
