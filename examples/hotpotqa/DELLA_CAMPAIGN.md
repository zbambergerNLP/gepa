# HotPotQA on Della

Use `codex/consolidated-della-experiments`, which combines the reviewed HotPotQA
and Terminal-Bench work with [Zach's PR #62](https://github.com/zbambergerNLP/gepa/pull/62).
The [integration record](../../docs/experiment-consolidation.md) identifies the
input commits, conflict resolutions, and deferred decisions. The old `169ddda`
runbook commit predates these decisions and is not a launch target.

These are instructions for future operations. Consolidation does not submit
jobs. Start with HotPotQA; Terminal-Bench's Apptainer integration is paused.

## Source and connection

Run the laptop launchers from the reviewed, clean consolidated checkout. They
stage `git archive HEAD` under `$REMOTE_DIR/sources/<commit>` and verify its
manifest. No branch switching is needed between experiment and Della tooling.

```bash
cd /path/to/gepa-consolidated
git status --short --branch
export HOTPOTQA_SOURCE_COMMIT="$(git rev-parse HEAD)"
```

Keep this SHA fixed during a campaign. Source, checkpoint, runtime, or experiment
changes require a fresh campaign. HotPotQA's consolidated run contract is schema 26.

Reuse the working Della connection configuration. If this checkout's ignored
`scripts/della/.env` is absent, create it from `.env.example`, fill in the real
user, hosts, allocation, scratch path, and model path, and set mode 600. Do not
overwrite an existing configured file. Credentials are never committed.
Needed local commands: `git`, `ssh`, `rsync`, `sha256sum`, and `uv`.

The SSH connection must work with `BatchMode=yes` and verified host keys. If the
account requires interactive authentication, use the ControlMaster helper:

```bash
scripts/della/della_session.sh status
# Authenticate interactively only when a new master is needed:
scripts/della/della_session.sh open
scripts/della/preflight_hotpotqa.sh
```

The helper uses the existing SSH configuration: `ControlMaster auto`,
`ControlPath ~/.ssh/cm/%r@%h:%p`, and `ControlPersist yes` for both configured
hosts. Preflight is read-only and requires the explicit `HOTPOTQA_SOURCE_COMMIT`
above. It checks source, permissions, SSH, CUDA, storage, and serving locks.

## Prepare the exact artifacts

Login nodes handle brief operations and submissions. `della-vis1` handles
builds and downloads. Allocated eight-H200 nodes run inference and optimization
with Hugging Face offline mode. Keep environments, caches, and outputs on
`SCRATCH_BASE`; checkpoints use the configured shared `MODEL_STORAGE`.

Each model has its own hash-locked vLLM environment. Qwen uses `.serving-venv`
(vLLM 0.25.1 / Torch 2.11). DeepSeek V4.1 uses `.serving-venv-deepseek-v4.1-flash`
from `serving/requirements-deepseek-v4.1-flash-x86_64-linux-py312.txt`, including
vLLM commit `e77daef89e18e08321ae7b8b24827eedd5fe8673` (package version
`0.1.1.dev5+ge77daef89`), Torch 2.13, and prebuilt FlashInfer kernels for offline
GPU nodes. Both serving environments use Python 3.12.7 / Transformers 5.13.
GEPA uses Python 3.11.13 / uv 0.9.13 with pinned DSPy. HotPotQA does not need
a POSIT checkout. HoVer's shared model references are updated, but its separate
POSIT-based deployment is outside this HotPotQA campaign.

```bash
scripts/della/build_env.sh qwen3.8-27b deepseek-v4.1-flash
```

This syncs code, builds environments and datasets, and starts detached model
downloads on the visualization node. A successful launcher exit means downloads
started, not finished. The printed log must end with `MODELS_DONE`; each model
directory must contain its byte-verified `.gepa-model-integrity.json`.

The remote stages are `scripts/della/remote/setup_env.sh`,
`download_dataset.sh`, and `download_model.sh <profile>`. Run them from the
synced checkout with `SCRATCH_BASE`, `MODEL_STORAGE`, and any custom
`WIKI17_DIR` exported. Artifact locks prevent rebuilding in-use environments
and indexes; model-directory locks protect served checkpoints.

Installation and ordered CUTLASS reinstalls use committed requirements hashes.
The launcher/job check both the lock and realized package manifest. Code sync
preserves `.venv`, `.serving-venv*`, `.tools`, caches, snapshots, and outputs.
FlashInfer builds use the serving environment's CUDA headers first.

## Frozen experiment choices

| Setting | Qwen | DeepSeek |
| --- | --- | --- |
| Model, all roles | Qwen3.8-27B | DeepSeek-V4.1-Flash |
| Revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | `dba1be0a40aa45a94ad051997016db3960a90277` |
| Serving | TP1 / DP8 / eight API servers | TP8 / EP8 / DP1 / one API server |
| Active sequences | One per replica | One |
| Context / output cap | 262,144 / 16,384 | 262,144 / 16,384 |
| Thinking effort | `xhigh` | `100` |
| Approved initial pilot workers | 12 | 4 |

The approved starting values incorporate Zach's observed queue timeout. Check
throughput and timeouts on training examples, then freeze the chosen concurrency
across all six cells per model. This calibration remains pending. The
new V4.1 runtime uses one sequence per replica, as in Zach's branch.

Temperature is 1.0 and top-p is 0.95 for both models and all roles. DeepSeek uses
numeric effort 100, matching its published instruct/agentic evaluations.
Requests have a 3,600-second deadline shared by at most three transient-error
attempts, with 1/2-second backoff and attempt logs; SDK retries stay disabled.
The 262,144-token server context is Zach's Della configuration, below the model
card's 1M capability. Output caps remain our approved practical budgets.
See the [provider review](../common/temperature_policy.md).

HotPotQA retains 150/300/300 examples, seed 0, the pinned DSPy two-stage program,
Wiki-2017 BM25 k=7, and minibatches of three. Evaluation and response caches stay
off. Text feedback and exact deduplication remain enabled. Character limits
default to unlimited and can be set with `HOTPOTQA_TEXT_LIMITS_JSON`; model token
limits remain. See [shared decisions](../../scripts/della/README.md).

Final testing includes one shared starting-prompt baseline per model. Alongside
the first completed cell's held-out evaluation, run the original prompts once
over the same 300 test questions using that model's frozen task runtime. All six
cells reference those same baseline scores and report EM/F1 gains. This adds 300
question executions per model, accounted separately from optimization. Training
pilot scores continue to serve calibration only.

Baseline artifacts live under `outputs/hotpotqa-baselines/<identity-sha256>/`
and are fetched with the other run outputs. The identity includes the campaign,
source/runtime, starting prompts, model settings, retriever, and exact data.
Completed per-question records are reused on resume; changed identities require
separate evidence. The result analyzer verifies each cell's baseline and gains,
then checks that all of a model's cells share the same reference evaluation.

## Verify before optimization

Zach's independent diagnostic records the request, rendered prompt, reasoning,
output, usage, and four edit-tool probes:

```bash
scripts/della/submit_deepseek_smoke.sh submit
scripts/della/submit_deepseek_smoke.sh fetch <job-id>
```

It uses the approved V4.1 serving flags and shared edit probes, including explicit
`<finish>`. It does not write a campaign qualification marker.

The production DeepSeek chain retains its mandatory 20-attempt canary on the
exact campaign runtime before six optimization jobs, with `afterok` dependencies.
Its marker is written only on success. A failed canary or native-tool preflight
cannot freeze campaign locks. Every optimization job verifies source, model
bytes, environment, H200 hardware, and native tool calls before training.

The approved HotPotQA calibration pilot then evaluates, for each model:

1. Three training questions to check the complete two-stage task pipeline.
2. All 150 training questions to measure throughput, timeouts, token usage,
   and output cutoffs with the initial prompts.

Use the pinned training split, Wiki-2017 BM25 k=7, and approved model settings.
Start with 12 Qwen workers / 4 DeepSeek workers. Keep validation and test
examples outside calibration, and review the evidence before freezing the
runtime settings across the six experiment cells per model. These stages
evaluate the initial prompts without optimizing them. Pilot execution remains
pending; the production optimization launcher is not a training-only pilot.

The approved pilot acceptance criteria are:

- Every question completes the pipeline and returns a usable prediction.
- Unresolved provider or parsing errors, and output cutoffs, require
  investigation before proceeding to the full campaign.
- Record EM/F1 as baseline measurements with no minimum accuracy threshold.

Check execution evidence separately from scores: the normal evaluator assigns
zero to malformed task output, which is a pilot reliability issue. An ordinary
wrong answer remains a valid baseline result.

Keep 12 Qwen / 4 DeepSeek workers if the pilot passes; the approved plan does
not include a search for higher parallelism. If queueing causes timeouts,
reduce concurrency and repeat the affected model's 150-question training
pilot. Investigate parsing failures and output cutoffs separately. Freeze the
successful setting across that model's six experiment cells.

## Campaign and results

Per model: standard `vanilla`, `react_v2`, `react_v2_random`, and `action` at
6,871 metric calls; then independent expanded `vanilla` and `react_v2` at
13,742 calls. There are 12 optimization cells across both models. Expanded runs
start from the initial prompt.

After source, training calibration, and runtime checks are reviewed, choose a
new campaign ID and use the production launcher:

```bash
export HOTPOTQA_CAMPAIGN_ID=<new-campaign-id>
MODEL_PROFILE=qwen3.8-27b scripts/della/submit_hotpotqa.sh
MODEL_PROFILE=deepseek-v4.1-flash scripts/della/submit_hotpotqa.sh
```

Jobs request one node, eight H200 GPUs, 64 CPUs, and 768G on `ailab`. Qwen standard
caps are 72 hours; expanded and DeepSeek caps are 144 hours. These are limits,
not estimates. Logs live at `$SCRATCH_BASE/logs/hotpotqa/<campaign>/<commit>/`.

Resume only with the same source/campaign/model/settings, using
`BUDGET_PROFILE=standard|expanded CONDITION=<cell>`. Inspect orphaned dependency
chains before resubmission. Fetch and validate completed evidence with
`scripts/della/fetch_hotpotqa_results.sh` under the same campaign ID. Its output
is `outputs/hotpotqa-campaigns/<campaign>/<commit>/`.

PR #62's previous job numbers and artifact-readiness claims are historical.
Check the actual connection and runtime before using this consolidated source.
