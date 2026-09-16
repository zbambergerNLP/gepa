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
```

Preflight and submission automatically record the latest committed `HEAD`;
no manual source hash is required. Preflight requires the consolidated branch
and rejects uncommitted changes. An optional `HOTPOTQA_SOURCE_COMMIT` asserts
a particular expected revision. Keep the recorded SHA fixed during a campaign.
Source, checkpoint, runtime, or experiment
changes require a fresh campaign. HotPotQA's consolidated run contract is schema 27.

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
hosts. Preflight is read-only and uses the current consolidated branch tip.
It checks source, permissions, SSH, CUDA, storage, and serving locks.

## Prepare the exact artifacts

Login nodes handle brief operations and submissions. `della-vis1` handles
builds and downloads. Allocated H200 GPUs run inference and optimization
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
| Serving | TP1 / DP1 / one API server | TP4 / EP4 / DP1 / one API server |
| Della allocation | 1 H200, 8 CPUs, 128G | 4 H200s on one node, 32 CPUs, 768G |
| Weights / KV cache | BF16 / BF16 (`auto`) | Native FP4 experts + FP8 / FP8 |
| Engram tables | Not applicable | Explicit CPU offload |
| Active sequences | Calibrate 1, 2, 4 | Calibrate 1, 2, 4 |
| Context / output caps | 262,144 / 65,536 solver; 32,768 optimizer | 262,144 / 65,536 solver; 131,072 optimizer |
| Solver reasoning budget | Native model termination | 32,768 within the 65,536 total output cap |
| Thinking effort | `xhigh` | `100` |
| Approved initial pilot workers | 12 | 4 |

The initial client-worker values incorporate Zach's observed queue timeout.
The smaller GPU allocations follow the checkpoint-byte sizing review and await
empirical qualification. Client workers may queue behind one active sequence. Check
throughput and timeouts on training examples, then freeze the chosen concurrency
across all six cells per model. This calibration remains pending. The
September 13 qualification now compares 1, 2, and 4 active requests on the same 12 training questions. This revises the earlier single-sequence profile.

Temperature is 1.0 and top-p is 0.95 for both models and all roles. DeepSeek uses
numeric effort 100, matching its published instruct/agentic evaluations.
Requests have a 3,600-second deadline shared by at most three transient-error
attempts, with 1/2-second backoff and attempt logs; SDK retries stay disabled.
The 262,144-token server context is Zach's Della configuration, below the model
card's 1M capability. Output caps remain our approved practical budgets.
See the [provider review](../common/temperature_policy.md).

HotPotQA retains 150/300/300 examples, seed 0, the pinned DSPy two-stage program,
Wiki-2017 BM25 k=7, and minibatches of three. Evaluation and response caches stay
off. Text feedback remains enabled; repeated context is preserved without
deduplication, as recorded by reflection-context policy version 2. Character limits
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
exact campaign runtime before six optimization jobs. A short Slurm controller
advances the chain only after each allocation completes successfully.
Its marker is written only on success. A failed canary or native-tool preflight
cannot freeze campaign locks. Every optimization job verifies source, model
bytes, environment, H200 hardware, and native tool calls before training.

The approved HotPotQA calibration pilot then evaluates, for each model:

1. Three training questions to check the complete two-stage task pipeline.
2. One real cycle for each of the four optimizer methods.
3. Twelve fixed training questions to compare active-request limits 1, 2, and 4.
4. All 150 training questions at the selected profile to measure throughput,
   timeouts, token usage, and output cutoffs with the initial prompts.

Use the pinned training split, Wiki-2017 BM25 k=7, and approved model settings.
Start with 12 Qwen workers / 4 DeepSeek workers. Keep validation and test
examples outside calibration, and review the evidence before freezing the
runtime settings across the six experiment cells per model. These stages
evaluate the initial prompts without optimizing them. The dedicated
`submit_hotpotqa_pilots.sh` launcher runs smoke and the four optimizer checks before throughput and full calibration on
each model. It uses the production serving gates and writes isolated outputs
under `outputs/hotpotqa-pilots/<pilot-id>/<model>/`. It does not freeze production
campaign locks. Qwen first verifies all four native edit tools; DeepSeek runs
its 20-attempt canary. Successful checks are recorded for the exact runtime.
Live pilot execution remains pending.

The approved pilot acceptance criteria are:

- Every question completes the pipeline and returns a usable prediction.
- Unresolved provider or parsing errors, and output cutoffs, require
  investigation before proceeding to the full campaign.
- Record EM/F1 as baseline measurements with no minimum accuracy threshold.

Check execution evidence separately from scores: the normal evaluator assigns
zero to malformed task output, which is a pilot reliability issue. An ordinary
wrong answer remains a valid baseline result.

Also include a small training-only optimizer-flow check per model for `vanilla`,
`react_v2`, `react_v2_random`, and `action`. Exercise real task feedback through
reflection/proposal and candidate reevaluation, including each method's actual
Manifestor, Controller, and Editor stages where applicable. Save the stage
evidence and the resulting normal acceptance or rejection decision separately
from the initial-prompt calibration measurements and production runs.

Run one completed proposal-and-reevaluation cycle on the normal three-example
training minibatch per method/model combination: four methods times two models
gives eight checks. Standard and doubled budgets share this process coverage;
their execution paths are the same. Stop after the completed cycle regardless
of metric direction or candidate acceptance. The three-question and full
150-question initial-prompt calibration stages remain separate.

The check passes when that process completes correctly. Tied or lower EM/F1,
wrong answers, and rejected candidates are valid outcomes, with no minimum
score or improvement requirement. Recovered editor tool errors are acceptable
when normal feedback allows completion. Investigate unresolved execution errors
and missing stage evidence; do not retry for an improved metric or change the
optimizer's acceptance rule. Keep validation/test examples out, and start every
production cell from the approved initial prompts, not a pilot revision.
The checks use the production optimizer builders, with only the three training
examples as both discovery and diagnostic evaluation data and a one-proposal
stop condition. `optimizer-cycle.json` retains callback evidence;
`optimizer-pilot-complete.json` verifies all required stages. A perfect-batch
skip does not count as an exercised cycle. The serving canary alone does not
exercise this complete flow. Live qualification remains pending.

Keep client workers fixed at 12 Qwen / 4 DeepSeek while comparing server
active-request limits 1, 2, and 4. Rank completed questions per hour using
`examples.hotpotqa.batching_report`; inspect memory, queueing, preemptions, and
request errors before selecting a profile. A timing winner alone is not production qualification. If queueing causes timeouts,
reduce concurrency and repeat the affected model's 150-question training
pilot. Investigate parsing failures and output cutoffs separately. Freeze the
successful setting across that model's six experiment cells.

Use one interactive allocation at a time, following the user's updated resource
instructions. Qwen and DeepSeek qualification runs are sequential; concurrency
calibration compares active requests within each model server. Record each job
ID, inference start/end times, throughput, errors, output cutoffs, and resource
peaks. Cross-model overlap has not been qualified.

After artifact preparation, commit the exact source and use a fresh pilot ID:

```bash
export HOTPOTQA_CAMPAIGN_ID=<new-pilot-id>
scripts/della/submit_hotpotqa_pilots.sh --dry-run
scripts/della/submit_hotpotqa_pilots.sh
HOTPOTQA_JOB_KIND=pilot scripts/della/fetch_hotpotqa_results.sh
```

The dry run prints the two submissions without contacting Della. The actual
launcher submits independent model arms; resource availability and canary
timing determine overlap. The fetch writes `pilot-report.json` under
`outputs/hotpotqa-pilot-fetches/<pilot-id>/<source-commit>/`. It validates
calibration records and optimizer stage evidence, retains incomplete results,
and measures overlap from physical request intervals, including queue time.
Idle gaps and merely queued jobs do not count as concurrent requests. Review
the report and the preserved Slurm/model-server logs before freezing settings.
Original provider records retain job IDs, failed attempts, and unknown usage;
completed question records are reused only when resuming the same pilot.

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

These commands submit independent model chains. Apply the schedule selected
from the pilots: submit both chains for concurrent scheduling, or wait for one
model's chain to finish before submitting the other for sequential scheduling.
Each model's six ablations and their test evaluations remain sequential.

Qwen requests one H200, 8 CPUs, and 128G; DeepSeek requests four H200s, 32 CPUs,
and 768G on one `ailab` node, with Engram tables explicitly offloaded to CPU.
The host-memory request includes headroom after the 640G pilot encountered
memory reclaim pressure without OOM.
Qwen standard
caps are 72 hours; expanded and DeepSeek caps are 144 hours. These are limits,
not estimates. Logs live at `$SCRATCH_BASE/logs/hotpotqa/<campaign>/<commit>/`.
Each allocation records its CUDA-visible GPU inventory and VRAM/utilization every
five seconds in `gpu-inventory-<job>.json` and `gpu-memory-<job>.csv`. Physical
UUIDs stay outside the resumable experiment identity. vLLM INFO logs retain
weight, KV-cache, and startup allocation details; reserved VRAM alone is not the
working memory needed by a request.

The approved allocation-recovery policy is to request a continuation
automatically after scheduler-confirmed allocation time expiry, using a
verified checkpoint with newly persisted work from that allocation. Completed
evaluations or iterations count as progress even when scores do not improve
or candidates are rejected. Preserve the same logical run, source/campaign,
data, model, runtime settings, saved state, and remaining original budget.
Keep downstream ablations waiting for this cell's optimization and test to
finish successfully. Stop for unresolved execution errors, cancellation,
missing/incompatible checkpoints, or no saved progress; retain recovery usage.
The launcher implements this through `examples.common.slurm_continuation`.
Each worker is submitted held until its `afterany` controller is recorded,
then released. The controller reads the allocation's final `sacct` state. Only
`TIMEOUT` plus new hash-verified work permits another allocation with the same
command and environment. Successful completion advances to the next cell;
errors, corruption, ambiguous submission replies, and duplicate starts stop
without blind resubmission. Plans and allocation history live in the campaign
log directory, with separate provider usage identified by allocation job ID.
These failure paths are tested locally; live Slurm recovery remains unverified.

For a stopped plan, inspect its state and queued/held jobs before any manual
resubmission; never start a second plan over an active logical run. Fetch and
validate completed experiment evidence with
`scripts/della/fetch_hotpotqa_results.sh` under the same campaign ID. Its output
is `outputs/hotpotqa-campaigns/<campaign>/<commit>/`.

PR #62's previous job numbers and artifact-readiness claims are historical.
Check the actual connection and runtime before using this consolidated source.

## Interactive qualification after the September 13 measurements

DeepSeek optimizer roles now use a 131,072-token ceiling, uniformly across
vanilla, FOREST, random Controller, and action-only. Following observed training-pilot
cutoffs and user approval, both models' solver calls use 65,536; Qwen optimizer
roles retain 32,768. The September 15 Qwen full pilot reached its previous 32,768
solver cap in the first evidence summary and returned no parseable summary.
These revised caps require fresh qualification. The 262,144 context and provider
sampling settings are unchanged. Run contracts use schema 28; pilot protocol
uses version 2.

The September 16 investigation retained two DeepSeek query calls that spent
all 65,536 output tokens on repeated reasoning without an answer. DeepSeek
solver calls now use vLLM's `thinking_token_budget=32768`, leaving the remaining
output allowance for the structured answer. This is a resource choice justified
by training diagnostics, not a provider recommendation. Optimizer calls retain
their existing limits and have no separate thinking budget. Qwen's earlier
empty response was not captured in full, so its cause remains unconfirmed;
its generation settings remain unchanged pending qualification.

The exact pinned DeepSeek vLLM also had a numerical bug at forced reasoning
boundaries: its `1e9` forced logit combined with top-p 0.95 masked every token
and emitted repeated BOS tokens. The launcher loads the guarded
`serving/safe_thinking_budget.py` repair before Triton compilation. It forces
the required token with probability one, preserves ordinary sampling, checks
the exact vLLM version and original source hash, and leaves installed package
files unchanged. Its file hash is part of serving identity; the whole repair
is pinned by source identity. Qwen does not load this DeepSeek-specific repair.
Each DeepSeek startup first runs a 256-token probe that verifies immediate
reasoning closure and nonempty final content. Failed probes stop qualification.
Empty or length-limited provider responses now retain their request and full
reasoning in private `provider-failures` artifacts linked from attempt logs.
Neither diagnostic continuations nor repeated trials replace failed evaluations.
Fresh qualification is required before launching ablations with these changes.
Serving and resource logs include the model and allocation step in their
directory, preserving both models' logs when one interactive allocation is reused.

Set `HOTPOTQA_JOB_KIND=pilot HOTPOTQA_PREPARE_ONLY=1` with a fresh campaign ID
and `MODEL_PROFILE`, then run `scripts/della/submit_hotpotqa.sh`. This verifies
and stages the clean current source and prints `INTERACTIVE_EXPORT_FILE` without
submitting jobs. Inside an approved allocation, run the staged
`scripts/della/remote/run_hotpotqa_interactive.sh <export-file> all` through
`srun`. Stages `smoke`, `optimizer`, `throughput`, and `full` are individually
resumable. Use `preliminary` to run smoke, all optimizer checks, and throughput
with one server startup, stopping before the full calibration. This supports
batching selection before the 150-question run. The complete qualification plan
uses one nine-hour `salloc` with four H200s, 32 CPUs and 768G host memory, releasing
it early when finished. After DeepSeek batching selection, run Qwen's full pilot
before the selected DeepSeek full pilot; Qwen uses an exact one-H200/eight-CPU/128G
step within that allocation. Preserve a failed full pilot and run the other model
when time permits; either failure prevents qualification. This replaces the
earlier 55-minute partial-pilot reservations. The DeepSeek interactive pilot runs its required
20-attempt runtime canary before inference when its exact-runtime marker is absent.

Store each batching profile under its own pilot ID. Compare the three server
settings on training only, then finish
the complete optimizer and 150-question checks at the selected setting.
`--stage optimizer --method <name>` can resume a single method through the
Python CLI against an already running endpoint. Completed checks are hash-verified
and reused only for the same run identity. Standard and doubled budgets share
method coverage; production candidates always start from the original prompts.

The bounded one-edit native diagnostic now requests `tool_choice=none` and an
explicit finish after spending its configured edit allowance. Its strict two-turn
check still requires a valid edit and finish. Production's unlimited tool/turn
policy is unchanged. Keep failed historical probes as evidence.

The `all` pilot stage also checks native error recovery before optimization.
It replaces the first returned replacement target with a deliberately missing
target, then requires the model to receive the actual editor error, correct its
edit, and explicitly finish. Evidence identifies this as a controlled injected
fault, not a spontaneous model error. The isolated probe allows four turns and
three tool calls; production editor limits remain unchanged.

Qualify interruption and resume with saved records, optimizer checkpoints, and
response journals before long runs. Check that original metric budgets remain
unchanged and completed work is not rerun. Compiler caches remain on scratch;
model-response and evaluation caches remain disabled. See
`DELLA_QUALIFICATION_2026-09-13.md` for the original measurements; the new batching
comparison, larger-token DeepSeek cycle, full calibration, and live recovery
are not established by that earlier report.
