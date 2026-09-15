# Paired benchmark models

Every benchmark arm uses the same model for execution and optimization:

Every model role uses the shared [provider retry policy](../../examples/common/provider_retries.md):
at most three attempts for temporary provider failures, with per-attempt logs
and nested SDK retries disabled. Completed answers and benchmark tasks are not
retried by this policy.

Evaluation-result caching is disabled for HotPotQA and TB2.1 across all methods
and budgets. HotPotQA also disables DSPy's disk and memory response caches, so
each newly requested evaluation runs the task model again. Completed checkpoint
records and optimizer response journals remain available for recovery; they do
not serve unrelated new evaluations. Cache settings are part of run identity,
so HotPotQA's earlier cache-enabled checkpoints require a fresh campaign. The
6,871/13,742-call budgets remain unchanged and now fund uncached evaluations.

HotPotQA training minibatches use a separate random stream initialized with the
experiment seed. Every method and budget therefore sees the same sequence of
shuffled training epochs for the same ordered dataset and seed, independent of
parent selection or reflection draws. Larger budgets continue that sequence;
metric-call budgets do not guarantee the same number of optimization iterations.
Checkpoints save the permutation, cursor, and private RNG state. Run-contract
schema 26 records this policy, the starting baseline, and serving provenance; use a fresh
campaign for older checkpoints.

Each HotPotQA model has one shared unoptimized starting-prompt baseline on the
same 300 test examples used by all six ablations. It runs alongside the first
completed ablation's test; later cells reuse that recorded reference and report
EM/F1 gains. Baseline task executions are separate from optimization budgets.
`outputs/hotpotqa-baselines/` contains the frozen identities and resumable
per-question evidence, which the existing fetcher includes and analyzer verifies.

| Arm | Student and proposer | Serving |
| --- | --- | --- |
| Qwen | `Qwen/Qwen3.8-27B` | Self-contained local vLLM |
| DeepSeek | `deepseek-ai/DeepSeek-V4.1-Flash` | Self-contained local vLLM |

The consolidated [Della runbook](../../examples/hotpotqa/DELLA_CAMPAIGN.md)
and [Della skill](../../.claude/skills/della/SKILL.md) include Zach's setup,
SSH, smoke-test, and launch fixes. Qwen uses the committed
vLLM 0.25.1 lock; DeepSeek V4.1 uses Zach's separate commit-wheel lock. A POSIT checkout is no longer required for HotPotQA.
V4.1 is the approved DeepSeek model for both benchmarks.

HotPotQA requests have a 3,600-second deadline shared across the bounded
provider attempts. Initial worker defaults are 12 for Qwen and 4 for DeepSeek;
training calibration must precede freezing them for a comparison.

For both HotPotQA and TB2.1, try overlapping Qwen and DeepSeek execution during
their training pilots on separate allocations. Record actual inference overlap
and the existing pilot reliability/throughput measurements. Decide model-arm
scheduling from those results: concurrent if qualified, otherwise sequential
with the required full pilots completed in that mode. This decision remains
pending until live pilot review and is separate from each arm's worker count.
Follow the benchmark pilot protocols; ablations within a model stay sequential.
Both protocols also include an approved training-only optimizer-flow check per
model and distinct method, covering both Terminal-Bench scopes. Correct process
completion qualifies even with unchanged/worse scores or a rejected candidate.
Use one completed proposal-and-reevaluation cycle on a three-example training
minibatch per combination: eight HotPotQA checks and 16 Terminal-Bench checks.
Standard and doubled budgets share these checks.
Both check runners are implemented, with offline tests of stage coverage and
acceptance of worse or tied scores. Live qualification remains pending; see
the benchmark runbooks for coverage and acceptance criteria.

Both benchmarks also have an approved automatic allocation-continuation policy:
resume after scheduler-confirmed allocation time expiry when a verified
checkpoint contains new saved work, preserving run identity and the remaining
budget. Metric improvement is not required. Unresolved errors, cancellation,
missing/incompatible checkpoints, or no saved progress stop continuation.
The shared Slurm controller and checkpoint hooks are implemented. HotPotQA
submission is wired to the controller; live allocation recovery remains
unverified. Terminal-Bench's Della backend remains paused.

DeepSeek is pinned to revision `dba1be0a40aa45a94ad051997016db3960a90277`.
The V4.1 arm uses the exact vLLM commit wheel `e77daef89` (version
`0.1.1.dev5+ge77daef89`) with native `deepseek_v41` tokenizer, reasoning, and
tool parsers. It runs TP4/EP4 on four H200s (32 CPUs, 512G), with Engram CPU offload
explicit, one active sequence, FP8 KV cache and automatic block size, no speculative decoding, and prebuilt offline FlashInfer kernels.

Provider guidance sets temperature 1.0 and top-p 0.95 for every role in both
models. Thinking is explicit: Qwen `xhigh`; DeepSeek V4.1 numeric effort 100.
The [provider review](../../examples/common/temperature_policy.md) records the
sources. Context is 262,144 for both Della servers. HotPotQA output caps are
32,768 for Qwen solver and optimizer calls, and 65,536 solver / 131,072 optimizer
for DeepSeek. TB2.1 uses 32,768 per call. These practical runtime limits are below
the providers' largest recommended budgets and require training-only review.

The FOREST ReAct editor has no assistant-turn or tool-call limit in HotPotQA
or TB2.1. It may make multiple edits within the Controller-selected section,
all serving the same semantic action and Manifestor steering, then explicitly
emit `<finish>`. Each successful edit returns the latest section text. With the
minimal tool basis, each replacement or move must complete its delete/insert
pair before another operation or finish. The protocol is recorded in run
contracts; older checkpoints require a fresh run. The runtime canary retains
its small diagnostic budget to verify one literal edit followed by finish.

All optimizer character limits default to unlimited and are independently
configurable through the shared [text-limit settings](../../examples/common/text_limits.md).
This covers complete prompts and skills, total candidate size, complete optimizer
requests, selector targets, feedback, Manifestor traces and steering, and saved
diagnostic text. Set `HOTPOTQA_TEXT_LIMITS_JSON` for this launcher or pass
`--text-limits` to the Python CLI. Model context and per-call output limits still
apply. Resolved settings are recorded in run contracts for resume validation.

After configuring `scripts/della/.env` from `.env.example`, prepare artifacts,
complete the pilots, and apply their reviewed schedule when submitting:

```bash
scripts/della/build_env.sh
export HOTPOTQA_CAMPAIGN_ID=<fresh-pilot-id>
scripts/della/submit_hotpotqa_pilots.sh --dry-run
scripts/della/submit_hotpotqa_pilots.sh
HOTPOTQA_JOB_KIND=pilot scripts/della/fetch_hotpotqa_results.sh
# After completing the pilots and reviewing their scheduling evidence:
export HOTPOTQA_CAMPAIGN_ID=<fresh-experiment-id>
MODEL_PROFILE=qwen3.8-27b scripts/della/submit_hotpotqa.sh
# For sequential scheduling, wait for Qwen's chain to finish first.
MODEL_PROFILE=deepseek-v4.1-flash scripts/della/submit_hotpotqa.sh
```

Each HotPotQA arm contains the existing six optimization cells. DeepSeek first
runs the multi-tool canary; a failed canary prevents its campaign from starting.
Use a fresh campaign ID after changing models or serving environments. Previous
GLM results and checkpoints cannot be resumed as DeepSeek runs.

Pilot fetches write `pilot-report.json` with stage and optimizer completion,
usage, allocation IDs, and measured full-stage request overlap. The report
preserves partial evidence and leaves the scheduling decision for review.
Pilot results and candidates never seed production runs. A persisted
continuation plan rejects duplicate submissions; inspect its status and queued
jobs before attempting manual recovery.

HotPotQA reflection uses the configured model context window without an
additional 8,000-character Manifestor trace cap. All supplied passages,
reasoning, task outcomes, and feedback remain intact, including repeated text
and log lines. FOREST receives feedback in its dedicated field and per-example
traces. Original evaluation records are unchanged. Context overflow remains a
provider error that stops the run; it does not silently truncate evidence.
Reflection-context policy version 2 records the removal of our deduplication;
earlier incompatible checkpoints require a fresh campaign.
