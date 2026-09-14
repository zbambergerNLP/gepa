# Consolidated experiment decisions

Local branch: `codex/consolidated-della-experiments`.
Use this branch's latest clean commit for preflight and submission; both record
the source revision automatically. A manually supplied source hash is optional.

This integration combines the complete reviewed HotPotQA/Terminal-Bench branch
`af602f3d7a7949f199883ff5a6a34afbd1580fcf` (PR #60, including HotPotQA PR #59 at
`8f7214fa9457fd853d607afe307ba236306581dc`) with Zach's PR #62 at
`0f2ffa6ee7cb53680a2ea87506e848d7c5c01de5`. Both histories remain ancestors of
the merge. The existing branches are preserved; consolidation is local.

## Resolved differences

| Area | Consolidated behavior |
| --- | --- |
| Models | Adopt Qwen3.8-27B and DeepSeek-V4.1-Flash, homogeneous across all roles, as explicitly approved during consolidation. Update both benchmarks and shared catalog consumers. |
| Serving environment | Adopt Zach's separate hash-locked vLLM environments: 0.25.1 for Qwen, exact commit wheel e77daef89 for V4.1. Replace HotPotQA's POSIT provenance with serving-lock and realized-environment hashes. HoVer shared model references are updated; its separate deployment remains outside this campaign. |
| Serving settings | Qwen TP1/DP1/one API/one sequence on 1 H200 (8 CPUs, 128G) and DeepSeek V4.1 TP4/EP4/DP1/one API/one sequence on 4 H200s (32 CPUs, 512G), with explicit Engram CPU offload, context 262,144, bfloat16 activations, checkpoint FP8/FP4 weights, FP8 KV/automatic block size, native V4.1 parsers, and no speculative decoding. |
| Workers | Confirmed for the initial HotPotQA pilot: 12 Qwen / 4 DeepSeek. Check throughput and timeouts on training examples, then freeze the chosen concurrency across all six cells per model. Calibration remains pending. |
| Request timeout/retries | Adopt 3,600 seconds per HotPotQA logical request, shared by our maximum of three transient-error attempts. Keep nested SDK retries zero, 1/2-second backoff, and per-attempt logs. |
| Runtime verification | Keep our mandatory exact-runtime DeepSeek canary and Zach's independent transcript/tool diagnostic. Both use the current editor probes. Write campaign locks only after canary/native-tool success. |
| Setup | Adopt checked-in remote stages, detached downloads, scratch storage, SSH helpers, CUDA-header precedence, and rsync environment exclusions. Propagate custom Wiki-2017 paths and use this checkout's source in uninstalled remote environments. |
| Wiki-2017 | Adopt corrected extracted-corpus size, 1,780,742,620 bytes. Corpus/archive hashes, document count, retrieval parameters, and split stay fixed. |
| Text/editor behavior | Preserve configurable unlimited-by-default character limits, repeated reflection evidence without deduplication, actionable tool-error feedback, repeated selected-section edits, and explicit finish. |
| Run identity | HotPotQA schema 27 records the shared starting-baseline protocol, timeout, and serving provenance. Terminal-Bench schema 32 records test-after-each-ablation timing; runtime freeze and pilot schema 9 are preserved. Old runtime contracts cannot be silently resumed. |
| Della skill/runbook | Include and reconcile PR #62's skill and runbook with the current decisions. Remove obsolete instructions to launch the old commit; use the approved V4.1 arm. |

The user explicitly approved V4.1 during integration. Its published instruct
and agentic evaluations use temperature 1.0, top-p 0.95, and effort 100; these
replace the older V4 role-specific top-p values and named effort. The Della
context is 262,144, as in Zach's runtime; the provider advertises 1M. Keep the
approved smaller output caps. See the provider review for sources.

## Decisions preserved across benchmarks

- Within each benchmark, every ablation and model arm uses identical pinned
  data and exact ordered train/validation/test examples. Match content and
  source revisions as well as IDs and counts. This is also required for future
  benchmark integrations; a method, scope, or budget must never resample splits.
- Evaluate the validation-selected winner after each ablation finishes, using
  the same held-out test set. Freeze each winner before its test calls, and
  keep test scores out of all later optimization and configuration decisions.
- Keep a shared unoptimized starting baseline per benchmark/model, using the
  same test examples and task runtime as its ablations. Preserve each benchmark's
  repetition protocol: one test pass for HotPotQA, three for Terminal-Bench.
- Do not add bootstrap confidence intervals by resampling questions or tasks.
  Report HotPotQA EM/F1 and baseline gains from its single test pass, without a
  claim about rerun variability. For Terminal-Bench, retain the three actual
  test repetition scores, their mean, and sample standard deviation.
- The model executing the benchmark is also the optimizer model within each arm.
- Provider guidance determines role-specific sampling and reasoning: temperature
  1.0 and top-p 0.95 for both models; Qwen xhigh and DeepSeek V4.1 numeric effort 100.
- HotPotQA output limits are 32,768 solver / 32,768 optimizer for Qwen and
  32,768 solver / 131,072 optimizer for DeepSeek. Terminal-Bench uses 32,768 per call.
  Context limits remain 262,144 Qwen and 262,144 DeepSeek.
- Model context/output limits still apply when character limits are unlimited.
  All relevant character limits remain independently configurable.
- Controller selection is a Verbalized-Sampling-based 90% model-probability /
  10% uniform-positive-choice adaptation. Zero-probability choices are excluded.
- ReAct editors may correct tool errors and make multiple edits within the
  selected section/action; no fixed turn cap; explicit finish is required.
- Reflection preserves supplied task evidence, including exact repeated text,
  paragraphs, log lines, feedback, document bodies, and Harbor copied history.
  Remove our added deduplication and trace projection for both benchmarks.
  Reflection-context policy version 2 and Terminal-Bench feedback version 5
  distinguish this behavior from older runs. Native Harbor incremental terminal
  output, summarization, and usage accounting remain as previously configured.
- Evaluation caching is off. HotPotQA DSPy response caches are off. Recovery
  checkpoints and optimizer response journals remain supported.
- For both HotPotQA and TB2.1, automatically request a continuation allocation
  after scheduler-confirmed cluster allocation time expiry, provided the run
  has a verified recoverable checkpoint and newly persisted work since the
  allocation began. Progress means completed evaluations or iterations, not
  improved metrics or accepted candidates. Resume the same logical run with
  the same source, campaign, data, model, and material runtime settings; retain
  saved optimizer/random-sampler state and the remaining original budget.
  Preserve completed evaluation evidence and record recovery costs separately.
  Keep dependent ablations waiting until the current optimization and its test
  finish successfully. Stop for unresolved execution errors, user cancellation,
  missing or incompatible checkpoints, or no saved progress. Provider retries
  and official Terminal-Bench task-timeout scoring retain their own policies;
  they do not trigger allocation continuation. Shared checkpoint sealing and
  the Slurm continuation controller are implemented for both benchmark entry
  points. The HotPotQA launcher uses the controller; live allocation recovery
  remains unverified. Terminal-Bench's Della backend work remains paused.
- Training batches use their own seeded random stream, independent of method
  decisions; larger budgets continue the same batch sequence.
- Add a small training-only optimizer-flow check for each model and distinct
  method (`vanilla`, `react_v2`, `react_v2_random`, `action`), covering both
  Terminal-Bench text scopes with `system_prompt` first. Exercise real task
  feedback, reflection/proposal, and candidate reevaluation, including the
  Manifestor, Controller, and Editor where used by the method. Record the
  resulting normal acceptance or rejection decision. A tied or lower score,
  wrong task answer, or rejected candidate is not a pilot failure; no minimum
  score or improvement is required. Recovered tool errors are acceptable when
  the normal feedback path lets the process complete. Unresolved execution
  errors or missing stage evidence require investigation. Do not retry for a
  better metric or change the optimizer's acceptance rule to pass the check.
  Keep these diagnostics separate from initial-prompt throughput calibration
  and production budgets; preserve their artifacts, but start every production
  cell from the approved initial prompts. Validation and test data remain
  excluded. These checks are implemented in `examples.hotpotqa.pilot` and
  `examples.terminalbench.optimizer_pilot`; live execution remains pending.
  Saved evidence verifies each stage and records rejected candidates as valid
  process outcomes. A perfect-batch skip remains uncovered and requires review.
- Size each optimizer-flow check at one completed proposal-and-reevaluation
  cycle on the normal three-example training minibatch. Cover four methods
  and two models for HotPotQA (eight checks), and both text scopes as well for
  TB2.1 (16 checks). Standard and doubled budgets share this process coverage
  because they use the same execution paths. Stop after the completed cycle
  regardless of metric direction or candidate acceptance. Keep the separate
  three-example and full-training-set initial-prompt calibration stages.
- Determine model-arm scheduling separately for HotPotQA and TB2.1 from their
  training pilots. Attempt overlapping Qwen and DeepSeek execution on separate
  allocations, with independent servers and output directories. Record job IDs,
  actual inference overlap, resource availability, throughput, timeouts, errors,
  and output cutoffs. Concurrent submission alone does not establish overlap.
  Use concurrent production scheduling only when both arms pass their existing
  pilot checks while overlapping. Otherwise use sequential scheduling and
  complete the required full training pilots in that mode. Review and record
  the selected schedule before production; neither schedule is selected yet.
  This is separate from workers/tasks within an arm. Ablations within each
  model remain sequential, with testing after each ablation.

## HotPotQA

The pinned two-stage DSPy program, 150/300/300 train/validation/test split, seed
0, frozen Wiki-2017 BM25 k=7, and three-example reflection minibatches remain.
Per model: standard `vanilla`, `react_v2`, `react_v2_random`, `action` at 6,871
metric calls; expanded independent `vanilla`, `react_v2` at 13,742 calls.
This gives six cells per model, 12 total. Preserve single mutation, merge off,
Pareto parent selection, strict training improvement, and validation-based final
selection. Start operational work with HotPotQA after this consolidation review.
The scientific launcher runs one cell per job and evaluates test at the end of
that cell. Its preflight checks the pinned dataset revision and the ordered
content hash of each of the three splits; keep these checks for every ablation.

Each model's starting prompts receive one evaluation on the same 300 held-out
questions alongside its first completed ablation's test. All six ablations share
that baseline and report test EM/F1 gains against it. The baseline adds 300
question executions per model outside the optimization budget. It is separate
from the training-only pilot. Baseline contracts include initial prompts, exact
data, retrieval, model settings, and campaign/runtime identity; per-question
checkpoints support interruption recovery. Fetched analysis verifies the baseline
evidence and rejects inconsistent references within a model's campaign.

The approved training-only pilot has two stages for each model: first three
training questions to check the complete task pipeline, then all 150 training
questions to measure throughput, timeouts, token usage, and output cutoffs.
Both stages use the initial prompts and the fixed two-stage task program.
Validation and test examples are excluded from calibration. Review the pilot
evidence before freezing runtime settings across that model's six cells.
Approved acceptance criteria: every question must complete the pipeline with a
usable prediction. Unresolved provider/parsing errors or output cutoffs require
investigation before the full campaign. Record baseline EM/F1 without a minimum
accuracy threshold. A wrong answer is a valid baseline observation; malformed
task output is a reliability issue even when the normal evaluator scores it as
zero. Pilot execution remains pending.

The approved calibration stopping rule is to retain 12 Qwen / 4 DeepSeek
workers when the pilot passes. Do not search for higher parallelism. If
queueing causes timeouts, reduce concurrency and repeat the affected model's
150-question training pilot before freezing its setting. Other errors and
output cutoffs require their own investigation.

## Terminal-Bench

| Decision | Preserved setting |
| --- | --- |
| Dataset | TB2.1 only, 89 pinned tasks, split 30 train / 19 validation / 40 test; TB4 removed. |
| Adapter | Maintained Harbor port of GEPA's Terminus adapter, Harbor 0.22.0, PromptedTerminus. |
| Default scope | Whole initial instruction block as one editable `system_prompt`; run this scope first. Later prompts and skills remain fixed. |
| Ablation | `all_text`: all 14 prompt components and two skills editable together, executable code fixed. Both scopes start from identical runtime text. |
| Skills | Fixed skill inventory; editable names, descriptions, bodies in all-text; metadata-first discovery and on-demand loading. |
| Optimization | Six cells per scope/model; standard four epochs and doubled eight epochs, minibatch three, 120/240 training draws. Doubled runs start independently. |
| Selection | One mutation, merge off, skip perfect minibatches, strict improvement on training minibatch, full validation for initial/accepted candidates, GEPA instance-Pareto parent choice, best mean validation with earliest tie. |
| Task execution | One task attempt, no job retries, official task resources and timeouts at 1x, task-complete submission. |
| Failures | Genuine task failures score zero; verified task timeout uses actual reward. Infrastructure/provider failures abort rather than invent scores. |
| Context | Automatic summary enabled at the agreed remaining-context trigger; traces retained. Summary text editable only in all-text. |
| Budget accounting | Training passes normalized to each benchmark; validation and agent costs reported separately. More draws are not claimed to be equal compute. |
| Pilot | Per model, three training tasks then all 30 with the initial harness for usage/cutoff/timeout/throughput review; this calibration evidence may serve identical initial text in both scopes. Add the separate optimizer-flow check above for each method and scope, with no score-improvement requirement. |
| Runtime | Verified local model-server identity at pilot, optimization, resume, and final evaluation. Freeze concurrency/hardware/settings per model after training calibration. |
| Final evaluation | Evaluate each completed ablation before starting the next. Accumulate the common initial harness and 12 immutable validation winners in one matched test directory per model, three repetitions each; report execution variability. Exact data and splits must match throughout. |
| Execution backend | Existing Docker runner retained. Proposed Della Apptainer work explicitly paused; no backend compatibility or live pilot qualification is claimed. |

Canonical detailed protocol and commands:
[`terminal_bench_adapter/README.md`](../src/gepa/adapters/terminal_bench_adapter/README.md).

## Local verification

The pilot/recovery implementation passes **529 focused offline tests**,
Ruff on changed Python files, targeted Pyright on the new runner/recovery
modules, and shell syntax checks. Coverage includes real engine cycles with
lower/tied scores, all four HotPotQA methods and both Terminal-Bench scopes,
training-only execution, evidence integrity, allocation timeout recovery,
unchanged budgets, failed/duplicate submission handling, and read-only reuse
of shared model checkpoints. Live serving and recovery remain unqualified.
Server commands require explicit user approval, including read-only preflight.
No builds, downloads, or benchmark jobs were launched during implementation.

The context-preservation update passes **546 focused offline tests**, with one
optional test skipped, plus **28** tests in the pinned Harbor environment. Ruff
and targeted Pyright pass. Existing formatting differences are unchanged from
the previous committed source. Tests verify repeated evidence, complete copied
and nested Harbor traces, unchanged native summarization, and rejection of the
previous deduplication policy on resume. No benchmark jobs were launched.

The shared HotPotQA baseline update passes **252 focused offline tests** and
Ruff. The new baseline module and result analyzer have no Pyright errors;
HotPotQA's main module retains its two pre-existing selector/config type errors,
verified against the previous committed source. No benchmark jobs were launched.

The subsequent test-after-each-ablation update passes **512 focused offline
tests**, Ruff on the changed Python files, and Pyright on all three changed
Terminal-Bench entry points. Coverage includes optimize/test ordering, incomplete
run rejection, incremental resume, unchanged baseline evidence, exact split
matching, and HotPotQA content-hash drift checks. No benchmark jobs were launched.

The original consolidation was verified as follows:

- Installed the unchanged `uv.lock` with `dev`, `wiki17`, and the pinned
  `hotpotqa-task-program` dependency group. No GPU serving packages or model
  checkpoints were downloaded locally.
- Final local suite: **1,972 passed, six optional modules/tests skipped, 20
  credential-dependent tests deselected**. The separate pinned Harbor environment
  passes all **28** document/token-runtime tests, covering the two Harbor modules
  skipped in the GEPA environment. The live Docker smoke remains unrun.
- Ruff passes for the core package and every changed Python file. The normal
  core Pyright check passes with zero errors. An expanded check of the modified
  examples and new diagnostics reports the same 60 diagnostics as the original
  branch under the same dependencies, with none added by consolidation.
- Both model arms produce 12 Terminal-Bench dry-run commands: six
  `system_prompt` cells followed by six `all_text` cells, without creating runs.
- Launcher tests execute the real shell expansion with local SSH/Slurm stand-ins,
  verify custom smoke paths, reject invalid submission replies, and exercise
  successful/failed runtime gates before campaign-lock creation.
- The four serving input/lock files are byte-identical to PR #62. Core optimizer
  and Terminal-Bench agent implementation match the reviewed pre-merge branch.
- The broad suite has external prerequisites: 20 existing OpenRouter tests
  fail with HTTP 401 without credentials, and the optional DSPy full-program
  adapter module needs `dspy.teleprompt.bootstrap_trace`, absent from the
  deliberately pinned HotPotQA DSPy version. These files and their core code
  are unchanged by the merge; they are excluded from the final local suite.

## Outstanding human review

- Ask Lakshya to confirm the system-prompt/all-text scope ablation.
- Ask Lakshya to confirm the TB2.1 30/19/40 split.
- Gilad's deduplication review is resolved by removing our added context
  deduplication and evidence pruning for HotPotQA and Terminal-Bench.
- Review the integrated source and live Della runtime before scheduling HotPotQA.

No messages to collaborators or Della jobs are sent by this consolidation.
The source integration does not establish GPU runtime compatibility or successful
benchmark execution; those require actual subsequent runtime evidence.

September 13–14 follow-up: DeepSeek HotPotQA optimizer calls use 131,072 output
tokens. After training-pilot cutoffs, the user approved 32,768 for DeepSeek solver
calls and Qwen optimizer calls. A later Qwen solver cutoff led to approval of
32,768 for Qwen solver calls as well. Both revised
profiles require fresh qualification. The approved qualification
compares server active-request limits 1/2/4 on the same twelve training examples,
with client workers fixed. Four small optimizer checks precede full calibration.
Source/runtime changes require new run identities; see the current Della runbook.
