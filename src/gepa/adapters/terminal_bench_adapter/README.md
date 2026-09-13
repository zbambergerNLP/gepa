### Terminal-Bench 2.1

Terminal-Bench 2.1 is the sole Terminal-Bench target. The optimization and training
pilot commands default to `--experiment tb2.1`. Each model arm compares two
editable scopes with identical initial model input, skill files, task splits,
and task/proposer models. Methods within a scope receive the same editable
components.

The pinned dataset has **89 tasks**, split into **30 training, 19 validation,
and 40 test tasks**. `--optimization-scope system_prompt` (the default)
exposes the whole initial instruction block as one editable prompt.
`--optimization-scope all_text` exposes all 14 prompts and two skill files.
Pass that flag explicitly when resuming an existing all-text run. The recorded
scope still prevents an old directory from silently switching experiments.

#### GEPA adapter provenance and TB2.1 compatibility

The [GEPA blog](https://gepa-ai.github.io/gepa/blog/) links to the official
[`TerminusAdapter` API reference](https://gepa-ai.github.io/gepa/api/adapters/TerminalBenchAdapter/).
Its [upstream source](https://github.com/gepa-ai/gepa/blob/4f1613773d0c13c8f1551543a801b299bd8acf73/src/gepa/adapters/terminal_bench_adapter/terminal_bench_adapter.py)
was verified on September 11, 2026: commit
`4f1613773d0c13c8f1551543a801b299bd8acf73`, file blob
`1786e06b8e4bdee4129d85521bca4a26fcab7c7e`.

This fork maintains a **Harbor port of that adapter**, in the same GEPA module.
It is not the unmodified upstream implementation or constructor API. The
upstream runner targets legacy `tb run`, `terminal-bench-core@head`, and one
`instruction_prompt`; replacing our port with that implementation would not
support the approved TB2.1 dataset and both editable scopes.

| Adapter responsibility | Published upstream | This TB2.1 port |
| --- | --- | --- |
| Agent execution | Legacy `tb run` and Terminus wrapper | Pinned Harbor CLI and `PromptedTerminus` |
| Editable text | One instruction prompt | One unified initial prompt, or all 14 prompts and two skills |
| Scores and feedback | Passed parser checks, episode messages, success/failure text | Official verifier reward, ATIF traces, verifier diagnostics |
| Execution errors | Result-reading errors become zero scores | Infrastructure or missing-evidence errors stop the run |

Optimization, training pilots, and final evaluation all instantiate
`gepa.adapters.terminal_bench_adapter.TerminusAdapter`; `TerminalBenchAdapter`
remains an alias for existing callers. Final evaluation uses the adapter's
evaluation path without constructing reflection feedback. There is no fallback
to the legacy runner. Run contract version 32 and pilot configuration version 9
record the adapter entry point, the explicit `harbor_port` implementation, and
upstream provenance. Missing or changed adapter identity prevents optimization
resume and final comparison; use fresh run directories for older contracts.

#### Methods and campaign matrix

Each scope in each model arm follows HotPotQA's six-configuration comparison:

| Condition | Method | Standard budget | Double budget |
| --- | --- | --- | --- |
| `vanilla` | Vanilla GEPA reflection | 4 epochs | 8 epochs |
| `react_v2` | Full FOREST: Controller, Manifestor, ReAct V2 | 4 epochs | 8 epochs |
| `react_v2_random` | FOREST with a uniformly random Controller | 4 epochs | — |
| `action` | Action-conditioned stateless GEPA | 4 epochs | — |

This gives **24 optimization runs**: six configurations × two scopes × two
models. There is one optimization run per scope/method/budget configuration.
Each model's common initial harness is an additional evaluation reference.

Full FOREST retains the same Verbalized Sampling-based Controller as HotPotQA.
The model scores every section/action pair; the sampler combines 90% of the
normalized model distribution with 10% uniform exploration among pairs assigned
positive probability. Pairs assigned zero stay excluded. This policy applies to
TB2.1 at both full-FOREST budgets.

This is our adaptation of [Verbalized Sampling](https://arxiv.org/html/2510.01171v3#S4):
we elicit probabilities over a fixed action catalog, then add uniform exploration.
The 10% mixture is our setting. The authors' [`tau=0.10` example](https://github.com/CHATS-lab/verbalized-sampling#quickstart)
instead requests responses from the low-probability tail; it does not specify a
90/10 exploration mixture. We retain that distinction in describing the method.

Random-Controller FOREST preserves Manifestor steering, the ReAct V2 editor,
and branch-local edit history. Only Controller selection changes. The `action`
condition uses HotPotQA's `VerbalizedActionSelector` and `StatelessReflectionLM`:
select a semantic action and section, then rewrite that section once without
Manifestor or a tool loop. Every selected prompt and skill gets its own
single-component action job with its matching section template. All edits see
the same parent harness and combine into one child candidate. `action` is no
longer an alias for full FOREST. Controller randomness is seeded, separate from
task sampling, and restored on resume.

Both FOREST variants use the same completion policy as HotPotQA: the ReAct
editor may make multiple edits within the selected section under one fixed
semantic action and Manifestor instruction, then emit `<finish>`. Assistant
turns and valid tool calls have no count limit. The editor receives the latest
section text after each edit; an atomic delete/insert pair must be complete
before another operation or submission. Run contracts record this policy and
reject earlier checkpoints. This changes the optimizer editor's stopping rule;
the approved per-call token budgets and benchmark task timeouts are unchanged.

Invalid editor calls return the specific protocol or edit error to the next
model turn without changing the selected section. The model can correct its
arguments using the latest section text already in the conversation. Recovery
attempts have no separate retry limit; the unlimited assistant-turn policy
includes failed calls. These attempts consume model calls and context, but do
not count as valid edits. Provider or infrastructure failures are separate from
this tool-error recovery loop.

All optimizer character limits default to unlimited and are independently
configurable through the shared [text-limit settings](../../../../examples/common/text_limits.md).
Pass `--text-limits` as a JSON object to configure complete prompts and skills,
total candidate size, complete optimizer requests, selector targets, feedback,
Manifestor traces and steering, saved diagnostic text, or verifier logs. Every
method receives the same settings. Model context and per-call output limits
still apply. Run contracts record all limits, reject changed settings on resume,
and require matching settings across the final comparison.

HotPotQA's budget levels are 6,871 and 13,742 metric calls. Terminal-Bench retains
the approved epoch-based rule: four and eight training epochs. Thus the method
matrix and 2× budget multiplier match HotPotQA; the budget unit differs.

#### Editable scopes, agent text, and skills

| `--optimization-scope` | Editable candidate | Fixed auxiliary text |
| --- | --- | --- |
| `system_prompt` (default; runs first) | One `instruction_prompt` containing the whole initial instruction block | Ten later prompts and both skills, including skill metadata |
| `all_text` | 14 prompts and two skills, edited as 16 separate components | None |

The smaller scope combines `instruction_prompt`, `terminal_tool`,
`skill_discovery`, and `command_format` into one editable component. It is not
limited to the original main instructions and does not propose four separate
edits. The `system_prompt` label selects this initial instruction block; Harbor's
message roles remain fixed. When materializing this candidate, the other three initial components
are empty because their guidance already resides in the unified prompt.
Later prompts and skill files are restored from the selected provider's seed;
extra candidate keys are rejected before Harbor executes anything.

Both scopes render the initial guidance documents with nested `###` headings.
This prevents repeated provider `##` section names from creating invalid
sections in the unified prompt. Component bodies and literal braces are
preserved. Initial model input and skill files are identical between scopes,
even though their candidate dictionaries differ. This rendering policy and
the edit boundary are recorded in run contracts; older runs need fresh
directories. Final testing shares one common initial reference.

`PromptedTerminus` runs the same 16-component document bundle for TB2.1:

| Surface | Components |
| --- | --- |
| Initial and tool instructions | `instruction_prompt`, `terminal_tool`, `command_format`, `skill_discovery` |
| Context management | `summary`, `summary_questions`, `summary_answers`, `handoff`, `short_summary`, `context_recovery` |
| Completion and recovery | `completion`, `timeout`, `parse_error`, `output_limit` |
| Reusable skills | `skill_debugging`, `skill_verification` |

The runtime component set stays fixed at 14 prompts and two skill files for
all methods, scopes, and budgets. Optimizers cannot add or remove editable
components or change stable file identities. In `all_text`, all component text
is editable, including each skill's name, description, instructions, and examples.

Prompts use the selected provider's `user_prompt` template. Skills use the `skill`
template: Name, Description, Instructions, and Examples. The approved loading
policy for TB2.1 is on demand: the task agent initially sees each skill's
name, description, and file path, then reads its full `SKILL.md` through the
terminal when relevant. This policy applies to all methods, scopes, and budgets.
In `all_text`, every method can rewrite both skills' metadata and bodies,
regardless of whether a particular task uses them. In `system_prompt`, skills
remain available to the task agent but their metadata and bodies stay fixed.

All methods explicitly use `module_selector="all"`: in `all_text`, each proposal
selects all 16 documents, revises them separately using the same minibatch
evidence, and evaluates the combined harness as one child candidate. This applies the
[GEPA FAQ's multi-module efficiency guidance](https://gepa-ai.github.io/gepa/guides/faq/#how-do-i-optimize-multi-module-dspy-programs-efficiently)
to TB2.1. Optimizer-side editing work is measured separately; selecting
all documents does not require a separate task evaluation for each document.
In `system_prompt`, the same selection policy selects the single unified prompt.
The selected epoch budget stays fixed across scopes, and perfectly scored
minibatches skip reflection and editing. Run contracts pin editable and runtime
component sets, fixed text, document bundle version, and selection policy for
resume and final-test comparisons.

The optimization target is **model-facing text and skills**. Python agent logic,
tool implementations, the actual JSON parser/command interface, task inputs,
runtime observations, and the official verifier stay fixed. Rewriting the text
that describes a tool does not change its implementation. Task and terminal-state
fields are appended separately, and candidate braces remain literal. TB2.1
retains their official task resource limits and agent timeouts, without
local overrides. All configurations in both scopes use the same task
limits, including the standard and double optimization budgets. The double
budget increases optimization opportunities while keeping per-task limits fixed.
An optional `--harbor-process-timeout-sec` is a whole-job operational limit
recorded in the run contract.

#### Task-agent completion

TB2.1 retains Harbor 0.22.0's completion confirmation mechanism across all
methods and both budgets. The first `task_complete: true` response returns
the editable `completion` prompt with the current terminal state. The agent
can confirm completion or continue working; continuing without the completion
flag resets the pending confirmation. Confirming ends the task-agent loop,
after which Harbor runs the official verifier to determine the score.

The confirmation prompt remains an optimization target, while the parser and
confirmation mechanism stay fixed. Confirmation calls count toward recorded
model usage and the task's existing timeout. The FOREST optimizer editor uses
its separate explicit `<finish>` operation described above.

#### Task-agent context summarization

Automatic summarization stays enabled for TB2.1 across all methods and
both budgets. The campaign explicitly supplies `enable_summarize=true` and
retains Harbor 0.22.0's existing proactive trigger: fewer than 8,000 estimated
tokens remaining in the context window. This is a token-space trigger, separate
from the optional character limits on optimizer inputs and candidate text.

The task model summarizes its working conversation, asks questions about missing
details, answers them using the prior history, and continues from a handoff.
The summary, question, answer, handoff, and recovery instructions remain editable
components for every optimizer. The trigger and execution flow stay fixed;
custom agent kwargs cannot disable summarization or change its threshold.
This preserves the pinned [Harbor flow](https://github.com/harbor-framework/harbor/blob/v0.22.0/src/harbor/agents/terminus_2/terminus_2.py).

Summarization can omit details and adds model calls. Original execution
trajectories and the summarization traces remain saved for reflection, and the
existing usage observer records the extra calls. Training pilots record the
context settings; optimization contracts reject missing or changed settings on
resume and before final comparison. All execution paths use the same settings.

#### Textual feedback for reflection

Every method receives the same training-evidence format: task identity, complete
ATIF trajectories, raw trial/process metadata, official rewards, and textual
verifier diagnostics. Task instructions, reasoning, commands, observations,
subagent relationships, copied context, and ATIF metadata remain as recorded.
The official verifier reward is the optimization score; diagnostic text supplies
evidence for reflection without changing that score.

`Feedback` includes the actual contents of Harbor's `verifier/test-stdout.txt`
and `verifier/test-stderr.txt` when present, plus corresponding logs under
`steps/*/verifier/` for multi-step trials. These are console outputs from the
verifier run, not the verifier implementation or benchmark solution files.
Every editable component receives the same feedback: 16 components in
`all_text`, or one unified prompt in `system_prompt`. Each optimizer retains
its existing reflection procedure.

TB2.1 and HotPotQA default to unlimited Manifestor
traces within the configured model's context window; `manifestor_trace_chars`
can set an explicit allowance. Repeated strings, paragraphs, log lines, feedback,
and document bodies remain intact. FOREST's Manifestor and editor receive feedback
in their dedicated field and in the per-example traces. Context overflow stops
the run through the provider error path instead of silently truncating evidence.

The adapter passes complete saved ATIF trajectories into reflection, including
copied-context steps and nested subagent history. It does not replace copies
with references, collapse repeated lines, or prune trajectory metadata. Each
example retains its editable document body. Reflection-context policy version 2
and feedback version 5 are pinned for resume and final comparison alongside the
Manifestor limit.

This removes our additional preprocessing, while preserving the configured
Harbor task agent. Harbor 0.22.0's native
[terminal reader](https://github.com/harbor-framework/harbor/blob/v0.22.0/src/harbor/agents/terminus_2/tmux_session.py)
returns incremental output when available, falling back to the current screen.
Its [summarization flow](https://github.com/harbor-framework/harbor/blob/v0.22.0/src/harbor/agents/terminus_2/terminus_2.py)
remains enabled; copied history retains messages while omitting duplicate usage
counters. The reference
[GEPA adapter](https://github.com/gepa-ai/gepa/blob/4f1613773d0c13c8f1551543a801b299bd8acf73/src/gepa/adapters/terminal_bench_adapter/terminal_bench_adapter.py)
passes recorded message history into reflection without our former text
deduplication pass.

Each log contributes its full text by default. Configure `verifier_log_chars`
to retain source characters from the beginning and end with an omission marker.
The limit counts Unicode characters, including for multibyte logs. Full logs
stay unchanged in the Harbor artifacts. Text uses UTF-8 with replacement
for invalid bytes. Missing logs are reported as unavailable and do not change a
valid reward; a present but unreadable log raises an evidence error.

Only training tasks may enter reflection. Validation still selects candidates,
and held-out test feedback cannot enter optimization. The run contract pins the
feedback version, log locations, optional character limit, decoding, and
missing-log policy. Earlier runs without this feedback contract require fresh
directories.

#### Task failures, timeouts, and recovery

All six configurations use the same policy. A completed
task keeps its official verifier reward, including zero for unsuccessful work.
An `AgentTimeoutError` also keeps the official reward when valid verification
and a trajectory exist; the timeout remains visible in reflection feedback.
Reaching the agent time limit can still score one if the final work passes.
Harbor's job error counter must match exactly these verified trial timeouts.
For multi-step tasks, a timed-out step must have its own verifier rewards;
an aggregate reward cannot hide an unverified step. ATIF trajectories are read
from both `agent/` and Harbor's archived `steps/*/agent/` directories.

Provider, container, verifier, subprocess, and missing/invalid-evidence failures
stop optimization or final testing without a fabricated score. This follows
HotPotQA's distinction between task outcomes and systemic failures; HotPotQA
also scores its specific malformed task-output case as zero. Terminal-Bench's
timeout handling uses Harbor's verifier instead of assigning an automatic zero.

Every Harbor job explicitly uses `n_attempts=1` and `retry.max_retries=0`.
There are no automatic retries of failed Harbor jobs or extra attempts to improve a
completed score. Repair infrastructure before explicitly resuming. Completed
test repetitions remain reusable; an interrupted, unrecorded repetition starts
again as described below. Failed-job logs and any Harbor-recorded token/cost
counters remain in their original evaluation directories, separate from scored
GEPA evaluations. Those recovery costs must be reported separately rather than
inferred from `total_metric_calls`; unavailable usage is not zero cost.
Run contracts pin this policy and reject earlier or changed policies on resume
and when freezing final comparisons.

Model requests use the shared [provider retry policy](../../../../examples/common/provider_retries.md):
at most three attempts for temporary connection/server failures, with one- and
two-second backoff. An explicit request timeout covers all attempts together.
Permanent request errors stop immediately. Both SDK retry settings remain zero,
and Harbor's nested retry decorators are bypassed so they cannot multiply the
three-attempt allowance. Exhausted summary and context-recovery requests stop
the run. Completed task scores and FOREST tool-error correction keep their
separate policies above.

Every physical attempt is recorded in `provider-attempts.jsonl` and in the
existing `token-usage.jsonl` files, including failures with unknown usage.
The two files describe the same requests, so their totals must not be added.
The policy is pinned in run contract version 32 and pilot configuration version
9; older or changed policies cannot resume or enter final evaluation.

For cluster execution, the approved policy matches HotPotQA: after
scheduler-confirmed allocation time expiry, automatically request a continuation
only when a verified recoverable checkpoint contains newly persisted work from
that allocation. Completed evaluations or iterations count as progress even
with zero, unchanged, or lower rewards or rejected candidates. Resume the same
logical run with matching source, campaign, data, model, runtime settings, saved
state, and remaining original budget. Preserve completed test evidence and
report recovery costs separately. Subsequent ablations wait for the current
optimization and test to finish successfully.

Unresolved execution errors, cancellation, missing/incompatible checkpoints,
or no saved progress stop continuation. Official task timeouts and provider or
Harbor process failures keep the policies above; they are not cluster allocation
expiry. Automatic allocation continuation is approved, but scheduler wiring and
live verification remain pending. This decision does not resume the paused
Terminal-Bench Della backend work.

#### Reference protocol and pending confirmation

The working decision is to compare unified-initial-prompt optimization with
full text-and-skill optimization on TB2.1. The
[AutoSaddler GEPA baseline](https://arxiv.org/html/2608.23041v1#A2) optimized one
unified prompt on Terminal-Bench 2.0, while AutoSaddler itself could also change
executable harness code. Our GEPA and FOREST methods have identical editable
components within each scope, with execution code fixed throughout. The smaller
scope matches the single editable prompt unit; it does not restore the paper's
runtime, exact seed, or model settings.

TB2.1 retains the approved 30/19/40 split sizes and task-name assignments,
four-epoch standard budget, and three repeated final evaluations. The dataset
contains revised task contents from the official TB2.1 release. The assignments
are our deterministic split; the authors' exact assignments and harness revision
were not established from released artifacts. Dataset version, editable surface,
seed documents, and model arms differ from the paper. This is our controlled
comparison, not a reproduction of its GEPA setup or reported scores.

- [ ] Ask Lakshya to confirm the TB2.1 scope ablation: one unified initial
  instruction block versus all model-facing text and skills, with identical
  editable components for GEPA and FOREST within each scope and fixed execution
  code. This is a research follow-up; the implementation uses the user's
  approved working decision.
- [ ] Ask Lakshya to confirm the TB2.1 train/validation/test split (30/19/40),
  including the preserved task-name assignments.
- [x] Gilad resolved the context-deduplication review by choosing to remove our
  added deduplication and evidence pruning in HotPotQA and TB2.1. Preserve
  repeated text, feedback, document bodies, and complete Harbor traces.
  Configurable character limits and native Harbor context management retain
  their previously approved settings.

#### Dataset pins and run identity

The experiment uses Harbor **0.22.0** in a separate Python 3.12 environment and
the official [Terminal-Bench 2.1](https://github.com/harbor-framework/terminal-bench-2-1)
Hub dataset `terminal-bench/terminal-bench-2-1`.

The checked-in `examples/terminalbench/terminalbench-v2.1-manifest.json` pins
all 89 task content hashes and the dataset content hash
`sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a`
(registry version ID `f92eea12-ff70-4d30-ace0-003abf294998`). Every Harbor job uses
that immutable dataset ref, never a moving `latest` label. Git source metadata
is informational; the official Hub content hashes identify the executed tasks.

All 89 task names match the previous approved task set. The manifest preserves
every training, validation, and held-out assignment while using the revised
TB2.1 contents. Hash ordering uses the original task name without Hub's
`terminal-bench/` namespace, with the existing split seed. Assignments stay
identical across models and optimizers.

Manifest validation checks exact source metadata, task-reference digest, split
sizes, deterministic ordering, and disjoint coverage. Each evaluation retains
its candidate, experiment identity, Harbor job configuration, verifier results,
and ATIF trajectories. Reflection identifies the pinned dataset.

The resume contract records the experiment, dataset, complete task refs and
splits, target, editable scope, materialized and common reference seed digests,
models, decoding, and budget. Only `tb2.1` is
accepted. Earlier benchmark contracts and checkpoints require fresh run
directories; they cannot silently resume or enter final comparisons as TB2.1.
Each completed full-split ablation automatically freezes its validation winner
and evaluates the held-out test split. All ablations must use the checked-in
manifest's exact data, immutable task refs, and ordered 30/19/40 split; a custom
manifest cannot change assignments while retaining the same counts.

#### Optimization budget

The standard budget is **four training epochs**, retaining the approved budget
inspired by the TB2 GEPA run in
[AutoSaddler, Appendix B](https://arxiv.org/html/2608.23041v1#A2).
`--budget double` gives vanilla GEPA and full FOREST **eight epochs**, with all
other settings fixed. With the default minibatch size of three, the stopping
rule is `epochs * ceil(train_tasks / 3)` iterations:

| Training tasks | Iterations per epoch | Standard / double iterations | Standard / double training draws |
| --- | --- | --- | --- |
| 30 | 10 | 40 / 80 | 120 / 240 |

No padding is needed for the full 30-task training split at minibatch size three.
A training limit or different minibatch size changes the iteration count using
the same rule; GEPA pads an incomplete minibatch if needed. These counts describe
sampled training tasks, not total task executions or model calls.

Training shuffles use their own random stream initialized with `--seed`, separate
from parent selection and reflection randomness. The same ordered training split,
minibatch size, and seed produce identical task order across methods, models, and
text scopes. Eight-epoch runs share the first four epochs with standard runs,
then continue the shuffle sequence. Checkpoints save the permutation, cursor,
and private RNG state so a resumed run retains every later epoch's task order.
Run contract version 32 records this policy; older checkpoints require fresh
runs. HotPotQA uses the same sampler, with its existing metric-call budgets.

Each iteration samples one minibatch for one mutation attempt; merging is off.
Perfect minibatches or unsuccessful proposals still consume their iteration.
Parent and proposed-candidate evaluations, plus initial and conditional full
validation, contribute to the measured `total_metric_calls`. Validation is
allowed to finish and does not reduce the number of training epochs. Thus the
methods have equal training-pass budgets, not necessarily equal task-execution,
token, or wall-time costs. The run contract records the epoch rule, iteration
limit, sampler, and padding. Resuming continues the original budget rather than
granting additional epochs. Doubling the budget doubles proposal opportunities;
it does not guarantee twice as many accepted candidates or total metric calls.
The two larger-budget runs in each scope start independently from the shared initial harness.
They cannot extend a standard-budget checkpoint in place.

`--max-metric-calls` is an optional additional early-stop cap for pilot or
operational runs. It is checked at iteration boundaries and can be exceeded by
the final iteration's evaluations. A run stopped by that cap before its selected
epoch budget does not complete the protocol. The normal commands omit this cap.
An incomplete full-split run cannot enter held-out evaluation and stops the
campaign until it is resumed to completion. Partial-split diagnostic runs skip
held-out testing and cannot be launched as campaign ablations.

#### Evaluation caching

Evaluation-result caching is explicitly disabled for TB2.1 and HotPotQA across
all methods and both budgets. A new request to evaluate the same harness on
the same task executes it again. TB2.1 creates a fresh Harbor job and task
environment; HotPotQA also disables DSPy's disk and memory response caches.
The approved budgets remain unchanged, and actual evaluation/model usage is
counted for these fresh executions.

Completed checkpoint records and optimizer response journals remain available
for recovery of the same logical work. They do not supply results for unrelated
new evaluations, and completed held-out repetitions remain resumable. Run
contract version 32 records `cache_evaluation=false`, forwards it to GEPA, and
rejects missing or changed policies on resume and before final comparison.

#### Parent selection

All six configurations in each scope use GEPA's existing Pareto parent selector with one
frontier key per validation task, matching HotPotQA. Track which previously
evaluated harnesses tie for the best score on each task, prune redundant
best-task coverage, then sample one remaining harness with probability
proportional to the number of tasks on which it is best. This can retain a
lower-average harness that handles tasks the higher-average harness misses.

Each iteration selects one complete parent harness. FOREST's Controller then
selects sections and actions within that parent; the random-Controller
ablation retains the same Pareto parent-selection algorithm. The existing
experiment-seeded optimizer RNG and checkpoint restoration govern sampling.
The final winner remains the harness with the highest mean validation score.

`candidate_selection_strategy="pareto"` and `frontier_type="instance"` are
explicit run contract fields forwarded to the optimizer for every method and
budget. Run contract version 32 rejects missing or changed parent-selection
policies on resume and before final comparison.

#### Proposal acceptance and validation

All six campaign configurations in both scopes use the same strict-improvement rule as
HotPotQA. Evaluate the parent and proposed harness on the same training
minibatch (three tasks by default). Advance the proposal only if its summed
reward is strictly higher; reject ties and regressions. A parent with a perfect
minibatch score (3/3 by default) skips reflection and editing. This matches
HotPotQA. Every iteration still consumes its approved training-pass budget.

Evaluate the initial harness on all 19 validation tasks. Every proposal that
passes the training comparison also receives full validation; rejected edits
receive no validation run. Validation scores govern the candidate frontier and
final winner selection. A training improvement does not guarantee a validation
improvement or automatically replace the existing best harness.

`acceptance_criterion="strict_improvement"`, `validation_evaluation="full_eval"`,
`skip_perfect_score=true`, and `perfect_score=1.0` are explicit run contract
fields forwarded to the optimizer. Run contract version 32 rejects missing or
changed policies on resume and before final comparison. This preserves the
prior runtime defaults while recording the approved experiment identity.

#### Repetitions and final testing

Each model arm uses one optimization run per scope/method/budget configuration, followed by
three test repetitions of each frozen harness. This follows
[AutoSaddler, section 5.1 and Table 3](https://arxiv.org/html/2608.23041v1): one
evolution run and three test executions, reporting mean and standard deviation
of Pass@1.

Testing follows each completed ablation, matching the HotPotQA campaign. The
next ablation starts after that cell's testing succeeds. Each winner is selected
independently by mean validation reward, with GEPA's earliest-candidate tie break,
and frozen before its test tasks run. Later ablations join the same comparison
without replacing earlier winners or their evidence. All cells must match the
benchmark data, ordered splits, model, decoding, optimization seed, and shared
runtime settings. Each must have the correct scope, method, and budget. Partial
training/validation selections and runs stopped before four or eight epochs are
rejected. Test scores never determine prompts, settings, budgets, or selection
for subsequent ablations.

Each repetition starts a distinct Harbor job over the entire test split with
`n_attempts=1` and fresh task environments. Training and validation evaluations
remain single-attempt. All three test success rates contribute equally to the
reported mean; no best-of-three selection or Pass@3 aggregation is performed.
The output records sample standard deviation (`ddof=1`) explicitly. Test repeats
measure execution variability for the fixed harness, not optimization-seed
variability.

For each model, the thirteen harnesses are the initial harness and the twelve
validation-selected winners:

| Experiment | Test tasks | Repetitions per harness | Attempts per harness | Attempts across all thirteen harnesses |
| --- | --- | --- | --- | --- |
| TB2.1 | 40 | 3 | 120 | 1,560 |

With the shared test directory, that is 39 Harbor jobs per model, or 3,120 task
attempts across both models. The common initial harness is tested once per model
(three repetitions); each ablation still gets its own three fresh repetitions,
even if its validation winner equals the initial harness.

`examples.terminalbench.main` tests automatically after optimization. For
individually launched cells, pass the same `--test-output-dir` for all of that
model's ablations; it defaults to `RUN_DIR/heldout` for a standalone run. The
campaign launcher supplies `RUN_ROOT/test` automatically. To resume testing a
completed cell directly, without waiting for any other optimization run:

```bash
uv run python -m examples.terminalbench.evaluate \
  --runtime-record runs/servers/qwen.json \
  --run-dir system_prompt__vanilla=runs/tb2.1/qwen/system_prompt/vanilla \
  --output-dir runs/tb2.1/qwen/test
```

Repeat `--run-dir CELL=PATH` to include more completed cells, or supply just the
next one using the same output directory. Already-frozen cells are retained.
Use separate corresponding directories for the DeepSeek arm.
The command reads student model, endpoint, decoding, and concurrency from the
optimization contracts. `--harbor-executable` and `--docker-executable` optionally
select installed binaries. Checkpoints must be trusted local optimization
artifacts because GEPA's checkpoint format uses Python pickle.

`frozen-comparison.json` version 4 accumulates the common baseline and completed
cells' immutable harnesses and source contracts, including each winner's scope.
Each completed repetition gets a JSON file with per-task verifier rewards and
its distinct Harbor job identity. Rerunning the same command reuses completed
repetitions and runs only missing ones; an interrupted, unrecorded repetition
starts again in fresh environments. Frozen harness or configuration changes
are rejected. The CLI locks the output directory against concurrent writers.
`summary.json` is written after all currently frozen harnesses finish testing and contains
the scope, three Pass@1 values, their mean and sample standard deviation, and the
completed task-attempt count for each harness. Scores are fractions in JSON
and percentages in console output. `completed_cells`, `pending_cells`, and
`campaign_complete` distinguish partial campaign coverage from a finished
twelve-cell comparison. Adding a new cell invalidates the earlier summary until
the added tests finish. Failed or incomplete Harbor jobs stop the
evaluation instead of becoming fabricated zero scores.

This aligns the repetition protocol with the paper; the previously documented
TB2.1 dataset, task-assignment, harness, and model differences still apply.

#### Record the local serving runtime

Launch the model through `examples.terminalbench.runtime` on the Linux GPU node,
using the prepared **serving environment's Python**. It verifies every checkpoint
file with the same model-snapshot verifier as HotPotQA, captures the installed
vLLM/PyTorch/CUDA/Transformers and Python versions, an installed-package fingerprint,
visible GPU models/counts/memory/compute capabilities and driver, precision,
parallelism, the exact forwarded launch arguments, and relevant vLLM/NCCL settings.
Credentials are excluded. It then executes vLLM with that same interpreter and
arguments, preserving the process identity recorded in the JSON file.

For the existing Qwen TP1/DP8 profile, set `VLLM_PY` to the prepared serving
interpreter and `SOLVER_MODEL_PATH` to its verified checkpoint, then run from this
repository root:

```bash
uv run --no-project --python "$VLLM_PY" python -m examples.terminalbench.runtime \
  --model hosted_vllm/Qwen/Qwen3.8-27B \
  --model-path "$SOLVER_MODEL_PATH" \
  --runtime-record runs/servers/qwen.json --port 8000 -- \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --tensor-parallel-size 1 --data-parallel-size 8 --api-server-count 8 \
  --gpu-memory-utilization 0.92 --max-model-len 262144 \
  --max-num-seqs 1 --max-num-batched-tokens 16384 \
  --dtype bfloat16 --kv-cache-dtype auto --seed 0 \
  --no-enable-prefix-caching --language-model-only
```

For DeepSeek V4.1, use its separate prepared serving interpreter from
`.serving-venv-deepseek-v4.1-flash` and its verified checkpoint. The pinned build
is `0.1.1.dev5+ge77daef89`; older V4 releases do not qualify. Use its TP8/EP8 flags:

```bash
FLASHINFER_NO_DOWNLOAD=1 VLLM_ENGINE_READY_TIMEOUT_S=3600 \
uv run --no-project --python "$VLLM_PY" python -m examples.terminalbench.runtime \
  --model hosted_vllm/deepseek-ai/DeepSeek-V4.1-Flash \
  --model-path "$SOLVER_MODEL_PATH" \
  --runtime-record runs/servers/deepseek.json --port 8000 -- \
  --language-model-only --tokenizer-mode deepseek_v41 \
  --reasoning-parser deepseek_v41 --enable-auto-tool-choice --tool-call-parser deepseek_v41 \
  --tensor-parallel-size 8 --enable-expert-parallel --data-parallel-size 1 --api-server-count 1 \
  --gpu-memory-utilization 0.92 --max-model-len 262144 \
  --max-num-seqs 1 --max-num-batched-tokens 16384 \
  --dtype bfloat16 --kv-cache-dtype fp8 --seed 0 --no-enable-prefix-caching
```

Keep the existing [Della serving environment](../../../../scripts/della/README.md)
and its environment settings. The launcher owns the checkpoint, served model
name, loopback host, and port; pass the remaining serving flags after `--`.
Numerical dtype, KV-cache dtype, tensor parallelism, and data parallelism must
be explicit. The examples do not pick concurrency or replace training calibration.
Wait for the model server to become ready before starting benchmark commands.

Pass `--runtime-record` on **every** pilot, optimization, resume, and final-test
invocation. If optimizer roles use a separate local server, launch it the same
way and supply `--proposer-runtime-record`; otherwise the task-server record is
used for both roles. Each role's endpoint must match its recorded local port.
Final evaluation only needs the task server because no optimizer runs there.

The entry points verify that the record belongs to a live process on the current
node using hostname, Linux boot ID, PID, and process start time, and that this
process or its API workers owns the listening socket. Recreate the
record after a server restart; a previous pilot's saved identity cannot substitute
for this live check. This is trusted local launcher evidence, not remote server
attestation, and does not support tunneled or remote model endpoints.

Only the material configuration enters pilot/run/frozen-comparison contracts.
Node names, job IDs, GPU device numbers, PIDs, and checkpoint/record paths do not
enter that comparison, so fresh servers on equivalent nodes remain compatible.
Endpoint URLs remain subject to the existing run contract. Material settings are
compared exactly, including launch-argument order. Missing or changed runtime
evidence stops execution before Harbor. A changed task runtime requires new
matching pilot evidence and a fresh campaign; changes to either role also reject
resume and mixed final comparisons. All twelve cells in a model arm must share
both role configurations. Historical contracts lacking this evidence are rejected
by run schema 32 and pilot schema 9.

#### Run

From the repository root, after completing and reviewing both pilot stages below,
with Docker and the same model endpoint available:

```bash
uv sync --extra dev
uv tool install --python 3.12 harbor==0.22.0

uv run python -m examples.terminalbench.main \
  --experiment tb2.1 \
  --optimization-scope system_prompt \
  --condition vanilla \
  --student-api-base http://localhost:8000/v1 \
  --proposer-api-base http://localhost:8000/v1 \
  --reviewed-pilot runs/canaries/tb2.1/qwen/full \
  --runtime-record runs/servers/qwen.json \
  --run-dir runs/tb2.1/qwen/system_prompt/vanilla \
  --test-output-dir runs/tb2.1/qwen/test \
  --harbor-work-dir runs/tb2.1/qwen/system_prompt/vanilla/harbor
```

Use `--condition react_v2`, `--condition react_v2_random`, and `--condition action`
with separate output directories for the other standard-budget methods.
The command above defaults to `--budget standard` (four epochs).
Launch each larger-budget run in its own fresh directory, for example:

```bash
uv run python -m examples.terminalbench.main \
  --experiment tb2.1 \
  --optimization-scope system_prompt \
  --condition vanilla --budget double \
  --student-api-base http://localhost:8000/v1 \
  --proposer-api-base http://localhost:8000/v1 \
  --reviewed-pilot runs/canaries/tb2.1/qwen/full \
  --runtime-record runs/servers/qwen.json \
  --run-dir runs/tb2.1/qwen/system_prompt/vanilla_2x \
  --test-output-dir runs/tb2.1/qwen/test \
  --harbor-work-dir runs/tb2.1/qwen/system_prompt/vanilla_2x/harbor
```

Use `--condition react_v2 --budget double` and
`runs/tb2.1/qwen/system_prompt/react_v2_2x` for the larger-budget FOREST run. Repeat
all six configurations with `--optimization-scope all_text` and paths
under `runs/tb2.1/qwen/all_text/`. Use the same twelve configurations for
the DeepSeek model arm, with separate paths under `runs/tb2.1/deepseek/`.
The CLI rejects double-budget ablations outside the approved pair.
An optional `--manifest` must match the pinned TB2.1 experiment.

Run the complete twelve-cell matrix for one model with the batch launcher:

```bash
uv run --no-sync python -m examples.terminalbench.run_ablations \
  --run-root runs/tb2.1/qwen \
  --reviewed-pilot runs/canaries/tb2.1/qwen/full \
  --runtime-record runs/servers/qwen.json \
  --student-api-base http://localhost:8000/v1 \
  --proposer-api-base http://localhost:8000/v1 \
  --dry-run
```

`--dry-run` prints commands without creating runs or invoking Harbor. Omit it
to execute sequentially: all six `system_prompt` configurations first, then all
six `all_text` configurations. Each cell completes optimization and held-out
testing before the next starts. The launcher uses the same ordered campaign
matrix as final evaluation and stops if either phase fails. Each scope and cell
has its own optimization directory; all cells share `RUN_ROOT/test` for matched,
incremental testing. Rerunning resumes the existing optimization and test
checkpoints without granting extra epochs or repeating completed test evidence.

Other optimization options, including homogeneous student/proposer models,
endpoints, runtime records, reviewed pilot, concurrency, seed, and text limits,
are forwarded to every cell.
Scope, condition, budget, per-run directories, and the shared test directory
are owned by the matrix. Partial train/validation limits are rejected.
Use a separate `--run-root runs/tb2.1/deepseek` with both DeepSeek model flags
and its endpoints for that model's campaign. Every model's campaign starts
with system-prompt optimization.

The campaign supports two separate model arms: Qwen3.8-27B with Qwen3.8-27B
(the model default), and DeepSeek V4.1 Flash with DeepSeek V4.1 Flash. Student,
proposer, and Controller use the same model within an arm. Both are served through
local vLLM. DeepSeek uses revision `dba1be0a40aa45a94ad051997016db3960a90277`,
numeric effort 100, and native `deepseek_v41` tokenizer/parsers on the exact
vLLM commit wheel `e77daef89`; see the
[serving configuration](../../../../scripts/della/README.md).

For the DeepSeek arm, point both roles at the prepared endpoint and use separate
output directories:

```bash
uv run python -m examples.terminalbench.main \
  --experiment tb2.1 \
  --condition vanilla \
  --student-model hosted_vllm/deepseek-ai/DeepSeek-V4.1-Flash \
  --proposer-model hosted_vllm/deepseek-ai/DeepSeek-V4.1-Flash \
  --student-api-base http://localhost:8000/v1 \
  --proposer-api-base http://localhost:8000/v1 \
  --reviewed-pilot runs/canaries/tb2.1/deepseek/full \
  --runtime-record runs/servers/deepseek.json \
  --run-dir runs/tb2.1/deepseek/vanilla \
  --test-output-dir runs/tb2.1/deepseek/test \
  --harbor-work-dir runs/tb2.1/deepseek/vanilla/harbor
```

Model identity, checkpoint revision, and thinking settings are
recorded in the resume contract; changing any of them requires a fresh run.

Temperatures follow the model author's applicable task/mode guidance, with the
general recommendation as the fallback. For both pinned thinking-mode models,
the current recommendation is 1.0 for task execution and every optimizer role,
including the Manifestor. The [provider source review](../../../../examples/common/temperature_policy.md)
records the HotPotQA, TB2.1, Controller, Manifestor, and proposer mappings.
The previous Manifestor-0.0 policy cannot resume or enter a final comparison
under the new contract. Both models use top-p 0.95 for every role, following
V4.1's instruct and agentic evaluation settings. Role decoding is recorded and
validated before resume or final comparison. These values apply at both
optimization budgets and during final task evaluation.

Every role explicitly enables thinking: Qwen requests `xhigh`, its provider
default, and DeepSeek V4.1 requests numeric effort `100`, used in its published
code-agent evaluations. Both pass the controls through
`extra_body.chat_template_kwargs`; Qwen uses `enable_thinking=true` and
DeepSeek uses `thinking=true`. Applying DeepSeek effort `100` to optimizer roles and
HotPotQA is our approved experimental choice, documented in the provider source
review. Contracts and final evaluation preserve these fields and reject
missing or changed reasoning settings. Every TB2.1 role uses a **32,768-token
output ceiling per call**, including reasoning and final output. HotPotQA keeps
16,384. This is the approved practical budget, not the providers' larger
maximum-performance recommendation. The consolidated Della profiles configure
262,144 context tokens for both models, below the provider's advertised maximum.
The output cap does not force a model to generate that many tokens.

Harbor 0.22 requires short model names for its local metadata registry. The
runtime registers the checkpoint basename there and retains the full original
`hosted_vllm/organization/model` identifier for requests, trajectories, and usage.
Both model arms use the same compatibility handling.

#### Concurrency review

Model-arm scheduling is a separate pilot decision from `--n-concurrent`.
Attempt overlapping Qwen and DeepSeek training pilots on separate allocations,
with independent model endpoints, runtime records, and output directories.
Retain each model's three-task smoke and full 30-task stages; coordinate the
full stages to overlap where capacity allows. Record job/evaluation IDs, actual
inference overlap, resource availability, throughput, timeouts, errors, and
output cutoffs. Concurrent submission alone does not qualify the schedule.

Decide after reviewing the pilot evidence. Use concurrent model-arm scheduling
only if both models pass the existing pilot checks while overlapping. If
overlap is unavailable or fails those checks, use sequential scheduling and
complete the required full training pilots in that mode. Record the selected
schedule and apply it consistently to the campaign; no schedule is selected in
advance. Within each model, ablations and their test evaluations remain
sequential. HotPotQA remains the first benchmark to run; this protocol does not
resume the paused Terminal-Bench Della backend work.

Choose task concurrency using training-only measurements before freezing each
benchmark/model comparison. Start the pilot with `--n-concurrent 1`, then test
higher values on the actual hardware with the same tasks, initial harness,
model settings, and serving configuration. Use a fresh output directory for
each pilot. Additional probes use `--stage calibration` and a `--train-limit`
at least as large as the concurrency being tested; these probes do not replace
the full stage at the selected settings. Record throughput, response latency, and task timeouts; compare Harbor
timing and exception artifacts alongside the model server's latency metrics.
The pilot records its concurrency in `canary-config.json`.

Once selected, pass the same `--n-concurrent` value to every scope, method, and budget
within that benchmark/model comparison. The run contract already records it,
resume rejects changes, and final evaluation requires matching source runs and
reuses their concurrency. Calibration does not use validation or test results.
The default remains one until the training measurements justify another value.

#### Two-stage training pilot and runtime review

Before freezing settings, run the initial harness in two stages for each model:

1. **Smoke:** exactly three training tasks to check setup (the default stage).
2. **Full:** all 30 training tasks, using a completed smoke check from the same
   model, endpoint, initial harness, and runtime settings. Concurrency may change
   after calibration; the full stage must use the final selected value.

For Qwen, with the default concurrency of one:

```bash
uv run --no-sync python -m examples.terminalbench.canary \
  --stage smoke \
  --model hosted_vllm/Qwen/Qwen3.8-27B \
  --api-base http://localhost:8000/v1 \
  --runtime-record runs/servers/qwen.json \
  --output-dir runs/canaries/tb2.1/qwen/smoke

uv run --no-sync python -m examples.terminalbench.canary \
  --stage full \
  --smoke-dir runs/canaries/tb2.1/qwen/smoke \
  --model hosted_vllm/Qwen/Qwen3.8-27B \
  --api-base http://localhost:8000/v1 \
  --runtime-record runs/servers/qwen.json \
  --n-concurrent 1 \
  --output-dir runs/canaries/tb2.1/qwen/full
```

Repeat both stages with `--model hosted_vllm/deepseek-ai/DeepSeek-V4.1-Flash`
with `--runtime-record runs/servers/deepseek.json` and separate directories under
`runs/canaries/tb2.1/deepseek`. This is 33 task
attempts per model: **60 full-stage attempts plus six smoke attempts** across
both arms, before additional calibration runs or the optimizer-flow checks
below. All tasks come from training; these two initial-harness stages do not
optimize prompts or use validation/held-out data. This
coverage is our experimental choice, not a requirement from the reference paper.

Review `pilot-summary.json` for elapsed time, tasks/hour, and verified task
timeouts; inspect `token-usage-summary.json` for actual usage, missing counts,
and cutoffs. `task-results.json` links the original trial/job evidence, including
Harbor timings. Successful pilot completion means complete measured coverage,
not that the model solved every task or never hit a cap. Keep official zero
rewards and verified timeouts visible. Provider/infrastructure errors or missing
results prevent completion, while available usage remains saved.

After reviewing token usage, cutoffs, timeouts, and throughput, pass the full
stage's directory as `--reviewed-pilot` to the optimization CLI or ablation
launcher. Supplying this flag explicitly records that review. The campaign
verifies the stage chain, complete task coverage, artifact hashes, and matching
runtime settings before contacting Harbor. The evidence is embedded in run
contracts, reused on resume without another review flag, and required again
at final comparison. Run contract version 32 and
pilot configuration version 9 reject older or changed policies; use fresh runs.
Partial-data diagnostic optimizations can still run without qualifying a final
comparison. A dry run only prints commands and does not attest review.

Both text scopes start with identical runtime text, so the same reviewed pilot
serves every method, budget, and scope within a model arm. The pilot defaults to
`system_prompt`; `all_text` is also supported. Keep the same hardware and model
server configuration between calibration, the full stage, and the campaign.
The command does not adjust caps or concurrency automatically. If settings need
to change, collect matching pilot evidence and start a fresh campaign.

In addition, include a small training-only optimizer-flow check for each model
and distinct method (`vanilla`, `react_v2`, `react_v2_random`, `action`) in both
text scopes, with `system_prompt` first. Exercise real task feedback through
reflection/proposal and candidate reevaluation, including each method's actual
Manifestor, Controller, and Editor stages where applicable. Preserve the stage
evidence and the resulting normal acceptance or rejection decision. Identical
initial text permits sharing task-calibration evidence above; it does not cover
the different optimizer paths and editing scopes.

Run one completed proposal-and-reevaluation cycle on the normal three-example
training minibatch per method/model/scope combination: four methods times two
models times two scopes gives 16 checks. Standard and doubled budgets share
this process coverage because they use the same execution paths. Stop after
the completed cycle regardless of reward direction or candidate acceptance.
The three-task and full 30-task initial-harness calibration stages remain
separate, and the 16 checks are additional to their task-attempt counts.

The optimizer-flow check passes when the process completes correctly. Zero,
tied, or lower rewards and rejected candidates are valid outcomes; there is no
minimum score or improvement requirement. Recovered editor tool errors are
acceptable when normal feedback allows completion. Investigate unresolved
execution errors or missing stage evidence. Do not retry for a better reward
or change the optimizer's acceptance rule. Keep validation/test tasks out and
preserve the existing task-failure and verified-timeout policies. Save these
diagnostics separately from throughput calibration and production budgets, and
start every production cell from the approved initial harness.

This additional optimizer-flow check is approved; implementation and live
execution remain pending. The current `canary` command and `--reviewed-pilot`
validation cover the two initial-harness stages, not this additional check.

Optimization writes `token-usage.jsonl` beside the run contract. Every Harbor
trial writes another in its agent log directory, including main-agent and
summarization calls. To aggregate optimizer and task usage, including failed jobs:

```bash
uv run python -m examples.terminalbench.token_usage \
  runs/tb2.1/vanilla runs/tb2.1/vanilla/harbor \
  --output runs/tb2.1/vanilla/token-usage-summary.json
```

Pass the actual Harbor work directory if it is outside the optimization run.
Overlapping paths are deduplicated. The same command works on final-evaluation
roots, but their outcomes must not inform cap selection. Reports group by model
and role, with input/output/reasoning totals, maximum observed output, cap hits,
and provider length finishes. A length finish is recorded separately from
reaching the configured output or context cap; it does not by itself identify
which limit caused the cutoff. Missing usage or finish reasons stay unknown
and get separate unreported counts. Inspect those counts and the file list;
an empty report is not evidence of zero usage or no cutoffs.

Optimizer observation covers plain, tool, and individual batch requests;
response-journal replay does not count as another physical call. A shared Qwen
Controller/editor client is reported as `controller-proposer`. Harbor records
provider responses before truncation recovery, plus provider-error types.
Provider errors and cooperative cancellations retain an attempt record with
unknown counts when none were returned. A process killed before logging can
still leave consumption unrecorded. This log contains no prompt or
response text, and raw execution evidence remains in the original artifacts.
Caps and observation policy are part of run identity, so older 16,384-token
Terminal-Bench runs cannot resume or enter the new final comparison unchanged.

Offline tests in `tests/harbor/` exercise the actual TB2.1 agent loop and Harbor job
schemas with simulated model and terminal boundaries. They make no paid model
calls and do not require Docker. The upstream prompt and adapted methods are
Apache-2.0; see `examples/terminalbench/HARBOR_LICENSE`.
