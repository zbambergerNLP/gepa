# Consolidated experiment decisions

Local branch: `codex/consolidated-della-experiments`.

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
| Serving settings | Preserve Qwen TP1/DP8/eight APIs/one sequence per replica and DeepSeek V4.1 TP8/EP8/DP1/one API/one sequence, context 262,144, bfloat16 activations, checkpoint FP8/FP4 weights, FP8 KV/automatic block size, native V4.1 parsers, and no speculative decoding. |
| Workers | Confirmed for the initial HotPotQA pilot: 12 Qwen / 4 DeepSeek. Check throughput and timeouts on training examples, then freeze the chosen concurrency across all six cells per model. Calibration remains pending. |
| Request timeout/retries | Adopt 3,600 seconds per HotPotQA logical request, shared by our maximum of three transient-error attempts. Keep nested SDK retries zero, 1/2-second backoff, and per-attempt logs. |
| Runtime verification | Keep our mandatory exact-runtime DeepSeek canary and Zach's independent transcript/tool diagnostic. Both use the current editor probes. Write campaign locks only after canary/native-tool success. |
| Setup | Adopt checked-in remote stages, detached downloads, scratch storage, SSH helpers, CUDA-header precedence, and rsync environment exclusions. Propagate custom Wiki-2017 paths and use this checkout's source in uninstalled remote environments. |
| Wiki-2017 | Adopt corrected extracted-corpus size, 1,780,742,620 bytes. Corpus/archive hashes, document count, retrieval parameters, and split stay fixed. |
| Text/editor behavior | Preserve configurable unlimited-by-default character limits, context deduplication, actionable tool-error feedback, repeated selected-section edits, and explicit finish. |
| Run identity | HotPotQA schema 25 records timeout and new serving provenance. Terminal-Bench schema 31, runtime freeze, and pilot schema 9 are preserved. Old runtime contracts cannot be silently resumed. |
| Della skill/runbook | Include and reconcile PR #62's skill and runbook with the current decisions. Remove obsolete instructions to launch the old commit; use the approved V4.1 arm. |

The user explicitly approved V4.1 during integration. Its published instruct
and agentic evaluations use temperature 1.0, top-p 0.95, and effort 100; these
replace the older V4 role-specific top-p values and named effort. The Della
context is 262,144, as in Zach's runtime; the provider advertises 1M. Keep the
approved smaller output caps. See the provider review for sources.

## Decisions preserved across benchmarks

- The model executing the benchmark is also the optimizer model within each arm.
- Provider guidance determines role-specific sampling and reasoning: temperature
  1.0 and top-p 0.95 for both models; Qwen xhigh and DeepSeek V4.1 numeric effort 100.
- Output limits are 16,384 for HotPotQA and 32,768 for Terminal-Bench per call.
  Context limits remain 262,144 Qwen and 262,144 DeepSeek.
- Model context/output limits still apply when character limits are unlimited.
  All relevant character limits remain independently configurable.
- Controller selection is a Verbalized-Sampling-based 90% model-probability /
  10% uniform-positive-choice adaptation. Zero-probability choices are excluded.
- ReAct editors may correct tool errors and make multiple edits within the
  selected section/action; no fixed turn cap; explicit finish is required.
- Reflection receives task feedback and deduplicated evidence. Keep distinct
  evidence and avoid repeating identical long blocks.
- Evaluation caching is off. HotPotQA DSPy response caches are off. Recovery
  checkpoints and optimizer response journals remain supported.
- Training batches use their own seeded random stream, independent of method
  decisions; larger budgets continue the same batch sequence.

## HotPotQA

The pinned two-stage DSPy program, 150/300/300 train/validation/test split, seed
0, frozen Wiki-2017 BM25 k=7, and three-example reflection minibatches remain.
Per model: standard `vanilla`, `react_v2`, `react_v2_random`, `action` at 6,871
metric calls; expanded independent `vanilla`, `react_v2` at 13,742 calls.
This gives six cells per model, 12 total. Preserve single mutation, merge off,
Pareto parent selection, strict training improvement, and validation-based final
selection. Start operational work with HotPotQA after this consolidation review.

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
| Pilot | Per model, three training tasks then all 30, initial harness only, with usage/cutoff/timeout/throughput review. Same pilot may serve identical initial text in both scopes. |
| Runtime | Verified local model-server identity at pilot, optimization, resume, and final evaluation. Freeze concurrency/hardware/settings per model after training calibration. |
| Final evaluation | Complete all 12 optimization cells per model before test. Initial plus 12 selected harnesses, three test repetitions each; report execution variability. |
| Execution backend | Existing Docker runner retained. Proposed Della Apptainer work explicitly paused; no backend compatibility or live pilot qualification is claimed. |

Canonical detailed protocol and commands:
[`terminal_bench_adapter/README.md`](../src/gepa/adapters/terminal_bench_adapter/README.md).

## Local verification

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
- Gilad to review implemented context deduplication and redundancy removal.
- Review the integrated source and live Della runtime before scheduling HotPotQA.

No messages to collaborators or Della jobs are sent by this consolidation.
The source integration does not establish GPU runtime compatibility or successful
benchmark execution; those require actual subsequent runtime evidence.
