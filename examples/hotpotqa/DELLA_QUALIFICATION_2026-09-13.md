# Della interactive measurements: September 13, 2026

Qwen3.8-27B passed the three-question HotPotQA smoke test and one complete FOREST cycle on **one H200**, generating about **63.77 tokens/s** without output cutoffs. DeepSeek-V4.1-Flash served successfully on four H200s, but its earlier FOREST editor request needed **46,901 output tokens** before returning its first tool call. Neither model has completed the entire campaign qualification.

No production decoding, token caps, benchmark data, or optimizer settings were changed. The larger output allowance was confined to a separate measurement requested by the user.

## Allocation and source

- Job: `13806836`, node `della-i20g3`; one interactive allocation, four H200s, 32 CPUs, 512G host memory, requested time 55 minutes.
- Runtime source: `246b9cba66331e211d8ed7fca812a81969d347fd`; the complete source manifest was verified before execution. The frozen remote source was not edited.
- The allocation started at 06:19:48 UTC. Slurm still reported it running after the requested time; it was explicitly canceled at 07:16:56 UTC. Accounting records 57:08 elapsed and final state `CANCELLED`. No additional allocation or batch job was submitted. The allocation-follow-up automation was paused.
- Only training examples were used. No held-out validation or test evaluation was run. The tiny optimizer diagnostic reused three training examples as its local evaluation set; that is not the production validation split.

## Natural completion of the editor turn

The Controller and Manifestor responses from the interrupted optimizer check were replayed locally to reconstruct its first editor request. The Controller, Manifestor, and editor request hashes matched their original response-journal records. No new model inference was used to reconstruct the request.

The editor then received the same model request and decoding settings with its fixed output cap removed for this diagnostic. Its remaining context supplied an output allowance of 256,094 tokens. This was a fresh replay of the first editor request, without the subsequent cutoff-recovery history.

| Measurement | Observed value |
| --- | ---: |
| Provider-reported input tokens | 6,030 |
| Output tokens | **46,901** |
| Reasoning tokens, included in output | **46,285** |
| Other output tokens, including tool syntax | 616 |
| Total context consumed | 52,931 |
| Request elapsed time | **397.734 seconds / 6m 38s** |
| Output tokens / request elapsed time | **117.92 tokens/s** |
| Decode throughput from server token-latency counters | **118.09 tokens/s** |
| Finish reason | `tool_calls` |
| Returned action | One `INSERT_TEXT` call with JSON arguments |

The tokenizer endpoint counted 6,050 tokens for its tokenizer-facing request; the actual completion reported 6,030. The larger count was used to reserve output space conservatively. The original request hash is `e812dc1b76b7c533c55d19bf5b427426969023c8b27d61191a93aba7891c8913`.

For this observed turn, 16,384 and 32,768 tokens were insufficient; 65,536 would have covered it. This measures one editor turn ending in a tool call, not the complete edit-and-submit sequence or the largest requirement across the dataset. A universal cap remains a decision to make after more measurements.

Before the larger-cap diagnostic, the editor's first two turns each returned `finish_reason=length` with all 16,384 output tokens attributed to reasoning. The editor continued after the first cutoff, using the existing corrective protocol feedback. The optimizer was interrupted before a complete proposal and reevaluation; no complete-cycle marker was produced.

## HotPotQA execution and runtime estimates

The existing four-component HotPotQA program completed two passes over the same three training questions: six full evaluations and 24 solver calls. The seed scored 2/3 exact match on this tiny set. Metric improvement was not a qualification requirement.

| Three-question pass | Solver calls | Output tokens | Model-request span | Span / completed question |
| --- | ---: | ---: | ---: | ---: |
| Initial evaluation | 12 | 29,650 | 252.119 s | 84.040 s |
| Reflection minibatch | 12 | 12,481 | 107.926 s | 35.975 s |

Spans run from the earliest recorded request start to the final request completion in each pass, including intervening queueing and pipeline gaps. They exclude initial dataset/client setup. The second pass briefly overlapped the separate tool-compatibility diagnostic. These are two passes over **three unique questions**, not six independent examples or a full calibration.

The average observed cost is about 60.0 seconds per full evaluation under the current single-active-sequence server. The following is conditional planning arithmetic if that tiny sample represented the eventual workload; it is not a reliable production completion forecast or a confidence interval.

| Ablation | Metric-call budget | Estimated evaluation time | Range using the two observed pass speeds |
| --- | ---: | ---: | ---: |
| Vanilla GEPA | 6,871 | 114.5 h | 68.7–160.4 h |
| FOREST / `react_v2` | 6,871 | 114.5 h | 68.7–160.4 h |
| FOREST random Controller / `react_v2_random` | 6,871 | 114.5 h | 68.7–160.4 h |
| Action-only / `action` | 6,871 | 114.5 h | 68.7–160.4 h |
| Vanilla GEPA, double budget | 13,742 | 229.1 h | 137.3–320.8 h |
| FOREST, double budget | 13,742 | 229.1 h | 137.3–320.8 h |

These estimates cover full question evaluations. The metric budget already counts training and optimizer-validation evaluations; validation must not be added a second time. Add about **5.0 hours per ablation** for its 300-question held-out test at the same assumed speed, and **5.0 hours once per model** for the shared starting baseline. Those held-out evaluations have not been executed.

Optimizer-model calls and startup are additional. Observed FOREST calls took 116.237 seconds for the Controller, 74.014 seconds for the Manifestor, and 397.734 seconds for the fresh larger-cap editor turn. Their sum is about 9.8 minutes, before a subsequent finish turn. This is a partial proposal measurement, not a completed iteration. Proposal counts depend on search behavior, acceptance, and validation work; vanilla, random-Controller, and action-only proposal overheads were not measured. A full ablation estimate therefore remains:

`startup + evaluation budget × seconds per evaluation + proposal-model time + held-out test time`

The serving profile remained TP4/EP4/DP1, one active sequence, one API server, native mixed FP4/FP8 weights, FP8 KV, Engram CPU offload, context 262,144, temperature 1.0, top-p 0.95, and reasoning effort 100. No throughput-tuning restart was performed.

## Memory, startup, and completed checks

- DeepSeek reported 71.46 GiB for model loading per GPU. Its slowest worker took 493.6 seconds for model loading, followed by 142.75 seconds for engine initialization, profiling, cache setup, and warmup. Checkpoint and environment verification were separate startup costs.
- Peak sampled GPU use was **137,689 MiB / 134.46 GiB per GPU**, out of 143,771 MiB visible capacity. This includes reserved cache and runtime allocations; it is not a minimum model-memory measurement. Long-context capacity was not fully exercised.
- Host cgroup memory peaked at **511.97 GiB**, with substantial file-cache pressure and rereads during loading. No OOM was observed. The setup served successfully, but startup memory headroom was tight.
- DeepSeek passed ordinary completion, native tool-result continuation, and one real proposal through each of `INSERT_TEXT`, `DELETE_TEXT`, `REPLACE_TEXT`, and `MOVE_TEXT`. This was the standalone four-edit diagnostic, not the campaign's 20-attempt canary.
- 42 HotPotQA/Terminal-Bench pilot-logic tests passed on Della. These are CPU tests, not real Terminal-Bench task execution.
- A quoting bug in the standalone DeepSeek diagnostic's embedded package-inventory Python was fixed locally. The repair and serving-smoke regression checks passed: 22 tests, Ruff, and Bash syntax validation.
- In the initial shared allocation, Qwen's source, checkpoint, and environment preflight passed. It loaded BF16 weights on one H200 and completed graph compilation, but was stopped before endpoint readiness and inference. The subsequent Qwen-only allocation below completed those checks.
- A real Terminal-Bench pilot could not run: Harbor and Docker were absent. Apptainer was available, but the current GEPA Terminal-Bench adapter requires Docker.

## Qwen follow-up: job 13816818

The separately approved interactive allocation used one H200, 8 CPUs, and 128G host memory on `della-i24g2`. It started at **07:41:19 UTC** and was explicitly released at **08:35:00 UTC**, after **53:41** of the requested 55 minutes. Slurm records `CANCELLED by 377417`; the queue was subsequently empty. The dedicated SSH shell was closed and the follow-up automation paused. No additional allocation, batch job, or DeepSeek run was started.

Two live failures exposed source bugs, both fixed on `codex/consolidated-della-experiments`:

- `dfc21a18de9d0a62f5f96aaa5f4930ae3aa4f05a`: FlashInfer's CUDA build selected incompatible wheel runtime headers through `CPATH`. Using `C_INCLUDE_PATH` and `CPLUS_INCLUDE_PATH` as system-header fallbacks preserved the cluster compiler's own headers while retaining missing cuBLAS headers. Four C/C++ regression checks passed, and the exact previously failing GDN kernel compiled on Della. The restarted server and FOREST diagnostic used this source, with manifest `8253eb54564b2718f980dd7a7ad8b8dd2f9b56504697e7887380e3fee5aea811`.
- `17d8896ea3e404cece13bf4b1696e5475f233f76`: the HotPotQA scientific guard still required eight GPUs for every model. It now requires one for Qwen and four for DeepSeek, including DeepSeek's TP4/EP4 and Engram CPU offload. All 163 configuration tests passed. The corrected guard was checked against the live Qwen metadata, then the standard HotPotQA smoke entry point passed using this source and manifest `dd17c4c91418b52864bc7b551960faa10c3ee3bb3a6265ddd88f3a920c9cfba4`. The running vLLM server was retained: this second repair changed client validation, not serving code or decoding.

The initial launch used `476c7bdb6a4a7bc007b40ef673cf5f375c6fc5c1`. Each source was staged separately; frozen source directories and environments were not edited in place.

### Completed execution and token measurements

| Check or measurement | Verified result |
| --- | --- |
| Ordinary inference | Paris; 30 output tokens in 0.63 s |
| Corrected HotPotQA smoke | **PASS**; three training questions, 12 solver calls, 237.697 s |
| Full FOREST cycle | **PASS**; 1,105.683 s / **18m 25.7s**, nine full question evaluations |
| FOREST decision | Candidate tied the seed at 2/3 exact match and was correctly rejected |
| FOREST editor | One `INSERT_TEXT` action followed by explicit `FINISH`; no editor errors |
| Server totals | **64 completed requests; 85,360 output tokens** |
| Decode throughput | **63.77 tokens/s**, from 85,296 inter-token intervals over 1,337.636 s |
| Largest completed output | **11,844 tokens**, the Controller call; 187.804 s |
| Manifestor | 5,616 output tokens; 89.243 s |
| Editor, both turns together | 710 output tokens; 12.284 s |
| Largest solver output | 10,878 tokens |
| Output cutoffs | **None** in the FOREST and corrected HotPotQA smoke logs |

The FOREST check covered actual reflection, action/section selection, Manifestor feedback, editing, explicit completion, reevaluation of all three questions, and rejection. Both its completed-cycle marker and the corrected smoke marker passed the repository's integrity checks after retrieval. Metric improvement was not required. The FOREST cycle made 36 solver calls plus four optimizer calls; the corrected smoke added 12 solver calls. These are **12 full evaluations of three unique training questions**, not 12 independent questions or a held-out result. The diagnostic candidate remains separate from production seeds and shared baselines.

The short native-tool diagnostic was **not an overall pass**: ordinary completion, tool-result continuation, `DELETE_TEXT`, and `MOVE_TEXT` passed. `INSERT_TEXT` and `REPLACE_TEXT` each applied a valid first edit, but their next turns chose a disallowed tool and did not finish within that diagnostic's two-turn allowance. The subsequent full FOREST editor did finish correctly under its normal unbounded-turn policy. Preserve both results; the later success does not erase the two failed toy probes.

The operational eight-minute FOREST watchdog was paused to permit the complete cycle, with the allocation deadline preserved. Afterward, the launcher was paused before the other methods to prioritize the corrected smoke test. The outer serving wrapper reached its planned timeout after both checks had completed. Its Slurm step exit `124` is therefore separate from the verified `PASS` artifacts. Vanilla, random-Controller, and action-only live cycles were **not run** in this allocation.

Qwen used vLLM 0.25.1, TP1/DP1, one active sequence, BF16 weights, automatic/BF16 KV cache, context 262,144, temperature 1.0, top-p 0.95, top-k 20, `xhigh` reasoning, and a 16,384-token output cap. Three question pipelines ran concurrently; full 12-worker calibration was not performed. Qwen's API did not report separate reasoning-token counts, so those remain unknown. The output counts above include reasoning.

### Memory, startup, and planning arithmetic

Qwen reported **50.22 GiB** for model loading, taking 10.168 s including setup. It reserved **75.12 GiB** for KV cache. Peak sampled GPU use was **132,371 MiB / 129.27 GiB**, out of 143,771 MiB. Host cgroup memory peaked at **110.59 GiB** out of 128 GiB, including file cache and compilation. No host OOM or server preemption was recorded. This proves the tested workload fits; it does not measure minimum required memory or exercise the full 262,144-token context.

The repaired startup reused its Torch graph in 4.67 s, but building and warming the remaining FlashInfer kernels made engine initialization take **562.06 s**. Checkpoint/environment verification was additional. A subsequent startup with the newly populated kernel cache was not timed.

The corrected seed smoke took **79.23 s per full question evaluation**. Reevaluation of the diagnostic candidate took **115.42 s per question**, despite its tied score. These observed costs illustrate why prompt changes can change runtime as well as quality.

| Qwen budget | Evaluation-only projection using those two observed costs |
| --- | ---: |
| Standard vanilla, FOREST, random Controller, or action-only: 6,871 calls | **151.2–220.3 h** |
| Expanded vanilla or FOREST: 13,742 calls | **302.4–440.6 h** |

These are conditional calculations from three questions, not measured ablation runtimes, statistical intervals, or reliable completion forecasts. On one GPU the numbers also equal GPU-hours. The metric budget already includes optimizer training and validation evaluations. Add startup, proposal-model time, and held-out evaluation: this FOREST proposal's four optimizer calls took **4.82 minutes** in total, but its eventual proposal count is unknown. The same assumed costs imply **6.60–9.62 h** per 300-question held-out test and **6.60 h once** for the shared unoptimized baseline at the seed rate. None of those held-out runs was performed.

## Evidence and remaining decisions

Remote evidence: `/scratch/gpfs/BSTEWART/gm8296/gepa/logs/hotpotqa/verify/13806836/`.

Local evidence is retained under `outputs/hardware-sizing-20260912/interactive-evidence/13806836/`, with the raw archive at `outputs/hardware-sizing-20260912/13806836-evidence.tar.gz`. Key files are `measurement-summary.json`, `deepseek-uncapped/result.json`, `deepseek-uncapped/response.json`, `deepseek-tool-verification.log`, original provider-attempt logs, GPU/host-memory samples, server metrics, and Qwen's startup log.

Qwen evidence is retained remotely under `/scratch/gpfs/BSTEWART/gm8296/gepa/logs/hotpotqa/verify/13816818/` and locally under `outputs/hardware-sizing-20260912/interactive-evidence/13816818/`, with archive `outputs/hardware-sizing-20260912/13816818-evidence.tar.gz`. It includes the final measurement summary, Slurm accounting, both startup logs, compiler probes, native-tool failures, complete FOREST cycle and request logs, corrected smoke records/marker, GPU/host-memory samples, and final server metrics. `outputs/hardware-sizing-20260912/summarize-qwen-13816818.py` reproduces the summary from these retained files.

Still to decide or verify:

1. Select production output limits after reviewing the 46,901-token editor turn and sampling more natural finishing lengths. Production caps remain unchanged.
2. Complete DeepSeek's full FOREST cycle and its required 20-attempt campaign canary. Qwen's single FOREST cycle is now verified.
3. Measure other optimizer ablations and a larger training sample before freezing runtime/resource estimates or concurrency.
4. Complete the remaining Qwen optimizer ablations and larger calibration, review the failed short tool probes, and prepare the actual Terminal-Bench container runtime.

Qwen's smoke test and one complete FOREST cycle are verified. DeepSeek inference and standalone tool compatibility are verified. The full 150-question/model pilot matrix, production campaigns, and live Terminal-Bench execution remain unqualified.


## Follow-up qualification and recovery fixes

Source `a07ea72e8b476a0d71e53d7805a8f90892cbd40c` was exercised on Qwen in interactive jobs 13834255 and 13836224. All four native edit probes passed (11 requests, 1,554 output tokens); the three-question smoke passed in 241.13 seconds with 14,361 output tokens and no errors or cutoffs. Vanilla GEPA and FOREST each completed reflection, proposal, three-question reevaluation, and ordinary tied-candidate rejection. Their request totals were 37 / 51,548 output tokens and 40 / 68,874 output tokens, respectively. These remain historical results for that exact source.

The random Controller selected MOVE_TEXT on an empty Tone section. The Manifestor confused feedback with the editable body, and the editor could not finish without a mutation. Fourteen invalid editor responses were saved before the first allocation's planned 53-minute timeout. The next allocation replayed those responses and the Manifestor response, but rebuilt the parent feedback through 12 repeated solver calls. The rebuilt feedback happened to match exactly; repeated inference is unnecessary and could invalidate reflection replay if its output differs.

The follow-up code makes the selected body explicit as a JSON string with its character count, repeats the actual current body in edit-error observations, and separates feedback from editable text. An editor may explicitly finish an inapplicable action; the unchanged proposal is discarded. Fixed action menus, multiple edits, and unlimited production turns/tool calls remain in force. Optimizer pilots now continue after dropped attempts until they record a proposal, reevaluation, and decision; they do not require metric improvement.

Completed parent and child evaluation batches are journaled by optimizer iteration and phase, including original outputs, scores, trajectories, and adapter state. Restart replays only that interrupted logical occurrence; a later iteration evaluates again even with identical inputs. Checksums and exact input identity reject corrupt or mismatched records. A partially completed batch still needs to restart; only completed batches are durable at this boundary.

A focused Qwen GPU diagnostic with the corrected context ended the empty-section action in one turn without editing feedback. A separate unbounded MOVE formatting diagnostic was interrupted after repeated attempts and is not a completed-cycle qualification. Full optimizer pilots, selected-concurrency calibration, and live interruption/replay must be verified on the new committed source before campaigns are qualified. Current caps are 131,072 tokens for DeepSeek optimizer roles, 16,384 for both solvers and all Qwen roles. DeepSeek's next qualification allocation uses 640G host memory for headroom above the previous 512G peak.

Evidence: `outputs/hotpotqa-qualification-20260913/` and its `live-state.json`, including the original pilot and server artifacts, 44 completed-file checksums before resume, and isolated empty-section diagnostics. No held-out data was used for these changes.
