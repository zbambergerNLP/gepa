# Preliminary results: action-conditioned reflection for GEPA

**Status: very preliminary. Single run per condition, no error bars - nothing here supports conclusions yet. Shared to show the structure of the outputs, the configuration, and the counters we track.**

*Last updated: August 3, 2026. This file lives outside the mkdocs source tree and is not published to the docs site. Companion docs: `HANDOVER_action_conditioned_reflection.md` (project map), `examples/ifbench/FIRST_EXPERIMENTAL_RESULTS.md` (Rev 1 writeup).*

## Setup

| | Rev 1 (July 22-23) | Rev 2 Wave A (Aug 1-2) |
|---|---|---|
| Benchmark | IFBench, exact GEPA-paper protocol (300 train / 300 val from IF-RLVR constraints; 294 novel-constraint test) | same |
| Task + reflection LM | Qwen3.5-9B | Qwen3-8B (paper's model) |
| Decoding | temp 0.6, top-p 0.95, max 2048 tok | temp 0.6, top-p 0.95, top-k 20, max 16384 tok (paper config) |
| Seed prompts | paper's plain seed sentences | markdown skeleton (Role/Task/Rules/Output Format/Examples), paper sentences preserved |
| Action space | 6 generic edit actions | 16 section-scoped actions (rewrite/append/condense x section + restructure) |
| Optimization budget | 3,593 metric calls per run (paper budget; 1 metric call = 1 full program rollout) | same |
| Other GEPA settings | minibatch 3, strict-improvement acceptance, Pareto candidate selection, no merge | same |

Metric: instruction-level accuracy = fraction of constraints satisfied per example (rule-based checkers, partial credit). "Val" is selection data; "test" is held out until the end. Conditions differ only in the reflection step: *vanilla* = free-form GEPA mutation; *random* = mutation constrained to a uniformly chosen action; *verbalized* = action chosen by verbalized sampling (reflection LM emits a probability distribution over actions; we sample its tails).

## Results (optimized = highest-val candidate, evaluated once on test)

**Rev 2 Wave A** (paper model, structured prompts):

| Program | Condition | Baseline test | Best val | Optimized test | Delta test vs baseline |
|---|---|---|---|---|---|
| 2-stage | vanilla | 37.6% | 0.83 | 18.7% | **-18.9** |
| 2-stage | random | 34.2% | 0.76 | 29.3% | -4.9 |
| 2-stage | verbalized | 31.5% | 0.80 | 29.4% | -2.0 |
| 1-stage | vanilla | 33.0% | 0.80 | 32.7% | -0.3 |
| 1-stage | random | 35.2% | 0.53 | 35.5% | +0.3 |
| 1-stage | verbalized | 32.1% | 0.68 | 32.8% | +0.7 |

**Rev 1** (Qwen3.5-9B, plain prompts) showed the same qualitative pattern: 2-stage vanilla -9.7 points vs its own baseline; action-conditioned runs -0.3 to -0.7; 1-stage all within noise. Full details: `examples/ifbench/FIRST_EXPERIMENTAL_RESULTS.md`.

Consistent observation across both revisions (still needing replication): every method drives val up sharply (0.37 to 0.75-0.83) while test never improves - IFBench's test set uses novel constraint types by design - and *vanilla GEPA's best-val candidate generalizes worst*, i.e., action-conditioning is so far acting as a regularizer against validation overfitting, not an accelerator.

## Counters per run (why run length matters)

Because every accepted mutation costs a full 300-example validation pass (~8% of the 3,593-rollout paper budget), each run explored only **11-14 candidate prompts across ~30-50 reflection iterations**, far too few for diversity mechanisms to matter; Wave B (submitted, queued) raises the budget to 15,000 rollouts per run (~50 accepted candidates) to probe the longer-run regime where saturation binds and we expect the proposed method to differentiate, if it does.

Everything (per-run candidate lineages, per-action proposal/acceptance counts, the verbalized selector's full logged probability distributions, and all rejected proposals) is dumped as JSON per run for deeper analysis (`outputs/<run>/candidates.json`, `action_summary.json`, `run_log.txt`; see `examples/ifbench/README.md`). Branch and docs: https://github.com/TillRS/gepa/pull/1
