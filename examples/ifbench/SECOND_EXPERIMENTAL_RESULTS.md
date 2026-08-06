# IFBench Rev 2: Structured Prompts, Section-Scoped Actions, Paper-Exact Model

**TL;DR:** On the paper's exact model (Qwen3-8B) with structured markdown prompts and section-scoped actions, the Rev 1 pattern replicated at paper budget: no method beats the unoptimized baseline on IFBench's novel-constraint test set, and vanilla GEPA's best-val candidate overfits worst (-18.9 points vs its own baseline, with action-conditioned runs at -2.0 to -4.9). At 4x budget (~50 explored candidates per run), however, every method overfits substantially and the condition ranking becomes inconsistent across programs, so Wave B does not support a clean protection claim. Two solid mechanistic findings: additive actions inflate prompts until they exhaust the model context unless explicitly length-controlled, and the verbalized selector's tail-sampling mechanism is mostly inactive as configured (tail rate 2.5-7.4%), making it closer to LM-guided softmax selection than true tail exploration.

*Single run per condition throughout; eval noise on the 294-example test set is roughly plus/minus 3 points. Treat every delta below accordingly.*

## Setup (deltas vs Rev 1)

Same benchmark protocol as Rev 1 (`FIRST_EXPERIMENTAL_RESULTS.md`): IFBench with the GEPA paper's exact data, splits (300 train / 300 val / 294 novel-constraint test), 2-stage program (plus 1-stage ablation), instruction-level accuracy metric, and Pareto/strict-improvement GEPA settings. Changes:

- **Model**: Qwen3-8B, the paper's model, with the artifact's exact decoding config (temp 0.6, top-p 0.95, top-k 20, max_tokens 16384). Thinking mode stays disabled (see Rev 1 pitfalls).
- **Structured seeds**: each component is a markdown skeleton (`## Role / ## Task / ## Rules / ## Output Format / ## Examples`) with the paper's seed sentence preserved in Task. All conditions, including vanilla, use structured seeds.
- **Action space**: 16 section-scoped actions ({rewrite, append, condense} x 5 sections + restructure) instead of the 6 generic Rev 1 actions.
- **Distribution logging**: the verbalized selector records every generated probability distribution, the sampled actions, and parse fallbacks.

Waves: **A** = paper budget (3,593 rollouts), jobs 11843931-42 + reruns; **B** = 15,000 rollouts targeting ~50 accepted candidates, jobs 11961663/67 (vanilla) and 12001198-202 (action runs, rerun under Rev 2.1 length control after the original action runs crashed; see below).

## Wave A results (paper budget, 11-14 candidates per run)

| Program | Condition | Baseline test | Best val | Optimized test | Delta |
|---|---|---|---|---|---|
| 2-stage | vanilla | 37.59% | 0.825 | 18.71% | **-18.9** |
| 2-stage | random | 34.18% | 0.764 | 29.25% | -4.9 |
| 2-stage | verbalized | 31.46% | 0.804 | 29.42% | -2.0 |
| 1-stage | vanilla | 32.99% | 0.795 | 32.65% | -0.3 |
| 1-stage | random | 35.20% | 0.531 | 35.54% | +0.3 |
| 1-stage | verbalized | 32.14% | 0.678 | 32.82% | +0.7 |

Reading: clean replication of Rev 1, amplified. 2-stage vanilla reaches the highest val score and collapses hardest on test; both action-conditioned 2-stage runs stay near baseline; 1-stage moves nothing. Baselines are lower than Rev 1's (~32-38% vs ~38-47%), consistent with the smaller model. Structured seeds alone raised the seed's val score substantially (0.37 vs 0.25-0.40 spread in Rev 1 validation probes).

## Finding: additive actions inflate prompts until the context dies

The first Wave B attempt crashed all four action-conditioned runs after 4-9 hours: over 23-45 acceptances, accretive actions (append, illustrations) grew candidate prompts to ~28,700 input tokens, exhausting Qwen3-8B's 32,768-token context mid-rollout. The two vanilla runs, whose wholesale rewrites are naturally self-limiting, completed unharmed. This is a real long-horizon failure mode of additive edit actions: nothing in the action space pushed back on length, so unbounded growth was the equilibrium.

**Rev 2.1 length control** (commit `20a49c8`), applied to the rerun action jobs: (1) every action suffix states an 8,000-char prompt budget and instructs merging or replacing over adding when near it; (2) the verbalized selector sees the current prompt length vs budget and is told to favor condense/rewrite/restructure when long; (3) a hard cap drops action-conditioned proposals over 10,000 chars at the reflection step. In the rerun, the cap fired 2-10 times per run and all four jobs completed with full budgets. Caveat: Wave B's vanilla runs predate this commit (irrelevant to vanilla, which uses no actions, but the runner's context-overflow retry ladder differs slightly).

## Wave B results (15k budget, 46-49 candidates per run)

| Program | Condition | Baseline test | Best val | Optimized test | Delta | Candidates |
|---|---|---|---|---|---|---|
| 2-stage | vanilla | 31.97% | 0.858 | 24.83% | -7.1 | 49 |
| 2-stage | random | 36.56% | 0.825 | 24.83% | -11.7 | 47 |
| 2-stage | verbalized | 34.18% | 0.833 | 32.31% | **-1.9** | 47 |
| 1-stage | vanilla | 34.01% | 0.850 | 29.59% | -4.4 | 46 |
| 1-stage | random | 34.18% | 0.858 | 30.27% | -3.9 | 49 |
| 1-stage | verbalized | 34.69% | 0.834 | 23.98% | -10.7 | 47 |

Reading, honestly: longer optimization made everyone overfit more (val 0.83-0.86 everywhere; every run below its baseline on test), exactly as the saturation argument predicted. But the condition ranking is inconsistent: 2-stage verbalized is the best performer (-1.9, extending its Wave A story) while 1-stage verbalized is the worst (-10.7), and random flips the other way. With one run per cell, Wave B does not support ranking the conditions; the defensible claim is only that all methods overfit at this scale and that no condition escapes it.

Why the overfitting: the winning prompts bake in val-set constants. Vanilla's 2-stage Wave B winner literally hard-codes "use exactly 8 highlighted sections", "at least 15 sentences", and "begin by repeating the user's query verbatim" as universal rules, which win val points on the constraint families present in validation and actively violate unseen test constraints.

## Action-choice diversity (the verbalized-sampling check)

From the logged distributions (Wave A + B verbalized runs; ~19-220 selection calls each):

| Statistic | Verbalized runs | Random baseline | Uniform reference |
|---|---|---|---|
| Choice entropy over sampled actions | 2.97-3.48 bits | 3.96 bits | 4.0 bits (16 actions) |
| Per-call distribution entropy | 2.08-2.17 bits | n/a | 2.32 bits (k=5 candidates) |
| Tail-sampling rate (p < 0.10) | **2.5-7.4%** | n/a | - |
| Parse-fallback rate | 0.0% everywhere | n/a | - |

Three conclusions. First, verbalized selection **is** meaningfully non-uniform: it concentrates on a preferred subset of actions (roughly 8-11 effective actions vs random's 16), so the LM forms real preferences from the feedback. Second, the tail-sampling mechanism is **mostly inactive as configured**: with k=5 candidates whose probabilities sum to 1, almost no candidate falls under tau=0.10, so the tail filter falls back to full-distribution sampling on more than 92% of draws. The mechanism as deployed is LM-guided sampling, not tail exploration; raising tau (e.g. to 1/k) or increasing k would be needed to actually exercise the tails. Third, parsing is fully reliable on Qwen3-8B (zero fallbacks across every run).

Proposal-level lexical diversity (mean pairwise Jaccard distance over all proposals) adds a scale effect: at 15k budget, **vanilla's proposals become the least diverse** (0.61-0.75 vs 0.74-0.82 for action-conditioned runs), a reversal of the short-run picture: over long runs, unconstrained reflection converges to paraphrases of its incumbent winner, while the rotating action constraint keeps forcing different edit types.

## Caveats

- One run per cell; the test-eval noise floor (~plus/minus 3 points) swallows most deltas except the vanilla collapses.
- Wave B action runs use Rev 2.1 length-controlled instructions; Wave B vanilla runs predate them (no effect on vanilla behavior, noted for exactness).
- The 2-stage verbalized Wave A run was resumed after an infrastructure crash (port collision, since fixed); its action counters cover only the post-resume segment, though its candidate lineage is complete.
- Baselines vary by up to ~5 points across identical seed prompts (sampling at temp 0.6), so cross-row baseline differences are noise, not signal.

## Next steps

1. **Seed-repeat runs** (3-5 per condition) on the 2-stage program at paper budget: the vanilla-collapse vs action-conditioned contrast is the one effect big enough to power with few repeats.
2. **Fix the tail mechanism** (tau ~ 1/k or larger k) and re-test whether true tail exploration changes anything; current results only evaluate LM-guided selection.
3. **Val-overfitting diagnostics**: behavioral diversity from per-example val score vectors, and holdout-val selection (select on a split the optimizer never scores against).
4. A high-headroom benchmark (HotpotQA at scale) to test acceleration rather than regularization.

## Artifacts

Jobs (della, Aug 1-4, 2026): Wave A 11843931/34/38/40, reruns 11901069/70, eval-recovery 11919577/79/82; Wave B vanilla 11961663/67; Wave B action (crashed) 11961664/65/68/69; Wave B action rerun 12001198/99/201/202. Per-run dumps (candidates + lineage, action summaries with full distribution histories, all proposals, candidate trees) fetched under `outputs/della_rev2/`; analysis via `examples/ifbench/analyze_actions.py` (writes `action_analysis.json` per run).
