# Handover: Action-Conditioned Reflection in GEPA

*Last updated: August 1, 2026. Author: Till, with Claude Code assistance. This file lives outside the mkdocs source tree and is not published to the docs site.*

## What this project is

We are testing whether GEPA's reflective mutation step improves when each mutation is **conditioned on a typed edit action** instead of being free-form, and whether the action should be chosen by **verbalized sampling** (the reflection LM writes out a probability distribution over actions; we sample from its tails for diversity) rather than at random. Hypotheses: better proposal diversity, better credit assignment, and better final performance.

Branch: `rev1_action-conditioned_reflection`.

## What is implemented (all in this branch)

| Piece | Where |
|---|---|
| Action space and selectors (`PromptEditAction`, `RandomActionSelector`, `VerbalizedActionSelector` with distribution history, `DEFAULT_ACTIONS` = 6 generic edit types) | `src/gepa/strategies/action_space.py` |
| Section-scoped structured actions (Rev 2): `build_structured_actions()` = rewrite/append/condense x {Role, Task, Rules, Output Format, Examples} + restructure, 16 actions targeting sections of a markdown prompt skeleton | same file |
| Reflection integration (action suffix on the reflection prompt, action recorded in proposal metadata, selector context injection) | `src/gepa/proposer/reflective_mutation/reflection_lm.py` |
| Observational tracking (per-action proposal/accept/reject counts, score deltas, sibling diversity) | `src/gepa/core/action_tracking.py` (`ActionDiversityCallback`) |
| Benchmarks | `examples/hotpotqa/` (smoke), `examples/ifbench/` (primary, paper-faithful; see its README) |
| Post-hoc analysis (choice entropy, distribution entropy vs uniform, tail-sampling rate, Jaccard proposal diversity) | `examples/ifbench/analyze_actions.py` |
| Cluster tooling (sync, env build, submit, fetch for Princeton della) | `scripts/della/`, `.claude/skills/della/` |
| Tests (47 for the action machinery alone) | `tests/test_action_space.py` |

## Experiment timeline and status

### Rev 1: generic actions, plain prompts (COMPLETE, results written up)

**The first experimental results are `examples/ifbench/FIRST_EXPERIMENTAL_RESULTS.md`, produced July 22-23, 2026.** That document is the historical record; do not edit it. What it covers, and the caveats to keep in mind when reading it:

- Setup: IFBench with the paper's exact data, splits, 2-stage program, metric, and 3,593-call budget, plus a 1-stage ablation. Solver and reflection LM: **Qwen3.5-9B** (not the paper's Qwen3-8B), plain seed prompts, the generic 6-action space. Single run per condition; eval noise is roughly plus/minus 3 points.
- Headline: no condition beat its own baseline on the novel-constraint test set (consistent with the paper's +1.7). The one effect outside noise: 2-stage vanilla GEPA overfit validation and lost ~10 test points, while both action-conditioned runs held at baseline. Action-conditioning acted as a regularizer.
- Two earlier full-run attempts (July 21) were **invalid** and discarded: Qwen's hidden thinking mode returned empty `message.content`, so near-every rollout scored 0 (baselines ~0.5% vs a true ~40%). Archived on della under `outputs/archive_thinking_bug/`. If you see absurdly low baselines anywhere, suspect this first.
- Raw artifacts for the valid runs are on della and mirrored locally under `outputs/della_full_v2/` (jobs 11476476, 11476485-87, 11500500-01).

### Rev 2: structured prompts, section-scoped actions, paper-exact model (COMPLETE)

**Results: `examples/ifbench/SECOND_EXPERIMENTAL_RESULTS.md`, produced August 4, 2026.** Changes vs Rev 1: markdown-skeleton seed prompts with 16 section-scoped actions; the paper's **Qwen3-8B** with the artifact's exact decoding config; full distribution logging for the verbalized selector. Two waves: Wave A at paper budget (replicated and amplified the Rev 1 overfitting story: 2-stage vanilla -18.9 test points vs baseline, action-conditioned -2.0/-4.9) and Wave B at 15,000 rollouts (~50 candidates per run: everyone overfits, condition ranking inconsistent, no clean protection claim).

Key mechanistic findings along the way, both detailed in the writeup:

- Additive actions inflate prompts until they exhaust the model context over long runs; fixed by three-layer length control (soft budget in action suffixes, length-aware selection, 10k-char hard cap at proposal time), commit `20a49c8`.
- The verbalized selector's tail-sampling mechanism is mostly inactive as configured (tail rate 2.5-7.4% with tau=0.10, k=5); what was actually evaluated is LM-guided selection, which does form real, non-uniform action preferences (choice entropy 3.0-3.5 bits vs random's 4.0).

### Rev 3 candidates (not started)

Seed-repeat runs for error bars on the 2-stage overfitting contrast; fix the tail mechanism (tau ~ 1/k) and re-test; holdout-val selection; a high-headroom benchmark (HotpotQA at scale).

## How to monitor / fetch / analyze

```bash
# status (from a machine on the Princeton VPN; <DELLA_USER> is your cluster username)
squeue -u <DELLA_USER>                 # on della
# fetch artifacts to laptop
sshpass -p "$REMOTE_PASSWORD" rsync -az <DELLA_USER>@della.princeton.edu:~/gepa/outputs/ outputs/fetched/
# analyze
uv run python examples/ifbench/analyze_actions.py outputs/fetched
```

Connection config and password live in `scripts/della/.env` (gitignored; ask Till). All submission goes through `scripts/della/submit_ifbench.sh`; the della skill (`.claude/skills/della/SKILL.md`) documents cluster conventions and pitfalls (offline compute nodes, /home quota, backfill).

## Open questions / next steps

1. Wave A results: does structure + section-scoping change the Rev 1 picture (regularization but no gains)?
2. Wave B: do diversity benefits emerge at ~50 explored candidates per run?
3. Action-choice diversity: is verbalized sampling actually diverse (entropy, tail rate from the logged distributions), or does it collapse onto favorite actions?
4. Not yet done, known limitations: seed-repeat runs for error bars; behavioral diversity from per-example val score vectors; embedding-based (e.g. Vendi score) proposal diversity to separate paraphrase churn from strategy diversity; a high-headroom benchmark (HotpotQA at scale) to test acceleration rather than regularization.
