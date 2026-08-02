# Handover: Action-Conditioned Reflection in GEPA

*Last updated: August 1, 2026. Author: Till (ts0800), with Claude Code assistance. This file lives outside the mkdocs source tree and is not published to the docs site.*

## What this project is

We are testing whether GEPA's reflective mutation step improves when each mutation is **conditioned on a typed edit action** instead of being free-form, and whether the action should be chosen by **verbalized sampling** (the reflection LM writes out a probability distribution over actions; we sample from its tails for diversity) rather than at random. Hypotheses: better proposal diversity, better credit assignment, and less validation overfitting.

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

### Rev 2: structured prompts, section-scoped actions, paper-exact model (IN FLIGHT)

Changes vs Rev 1, decided July 24: seed prompts get a best-practice markdown skeleton and actions edit specific sections; model switched to the paper's **Qwen3-8B** with the artifact's exact decoding config (temp 0.6, top-p 0.95, top-k 20, max_tokens 16384); the verbalized selector's full distributions are now logged for an action-choice-diversity analysis.

- Validation run passed July 31 (job 11808722: baseline 30% on 20 test examples, seed val 0.70, ~6 min). Structured seeds alone lift the seed val score noticeably (0.70 vs 0.40 in Rev 1).
- **Wave A** (paper budget 3,593; 6 jobs = {2-stage, 1-stage} x {vanilla, random, verbalized}; tag `rev2`): jobs **11843931, 11843934, 11843936, 11843938, 11843940, 11843942**, submitted July 31, still queued on `ailab` as of August 1 (deep queue).
- **Wave B** (exploration scale: 15,000 calls for ~50 accepted candidates per run, tag `15k`): planned, submit after Wave A spot-check.
- Writeup destination when done: `examples/ifbench/SECOND_EXPERIMENTAL_RESULTS.md`, same format as the first.

## How to monitor / fetch / analyze

```bash
# status (from a machine on the Princeton VPN)
squeue -u ts0800                       # on della
# fetch artifacts to laptop
sshpass -p "$REMOTE_PASSWORD" rsync -az ts0800@della.princeton.edu:~/gepa/outputs/ outputs/fetched/
# analyze
uv run python examples/ifbench/analyze_actions.py outputs/fetched
```

Connection config and password live in `scripts/della/.env` (gitignored; ask Till). All submission goes through `scripts/della/submit_ifbench.sh`; the della skill (`.claude/skills/della/SKILL.md`) documents cluster conventions and pitfalls (offline compute nodes, /home quota, backfill).

## Open questions / next steps

1. Wave A results: does structure + section-scoping change the Rev 1 picture (regularization but no gains)?
2. Wave B: do diversity benefits emerge at ~50 explored candidates per run?
3. Action-choice diversity: is verbalized sampling actually diverse (entropy, tail rate from the logged distributions), or does it collapse onto favorite actions?
4. Not yet done, known limitations: seed-repeat runs for error bars; behavioral diversity from per-example val score vectors; embedding-based (e.g. Vendi score) proposal diversity to separate paraphrase churn from strategy diversity; a high-headroom benchmark (HotpotQA at scale) to test acceleration rather than regularization.
