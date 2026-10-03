# Edit diversity and training outcome memory

`forest-diversity-quality-v1` is an opt-in proposal policy for local review. It
extends the sibling policy with evaluated-edit memory and a duplicate check before
training evaluation. It has not been qualified in a live optimization run.

## Enable the policy

```python
from gepa.proposer.reflective_mutation.three_role import ThreeRoleReflectionLM

reflection = ThreeRoleReflectionLM(
    editor_lm,
    controller_lm=controller_lm,
    manifestor_lm=manifestor_lm,
    level=2,
    edit_tool_set="broad",
    proposal_policy="diversity_quality",
    novelty_backend="generative",  # or "jev"
)
```

Pass this strategy to the normal optimizer with a persistent `run_dir`. This
checkout uses the generative level-2 Controller. The verifier backend is a separate
choice; selecting Jev here changes verification, not Controller selection. The
separate Jev Controller experiment is unchanged.

| Verifier | Model and decision |
| --- | --- |
| `generative` | Reuses the Manifestor model and its provider boundary. Requests a structured `duplicate`, `distinct`, or `uncertain` verdict with a reason. |
| `jev` | Uses TypeSafe Jev through the pinned `typesafe-sdk==0.7.1`. Chooses deterministically among distinct, uncertain, and one duplicate alternative per prior edit. |

For Jev, install the `jev` extra and set `TYPESAFE_API_KEY` in the environment.
The optimizer binds separate verifier response and attempt journals. Jev verification
cannot issue requests without these journals and a logical request scope.

## What is excluded

Permanent action exclusions are keyed by **parent node, component, action, and
section**. Only insertion of an accepted child consumes a pair. Each child starts
with an empty outgoing record. Equal prompt text on different nodes does not share
exclusions or outcome memory. Repeating an action further down a branch remains
allowed.

The existing skip when the parent answers all three training questions correctly
is retained. Otherwise, the Editor must produce one valid atomic batch with a net
change. Invalid, empty, incompatible, and unchanged generations return to the
planner. Recovery tries each available pair at most once in that parent/minibatch
opportunity: positive-weight pairs first, then executable zero-weight pairs.

A changed candidate that passes verification is evaluated normally. A training tie
or loss ends that proposal; it does not trigger another generation attempt.

## Duplicate verification

The comparison set contains evaluated edits from the same parent and component,
plus pending edits in the same proposal batch. It excludes ancestor and unrelated
node history.

1. An exact repeated parent/proposal text pair on the same ordered training batch
   is blocked without a model call.
2. Other proposals are compared semantically against up to eight recent eligible
   edits. The verifier receives the full parent and proposal texts, prior texts,
   training traces, and available training outcomes. Text overlap is recorded as
   a diagnostic; it cannot alone reject a candidate.
3. The verifier must identify the previous attempt it considers redundant.
   Different training evidence can justify a similar edit. Missing evidence or an
   uncertain judgment leaves the proposal eligible for evaluation.
4. A blocked proposal and its verdict remain in the generation record. The
   Controller receives the matching edit and feedback, then replans among remaining
   pairs or components using the same current training traces.

For Jev, a duplicate alternative must have the largest normalized probability and
at least `0.8` probability to block an edit. This threshold is an engineering
heuristic, not calibrated confidence. No exploration is applied to verification.
A generative response that fails verdict-schema validation is recorded as
uncertain. Provider, journal, and invalid Jev distribution failures stop the run
instead of silently bypassing verification.

## Training outcome memory

Only changed proposals with matched parent/child training scores enter outcome
memory. Generation errors and duplicate rejections receive no invented score.
Validation and held-out scores are never fed into this memory.

The Controller sees the latest eight outcomes for the selected parent/component,
with compact diffs. Repeated ties or losses for an action/section receive a gentle
weight reduction before normalization and the existing exploration mixture:

```text
multiplier = max(0.5, 1 / (1 + 0.25 * max(0, unsuccessful_attempts - 1)))
```

The first unsuccessful attempt has no penalty. A strict training improvement resets
the matching count. Counts use the recent window and exact parent component text;
the multiplier never removes an otherwise eligible pair. This is a fixed heuristic,
not an Optuna study or a learned bandit.

## Persistence and accounting

The checkpoint records the complete policy and verifier contract, evaluated edits,
temporary batch reservations, and RNG state. Switching verifier backends cannot
silently reuse a checkpoint. Recovery steps retain canonical inputs and their
results so an interrupted proposal reproduces the same choices and metadata.

Keep `sibling-recovery.sqlite3`, the optimizer checkpoint, evaluation and role
response journals, and provider ledgers together. Jev additionally writes
`novelty-responses.sqlite3` and `novelty-provider-attempts.jsonl`.

Novelty records include the verdict, matching attempt, evidence, elapsed time, and
available cost/input/output-token deltas. Unavailable usage is `None`; paths that
make no model call record zero. Jev also preserves its raw provider usage and
probabilities. Provider retries retain their existing shared allowance. A response
lost before durable storage can require repeated work; every physical attempt must
remain accounted for.

Reports distinguish duplicate generations, other generation errors, exhaustion,
training ties/losses, and accepted children. Sibling pair reuse and action reuse
along ancestor paths remain separate measures.

## Verification boundary

The tests cover sibling and descendant scope, identical text on independent nodes,
exact and semantic duplicates, changed training evidence, uncertain decisions,
zero-weight recovery, batching, outcome attribution, verifier contracts and exact
interruption replay. Model responses are mocked. These checks establish software
behavior; they do not establish semantic-verifier accuracy, improved task scores,
or lower end-to-end cost. Those require a separately authorized experiment.

## Bounded training pilot

`examples.hotpotqa.diversity_quality_pilot` compares three arms with the same
generative Controller, Manifestor, Editor, solver, data and decoding:

- Sibling rules alone (`sibling_diverse`).
- Combined memory and duplicate policy with generative verification.
- Combined memory and duplicate policy with Jev verification.

There are three fixed opportunities for each of the four components: twelve
opportunities, with three proposal-training questions each. Every opportunity
starts from original parent node 0. Each arm retains its own evaluated-edit memory
and real accepted children, so later proposals can exercise sibling exclusions.
Strict improvement on the three proposal questions admits a child. No validation
score is invented, and these children are not a completed Pareto search.

Twelve separate training questions measure transfer after all proposal generation
and training decisions finish. They never enter reflection or admission. In total,
the protocol uses 48 distinct training questions, at most 36 proposed candidates,
and at most 588 task evaluations. Recovery, verifier and startup model calls are
additional and recorded separately. Execution rotates the arm order.

The comparison reports paired transfer changes, accepted edits, generation errors,
duplicate blocks, verifier uncertainty and cost. Repeated transfer evaluations are
paired observations on the same twelve questions, not independent new examples.
Coverage of each mechanism is reported from actual execution; the pilot does not
force duplicate edits or fabricate history to make a gate activate.

Use the native prepared-pilot launcher with stage `diversity-quality`. The pilot
has a separate source identity and output directory; it does not resume or replace
completed standard/expanded searches or the separate Jev Controller pilot.
