# Generation recovery

For level-2 generative, Jev and uniform-random Controllers, `ThreeRoleReflectionLM` defaults to
`proposal_policy="real_edit"`. It requires the broad tool basis and one atomic
Editor batch per attempt. `proposal_policy="independent"` retains the historical
behavior; lower reflection levels are unchanged. The recovery identity is
`forest-real-edit-recovery-v2`.

The Manifestor must return structured executable guidance. Incompatible, blank,
malformed or incomplete guidance returns a generation error. Finish-only,
invalid, unchanged or canonically whitespace-only Editor output does the same.
When a native Editor returns text without native function calls, it gets one
protocol correction on the same action, section, original text and evidence.
The original response is retained as diagnostic data. XML text is never executed
as a substitute for native calls. A second failure returns to pair selection.
Finish-only responses immediately return to pair selection.
The planner tries each remaining executable action/section pair at most once,
on the same selected component and captured training evidence. Positive Controller
weights are exhausted before uniform zero-weight fallback. Exhaustion returns no
candidate and preserves every failure. Provider and journal failures propagate.

Jev supplies one journaled distribution per selected component. The planner
samples remaining pairs without replacement after a generation failure, without
calling a generative Controller or repeatedly billing for the same Jev decision.
The first sample uses the same seeded probability mixture as independent selection.
This is separate from Jev's existing four-attempt, one-deadline handling of invalid
typed responses and transport failures; exhausting those retries still stops the run.

Only a changed candidate is evaluated. An evaluated tie or loss is never retried.
Failed generation records never inherit that candidate's training or validation
score. Completed recovery steps, provider responses and RNG state are journaled
for interruption replay. The run contract rejects incompatible checkpoints.

All actions remain available on later opportunities, including the same parent
and same-parent batched proposals. This feature adds no sibling exclusions,
duplicate verifier, training-outcome memory, parent deferral or module fallback.
Those experimental policies are reviewed separately. Existing frozen allocations
and source identities must not be migrated to this policy.

Mock tests verify recovery, bounded exhaustion, unchanged evidence, no retry after
scoring, and replay without repeated completed calls. They do not establish an
increase in accepted proposals or held-out accuracy. The earlier diversity pilot
already used recovery in every arm and cannot measure its causal benefit.

## Archived no-edit audit, October 3, 2026

A no-edit outcome here means a completed proposal cycle whose
`action_summary.json` record has empty `texts_by_component`. Perfect-batch skips
never enter this denominator. Changed proposals that subsequently tie, lose, or
fail admission are separate outcomes.

| Complete archived search | Proposal cycles | No edit | Finish without editing | Text instead of native calls |
| --- | ---: | ---: | ---: | ---: |
| Generative expanded search | 250 | 82 (32.8%) | 73 | 9 |
| Jev standard continuation | 156 | 58 (37.2%) | 50 | 8 |

These are different searches, not a controlled comparison of no-edit rates.
The original 6/20 (30%) generative snapshot is included in the expanded search,
not counted again. The Jev predecessor and continuation form one search.

All 140 outcomes had zero native tool calls. The regression fixture
`tests/fixtures/forest_observed_noops.json` retains every original Editor response,
Manifestor response, action, selected component and original section. Its source
metadata records input hashes. Journal response hashes were verified, and the
round-robin component schedule was reconstructed and matched against every
changed component and the final saved component cursors. These complete archives
do not describe later running allocations.

| Observed failure | Example fixture | Recovery |
| --- | --- | --- |
| Context-only action asked to add operative rules | `generative_expanded-1` | Manifestor reports incompatibility; try another pair before evaluation. |
| Surface-only rewrite cannot fix a semantic problem | `jev_standard-5` | Jev selections now enter the same finite pair-recovery loop. |
| Manifestor explicitly directs finishing without editing | `jev_standard-3` | Require structured actionable guidance; an incompatible result never reaches the Editor. |
| Empty insertion region mistaken for a missing target; reusable prompt confused with a task answer | `jev_standard-29`, `jev_standard-126` | Clarify that insertion needs no existing anchor and task inputs are in the traces; preserve semantic action constraints. |
| Intended edit returned as XML rather than native calls | `jev_standard-19`, `generative_expanded-100` | Expose the selected operator's native schema and allow one protocol correction. |

Every one of the 17 text-only batches changed its original section when its
literal arguments were manually represented as native calls and replayed through
the strict executor. Those scripted corrections are included in the fixture.
This proves mechanical executability, not semantic validity, model correction
success, or acceptance. Under `real_edit`, all 123 finish-only responses are
generation errors eligible for pair recovery instead of ending the proposal opportunity.

The tests also cover exhaustion, failed protocol correction, atomic rollback,
interruption replay, and not retrying scored ties/losses. They make no inference
calls. A live before/after acceptance or held-out improvement remains unmeasured;
existing frozen jobs and their scientific results are unchanged.
