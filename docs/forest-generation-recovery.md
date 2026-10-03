# Generation recovery

For a generative level-2 Controller, `ThreeRoleReflectionLM` defaults to
`proposal_policy="real_edit"`. It requires the broad tool basis and one atomic
Editor response per attempt. `proposal_policy="independent"` retains the historical
behavior; Jev action selection, random selection and lower levels retain theirs.

The Manifestor must return structured executable guidance. Incompatible, blank,
malformed or incomplete guidance returns a generation error. Finish-only, empty,
invalid, unchanged or canonically whitespace-only Editor output does the same.
The planner tries each remaining executable action/section pair at most once,
on the same selected component and captured training evidence. Positive Controller
weights are exhausted before uniform zero-weight fallback. Exhaustion returns no
candidate and preserves every failure. Provider and journal failures propagate.

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
