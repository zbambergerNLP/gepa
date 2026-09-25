# Real edits and diversity among siblings

This local review revision starts from `72d0e8c42977e2d11a4b16f85771750702150dee`.
It does not include the separate outcome-history/soft-penalty revision or change any existing experiment.
Its policy identity is `forest-sibling-real-edits-v1` (Controller contract version 7).

## Scope of diversity

The permanent key is **parent node ID + component + semantic action + section**.
Only a changed component entering an accepted child consumes that choice. Each child
starts with an empty outgoing ledger, even if another component or its complete text
is identical to an ancestor or another node.

| Edges for the same component | Result |
| --- | --- |
| A → B: contextualize/Style; A → D: contextualize/Style | Blocked |
| A → B: contextualize/Style; A → D: contextualize/Objective | Allowed |
| A → B: contextualize/Style; B → C: contextualize/Style | Allowed |

Rejected candidates, generation errors, and ancestors' failed attempts consume no
permanent choices. There is no inherited blacklist and no prompt-hash exclusion.

## Candidate generation

The default generative level-2 `ThreeRoleReflectionLM` uses this policy. It requires
one atomic Editor response with a net change after canonical section rendering.
The historical `proposal_policy="independent"` remains available explicitly for
comparisons. Level 0, level 1, and uniform-random selection keep their previous behavior.

1. Keep the existing perfect-minibatch skip before any role is called.
2. Remove accepted sibling pairs, temporary batch reservations, and pairs already
   attempted in this parent/minibatch opportunity. Record mechanical exclusions,
   including missing target text or an unavailable direct operator.
3. Begin with the selected component. The Controller receives the full canonical
   action catalog and training evidence. It supplies relative weights and an intended
   change that fits each pair; Python normalizes the weights.
4. The Manifestor receives the same catalog and the selected direction. It returns
   JSON with `status="ready"`, `observation`, `hypothesis`, `general_change`, and `scope`,
   or `status="incompatible"` with a reason. Incompatibility returns to the planner.
5. The Editor receives that same catalog and direction, and stages an ordered batch
   against the selected section. Every call must be valid. A finish-only response,
   invalid batch, or net-unchanged result is a generation error, including changes
   erased by canonical rendering. No partial batch becomes a candidate.
6. On error, try another pair or component against the **same saved training traces**.
   Remaining positive weights use 90% normalized weights plus 10% uniform exploration
   over positive choices. Once positives are exhausted, executable zero-weight pairs
   are sampled uniformly. Other components are considered before an all-zero fallback.
   Each pair is attempted at most once per parent/minibatch opportunity.
7. A changed candidate gets the normal training evaluation. A tie or loss ends its
   proposal. It does not buy another candidate-generation attempt. Exhaustion is
   reported explicitly, and the exhausted parent is deferred for the next draw when
   another eligible frontier node exists; otherwise the next scheduled minibatch proceeds.

The structural checks enforce a valid, changed, section-scoped edit. Semantic agreement
still depends on the roles interpreting the shared constraints correctly; it is not a
proof of improved task accuracy.

## Acceptance and interruption

Same-parent proposals in a batch share temporary reservations. The engine checks the
whole selected batch before validation and rechecks each child at insertion. It records
the accepted choice before subsequent reporting callbacks.

The optimizer checkpoint stores accepted choices, node edges, deferred parents, and RNG
state. `sibling-recovery.sqlite3` stores completed scoring/edit steps, including failed
attempts and post-step RNG state, with request and response hashes. Recovery keys use
stable optimizer iteration numbers, parent IDs, minibatch IDs, and proposal slots;
display iteration IDs can change on restart and are not used as recovery identities.
Keep this journal alongside `gepa_state.bin`, the evaluation journal, role response
journals, and provider-attempt ledgers when archiving or resuming.

Restart replays completed recovery steps and rebuilds temporary reservations. The
existing role response journals replay completed calls inside an interrupted step.
Custom role clients must also provide durable response replay to avoid repeating a
completed call in such an unfinished step. Work interrupted before a durable response
exists may need to be repeated; every physical attempt remains charged separately.
Provider transport/output retries retain their existing shared allowance.

## Reporting and verification

`ActionDiversityCallback` reports accepted sibling pair counts per parent/component
separately from repeated pairs along ancestor paths. It retains generation errors,
generation exhaustion, perfect-batch skips, evaluated ties/losses, and proposal text
records. A failed attempt never inherits the score or acceptance of the later successful edit.

`tests/test_sibling_diversity.py` covers sibling invariants, vertical reuse, identical
text on distinct nodes, component fallback, positive/zero-weight recovery, atomic
errors, exhaustion, batch reservations, engine acceptance, and deterministic restart
of both recovery and partially generated sibling batches. Historical role tests
explicitly select the independent policy.

This revision requires its own future model qualification. It is not deployed and is
not used by the current Della comparison.
