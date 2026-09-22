# FOREST evidence and generalization qualification

This revision gives the generative Controller complete ordered reflection
records, including actual component inputs and outputs. All three roles receive
the same guidance to infer a reusable, scoped lesson from training evidence.
The Controller chooses the region and action and supplies the intended direction
in that option's rationale. The sampled rationale reaches both downstream roles
verbatim and is recorded with the decision. The Manifestor's observation,
hypothesis, general change, and scope concretize that direction; they do not
authorize it to choose a different improvement goal. The Editor receives that
same Controller direction independently of the Manifestor, along with the
original selected action's description and instruction. Conflicting advice or
unsupported direction permits a no-op, not a substitute action. Random Controller
selection supplies no invented model rationale. The action catalog, sampling,
no-op handling, single-response
editing, and atomic rollback behavior are unchanged. These are instructions to
the models, not a semantic validator or a guarantee of generalization.

HotPotQA's richer diagnostic fields are enabled for FOREST and its random
Controller ablation. They distinguish the final exact-match outcome from causal
attribution to a component, and state the component's downstream information
boundary. Vanilla and stateless action baselines keep the original feedback.
The reflection and run contracts change, so old checkpoints cannot silently
resume under this revision. Existing campaigns continue on their pinned source.

## Separate technical and usefulness checks

The `generalization` interactive qualification stage first runs the existing
technical preliminary qualification: endpoint and thinking-budget checks,
native edit probes, real optimizer cycles, and training throughput. It then
executes `examples.hotpotqa.generalization_pilot` on the same model allocation.
The control source must be supplied explicitly as an immutable staged directory
and exact commit; both source manifests are verified before inference.

Before any comparison inference, a sealed contract records:

- Original prompts, runtime, model/retrieval identities, both sources, and all
  example records and their order.
- Four independent three-question proposal batches: ordered training indices
  0–2, 3–5, 6–8, and 9–11, one per program component.
- Twenty-four transfer questions at ordered training indices 12–35. These never
  enter reflection, acceptance, or subsequent proposals.
- Eight separately reported synthetic cases: descriptions without identities,
  explicitly stated identities, relationships across passages, and direct-answer
  controls. Their fixed retriever tests use of evidence, not BM25 performance.

For each component, each revision generates exactly one proposal from the same
original parent and saved solver traces. Control and revised execution order
alternate by component. The control imports its old source and receives its
original reflection fields; the revised source receives the added diagnostics.
No candidate is selected for another optimization step. Every valid proposal is
evaluated on its proposal batch, the reserved transfer questions, and the
synthetic cases, including proposals GEPA would reject on the proposal batch.
No-ops retain their original scores and are counted as no-ops without repeating
identical solver work. Invalid batches remain ordinary discarded proposals.

At most 324 distinct metric evaluations are planned for the comparison:
44 original evaluations plus 8 proposals × 35 cases. The technical preliminary
qualification is additional work. Actual evaluations, task-format errors,
physical provider attempts, retries, and any recovery work are recorded
separately. This bounded diagnostic is not a GEPA optimization-budget run.

The preregistered positive usefulness signal requires the revised mean across
all four transfer proposals to exceed both the control mean and the original
mean, without additional synthetic regressions compared with the control.
Per-component wins, losses, already-correct regressions, and synthetic category
results remain visible. Technical completion can succeed while usefulness is
not demonstrated. Four proposal pairs and 24 transfer questions are too small
to establish robust generalization; no validation/test tuning, final held-out
claim, campaign replacement, or automatic production promotion follows.

Inspect all role prompts, semantic action selections, edits, scored records,
tracking errors, runtime identities, and provider logs before qualification
review. The synthetic cases and small development set diagnose this revision;
they must not be relabeled as independent scientific results.
