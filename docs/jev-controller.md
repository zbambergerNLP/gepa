# Jev Controller

`controller_selection="jev"` replaces FOREST's generative Controller with a
TypeSafe `Choice` request over the joint action/section menu. It uses the pinned
`jev-1.13.0` model and SDK 0.7.1. Existing runs default to their original Controller.

Jev returns probabilities, not a written rationale. The Manifestor develops the
edit direction from the selected pair and the training evidence; the Editor still
receives the full canonical action constraints independently. This is a new
policy, `jev_joint_action_section_v2`, rather than a behavior-preserving model swap.
It contains neither the outcome-history nor sibling-diversity revisions.

## Setup

```sh
uv sync --extra dev --extra jev --extra wiki17 --group hotpotqa-task-program
export TYPESAFE_API_KEY="$(cat "$HOME/.config/typesafe/api-key")"
```

For a separately authorized HotPotQA run, add these options to the normal command:

```sh
--condition react_v2 --reflection-level 2 --controller-selection jev --tag jev
```

The API key is read from the environment and never included in saved run
contracts. The request goes to `https://api.typesafe.ai/v1/systemone`, not the local
model server. The solver, Manifestor and Editor keep their existing model settings.
Do not pass a Jev key as a command-line argument.

The benchmark derives a distinct run key and refuses incompatible checkpoint
resumption. `--enforce-scientific-contract` rejects Jev because it is not part of
the qualified Della campaign. None of this changes an existing source snapshot,
export, checkpoint, queued worker or controller.

Programmatic use:

```python
from gepa.proposer.reflective_mutation.three_role import ThreeRoleReflectionLM
from gepa.strategies.jev_controller import JevController

controller = JevController(
    response_journal_path="new-run/.lm-response-journal/responses.sqlite3",
    attempt_log_path="new-run/jev-provider-attempts.jsonl",
)
strategy = ThreeRoleReflectionLM(
    base_lm=editor_lm,
    manifestor_lm=manifestor_lm,
    level=2,
    controller_selection="jev",
    jev_controller=controller,
    editor_mode="single_call",
    template_family="alibaba",
)
# Close controller.close() when the owning application finishes.
```

## Selection and recovery

- Send every component section, its description, full structured training traces,
  and every action's canonical constraints. No evidence is silently truncated.
- Describe each action by what it changes and what it must preserve. Classify the
  intended correction before selecting its action and section: adding a behavior
  rule changes meaning, while adding background preserves every operative rule.
  These selection glosses do not change the canonical constraints used by the
  Manifestor and Editor. Version 2 has a separate request and run identity from
  version 1, so existing decisions cannot silently resume under revised wording.
- Exclude delete, replace and move operations on empty sections, recording each
  reason. Other existing menu entries remain eligible.
- Validate the returned model, complete probability map, argmax, confidence and
  usage. Locally sample with the existing seeded 90% model probability plus 10%
  uniform-positive-support mixture. The API's argmax is recorded separately.
  Model-assigned zeros remain zero.
- Use at most four physical attempts for transient transport errors, HTTP 408,
  429 or 5xx, sharing one 30-second deadline. SDK retries are disabled. Backoff
  does not consume the selection RNG; numeric `Retry-After` is respected within
  the deadline. Authentication, context-limit and malformed-output errors stop
  immediately. No failure invokes the generative Controller or another sample.
- Journal a completed response before exposing it. Iteration replay restores that
  response and the optimizer's RNG, without another API call or another charge.
  Batch fallback rewinds journal cursors together with the sampling state.

The provider currently limits a request to 64k tokens overall and 32k for state
plus the longest question. This integration has one question, so the 32k limit
applies. Oversized evidence fails visibly; it is not clipped or rerouted.
See [TypeSafe's model documentation](https://docs.typesafe.ai/models).

## Accounting

`jev-provider-attempts.jsonl` records a `started` event before every physical call
and a `finished` event with its response, usage, duration and retry status. An
unmatched start identifies interrupted work whose server usage is unknown.
Failed responses remain in this ledger, with credentials redacted. Unknown token
usage is `null`, not an assertion of zero provider consumption.

Proposal metadata records raw and sampled probabilities, excluded pairs, the
selected action, API argmax, request identity, replay status, latency and usage.
Cost is an estimate from the documented input price of **$0.042 per million
tokens**, with free output tokens, pinned in the policy as of September 27, 2026.
It is not an invoice. Known usage from failed attempts remains charged.

## Verification

Mocked HTTP tests exercise the real SDK, downstream role routing, probability
sampling, mechanical exclusions, invalid responses, authentication failures,
bounded retries, batch fallback, accounting and deterministic response replay.
Existing Controller tests cover compatibility with the original defaults.

An earlier version-1 integration check used the first saved training context for each of the
four components, recovered from the checksum-verified completed FOREST archive.
Each contained the original three training examples and the complete action menu
before mechanical exclusions. Four API requests succeeded without retries:

| Component | API seconds | Input tokens | Output tokens | Estimated USD |
|---|---:|---:|---:|---:|
| summarize1 | 0.260 | 9,289 | 279 | 0.000390 |
| create_query_hop2 | 0.140 | 5,082 | 280 | 0.000213 |
| summarize2 | 0.141 | 7,748 | 279 | 0.000325 |
| final_answer | 0.127 | 5,066 | 279 | 0.000213 |

Total estimated cost: **$0.001142**. Median API latency: **0.140 seconds**.
These are four Controller-only checks, not optimization results. No candidates
were generated or evaluated, and no held-out data was used. The sampled action
was `contextualize/Response` in all four checks (each used a reset seed-0 sampling
RNG); that is not evidence of action diversity or edit quality. A future matched
training-only pilot must assess useful edits and total proposal-cycle latency,
including any changed Manifestor work, before claiming preserved performance.

Local evidence is in `outputs/jev-controller-live-check-20260927/`: the full
request/response ledger, replay journal and `proof.json` with source-file hashes.
These operational files are ignored and are not part of a deployment.

### Version-2 selection diagnostic

Four wording variants were compared on ten explicit edit-classification cases.
The selection score was probability on the fixed correct action/section under
the unchanged 90/10 sampler, with invalid responses scored zero. The combination
of contrastive descriptions and effect-first guidance won and was frozen before
querying 20 fresh authored cases, two per action. All labels and requests were
saved before querying. This was a small manual variant comparison, not a GEPA
optimization run or an external benchmark.

| Fresh cases | Version 1 | Version 2 |
|---|---:|---:|
| Correct top pair, including rejected responses | 17/20 | 20/20 |
| Valid response | 12/20 | 15/20 |
| Correct top pair and valid response | 10/20 | 15/20 |
| Mean correct-pair sampling probability; invalid = 0 | 28.6% | 54.0% |
| Mean input tokens | 8,898 | 9,995 |
| Median API seconds | 0.157 | 0.168 |

Five version-2 responses were rejected because their probabilities summed to
0.99. The strict validator, failure accounting and no-reroll behavior remain
unchanged. Thus 20 correct top labels do not mean 20 usable decisions. Even among
the seven cases with valid responses in both arms, correct-pair probability
increased from 55.8% to 75.7%; this restricted comparison is descriptive.

On twelve archived training states, version 1 preferred `contextualize` in all
twelve; version 2 preferred it in five and `restrict_meaning` in seven. Both had
eleven valid responses. These are top choices, not accepted edits, and those
states have no unique action gold labels. Some failures originate in another
component, so choosing a different action does not necessarily fix them.

The study made 104 physical Controller calls, including 19 rejected responses,
with no transport retries, at an estimated total cost of $0.039144. It made no
Manifestor, Editor, solver, validation or test requests. This supports improved
classification on the authored cases, not improved optimization scores or
production readiness. The remaining distribution failures need resolution before
deployment. Full ignored evidence is in `outputs/jev-definition-study-20260927/`.
