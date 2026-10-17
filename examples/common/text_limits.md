# Optional character limits

The GEPA library exposes a `TextLimits` configuration. The shared benchmark
runner uses its unlimited defaults. Every field defaults to `None` in Python or `null` in JSON,
meaning **unlimited**. These settings are separate from model output-token
budgets and context capacity.

| Setting | Applies to | When configured |
| --- | --- | --- |
| `max_component_chars` | Each complete optimized prompt or skill, including its section headers | Reject an oversized proposed component; the FOREST editor can revise an oversized section before finishing |
| `max_candidate_chars` | Sum of the text in the complete candidate's prompt and skill components | Reject an oversized combined proposal, including merges |
| `selector_target_chars` | Controller and stateless action-selection/rewrite instructions | Add a soft size preference; this does not reject or truncate text |
| `max_prompt_chars` | Each complete Controller, Manifestor, stateless-proposer, or ReAct editor request | Stop before the model call if the assembled request exceeds the limit; includes conversation history and native tool definitions |
| `controller_feedback_chars` | Failure feedback passed to the FOREST Controller | Retain a marked prefix |
| `stateless_feedback_chars` | Feedback passed to the stateless action selector | Retain a marked prefix |
| `manifestor_trace_chars` | Execution evidence passed to the Manifestor | Retain a marked prefix |
| `manifestor_steering_chars` | Generated or fixed guidance passed from Manifestor to editor | Retain a marked prefix |
| `history_text_chars` | Individual saved diagnostic text fields, such as assistant output, feedback, and edit descriptions | Retain a marked prefix; original responses and the replayable edit conversation remain complete |
| `verifier_log_chars` | Each Terminal-Bench verifier stdout/stderr log, including step logs | Retain the beginning and end with an omission marker; source files remain intact |

Lengths use Unicode characters, not UTF-8 bytes or tokens. Evidence cutoffs count
retained **source characters**; their omission markers are additional characters.
The full-request check measures a plain string directly, or compact JSON with
Unicode preserved for structured messages and tool definitions. It never cuts a
complete request to make it fit. Document caps count actual component text and
candidate sums, without JSON serialization overhead.

Library integrations can pass a `TextLimits` instance to the optimizer and
reflection strategy. The shared benchmark CLI keeps the standard unlimited
profile; it does not expose a separate text-limits override.
