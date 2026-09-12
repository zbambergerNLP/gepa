# Optional character limits

HotPotQA and Terminal-Bench 2.1 use the same `TextLimits`
configuration. Every field defaults to `None` in Python or `null` in JSON,
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

Both experiment CLIs accept a JSON object; omitted fields remain unlimited:

```sh
--text-limits '{"max_component_chars":10000,"selector_target_chars":8000,"manifestor_steering_chars":1200}'
```

For the Della launcher, set `HOTPOTQA_TEXT_LIMITS_JSON` to the same JSON object.
The launcher validates it and passes it to every campaign cell. All resolved
values enter run contracts and run identity. Changing any value requires a fresh
run directory; final comparisons require matching shared settings.

The Python API accepts the same configuration:

```python
from gepa.strategies.text_limits import TextLimits

limits = TextLimits(max_component_chars=10000, selector_target_chars=8000)
# gepa.optimize(..., text_limits=limits)
# ReflectionConfig(..., text_limits=limits)
# ThreeRoleReflectionLM(..., text_limits=limits)
```

Direct Controller, Manifestor, stateless-reflector, and ReAct constructors also
accept `text_limits`. The shared experiment strategy builder forwards it to all
roles. Existing explicit `max_chars` and `manifestor_traces_chars` arguments on
the three-role strategy override their corresponding settings. An injected
strategy and an explicit front-door configuration must agree. The optional
per-evaluation refiner uses `RefinerConfig(text_limits=limits)` for its prompt,
proposed text, and saved-output fields.

Positive integers enable limits. Zero, negative numbers, booleans, non-integers,
and unknown field names are rejected. With no configuration, all supplied text
remains complete, including repeated passages, feedback, and log lines. No
context deduplication is applied. Conciseness instructions remain guidance,
without a numerical character target.
