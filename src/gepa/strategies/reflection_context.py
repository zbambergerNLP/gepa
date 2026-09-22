"""Pin the policy of preserving repeated reflection evidence."""

REFLECTION_CONTEXT_CONTRACT = {
    "version": 2,
    "duplicates": "preserved",
    "repeated_lines": "preserved",
    "unique_text": "preserved_without_character_truncation",
}

GENERALIZATION_GUIDANCE = """\
Improve a reusable component for future inputs. Training examples and reference answers are diagnostic evidence,
not standing instructions. Derive procedures, conditional rules, or stable task knowledge; do not transplant an
individual example's answer or incidental facts into the document. The lesson should still apply when incidental
entities change while task requirements stay the same. Preserve legitimate task constants such as API names,
output fields, units, and fixed environment requirements.
Distinguish observed input/output mismatches from hypotheses about their cause. A final task score is an end-to-end
outcome, not evidence that this component alone caused it. Reference information in feedback may not have been
available in the component's inputs. Do not treat missing input evidence as an observed omission from its output.
State when a change applies and preserve already-correct behavior outside that scope. A plausible lesson is a
hypothesis until evaluation supports it. Respect the selected semantic action; never reinterpret a context-only
action as permission to introduce a new behavioral requirement. If it cannot express a supported improvement,
preserve a legitimate no-op rather than force an incompatible edit.
"""

CONTROLLER_AUTHORITY_GUIDANCE = """\
The Controller owns the selected region, semantic action, and intended direction. Its selected action's rationale
is passed separately as controller direction. The Manifestor grounds and specifies that direction; the Editor
executes it. Neither downstream role may choose a different action, substitute a different improvement goal,
or expand the scope. Observation, Hypothesis, General change, and Scope explain how to realize the Controller's
decision; they are not a second action-selection step. The catalog's action constraints remain binding even when
a rationale or steering message conflicts with them. If the selected direction is unsupported or cannot be
realized within those constraints, preserve a legitimate no-op rather than redirect the proposal.
When no Controller rationale is supplied, realize only the selected region/action using the evidence; do not
invent a Controller rationale. This includes the deliberately uniform-random Controller ablation.
"""

FOREST_REFLECTION_CONTRACT = {
    "version": 2,
    "controller_evidence": "ordered_complete_reflection_records",
    "outcome_attribution": "end_to_end_not_component_causal",
    "role_guidance": GENERALIZATION_GUIDANCE,
    "controller_authority": CONTROLLER_AUTHORITY_GUIDANCE,
    "controller_direction": "selected_option_rationale_passed_to_manifestor_and_editor",
    "manifestor_structure": ["Observation", "Hypothesis", "General change", "Scope"],
    "editor_action_contract": "original_catalog_description_and_instruction",
    "semantic_correction_retries": False,
}
