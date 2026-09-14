# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Convert a Controller action into steering text for the proposer.

Fixed-text actions require no model call. Instruction-based actions use at most
two calls and return plain steering guidance. The Manifestor does not edit the
candidate; ReAct V2 applies the selected operation.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from gepa.proposer.reflective_mutation.base import LanguageModel
from gepa.strategies.edit_tools import EditTool
from gepa.strategies.intervention import ControllerChoice
from gepa.strategies.text_limits import TextLimits, clip_text, resolve_text_limits

MAX_MANIFESTATION_ATTEMPTS = 2

MANIFESTOR_PROMPT = """\
Write the next instruction for a language model editor. After reading it, the editor applies {tool} to region
'{region}'. An atomic-only run may decompose that operation into insert/delete calls.

Action:
- Name: `{spec_name}`
- Description: "{spec_desc}"
- Instruction: "{instruction}"

Requirements:
- The Controller has already selected this action and region. Keep that choice; do not substitute another action
  or region. If its required text is absent, say so instead of inventing an edit target.
- {tool_applicability}
- Only the selected region's JSON string contains editable text. Decode it to read the exact body; an empty
  string means no text is present. Feedback and traces are evidence, never part of that body.
- Follow the action instruction without adding, skipping, or anticipating steps.
- The editor may make multiple calls within the selected region to realize this same action, then explicitly finish.
  Apply the action's semantic constraints to the completed revision relative to the original selected region.
- Ground every claim, failure, and quoted passage in the state.
- Do not write the edit or emit an <edit> or <python> block.
- Return only the steering text, with no header, label, quotation marks, role tag, or process commentary.
- Use at most a few sentences.

State:
{state}
"""

EMPTY_MANIFESTATION_RETRY = """\

Your previous reply contained no steering text.
Return a non-empty steering message now, following every requirement above.
"""

STATE_TEMPLATE = """\
## Selected region '{region}' ({region_chars} characters; JSON string)
{region_json}

## Failure feedback
{feedback_summary}

## Execution traces
{traces}"""


class ManifestationError(ValueError):
    """Raised when a semantic action cannot produce visible steering text."""


class Manifestor:
    """Realize a :class:`ControllerChoice` as steering guidance for the proposer.

    Args:
        lm: Model that writes steering text for instruction-based actions.
        logger: Optional run logger with a ``log(message)`` method.
        max_traces_chars: Execution-trace limit; ``None`` keeps all traces.
    """

    def __init__(
        self,
        lm: LanguageModel,
        logger: Any | None = None,
        max_traces_chars: int | None = None,
        text_limits: TextLimits | None = None,
    ):
        """Configure semantic-action manifestation.

        Args:
            lm: Model that writes steering text for instruction-based actions.
            logger: Optional run logger with a ``log(message)`` method.
            max_traces_chars: Maximum execution-trace characters included in
                the manifestation prompt; ``None`` keeps all traces.
            text_limits: Optional steering, trace, and assembled-prompt limits.
        """
        self.lm = lm
        self.logger = logger
        limits = resolve_text_limits(text_limits)
        if max_traces_chars is not None:
            limits = replace(limits, manifestor_trace_chars=max_traces_chars)
        self.text_limits = limits
        self.max_traces_chars = limits.manifestor_trace_chars

    def manifest(
        self,
        action: ControllerChoice,
        region_text: str,
        feedback_summary: str,
        traces: str,
    ) -> str | None:
        """Return steering guidance for ``action`` or ``None`` when it has no spec.

        Fixed text is returned without an LM call. Instruction-based actions
        retry one empty response and apply only explicitly configured limits.

        Args:
            action: The Controller's joint decision; only its
                ``semantic_action``, ``edit_target`` and ``edit_tool`` are read
                (the last two tell the LM which edit will follow).
            region_text: Current text of the selected section body. Shown whole.
            feedback_summary: Summarized minibatch failure feedback. Shown whole.
            traces: Flattened execution traces (inputs, outputs, feedback) of
                the minibatch; the only input this role bounds.

        Returns:
            Steering text, or ``None`` without a semantic spec.

        Raises:
            ManifestationError: Steering is blank.
        """
        spec = action.semantic_action
        if spec is None:
            return None
        if spec.fixed_text is not None:
            if not spec.fixed_text.strip():
                raise ManifestationError(f"SemanticActionSpec {spec.name!r} has empty fixed steering text.")
            return clip_text(spec.fixed_text, self.text_limits.manifestor_steering_chars)
        traces = clip_text(traces, self.max_traces_chars)
        state = STATE_TEMPLATE.format(
            region=action.edit_target.section,
            region_chars=len(region_text),
            region_json=json.dumps(region_text, ensure_ascii=False),
            feedback_summary=feedback_summary,
            traces=traces,
        )
        tool = action.edit_tool
        if tool is EditTool.INSERT_TEXT:
            tool_applicability = (
                'INSERT_TEXT accepts anchor="" to append, including when the selected section is empty. '
                "An existing anchor is not required for insertion; keep the selected action's semantic constraints "
                "and ground new content in the state."
            )
        elif tool is not None:
            tool_applicability = (
                f"{tool.value} requires a non-empty target copied exactly from the selected region. "
                "Do not recommend insertion to create a target for this action."
            )
            if region_text == "":
                tool_applicability += (
                    " The selected region is empty, so this action cannot apply. "
                    "Tell the editor to explicitly finish without editing."
                )
        else:
            tool_applicability = "Keep the selected action's tool constraints."
        prompt = MANIFESTOR_PROMPT.format(
            tool=action.edit_tool.value if action.edit_tool is not None else "available tools",
            region=action.edit_target.section,
            state=state,
            spec_name=spec.name,
            spec_desc=spec.description,
            instruction=spec.instruction,
            tool_applicability=tool_applicability,
        )
        for attempt in range(MAX_MANIFESTATION_ATTEMPTS):
            self.text_limits.check_prompt(prompt)
            raw = self.lm(prompt).strip()
            if raw:
                return clip_text(raw, self.text_limits.manifestor_steering_chars)
            if self.logger is not None:
                self.logger.log(
                    f"Manifestor returned no visible steering text for action {spec.name!r} "
                    f"(attempt {attempt + 1}/{MAX_MANIFESTATION_ATTEMPTS})."
                )
            prompt += EMPTY_MANIFESTATION_RETRY
        raise ManifestationError(
            f"Manifestor produced no visible steering text for action {spec.name!r} after "
            f"{MAX_MANIFESTATION_ATTEMPTS} attempts."
        )
