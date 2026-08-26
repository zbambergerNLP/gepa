# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Convert a Controller action into steering text for the proposer.

Fixed-text actions require no model call. Instruction-based actions use at most
two calls and return plain steering guidance. The Manifestor does not edit the
candidate; ReAct V2 applies the selected operation.
"""

from __future__ import annotations

from typing import Any

from gepa.proposer.reflective_mutation.base import LanguageModel
from gepa.strategies.intervention import ControllerChoice

MAX_STEERING_MESSAGE_CHARS = 1200
MAX_MANIFESTATION_ATTEMPTS = 2
# Traces are the only unbounded input; the selected region and feedback stay whole.
MAX_TRACES_CHARS = 8000

MANIFESTOR_PROMPT = """\
Write the next instruction for a language model editor. After reading it, the editor applies {tool} to region
'{region}'. An atomic-only run may decompose that operation into insert/delete calls.

Action:
- Name: `{spec_name}`
- Description: "{spec_desc}"
- Instruction: "{instruction}"

Requirements:
- The Controller has already selected this action. Treat that choice as final: do not reassess its preconditions,
  reject it, or substitute another action. Any applicability language in the action instruction is a Controller
  selection rule, not a Manifestor decision.
- Follow the action instruction without adding, skipping, or anticipating steps.
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
## Selected region '{region}'
{region_text}

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
        max_traces_chars: int | None = MAX_TRACES_CHARS,
    ):
        """Configure semantic-action manifestation.

        Args:
            lm: Model that writes steering text for instruction-based actions.
            logger: Optional run logger with a ``log(message)`` method.
            max_traces_chars: Maximum execution-trace characters included in
                the manifestation prompt; ``None`` keeps all traces.
        """
        self.lm = lm
        self.logger = logger
        self.max_traces_chars = max_traces_chars

    def manifest(
        self,
        action: ControllerChoice,
        region_text: str,
        feedback_summary: str,
        traces: str,
    ) -> str | None:
        """Return steering guidance for ``action`` or ``None`` when it has no spec.

        Fixed text is returned without an LM call. Instruction-based actions
        retry one empty response and enforce the text and trace limits.

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
            return spec.fixed_text
        if self.max_traces_chars is not None and len(traces) > self.max_traces_chars:
            traces = traces[: self.max_traces_chars] + f"\n...(+{len(traces) - self.max_traces_chars} chars)"
        state = STATE_TEMPLATE.format(
            region=action.edit_target.section,
            region_text=region_text,
            feedback_summary=feedback_summary,
            traces=traces,
        )
        prompt = MANIFESTOR_PROMPT.format(
            tool=action.edit_tool.value if action.edit_tool is not None else "available tools",
            region=action.edit_target.section,
            state=state,
            spec_name=spec.name,
            spec_desc=spec.description,
            instruction=spec.instruction,
        )
        for attempt in range(MAX_MANIFESTATION_ATTEMPTS):
            raw = self.lm(prompt).strip()
            if raw:
                if len(raw) > MAX_STEERING_MESSAGE_CHARS:
                    raw = raw[:MAX_STEERING_MESSAGE_CHARS] + "..."
                return raw
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
