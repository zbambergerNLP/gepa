# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Convert a Controller action into steering text for the proposer.

Fixed-text actions require no model call. Instruction-based actions use at most
two calls and return a user message by default. The Manifestor does not edit the
candidate; ReAct V2 or the RLM proposer applies the selected operation.
"""

from __future__ import annotations

import re
from typing import Any

from gepa.proposer.reflective_mutation.base import LanguageModel
from gepa.strategies.intervention import ControllerAction, InjectionSite, Intervention
from gepa.utils.text import strip_think_tags

MAX_INTERVENTION_CHARS = 1200
MAX_MANIFESTATION_ATTEMPTS = 2
# Traces are the only unbounded input; the candidate, region, and feedback stay whole.
MAX_TRACES_CHARS = 8000

INJECTION_SITE_DESCRIPTIONS: dict[InjectionSite, str] = {
    "assistant_reasoning": (
        "the editor's private reasoning; write in the first person as the editor thinking before it acts"
    ),
    "user": "a user message addressed to the editor",
    "system": "a concise system directive to the editor",
    "developer": "a concise developer directive to the editor",
}

MANIFESTOR_PROMPT = """\
Write the next instruction for a language model editor. After reading it, the editor applies {tool} to region
'{region}'. An atomic-only run may decompose that operation into insert/delete calls.

Message role:
{site}

Action:
- Name: `{spec_name}`
- Description: "{spec_desc}"
- Instruction: "{instruction}"

Requirements:
- Confirm that the selected region and failures support the action's precondition. Otherwise output exactly
  <not_applicable>brief grounded reason</not_applicable> and nothing else.
- Follow the action instruction without adding, skipping, or anticipating steps.
- Ground every claim, failure, and quoted passage in the state.
- Do not write the edit or emit an <edit> or <python> block.
- Return only the steering text, with no header, label, quotation marks, role tag, or process commentary.
- Use at most a few sentences.

State:
{state}
"""

EMPTY_MANIFESTATION_RETRY = """\

Your previous reply contained no visible steering text after hidden-thought tags were removed.
Return a non-empty steering message now, following every requirement above.
"""

STATE_TEMPLATE = """\
## Document being optimized (X)
{full_text}

## Region '{region}' (the edit lands here)
{region_text}

## Failure feedback
{feedback_summary}

## Execution traces
{traces}"""


class ManifestationError(ValueError):
    """Raised when a semantic action cannot produce visible steering text."""


def infer_manifestor_injection_site(_model: str | None) -> InjectionSite:
    """Return the user role used for proposer steering.

    Every provider receives Manifestor output as a user turn, avoiding
    differences in assistant-prefill support.

    Args:
        _model: Provider/model identifier retained for API compatibility.

    Returns:
        Always ``"user"``.
    """
    return "user"


class Manifestor:
    """Realize a :class:`ControllerAction` as an :class:`Intervention`.

    Args:
        lm: Model that writes steering text for instruction-based actions.
        logger: Optional run logger with a ``log(message)`` method.
        max_traces_chars: Execution-trace limit; ``None`` keeps all traces.
        inject_as: Chat role that overrides the action's default role.
    """

    def __init__(
        self,
        lm: LanguageModel,
        logger: Any | None = None,
        max_traces_chars: int | None = MAX_TRACES_CHARS,
        inject_as: InjectionSite | None = None,
    ):
        """Store the LM, logger, traces bound, and configured injection site."""
        self.lm = lm
        self.logger = logger
        self.max_traces_chars = max_traces_chars
        self.inject_as: InjectionSite | None = inject_as

    def _log(self, message: str) -> None:
        """Forward ``message`` to the bound run logger, if any."""
        if self.logger is not None:
            self.logger.log(message)

    def manifest(
        self,
        action: ControllerAction,
        region_text: str,
        full_text: str,
        feedback_summary: str,
        traces: str,
    ) -> Intervention | None:
        """Return steering text for ``action`` or ``None`` when it has no spec.

        Fixed text is returned without an LM call. Instruction-based actions
        retry one empty response, strip hidden thoughts, and enforce the text
        and trace limits.

        Args:
            action: The Controller's joint decision; only its
                ``intervention_spec``, ``edit_target`` and ``edit_tool`` are
                read (the last two so the LM knows which edit will follow).
            region_text: Current text of ``action.edit_target`` (a section body,
                or the whole document for a global target). Shown whole.
            full_text: Current text of the whole component (the candidate
                ``X``), in canonical section format. Shown whole.
            feedback_summary: Summarized minibatch failure feedback. Shown whole.
            traces: Flattened execution traces (inputs, outputs, feedback) of
                the minibatch; the only input this role bounds.

        Returns:
            Steering text and its chat role, or ``None`` without a semantic spec.

        Raises:
            ManifestationError: Steering is blank or the action does not apply.
        """
        spec = action.intervention_spec
        if spec is None:
            return None
        inject_as: InjectionSite = self.inject_as if self.inject_as is not None else spec.inject_as
        if spec.fixed_text is not None:
            if not spec.fixed_text.strip():
                raise ManifestationError(f"InterventionSpec {spec.name!r} has empty fixed steering text.")
            return Intervention(spec.fixed_text, inject_as)
        if self.max_traces_chars is not None and len(traces) > self.max_traces_chars:
            traces = traces[: self.max_traces_chars] + f"\n...(+{len(traces) - self.max_traces_chars} chars)"
        state = STATE_TEMPLATE.format(
            full_text=full_text,
            region=action.edit_target.name,
            region_text=region_text,
            feedback_summary=feedback_summary,
            traces=traces,
        )
        prompt = MANIFESTOR_PROMPT.format(
            tool=action.edit_tool.value if action.edit_tool is not None else "available tools",
            region=action.edit_target.name,
            site=INJECTION_SITE_DESCRIPTIONS[inject_as],
            state=state,
            spec_name=spec.name,
            spec_desc=spec.description,
            instruction=spec.instruction,
        )
        for attempt in range(MAX_MANIFESTATION_ATTEMPTS):
            raw = strip_think_tags(self.lm(prompt)).strip()
            if raw:
                inapplicable = re.fullmatch(r"<not_applicable>(.*?)</not_applicable>", raw, re.DOTALL)
                if inapplicable is not None:
                    reason = inapplicable.group(1).strip() or "the selected action's precondition is not met"
                    raise ManifestationError(f"Semantic action {spec.name!r} is not applicable: {reason}")
                if len(raw) > MAX_INTERVENTION_CHARS:
                    raw = raw[:MAX_INTERVENTION_CHARS] + "..."
                return Intervention(raw, inject_as)
            self._log(
                f"Manifestor returned no visible steering text for action {spec.name!r} "
                f"(attempt {attempt + 1}/{MAX_MANIFESTATION_ATTEMPTS})."
            )
            prompt += EMPTY_MANIFESTATION_RETRY
        raise ManifestationError(
            f"Manifestor produced no visible steering text for action {spec.name!r} after "
            f"{MAX_MANIFESTATION_ATTEMPTS} attempts."
        )
