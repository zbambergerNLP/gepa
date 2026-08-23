# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Manifestor role of the 3-role reflection architecture.

The Manifestor receives the Controller's :class:`~gepa.strategies.intervention.ControllerAction`
(the selected ``EditTarget`` + ``EditTool`` + ``InterventionSpec``) together with
the reflection state, and *manifests* the spec into a concrete
:class:`~gepa.strategies.intervention.Intervention`, following POSIT's
manifestation step. A spec with ``fixed_text`` is used verbatim with no model
call. A spec with an ``instruction`` is realized by at most two LM calls that write the
steering text the instruction calls for, in the voice the spec's ``inject_as``
site requires (by default a portable user message),
grounded in the current candidate and its failures. The Manifestor deliberately
does **not** write the edit itself: it steers, and ReAct V2 performs the edit.

POSIT manifests deterministically (temperature 0, thinking off). The
``LanguageModel`` seam carries no sampling knobs, so callers that want that
pass an LM configured that way (``ThreeRoleReflectionLM(manifestor_lm=...)``;
:func:`gepa.optimize` builds one when ``reflection_lm`` is a model name).
"""

from __future__ import annotations

import re
from typing import Any

from gepa.proposer.reflective_mutation.base import LanguageModel
from gepa.strategies.intervention import ControllerAction, InjectionSite, Intervention
from gepa.utils.text import strip_think_tags

# Cap the manifested intervention so it steers without ballooning the proposer context.
MAX_INTERVENTION_CHARS = 1200
# Retry once when the visible reply is empty after hidden-thought removal.
MAX_MANIFESTATION_ATTEMPTS = 2
# Default bound on the execution traces shown to the Manifestor. The candidate,
# the region and the feedback summary are always shown whole (they are the state
# the steering text must be grounded in); traces are the one unbounded input.
# ``None`` shows them whole too.
MAX_TRACES_CHARS = 8000

# How each ``inject_as`` site reads to the Editor, so the Manifestor writes in
# the matching voice (POSIT's injection-site descriptions, addressed to an editor).
INJECTION_SITE_DESCRIPTIONS: dict[InjectionSite, str] = {
    "assistant_reasoning": (
        "the editor's own private thinking -- it will be shown to the editor as its own opening reasoning, which "
        "it reads and continues from before it acts, so write it in the first person, as the editor thinking to "
        "itself"
    ),
    "user": (
        "a user turn addressed to the editor, which the editor will read and respond to -- write it the way a "
        "user speaking to the editor would phrase it"
    ),
    "system": (
        "a system instruction that conditions the editor's next turn -- write it as a concise, authoritative "
        "directive to the editor"
    ),
    "developer": (
        "a developer instruction that conditions the editor's next turn -- write it as a concise, authoritative "
        "directive to the editor"
    ),
}

MANIFESTOR_PROMPT = """\
You steer a language model editor one step at a time.
Write the text that will steer its next step, following the instruction below exactly.
Your text should read naturally as the editor's next move and push it toward fixing the failures the document \
caused.

NOTE: After your text, the editor inspects the state and realizes this action on region '{region}'. Its direct
semantic tool is {tool}; an atomic-only run may decompose that operation into insert/delete calls.

Where your text goes:
{site}

Hard requirements:
- First verify that the selected action's stated precondition is evidenced in the selected region and failures. If it
  is not, output exactly <not_applicable>brief grounded reason</not_applicable> and nothing else.
- Carry out the instruction precisely; do not add, skip, or anticipate steps it does not ask for.
- Ground every claim, failure, and piece of text you name in the state below.
- Do not write the edited text yourself and do not emit an <edit> or <python> block.
- Output only the text itself: no headers, labels, quotation marks, role tags, or commentary about what you are doing.
- Keep it concise and natural -- a few sentences at most.

STATE SO FAR:
{state}

A single next step has been chosen for the editor.
It is defined by:
- Name: `{spec_name}`
- What it accomplishes: "{spec_desc}"
- Instruction to carry out (write text that does exactly this): "{instruction}"
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
    """Choose the portable user-message role used to steer the proposer.

    The Manifestor emits a user turn for every provider. This avoids assistant
    prefills, which are not portable across closed-model APIs, while preserving
    the same steering semantics for OpenAI, Anthropic, and other providers.

    Args:
        _model: Provider/model identifier retained for API compatibility.

    Returns:
        Always ``"user"``.
    """
    return "user"


class Manifestor:
    """Realizes a :class:`ControllerAction` into a concrete :class:`Intervention`.

    The Manifestor is the middle role of the 3-role pipeline (Controller ->
    Manifestor -> ReAct V2) and runs only at reflection level 2. Given the
    Controller's chosen ``(EditTarget, EditTool, InterventionSpec)`` and the
    reflection state, it produces the steering text the Editor will receive at
    the configured ``inject_as`` site. It steers but never edits: ReAct V2
    is the only role that changes the candidate.

    One instance is cheap and stateless between calls, so
    :class:`~gepa.proposer.reflective_mutation.three_role.ThreeRoleReflectionLM`
    constructs one per component per reflection.

    Args:
        lm: Language model that writes the steering text for ``instruction``
            specs. POSIT manifests deterministically (temperature 0, thinking
            off); pass an LM configured that way to match, since the
            ``LanguageModel`` seam itself carries no sampling knobs. Not called
            for ``fixed_text`` specs.
        logger: Optional run logger with a ``log(message)`` method; ``None``
            silences the role.
        max_traces_chars: Bound on the execution traces included in the state
            shown to the LM. The candidate, region and feedback summary are
            always shown whole; traces are the one unbounded input, so they are
            head-truncated with a ``(+N chars)`` marker past this many
            characters. ``None`` shows them whole too.
        inject_as: Configured chat role. When supplied it overrides the spec's
            default user-message injection site.
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
        """Manifest the action's spec into the Intervention the Editor will receive.

        Follows POSIT's manifestation step. Three cases, in order:

        1. The action carries no ``InterventionSpec`` (reflection level 1, tool
           and target only): there is no semantic action to realize, so nothing
           is manifested and no LM call is made.
        2. The spec has ``fixed_text``: non-empty text is returned verbatim,
           tagged with the spec's ``inject_as`` site, with no LM call.
        3. The spec has an ``instruction``: up to two LM calls are made over
           :data:`MANIFESTOR_PROMPT`, whose state block is the whole candidate
           (``full_text``), the region the edit will land on (``region_text``),
           the minibatch ``feedback_summary`` and the execution ``traces``
           (head-truncated to ``max_traces_chars`` unless that is ``None``).
           The prompt tells the LM where its text will land (in the voice
           matching ``inject_as``), forbids writing the edit itself, and asks
           it to carry out the instruction precisely, grounded in that state.
           The reply is stripped of think tags and surrounding whitespace and
           capped at :data:`MAX_INTERVENTION_CHARS` (with a ``...`` marker) so
           it steers without ballooning the Editor's prompt. An empty or
           think-only first reply is retried once; a second empty reply raises
           :class:`ManifestationError` so the action can be dropped explicitly.

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
            The realized :class:`Intervention` (steering text plus its
            ``inject_as`` site), or ``None`` when ``action.intervention_spec``
            is ``None``.

        Raises:
            ManifestationError: A fixed intervention is blank or both
                instruction-manifestation attempts are empty after hidden
                thoughts are removed, or the action precondition is not met.
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
            tool=action.edit_tool.value if action.edit_tool is not None else "available-tool-basis",
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
