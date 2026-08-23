# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""RLM Proposer role of the 3-role reflection architecture.

The Proposer is a Recursive Language Model (RLM) agent with externalized
context, a persistent guarded in-process Python environment, leaf LM delegation,
optional recursive RLM delegation, and deterministic edit commitment. Its
in-process executor is a guardrail for trusted model output, not a security
boundary for hostile code.

The context (the region to edit, the whole component, the failure feedback and
the execution traces) is *not* in the model's prompt: it lives in an
:class:`~gepa.proposer.reflective_mutation.rlm_environment.RLMEnvironment` as
Python variables. Each turn the model either runs one ``<python>`` block against
that environment (slicing, searching and summarizing the context, keeping
intermediate state between turns, delegating with ``llm_query`` /
``rlm_query``) or commits its edit. The prompt only says what variables exist,
which edit operation is required, what the planner advised, and how to
terminate.

The loop is *forced*: the only terminating action is an ``<edit>`` block, and
the harness always applies the Controller-selected
:class:`~gepa.strategies.edit_tools.EditTool` to the Controller-selected
:class:`~gepa.strategies.document_template.EditTarget` (spec step 5). Python
never mutates the candidate; the edit is parsed and validated here, applied by
:func:`~gepa.strategies.edit_tools.apply_edit` (whose atomic-op log becomes the
step-8 ``executed_edit`` record), and the result must still parse as a
canonical document and fit the size budget, or it is dropped.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from gepa.proposer.reflective_mutation.base import LanguageModel
from gepa.proposer.reflective_mutation.react_v2_proposer import ReActV2ContextError
from gepa.proposer.reflective_mutation.rlm_environment import (
    ALLOWED_MODULES,
    CHILD_RLM_PROMPT,
    LAST_TURN_NOTE,
    RLMBudget,
    RLMEnvironment,
    RLMProtocolError,
    RLMStep,
    parse_action,
    rlm_environment_contract,
)
from gepa.strategies.document_template import DocumentTemplate, EditTarget, MalformedDocumentError
from gepa.strategies.edit_tools import (
    DeleteTextArgs,
    EditApplicationError,
    EditArgs,
    EditTool,
    InsertTextArgs,
    MoveTextArgs,
    Placement,
    ReplaceTextArgs,
    apply_edit,
)
from gepa.utils.text import strip_think_tags

_EDIT_SCHEMAS: dict[EditTool, str] = {
    EditTool.INSERT_TEXT: (
        "<edit>\n"
        "  <anchor>existing text to anchor on, or empty to append at the end</anchor>\n"
        "  <where>before|after</where>\n"
        "  <text>the new text to insert</text>\n"
        "</edit>"
    ),
    EditTool.DELETE_TEXT: ("<edit>\n  <target>the exact existing text to remove</target>\n</edit>"),
    EditTool.REPLACE_TEXT: (
        "<edit>\n  <target>the exact existing text to replace</target>\n  <text>the replacement text</text>\n</edit>"
    ),
    EditTool.MOVE_TEXT: (
        "<edit>\n"
        "  <target>the exact existing text to move</target>\n"
        "  <anchor>existing text marking the destination</anchor>\n"
        "  <where>before|after</where>\n"
        "</edit>"
    ),
}

MAX_BRANCH_HISTORY_CHARS = 12_000


class RLMContextError(ReActV2ContextError):
    """Raise a fatal branch-history error compatible with GEPA's proposal boundary."""


def _validated_branch_history(history: Sequence[Mapping[str, Any]]) -> str:
    """Validate and serialize one branch's user/assistant transcript.

    Args:
        history: Messages retained by the selected parent candidate.

    Returns:
        Stable JSON exposed to the RLM as its read-only ``history`` variable.

    Raises:
        TypeError: An entry or its content has the wrong type.
        ValueError: An entry has extra fields or an unsupported role.
        RLMContextError: The serialized transcript exceeds the fixed budget.
    """
    messages: list[dict[str, str]] = []
    for message in history:
        if not isinstance(message, Mapping):
            raise TypeError("Every branch-local history entry must be a chat-message mapping.")
        if set(message) != {"role", "content"}:
            raise ValueError("Every branch-local history entry must contain only 'role' and 'content'.")
        role = message["role"]
        content = message["content"]
        if role not in {"user", "assistant"}:
            raise ValueError("Branch-local history roles must be 'user' or 'assistant'.")
        if not isinstance(content, str):
            raise TypeError("Branch-local history message content must be a string.")
        messages.append({"role": role, "content": content})
    rendered = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    if len(rendered) > MAX_BRANCH_HISTORY_CHARS:
        raise RLMContextError(
            f"Branch-local user/assistant history is {len(rendered)} characters, exceeding the "
            f"{MAX_BRANCH_HISTORY_CHARS}-character RLM history budget. Global history and automatic compression "
            "are intentionally disabled."
        )
    return rendered


RLM_PROPOSER_PROMPT = """\
You are the Editor. Make ONE precise edit to a region of a {kind} document to fix the failures it caused.

## Required operation: {tool} on region '{region}'
You MUST finish by emitting exactly one <edit> block using this operation. The edit is applied verbatim, so \
anchors/targets must be copied exactly from `region`. The document's `## <Section>` headers are fixed \
structure: never add, remove, or rename one.

## Guidance from the planner
{intervention}

## Your workspace
The context is NOT in this prompt. It lives in a persistent Python environment as read-only variables:
{variables}
Run code with a <python>...</python> block (one per turn); whatever you print() is returned to you next turn. \
Variables you assign persist across turns, so you can build up intermediate results (e.g. the failures you \
found, the sentences you want to change) without re-reading. Available: plain Python, {modules}, and
{tools}
This is guarded in-process execution for trusted model output, not security isolation. Ordinary file, network and \
OS interfaces are not exposed. Rebinding `region` or `component` in Python has no effect on the real document \
(they are restored each turn): only your final <edit> changes it. You have {turns} turns in total.

## How to answer each turn
Reply with exactly ONE action: a single <python>...</python> block, or, when ready, the final <edit> block:
{edit_schema}
"""

RLM_PROPOSER_PROTOCOL_VERSION = 1


def rlm_protocol_contract() -> dict[str, Any]:
    """Return the behavior-bearing RLM prompt and protocol identity."""
    return {
        "version": RLM_PROPOSER_PROTOCOL_VERSION,
        "environment": rlm_environment_contract(),
        "root_prompt_template": RLM_PROPOSER_PROMPT,
        "child_prompt_template": CHILD_RLM_PROMPT,
        "last_turn_note": LAST_TURN_NOTE,
        "edit_schemas": {tool.value: schema for tool, schema in _EDIT_SCHEMAS.items()},
        "guarded_executor_allowed_modules": list(ALLOWED_MODULES),
        "branch_history_max_chars": MAX_BRANCH_HISTORY_CHARS,
        "context_transport": "read_only_python_variables",
        "history_transport": "read_only_json_user_assistant_messages",
    }


@dataclass
class RLMResult:
    """Outcome of one RLM proposal.

    ``repl_calls``, ``llm_queries`` and ``rlm_queries`` are totals over the
    root and every child RLM it spawned; ``steps`` is the root's turn log
    (child turns hang off each step's ``child_calls``).

    Attributes:
        new_text: The edited component text, or the untouched parent text when
            the edit was dropped.
        changed: Whether ``new_text`` differs from the parent text.
        executed_edit: The atomic-operation log of the applied edit (empty when
            nothing was applied).
        iterations: Root turns consumed, including the terminating one.
        repl_calls: ``<python>`` executions across the whole recursion tree.
        llm_queries: Leaf ``llm_query`` calls across the whole tree.
        rlm_queries: Child RLMs spawned across the whole tree.
        dropped_reason: Why no edit was applied (over budget, no valid edit in
            time); ``None`` when an edit was applied.
        final_output: The model's last raw reply.
        steps: The root's turn log.
        chat_messages: Actual root assistant turns and user observations, used
            as the next revision's branch-local history.
    """

    new_text: str
    changed: bool
    executed_edit: list[str] = field(default_factory=list)
    iterations: int = 0
    repl_calls: int = 0
    llm_queries: int = 0
    rlm_queries: int = 0
    dropped_reason: str | None = None
    final_output: str = ""
    steps: list[RLMStep] = field(default_factory=list)
    chat_messages: list[dict[str, str]] = field(default_factory=list)


def _extract_field(block: str, tag: str, *, strip: bool = True) -> str | None:
    """Read one named child field (``<tag>value</tag>``) out of an ``<edit>`` block.

    Args:
        block: Inner text of the ``<edit>`` block.
        tag: Child tag name to read, without angle brackets.
        strip: Trim surrounding whitespace from the value. Pass ``False`` for
            payloads that are applied verbatim (e.g. inserted text) so leading
            newlines and indentation survive.

    Returns:
        The field's text, or ``None`` when the child tag is absent (an empty
        ``<tag></tag>`` yields ``""``).
    """
    m = re.search(rf"(?is)<{tag}\s*>(.*?)</{tag}>", block)
    if m is None:
        return None
    return m.group(1).strip() if strip else m.group(1)


def _required_field(block: str, tag: str, tool: EditTool, *, strip: bool = True, non_empty: bool = True) -> str:
    """Read a child field the tool's schema requires, rejecting a missing (or empty) one.

    Args:
        block: Inner text of the ``<edit>`` block.
        tag: Child tag name to read, without angle brackets.
        tool: The forced tool, named in the error message.
        strip: Trim surrounding whitespace from the value (see :func:`_extract_field`).
        non_empty: Also reject an empty value.

    Returns:
        The field's text.

    Raises:
        RLMProtocolError: The tag is absent, or empty while ``non_empty`` is set.
    """
    value = _extract_field(block, tag, strip=strip)
    if value is None:
        raise RLMProtocolError(f"{tool.value} requires a <{tag}> field in the <edit> block.")
    if non_empty and not value:
        raise RLMProtocolError(f"{tool.value} requires a non-empty <{tag}> field.")
    return value


def _placement(block: str, tool: EditTool) -> Placement:
    """Read ``<where>`` and require it to be exactly ``before`` or ``after``.

    Args:
        block: Inner text of the ``<edit>`` block.
        tool: The forced tool, named in the error message.

    Returns:
        ``"before"`` or ``"after"`` (case-insensitive match).

    Raises:
        RLMProtocolError: The field is missing or any other value; nothing is
            silently defaulted.
    """
    where = _required_field(block, "where", tool).lower()
    if where == "before":
        return "before"
    if where == "after":
        return "after"
    raise RLMProtocolError(f"Invalid <where> value {where!r}: must be 'before' or 'after'.")


class RLMProposer:
    """RLM Editor: reasons in a guarded in-process REPL, then commits one forced edit.

    The Proposer role of the three-role reflection stack. Instead of receiving
    the candidate, feedback and traces inline, the model gets them as variables
    of a persistent :class:`RLMEnvironment` and inspects them with Python (and
    delegated ``llm_query``/``rlm_query`` calls) turn by turn, then commits a
    single edit using the tool and region the Controller fixed. See
    :meth:`propose` for the loop and its invariants.

    Args:
        lm: The language model driving the Editor and its delegations.
        template: The document template of the components this proposer edits;
            used to split the candidate into regions and to re-validate the
            canonical format after every edit.
        budget: Turn, REPL, delegation and time limits; the :class:`RLMBudget`
            defaults when ``None``.
        logger: Optional run logger with a ``log(message)`` method for dropped
            edits.
    """

    def __init__(
        self,
        lm: LanguageModel,
        template: DocumentTemplate,
        *,
        budget: RLMBudget | None = None,
        logger: Any | None = None,
    ):
        """Store the LM, template, budget and logger; nothing runs until :meth:`propose`."""
        self.lm = lm
        self.template = template
        self.budget = budget if budget is not None else RLMBudget()
        self.logger = logger

    def _log(self, message: str) -> None:
        """Forward ``message`` to the bound run logger, if any."""
        if self.logger is not None:
            self.logger.log(message)

    def propose(
        self,
        component_text: str,
        edit_target: EditTarget,
        edit_tool: EditTool,
        intervention: str,
        feedback_summary: str,
        traces_text: str,
        max_chars: int | None,
        branch_history: Sequence[Mapping[str, Any]] = (),
    ) -> RLMResult:
        """Run the RLM loop and return the resulting component text and its trace.

        The context is loaded into a fresh :class:`RLMEnvironment` (``region``,
        ``component``, ``feedback``, ``traces``). For up to
        ``budget.max_root_iterations`` turns the model runs one ``<python>``
        block per turn (its output is fed back) until it emits a valid ``<edit>``
        for ``edit_tool`` on ``edit_target``. A section edit is applied to that
        section's exact body span without rewriting surrounding bytes; a whole-document
        edit is applied to the full text. Either way the result must still parse
        as a ``template`` document (an edit that injects or removes a ``## ``
        header is rejected and fed back to the model like a bad anchor), so the
        canonical format is an invariant of the run. An over-budget or
        unrealizable edit is dropped (``changed=False``), leaving the parent
        text unchanged so the proposal dies at the acceptance gate rather than
        corrupting the candidate.

        Args:
            component_text: Current text of the component (canonical format).
            edit_target: The Controller-selected region.
            edit_tool: The Controller-selected operation the edit must use.
            intervention: The Manifestor's steering guidance (may be empty).
            feedback_summary: Failure feedback, exposed as ``feedback``.
            traces_text: Execution traces, exposed as ``traces``.
            max_chars: Size budget of the edited component; ``None`` = unbounded.
            branch_history: User/assistant transcript from this parent branch
                only. It is exposed as read-only JSON in ``history``; no global
                or cross-branch history is constructed.

        Returns:
            An :class:`RLMResult` with the new component text (``changed=True``)
            or the untouched parent text plus a ``dropped_reason``
            (``changed=False``), together with the executed edit, the turn /
            REPL / delegation counts and the per-turn log.

        Raises:
            MalformedDocumentError: ``component_text`` itself is not in the
                template's canonical format (a caller bug: seeds are validated
                at the front door and every accepted edit re-validated here).
        """
        self.template.parse(component_text)
        section = edit_target.section
        if section is None:
            body_start, body_end = 0, len(component_text)
        else:
            body_start, body_end = self.template.section_body_span(component_text, section)
        region_text = component_text[body_start:body_end]
        history_text = _validated_branch_history(branch_history)
        env = RLMEnvironment(
            {
                "region": region_text,
                "component": component_text,
                "feedback": feedback_summary,
                "traces": traces_text,
                "history": history_text,
            },
            self.lm,
            self.budget,
        )
        variables = "\n".join(
            f"- {name}: str, {len(value):,} chars — {what}"
            for name, value, what in (
                ("region", region_text, f"the text of region '{edit_target.name}', which your edit modifies"),
                ("component", component_text, "the whole document the region belongs to"),
                ("feedback", feedback_summary, "the failure feedback from the recent evaluations"),
                ("traces", traces_text, "the execution traces (inputs, outputs, feedback) of those evaluations"),
                (
                    "history",
                    history_text,
                    "JSON user/assistant messages from this candidate branch only; no global history",
                ),
            )
        )
        turns = self.budget.max_root_iterations
        base_prompt = RLM_PROPOSER_PROMPT.format(
            kind=self.template.kind,
            tool=edit_tool.value,
            region=edit_target.name,
            intervention=intervention or "(no additional guidance)",
            variables=variables,
            modules=", ".join(ALLOWED_MODULES),
            tools=env.tools_help(),
            turns=turns,
            edit_schema=_EDIT_SCHEMAS[edit_tool],
        )
        transcript = ""
        steps: list[RLMStep] = []
        last_error = ""
        final_output = ""
        chat_messages: list[dict[str, str]] = []

        for iteration in range(1, turns + 1):
            note = LAST_TURN_NOTE if iteration == turns else ""
            raw = strip_think_tags(self.lm(base_prompt + transcript + note))
            final_output = raw
            chat_messages.append({"role": "assistant", "content": raw})
            try:
                action, payload = parse_action(raw, "edit")
            except RLMProtocolError as exc:
                last_error = str(exc)
                steps.append(RLMStep(iteration, "invalid", error=last_error))
                chat_messages.append({"role": "user", "content": last_error})
                transcript += f"\n\n<your-output>{raw}</your-output>\n<error>{last_error}</error>\n"
                continue

            if action == "python":
                execution = env.execute(payload)
                steps.append(RLMStep(iteration, "python", payload, execution.stdout, execution.error, execution.calls))
                chat_messages.append({"role": "user", "content": execution.render(self.budget.max_output_chars)})
                transcript += (
                    f"\n\n<your-output>{raw}</your-output>\n{execution.render(self.budget.max_output_chars)}\n"
                )
                continue

            try:
                args = self._parse_edit_args(payload, edit_tool)
                new_region, ops = apply_edit(region_text, args)
                if section is None:
                    new_component = new_region
                else:
                    replacement = new_region
                    if body_start == body_end and replacement:
                        if body_start == 0 or component_text[body_start - 1] not in "\r\n":
                            replacement = "\n" + replacement
                        if body_end < len(component_text) and component_text[body_end:].startswith("## "):
                            replacement += "\n"
                    new_component = component_text[:body_start] + replacement + component_text[body_end:]
                self.template.parse(new_component)
                if new_component == component_text:
                    raise EditApplicationError(
                        f"{edit_tool.value} produced no text change; choose distinct replacement text or placement."
                    )
                if max_chars is not None and len(new_component) > max_chars:
                    raise EditApplicationError(
                        f"Edited component is {len(new_component)} characters, exceeding max_chars={max_chars}."
                    )
            except (RLMProtocolError, EditApplicationError, MalformedDocumentError) as exc:
                last_error = str(exc)
                steps.append(RLMStep(iteration, "edit", error=last_error))
                chat_messages.append({"role": "user", "content": f"{last_error} Try again."})
                transcript += f"\n\n<your-output>{raw}</your-output>\n<error>{last_error} Try again.</error>\n"
                continue

            steps.append(RLMStep(iteration, "edit"))
            chat_messages.append(
                {
                    "role": "user",
                    "content": f"OK: {edit_tool.value} applied.\nExecuted edit:\n" + "\n".join(ops),
                }
            )
            return self._result(
                new_component,
                new_component != component_text,
                env,
                steps,
                iteration,
                final_output,
                chat_messages,
                executed_edit=ops,
            )

        reason = f"no_valid_edit after {turns} iterations"
        if last_error:
            reason += f" (last error: {last_error})"
        self._log(f"RLMProposer dropped edit on '{edit_target.label}': {reason}")
        return self._result(
            component_text,
            False,
            env,
            steps,
            turns,
            final_output,
            chat_messages,
            dropped_reason=reason,
        )

    @staticmethod
    def _result(
        new_text: str,
        changed: bool,
        env: RLMEnvironment,
        steps: list[RLMStep],
        iterations: int,
        final_output: str,
        chat_messages: list[dict[str, str]],
        *,
        executed_edit: list[str] | None = None,
        dropped_reason: str | None = None,
    ) -> RLMResult:
        """Assemble the result, pulling the tree-wide usage counters out of the environment.

        Args:
            new_text: The component text to return (edited or untouched).
            changed: Whether ``new_text`` differs from the parent text.
            env: The root environment; its ``usage`` counters are copied out.
            steps: The root's turn log.
            iterations: Root turns consumed.
            final_output: The model's last raw reply.
            chat_messages: Root assistant turns and user observations.
            executed_edit: Atomic-operation log of the applied edit, if any.
            dropped_reason: Why no edit was applied, if none was.

        Returns:
            The populated :class:`RLMResult`.
        """
        return RLMResult(
            new_text=new_text,
            changed=changed,
            executed_edit=executed_edit or [],
            iterations=iterations,
            repl_calls=env.usage.repl_calls,
            llm_queries=env.usage.llm_queries,
            rlm_queries=env.usage.rlm_queries,
            dropped_reason=dropped_reason,
            final_output=final_output,
            steps=steps,
            chat_messages=chat_messages,
        )

    @staticmethod
    def _parse_edit_args(block: str, tool: EditTool) -> EditArgs:
        """Turn the model's ``<edit>`` block into the typed arguments of the forced tool.

        The Controller has already fixed which :class:`EditTool` this turn must
        use, so the block is read against that tool's schema (see
        ``_EDIT_SCHEMAS``) rather than inferred from its contents. The fields
        each tool requires are validated here, at the protocol level:

        * INSERT_TEXT: non-empty ``<text>``, ``<where>``; ``<anchor>`` may be
          empty (append at the end).
        * DELETE_TEXT: non-empty ``<target>``.
        * REPLACE_TEXT: non-empty ``<target>`` and a ``<text>`` field (empty
          means "replace with nothing").
        * MOVE_TEXT: non-empty ``<target>`` and ``<anchor>``, ``<where>``.

        ``<where>`` must be exactly ``before`` or ``after``. Whether anchors and
        targets actually occur in the region is left to
        :func:`~gepa.strategies.edit_tools.apply_edit`.

        Args:
            block: Inner text of the ``<edit>...</edit>`` block the model emitted.
            tool: The tool the Controller forced for this proposal.

        Returns:
            The args object for ``tool``: :class:`InsertTextArgs`,
            :class:`DeleteTextArgs`, :class:`ReplaceTextArgs`, or
            :class:`MoveTextArgs`.

        Raises:
            RLMProtocolError: A required field is missing/empty or ``<where>``
                is invalid.
            EditApplicationError: ``tool`` is not one of the four known tools.
        """
        if tool == EditTool.INSERT_TEXT:
            return InsertTextArgs(
                # Preserve whitespace: the inserted payload is applied verbatim.
                text=_required_field(block, "text", tool, strip=False),
                anchor=_extract_field(block, "anchor") or "",
                where=_placement(block, tool),
            )
        if tool == EditTool.DELETE_TEXT:
            return DeleteTextArgs(target=_required_field(block, "target", tool))
        if tool == EditTool.REPLACE_TEXT:
            return ReplaceTextArgs(
                target=_required_field(block, "target", tool),
                text=_required_field(block, "text", tool, strip=False, non_empty=False),
            )
        if tool == EditTool.MOVE_TEXT:
            return MoveTextArgs(
                target=_required_field(block, "target", tool),
                anchor=_required_field(block, "anchor", tool),
                where=_placement(block, tool),
            )
        raise EditApplicationError(f"Unknown edit tool: {tool}")
