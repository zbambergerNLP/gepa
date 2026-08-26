# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Propose action-conditioned edits with an RLM workspace.

The Controller fixes the document section and edit tool. The model inspects the
section body, failures, traces, and branch history as read-only variables in a
persistent :class:`RLMEnvironment`. Each turn runs one ``<python>`` block or
returns one ``<edit>`` block. The proposer validates and applies that edit, then
checks the section template and size limit.
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
    RLM_ENVIRONMENT_CONTRACT,
    RLMBudget,
    RLMEnvironment,
    RLMProtocolError,
    RLMStep,
    parse_action,
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
        history: Messages retained by the selected parent branch.

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
Edit one region of a {kind} document to address the recorded failures.

## Required operation: {tool} on region '{region}'
Finish by emitting exactly one <edit> block using this operation. The edit is applied verbatim, so \
anchors/targets must be copied exactly from `region`. The template defines the allowed `## <Section>` \
headers and their order. Empty sections are absent until the harness gives them content. Edit body text \
only: never write, delete, rename, or move a header line yourself.

## Guidance from the planner
{steering_message}

## Your workspace
Read the context from these read-only variables in the persistent Python environment:
{variables}
Run code with a <python>...</python> block (one per turn); whatever you print() is returned to you next turn. \
Variables you assign persist across turns, so you can build up intermediate results (e.g. the failures you \
found, the sentences you want to change) without re-reading. Available: plain Python, {modules}, and
{tools}
The executor runs in process without security isolation and accepts only trusted model output. File, network, and \
OS interfaces are unavailable. Rebinding `region` cannot change the section; the final <edit> is \
the only mutation. You have {turns} turns in total.

## Turn protocol
Reply with exactly one action: a single <python>...</python> block, or, when ready, the final <edit> block:
{edit_schema}
"""

RLM_PROPOSER_PROTOCOL_VERSION = 1
RLM_PROTOCOL_CONTRACT: dict[str, Any] = {
    "version": RLM_PROPOSER_PROTOCOL_VERSION,
    "environment": RLM_ENVIRONMENT_CONTRACT,
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
        new_text: The edited selected-section body, or the untouched body when
            the edit was dropped.
        changed: Whether ``new_text`` differs from the parent section body.
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
    the section body, feedback and traces inline, the model gets them as variables
    of a persistent :class:`RLMEnvironment` and inspects them with Python (and
    delegated ``llm_query``/``rlm_query`` calls) turn by turn, then commits a
    single edit using the tool and section the Controller fixed. See
    :meth:`propose` for the loop and its invariants.

    Args:
        lm: The language model driving the Editor and its delegations.
        template: The document template of the section this proposer edits;
            used to re-validate the selected section after every edit.
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
        """Store the dependencies for a fresh environment per proposal.

        Args:
            lm: Model driving the root editor and delegated calls.
            template: Document template used to validate edited section bodies.
            budget: Tree-wide turn and execution limits, or ``None`` for the
                default matched budget.
            logger: Optional run logger with a ``log(message)`` method.
        """
        self.lm = lm
        self.template = template
        self.budget = budget if budget is not None else RLMBudget()
        self.logger = logger

    def propose(
        self,
        region_text: str,
        edit_target: EditTarget,
        edit_tool: EditTool,
        steering_message: str,
        feedback_summary: str,
        traces_text: str,
        max_chars: int | None,
        branch_history: Sequence[Mapping[str, Any]] = (),
    ) -> RLMResult:
        """Run the RLM loop and return the resulting selected-section body.

        The context is loaded into a fresh :class:`RLMEnvironment` (``region``,
        ``feedback``, ``traces``). For up to
        ``budget.max_root_iterations`` turns the model runs one ``<python>``
        block per turn (its output is fed back) until it emits a valid ``<edit>``
        for ``edit_tool`` on ``edit_target``. An over-budget or unrealizable
        edit is dropped (``changed=False``), leaving the section body unchanged.

        Args:
            region_text: Complete body of the Controller-selected section.
            edit_target: The Controller-selected region.
            edit_tool: The Controller-selected operation the edit must use.
            steering_message: The Manifestor's steering guidance (may be empty).
            feedback_summary: Failure feedback, exposed as ``feedback``.
            traces_text: Execution traces, exposed as ``traces``.
            max_chars: Size budget of the edited section body; ``None`` = unbounded.
            branch_history: User/assistant transcript from this parent branch
                only. It is exposed as read-only JSON in ``history``; no global
                or cross-branch history is constructed.

        Returns:
            An :class:`RLMResult` with the new section body (``changed=True``)
            or the untouched body plus a ``dropped_reason``
            (``changed=False``), together with the executed edit, the turn /
            REPL / delegation counts and the per-turn log.

        Raises:
            TypeError: A branch-history entry or its content has the wrong type.
            ValueError: A branch-history entry has extra fields or a non-chat
                role.
            RLMContextError: Serialized branch history exceeds its fixed budget.
        """
        section = edit_target.section
        history_text = _validated_branch_history(branch_history)
        env = RLMEnvironment(
            {
                "region": region_text,
                "feedback": feedback_summary,
                "traces": traces_text,
                "history": history_text,
            },
            self.lm,
            self.budget,
        )
        variables = "\n".join(
            f"- {name}: str, {len(value):,} chars; {what}"
            for name, value, what in (
                ("region", region_text, f"the text of section '{section}', which your edit modifies"),
                ("feedback", feedback_summary, "the failure feedback from the recent evaluations"),
                ("traces", traces_text, "the execution traces (inputs, outputs, feedback) of those evaluations"),
                (
                    "history",
                    history_text,
                    "JSON user/assistant messages from this parent branch only; no global history",
                ),
            )
        )
        turns = self.budget.max_root_iterations
        base_prompt = RLM_PROPOSER_PROMPT.format(
            kind=self.template.kind,
            tool=edit_tool.value,
            region=section,
            steering_message=steering_message or "(no additional guidance)",
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
            raw = self.lm(base_prompt + transcript + note)
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
                self.template.replace_section_body("", section, new_region)
                if new_region == region_text:
                    raise EditApplicationError(
                        f"{edit_tool.value} produced no text change; choose distinct replacement text or placement."
                    )
                if max_chars is not None and len(new_region) > max_chars:
                    raise EditApplicationError(
                        f"Edited section is {len(new_region)} characters, exceeding max_chars={max_chars}."
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
            return RLMResult(
                new_text=new_region,
                changed=new_region != region_text,
                executed_edit=ops,
                iterations=iteration,
                repl_calls=env.usage.repl_calls,
                llm_queries=env.usage.llm_queries,
                rlm_queries=env.usage.rlm_queries,
                final_output=final_output,
                steps=steps,
                chat_messages=chat_messages,
            )

        reason = f"no_valid_edit after {turns} iterations"
        if last_error:
            reason += f" (last error: {last_error})"
        if self.logger is not None:
            self.logger.log(f"RLMProposer dropped edit on '{edit_target.label}': {reason}")
        return RLMResult(
            new_text=region_text,
            changed=False,
            executed_edit=[],
            iterations=turns,
            repl_calls=env.usage.repl_calls,
            llm_queries=env.usage.llm_queries,
            rlm_queries=env.usage.rlm_queries,
            dropped_reason=reason,
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
