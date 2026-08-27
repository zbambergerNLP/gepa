# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Apply section-scoped prompt and skill revisions with ReAct V2.

The Controller selects a document region and, at level 2, a semantic action.
The proposer runs a bounded tool loop and returns each observation to the
model. The broad tool set applies an action directly; the minimal set
decomposes replace and move into delete and insert calls.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, cast

from gepa.lm import NativeToolCall, ToolCompletion
from gepa.proposer.reflective_mutation.base import LanguageModel
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

MAX_BRANCH_HISTORY_CHARS = 12_000
MAX_INITIAL_CONTEXT_CHARS = 64_000


class ReActV2ProtocolError(ValueError):
    """Raised when a ReAct V2 reply is not a valid tool or finish action."""


class ReActV2ContextError(ValueError):
    """Raised when branch-local history or the initial request exceeds its budget."""


@dataclass
class ReActV2Step:
    """One assistant action and the observation returned by the harness.

    Attributes:
        turn: One-based assistant turn number.
        assistant: Raw answer-only assistant reply.
        action: Tool name, ``"FINISH"``, or ``"INVALID"``.
        observation: Tool result or protocol error returned to the next turn.
        error: Error text for a rejected action, otherwise ``None``.
        executed_edit: Atomic insert/delete operations produced by a valid tool.
        region_text: Complete selected-section body after the step.
    """

    turn: int
    assistant: str
    action: str
    observation: str
    error: str | None
    executed_edit: list[str] = field(default_factory=list)
    region_text: str = ""


@dataclass
class ReActV2Result:
    """Outcome of one ReAct V2 proposal.

    Attributes:
        new_text: Edited section body, or the untouched body when the proposal is dropped.
        changed: Whether a completed proposal changed the component.
        executed_edit: Atomic insert/delete log across all successful tool calls.
        iterations: Assistant turns consumed.
        tool_calls: Valid tool calls applied.
        dropped_reason: Why no completed edit was returned, otherwise ``None``.
        final_output: Last raw assistant reply.
        steps: Full local tool/observation trajectory.
    """

    new_text: str
    changed: bool
    executed_edit: list[str] = field(default_factory=list)
    iterations: int = 0
    tool_calls: int = 0
    dropped_reason: str | None = None
    final_output: str = ""
    steps: list[ReActV2Step] = field(default_factory=list)

_TOOL_SCHEMAS: dict[EditTool, str] = {
    EditTool.INSERT_TEXT: (
        "<tool_call>\n"
        "  <tool>INSERT_TEXT</tool>\n"
        "  <anchor>exact existing text, or empty to append</anchor>\n"
        "  <where>before|after</where>\n"
        "  <text>new text inserted verbatim</text>\n"
        "</tool_call>"
    ),
    EditTool.DELETE_TEXT: (
        "<tool_call>\n  <tool>DELETE_TEXT</tool>\n  <target>exact existing text</target>\n</tool_call>"
    ),
    EditTool.REPLACE_TEXT: (
        "<tool_call>\n"
        "  <tool>REPLACE_TEXT</tool>\n"
        "  <target>exact existing text</target>\n"
        "  <text>replacement text applied verbatim</text>\n"
        "</tool_call>"
    ),
    EditTool.MOVE_TEXT: (
        "<tool_call>\n"
        "  <tool>MOVE_TEXT</tool>\n"
        "  <target>exact existing text to move</target>\n"
        "  <anchor>exact existing destination text</anchor>\n"
        "  <where>before|after</where>\n"
        "</tool_call>"
    ),
}

_NATIVE_TOOL_PARAMETERS: dict[EditTool, dict[str, Any]] = {
    EditTool.INSERT_TEXT: {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "New text inserted verbatim."},
            "anchor": {
                "type": "string",
                "description": "Exact existing text, or an empty string to append.",
            },
            "where": {"type": "string", "enum": ["before", "after"]},
        },
        "required": ["text", "anchor", "where"],
        "additionalProperties": False,
    },
    EditTool.DELETE_TEXT: {
        "type": "object",
        "properties": {"target": {"type": "string", "description": "Exact existing text to delete."}},
        "required": ["target"],
        "additionalProperties": False,
    },
    EditTool.REPLACE_TEXT: {
        "type": "object",
        "properties": {
            "target": {"type": "string", "description": "Exact existing text to replace."},
            "text": {"type": "string", "description": "Replacement text applied verbatim."},
        },
        "required": ["target", "text"],
        "additionalProperties": False,
    },
    EditTool.MOVE_TEXT: {
        "type": "object",
        "properties": {
            "target": {"type": "string", "description": "Exact existing text to move."},
            "anchor": {"type": "string", "description": "Exact existing destination text."},
            "where": {"type": "string", "enum": ["before", "after"]},
        },
        "required": ["target", "anchor", "where"],
        "additionalProperties": False,
    },
}

_NATIVE_TOOL_DESCRIPTIONS: dict[EditTool, str] = {
    EditTool.INSERT_TEXT: "Insert new text before or after an exact anchor in the selected region.",
    EditTool.DELETE_TEXT: "Delete the first occurrence of exact text from the selected region.",
    EditTool.REPLACE_TEXT: "Replace the first occurrence of exact text in the selected region.",
    EditTool.MOVE_TEXT: "Move exact text before or after an exact anchor in the selected region.",
}

REACT_V2_SYSTEM_PROMPT = """\
Revise only the selected section body of this structured {kind} document.

On every turn, emit exactly one action: {action_protocol}, or <finish>briefly state why the
revision is complete</finish>. Never emit both. Tool arguments are literal: copy targets and
anchors exactly from the latest region in the most recent observation.
The harness applies a call and returns an observation; use that observation before acting again.
Invalid calls do not change the document and return an error you must correct.

The harness owns the surrounding document and its headers. You receive only one section body. Never write
a `## <Section>` header; every target and anchor must come from the selected body.

Available tools:
{tool_schemas}

{completion_rule}
"""

REACT_V2_TASK_PROMPT = """\
## Selected component and region
Component: {component}
Region: {region}

## Current selected section body
{region_text}

## Failure feedback
{feedback}

## Execution traces
{traces}

## Branch-local conversation
Earlier assistant edit attempts and user tool/optimizer feedback from this candidate branch are replayed as
chat messages before this task. No global history is available.

Make the smallest revision that addresses the evidence. Begin with one action.
"""

_TOOL_CALL_BLOCK_RE = re.compile(r"(?is)<tool_call(?:\s[^>]*)?>(.*?)</tool_call>")
_FINISH_BLOCK_RE = re.compile(r"(?is)<finish(?:\s[^>]*)?>(.*?)</finish>")


def _lowered_semantic_tool(preferred_tool: EditTool | None, allowed_tools: Sequence[EditTool]) -> EditTool | None:
    """Identify a hidden semantic operator reproducible by the atomic basis.

    Args:
        preferred_tool: Direct operator coupled to the selected semantic action.
        allowed_tools: Operators exposed to the proposer.

    Returns:
        ``REPLACE_TEXT`` or ``MOVE_TEXT`` when that preferred operator is
        hidden but insert and delete can reproduce it, otherwise ``None``.
    """
    if preferred_tool not in {EditTool.REPLACE_TEXT, EditTool.MOVE_TEXT}:
        return None
    available = set(allowed_tools)
    if preferred_tool in available:
        return None
    required_basis = {EditTool.INSERT_TEXT, EditTool.DELETE_TEXT}
    return preferred_tool if required_basis.issubset(available) else None


def _extract_field(block: str, tag: str, *, strip: bool = True) -> str | None:
    """Extract a child field from a tool-call block.

    Args:
        block: Inner tool-call content.
        tag: Child field name.
        strip: Whether to trim surrounding whitespace.

    Returns:
        Field content, or ``None`` when absent.
    """
    match = re.search(rf"(?is)<{tag}\s*>(.*?)</{tag}>", block)
    if match is None:
        return None
    return match.group(1).strip() if strip else match.group(1)


def _required_field(
    block: str,
    tag: str,
    tool: EditTool,
    *,
    strip: bool = True,
    allow_empty: bool = False,
) -> str:
    """Read a required tool argument with a precise protocol error.

    Args:
        block: Inner tool-call content.
        tag: Required child field.
        tool: Tool whose schema is being parsed.
        strip: Whether to trim surrounding whitespace.
        allow_empty: Whether an explicitly empty field is valid.

    Returns:
        Parsed field text.

    Raises:
        ReActV2ProtocolError: The field is missing or disallowed empty text.
    """
    value = _extract_field(block, tag, strip=strip)
    if value is None:
        raise ReActV2ProtocolError(f"{tool.value} requires a <{tag}> field.")
    if not allow_empty and not value:
        raise ReActV2ProtocolError(f"{tool.value} requires a non-empty <{tag}> field.")
    return value


def _placement(block: str, tool: EditTool) -> Placement:
    """Parse a before/after placement field.

    Args:
        block: Inner tool-call content.
        tool: Tool whose placement is being parsed.

    Returns:
        ``"before"`` or ``"after"``.

    Raises:
        ReActV2ProtocolError: The value is not a supported placement.
    """
    where = _required_field(block, "where", tool).lower()
    if where not in {"before", "after"}:
        raise ReActV2ProtocolError(f"{tool.value} <where> must be 'before' or 'after'; got {where!r}.")
    return cast(Placement, where)


def parse_tool_call(reply: str) -> tuple[EditTool, EditArgs]:
    """Parse one ReAct V2 ``<tool_call>`` reply into typed edit arguments.

    Args:
        reply: Assistant reply containing one tool-call block.

    Returns:
        Selected tool and its typed arguments.

    Raises:
        ReActV2ProtocolError: The block, tool name, or required arguments are invalid.
    """
    blocks = _TOOL_CALL_BLOCK_RE.findall(reply)
    if len(blocks) != 1:
        raise ReActV2ProtocolError("Reply must contain exactly one <tool_call> or <finish> block.")
    if _FINISH_BLOCK_RE.findall(reply):
        raise ReActV2ProtocolError("Reply contained both <tool_call> and <finish>; emit exactly one action.")
    block = blocks[0]
    tool_name = _extract_field(block, "tool")
    if tool_name is None:
        raise ReActV2ProtocolError("<tool_call> requires a non-empty <tool> field.")
    try:
        tool = EditTool(tool_name.upper())
    except ValueError as exc:
        raise ReActV2ProtocolError(f"Unknown edit tool {tool_name!r}.") from exc

    if tool is EditTool.INSERT_TEXT:
        args: EditArgs = InsertTextArgs(
            text=_required_field(block, "text", tool, strip=False),
            anchor=_required_field(block, "anchor", tool, allow_empty=True),
            where=_placement(block, tool),
        )
    elif tool is EditTool.DELETE_TEXT:
        args = DeleteTextArgs(target=_required_field(block, "target", tool))
    elif tool is EditTool.REPLACE_TEXT:
        args = ReplaceTextArgs(
            target=_required_field(block, "target", tool),
            text=_required_field(block, "text", tool, strip=False, allow_empty=True),
        )
    else:
        args = MoveTextArgs(
            target=_required_field(block, "target", tool),
            anchor=_required_field(block, "anchor", tool),
            where=_placement(block, tool),
        )
    return tool, args


def parse_native_tool_call(call: NativeToolCall) -> tuple[EditTool, EditArgs]:
    """Parse one provider-native function call into typed edit arguments.

    Args:
        call: Normalized function name, JSON arguments, and provider call ID.

    Returns:
        Selected edit tool and its validated typed arguments.

    Raises:
        ReActV2ProtocolError: The call ID, function name, JSON object, field
            names, field types, or placement value is invalid.
    """
    if not call.id:
        raise ReActV2ProtocolError("Provider-native tool call is missing its call id.")
    try:
        tool = EditTool(call.name.upper())
    except ValueError as exc:
        raise ReActV2ProtocolError(f"Unknown edit tool {call.name!r}.") from exc
    try:
        payload = json.loads(call.arguments)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ReActV2ProtocolError(f"{tool.value} arguments must be one JSON object.") from exc
    if not isinstance(payload, dict):
        raise ReActV2ProtocolError(f"{tool.value} arguments must be one JSON object.")

    expected = set(_NATIVE_TOOL_PARAMETERS[tool]["properties"])
    extra = set(payload) - expected
    if extra:
        raise ReActV2ProtocolError(f"{tool.value} received unknown argument(s): {', '.join(sorted(extra))}.")

    def string_field(name: str, *, allow_empty: bool = False) -> str:
        """Read and validate one string argument from the native payload.

        Args:
            name: Argument name to read.
            allow_empty: Whether an empty string is valid for this field.

        Returns:
            The validated string value.

        Raises:
            ReActV2ProtocolError: The field is absent, is not a string, or is
                empty when emptiness is disallowed.
        """
        value = payload.get(name)
        if not isinstance(value, str):
            raise ReActV2ProtocolError(f"{tool.value} requires a string '{name}' argument.")
        if not allow_empty and not value:
            raise ReActV2ProtocolError(f"{tool.value} requires a non-empty '{name}' argument.")
        return value

    def native_placement() -> Placement:
        """Read the native call's placement argument.

        Returns:
            The validated ``"before"`` or ``"after"`` placement.

        Raises:
            ReActV2ProtocolError: The placement is missing or unsupported.
        """
        where = string_field("where").lower()
        if where not in {"before", "after"}:
            raise ReActV2ProtocolError(f"{tool.value} 'where' must be 'before' or 'after'; got {where!r}.")
        return cast(Placement, where)

    if tool is EditTool.INSERT_TEXT:
        args: EditArgs = InsertTextArgs(
            text=string_field("text"),
            anchor=string_field("anchor", allow_empty=True),
            where=native_placement(),
        )
    elif tool is EditTool.DELETE_TEXT:
        args = DeleteTextArgs(target=string_field("target"))
    elif tool is EditTool.REPLACE_TEXT:
        args = ReplaceTextArgs(
            target=string_field("target"),
            text=string_field("text", allow_empty=True),
        )
    else:
        args = MoveTextArgs(
            target=string_field("target"),
            anchor=string_field("anchor"),
            where=native_placement(),
        )
    return tool, args


def _validated_branch_history(
    history: Sequence[Mapping[str, Any]],
    max_chars: int,
) -> list[dict[str, str]]:
    """Validate one branch's user/assistant transcript without truncation.

    Args:
        history: User/assistant messages for the selected parent candidate.
        max_chars: Maximum serialized history length.

    Returns:
        Copied provider-ready chat messages.

    Raises:
        TypeError: A message is not a mapping or its content is not a string.
        ValueError: A message contains extra fields or a role other than user/assistant.
        ReActV2ContextError: The history exceeds the configured budget.
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
    rendered = json.dumps(messages, ensure_ascii=False)
    if len(rendered) > max_chars:
        raise ReActV2ContextError(
            f"Branch-local user/assistant history is {len(rendered)} characters, exceeding the {max_chars}-character "
            "ReAct V2 history budget. Global history and automatic compression are intentionally disabled."
        )
    return messages


class ReActV2Proposer:
    """Run a bounded ReAct tool loop over one selected document region.

    Args:
        lm: Model driving the ReAct conversation.
        template: Canonical document template.
        allowed_tools: Tools exposed by the configured edit basis.
        max_iterations: Maximum assistant turns, including invalid retries and finish.
        max_tool_calls: Maximum valid calls in an atomic-basis proposal.
        max_history_chars: Branch-local chat-history budget. Overflow raises instead of
            silently introducing global memory or lossy compression.
        max_initial_context_chars: Total serialized message-and-tool budget before
            every provider call. The historical argument name is retained for
            compatibility.
        logger: Optional run logger.
    """

    def __init__(
        self,
        lm: LanguageModel,
        template: DocumentTemplate,
        allowed_tools: Sequence[EditTool],
        *,
        max_iterations: int = 8,
        max_tool_calls: int = 4,
        max_history_chars: int = MAX_BRANCH_HISTORY_CHARS,
        max_initial_context_chars: int = MAX_INITIAL_CONTEXT_CHARS,
        logger: Any | None = None,
    ):
        """Validate and store the bounded ReAct proposer configuration.

        Args:
            lm: Model driving the ReAct conversation.
            template: Canonical document template for section validation.
            allowed_tools: Non-empty edit-tool basis exposed to the proposer.
            max_iterations: Maximum assistant turns, including invalid retries.
            max_tool_calls: Maximum valid calls in one proposal.
            max_history_chars: Maximum serialized branch-history length.
            max_initial_context_chars: Maximum serialized messages and tool
                schemas before each provider call.
            logger: Optional run logger.

        Raises:
            ValueError: The tool basis is empty or any configured limit is
                below one.
        """
        if not allowed_tools:
            raise ValueError("ReAct V2 requires at least one allowed edit tool.")
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least 1.")
        if max_initial_context_chars < 1:
            raise ValueError("max_initial_context_chars must be at least 1.")
        self.lm = lm
        self.template = template
        self.allowed_tools = tuple(allowed_tools)
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.max_history_chars = max_history_chars
        self.max_initial_context_chars = max_initial_context_chars
        self.logger = logger

    def _initial_messages(
        self,
        region_text: str,
        edit_target: EditTarget,
        preferred_tool: EditTool | None,
        steering_message: str | None,
        feedback_summary: str,
        traces_text: str,
        branch_history: Sequence[Mapping[str, Any]],
        *,
        native_tools: bool,
    ) -> list[dict[str, Any]]:
        """Build the conversation prefix with steering in the user task.

        Args:
            region_text: Complete body of the Controller-selected section.
            edit_target: Controller-selected region.
            preferred_tool: Tool directly coupled to a semantic action, if any.
            steering_message: Manifested semantic steering text.
            feedback_summary: Minibatch failure feedback.
            traces_text: Flattened execution traces.
            branch_history: User/assistant transcript along this parent branch.
            native_tools: Whether the LM exposes provider-native tool completion.

        Returns:
            Chat messages ready for the first ReAct turn.
        """
        direct_tool = preferred_tool if preferred_tool is not None and preferred_tool in self.allowed_tools else None
        lowered_tool = _lowered_semantic_tool(preferred_tool, self.allowed_tools)
        if direct_tool is not None:
            completion_rule = (
                f"This semantic action is coupled to {direct_tool.value}. Make exactly one valid "
                f"{direct_tool.value} call; that call completes the proposal automatically."
            )
        elif lowered_tool is EditTool.REPLACE_TEXT:
            completion_rule = (
                "This semantic action is coupled to REPLACE_TEXT, which is hidden by the atomic basis. "
                "Reproduce it with exactly one DELETE_TEXT call followed by one INSERT_TEXT call at the "
                "deleted target's original location, then emit <finish>."
            )
        elif lowered_tool is EditTool.MOVE_TEXT:
            completion_rule = (
                "This semantic action is coupled to MOVE_TEXT, which is hidden by the atomic basis. "
                "Reproduce it with exactly one DELETE_TEXT call followed by one INSERT_TEXT call that "
                "reinserts the exact deleted bytes at a distinct valid anchor, then emit <finish>."
            )
        elif preferred_tool is not None:
            completion_rule = (
                f"The semantic action's direct tool is {preferred_tool.value}, which is intentionally absent from "
                "this atomic basis. Compose the available tools, then emit <finish> only after the revision is complete."
            )
        else:
            completion_rule = "Compose the available tools as needed, then emit <finish> when the revision is complete."
        if native_tools:
            action_protocol = "one provider-native function call"
            tool_schemas = "\n".join(
                f"- {tool.value}: {_NATIVE_TOOL_DESCRIPTIONS[tool]}" for tool in self.allowed_tools
            )
        else:
            action_protocol = (
                "one <tool_call> block using the compatibility text schema below "
                "(the configured callable has no native-tool interface)"
            )
            tool_schemas = "\n\n".join(_TOOL_SCHEMAS[tool] for tool in self.allowed_tools)
        system = REACT_V2_SYSTEM_PROMPT.format(
            kind=self.template.kind,
            action_protocol=action_protocol,
            tool_schemas=tool_schemas,
            completion_rule=completion_rule,
        )
        task = REACT_V2_TASK_PROMPT.format(
            component=edit_target.component_name,
            region=edit_target.section,
            region_text=region_text,
            feedback=feedback_summary,
            traces=traces_text,
        )
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        messages.extend(_validated_branch_history(branch_history, self.max_history_chars))
        if steering_message:
            task = f"{steering_message}\n\n{task}"
        messages.append({"role": "user", "content": task})
        return messages

    def _check_context(
        self,
        messages: Sequence[Mapping[str, Any]],
        native_tools: Sequence[Mapping[str, Any]],
    ) -> None:
        """Reject an oversized conversation before making a provider call.

        Args:
            messages: Conversation accumulated for the next model turn.
            native_tools: Native tool schemas included with that turn.

        Raises:
            ReActV2ContextError: The serialized request exceeds the configured
                total-context budget.
        """
        payload: dict[str, Any] = {"messages": list(messages)}
        if native_tools:
            payload["tools"] = list(native_tools)
        rendered = json.dumps(payload, ensure_ascii=False, default=str)
        if len(rendered) > self.max_initial_context_chars:
            raise ReActV2ContextError(
                f"ReAct V2 context is {len(rendered)} characters, exceeding the "
                f"{self.max_initial_context_chars}-character total context budget. The total includes system "
                "instructions, native tool schemas, the selected section, feedback, traces, steering message, and "
                "branch-local history plus prior ReAct turns; automatic compression is intentionally disabled."
            )

    @staticmethod
    def _native_assistant_message(completion: ToolCompletion) -> dict[str, Any]:
        """Convert a normalized tool completion into a provider chat turn.

        Args:
            completion: Assistant content and normalized native tool calls.

        Returns:
            Assistant message with provider-shaped tool-call entries when any
            calls are present.
        """
        message: dict[str, Any] = {"role": "assistant", "content": completion.content}
        if completion.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in completion.tool_calls
            ]
        return message

    @staticmethod
    def _native_assistant_history_content(completion: ToolCompletion) -> str:
        """Render a native assistant turn as branch-history text.

        Args:
            completion: Assistant content and normalized function calls.

        Returns:
            Readable content followed by each function call, or an explicit
            empty-response marker.
        """
        parts = [completion.content] if completion.content else []
        parts.extend(f"Tool call: {call.name}\nArguments: {call.arguments}" for call in completion.tool_calls)
        return "\n\n".join(parts) or "(empty assistant response)"

    @staticmethod
    def _append_observation(
        messages: list[dict[str, Any]],
        observation: str,
        native_call: NativeToolCall | None,
    ) -> None:
        """Append an observation using the active tool protocol.

        Args:
            messages: Conversation to mutate in place.
            observation: Result or error returned to the proposer.
            native_call: Native call receiving the observation, or ``None``
                when using the text compatibility protocol.
        """
        if native_call is None:
            messages.append({"role": "user", "content": observation})
            return
        messages.append(
            {
                "role": "tool",
                "tool_call_id": native_call.id,
                "content": observation,
            }
        )

    def _apply_to_region(
        self,
        region_text: str,
        edit_target: EditTarget,
        args: EditArgs,
    ) -> tuple[str, list[str]]:
        """Apply typed arguments inside the selected section body.

        Args:
            region_text: Current complete selected-section body.
            edit_target: Named section used to validate the resulting body.
            args: Parsed tool arguments.

        Returns:
            Updated selected-section body and atomic log.

        Raises:
            EditApplicationError: The literal target or anchor is invalid.
            MalformedDocumentError: The result contains a section header.
        """
        new_region, executed = apply_edit(region_text, args)
        self.template.replace_section_body("", edit_target.section, new_region)
        return new_region, executed

    def propose(
        self,
        region_text: str,
        edit_target: EditTarget,
        preferred_tool: EditTool | None,
        steering_message: str | None,
        feedback_summary: str,
        traces_text: str,
        branch_history: Sequence[Mapping[str, Any]],
        max_chars: int | None,
    ) -> ReActV2Result:
        """Run ReAct V2 until a direct semantic call or explicit finish succeeds.

        Args:
            region_text: Complete body of the Controller-selected section.
            edit_target: Controller-selected named section.
            preferred_tool: Direct tool coupled to the semantic action. ``None``
                means the proposer is operating directly over the configured basis.
            steering_message: Manifestor steering delivered as part of the user task.
            feedback_summary: Minibatch failure feedback.
            traces_text: Execution traces grounding the revision.
            branch_history: User/assistant messages from this parent candidate's lineage only.
            max_chars: Maximum completed section-body length, or ``None`` for no limit.

        Returns:
            Completed proposal or an unchanged result with a drop reason.

        Raises:
            ValueError: The selected semantic operator is neither available
                directly nor reproducible by the configured atomic basis.
            TypeError: Native completion support returns an unexpected value or
                branch-history entries have invalid types.
            ReActV2ContextError: Branch history or the accumulated request
                exceeds its configured context budget.
        """
        native_complete = getattr(self.lm, "complete_with_tools", None)
        use_native_tools = callable(native_complete)
        provider_tools = (
            [
                {
                    "type": "function",
                    "function": {
                        "name": tool.value,
                        "description": _NATIVE_TOOL_DESCRIPTIONS[tool],
                        "parameters": _NATIVE_TOOL_PARAMETERS[tool],
                    },
                }
                for tool in self.allowed_tools
            ]
            if use_native_tools
            else []
        )
        direct_tool = preferred_tool if preferred_tool is not None and preferred_tool in self.allowed_tools else None
        lowered_tool = _lowered_semantic_tool(preferred_tool, self.allowed_tools)
        if preferred_tool is not None and direct_tool is None and lowered_tool is None:
            available = ", ".join(tool.value for tool in self.allowed_tools)
            raise ValueError(
                f"Semantic operator {preferred_tool.value} is not directly available and cannot be lowered through "
                f"the configured tool basis: {available}."
            )
        messages = self._initial_messages(
            region_text,
            edit_target,
            preferred_tool,
            steering_message,
            feedback_summary,
            traces_text,
            branch_history,
            native_tools=use_native_tools,
        )
        current = region_text
        executed_all: list[str] = []
        steps: list[ReActV2Step] = []
        valid_calls = 0
        lowered_delete: DeleteTextArgs | None = None
        lowering_complete = False
        last_output = ""

        for turn in range(1, self.max_iterations + 1):
            self._check_context(messages, provider_tools)
            native_calls: tuple[NativeToolCall, ...] = ()
            action_text = ""
            assistant_history_content = ""
            if use_native_tools:
                completion = cast(Any, native_complete)(messages, provider_tools, tool_choice="auto")
                if not isinstance(completion, ToolCompletion):
                    raise TypeError("complete_with_tools must return gepa.lm.ToolCompletion.")
                content = completion.content.strip()
                completion = ToolCompletion(content=content, tool_calls=completion.tool_calls)
                native_calls = completion.tool_calls
                action_text = completion.content
                messages.append(self._native_assistant_message(completion))
                raw = json.dumps(messages[-1], ensure_ascii=False)
                assistant_history_content = self._native_assistant_history_content(completion)
            else:
                raw = self.lm(messages).strip()
                action_text = raw
                assistant_history_content = raw or "(empty assistant response)"
                messages.append({"role": "assistant", "content": raw})
            last_output = assistant_history_content
            finish_blocks = _FINISH_BLOCK_RE.findall(action_text)
            text_tool_blocks = _TOOL_CALL_BLOCK_RE.findall(action_text)
            action_count = len(finish_blocks) + len(text_tool_blocks) + len(native_calls)

            protocol_error: str | None = None
            if use_native_tools and text_tool_blocks:
                protocol_error = (
                    "Text <tool_call> blocks are disabled for this LM; use exactly one provider-native tool call."
                )
            elif action_count != 1:
                protocol_error = (
                    "Reply must contain exactly one action: one tool call or one <finish> block; "
                    f"received {action_count}."
                )
            if protocol_error is not None:
                error = protocol_error
                observation = f"ERROR: {error}"
                steps.append(
                    ReActV2Step(
                        turn,
                        assistant_history_content,
                        "INVALID",
                        observation,
                        error,
                        region_text=current,
                    )
                )
                if native_calls:
                    for call in native_calls:
                        self._append_observation(messages, observation, call)
                else:
                    self._append_observation(messages, observation, None)
                continue
            finish = finish_blocks[0] if finish_blocks else None
            if finish is not None:
                if direct_tool is not None:
                    error = f"The semantic action requires one valid {direct_tool.value} call before finishing."
                    observation = f"ERROR: {error}"
                    steps.append(
                        ReActV2Step(
                            turn,
                            assistant_history_content,
                            "INVALID",
                            observation,
                            error,
                            region_text=current,
                        )
                    )
                    self._append_observation(messages, observation, None)
                    continue
                if lowered_tool is not None and not lowering_complete:
                    required = EditTool.DELETE_TEXT if lowered_delete is None else EditTool.INSERT_TEXT
                    error = (
                        f"The {lowered_tool.value} lowering is incomplete; make the required "
                        f"{required.value} call before finishing."
                    )
                    observation = f"ERROR: {error}"
                    steps.append(
                        ReActV2Step(
                            turn,
                            assistant_history_content,
                            "INVALID",
                            observation,
                            error,
                            region_text=current,
                        )
                    )
                    self._append_observation(messages, observation, None)
                    continue
                if valid_calls == 0 or current == region_text:
                    error = "Cannot finish before at least one valid tool call changes the selected region."
                    observation = f"ERROR: {error}"
                    steps.append(
                        ReActV2Step(
                            turn,
                            assistant_history_content,
                            "INVALID",
                            observation,
                            error,
                            region_text=current,
                        )
                    )
                    self._append_observation(messages, observation, None)
                    continue
                steps.append(
                    ReActV2Step(
                        turn,
                        assistant_history_content,
                        "FINISH",
                        finish.strip(),
                        None,
                        region_text=current,
                    )
                )
                return ReActV2Result(
                    new_text=current,
                    changed=True,
                    executed_edit=executed_all,
                    iterations=turn,
                    tool_calls=valid_calls,
                    final_output=raw,
                    steps=steps,
                )

            native_call = native_calls[0] if native_calls else None
            try:
                if native_call is not None:
                    tool, args = parse_native_tool_call(native_call)
                else:
                    tool, args = parse_tool_call(raw)
                if tool not in self.allowed_tools:
                    allowed = ", ".join(item.value for item in self.allowed_tools)
                    raise ReActV2ProtocolError(f"{tool.value} is unavailable in this run; use one of: {allowed}.")
                if direct_tool is not None and tool is not direct_tool:
                    raise ReActV2ProtocolError(
                        f"This semantic action is coupled to {direct_tool.value}; {tool.value} is not valid here."
                    )
                expected_region: str | None = None
                if lowered_tool is not None:
                    if lowering_complete:
                        raise ReActV2ProtocolError(
                            f"The {lowered_tool.value} decomposition is complete; emit <finish> without "
                            "another tool call."
                        )
                    expected_atomic_tool = EditTool.DELETE_TEXT if lowered_delete is None else EditTool.INSERT_TEXT
                    if tool is not expected_atomic_tool:
                        raise ReActV2ProtocolError(
                            f"The {lowered_tool.value} lowering requires {expected_atomic_tool.value} next; "
                            f"{tool.value} is not valid at this step."
                        )
                    if lowered_delete is not None:
                        assert isinstance(args, InsertTextArgs)
                        if lowered_tool is EditTool.REPLACE_TEXT:
                            direct_args: EditArgs = ReplaceTextArgs(
                                target=lowered_delete.target,
                                text=args.text,
                            )
                        else:
                            if args.text != lowered_delete.target:
                                raise ReActV2ProtocolError(
                                    "The MOVE_TEXT lowering must reinsert exactly the bytes removed by DELETE_TEXT."
                                )
                            direct_args = MoveTextArgs(
                                target=lowered_delete.target,
                                anchor=args.anchor,
                                where=args.where,
                            )
                        expected_region, _ = self._apply_to_region(region_text, edit_target, direct_args)
                        if expected_region == region_text:
                            raise ReActV2ProtocolError(
                                f"The {lowered_tool.value} lowering must change the original selected section; "
                                "choose a distinct replacement or MOVE_TEXT destination."
                            )
                if valid_calls >= self.max_tool_calls:
                    raise ReActV2ProtocolError(
                        f"Tool-call budget exhausted after {self.max_tool_calls} valid calls; emit <finish>."
                    )
                new_region, executed = self._apply_to_region(current, edit_target, args)
                if expected_region is not None and new_region != expected_region:
                    assert lowered_tool is not None
                    raise ReActV2ProtocolError(
                        f"The INSERT_TEXT call does not reproduce one {lowered_tool.value} operation "
                        "on the original selected region. Use the exact deleted location for REPLACE_TEXT, or the "
                        "same destination anchor and placement as MOVE_TEXT."
                    )
                if new_region == current:
                    raise EditApplicationError(f"{tool.value} produced no text change.")
                if max_chars is not None and len(new_region) > max_chars:
                    raise EditApplicationError(
                        f"Edited section is {len(new_region)} characters, exceeding max_chars={max_chars}."
                    )
            except (ReActV2ProtocolError, EditApplicationError, MalformedDocumentError) as exc:
                error = str(exc)
                observation = (
                    f"ERROR: {error}\nThe selected section is unchanged. Correct the call using its latest text."
                )
                steps.append(
                    ReActV2Step(
                        turn,
                        assistant_history_content,
                        "INVALID",
                        observation,
                        error,
                        region_text=current,
                    )
                )
                self._append_observation(messages, observation, native_call)
                continue

            current = new_region
            valid_calls += 1
            if lowered_tool is not None:
                if lowered_delete is None:
                    assert isinstance(args, DeleteTextArgs)
                    lowered_delete = args
                else:
                    lowering_complete = True
            executed_all.extend(executed)
            observation = f"OK: {tool.value} applied.\nLatest selected region:\n{new_region}"
            steps.append(
                ReActV2Step(
                    turn=turn,
                    assistant=assistant_history_content,
                    action=tool.value,
                    observation=observation,
                    error=None,
                    executed_edit=executed,
                    region_text=current,
                )
            )
            if direct_tool is not None:
                return ReActV2Result(
                    new_text=current,
                    changed=True,
                    executed_edit=executed_all,
                    iterations=turn,
                    tool_calls=valid_calls,
                    final_output=raw,
                    steps=steps,
                )
            self._append_observation(messages, f"{observation}\nContinue with one action.", native_call)

        reason = f"No completed revision within {self.max_iterations} ReAct V2 turns."
        if self.logger is not None:
            self.logger.log(reason)
        return ReActV2Result(
            new_text=region_text,
            changed=False,
            executed_edit=executed_all,
            iterations=self.max_iterations,
            tool_calls=valid_calls,
            dropped_reason=reason,
            final_output=last_output,
            steps=steps,
        )
