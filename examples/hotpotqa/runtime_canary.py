"""Fail closed when a local HotPotQA model cannot sustain ReAct V2 tools."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from typing import Any
from urllib.parse import urlsplit

from examples.hotpotqa.utils import resolve_hotpotqa_lm_kwargs
from gepa.lm import LM, ToolCompletion
from gepa.proposer.reflective_mutation.react_v2_proposer import ReActV2Proposer
from gepa.strategies.document_template import TEMPLATE_FAMILIES, EditTarget
from gepa.strategies.edit_tools import EDIT_TOOL_SETS, EditTool

_CANARY_TIMEOUT_SECONDS = 600
_MINIMUM_ATTEMPTS = 20
_REPEATED_CHARACTER_RE = re.compile(r"(\S)\1{31,}")
_EDIT_REGION = (
    "Act as a careful research assistant. Verify every claim. Cite primary sources. "
    "Avoid unsupported conclusions. Return a concise answer."
)
_EDIT_STEERING = {
    EditTool.INSERT_TEXT: (
        'Use INSERT_TEXT with anchor "Verify every claim.", where "after", and text '
        '" State uncertainty explicitly."'
    ),
    EditTool.DELETE_TEXT: 'Use DELETE_TEXT with target "Avoid unsupported conclusions."',
    EditTool.REPLACE_TEXT: (
        'Use REPLACE_TEXT with target "Cite primary sources." and text "Cite primary sources inline."'
    ),
    EditTool.MOVE_TEXT: (
        'Use MOVE_TEXT with target "Return a concise answer.", anchor "Verify every claim.", and where "before".'
    ),
}


class RuntimeCanaryError(RuntimeError):
    """Signal that a local model or serving runtime failed the compatibility gate."""


def _validate_loopback_api_base(api_base: str) -> None:
    """Require an explicit local OpenAI-compatible /v1 endpoint.

    Args:
        api_base: Candidate completion endpoint supplied on the command line.

    Raises:
        RuntimeCanaryError: The endpoint is not HTTP loopback with an explicit
            port and /v1 path.
    """
    parsed = urlsplit(api_base)
    try:
        valid = (
            parsed.scheme == "http"
            and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            and parsed.port is not None
            and parsed.path.rstrip("/") == "/v1"
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        valid = False
    if not valid:
        raise RuntimeCanaryError(
            "The runtime canary requires a local HTTP loopback endpoint with an explicit port and /v1 path."
        )


def _require_healthy_text(text: str, label: str) -> str:
    """Reject empty, leaked-reasoning, or degenerate model text.

    Args:
        text: Model-produced content or serialized tool arguments.
        label: Human-readable probe stage included in failures.

    Returns:
        The stripped non-degenerate text.

    Raises:
        RuntimeCanaryError: The text is empty, exposes inline reasoning tags,
            or degenerates into a repeated character pattern.
    """
    stripped = text.strip()
    if not stripped:
        raise RuntimeCanaryError(f"{label} returned empty text.")
    if "<think>" in stripped or "</think>" in stripped:
        raise RuntimeCanaryError(f"{label} leaked inline reasoning markup.")
    repeated = _REPEATED_CHARACTER_RE.search(stripped)
    visible_characters = [character for character in stripped if not character.isspace()]
    low_diversity = len(visible_characters) >= 64 and len(set(visible_characters)) <= 2
    if repeated is not None or low_diversity:
        raise RuntimeCanaryError(f"{label} returned repeated-character degeneration.")
    return stripped


def _ordinary_completion_probe(lm: LM) -> None:
    """Verify one ordinary answer-only completion through the local runtime.

    Args:
        lm: Configured local model client.

    Raises:
        RuntimeCanaryError: The response is empty, degenerate, or does not
            follow the readiness instruction.
    """
    response = lm(
        [
            {
                "role": "system",
                "content": "This is a serving-runtime compatibility probe. Follow the user instruction exactly.",
            },
            {"role": "user", "content": "Reply with the exact token CANARY_READY."},
        ]
    )
    response = _require_healthy_text(response, "Ordinary completion probe")
    if "CANARY_READY" not in response:
        raise RuntimeCanaryError(
            f"Ordinary completion probe did not return the readiness token: {response!r}"
        )


def _tool_continuation_probe(lm: LM) -> None:
    """Verify a native tool call followed by its tool-result continuation.

    Args:
        lm: Configured local model client with provider-native tool support.

    Raises:
        RuntimeCanaryError: Either turn is malformed, calls an unexpected
            function, ignores the tool result, or returns degenerate content.
    """
    tool = {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Return a supplied runtime-readiness value.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    }
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "This is a serving-runtime compatibility probe. Call the requested tool exactly once, then use its "
                "result when it is returned."
            ),
        },
        {"role": "user", "content": "Call echo exactly once with value CANARY_TOOL_READY."},
    ]
    completion = lm.complete_with_tools(messages, [tool], tool_choice="auto")
    if not isinstance(completion, ToolCompletion):
        raise RuntimeCanaryError("Native tool probe returned an unexpected completion type.")
    if completion.reasoning_content:
        _require_healthy_text(completion.reasoning_content, "Native echo reasoning")
    if len(completion.tool_calls) != 1:
        raise RuntimeCanaryError(
            f"Native tool probe returned {len(completion.tool_calls)} calls instead of exactly one."
        )
    call = completion.tool_calls[0]
    if call.name != "echo" or not call.id:
        raise RuntimeCanaryError(f"Native tool probe returned an unknown or unaddressable call: {call!r}")
    arguments_text = _require_healthy_text(call.arguments, "Native echo arguments")
    try:
        arguments = json.loads(arguments_text)
    except json.JSONDecodeError as exc:
        raise RuntimeCanaryError("Native echo arguments are not valid JSON.") from exc
    if arguments != {"value": "CANARY_TOOL_READY"}:
        raise RuntimeCanaryError(f"Native echo arguments do not match the requested schema: {arguments!r}")

    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "content": completion.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
        ],
    }
    if completion.reasoning_content:
        assistant_message["reasoning_content"] = completion.reasoning_content
    messages.append(assistant_message)
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call.id,
            "content": "CANARY_TOOL_READY. Reply with the exact token CANARY_CONTINUED and do not call a tool.",
        }
    )
    continuation = lm.complete_with_tools(messages, [tool], tool_choice="auto")
    if continuation.reasoning_content:
        _require_healthy_text(continuation.reasoning_content, "Tool-result continuation reasoning")
    if continuation.tool_calls:
        raise RuntimeCanaryError("Tool-result continuation called another function instead of returning content.")
    continuation_text = _require_healthy_text(
        continuation.content,
        "Tool-result continuation",
    )
    if "CANARY_CONTINUED" not in continuation_text:
        raise RuntimeCanaryError(
            f"Tool-result continuation did not incorporate the observation: {continuation_text!r}"
        )


def _validate_edit_result(tool: EditTool, edited_text: str) -> None:
    """Require the requested atomic edit to have its exact intended effect.

    Args:
        tool: Edit operator selected for the representative ReAct V2 probe.
        edited_text: Section body returned by the proposer.

    Raises:
        RuntimeCanaryError: The resulting body does not reflect the requested
            insert, delete, replace, or move operation.
    """
    if tool is EditTool.INSERT_TEXT and "State uncertainty explicitly." not in edited_text:
        raise RuntimeCanaryError("INSERT_TEXT probe omitted the requested inserted text.")
    if tool is EditTool.DELETE_TEXT and "Avoid unsupported conclusions." in edited_text:
        raise RuntimeCanaryError("DELETE_TEXT probe retained the requested deletion target.")
    if tool is EditTool.REPLACE_TEXT and (
        "Cite primary sources inline." not in edited_text
        or "Cite primary sources. " in edited_text
    ):
        raise RuntimeCanaryError("REPLACE_TEXT probe did not apply the exact replacement.")
    if tool is EditTool.MOVE_TEXT:
        moved = edited_text.find("Return a concise answer.")
        anchor = edited_text.find("Verify every claim.")
        if moved == -1 or anchor == -1 or moved > anchor:
            raise RuntimeCanaryError("MOVE_TEXT probe did not move the target before its anchor.")


def _edit_probe(lm: LM, tool: EditTool, attempt: int) -> None:
    """Exercise one real ReAct V2 proposal with the complete four-tool menu.

    Args:
        lm: Configured local model client.
        tool: Direct operator coupled to this attempt's semantic action.
        attempt: One-based repetition number used in diagnostic labels.

    Raises:
        RuntimeCanaryError: ReAct V2 emits malformed or unknown tool JSON,
            retries after a protocol error, fails to edit, or degenerates.
    """
    proposer = ReActV2Proposer(
        lm,
        TEMPLATE_FAMILIES["generic"]["system_prompt"],
        EDIT_TOOL_SETS["broad"],
        max_iterations=2,
        max_tool_calls=1,
    )
    result = proposer.propose(
        region_text=_EDIT_REGION,
        edit_target=EditTarget("final_answer", "Task"),
        preferred_tool=tool,
        steering_message=(
            "The Controller selected a bounded semantic revision. Apply only this literal operation: "
            f"{_EDIT_STEERING[tool]}"
        ),
        feedback_summary=(
            "The current answer sometimes overlooks uncertainty and source attribution. Preserve the task's "
            "other requirements and make only the Controller-selected change."
        ),
        traces_text=(
            "Example 1: the answer stated an unsupported conclusion.\n"
            "Example 2: the answer used a secondary source when a primary source was available."
        ),
        branch_history=[
            {"role": "user", "content": "Keep each revision scoped to the selected section."},
            {"role": "assistant", "content": "I will preserve unrelated instructions."},
        ],
        max_chars=2_000,
    )
    if not result.changed or result.tool_calls != 1 or result.dropped_reason is not None:
        raise RuntimeCanaryError(
            f"ReAct V2 {tool.value} attempt {attempt} did not complete exactly one edit: {result!r}"
        )
    if len(result.steps) != 1 or result.steps[0].error is not None or result.steps[0].action != tool.value:
        raise RuntimeCanaryError(
            f"ReAct V2 {tool.value} attempt {attempt} contained a malformed or retried action: {result.steps!r}"
        )
    _require_healthy_text(result.steps[0].assistant, f"ReAct V2 {tool.value} attempt {attempt}")
    _validate_edit_result(tool, result.new_text)


def run_runtime_canary(model: str, api_base: str, attempts: int) -> dict[str, object]:
    """Run the complete local completion and ReAct V2 compatibility gate.

    Args:
        model: Exact local LiteLLM model identifier served by the Slurm job.
        api_base: Local OpenAI-compatible /v1 endpoint.
        attempts: Number of representative four-tool-menu repetitions.

    Returns:
        JSON-serializable pass summary with per-operator attempt counts.

    Raises:
        RuntimeCanaryError: The endpoint is non-local, fewer than twenty
            repetitions are requested, or any completion/tool probe fails.
        ValueError: The model identifier is outside the scientific catalog.
    """
    _validate_loopback_api_base(api_base)
    if attempts < _MINIMUM_ATTEMPTS:
        raise RuntimeCanaryError(
            f"The fail-closed runtime gate requires at least {_MINIMUM_ATTEMPTS} repetitions; received {attempts}."
        )
    lm_kwargs = resolve_hotpotqa_lm_kwargs(model, api_base, "scientific")
    lm_kwargs["timeout"] = _CANARY_TIMEOUT_SECONDS
    lm = LM(model, **lm_kwargs)
    _ordinary_completion_probe(lm)
    _tool_continuation_probe(lm)

    tool_counts: Counter[str] = Counter()
    tools = EDIT_TOOL_SETS["broad"]
    for offset in range(attempts):
        tool = tools[offset % len(tools)]
        _edit_probe(lm, tool, offset + 1)
        tool_counts[tool.value] += 1
    missing_tools = [tool.value for tool in tools if tool_counts[tool.value] == 0]
    if missing_tools:
        raise RuntimeCanaryError(
            f"Runtime canary did not exercise every broad edit tool: {', '.join(missing_tools)}"
        )
    return {
        "status": "passed",
        "model": model,
        "api_base": api_base,
        "attempts": attempts,
        "tool_attempts": dict(sorted(tool_counts.items())),
        "ordinary_completion": "passed",
        "tool_result_continuation": "passed",
    }


def main() -> None:
    """Parse the runtime-canary CLI and print its pass attestation."""
    parser = argparse.ArgumentParser(
        description="Fail closed unless a local HotPotQA runtime supports ordinary and ReAct V2 tool calls."
    )
    parser.add_argument("--model", required=True, help="Exact local LiteLLM model identifier")
    parser.add_argument("--api-base", required=True, help="Local OpenAI-compatible /v1 endpoint")
    parser.add_argument(
        "--attempts",
        required=True,
        type=int,
        help="Representative four-tool-menu repetitions; production requires at least 20",
    )
    args = parser.parse_args()
    try:
        summary = run_runtime_canary(args.model, args.api_base, args.attempts)
    except (RuntimeError, ValueError, TypeError) as exc:
        parser.exit(1, f"Runtime canary failed: {exc}\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
