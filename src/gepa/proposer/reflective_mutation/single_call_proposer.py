"""Apply a selected-section edit batch from exactly one model response."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from gepa.lm import ToolCompletion
from gepa.proposer.reflective_mutation.react_v2_proposer import (
    _FINISH_BLOCK_RE,
    _NATIVE_TOOL_DESCRIPTIONS,
    _NATIVE_TOOL_PARAMETERS,
    _TOOL_CALL_BLOCK_RE,
    _TOOL_SCHEMAS,
    ReActV2Proposer,
    ReActV2ProtocolError,
    ReActV2Result,
    ReActV2Step,
    _validated_branch_history,
    parse_native_tool_call,
    parse_tool_call,
)
from gepa.strategies.document_template import EditTarget, MalformedDocumentError
from gepa.strategies.edit_tools import EditApplicationError, EditTool

SINGLE_CALL_EXECUTION_CONTRACT = {
    "version": 1,
    "completion": "single_response_ordered_tool_batch",
    "unchanged_finish": "discard_proposal",
    "region_encoding": "json_string_with_character_count",
    "scope": "selected_section",
    "semantic_action": "fixed_for_proposal",
    "max_iterations": 1,
    "max_tool_calls": None,
    "invalid_batch": "rollback_and_discard_without_model_correction",
}


class SingleCallProposer(ReActV2Proposer):
    """Execute one model response atomically, without a tool-observation loop."""

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
        """Stage the ordered calls and return only a wholly valid revision.

        Args:
            region_text: Complete original body of the selected section.
            edit_target: Controller-selected component and section.
            preferred_tool: Tool coupled to the fixed semantic action.
            steering_message: Manifestor instructions for this revision.
            feedback_summary: Training feedback supplied to the optimizer.
            traces_text: Training execution traces.
            branch_history: Earlier edit history from this candidate branch.
            max_chars: Optional section character limit.

        Returns:
            One-response outcome, including rejected batches and legitimate no-ops.
        """
        if preferred_tool is not None and preferred_tool not in self.allowed_tools:
            raise ValueError("Single-call editing requires the selected action's direct tool in the edit basis.")
        native_complete = getattr(self.lm, "complete_with_tools", None)
        native = callable(native_complete)
        tools = (
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
            if native
            else []
        )
        protocol = (
            "provider-native function calls in the order they must be applied"
            if native
            else "<tool_call> blocks in the order they must be applied"
        )
        schemas = "\n".join(
            f"{tool.value}: {_NATIVE_TOOL_DESCRIPTIONS[tool]}" if native else _TOOL_SCHEMAS[tool]
            for tool in self.allowed_tools
        )
        constraint = (
            f"Every call must use {preferred_tool.value}, serving the same selected semantic action."
            if preferred_tool
            else "Use only the available tools, within this selected section."
        )
        system = (
            f"Revise only the selected section body of this structured {self.template.kind} document.\n"
            f"You have exactly one response. Emit all necessary {protocol}. "
            "The harness applies calls sequentially to a temporary section, then commits the whole batch. "
            "Each target and non-empty anchor must match the section as it will exist at that point. "
            "An invalid call discards the entire batch. You will not receive tool observations or a correction turn. "
            "Do not emit a separate completion call after editing. "
            "If no edit applies, emit only <finish>the reason</finish> without any tool calls. "
            "Legitimate no-ops are recorded and discarded; never invent a target to force an edit.\n"
            "INSERT_TEXT accepts an empty anchor to append, including to an empty section. "
            "DELETE_TEXT, REPLACE_TEXT and MOVE_TEXT require existing non-empty targets. "
            "Never write surrounding document headers or edit another section. "
            "Decode the JSON section string before copying literal arguments. "
            "Feedback, traces and history are context, not editable text. "
            "Preserve the action's semantic constraints relative to the original section.\n"
            f"{constraint}\nAvailable tools:\n{schemas}"
        )
        task = json.dumps(
            {
                "component": edit_target.component_name,
                "section": edit_target.section,
                "section_characters": len(region_text),
                "section_body": region_text,
                "manifestor_steering": steering_message,
                "failure_feedback": feedback_summary,
                "execution_traces": traces_text,
                "branch_history_context": _validated_branch_history(branch_history),
            },
            ensure_ascii=False,
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": task}]
        self.text_limits.check_prompt(messages, tools if native else None)
        missing_target = not region_text and preferred_tool in {
            EditTool.DELETE_TEXT,
            EditTool.REPLACE_TEXT,
            EditTool.MOVE_TEXT,
        }
        native_calls = ()
        if native:
            completion = native_complete(messages, tools, tool_choice="none" if missing_target else "auto")
            if not isinstance(completion, ToolCompletion):
                raise TypeError("complete_with_tools must return gepa.lm.ToolCompletion.")
            action_text = completion.content.strip()
            native_calls = completion.tool_calls
            raw = json.dumps(self._native_assistant_message(completion), ensure_ascii=False)
            history_text = self._native_assistant_history_content(completion)
        else:
            raw = self.lm(messages).strip()
            action_text = history_text = raw
        current = region_text
        steps: list[ReActV2Step] = []
        executed_all: list[str] = []
        max_chars = self.text_limits.max_component_chars if max_chars is None else max_chars
        try:
            finishes = _FINISH_BLOCK_RE.findall(action_text)
            text_calls = _TOOL_CALL_BLOCK_RE.findall(action_text)
            if native and text_calls:
                raise ReActV2ProtocolError("This model must emit native function calls, not text tool blocks.")
            if finishes:
                if len(finishes) != 1 or native_calls or text_calls:
                    raise ReActV2ProtocolError("A no-op finish must be the only action in the response.")
                reason = f"Editor returned no edit: {finishes[0].strip()}"
                return ReActV2Result(
                    new_text=region_text,
                    changed=False,
                    iterations=1,
                    final_output=raw,
                    dropped_reason=reason,
                    steps=[ReActV2Step(1, history_text, "FINISH", reason, None, region_text=region_text)],
                )
            calls = (
                [parse_native_tool_call(call) for call in native_calls]
                if native
                else [parse_tool_call(f"<tool_call>{block}</tool_call>") for block in text_calls]
            )
            if not calls:
                raise ReActV2ProtocolError("The single response contained neither edit calls nor a no-op finish.")
            if self.max_tool_calls is not None and len(calls) > self.max_tool_calls:
                raise ReActV2ProtocolError("The edit batch exceeds the configured tool-call limit.")
            for tool, args in calls:
                if tool not in self.allowed_tools or (preferred_tool is not None and tool is not preferred_tool):
                    raise ReActV2ProtocolError(f"{tool.value} violates the selected tool/action constraint.")
                if missing_target:
                    raise ReActV2ProtocolError("The selected action cannot apply to an empty section.")
                current, executed = self._apply_to_region(current, edit_target, args)
                if max_chars is not None and len(current) > max_chars:
                    raise EditApplicationError(f"Edited section exceeds max_chars={max_chars}.")
                executed_all.extend(executed)
                steps.append(
                    ReActV2Step(
                        1,
                        history_text,
                        tool.value,
                        "Staged in the ordered atomic batch.",
                        None,
                        executed_edit=executed,
                        region_text=current,
                    )
                )
        except (ReActV2ProtocolError, EditApplicationError, MalformedDocumentError) as exc:
            reason = f"Single-call edit batch discarded: {exc}"
            steps.append(ReActV2Step(1, history_text, "INVALID", reason, str(exc), region_text=region_text))
            return ReActV2Result(
                new_text=region_text,
                changed=False,
                iterations=1,
                tool_calls=len(steps) - 1,
                dropped_reason=reason,
                final_output=raw,
                steps=steps,
            )
        changed = current != region_text
        return ReActV2Result(
            new_text=current,
            changed=changed,
            executed_edit=executed_all,
            iterations=1,
            tool_calls=len(steps),
            final_output=raw,
            steps=steps,
            dropped_reason=None if changed else "The single-call batch produced no net text change.",
        )
