"""Verify controlled native error delivery and recovery without model requests."""

import json
from copy import deepcopy

import pytest

from examples.hotpotqa.error_recovery_probe import FaultInjectingLM, verify_error_recovery
from examples.hotpotqa.runtime_canary import _EDIT_REGION
from gepa.lm import NativeToolCall, ToolCompletion


class ScriptedNativeLM:
    """Return native replies while recording the messages delivered by the editor."""

    model = "hosted_vllm/deepseek-ai/DeepSeek-V4.1-Flash"

    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def complete_with_tools(self, messages, tools, *, tool_choice=None):
        self.calls.append(deepcopy(messages))
        return next(self.replies)


def replacement() -> ToolCompletion:
    """Construct one valid native replacement before controlled corruption."""
    return ToolCompletion(
        "",
        (
            NativeToolCall(
                "call-1",
                "REPLACE_TEXT",
                json.dumps(
                    {
                        "target": "Cite primary sources.",
                        "text": "Cite primary sources inline.",
                    }
                ),
            ),
        ),
        reasoning_content="Preserve unrelated instructions.",
    )


def test_controlled_error_reaches_model_and_recovery_preserves_reasoning():
    lm = ScriptedNativeLM([replacement(), replacement(), ToolCompletion("<finish>Done.</finish>", ())])
    report = verify_error_recovery(lm)
    assert report["status"] == "passed"
    assert report["injected_fault"]["spontaneous_model_error"] is False
    second_turn = lm.calls[1]
    assert any(
        row.get("role") == "tool" and "ERROR:" in row["content"] and _EDIT_REGION in row["content"]
        for row in second_turn
    )
    assert any(
        row.get("role") == "assistant" and row.get("reasoning_content") == "Preserve unrelated instructions."
        for row in second_turn
    )
    assert report["result"]["steps"][0]["error"]
    assert report["result"]["steps"][-1]["action"] == "FINISH"


def test_finish_without_repair_cannot_qualify():
    lm = ScriptedNativeLM([replacement(), ToolCompletion("<finish>Done.</finish>", ())])
    with pytest.raises(RuntimeError, match="failed the controlled"):
        verify_error_recovery(lm)


def test_fault_injection_does_not_change_the_original_provider_response():
    original = replacement()
    proxy = FaultInjectingLM(ScriptedNativeLM([original]))
    injected = proxy.complete_with_tools([], [])
    assert json.loads(original.tool_calls[0].arguments)["target"] == "Cite primary sources."
    assert json.loads(injected.tool_calls[0].arguments)["target"] == "__QUALIFICATION_MISSING_TARGET__"
    assert injected.reasoning_content == original.reasoning_content


def test_non_native_initial_reply_cannot_qualify():
    proxy = FaultInjectingLM(ScriptedNativeLM([ToolCompletion("REPLACE_TEXT", ())]))
    with pytest.raises(RuntimeError, match="initial native"):
        proxy.complete_with_tools([], [])
