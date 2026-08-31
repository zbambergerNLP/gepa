"""Tests for the fail-closed local HotPotQA runtime canary."""

import sys
from pathlib import Path
from unittest.mock import Mock, call

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from examples.hotpotqa import runtime_canary
from gepa.strategies.edit_tools import EDIT_TOOL_SETS


@pytest.mark.parametrize(
    "api_base",
    [
        "http://localhost:8000/v1",
        "http://127.0.0.1:8001/v1/",
        "http://[::1]:8002/v1",
    ],
)
def test_validate_loopback_api_base_accepts_explicit_local_v1_endpoints(api_base: str) -> None:
    """Accept HTTP loopback endpoints with an explicit port and v1 path.

    Args:
        api_base: Valid local endpoint under test.
    """
    runtime_canary._validate_loopback_api_base(api_base)


@pytest.mark.parametrize(
    "api_base",
    [
        "https://127.0.0.1:8000/v1",
        "http://127.0.0.1/v1",
        "http://0.0.0.0:8000/v1",
        "http://example.com:8000/v1",
        "http://127.0.0.1:8000/chat/completions",
        "http://127.0.0.1:8000/v1?token=secret",
        "http://127.0.0.1:not-a-port/v1",
    ],
)
def test_validate_loopback_api_base_rejects_nonlocal_or_ambiguous_endpoints(api_base: str) -> None:
    """Reject endpoints that do not match the local scientific-runtime contract.

    Args:
        api_base: Invalid endpoint under test.
    """
    with pytest.raises(runtime_canary.RuntimeCanaryError, match="local HTTP loopback"):
        runtime_canary._validate_loopback_api_base(api_base)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("   ", "empty text"),
        ("<think>private reasoning</think>answer", "leaked inline reasoning"),
        ("!" * 32, "repeated-character degeneration"),
        ("ab" * 32, "repeated-character degeneration"),
    ],
)
def test_require_healthy_text_rejects_degenerate_output(text: str, message: str) -> None:
    """Reject empty, leaked-reasoning, and low-diversity model output.

    Args:
        text: Unhealthy model text under test.
        message: Expected failure description.
    """
    with pytest.raises(runtime_canary.RuntimeCanaryError, match=message):
        runtime_canary._require_healthy_text(text, "Test probe")


def test_run_runtime_canary_requires_twenty_attempts_before_model_setup(monkeypatch) -> None:
    """Reject a weak canary before resolving or constructing its model client.

    Args:
        monkeypatch: Pytest fixture used to guard model setup calls.
    """
    resolve_kwargs = Mock()
    lm_factory = Mock()
    monkeypatch.setattr(runtime_canary, "resolve_hotpotqa_lm_kwargs", resolve_kwargs)
    monkeypatch.setattr(runtime_canary, "LM", lm_factory)

    with pytest.raises(runtime_canary.RuntimeCanaryError, match="at least 20 repetitions"):
        runtime_canary.run_runtime_canary(
            "hosted_vllm/zai-org/GLM-5.3-Flash",
            "http://127.0.0.1:8000/v1",
            19,
        )

    resolve_kwargs.assert_not_called()
    lm_factory.assert_not_called()


def test_run_runtime_canary_cycles_all_four_tools_for_twenty_attempts(monkeypatch) -> None:
    """Run every probe and distribute twenty edit attempts evenly across tools.

    Args:
        monkeypatch: Pytest fixture used to isolate the orchestration contract.
    """
    lm = object()
    resolve_kwargs = Mock(return_value={"temperature": 1.0})
    lm_factory = Mock(return_value=lm)
    ordinary_probe = Mock()
    continuation_probe = Mock()
    edit_probe = Mock()
    monkeypatch.setattr(runtime_canary, "resolve_hotpotqa_lm_kwargs", resolve_kwargs)
    monkeypatch.setattr(runtime_canary, "LM", lm_factory)
    monkeypatch.setattr(runtime_canary, "_ordinary_completion_probe", ordinary_probe)
    monkeypatch.setattr(runtime_canary, "_tool_continuation_probe", continuation_probe)
    monkeypatch.setattr(runtime_canary, "_edit_probe", edit_probe)

    model = "hosted_vllm/zai-org/GLM-5.3-Flash"
    api_base = "http://127.0.0.1:8000/v1"
    summary = runtime_canary.run_runtime_canary(model, api_base, 20)

    resolve_kwargs.assert_called_once_with(model, api_base, "scientific")
    lm_factory.assert_called_once_with(model, temperature=1.0, timeout=600)
    ordinary_probe.assert_called_once_with(lm)
    continuation_probe.assert_called_once_with(lm)
    tools = EDIT_TOOL_SETS["broad"]
    assert edit_probe.call_args_list == [
        call(lm, tools[offset % len(tools)], offset + 1) for offset in range(20)
    ]
    assert summary == {
        "status": "passed",
        "model": model,
        "api_base": api_base,
        "attempts": 20,
        "tool_attempts": {tool.value: 5 for tool in sorted(tools, key=lambda item: item.value)},
        "ordinary_completion": "passed",
        "tool_result_continuation": "passed",
    }


def test_run_runtime_canary_propagates_probe_failure_and_stops(monkeypatch) -> None:
    """Propagate a failed continuation probe without attempting any edits.

    Args:
        monkeypatch: Pytest fixture used to inject a runtime-probe failure.
    """
    lm = object()
    failure = runtime_canary.RuntimeCanaryError("native tool continuation failed")
    ordinary_probe = Mock()
    continuation_probe = Mock(side_effect=failure)
    edit_probe = Mock()
    monkeypatch.setattr(runtime_canary, "resolve_hotpotqa_lm_kwargs", Mock(return_value={}))
    monkeypatch.setattr(runtime_canary, "LM", Mock(return_value=lm))
    monkeypatch.setattr(runtime_canary, "_ordinary_completion_probe", ordinary_probe)
    monkeypatch.setattr(runtime_canary, "_tool_continuation_probe", continuation_probe)
    monkeypatch.setattr(runtime_canary, "_edit_probe", edit_probe)

    with pytest.raises(runtime_canary.RuntimeCanaryError) as exc_info:
        runtime_canary.run_runtime_canary(
            "hosted_vllm/zai-org/GLM-5.3-Flash",
            "http://127.0.0.1:8000/v1",
            20,
        )

    assert exc_info.value is failure
    ordinary_probe.assert_called_once_with(lm)
    continuation_probe.assert_called_once_with(lm)
    edit_probe.assert_not_called()
