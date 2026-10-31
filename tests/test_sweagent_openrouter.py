"""Exercise the pinned agent and real HTTP serialization with an offline provider."""

import json
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from examples.common.benchmark_types import BenchmarkModels
from examples.common.provider_retries import provider_retry_kwargs
from examples.swebench_live import swe_agent


@pytest.fixture
def offline_rollout(tmp_path, monkeypatch):
    value = os.environ.get("SWE_AGENT_TEST_ROOT")
    if not value:
        pytest.skip("Set SWE_AGENT_TEST_ROOT to the pinned prepared source for integration tests")
    root = Path(value)
    swe_agent.activate(root)
    from sweagent.agent.agents import DefaultAgent
    from sweagent.environment import swe_env

    from examples.swebench_live import swe_agent_rex

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    requests = []
    detail_template = [
        {"type": "reasoning.text", "text": "Keep the reproduction plan.", "format": "unknown", "index": 0},
        {"type": "reasoning.encrypted", "data": "opaque-fixture", "id": "reasoning-id", "index": 1},
    ]
    details = []

    def send(_, request, **kwargs):
        assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
        requests.append(json.loads(request.content))
        current_details = deepcopy(detail_template)
        current_details[0]["text"] += f" Turn {len(requests)}."
        details.append(current_details)
        return httpx.Response(
            200,
            request=request,
            json={
                "id": f"offline-{len(requests)}",
                "object": "chat.completion",
                "created": 1,
                "model": "deepseek/deepseek-v4.1-flash-1017",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning": "Keep the reproduction plan.",
                            "reasoning_details": current_details,
                            "tool_calls": [
                                {
                                    "id": "call-shared",
                                    "type": "function",
                                    "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            },
        )

    monkeypatch.setattr(httpx.Client, "send", send)
    environment = SimpleNamespace(name="offline", start=Mock(), close=Mock(), repo=SimpleNamespace(repo_name="testbed"))
    monkeypatch.setattr(swe_env, "SWEEnv", lambda **_: environment)
    monkeypatch.setattr(swe_agent_rex, "ApptainerDeployment", lambda *_: None)

    def run(agent, **kwargs):
        agent._env = environment
        agent._problem_statement = kwargs["problem_statement"]
        agent.history = [{"role": "user", "content": "Fix the fixture.", "agent": agent.name}]

        def action(step):
            step.observation = "/testbed"
            step.state = {"working_dir": "/testbed"}
            return step

        monkeypatch.setattr(agent, "handle_action", action)
        monkeypatch.setattr(agent, "_get_edited_files_with_context", lambda **_: {})
        agent.step()
        agent.step()
        return SimpleNamespace(info={"exit_status": "submitted", "submission": "offline fixture"})

    monkeypatch.setattr(DefaultAgent, "run", run)
    provider = {"only": ["deepinfra/fp8"], "allow_fallbacks": False, "require_parameters": True}
    kwargs = {
        "api_base": "https://openrouter.ai/api/v1",
        "seed": 0,
        "max_tokens": 65536,
        "temperature": 1.0,
        "top_p": 0.95,
        "timeout": 3600,
        "drop_params": False,
        "extra_body": {"provider": provider, "reasoning": {"effort": "low"}},
        **provider_retry_kwargs(tmp_path / "provider-attempts.jsonl", "solver"),
    }
    original = deepcopy(kwargs)
    models = BenchmarkModels("openrouter/deepseek/deepseek-v4.1-flash-1017", "unused", None, None, kwargs, {})
    result = swe_agent.rollout(
        root,
        {"id": "fixture", "problem_statement": "fixture"},
        {},
        models,
        "Fixed instructions",
        tmp_path,
        16,
        max_steps=30,
    )
    assert kwargs == original
    environment.close.assert_called_once()
    assert len(requests) == 2
    for request in requests:
        assert request["provider"] == provider
        assert request["reasoning"] == {"effort": "low"}
        assert request["seed"] == 0
        assert request["max_tokens"] == 65536
        assert request["temperature"] == 1.0 and request["top_p"] == 0.95
    rows = [json.loads(line) for line in (tmp_path / "provider-attempts.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and all(row["attempt"] == 1 for row in rows)
    return requests, result, details


def test_native_one_tool_parser_disables_parallel_calls(offline_rollout):
    requests, _, _ = offline_rollout
    assert all(request.get("parallel_tool_calls") is False for request in requests)


def test_tool_reasoning_survives_history_and_next_wire_request(offline_rollout):
    requests, result, details = offline_rollout
    assistant = next(message for message in requests[1]["messages"] if message["role"] == "assistant")
    assert assistant.get("reasoning_details") == details[0]
    assert assistant["content"] == ""
    assert assistant["tool_calls"][0]["id"] == "call-shared"
    saved = [message for message in result["trajectory"]["history"] if message["role"] == "assistant"]
    assert [message.get("reasoning_details") for message in saved] == details


def test_native_stats_count_tool_and_reasoning_output(offline_rollout):
    _, result, _ = offline_rollout
    stats = result["trajectory"]["info"]["model_stats"]
    assert stats["api_calls"] == 2
    assert stats["tokens_received"] == 40


def test_repaired_model_stops_before_query_31_and_preserves_provider_failure(tmp_path, monkeypatch):
    value = os.environ.get("SWE_AGENT_TEST_ROOT")
    if not value:
        pytest.skip("Set SWE_AGENT_TEST_ROOT to the pinned prepared source for integration tests")
    swe_agent.activate(Path(value))
    from sweagent.agent.agents import DefaultAgentConfig
    from sweagent.exceptions import InstanceCallLimitExceededError

    from examples.common.provider_retries import ProviderRequestError

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test")
    requests = []

    def send(_, request, **kwargs):
        requests.append(request)
        return httpx.Response(400, request=request, json={"error": {"message": "Definitive offline rejection"}})

    monkeypatch.setattr(httpx.Client, "send", send)
    models = BenchmarkModels(
        "openrouter/deepseek/deepseek-v4.1-flash-1017",
        "unused",
        None,
        None,
        provider_retry_kwargs(tmp_path / "provider-attempts.jsonl", "solver"),
        {},
    )
    config = DefaultAgentConfig.model_validate(swe_agent.configuration(Path(value), models, max_steps=30))
    model = swe_agent._StepLimitedModel(swe_agent.create_agent(config).model, 30)
    # A failed query still consumes the same strict budget; the test deliberately
    # invokes later independent queries to check the guard, not a production retry.
    for _ in range(30):
        with pytest.raises(ProviderRequestError):
            model.query([{"role": "user", "content": "Offline failure fixture"}])
    with pytest.raises(InstanceCallLimitExceededError):
        model.query([])
    assert len(requests) == 30
    rows = [json.loads(line) for line in (tmp_path / "provider-attempts.jsonl").read_text().splitlines()]
    assert len(rows) == 30
    assert all(row["attempt"] == 1 and row["will_retry"] is False for row in rows)
