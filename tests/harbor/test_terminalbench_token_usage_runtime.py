"""Capture raw usage through Harbor's actual LiteLLM and truncation recovery."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("harbor.agents.terminus_2")

import litellm
from harbor.llms.base import ContextLengthExceededError
from harbor.llms.chat import Chat
from harbor.models.trajectories import Step

from examples.common import provider_retries
from examples.common.experiment_models import EXPERIMENT_MODELS, experiment_request_overrides
from examples.common.provider_retries import ProviderRequestError, install_provider_retries
from examples.terminalbench.model_settings import terminalbench_decoding, terminalbench_limits, terminalbench_model_info
from examples.terminalbench.terminus_agent import PromptedTerminus
from gepa.adapters.terminal_bench_adapter.documents import seed_documents, write_document_bundle


@pytest.fixture(params=EXPERIMENT_MODELS)
def runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> tuple:
    """Build the real agent and model client with only the provider transport replaced."""
    model = request.param
    bundle = write_document_bundle(tmp_path, seed_documents("generic"))
    agent = PromptedTerminus(
        logs_dir=tmp_path / "logs",
        model_name=model,
        prompt_template_path=str(tmp_path / "terminus-prompt.txt"),
        document_bundle_path=str(bundle),
        token_limits=terminalbench_limits(model),
        model_info=terminalbench_model_info(model),
        llm_kwargs={
            "num_retries": 0,
            **terminalbench_decoding(model),
            **experiment_request_overrides(model, explicit_reasoning=True),
        },
        api_base="http://localhost:8000/v1",
        record_terminal_session=False,
    )
    provider = AsyncMock()
    monkeypatch.setattr(litellm, "acompletion", provider)
    monkeypatch.setattr(litellm, "completion_cost", Mock(return_value=0.0))
    monkeypatch.setattr(provider_retries.asyncio, "sleep", AsyncMock())
    install_provider_retries()
    return agent, provider, tmp_path, model


def response(model: str, content: str, finish: str = "stop") -> litellm.ModelResponse:
    """Return provider-reported counts independently of the small simulated content."""
    output = 32_768 if finish == "length" else 20
    return litellm.ModelResponse(
        model=model,
        choices=[{"message": {"role": "assistant", "content": content}, "finish_reason": finish}],
        usage={
            "prompt_tokens": 100,
            "completion_tokens": output,
            "total_tokens": 100 + output,
            "completion_tokens_details": {"reasoning_tokens": output - 10},
        },
    )


@pytest.mark.parametrize("valid_prefix", [False, True])
def test_truncated_usage_survives_json_repair(runtime: tuple, valid_prefix: bool) -> None:
    """Retain the first capped completion even when Harbor successfully recovers."""
    agent, provider, root, model = runtime
    valid = json.dumps({"analysis": "done", "plan": "finish", "commands": [], "task_complete": True})
    provider.side_effect = [response(model, valid if valid_prefix else "{incomplete", "length"), response(model, valid)]
    result = asyncio.run(agent._query_llm(Chat(agent._llm), "input"))
    assert result.content == valid
    assert provider.call_count == 2
    assert agent._llm.get_model_output_limit() == 32_768
    assert agent._llm.get_model_context_limit() == terminalbench_limits(model)["context_tokens"]
    assert agent._model_name == model
    for call in provider.call_args_list:
        assert call.kwargs["model"] == model
        assert call.kwargs["max_tokens"] == 32_768
        assert call.kwargs["extra_body"] == experiment_request_overrides(model, explicit_reasoning=True)["extra_body"]
    records = [json.loads(line) for line in (root / "logs" / "token-usage.jsonl").read_text().splitlines()]
    assert len(records) == provider.call_count
    assert records[0]["length_finish"] is True
    assert records[0]["output_cap_reached"] is True
    assert records[0]["completion_tokens"] == 32_768
    assert records[0]["reasoning_tokens"] == 32_758
    assert records[0]["requested_model"] == model
    assert records[1]["length_finish"] is False
    assert records[1]["completion_tokens"] == 20
    assert "32768 tokens" in provider.call_args.kwargs["messages"][-1]["content"]


def test_summary_calls_use_the_same_cap_and_usage_log(runtime: tuple) -> None:
    """Observe all three native summarization calls without double-counting copied history."""
    agent, provider, root, model = runtime
    provider.side_effect = [response(model, content) for content in ("SUMMARY", "QUESTIONS", "ANSWERS")]
    chat = Chat(agent._llm)
    chat._messages = [{"role": "user", "content": "TASK_INPUT"}]
    agent._trajectory_steps = [Step(step_id=1, source="user", message="TASK_INPUT")]
    session = SimpleNamespace(capture_pane=AsyncMock(return_value="REAL_STATE"))
    asyncio.run(agent._summarize(chat, "TASK_INPUT", session))
    records = [json.loads(line) for line in (root / "logs" / "token-usage.jsonl").read_text().splitlines()]
    assert len(records) == provider.call_count == 3
    assert all(call.kwargs["max_tokens"] == 32_768 for call in provider.call_args_list)
    assert sum(record["completion_tokens"] for record in records) == 60


def test_provider_failure_retains_unknown_usage_and_original_error(runtime: tuple) -> None:
    """Record a failed transport attempt without inventing token counts or retries."""
    agent, provider, root, model = runtime
    error = litellm.AuthenticationError(message="private error", model=model, llm_provider="hosted_vllm")
    provider.side_effect = error
    with pytest.raises(ProviderRequestError):
        asyncio.run(agent._llm.call("input"))
    provider.assert_called_once()
    record = json.loads((root / "logs" / "token-usage.jsonl").read_text())
    assert record["error_type"] == "AuthenticationError"
    assert record["completion_tokens"] is record["prompt_tokens"] is record["length_finish"] is None
    assert "private error" not in json.dumps(record)


@pytest.mark.parametrize("succeeds", [True, False])
def test_main_agent_provider_retries_do_not_multiply(runtime: tuple, succeeds: bool) -> None:
    """Count real transport attempts across both formerly nested Harbor layers."""
    agent, provider, root, model = runtime
    error = litellm.ServiceUnavailableError(message="private error", model=model, llm_provider="hosted_vllm")
    provider.side_effect = [error, error, response(model, "done") if succeeds else error]
    if succeeds:
        assert asyncio.run(agent._query_llm(Chat(agent._llm), "input")).content == "done"
    else:
        with pytest.raises(ProviderRequestError):
            asyncio.run(agent._query_llm(Chat(agent._llm), "input"))
    assert provider.call_count == 3
    records = [json.loads(line) for line in (root / "logs" / "token-usage.jsonl").read_text().splitlines()]
    assert len(records) == 3
    assert records[0]["error_type"] == "ServiceUnavailableError"
    assert records[-1]["completion_tokens"] == (20 if succeeds else None)
    assert all(call.kwargs["num_retries"] == call.kwargs["max_retries"] == 0 for call in provider.call_args_list)


def test_bad_request_does_not_trigger_harbor_parameter_fallback(runtime: tuple) -> None:
    """Stop before Harbor can silently remove request fields and call again."""
    agent, provider, _root, model = runtime
    provider.side_effect = litellm.BadRequestError(
        message="Unrecognized request argument session_id",
        model=model,
        llm_provider="hosted_vllm",
    )
    with pytest.raises(ProviderRequestError):
        asyncio.run(agent._query_llm(Chat(agent._llm), "input"))
    provider.assert_called_once()


def test_context_overflow_in_bad_request_keeps_harbor_recovery(runtime: tuple) -> None:
    """Translate a context-shaped HTTP 400 without automatically retrying it."""
    agent, provider, _root, model = runtime
    provider.side_effect = litellm.BadRequestError(
        message="Input exceeds the model's context length",
        model=model,
        llm_provider="hosted_vllm",
    )
    with pytest.raises(ContextLengthExceededError):
        asyncio.run(agent._llm.call("input"))
    provider.assert_called_once()


@pytest.mark.parametrize("during_context_recovery", [False, True])
def test_summary_provider_failure_stops_without_fallback(runtime: tuple, during_context_recovery: bool) -> None:
    """Propagate exhausted summary requests instead of manufacturing continuation text."""
    agent, provider, _root, model = runtime
    error = litellm.ServiceUnavailableError(message="private error", model=model, llm_provider="hosted_vllm")
    chat = Chat(agent._llm)
    chat._messages = [{"role": "user", "content": "TASK_INPUT"}]
    agent._trajectory_steps = [Step(step_id=1, source="user", message="TASK_INPUT")]
    session = SimpleNamespace(capture_pane=AsyncMock(return_value="REAL_STATE"))
    if during_context_recovery:
        provider.side_effect = [
            litellm.ContextWindowExceededError(message="context full", model=model, llm_provider="hosted_vllm"),
            error,
            error,
            error,
        ]
        operation = agent._query_llm(chat, "input", "TASK_INPUT", session)
    else:
        provider.side_effect = error
        agent._count_total_tokens = Mock(return_value=agent._llm.get_model_context_limit())
        operation = agent._check_proactive_summarization(chat, "TASK_INPUT", session)
    with pytest.raises(ProviderRequestError):
        asyncio.run(operation)
    assert provider.call_count == (4 if during_context_recovery else 3)
