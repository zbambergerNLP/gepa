"""Verify physical-attempt limits at the shared provider boundary, offline."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx
import litellm
import pytest

from examples.common import provider_retries
from examples.common.provider_retries import (
    PROVIDER_RETRY_KEY,
    ProviderRequestError,
    install_provider_retries,
    provider_retry_kwargs,
)
from gepa.lm import LM


def response() -> litellm.ModelResponse:
    """Return a completed response with provider-reported usage."""
    return litellm.ModelResponse(
        model="local-model",
        choices=[{"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
        usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    )


def rows(path: Path) -> list[dict]:
    """Read persisted physical attempts in append order."""
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("succeeds", [False, True])
def test_exactly_three_attempts_with_backoff_and_usage(tmp_path, monkeypatch, asynchronous, succeeds):
    """Bound consecutive connection failures and retain every attempt's evidence."""
    raw = response()
    outcomes = [
        ConnectionError("private diagnostic"),
        ConnectionError("private diagnostic"),
        raw if succeeds else ConnectionError("private diagnostic"),
    ]
    provider = AsyncMock(side_effect=outcomes) if asynchronous else Mock(side_effect=outcomes)
    sleep = AsyncMock() if asynchronous else Mock()
    monkeypatch.setattr(litellm, "acompletion" if asynchronous else "completion", provider)
    monkeypatch.setattr(provider_retries.asyncio if asynchronous else provider_retries.time, "sleep", sleep)
    path = tmp_path / "provider-attempts.jsonl"
    kwargs = {
        "model": "hosted_vllm/test",
        "messages": [{"role": "user", "content": "private prompt"}],
        **provider_retry_kwargs(path, "editor"),
    }
    install_provider_retries()

    def invoke():
        """Exercise the selected public LiteLLM boundary."""
        return asyncio.run(litellm.acompletion(**kwargs)) if asynchronous else litellm.completion(**kwargs)

    if succeeds:
        assert invoke() is raw
    else:
        with pytest.raises(ProviderRequestError) as caught:
            invoke()
        assert isinstance(caught.value.__cause__, ConnectionError)
    assert provider.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == [1.0, 2.0]
    for call in provider.call_args_list:
        assert call.kwargs["num_retries"] == call.kwargs["max_retries"] == 0
        assert PROVIDER_RETRY_KEY not in call.kwargs
    records = rows(path)
    assert [record["attempt"] for record in records] == [1, 2, 3]
    assert len({record["request_id"] for record in records}) == 1
    assert [record["will_retry"] for record in records] == [True, True, False]
    assert records[0]["prompt_tokens"] is records[0]["cost_usd"] is None
    assert records[-1]["completion_tokens"] == (2 if succeeds else None)
    assert "private" not in path.read_text()


@pytest.mark.parametrize("status", [400, 401, 403, 404, 408, 422, 429, 500, 501, 502, 503, 504])
def test_only_temporary_http_errors_retry(tmp_path, monkeypatch, status):
    """Reject permanent failures immediately and preserve status codes without bodies."""
    request = httpx.Request("POST", "http://localhost/test")
    error = httpx.HTTPStatusError("private message", request=request, response=httpx.Response(status, request=request))
    provider = Mock(side_effect=error)
    monkeypatch.setattr(litellm, "completion", provider)
    monkeypatch.setattr(provider_retries.time, "sleep", Mock())
    path = tmp_path / "attempts.jsonl"
    settings = provider_retry_kwargs(path)
    with pytest.raises(ProviderRequestError):
        litellm.completion(model="hosted_vllm/test", **settings)
    expected = 3 if status in {408, 429, 500, 502, 503, 504} else 1
    assert provider.call_count == len(rows(path)) == expected
    assert all(record["status_code"] == status for record in rows(path))


@pytest.mark.parametrize(
    "error", [ValueError("invalid"), httpx.LocalProtocolError("invalid"), httpx.UnsupportedProtocol("invalid")]
)
def test_programming_and_configuration_errors_are_not_retried(tmp_path, monkeypatch, error):
    """Keep local request bugs outside the transient retry policy."""
    provider = Mock(side_effect=error)
    monkeypatch.setattr(litellm, "completion", provider)
    settings = provider_retry_kwargs(tmp_path / "attempts.jsonl")
    with pytest.raises(ProviderRequestError):
        litellm.completion(model="hosted_vllm/test", **settings)
    provider.assert_called_once()


def test_deadline_covers_all_attempts_and_backoff(tmp_path, monkeypatch):
    """Spend one shared timeout rather than resetting it on each request."""
    now = [0.0]

    def fail(**kwargs):
        """Consume two seconds before returning a connection failure."""
        now[0] += 2
        raise ConnectionError("failed")

    provider = Mock(side_effect=fail)
    monkeypatch.setattr(litellm, "completion", provider)
    monkeypatch.setattr(provider_retries.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(provider_retries.time, "sleep", lambda delay: now.__setitem__(0, now[0] + delay))
    settings = provider_retry_kwargs(tmp_path / "attempts.jsonl")
    with pytest.raises(ProviderRequestError):
        litellm.completion(model="hosted_vllm/test", timeout=5, **settings)
    assert [call.kwargs["timeout"] for call in provider.call_args_list] == [5.0, 2.0]
    assert now[0] == 5.0


def test_async_cancellation_never_retries(tmp_path, monkeypatch):
    """Propagate an existing task timeout's cancellation without another request."""
    provider = AsyncMock(side_effect=asyncio.CancelledError())
    monkeypatch.setattr(litellm, "acompletion", provider)
    path = tmp_path / "attempts.jsonl"
    settings = provider_retry_kwargs(path)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(litellm.acompletion(model="hosted_vllm/test", **settings))
    provider.assert_called_once()
    assert rows(path)[0]["outcome"] == "cancelled"


def test_batch_retries_only_the_failed_item(tmp_path, monkeypatch):
    """Keep a completed batch item while another request recovers independently."""
    calls = {"good": 0, "flaky": 0}

    def complete(**kwargs):
        """Fail only the first two attempts for the flaky item."""
        key = kwargs["messages"][0]["content"]
        calls[key] += 1
        if key == "flaky" and calls[key] < 3:
            raise ConnectionError("temporary")
        return response()

    monkeypatch.setattr(litellm, "completion", complete)
    monkeypatch.setattr(litellm, "completion_cost", Mock(return_value=0))
    monkeypatch.setattr(provider_retries.time, "sleep", Mock())
    path = tmp_path / "attempts.jsonl"
    lm = LM("hosted_vllm/test", **provider_retry_kwargs(path, "optimizer"))
    assert lm.batch_complete([[{"role": "user", "content": key}] for key in calls]) == ["done", "done"]
    assert calls == {"good": 1, "flaky": 3}
    assert len(rows(path)) == 4


def test_each_retry_uses_the_original_request(tmp_path, monkeypatch):
    """Discard provider-side mutations before retrying the same model input."""
    seen = []

    def complete(**kwargs):
        """Mutate the first attempt and fail it after saving what was received."""
        seen.append(kwargs["messages"][0]["content"])
        kwargs["messages"][0]["content"] = "mutated"
        if len(seen) == 1:
            raise ConnectionError("temporary")
        return response()

    monkeypatch.setattr(litellm, "completion", complete)
    monkeypatch.setattr(provider_retries.time, "sleep", Mock())
    settings = provider_retry_kwargs(tmp_path / "attempts.jsonl")
    messages = [{"role": "user", "content": "original"}]
    litellm.completion(model="hosted_vllm/test", messages=messages, **settings)
    assert seen == ["original", "original"]
    assert messages[0]["content"] == "original"


def test_unmarked_requests_are_unchanged(monkeypatch):
    """Leave other library users' requests and retry semantics alone."""
    provider = Mock(return_value=response())
    monkeypatch.setattr(litellm, "completion", provider)
    install_provider_retries()
    litellm.completion(model="other", messages=[], num_retries=5)
    provider.assert_called_once_with(model="other", messages=[], num_retries=5)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_real_sdk_makes_three_http_attempts_without_nested_retries(tmp_path, monkeypatch, asynchronous):
    """Exercise real LiteLLM and SDK transports with offline HTTP responses."""
    sent = []

    def send(client, request, **kwargs):
        """Fail every HTTP request without touching the network."""
        sent.append(request)
        return httpx.Response(
            503, json={"error": {"message": "offline failure", "type": "server_error"}}, request=request
        )

    async def asend(client, request, **kwargs):
        """Return the same offline failure through the async SDK path."""
        return send(client, request, **kwargs)

    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.setattr(httpx.AsyncClient, "send", asend)
    monkeypatch.setattr(provider_retries.time, "sleep", Mock())
    monkeypatch.setattr(provider_retries.asyncio, "sleep", AsyncMock())
    path = tmp_path / "attempts.jsonl"
    settings = provider_retry_kwargs(path)
    request = {
        "model": "hosted_vllm/test",
        "api_key": "offline",
        "api_base": "http://127.0.0.1:1/v1",
        "messages": [{"role": "user", "content": "test"}],
        **settings,
    }
    with pytest.raises(ProviderRequestError):
        if asynchronous:
            asyncio.run(litellm.acompletion(**request))
        else:
            litellm.completion(**request)
    assert len(sent) == len(rows(path)) == 3
    assert all(PROVIDER_RETRY_KEY not in request.content.decode() for request in sent)
