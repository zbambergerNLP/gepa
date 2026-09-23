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
from gepa.lm import LM, LMRequestExhaustedError
from gepa.proposer.reflective_mutation.reflection_lm import StatelessReflectionLM
from gepa.response_journal import response_journal_scope


@pytest.fixture(autouse=True)
def fixed_jitter(monkeypatch):
    """Make timing assertions reproducible without using the optimizer RNG."""
    monkeypatch.setattr(provider_retries._JITTER, "uniform", lambda _start, end: end)


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
def test_initial_attempt_plus_three_retries_with_backoff_and_usage(tmp_path, monkeypatch, asynchronous, succeeds):
    """Bound consecutive connection failures and retain every attempt's evidence."""
    raw = response()
    outcomes = [
        ConnectionError("private diagnostic"),
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
    assert provider.call_count == 4
    assert [call.args[0] for call in sleep.call_args_list] == [1.0, 2.0, 4.0]
    for call in provider.call_args_list:
        assert call.kwargs["num_retries"] == call.kwargs["max_retries"] == 0
        assert PROVIDER_RETRY_KEY not in call.kwargs
    records = rows(path)
    assert [record["attempt"] for record in records] == [1, 2, 3, 4]
    assert len({record["request_id"] for record in records}) == 1
    assert [record["will_retry"] for record in records] == [True, True, True, False]
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
    expected = 4 if status in {408, 429, 500, 502, 503, 504} else 1
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
@pytest.mark.parametrize("finish_reason,content", [("length", None), ("stop", None), ("length", "unfinished")])
def test_incomplete_model_output_recovers_and_preserves_failed_attempt(
    tmp_path, monkeypatch, asynchronous, finish_reason, content
):
    """Recover from the observed cutoff while retaining its raw output and cost."""
    raw = litellm.ModelResponse(
        choices=[
            {
                "message": {"role": "assistant", "content": content, "reasoning_content": "unfinished reasoning"},
                "finish_reason": finish_reason,
            }
        ],
        usage={
            "prompt_tokens": 260,
            "completion_tokens": 65536,
            "total_tokens": 65796,
            "completion_tokens_details": {"reasoning_tokens": 65536},
        },
    )
    good = response()
    provider = AsyncMock(side_effect=[raw, good]) if asynchronous else Mock(side_effect=[raw, good])
    monkeypatch.setattr(litellm, "acompletion" if asynchronous else "completion", provider)
    monkeypatch.setattr(provider_retries.time, "sleep", Mock())
    monkeypatch.setattr(provider_retries.asyncio, "sleep", AsyncMock())
    path = tmp_path / "provider-attempts.jsonl"
    kwargs = {
        "model": "hosted_vllm/test",
        "seed": 0,
        "api_key": "secret-key",
        "extra_headers": {"Authorization": "secret-header"},
        "messages": [{"role": "user", "content": "captured training input"}],
        "extra_body": {"thinking_token_budget": 32768, "api_key": "secret-body"},
        **provider_retry_kwargs(path),
    }
    result = asyncio.run(litellm.acompletion(**kwargs)) if asynchronous else litellm.completion(**kwargs)
    assert result is good
    assert provider.call_count == 2
    row, recovered = rows(path)
    assert row["will_retry"] is True
    assert row["outcome"] == "error" and row["transport_outcome"] == "success"
    assert row["response_error"] == ("output_length" if finish_reason == "length" else "empty_completion")
    assert [item["seed"] for item in (row, recovered)] == [0, 1]
    assert [call.kwargs["seed"] for call in provider.call_args_list] == [0, 1]
    assert kwargs["seed"] == 0
    assert row["empty_completion"] is (content is None)
    assert row["thinking_token_budget"] == 32768
    assert row["thinking_budget_reached"] is True
    artifact = Path(row["response_artifact"])
    saved = json.loads(artifact.read_text())
    assert saved["request"]["messages"] == kwargs["messages"]
    assert saved["response"]["choices"][0]["message"]["reasoning_content"] == "unfinished reasoning"
    assert "secret-" not in artifact.read_text()
    assert artifact.stat().st_mode & 0o777 == 0o600


def test_native_tool_call_is_not_mistaken_for_empty_completion(tmp_path, monkeypatch):
    """Accept content-free native tools without generating spurious failure artifacts."""
    raw = litellm.ModelResponse(
        choices=[
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "echo", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    )
    monkeypatch.setattr(litellm, "completion", Mock(return_value=raw))
    path = tmp_path / "provider-attempts.jsonl"
    settings = provider_retry_kwargs(path)
    assert litellm.completion(model="hosted_vllm/test", **settings) is raw
    assert rows(path)[0]["empty_completion"] is False
    assert not (tmp_path / "provider-failures").exists()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_real_sdk_makes_four_http_attempts_without_nested_retries(tmp_path, monkeypatch, asynchronous):
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
    assert len(sent) == len(rows(path)) == 4
    assert all(PROVIDER_RETRY_KEY not in request.content.decode() for request in sent)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_transport_and_incomplete_responses_share_one_retry_budget(tmp_path, monkeypatch, asynchronous):
    """Keep retries bounded across mixed failure classes and vary only failed-output seeds."""
    unfinished = litellm.ModelResponse(
        choices=[{"message": {"role": "assistant", "content": None}, "finish_reason": "length"}],
        usage={"prompt_tokens": 7430, "completion_tokens": 131072, "total_tokens": 138502},
    )
    outcomes = [ConnectionError("temporary"), unfinished, ConnectionError("temporary"), unfinished]
    provider = AsyncMock(side_effect=outcomes) if asynchronous else Mock(side_effect=outcomes)
    monkeypatch.setattr(litellm, "acompletion" if asynchronous else "completion", provider)
    monkeypatch.setattr(provider_retries.time, "sleep", Mock())
    monkeypatch.setattr(provider_retries.asyncio, "sleep", AsyncMock())
    path = tmp_path / "attempts.jsonl"
    kwargs = {"model": "hosted_vllm/test", "seed": 0, **provider_retry_kwargs(path, "controller")}
    with pytest.raises(LMRequestExhaustedError):
        if asynchronous:
            asyncio.run(litellm.acompletion(**kwargs))
        else:
            litellm.completion(**kwargs)
    records = rows(path)
    assert provider.call_count == len(records) == 4
    assert [record["seed"] for record in records] == [0, 0, 1, 1]
    assert [record["empty_completion"] for record in records] == [False, True, False, True]
    assert [record["will_retry"] for record in records] == [True, True, True, False]
    assert sum(record["completion_tokens"] or 0 for record in records) == 262144
    assert len(list((tmp_path / "provider-failures").glob("*.json"))) == 2


@pytest.mark.parametrize("content", [None, "", "  \n "])
def test_empty_output_exhausts_four_attempts(tmp_path, monkeypatch, content):
    """Never pass empty successful HTTP responses onward as a task answer."""
    raw = litellm.ModelResponse(
        choices=[{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]
    )
    provider = Mock(return_value=raw)
    monkeypatch.setattr(litellm, "completion", provider)
    monkeypatch.setattr(provider_retries.time, "sleep", Mock())
    path = tmp_path / "attempts.jsonl"
    settings = provider_retry_kwargs(path)
    with pytest.raises(ProviderRequestError):
        litellm.completion(model="hosted_vllm/test", seed=0, **settings)
    assert [record["seed"] for record in rows(path)] == [0, 1, 2, 3]
    assert provider.call_count == 4


def test_missing_choices_is_retryable(tmp_path, monkeypatch):
    """Reject an incomplete response envelope before indexing its first choice."""
    provider = Mock(side_effect=[{"choices": []}, response()])
    monkeypatch.setattr(litellm, "completion", provider)
    monkeypatch.setattr(provider_retries.time, "sleep", Mock())
    path = tmp_path / "attempts.jsonl"
    settings = provider_retry_kwargs(path)
    litellm.completion(model="hosted_vllm/test", **settings)
    assert rows(path)[0]["response_error"] == "missing_choices"
    assert provider.call_count == 2


@pytest.mark.parametrize(
    "finish,content,refusal",
    [
        ("stop", "<finish>No applicable edit.</finish>", None),
        ("stop", "incorrect answer", None),
        ("content_filter", None, None),
        ("stop", None, "refused"),
    ],
)
def test_valid_noops_answers_and_refusals_are_not_resampled(tmp_path, monkeypatch, finish, content, refusal):
    """Keep scientific outcomes and provider refusals outside recovery sampling."""
    raw = {"choices": [{"finish_reason": finish, "message": {"content": content, "refusal": refusal}}]}
    provider = Mock(return_value=raw)
    monkeypatch.setattr(litellm, "completion", provider)
    settings = provider_retry_kwargs(tmp_path / "attempts.jsonl")
    assert litellm.completion(model="hosted_vllm/test", **settings) is raw
    provider.assert_called_once()


@pytest.mark.parametrize("role", ["controller", "manifestor", "editor", "vanilla"])
def test_every_optimizer_role_recovers_once_and_journals_only_usable_output(tmp_path, monkeypatch, role):
    """Apply recovery before journaling while replaying the exact accepted response."""
    bad = litellm.ModelResponse(
        choices=[{"message": {"role": "assistant", "content": None}, "finish_reason": "length"}]
    )
    provider = Mock(side_effect=[bad, response()])
    monkeypatch.setattr(litellm, "completion", provider)
    monkeypatch.setattr(litellm, "completion_cost", Mock(return_value=0))
    monkeypatch.setattr(provider_retries.time, "sleep", Mock())
    path = tmp_path / "attempts.jsonl"
    kwargs = {
        "seed": 0,
        "response_journal_path": str(tmp_path / "responses.sqlite"),
        "response_journal_namespace": role,
        **provider_retry_kwargs(path, role),
    }
    for _ in range(2):
        lm = LM("hosted_vllm/test", **kwargs)
        with response_journal_scope("iteration-1"):
            if role == "editor":
                assert lm.complete_with_tools([{"role": "user", "content": "edit"}], []).content == "done"
            else:
                assert lm("proposal") == "done"
    assert provider.call_count == len(rows(path)) == 2
    assert {record["role"] for record in rows(path)} == {role}


def test_exhausted_batch_cannot_restart_at_the_reflection_layer(tmp_path, monkeypatch):
    """Prevent four provider attempts from multiplying during batch fallback."""
    provider = Mock(side_effect=ConnectionError("temporary"))
    monkeypatch.setattr(litellm, "completion", provider)
    monkeypatch.setattr(provider_retries.time, "sleep", Mock())
    path = tmp_path / "attempts.jsonl"
    strategy = StatelessReflectionLM(LM("hosted_vllm/test", **provider_retry_kwargs(path)))
    jobs = [({"c": value}, {"c": [{"feedback": "failure"}]}, ["c"]) for value in ("one", "two")]
    with pytest.raises(LMRequestExhaustedError):
        list(strategy.reflect_many(jobs))
    records = rows(path)
    assert provider.call_count == len(records) == 8
    assert len({record["request_id"] for record in records}) == 2
    assert [record["attempt"] for record in records].count(4) == 2
