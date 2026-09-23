"""Tests for the one-exchange HotPotQA serving smoke test."""

import io
import json
import sys
from pathlib import Path
from urllib.error import HTTPError

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from examples.common.experiment_models import DEEPSEEK_V4_1_FLASH_MODEL, experiment_decoding
from examples.common import provider_retries
from examples.hotpotqa import smoke_serving

LOCAL_API_BASE = "http://127.0.0.1:8000/v1"
SERVED_NAME = "deepseek-ai/DeepSeek-V4.1-Flash"


def test_chat_request_matches_the_body_the_campaign_client_sends() -> None:
    """Lift chat-template arguments to the top level and drop client-only settings."""
    body = smoke_serving.build_chat_request(DEEPSEEK_V4_1_FLASH_MODEL, SERVED_NAME, LOCAL_API_BASE)

    assert body["model"] == SERVED_NAME
    assert body["messages"] == smoke_serving.SMOKE_MESSAGES
    assert body["chat_template_kwargs"] == {"thinking": True, "reasoning_effort": 75}
    assert body["seed"] == 0
    expected = experiment_decoding(DEEPSEEK_V4_1_FLASH_MODEL, agentic=False)
    expected["max_tokens"] = 65_536
    for field, value in expected.items():
        assert body[field] == value
    for client_only in ("api_base", "num_retries", "max_retries", "model_info", "timeout", "extra_body"):
        assert client_only not in body


def test_smoke_exchange_records_rendered_prompt_reasoning_and_content(monkeypatch) -> None:
    """Record the server-rendered prompt and both parts of the response.

    Args:
        monkeypatch: Pytest fixture used to replace the HTTP calls.
    """
    posts = []

    def post_json(url: str, payload: dict, timeout: float, **kwargs) -> dict:
        """Answer each local endpoint with a fixed payload.

        Args:
            url: Requested endpoint.
            payload: Request body.
            timeout: Per-call timeout.

        Returns:
            Fixed endpoint response.
        """
        posts.append((url, payload))
        if url.endswith("/tokenize"):
            return {"tokens": [1, 2, 3]}
        if url.endswith("/detokenize"):
            return {"prompt": "<rendered prompt>"}
        return {
            "choices": [
                {"message": {"content": "Paris", "reasoning": "France's capital is Paris."}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 9},
        }

    def get_json(url: str, timeout: float) -> dict:
        """Answer the read-only endpoints.

        Args:
            url: Requested endpoint.
            timeout: Per-call timeout.

        Returns:
            Fixed endpoint response.
        """
        return {"version": "0.1.1.dev5"} if url.endswith("/version") else {"data": [{"id": SERVED_NAME}]}

    monkeypatch.setattr(smoke_serving, "_post_json", post_json)
    monkeypatch.setattr(smoke_serving, "_get_json", get_json)

    transcript = smoke_serving.run_smoke_exchange(DEEPSEEK_V4_1_FLASH_MODEL, SERVED_NAME, LOCAL_API_BASE, 60)

    assert [url for url, _ in posts] == [
        "http://127.0.0.1:8000/tokenize",
        "http://127.0.0.1:8000/detokenize",
        "http://127.0.0.1:8000/v1/chat/completions",
    ]
    assert posts[0][1]["chat_template_kwargs"] == {"thinking": True, "reasoning_effort": 75}
    assert transcript["rendered_prompt"] == "<rendered prompt>"
    assert transcript["rendered_prompt_token_count"] == 3
    assert transcript["reasoning"] == "France's capital is Paris."
    assert transcript["content"] == "Paris"
    markdown = smoke_serving.render_markdown(transcript)
    assert "<rendered prompt>" in markdown
    assert "France's capital is Paris." in markdown


@pytest.mark.parametrize("succeeds", [True, False])
def test_raw_serving_probes_share_recovery_and_preserve_native_evidence(tmp_path, monkeypatch, succeeds):
    """Bound mixed raw HTTP/output failures and retain response token IDs."""
    cutoff = {"choices": [{"finish_reason": "length", "message": {"content": None}}]}
    complete = {"choices": [{"finish_reason": "stop", "message": {"content": "Paris"}, "token_ids": [1, 2]}]}
    outcomes = [HTTPError("http://local", 503, "unavailable", {}, None), cutoff, cutoff, complete if succeeds else cutoff]
    sent = []

    def send(request, **kwargs):
        sent.append(json.loads(request.data))
        result = outcomes.pop(0)
        if isinstance(result, Exception):
            raise result
        return io.BytesIO(json.dumps(result).encode())

    monkeypatch.setattr(smoke_serving.urllib.request, "urlopen", send)
    monkeypatch.setattr(provider_retries.time, "sleep", lambda _: None)
    path = tmp_path / "provider-attempts.jsonl"
    payload = {"model": "local", "seed": 0, "thinking_token_budget": 0, "return_token_ids": True}
    if succeeds:
        result = smoke_serving._post_json("http://local/v1/chat/completions", payload, 60, attempt_log=path)
        assert result["choices"][0]["token_ids"] == [1, 2]
    else:
        with pytest.raises(provider_retries.ProviderRequestError):
            smoke_serving._post_json("http://local/v1/chat/completions", payload, 60, attempt_log=path)
    assert [row["seed"] for row in sent] == [0, 0, 1, 2]
    assert all(row.keys() == payload.keys() and row["thinking_token_budget"] == 0 for row in sent)
    attempts = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(attempts) == 4
    assert attempts[0]["status_code"] == 503
    assert attempts[1]["response_error"] == "output_length"
    assert attempts[-1]["will_retry"] is False
