"""Bound and record provider attempts for the reviewed benchmark campaigns."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import random
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError, URLError

import httpx
import litellm
from litellm.exceptions import APIConnectionError, AuthenticationError, BadRequestError, ContextWindowExceededError

from gepa.lm import LMRequestExhaustedError

PROVIDER_RETRY_KEY = "_gepa_provider_retry"
PROVIDER_RETRY_POLICY = {
    "version": 2,
    "max_retries": 3,
    "max_attempts": 4,
    "backoff_seconds": [1.0, 2.0, 4.0],
    "backoff_jitter": "uniform_zero_to_backoff",
    "retryable_http_statuses": [408, 429, 500, 502, 503, 504],
    "retryable_errors": "connection_or_transport_timeout",
    "retryable_responses": ["output_length", "empty_completion", "missing_choices"],
    "incomplete_response_seed": "advance_explicit_seed_by_one_modulo_2**32",
    "semantic_retries": False,
    "sdk_retries": 0,
    "timeout_scope": "all_attempts_share_explicit_request_timeout",
    "attempt_log": "provider-attempts.jsonl",
    "missing_usage": None,
}
_WRITE_LOCK = threading.Lock()
_JITTER = random.SystemRandom()


class ProviderRequestError(LMRequestExhaustedError):
    """Stop higher-level retries after a provider request fails its policy."""


class IncompleteResponseError(RuntimeError):
    """Reject a transport-successful response that cannot complete the request."""

    def __init__(self, reason: str):
        """Retain the machine-readable failure without embedding model output."""
        super().__init__(reason)
        self.reason = reason


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read a response attribute from either a mapping or a provider object."""
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def is_provider_request_error(error: BaseException) -> bool:
    """Recognize exhausted requests through Harbor's exception translation."""
    seen: set[int] = set()
    while id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, ProviderRequestError):
            return True
        cause = error.__cause__
        if cause is None:
            return False
        error = cause
    return False


def _retryable(error: BaseException) -> bool:
    """Accept only temporary HTTP failures and connection/transport timeouts."""
    if isinstance(error, IncompleteResponseError):
        return True
    if isinstance(error, AuthenticationError | BadRequestError):
        return False
    if isinstance(error, HTTPError):
        return error.code in PROVIDER_RETRY_POLICY["retryable_http_statuses"]
    if isinstance(error, URLError):
        return isinstance(error.reason, ConnectionError | TimeoutError)
    status = _field(error, "status_code")
    if status is None:
        status = _field(_field(error, "response"), "status_code")
    if status is not None:
        return status in PROVIDER_RETRY_POLICY["retryable_http_statuses"]
    return isinstance(
        error,
        ConnectionError
        | TimeoutError
        | httpx.NetworkError
        | httpx.TimeoutException
        | httpx.RemoteProtocolError
        | APIConnectionError,
    )


def _response_error(response: Any) -> IncompleteResponseError | None:
    """Reject cutoffs and missing visible output before downstream parsing or tools."""
    choices = _field(response, "choices", [])
    if not choices:
        return IncompleteResponseError("missing_choices")
    for choice in choices:
        if _field(choice, "finish_reason") == "length":
            return IncompleteResponseError("output_length")
        # A provider refusal is a completed decision, not an empty generation
        # that should be resampled to evade the refusal.
        message = _field(choice, "message")
        if _field(choice, "finish_reason") == "content_filter" or _field(message, "refusal"):
            continue
        content = _field(message, "content")
        if not (isinstance(content, str) and content.strip()) and not _field(message, "tool_calls"):
            return IncompleteResponseError("empty_completion")
    return None


def _save_incomplete_response(
    path: Path, request: dict[str, Any], response: Any, request_id: str, attempt: int
) -> None:
    """Retain incomplete model output without copying transport credentials."""
    choices = []
    for choice in _field(response, "choices", []):
        message = _field(choice, "message")
        choices.append(
            {
                "finish_reason": _field(choice, "finish_reason"),
                "message": {
                    name: _field(message, name)
                    for name in ("role", "content", "reasoning_content", "reasoning", "tool_calls")
                },
            }
        )
    payload = {
        "request_id": request_id,
        "attempt": attempt,
        "request": {
            key: request[key]
            for key in ("model", "messages", "tools", "max_tokens", "temperature", "top_p", "top_k", "seed")
            if key in request
        },
        "response": {"id": _field(response, "id"), "model": _field(response, "model"), "choices": choices},
    }
    extra = {**request, **(request.get("extra_body") or {})}
    payload["request"]["extra_body"] = {
        key: extra[key] for key in ("thinking_token_budget", "top_k", "min_p") if key in extra
    }
    payload["request"]["extra_body"]["chat_template_kwargs"] = {
        key: value
        for key, value in extra.get("chat_template_kwargs", {}).items()
        if key in {"enable_thinking", "thinking", "reasoning_effort", "preserve_thinking"}
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            allow_nan=False,
            default=lambda value: value.model_dump(),
        )
        stream.flush()
        os.fsync(stream.fileno())


def _record(
    settings: dict[str, Any],
    request: dict[str, Any],
    request_id: str,
    attempt: int,
    started: float,
    response: Any,
    error: BaseException | None,
    will_retry: bool,
) -> None:
    """Append one secret-free physical-attempt record, retaining unknown usage."""
    if response is None and error is not None:
        response = _field(error, "response")
    usage = _field(response, "usage")
    if usage is None and isinstance(response, httpx.Response):
        try:
            usage = _field(response.json(), "usage")
        except (ValueError, httpx.ResponseNotRead):
            pass
    limits = settings.get("token_limits")
    prompt_tokens = _field(usage, "prompt_tokens")
    completion_tokens = _field(usage, "completion_tokens")
    reasoning_tokens = _field(_field(usage, "completion_tokens_details"), "reasoning_tokens")
    thinking_budget = (request.get("extra_body") or {}).get("thinking_token_budget", request.get("thinking_token_budget"))
    choices = _field(response, "choices", [])
    reasons = [_field(choice, "finish_reason") for choice in choices]
    empty_completion = (error is None or isinstance(error, IncompleteResponseError)) and (
        not choices
        or any(
            not str(_field(_field(choice, "message"), "content") or "").strip()
            and not _field(_field(choice, "message"), "tool_calls")
            for choice in choices
        )
    )
    row = {
        "schema_version": 2,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "allocation_job_id": os.environ.get("SLURM_JOB_ID"),
        "request_id": request_id,
        "attempt": attempt,
        "role": settings["role"],
        "model": request.get("model"),
        "requested_model": request.get("model"),
        "response_model": _field(response, "model"),
        "response_id": _field(response, "id"),
        "elapsed_seconds": time.monotonic() - started,
        "outcome": "success" if error is None else "error" if isinstance(error, Exception) else "cancelled",
        "transport_outcome": (
            "success"
            if error is None or isinstance(error, IncompleteResponseError)
            else "error"
            if isinstance(error, Exception)
            else "cancelled"
        ),
        "response_error": error.reason if isinstance(error, IncompleteResponseError) else None,
        "seed": request.get("seed"),
        "error_type": type(error).__name__ if error is not None else None,
        "status_code": _field(error, "status_code", _field(error, "code", _field(response, "status_code"))),
        "will_retry": will_retry,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cost_usd": _field(_field(response, "_hidden_params"), "response_cost"),
        "finish_reasons": reasons,
        "empty_completion": empty_completion,
        "thinking_token_budget": thinking_budget,
        "thinking_budget_reached": (
            reasoning_tokens >= thinking_budget
            if thinking_budget is not None and thinking_budget >= 0 and reasoning_tokens is not None
            else None
        ),
        "length_finish": "length" in reasons if any(reason is not None for reason in reasons) else None,
        "output_cap_reached": (
            completion_tokens >= limits["max_output_tokens"]
            if limits is not None and completion_tokens is not None
            else None
        ),
        "context_cap_reached": (
            prompt_tokens + completion_tokens >= limits["context_tokens"]
            if limits is not None and prompt_tokens is not None and completion_tokens is not None
            else None
        ),
        "limits": limits,
    }
    path = settings.get("log_path")
    if (
        path is not None
        and (error is None or isinstance(error, IncompleteResponseError))
        and (empty_completion or "length" in reasons)
    ):
        artifact = Path(path).parent / "provider-failures" / f"{request_id}-{attempt}.json"
        _save_incomplete_response(artifact, request, response, request_id, attempt)
        row["response_artifact"] = str(artifact)
    line = json.dumps(row, sort_keys=True, allow_nan=False)
    if path is None:
        logging.getLogger(__name__).warning("Provider attempt: %s", line)
        return
    destinations = {Path(path)}
    if settings.get("token_usage_log") is not None:
        destinations.add(Path(settings["token_usage_log"]))
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with _WRITE_LOCK, destination.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def _request(request: dict[str, Any], deadline: float | None) -> dict[str, Any]:
    """Disable nested SDK retries and carry the remaining explicit timeout."""
    result = {**request, "num_retries": 0, "max_retries": 0}
    for field in ("messages", "tools", "extra_body"):
        if field in result:
            result[field] = deepcopy(result[field])
    if deadline is not None:
        remaining = max(deadline - time.monotonic(), 0.001)
        for field in ("timeout", "request_timeout"):
            if field in result:
                result[field] = remaining
    return result


def _deadline(request: dict[str, Any]) -> float | None:
    """Share an explicit numeric timeout across attempts and backoff."""
    values = [request.get(field) for field in ("timeout", "request_timeout")]
    seconds = [float(value) for value in values if isinstance(value, float | int) and not isinstance(value, bool)]
    return time.monotonic() + min(seconds) if seconds else None


def _delay(error: BaseException, attempt: int, deadline: float | None) -> float | None:
    """Choose a permitted backoff only while another attempt fits the deadline."""
    if attempt >= PROVIDER_RETRY_POLICY["max_attempts"] or not _retryable(error):
        return None
    delay = _JITTER.uniform(0, PROVIDER_RETRY_POLICY["backoff_seconds"][attempt - 1])
    if deadline is not None and time.monotonic() + delay >= deadline:
        return None
    return delay


def _advance_output_retry_seed(request: dict[str, Any], error: BaseException) -> None:
    """Avoid deterministic regeneration of an unusable completion, retaining transport seeds."""
    seed = request.get("seed")
    if isinstance(error, IncompleteResponseError) and isinstance(seed, int) and not isinstance(seed, bool):
        request["seed"] = (seed + 1) % (2**32)


def complete_with_retries(
    send: Callable[..., Any], request: dict[str, Any], settings: dict[str, Any]
) -> Any:
    """Apply the same bounded request policy to SDK calls and raw serving probes."""
    request = dict(request)
    request_id, deadline = str(uuid.uuid4()), _deadline(request)
    last_error: Exception | None = None
    for attempt in range(1, PROVIDER_RETRY_POLICY["max_attempts"] + 1):
        if attempt > 1 and deadline is not None and time.monotonic() >= deadline:
            raise ProviderRequestError("Provider request timeout exhausted during backoff.") from last_error
        started = time.monotonic()
        response = None
        try:
            response = send(**_request(request, deadline))
            response_error = _response_error(response)
            if response_error is not None:
                raise response_error
        except BaseException as error:
            delay = _delay(error, attempt, deadline) if isinstance(error, Exception) else None
            _record(settings, request, request_id, attempt, started, response, error, delay is not None)
            if not isinstance(error, Exception):
                raise
            last_error = error
            if delay is None:
                if isinstance(error, ContextWindowExceededError):
                    raise
                raise ProviderRequestError("Provider request failed; inspect provider-attempts.jsonl.") from error
            _advance_output_retry_seed(request, error)
            time.sleep(delay)
        else:
            _record(settings, request, request_id, attempt, started, response, None, False)
            return response


def install_provider_retries() -> None:
    """Install opt-in transport wrappers without changing unmarked requests.

    The marker is consumed before LiteLLM or provider code sees the request.
    Batch workers and DSPy's pinned client also dispatch through these public
    completion functions. Installation is idempotent; importing this module
    alone does not modify LiteLLM.
    """
    if getattr(litellm.completion, "_gepa_retry_wrapper", False) is not True:
        original = litellm.completion

        @functools.wraps(original)
        def completion(*args: Any, **kwargs: Any) -> Any:
            """Retry one marked synchronous request without reissuing a batch."""
            settings = kwargs.pop(PROVIDER_RETRY_KEY, None)
            if settings is None:
                return original(*args, **kwargs)
            return complete_with_retries(functools.partial(original, *args), kwargs, settings)

        cast(Any, completion)._gepa_retry_wrapper = True
        litellm.completion = completion

    if getattr(litellm.acompletion, "_gepa_retry_wrapper", False) is not True:
        original_async = litellm.acompletion

        @functools.wraps(original_async)
        async def acompletion(*args: Any, **kwargs: Any) -> Any:
            """Retry one marked asynchronous request while preserving cancellation."""
            settings = kwargs.pop(PROVIDER_RETRY_KEY, None)
            if settings is None:
                return await original_async(*args, **kwargs)
            request_id, deadline = str(uuid.uuid4()), _deadline(kwargs)
            last_error: Exception | None = None
            for attempt in range(1, PROVIDER_RETRY_POLICY["max_attempts"] + 1):
                if attempt > 1 and deadline is not None and time.monotonic() >= deadline:
                    raise ProviderRequestError("Provider request timeout exhausted during backoff.") from last_error
                started = time.monotonic()
                response = None
                try:
                    response = await original_async(*args, **_request(kwargs, deadline))
                    response_error = _response_error(response)
                    if response_error is not None:
                        raise response_error
                except BaseException as error:
                    delay = _delay(error, attempt, deadline) if isinstance(error, Exception) else None
                    _record(settings, kwargs, request_id, attempt, started, response, error, delay is not None)
                    if not isinstance(error, Exception):
                        raise
                    last_error = error
                    if delay is None:
                        if isinstance(error, ContextWindowExceededError):
                            raise
                        raise ProviderRequestError(
                            "Provider request failed; inspect provider-attempts.jsonl."
                        ) from error
                    _advance_output_retry_seed(kwargs, error)
                    await asyncio.sleep(delay)
                else:
                    _record(settings, kwargs, request_id, attempt, started, response, None, False)
                    return response

        cast(Any, acompletion)._gepa_retry_wrapper = True
        litellm.acompletion = acompletion


def provider_retry_kwargs(log_path: Path | None = None, role: str = "model") -> dict[str, Any]:
    """Enable the fixed policy and return serializable, local-only request metadata."""
    install_provider_retries()
    return {
        "num_retries": 0,
        "max_retries": 0,
        PROVIDER_RETRY_KEY: {"log_path": str(log_path) if log_path is not None else None, "role": role},
    }
