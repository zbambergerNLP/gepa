# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Thin LM abstraction over LiteLLM that handles retries, truncation
warnings, and cross-model compatibility.

Usage::

    from gepa.lm import LM

    lm = LM("openai/gpt-4.1", temperature=0.7, max_tokens=4096)
    response: str = lm("Solve this problem...")

    # Also works with chat messages
    response = lm([{"role": "user", "content": "Hello"}])

The returned callable conforms to the ``LanguageModel`` protocol
(``(str | list[dict]) -> str``) used throughout GEPA.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from gepa.response_journal import (
    ACTIVE_RESPONSE_JOURNAL_SCOPE,
    ResumeResponseJournal,
    canonical_request_digest,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NativeToolCall:
    """One provider-native function call returned by a language model."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ToolCompletion:
    """Assistant content, native function calls, and provider reasoning state."""

    content: str
    tool_calls: tuple[NativeToolCall, ...]
    reasoning_content: str = ""


def _response_value(value: Any, key: str, default: Any = None) -> Any:
    """Read a response field through LiteLLM's object-or-mapping boundary.

    Args:
        value: Response object or mapping that may contain the field.
        key: Field name to read.
        default: Value returned when the field is absent.

    Returns:
        The mapping value or object attribute when present, otherwise
        ``default``.
    """
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


@dataclass(frozen=True)
class ProviderResponseIdentity:
    """Model identity returned by a completion provider."""

    model: str | None
    system_fingerprint: str | None


class ProviderIdentityMismatchError(RuntimeError):
    """Signal that a provider response differs from the launch-time identity."""


class LMProviderError(RuntimeError):
    """Signal that the configured completion provider failed a request."""


def provider_response_identity(completion: Any) -> ProviderResponseIdentity:
    """Extract the provider model and system fingerprint from one response.

    LiteLLM normally exposes both fields on the top-level response. The
    ``_hidden_params`` fallback retains compatibility with providers whose
    adapter stores the fingerprint outside the normalized response schema.

    Args:
        completion: LiteLLM response object or equivalent mapping.

    Returns:
        Provider response identity with ``None`` for missing or non-string
        fields.
    """
    model_value = _response_value(completion, "model")
    fingerprint_value = _response_value(completion, "system_fingerprint")
    if not isinstance(fingerprint_value, str) or not fingerprint_value:
        hidden_params = _response_value(completion, "_hidden_params", {})
        fingerprint_value = _response_value(hidden_params, "system_fingerprint")
    model = model_value if isinstance(model_value, str) and model_value else None
    system_fingerprint = (
        fingerprint_value if isinstance(fingerprint_value, str) and fingerprint_value else None
    )
    return ProviderResponseIdentity(model=model, system_fingerprint=system_fingerprint)


def validate_provider_response_identity(
    completion: Any,
    expected_model: str,
    expected_system_fingerprint: str,
) -> ProviderResponseIdentity:
    """Require one response to match the identity captured at launch.

    Args:
        completion: LiteLLM response object or equivalent mapping.
        expected_model: Exact provider-returned model captured by preflight.
        expected_system_fingerprint: Exact provider fingerprint captured by
            preflight.

    Returns:
        Validated response identity.

    Raises:
        ValueError: Either expected identity field is empty.
        ProviderIdentityMismatchError: The response omits or changes either
            provider identity field.
    """
    if not expected_model or not expected_system_fingerprint:
        raise ValueError("Expected provider model and system fingerprint must both be non-empty.")
    identity = provider_response_identity(completion)
    if identity.model != expected_model or identity.system_fingerprint != expected_system_fingerprint:
        raise ProviderIdentityMismatchError(
            "Provider response identity changed during the run: "
            f"expected model={expected_model!r}, system_fingerprint={expected_system_fingerprint!r}; "
            f"received model={identity.model!r}, system_fingerprint={identity.system_fingerprint!r}."
        )
    return identity


class InlineReasoningError(ValueError):
    """Raised when a leading inline reasoning block is malformed or truncated."""


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _starts_think_tag_shape(text: str, cursor: int, *, closing: bool = False) -> bool:
    """Recognize the prefix shape of an opening or closing reasoning tag.

    This deliberately accepts malformed suffixes so callers can distinguish
    ordinary answer text from a broken ``<think>`` protocol.

    Args:
        text: Model response being inspected.
        cursor: Character offset at which a tag may begin.
        closing: Whether to look for a closing rather than opening tag.

    Returns:
        Whether the text at ``cursor`` has the requested tag's prefix shape.
    """
    prefix = "</think" if closing else "<think"
    if not text.startswith(prefix, cursor):
        return False
    suffix = cursor + len(prefix)
    return suffix == len(text) or text[suffix].isspace() or text[suffix] in "/>"


def _answer_from_inline_reasoning(text: str) -> str:
    """Remove complete leading reasoning blocks without rewriting the answer.

    Consecutive leading ``<think>`` blocks are discarded. Text that does not
    begin with the reasoning protocol is returned byte-for-byte.

    Args:
        text: Raw assistant content from an inline-reasoning model.

    Returns:
        Content following the last valid leading reasoning block.

    Raises:
        InlineReasoningError: A leading tag is malformed, nested, unexpected,
            or truncated before its closing tag.
    """
    first_content = 0
    while first_content < len(text) and text[first_content].isspace():
        first_content += 1
    if text.startswith(_THINK_CLOSE, first_content):
        raise InlineReasoningError("Inline reasoning response starts with an unexpected closing tag.")
    if _starts_think_tag_shape(text, first_content, closing=True):
        raise InlineReasoningError("Inline reasoning response starts with a malformed closing tag.")
    if not text.startswith(_THINK_OPEN, first_content):
        if _starts_think_tag_shape(text, first_content):
            raise InlineReasoningError("Inline reasoning response starts with a malformed opening tag.")
        return text

    cursor = first_content
    while True:
        body_start = cursor + len(_THINK_OPEN)
        close = text.find(_THINK_CLOSE, body_start)
        if close == -1:
            raise InlineReasoningError("Inline reasoning response ended before its closing tag.")
        nested = text.find(_THINK_OPEN, body_start)
        if nested != -1 and nested < close:
            raise InlineReasoningError("Inline reasoning response contains a nested opening tag.")

        cursor = close + len(_THINK_CLOSE)
        next_content = cursor
        while next_content < len(text) and text[next_content].isspace():
            next_content += 1
        if text.startswith(_THINK_OPEN, next_content):
            cursor = next_content
            continue
        if text.startswith(_THINK_CLOSE, next_content):
            raise InlineReasoningError("Inline reasoning response contains an unexpected closing tag.")
        if _starts_think_tag_shape(text, next_content, closing=True) or _starts_think_tag_shape(
            text, next_content
        ):
            raise InlineReasoningError("Inline reasoning response contains a malformed leading tag.")
        return text[cursor:]


@dataclass(eq=False, repr=False)
class InlineReasoningLM:
    """Adapt an LM that embeds leading reasoning tags to answer-only output.

    This wrapper is opt-in for model endpoints that put private reasoning in
    ``<think>...</think>`` blocks inside assistant content. Provider clients
    that already expose reasoning separately should use :class:`LM` directly.

    Args:
        _lm: Callable model, optionally with batch or native-tool methods, whose
            string results should be normalized.
    """

    _lm: Any

    @staticmethod
    def _answer(text: Any) -> str:
        """Validate one wrapped response and expose only its answer content.

        Args:
            text: Value returned by the wrapped model.

        Returns:
            String content with valid leading reasoning blocks removed.

        Raises:
            TypeError: The wrapped model did not return a string.
            InlineReasoningError: The string contains malformed leading
                reasoning markup.
        """
        if not isinstance(text, str):
            raise TypeError("InlineReasoningLM requires wrapped calls to return strings.")
        return _answer_from_inline_reasoning(text)

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        """Run one wrapped completion and return answer-only content.

        Args:
            prompt: Plain prompt or provider-ready chat messages.

        Returns:
            The wrapped model's response with leading reasoning removed.
        """
        return self._answer(self._lm(prompt))

    def supports_cost_tracking(self) -> bool:
        """Report whether the wrapped model exposes real provider spend.

        Returns:
            ``True`` when the wrapped model explicitly supports cost tracking
            or exposes a non-estimated ``total_cost`` value.
        """
        inner_support = getattr(self._lm, "supports_cost_tracking", None)
        if callable(inner_support):
            return bool(inner_support())
        return hasattr(self._lm, "total_cost") and not isinstance(self._lm, TrackingLM)

    def __getattr__(self, name: str) -> Any:
        """Delegate attributes while adapting optional completion interfaces.

        Batch and native-tool completion methods are wrapped so their assistant
        content follows the same reasoning-removal contract as ``__call__``.

        Args:
            name: Attribute requested from the wrapper.

        Returns:
            Adapted optional completion method or the wrapped attribute.

        Raises:
            AttributeError: The requested optional interface is unavailable.
        """
        if name == "complete_with_tools":
            inner = getattr(self._lm, name, None)
            if not callable(inner):
                raise AttributeError(name)

            def complete_with_tools(*args: Any, **kwargs: Any) -> ToolCompletion:
                """Normalize assistant content from one native-tool completion.

                Args:
                    *args: Positional arguments forwarded to the wrapped method.
                    **kwargs: Keyword arguments forwarded to the wrapped method.

                Returns:
                    Tool completion with answer-only assistant content and the
                    original tool calls.

                Raises:
                    TypeError: The wrapped method does not return
                        :class:`ToolCompletion`.
                """
                result = cast(Any, inner)(*args, **kwargs)
                if not isinstance(result, ToolCompletion):
                    raise TypeError("complete_with_tools must return gepa.lm.ToolCompletion.")
                return ToolCompletion(
                    content=self._answer(result.content),
                    tool_calls=result.tool_calls,
                    reasoning_content=result.reasoning_content,
                )

            return complete_with_tools
        if name == "batch_complete":
            inner = getattr(self._lm, name, None)
            if not callable(inner):
                raise AttributeError(name)

            def batch_complete(*args: Any, **kwargs: Any) -> list[str]:
                """Normalize every response from one wrapped batch call.

                Args:
                    *args: Positional arguments forwarded to the wrapped method.
                    **kwargs: Keyword arguments forwarded to the wrapped method.

                Returns:
                    Answer-only responses in the wrapped batch's order.
                """
                return [self._answer(result) for result in cast(Any, inner)(*args, **kwargs)]

            return batch_complete
        return getattr(self._lm, name)


class LM:
    """A lightweight language model wrapper over LiteLLM.

    Handles:

    - **Retries** with exponential backoff via LiteLLM's ``num_retries``.
    - **Truncation detection** — logs a warning when ``finish_reason='length'``.
    - **drop_params=True** so unsupported params are silently ignored
      (with a warning logged for transparency).

    Conforms to the :class:`~gepa.proposer.reflective_mutation.base.LanguageModel`
    protocol, so it can be used anywhere GEPA expects a ``LanguageModel``.

    Args:
        model: LiteLLM model identifier, e.g. ``"openai/gpt-4.1"`` or ``"anthropic/claude-sonnet-4-6"``.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        num_retries: Number of retries on transient failures (default 3).
        expected_response_model: Optional provider-returned model captured by
            a launch preflight. Must be paired with
            ``expected_system_fingerprint``.
        expected_system_fingerprint: Optional provider fingerprint captured by
            a launch preflight. Every response must match it when configured.
        response_journal_path: Optional condition-local SQLite journal used to
            replay completed calls after an interrupted optimizer iteration.
        response_journal_namespace: Stable logical LM role within the journal.
        **kwargs: Extra keyword arguments forwarded to ``litellm.completion``
            (e.g. ``top_p``, ``stop``, ``api_key``, ``api_base``).
    """

    def __init__(
        self,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        num_retries: int = 3,
        expected_response_model: str | None = None,
        expected_system_fingerprint: str | None = None,
        response_journal_path: str | None = None,
        response_journal_namespace: str | None = None,
        **kwargs: Any,
    ):
        """Configure a LiteLLM client and optional exact-response journal.

        Args:
            model: LiteLLM provider and model identifier.
            temperature: Optional sampling temperature.
            max_tokens: Optional completion-token allowance.
            num_retries: Provider retry count.
            expected_response_model: Optional provider-returned model captured
                by launch preflight.
            expected_system_fingerprint: Optional provider fingerprint captured
                by launch preflight.
            response_journal_path: Optional condition-local SQLite journal.
            response_journal_namespace: Stable logical role within the journal.
            **kwargs: Additional LiteLLM completion arguments.

        Raises:
            ValueError: Provider identity or journal settings are only partly
                specified, or an identity field is empty.
        """
        if (expected_response_model is None) != (expected_system_fingerprint is None):
            raise ValueError(
                "expected_response_model and expected_system_fingerprint must be provided together."
            )
        if expected_response_model == "" or expected_system_fingerprint == "":
            raise ValueError("Expected provider identity fields must be non-empty when configured.")
        if (response_journal_path is None) != (response_journal_namespace is None):
            raise ValueError(
                "response_journal_path and response_journal_namespace must be provided together."
            )
        self.model = model
        self.num_retries = num_retries
        self._expected_response_identity = (
            ProviderResponseIdentity(expected_response_model, expected_system_fingerprint)
            if expected_response_model is not None and expected_system_fingerprint is not None
            else None
        )
        self.last_response_identity = ProviderResponseIdentity(None, None)
        self._total_cost: float = 0.0
        self._total_tokens_in: int = 0
        self._total_tokens_out: int = 0
        self._cost_lock = threading.Lock()
        self._response_journal = (
            ResumeResponseJournal(response_journal_path, response_journal_namespace)
            if response_journal_path is not None and response_journal_namespace is not None
            else None
        )
        if self._response_journal is not None:
            (
                self._total_cost,
                self._total_tokens_in,
                self._total_tokens_out,
            ) = self._response_journal.usage_totals()
        self._response_journal_lock = threading.Lock()
        self._response_journal_ordinals: dict[str, int] = {}

        self.completion_kwargs: dict[str, Any] = {
            **({"temperature": temperature} if temperature is not None else {}),
            **({"max_tokens": max_tokens} if max_tokens is not None else {}),
            **kwargs,
        }

    @property
    def total_cost(self) -> float:
        """Cumulative USD cost of all calls made through this LM instance."""
        return self._total_cost

    @property
    def total_tokens_in(self) -> int:
        """Cumulative input (prompt) tokens across all calls."""
        return self._total_tokens_in

    @property
    def total_tokens_out(self) -> int:
        """Cumulative output (completion) tokens across all calls."""
        return self._total_tokens_out

    def _capture_and_validate_response_identity(self, completion: Any) -> None:
        """Capture one response identity and enforce an optional launch pin.

        Args:
            completion: LiteLLM response returned by the provider.

        Raises:
            ProviderIdentityMismatchError: A configured identity is missing or
                differs from the response.
        """
        expected = self._expected_response_identity
        if expected is None:
            identity = provider_response_identity(completion)
        else:
            assert expected.model is not None
            assert expected.system_fingerprint is not None
            identity = validate_provider_response_identity(
                completion,
                expected.model,
                expected.system_fingerprint,
            )
        self.last_response_identity = identity

    def _check_truncation(self, choices: list[Any]) -> None:
        if any(getattr(c, "finish_reason", None) == "length" for c in choices):
            max_tok = self.completion_kwargs.get("max_tokens") or self.completion_kwargs.get("max_completion_tokens")
            logger.warning(
                f"LM response was truncated (finish_reason='length', max_tokens={max_tok}). "
                "Consider increasing max_tokens for better results."
            )

    def _record_completion_usage(self, completion: Any) -> dict[str, float | int]:
        """Accumulate cost and token usage from one LiteLLM completion.

        Cost lookup failures are treated as zero so usage accounting never
        changes completion behavior.

        Args:
            completion: LiteLLM response carrying choices and optional usage.

        Returns:
            JSON-ready cost and token counts for durable replay accounting.
        """
        import litellm

        self._capture_and_validate_response_identity(completion)
        self._check_truncation(completion.choices)
        try:
            cost = litellm.completion_cost(completion_response=completion) or 0.0
        except Exception:
            cost = 0.0

        usage = getattr(completion, "usage", None)
        tokens_in = (getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else 0
        tokens_out = (getattr(usage, "completion_tokens", 0) or 0) if usage is not None else 0

        with self._cost_lock:
            self._total_cost += cost
            self._total_tokens_in += tokens_in
            self._total_tokens_out += tokens_out
        return {
            "cost": float(cost),
            "tokens_in": int(tokens_in),
            "tokens_out": int(tokens_out),
        }

    def _restore_cached_identity(self, value: Any) -> ProviderResponseIdentity:
        """Validate and restore a provider identity from one journal record.

        Args:
            value: Persisted identity mapping.

        Returns:
            Restored provider identity.

        Raises:
            ProviderIdentityMismatchError: The cached identity differs from the
                launch-time provider identity.
            TypeError: The cached identity has an invalid shape.
        """
        if not isinstance(value, Mapping):
            raise TypeError("Response journal contains an invalid provider identity.")
        model = value.get("model")
        system_fingerprint = value.get("system_fingerprint")
        if model is not None and not isinstance(model, str):
            raise TypeError("Response journal contains an invalid provider model.")
        if system_fingerprint is not None and not isinstance(system_fingerprint, str):
            raise TypeError("Response journal contains an invalid provider fingerprint.")
        identity = ProviderResponseIdentity(model=model, system_fingerprint=system_fingerprint)
        expected = self._expected_response_identity
        if expected is not None and identity != expected:
            raise ProviderIdentityMismatchError(
                "Cached provider response identity differs from the launch-time identity: "
                f"expected model={expected.model!r}, system_fingerprint={expected.system_fingerprint!r}; "
                f"recorded model={identity.model!r}, system_fingerprint={identity.system_fingerprint!r}."
            )
        self.last_response_identity = identity
        return identity

    def _restore_journal_result(self, payload: Mapping[str, Any], expected_kind: str) -> Any:
        """Reconstruct one normalized LM result from a journal payload.

        Args:
            payload: Validated journal response mapping.
            expected_kind: Completion interface expected by the caller.

        Returns:
            Plain text, ordered batch texts, or :class:`ToolCompletion`.

        Raises:
            TypeError: The response payload has an invalid shape.
            ValueError: The payload belongs to a different completion interface.
        """
        if payload.get("kind") != expected_kind:
            raise ValueError(
                f"Response journal contains {payload.get('kind')!r}; expected {expected_kind!r}."
            )
        if expected_kind == "completion":
            content = payload.get("content")
            if not isinstance(content, str):
                raise TypeError("Response journal contains invalid completion text.")
            self._restore_cached_identity(payload.get("identity"))
            return content
        if expected_kind == "batch_completion":
            outputs = payload.get("outputs")
            identities = payload.get("identities")
            if (
                not isinstance(outputs, list)
                or not all(isinstance(output, str) for output in outputs)
                or not isinstance(identities, list)
                or len(identities) != len(outputs)
            ):
                raise TypeError("Response journal contains an invalid batch completion.")
            for identity in identities:
                self._restore_cached_identity(identity)
            return outputs
        if expected_kind != "tool_completion":
            raise ValueError(f"Unsupported response-journal result kind {expected_kind!r}.")
        content = payload.get("content")
        reasoning_content = payload.get("reasoning_content")
        tool_calls = payload.get("tool_calls")
        if not isinstance(content, str) or not isinstance(reasoning_content, str) or not isinstance(tool_calls, list):
            raise TypeError("Response journal contains an invalid native-tool completion.")
        restored_calls = []
        for call in tool_calls:
            if not isinstance(call, Mapping):
                raise TypeError("Response journal contains an invalid native tool call.")
            call_id = call.get("id")
            name = call.get("name")
            arguments = call.get("arguments")
            if not isinstance(call_id, str) or not isinstance(name, str) or not isinstance(arguments, str):
                raise TypeError("Response journal contains an invalid native tool call.")
            restored_calls.append(NativeToolCall(id=call_id, name=name, arguments=arguments))
        self._restore_cached_identity(payload.get("identity"))
        return ToolCompletion(
            content=content,
            tool_calls=tuple(restored_calls),
            reasoning_content=reasoning_content,
        )

    def _run_journaled(
        self,
        request: Mapping[str, Any],
        kind: str,
        live_call: Callable[[], tuple[Any, dict[str, Any]]],
    ) -> Any:
        """Replay or durably commit one logical LM call occurrence.

        The per-LM lock is intentionally held through the provider call. It
        keeps call ordinals deterministic; batch concurrency remains inside the
        provider's batch interface and ordinary unscoped calls are unaffected.

        Args:
            request: Exact effective request used only to compute a digest.
            kind: Normalized completion-interface name.
            live_call: Provider call returning the public result and a
                JSON-serializable journal payload.

        Returns:
            Replayed or newly completed public LM result.
        """
        journal = self._response_journal
        scope = ACTIVE_RESPONSE_JOURNAL_SCOPE.get()
        if journal is None or scope is None:
            result, _payload = live_call()
            return result
        request_sha256 = canonical_request_digest(request)
        with self._response_journal_lock:
            ordinal = self._response_journal_ordinals.get(scope, 0)
            cached = journal.load(scope, ordinal, request_sha256)
            if cached is None:
                result, payload = live_call()
                payload["kind"] = kind
                journal.store(scope, ordinal, request_sha256, payload)
            else:
                result = self._restore_journal_result(cached, kind)
            self._response_journal_ordinals[scope] = ordinal + 1
            return result

    def response_journal_cursor_state(self) -> dict[str, int]:
        """Snapshot logical response positions for a same-process retry.

        Returns:
            Independent mapping from restart-stable scope to its next call
            ordinal. An LM without an active journal returns an empty mapping.
        """
        with self._response_journal_lock:
            state = dict(self._response_journal_ordinals)
        return state

    def restore_response_journal_cursor_state(self, state: Mapping[str, int]) -> None:
        """Restore logical response positions before retrying failed work.

        Args:
            state: Snapshot returned by
                :meth:`response_journal_cursor_state`.

        Raises:
            TypeError: A scope or ordinal has the wrong type.
            ValueError: A scope is empty or an ordinal is negative.
        """
        restored: dict[str, int] = {}
        for scope, ordinal in state.items():
            if not isinstance(scope, str) or not isinstance(ordinal, int) or isinstance(ordinal, bool):
                raise TypeError("Response-journal cursor state must map string scopes to integer ordinals.")
            if not scope or ordinal < 0:
                raise ValueError("Response-journal cursor scopes must be non-empty and ordinals non-negative.")
            restored[scope] = ordinal
        with self._response_journal_lock:
            self._response_journal_ordinals = restored

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        """Run one LiteLLM completion and record provider usage.

        Args:
            prompt: Plain user prompt or provider-ready chat messages.

        Returns:
            Assistant message content from the first completion choice.
        """
        import litellm

        if isinstance(prompt, str):
            messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "num_retries": self.num_retries,
            "drop_params": True,
            **self.completion_kwargs,
        }

        def live_call() -> tuple[str, dict[str, Any]]:
            """Complete and normalize one uncached plain request.

            Returns:
                Public text and its secret-free journal payload.

            Raises:
                TypeError: The provider omits string assistant content.
            """
            try:
                completion = litellm.completion(**request)
            except Exception as exc:
                raise LMProviderError(f"Completion provider failed for model {self.model!r}.") from exc
            usage = self._record_completion_usage(completion)
            content = cast(Any, completion).choices[0].message.content
            if not isinstance(content, str):
                raise TypeError("LM completion did not return string assistant content.")
            payload = {
                "content": content,
                "identity": {
                    "model": self.last_response_identity.model,
                    "system_fingerprint": self.last_response_identity.system_fingerprint,
                },
                "usage": usage,
            }
            return content, payload

        return cast(str, self._run_journaled(request, "completion", live_call))

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: str = "auto",
    ) -> ToolCompletion:
        """Complete a chat turn with provider-native function tools enabled.

        Args:
            messages: Conversation sent to the provider.
            tools: Provider-native function schemas available for this turn.
            tool_choice: Provider tool-selection mode.

        Returns:
            Normalized assistant content and function calls, with generated
            identifiers for calls whose provider omitted one.
        """
        import litellm

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "num_retries": self.num_retries,
            "drop_params": True,
            **self.completion_kwargs,
        }
        # DeepSeek V4 thinking mode uses automatic tool selection when tools
        # are present but rejects an explicit tool_choice field.
        if not self.model.startswith("deepseek/deepseek-v4-"):
            request_kwargs["tool_choice"] = tool_choice

        def live_call() -> tuple[ToolCompletion, dict[str, Any]]:
            """Complete and normalize one uncached native-tool request.

            Returns:
                Public tool completion and its secret-free journal payload.
            """
            try:
                completion = cast(Any, litellm.completion(**request_kwargs))
            except Exception as exc:
                raise LMProviderError(f"Completion provider failed for model {self.model!r}.") from exc
            usage = self._record_completion_usage(completion)

            message = completion.choices[0].message
            content_value = _response_value(message, "content", "")
            content = content_value if isinstance(content_value, str) else ""
            reasoning_value = _response_value(message, "reasoning_content", "")
            reasoning_content = reasoning_value if isinstance(reasoning_value, str) else ""
            calls: list[NativeToolCall] = []
            for index, call in enumerate(_response_value(message, "tool_calls", ()) or ()):
                function = _response_value(call, "function")
                name = _response_value(function, "name", "")
                arguments = _response_value(function, "arguments", "")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                calls.append(
                    NativeToolCall(
                        id=str(_response_value(call, "id", "") or f"tool_call_{index}"),
                        name=str(name),
                        arguments=arguments,
                    )
                )
            result = ToolCompletion(
                content=content,
                tool_calls=tuple(calls),
                reasoning_content=reasoning_content,
            )
            payload = {
                "content": content,
                "reasoning_content": reasoning_content,
                "tool_calls": [
                    {"id": call.id, "name": call.name, "arguments": call.arguments}
                    for call in calls
                ],
                "identity": {
                    "model": self.last_response_identity.model,
                    "system_fingerprint": self.last_response_identity.system_fingerprint,
                },
                "usage": usage,
            }
            return result, payload

        return cast(ToolCompletion, self._run_journaled(request_kwargs, "tool_completion", live_call))

    def _normalize_batch_response(self, response: Any) -> tuple[str, dict[str, Any], float, int, int]:
        """Validate and normalize one provider response from a live batch.

        Args:
            response: One LiteLLM completion response.

        Returns:
            Output text, journal payload, cost, input tokens, and
            output tokens.

        Raises:
            LMProviderError: The provider returned an exception in this slot.
            TypeError: A response omits string assistant content.
            ProviderIdentityMismatchError: A response changes a pinned
                provider identity.
        """
        import litellm

        if isinstance(response, Exception):
            raise LMProviderError(f"Batch completion provider failed for model {self.model!r}.") from response
        self._capture_and_validate_response_identity(response)
        self._check_truncation(response.choices)
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise TypeError("LM batch completion did not return string assistant content.")
        output = content.strip()
        try:
            response_cost = litellm.completion_cost(completion_response=response) or 0.0
        except Exception:
            response_cost = 0.0
        usage = getattr(response, "usage", None)
        response_tokens_in = (getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else 0
        response_tokens_out = (getattr(usage, "completion_tokens", 0) or 0) if usage is not None else 0
        payload = {
            "content": output,
            "identity": {
                "model": self.last_response_identity.model,
                "system_fingerprint": self.last_response_identity.system_fingerprint,
            },
            "usage": {
                "cost": float(response_cost),
                "tokens_in": int(response_tokens_in),
                "tokens_out": int(response_tokens_out),
            },
        }
        return output, payload, response_cost, response_tokens_in, response_tokens_out

    def _normalize_batch_responses(
        self,
        responses: list[Any],
    ) -> tuple[list[str], list[dict[str, Any]], float, int, int]:
        """Validate and normalize provider responses from one live batch.

        Args:
            responses: Ordered LiteLLM completion responses.

        Returns:
            Output texts, per-item journal payloads, cost, input tokens, and
            output tokens.

        Raises:
            LMProviderError: The provider returned an exception in any slot.
            TypeError: A response omits string assistant content.
            ProviderIdentityMismatchError: A response changes a pinned
                provider identity.
        """
        batch_cost = 0.0
        batch_tokens_in = 0
        batch_tokens_out = 0
        results: list[str] = []
        payloads: list[dict[str, Any]] = []
        for response in responses:
            output, payload, response_cost, response_tokens_in, response_tokens_out = (
                self._normalize_batch_response(response)
            )
            results.append(output)
            payloads.append(payload)
            batch_cost += response_cost
            batch_tokens_in += response_tokens_in
            batch_tokens_out += response_tokens_out
        return results, payloads, batch_cost, batch_tokens_in, batch_tokens_out

    def batch_complete(
        self, messages_list: list[list[dict[str, Any]]], max_workers: int = 10, **kwargs: Any
    ) -> list[str]:
        """Run multiple completions in parallel using ``litellm.batch_completion``.

        Args:
            messages_list: List of message lists, one per request.
            max_workers: Maximum concurrent requests.
            **kwargs: Extra keyword arguments forwarded to ``litellm.batch_completion``
                (e.g. ``timeout``, ``api_base``).  These override any matching keys
                set during ``__init__``.

        Returns:
            List of response strings, one per input.
        """
        import litellm

        if not messages_list:
            return []
        merged = {**self.completion_kwargs, **kwargs}
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages_list,
            "max_workers": max_workers,
            "num_retries": self.num_retries,
            "drop_params": True,
            **merged,
        }
        journal = self._response_journal
        scope = ACTIVE_RESPONSE_JOURNAL_SCOPE.get()
        if journal is None or scope is None:
            try:
                responses = litellm.batch_completion(**request)
            except Exception as exc:
                raise LMProviderError(f"Batch completion provider failed for model {self.model!r}.") from exc
            batch_results, _payloads, batch_cost, batch_tokens_in, batch_tokens_out = (
                self._normalize_batch_responses(responses)
            )
            with self._cost_lock:
                self._total_cost += batch_cost
                self._total_tokens_in += batch_tokens_in
                self._total_tokens_out += batch_tokens_out
            return batch_results

        individual_requests = [
            {
                "model": self.model,
                "messages": messages,
                "num_retries": self.num_retries,
                "drop_params": True,
                **merged,
            }
            for messages in messages_list
        ]
        request_digests = [canonical_request_digest(item) for item in individual_requests]
        with self._response_journal_lock:
            first_ordinal = self._response_journal_ordinals.get(scope, 0)
            results: list[str | None] = [None] * len(messages_list)
            missing_indices: list[int] = []
            for index, request_sha256 in enumerate(request_digests):
                cached = journal.load(scope, first_ordinal + index, request_sha256)
                if cached is None:
                    missing_indices.append(index)
                else:
                    results[index] = cast(str, self._restore_journal_result(cached, "completion"))

            if missing_indices:
                missing_request = {
                    **request,
                    "messages": [messages_list[index] for index in missing_indices],
                }
                try:
                    responses = litellm.batch_completion(**missing_request)
                except Exception as exc:
                    raise LMProviderError(f"Batch completion provider failed for model {self.model!r}.") from exc
                if len(responses) != len(missing_indices):
                    raise ValueError(
                        f"LiteLLM batch completion returned {len(responses)} responses for "
                        f"{len(missing_indices)} requests."
                    )
                normalized: list[tuple[int, str, dict[str, Any], float, int, int]] = []
                provider_errors: list[Exception] = []
                for result_index, response in zip(missing_indices, responses, strict=True):
                    if isinstance(response, Exception):
                        provider_errors.append(response)
                        continue
                    output, payload, response_cost, response_tokens_in, response_tokens_out = (
                        self._normalize_batch_response(response)
                    )
                    normalized.append(
                        (
                            result_index,
                            output,
                            payload,
                            response_cost,
                            response_tokens_in,
                            response_tokens_out,
                        )
                    )
                batch_cost = 0.0
                batch_tokens_in = 0
                batch_tokens_out = 0
                for result_index, output, payload, cost, tokens_in, tokens_out in normalized:
                    payload["kind"] = "completion"
                    journal.store(
                        scope,
                        first_ordinal + result_index,
                        request_digests[result_index],
                        payload,
                    )
                    results[result_index] = output
                    batch_cost += cost
                    batch_tokens_in += tokens_in
                    batch_tokens_out += tokens_out
                with self._cost_lock:
                    self._total_cost += batch_cost
                    self._total_tokens_in += batch_tokens_in
                    self._total_tokens_out += batch_tokens_out
                if provider_errors:
                    raise LMProviderError(
                        f"Batch completion provider failed for model {self.model!r}."
                    ) from provider_errors[0]

            if not all(isinstance(result, str) for result in results):
                raise RuntimeError("Response journal did not resolve every batch completion.")
            self._response_journal_ordinals[scope] = first_ordinal + len(results)
            return cast(list[str], results)

    def __repr__(self) -> str:
        params = [f"model={self.model!r}"]
        for k, v in self.completion_kwargs.items():
            params.append(f"{k}={v!r}")
        return f"LM({', '.join(params)})"


class TrackingLM:
    """Wraps an arbitrary callable to track estimated token usage.

    For callables that don't go through LiteLLM, we can't get real token
    counts or costs.  This wrapper estimates tokens from string lengths
    (~4 chars/token) and reports ``total_cost = 0.0``.

    This ensures that *every* reflection LM — whether an ``LM`` instance
    or a plain callable — exposes ``total_cost``, ``total_tokens_in``,
    and ``total_tokens_out``.
    """

    _CHARS_PER_TOKEN = 4

    def __init__(self, fn: Any):
        self._fn = fn
        self._total_cost: float = 0.0
        self._total_tokens_in: int = 0
        self._total_tokens_out: int = 0

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_tokens_in(self) -> int:
        return self._total_tokens_in

    @property
    def total_tokens_out(self) -> int:
        return self._total_tokens_out

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // self._CHARS_PER_TOKEN)

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        if isinstance(prompt, str):
            self._total_tokens_in += self._estimate_tokens(prompt)
        else:
            self._total_tokens_in += self._estimate_tokens(str(prompt))

        result = self._fn(prompt)

        if isinstance(result, str):
            self._total_tokens_out += self._estimate_tokens(result)

        return result

    def __getattr__(self, name: str):
        """Delegate selected optional interfaces with token estimation.

        Args:
            name: Requested wrapped-model attribute.

        Returns:
            Wrapped model name or an adapted native-tool or batch method.

        Raises:
            AttributeError: The attribute is not one of the supported delegated
                interfaces or the wrapped callable does not provide it.
        """
        if name == "model":
            return getattr(self._fn, name)
        # Conditionally expose batch_complete: hasattr(tracking_lm,
        # "batch_complete") must be True exactly when the wrapped callable
        # provides it, so batched reflection (StatelessReflectionLM) is not
        # silently downgraded to the per-task path by this wrapper.
        if name == "complete_with_tools":
            inner = getattr(self._fn, "complete_with_tools", None)
            if not callable(inner):
                raise AttributeError(name)

            def tracked_complete_with_tools(messages, tools, *, tool_choice="auto"):
                """Track estimated usage around one native-tool completion.

                Args:
                    messages: Conversation forwarded to the wrapped callable.
                    tools: Native function schemas forwarded with the request.
                    tool_choice: Provider tool-selection mode.

                Returns:
                    The wrapped callable's native-tool completion unchanged.
                """
                self._total_tokens_in += self._estimate_tokens(str((messages, tools)))
                result = cast(Any, inner)(messages, tools, tool_choice=tool_choice)
                self._total_tokens_out += self._estimate_tokens(str(result))
                return result

            return tracked_complete_with_tools
        if name == "batch_complete":
            inner = getattr(self._fn, "batch_complete", None)
            if not callable(inner):
                raise AttributeError(name)

            def tracked_batch_complete(messages_list):
                """Estimate tokens for one wrapped batch completion.

                Args:
                    messages_list: Batch of conversations forwarded unchanged.

                Returns:
                    Materialized wrapped responses in input order.
                """
                for messages in messages_list:
                    self._total_tokens_in += self._estimate_tokens(str(messages))
                results = list(cast(Any, inner)(messages_list))
                for result in results:
                    if isinstance(result, str):
                        self._total_tokens_out += self._estimate_tokens(result)
                return results

            return tracked_batch_complete
        raise AttributeError(name)

    def __repr__(self) -> str:
        return f"TrackingLM({self._fn!r})"
