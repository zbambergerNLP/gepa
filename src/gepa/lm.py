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
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NativeToolCall:
    """One provider-native function call returned by a language model."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ToolCompletion:
    """Assistant content and provider-native function calls from one completion."""

    content: str
    tool_calls: tuple[NativeToolCall, ...]


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
                return ToolCompletion(content=self._answer(result.content), tool_calls=result.tool_calls)

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
        **kwargs: Extra keyword arguments forwarded to ``litellm.completion``
            (e.g. ``top_p``, ``stop``, ``api_key``, ``api_base``).
    """

    def __init__(
        self,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        num_retries: int = 3,
        **kwargs: Any,
    ):
        self.model = model
        self.num_retries = num_retries
        self._total_cost: float = 0.0
        self._total_tokens_in: int = 0
        self._total_tokens_out: int = 0
        self._cost_lock = threading.Lock()

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

    def _check_truncation(self, choices: list[Any]) -> None:
        if any(getattr(c, "finish_reason", None) == "length" for c in choices):
            max_tok = self.completion_kwargs.get("max_tokens") or self.completion_kwargs.get("max_completion_tokens")
            logger.warning(
                f"LM response was truncated (finish_reason='length', max_tokens={max_tok}). "
                "Consider increasing max_tokens for better results."
            )

    def _record_completion_usage(self, completion: Any) -> None:
        """Accumulate cost and token usage from one LiteLLM completion.

        Cost lookup failures are treated as zero so usage accounting never
        changes completion behavior.

        Args:
            completion: LiteLLM response carrying choices and optional usage.
        """
        import litellm

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

        completion = litellm.completion(
            model=self.model,
            messages=messages,
            num_retries=self.num_retries,
            drop_params=True,
            **self.completion_kwargs,
        )

        self._record_completion_usage(completion)

        return completion.choices[0].message.content  # type: ignore[union-attr]

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

        completion = cast(
            Any,
            litellm.completion(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                num_retries=self.num_retries,
                drop_params=True,
                **self.completion_kwargs,
            ),
        )
        self._record_completion_usage(completion)

        message = completion.choices[0].message
        content_value = _response_value(message, "content", "")
        content = content_value if isinstance(content_value, str) else ""
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
        return ToolCompletion(content=content, tool_calls=tuple(calls))

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

        merged = {**self.completion_kwargs, **kwargs}
        responses = litellm.batch_completion(
            model=self.model,
            messages=messages_list,
            max_workers=max_workers,
            num_retries=self.num_retries,
            drop_params=True,
            **merged,
        )

        batch_cost = 0.0
        batch_tokens_in = 0
        batch_tokens_out = 0
        results: list[str] = []
        for resp in responses:
            self._check_truncation(resp.choices)
            results.append(resp.choices[0].message.content.strip())
            try:
                batch_cost += litellm.completion_cost(completion_response=resp) or 0.0  # type: ignore[attr-defined]
            except Exception:
                pass
            usage = getattr(resp, "usage", None)
            if usage is not None:
                batch_tokens_in += getattr(usage, "prompt_tokens", 0) or 0
                batch_tokens_out += getattr(usage, "completion_tokens", 0) or 0

        with self._cost_lock:
            self._total_cost += batch_cost
            self._total_tokens_in += batch_tokens_in
            self._total_tokens_out += batch_tokens_out

        return results

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
