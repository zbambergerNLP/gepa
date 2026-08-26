# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from unittest.mock import MagicMock, patch

import pytest

from gepa.lm import LM, InlineReasoningError, InlineReasoningLM, NativeToolCall, ToolCompletion, TrackingLM


class TestLMInit:
    """Test LM constructor parameter handling."""

    def test_defaults(self):
        lm = LM("openai/gpt-4.1")
        assert "temperature" not in lm.completion_kwargs
        assert "max_tokens" not in lm.completion_kwargs

    def test_custom_params(self):
        lm = LM("openai/gpt-4.1", temperature=0.5, max_tokens=4096)
        assert lm.completion_kwargs["temperature"] == 0.5
        assert lm.completion_kwargs["max_tokens"] == 4096

    def test_extra_kwargs_forwarded(self):
        lm = LM("openai/gpt-4.1", top_p=0.9, stop=["\n"])
        assert lm.completion_kwargs["top_p"] == 0.9
        assert lm.completion_kwargs["stop"] == ["\n"]

    def test_reasoning_model_no_special_treatment(self):
        """Reasoning models should NOT get special parameter handling."""
        lm = LM("openai/gpt-5-mini", temperature=0.7, max_tokens=4096)
        assert lm.completion_kwargs["temperature"] == 0.7
        assert lm.completion_kwargs["max_tokens"] == 4096
        assert "max_completion_tokens" not in lm.completion_kwargs


class TestLMCall:
    """Test LM __call__ method."""

    @patch("litellm.completion")
    def test_string_prompt(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "response text"
        mock_response.choices[0].finish_reason = "stop"
        mock_completion.return_value = mock_response

        lm = LM("openai/gpt-4.1", temperature=0.5)
        result = lm("hello")

        assert result == "response text"
        mock_completion.assert_called_once_with(
            model="openai/gpt-4.1",
            messages=[{"role": "user", "content": "hello"}],
            num_retries=3,
            drop_params=True,
            temperature=0.5,
        )

    @patch("litellm.completion")
    def test_messages_prompt(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "chat response"
        mock_response.choices[0].finish_reason = "stop"
        mock_completion.return_value = mock_response

        lm = LM("openai/gpt-4.1")
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        result = lm(messages)

        assert result == "chat response"
        mock_completion.assert_called_once_with(
            model="openai/gpt-4.1",
            messages=messages,
            num_retries=3,
            drop_params=True,
        )

    @patch("litellm.completion")
    def test_provider_separated_reasoning_is_not_mixed_into_content(self, mock_completion):
        """Verify provider-separated reasoning never enters answer content.

        Args:
            mock_completion: Patched LiteLLM completion callable.
        """
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "answer"
        mock_response.choices[0].message.reasoning_content = "private reasoning"
        mock_response.choices[0].finish_reason = "stop"
        mock_completion.return_value = mock_response

        assert LM("openai/reasoning-model")("question") == "answer"

    @patch("litellm.completion")
    def test_truncation_warning(self, mock_completion, caplog):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "truncated"
        mock_response.choices[0].finish_reason = "length"
        mock_completion.return_value = mock_response

        lm = LM("openai/gpt-4.1", max_tokens=100)
        result = lm("hello")

        assert result == "truncated"
        assert "truncated" in caplog.text.lower()


class TestLMNativeTools:
    """Test the provider-native function-tool completion path."""

    @patch("litellm.completion")
    def test_complete_with_tools_uses_auto_choice_and_normalizes_calls(self, mock_completion):
        """Verify native tool calls use automatic choice and normalize output.

        Args:
            mock_completion: Patched LiteLLM completion callable.
        """
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].finish_reason = "tool_calls"
        response.choices[0].message.content = None
        response.choices[0].message.reasoning_content = "private reasoning"
        native_call = MagicMock()
        native_call.id = "call-1"
        native_call.function.name = "REPLACE_TEXT"
        native_call.function.arguments = '{"target":"old","text":"new"}'
        response.choices[0].message.tool_calls = [native_call]
        mock_completion.return_value = response
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "REPLACE_TEXT",
                    "parameters": {"type": "object"},
                },
            }
        ]
        messages = [{"role": "user", "content": "edit"}]

        lm = LM("openai/gpt-4.1", temperature=0.5)
        result = lm.complete_with_tools(messages, tools)

        assert result == ToolCompletion(
            content="",
            tool_calls=(NativeToolCall("call-1", "REPLACE_TEXT", '{"target":"old","text":"new"}'),),
        )
        mock_completion.assert_called_once_with(
            model="openai/gpt-4.1",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            num_retries=3,
            drop_params=True,
            temperature=0.5,
        )

    def test_tracking_wrapper_conditionally_preserves_native_tool_interface(self):
        """Verify tracking preserves native tools only when the callable has them."""
        native_callable = MagicMock(return_value="unused")
        native_callable.complete_with_tools.return_value = ToolCompletion(
            "", (NativeToolCall("call-1", "DELETE_TEXT", "{}"),)
        )
        wrapped_native = TrackingLM(native_callable)
        result = wrapped_native.complete_with_tools([], [], tool_choice="auto")

        assert result.tool_calls[0].name == "DELETE_TEXT"
        assert hasattr(wrapped_native, "complete_with_tools")
        assert not hasattr(TrackingLM(MagicMock(return_value="fallback", spec=[])), "complete_with_tools")


class TestLMBatchComplete:
    """Test LM batch_complete method."""

    @patch("litellm.batch_completion")
    def test_batch_complete(self, mock_batch):
        """Verify batch completion preserves order and strips outer whitespace.

        Args:
            mock_batch: Patched LiteLLM batch-completion callable.
        """
        resp1 = MagicMock()
        resp1.choices = [MagicMock()]
        resp1.choices[0].message.content = " answer1 "
        resp1.choices[0].message.reasoning_content = "private reasoning one"
        resp1.choices[0].finish_reason = "stop"
        resp2 = MagicMock()
        resp2.choices = [MagicMock()]
        resp2.choices[0].message.content = " answer2 "
        resp2.choices[0].message.reasoning_content = "private reasoning two"
        resp2.choices[0].finish_reason = "stop"
        mock_batch.return_value = [resp1, resp2]

        lm = LM("openai/gpt-4.1")
        msgs = [
            [{"role": "user", "content": "q1"}],
            [{"role": "user", "content": "q2"}],
        ]
        results = lm.batch_complete(msgs, max_workers=5)

        assert results == ["answer1", "answer2"]
        mock_batch.assert_called_once_with(
            model="openai/gpt-4.1",
            messages=msgs,
            max_workers=5,
            num_retries=3,
            drop_params=True,
        )

    @patch("litellm.batch_completion")
    def test_batch_complete_forwards_extra_kwargs(self, mock_batch):
        """Extra kwargs passed to batch_complete should be forwarded to litellm."""
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "ans"
        resp.choices[0].finish_reason = "stop"
        mock_batch.return_value = [resp]

        lm = LM("openai/gpt-4.1", temperature=0.5)
        lm.batch_complete(
            [[{"role": "user", "content": "q"}]],
            max_workers=3,
            timeout=30,
            api_base="https://custom.api",
        )

        call_kwargs = mock_batch.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["timeout"] == 30
        assert call_kwargs["api_base"] == "https://custom.api"

    @patch("litellm.batch_completion")
    def test_batch_complete_kwargs_override_init(self, mock_batch):
        """Kwargs passed at call time should override init kwargs."""
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "ans"
        resp.choices[0].finish_reason = "stop"
        mock_batch.return_value = [resp]

        lm = LM("openai/gpt-4.1", temperature=0.5)
        lm.batch_complete(
            [[{"role": "user", "content": "q"}]],
            temperature=0.9,  # override init value
        )

        call_kwargs = mock_batch.call_args[1]
        assert call_kwargs["temperature"] == 0.9


class TestInlineReasoningLM:
    """Test the opt-in adapter for endpoints that embed reasoning in content."""

    def test_preserves_ordinary_output_byte_for_byte(self):
        """Verify the inline-reasoning wrapper preserves ordinary output byte for byte."""
        output = "\n  Answer with the literal example <think>draft</think>.  \n"
        lm = InlineReasoningLM(MagicMock(return_value=output))

        assert lm("question") == output

    @pytest.mark.parametrize(
        ("output", "expected"),
        [
            ("<think>private</think>answer", "answer"),
            (" \n<think>first</think>\n\t<think>second</think>\nanswer", "\nanswer"),
            ("<think></think>", ""),
        ],
    )
    def test_removes_only_complete_leading_reasoning_blocks(self, output, expected):
        """Verify only complete leading reasoning blocks are removed.

        Args:
            output: Raw model content under test.
            expected: Expected answer-only content.
        """
        lm = InlineReasoningLM(MagicMock(return_value=output))

        assert lm("question") == expected

    @pytest.mark.parametrize(
        "output",
        [
            "<think>private reasoning",
            "<think",
            "<think private reasoning",
            "<think/>private reasoning",
            "<think>outer<think>nested</think>",
            "</think>answer",
            "</think",
            "<think>private</think></think>answer",
            "<think>private</think><think/>",
        ],
    )
    def test_rejects_malformed_leading_reasoning_without_leaking_it(self, output):
        """Verify malformed reasoning is rejected without echoing private text.

        Args:
            output: Malformed raw model content under test.
        """
        lm = InlineReasoningLM(MagicMock(return_value=output))

        with pytest.raises(InlineReasoningError) as exc_info:
            lm("question")

        assert "private reasoning" not in str(exc_info.value)
        assert "outer" not in str(exc_info.value)

    def test_requires_string_results(self):
        """Verify the inline-reasoning wrapper requires string results."""
        lm = InlineReasoningLM(MagicMock(return_value=None))

        with pytest.raises(TypeError, match="return strings"):
            lm("question")

    def test_conditionally_exposes_optional_interfaces_and_delegates_attributes(self):
        """Verify the inline-reasoning wrapper conditionally exposes optional interfaces and delegates attributes."""
        inner = MagicMock(
            return_value="answer",
            spec=["lm", "model", "completion_kwargs", "total_cost", "total_tokens_in", "total_tokens_out"],
        )
        inner.lm = object()
        inner.model = "custom/reasoning-model"
        inner.completion_kwargs = {"temperature": 0.2}
        inner.total_cost = 1.25
        inner.total_tokens_in = 10
        inner.total_tokens_out = 20
        lm = InlineReasoningLM(inner)

        assert lm.lm is inner.lm
        assert lm.model == inner.model
        assert lm.completion_kwargs is inner.completion_kwargs
        assert lm.total_cost == inner.total_cost
        assert lm.total_tokens_in == inner.total_tokens_in
        assert lm.total_tokens_out == inner.total_tokens_out
        assert lm.supports_cost_tracking()
        assert not hasattr(lm, "batch_complete")
        assert not hasattr(lm, "complete_with_tools")

    def test_adapts_batch_results_and_forwards_arguments(self):
        """Verify the inline-reasoning wrapper adapts batch results and forwards arguments."""
        inner = MagicMock(return_value="unused", spec=["batch_complete"])
        inner.batch_complete.return_value = ["<think>one</think>first", "second <think>literal</think>"]
        lm = InlineReasoningLM(inner)
        messages = [[{"role": "user", "content": "one"}], [{"role": "user", "content": "two"}]]

        assert lm.batch_complete(messages, max_workers=3) == ["first", "second <think>literal</think>"]
        inner.batch_complete.assert_called_once_with(messages, max_workers=3)

    def test_rejects_malformed_reasoning_in_batch_results(self):
        """Verify the inline-reasoning wrapper rejects malformed reasoning in batch results."""
        inner = MagicMock(return_value="unused", spec=["batch_complete"])
        inner.batch_complete.return_value = ["answer", "<think>truncated"]
        with pytest.raises(InlineReasoningError, match="closing tag"):
            InlineReasoningLM(inner).batch_complete([])

    def test_adapts_native_content_without_changing_tool_calls(self):
        """Verify the inline-reasoning wrapper adapts native content without changing tool calls."""
        calls = (NativeToolCall("call-1", "REPLACE_TEXT", '{"target":"old","text":"new"}'),)

        inner = MagicMock(return_value="unused", spec=["complete_with_tools"])
        inner.complete_with_tools.return_value = ToolCompletion("<think>private</think>ready", calls)
        lm = InlineReasoningLM(inner)

        result = lm.complete_with_tools([], [], tool_choice="required")

        assert result == ToolCompletion("ready", calls)
        assert result.tool_calls is calls
        inner.complete_with_tools.assert_called_once_with([], [], tool_choice="required")

    def test_rejects_malformed_reasoning_in_native_content(self):
        """Verify the inline-reasoning wrapper rejects malformed reasoning in native content."""
        inner = MagicMock(return_value="unused", spec=["complete_with_tools"])
        inner.complete_with_tools.return_value = ToolCompletion("<think/>private reasoning", ())
        with pytest.raises(InlineReasoningError) as exc_info:
            InlineReasoningLM(inner).complete_with_tools([], [])

        assert "private reasoning" not in str(exc_info.value)

    def test_composes_with_usage_tracking(self):
        """Verify the inline-reasoning wrapper composes with usage tracking."""
        adapted = InlineReasoningLM(MagicMock(return_value="<think>private</think>answer"))
        lm = TrackingLM(adapted)

        assert lm("question") == "answer"
        assert lm.total_tokens_in > 0
        assert lm.total_tokens_out == 1

    def test_preserves_estimated_cost_capability_through_wrapper(self):
        """Verify reasoning adaptation preserves estimated-cost capability."""
        lm = InlineReasoningLM(TrackingLM(MagicMock(return_value="answer")))

        assert not lm.supports_cost_tracking()


class TestLMRepr:
    def test_repr(self):
        lm = LM("openai/gpt-4.1", temperature=0.5)
        assert "gpt-4.1" in repr(lm)
        assert "temperature=0.5" in repr(lm)


class TestLMConformsToProtocol:
    """Verify LM satisfies the LanguageModel protocol."""

    def test_callable(self):
        """Verify the LM implementation satisfies the callable protocol."""
        lm = LM("openai/gpt-4.1")
        assert callable(lm)
