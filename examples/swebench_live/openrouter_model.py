"""Preserve OpenRouter tool reasoning across the pinned SWE-agent boundary.

Import this module only after activating the pinned upstream source. Its model
keeps the native query/retry, context checks, cost checks and tool parser. The
response conversion differs because upstream keeps only Anthropic thinking
blocks and counts only visible text as output tokens.
"""

from copy import deepcopy

import litellm
from sweagent import __version__
from sweagent.agent.hooks.abstract import AbstractAgentHook
from sweagent.agent.models import LiteLLMModel
from sweagent.exceptions import ContentPolicyViolationError, ContextWindowExceededError, ModelConfigurationError


class OpenRouterModel(LiteLLMModel):
    """Retain provider reasoning without placing it in visible assistant text."""

    def __init__(self, config, tools):
        super().__init__(config, tools)
        self.last_tool_response = {}

    def reset_stats(self):
        super().reset_stats()
        self.last_tool_response = {}

    def _history_to_messages(self, history):
        messages = super()._history_to_messages(history)
        for entry, message in zip(history, messages, strict=True):
            if message["role"] == "assistant" and message.get("tool_calls") and entry.get("reasoning_details"):
                message["reasoning_details"] = deepcopy(entry["reasoning_details"])
        return messages

    def _single_query(self, messages, n=None, temperature=None):
        """Use the pinned model's request contract with complete response accounting."""
        self.last_tool_response = {}
        self._sleep()
        counted = deepcopy(messages)
        for message in counted:
            message.pop("cache_control", None)
            message.pop("thinking_blocks", None)
        input_tokens = litellm.utils.token_counter(
            messages=counted,
            model=self.custom_tokenizer["identifier"] if self.custom_tokenizer else self.config.name,
            custom_tokenizer=self.custom_tokenizer,
        )
        if self.model_max_input_tokens and input_tokens > self.model_max_input_tokens > 0:
            raise ContextWindowExceededError(
                f"Input tokens {input_tokens} exceed max tokens {self.model_max_input_tokens}"
            )
        kwargs = deepcopy(self.config.completion_kwargs)
        kwargs.setdefault("extra_headers", {}).setdefault("User-Agent", f"swe-agent/{__version__}")
        if self.config.api_base:
            kwargs["api_base"] = self.config.api_base
        if self.tools.use_function_calling:
            kwargs["tools"] = self.tools.tools
        try:
            response = litellm.completion(
                model=self.config.name,
                messages=messages,
                api_key=self.config.choose_api_key(),
                fallbacks=self.config.fallbacks,
                n=n,
                **kwargs,
            )
        except litellm.exceptions.ContextWindowExceededError as error:
            raise ContextWindowExceededError from error
        except litellm.exceptions.ContentPolicyViolationError as error:
            raise ContentPolicyViolationError from error
        except litellm.exceptions.BadRequestError as error:
            if "is longer than the model's context length" in str(error):
                raise ContextWindowExceededError from error
            raise
        self.logger.debug(f"Response: {response}")
        try:
            cost = litellm.cost_calculator.completion_cost(response, model=self.config.name)
        except Exception as error:
            if self.config.per_instance_cost_limit > 0 or self.config.total_cost_limit > 0:
                raise ModelConfigurationError(f"Cannot enforce cost limits for {self.config.name}") from error
            cost = 0
        outputs = []
        output_tokens = 0
        for choice in response.choices[: n or 1]:
            message = choice.message
            output = {"message": message.content or ""}
            output_tokens += litellm.utils.token_counter(
                text=output["message"],
                model=self.custom_tokenizer["identifier"] if self.custom_tokenizer else self.config.name,
                custom_tokenizer=self.custom_tokenizer,
            )
            if self.tools.use_function_calling:
                calls = [call.to_dict() for call in message.tool_calls or []]
                output["tool_calls"] = calls
                fields = getattr(message, "provider_specific_fields", None) or {}
                details = fields.get("reasoning_details", getattr(message, "reasoning_details", None))
                if calls and details:
                    self.last_tool_response = {"tool_calls": deepcopy(calls), "reasoning_details": deepcopy(details)}
            if getattr(message, "thinking_blocks", None):
                output["thinking_blocks"] = message.thinking_blocks
            outputs.append(output)
        usage = getattr(response, "usage", None)
        # Provider usage includes tool arguments and reasoning; visible text may be empty.
        if isinstance(getattr(usage, "prompt_tokens", None), int):
            input_tokens = usage.prompt_tokens
        if isinstance(getattr(usage, "completion_tokens", None), int):
            output_tokens = usage.completion_tokens
        self._update_stats(input_tokens=input_tokens, output_tokens=output_tokens, cost=cost)
        return outputs


class ReasoningHistoryHook(AbstractAgentHook):
    """Save the same reasoning that the next solver request receives."""

    def __init__(self, model):
        self.model = model

    def on_init(self, *, agent):
        self.agent = agent

    def on_step_done(self, *, step, info):
        response = self.model.last_tool_response
        if not step.tool_calls or step.tool_calls != response.get("tool_calls"):
            return
        # Tool IDs can repeat across turns. Update only the step just appended.
        message = next(entry for entry in reversed(self.agent.history) if entry["role"] == "assistant")
        message["reasoning_details"] = deepcopy(response["reasoning_details"])
