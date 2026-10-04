"""Provide offline model responses while exercising real benchmark optimizers."""

import json
import re

from examples.common import benchmark_runner, react_v2
from examples.common.react_v2 import structured_prompt
from gepa.lm import NativeToolCall, ToolCompletion


class BenchmarkProposerLM:
    """Answer real GEPA/FOREST role protocols without replacing their implementations."""

    def __init__(self, model="offline-proposer", **kwargs):
        self.model = model
        self.completion_kwargs = kwargs
        self.calls = []

    def __call__(self, prompt):
        self.calls.append(prompt)
        if isinstance(prompt, list):
            prompt = "\n".join(str(message["content"]) for message in prompt)
        if "Validate and manifest" in prompt:
            return json.dumps(
                {
                    "status": "ready",
                    "observation": "Training feedback needs clarification.",
                    "hypothesis": "Explicit instructions help.",
                    "general_change": "Add improved guidance.",
                    "scope": "Preserve other instructions.",
                }
            )
        if "Write the next instruction for a language model editor" in prompt:
            return "Add improved guidance supported by the training feedback."
        if "Choose edit actions that address" in prompt:
            options = re.findall(r"^- (\S+): ", prompt, re.M)
            chosen = next((item for item in options if "contextualize@Objective/" in item), None)
            chosen = chosen or next((item for item in options if "EDIT@Objective" in item), None)
            chosen = chosen or next((item for item in options if "/INSERT_TEXT" in item), options[0])
            return (
                "<response>"
                + "".join(
                    f"<candidate><action>{item}</action><reasoning>Clarify the training instructions.</reasoning>"
                    f"<probability>{int(item == chosen)}</probability></candidate>"
                    for item in options
                )
                + "</response>"
            )
        if "--- Edit constraint ---" in prompt:
            return "```\nUse improved instructions.\n```"
        return "```\n" + structured_prompt("Use improved instructions.", "alibaba") + "\n```"

    def complete_with_tools(self, messages, tools, **kwargs):
        self.calls.append(messages)
        if any(message.get("role") == "tool" for message in messages):
            return ToolCompletion("<finish>The edit is complete.</finish>", ())
        try:
            task = json.loads(messages[-1]["content"])
            body = task["section_body"]
        except (json.JSONDecodeError, KeyError):
            match = re.search(r"## Current selected section body[^\n]*\n([^\n]+)", messages[-1]["content"])
            body = json.loads(match[1])
        selected = re.search(r"Every call must use (\w+)", messages[0]["content"])
        operator = selected[1] if selected else "INSERT_TEXT"
        if operator == "INSERT_TEXT":
            args = {"anchor": "", "text": " improved guidance.", "where": "after"}
        elif not body:
            return ToolCompletion("<finish>The selected section is empty.</finish>", ())
        elif operator == "DELETE_TEXT":
            args = {"target": body[0]}
        elif operator == "REPLACE_TEXT":
            args = {"target": body, "text": body + " improved guidance."}
        else:
            args = {"target": body[0], "anchor": "", "where": "after"}
        return ToolCompletion("", (NativeToolCall("edit", operator, json.dumps(args)),))


def install_proposer(monkeypatch):
    """Replace only shared proposer transports and return their observable instances."""
    instances = []

    def factory(*args, **kwargs):
        lm = BenchmarkProposerLM(*args, **kwargs)
        instances.append(lm)
        return lm

    monkeypatch.setattr(benchmark_runner, "LM", factory)
    monkeypatch.setattr(react_v2, "LM", factory)
    return instances
