"""Keep the public upstream demonstration fixed and edit only agent instructions."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from examples.appworld.benchmark_settings import COMPONENT
from examples.common.react_v2 import structured_prompt

UPSTREAM_PROMPT = Path(__file__).with_name("upstream_react.txt")


def _prompt_parts() -> tuple[str, str, str]:
    """Separate the upstream instructions, demonstration, and actual task template."""
    text = UPSTREAM_PROMPT.read_text(encoding="utf-8")
    introduction, rest = text.removeprefix("USER:\n").split("My name is:", 1)
    demo, rest = ("My name is:" + rest).split("----------------------------------------------", 1)
    instructions, task = rest.strip().removeprefix("USER:\n").split("\nUSER:\n", 1)
    return introduction.strip() + "\n\n" + instructions.strip(), "USER:\n" + demo.strip(), task.strip()


def seed_candidate(template_family: str) -> dict[str, str]:
    """Wrap the upstream instructions in the shared FOREST-compatible structure."""
    instructions, _, _ = _prompt_parts()
    return {COMPONENT: structured_prompt(instructions, template_family)}


def initial_messages(candidate: dict[str, str], context: dict[str, Any]) -> list[dict[str, str]]:
    """Render only public context into fixed messages and use the candidate verbatim."""
    if set(candidate) != {COMPONENT} or not isinstance(candidate[COMPONENT], str) or not candidate[COMPONENT].strip():
        raise ValueError("AppWorld requires exactly one nonempty system_prompt component.")
    _, demo, task = _prompt_parts()
    supervisor = context["supervisor"]
    values = {
        **{f"main_user.{key}": str(supervisor[key]) for key in ("first_name", "last_name", "email", "phone_number")},
        "app_descriptions": json.dumps(
            [{"name": name, "description": description} for name, description in context["app_descriptions"].items()],
            indent=1,
        ),
        "input_str": context["instruction"],
    }

    def render(text: str) -> str:
        # One substitution pass prevents task text from introducing template expressions.
        return re.sub(r"{{\s*([\w.]+)\s*}}", lambda match: values[match[1]], text)

    messages = [{"role": "system", "content": candidate[COMPONENT]}]
    parts = re.split(r"^(USER|ASSISTANT):\n", demo, flags=re.MULTILINE)
    for index in range(1, len(parts), 2):
        messages.append({"role": parts[index].lower(), "content": render(parts[index + 1].strip())})
    messages.append({"role": "user", "content": render(task)})
    return messages


def extract_code(text: str) -> tuple[str, str]:
    """Execute the first closed Python fence, as in upstream ignore_multiple_calls."""
    match = re.search(r"```python\n(.*?)```", text, flags=re.DOTALL)
    if match is None or not match[1].strip():
        raise ValueError("Expected a nonempty, closed Python code block.")
    return match[1].strip(), text[: match.end()]
