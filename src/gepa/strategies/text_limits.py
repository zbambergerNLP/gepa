"""Configure optional character limits without imposing a default text budget."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


class TextLimitError(ValueError):
    """Report an explicitly configured limit without silently cutting a prompt."""


def validate_char_limit(name: str, value: int | None) -> None:
    """Require a positive integer character limit or ``None`` for unlimited."""
    if value is not None and (type(value) is not int or value < 1):
        raise ValueError(f"{name} must be a positive integer or null (unlimited).")


def clip_text(text: str, max_chars: int | None, *, head_and_tail: bool = False) -> str:
    """Retain at most the requested source characters, plus an omission marker.

    Args:
        text: Original Unicode text, preserved exactly when it fits.
        max_chars: Source-character allowance, or ``None`` to preserve all text.
        head_and_tail: Retain both ends instead of only the beginning.

    Returns:
        Original text or a marked excerpt. Marker characters are additional.
    """
    validate_char_limit("max_chars", max_chars)
    if max_chars is None or len(text) <= max_chars:
        return text
    marker = f"\n[... {len(text) - max_chars} characters omitted ...]\n"
    if head_and_tail:
        tail_size = max_chars // 2
        tail = text[-tail_size:] if tail_size else ""
        return text[: max_chars - tail_size] + marker + tail
    return text[:max_chars] + marker


@dataclass(frozen=True)
class TextLimits:
    """Set independent text allowances; every default is unlimited.

    Component and candidate limits reject oversized proposed documents. The
    selector target is guidance only. Prompt limits reject complete optimizer
    requests before a model call. Evidence and saved-field limits retain marked
    excerpts; full provider responses and source artifacts remain available.
    """

    max_component_chars: int | None = None
    max_candidate_chars: int | None = None
    selector_target_chars: int | None = None
    max_prompt_chars: int | None = None
    controller_feedback_chars: int | None = None
    stateless_feedback_chars: int | None = None
    manifestor_trace_chars: int | None = None
    manifestor_steering_chars: int | None = None
    history_text_chars: int | None = None
    verifier_log_chars: int | None = None

    def __post_init__(self) -> None:
        """Validate every supplied limit, including values loaded from JSON."""
        for name, value in asdict(self).items():
            validate_char_limit(name, value)

    def to_dict(self) -> dict[str, int | None]:
        """Return all resolved settings for configuration and run identity."""
        return asdict(self)

    def document_contract(self) -> dict[str, Any]:
        """Record the selected component cap and soft selector target."""
        return {
            "version": 2,
            "max_component_chars": self.max_component_chars,
            "max_candidate_chars": self.max_candidate_chars,
            "selector_target_chars": self.selector_target_chars,
        }

    def check_prompt(self, prompt: Any, tools: Any = None) -> None:
        """Check a complete optimizer request, including native tool definitions.

        Strings are measured directly. Structured requests are measured as
        compact, Unicode-preserving JSON; tool definitions use the same format.
        No part of the assembled request is cut to make it fit.
        """
        if self.max_prompt_chars is None:
            return
        rendered = prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))
        size = len(rendered)
        if tools is not None:
            size += len(json.dumps(tools, ensure_ascii=False, separators=(",", ":")))
        if size > self.max_prompt_chars:
            raise TextLimitError(f"Optimizer prompt has {size} characters; max_prompt_chars={self.max_prompt_chars}.")

    def check_candidate(self, candidate: Mapping[str, str]) -> None:
        """Reject components or a complete prompt/skill bundle above an explicit cap."""
        if self.max_component_chars is not None:
            for name, text in candidate.items():
                if len(text) > self.max_component_chars:
                    raise TextLimitError(
                        f"Component {name!r} has {len(text)} characters; max_component_chars={self.max_component_chars}."
                    )
        size = sum(len(text) for text in candidate.values())
        if self.max_candidate_chars is not None and size > self.max_candidate_chars:
            raise TextLimitError(f"Candidate has {size} characters; max_candidate_chars={self.max_candidate_chars}.")


def resolve_text_limits(value: TextLimits | Mapping[str, Any] | None = None) -> TextLimits:
    """Resolve typed or serialized settings, rejecting misspelled configuration keys."""
    if isinstance(value, TextLimits):
        return value
    if value is None:
        return TextLimits()
    if not isinstance(value, Mapping):
        raise ValueError("text_limits must be a JSON object or TextLimits instance.")
    unknown = set(value) - set(TextLimits.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown text_limits settings: {sorted(unknown)}")
    return TextLimits(**dict(value))


def parse_text_limits(value: str) -> TextLimits:
    """Parse a CLI JSON object whose omitted or null fields mean unlimited."""
    return resolve_text_limits(json.loads(value))
