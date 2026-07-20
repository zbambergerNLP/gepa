# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Text post-processing helpers for LM outputs."""

import re

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think_tags(text: str) -> str:
    """Remove reasoning-model ``<think>...</think>`` blocks from LM output.

    Closed blocks are removed wherever they appear. If an unmatched ``<think>``
    remains afterwards, the reasoning was truncated before ``</think>``; since
    reasoning models emit the answer only after the closing tag, everything from
    the dangling ``<think>`` onward is dropped rather than leaked into the result.
    """
    text = _THINK_BLOCK_RE.sub("", text)
    dangling = text.find("<think>")
    if dangling != -1:
        text = text[:dangling]
    return text
