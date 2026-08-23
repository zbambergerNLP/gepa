# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import pytest

from gepa.strategies.instruction_proposal import InstructionProposalSignature
from gepa.utils.text import strip_think_tags


class TestInstructionProposalSignature:
    """Test InstructionProposalSignature functions."""

    @pytest.mark.parametrize(
        "lm_output,expected_instruction",
        [
            # Test with language specifier
            (
                """Here's the improved instruction:
```markdown
This is the actual instruction content.
It should not include the word 'markdown'.
```
""",
                "This is the actual instruction content.\nIt should not include the word 'markdown'.",
            ),
            # Test without language specifier (original behavior)
            (
                """Here's the instruction:
```
This is the instruction without language specifier.
```
Done.""",
                "This is the instruction without language specifier.",
            ),
            (
                """```markdown
Don't get confused by these backticks: ```
```""",
                "Don't get confused by these backticks: ```",
            ),
            # Test stripping the output string
            (
                """```

Here are the instructions.

```""",
                "Here are the instructions.",
            ),
            # Test multiple sets of backticks (should take the "outermost" block)
            (
                """Begin text
```plaintext
Begin instructions

```
Internal block 1
```

```python
Internal block 2
```

End instructions
```
End text
""",
                "Begin instructions\n\n```\nInternal block 1\n```\n\n```python\nInternal block 2\n```\n\nEnd instructions",
            ),
            # Test when the output starts with ``` but doesn't end with it
            (
                """```text
Here are the instructions.""",
                "Here are the instructions.",
            ),
            # Test when the output ends with ``` but doesn't start with it
            (
                """Here are the instructions.
```""",
                "Here are the instructions.",
            ),
            # Test only backticks in the middle
            (
                """
Here are some backticks:
```
I hope you didn't get confused.
                """,
                "Here are some backticks:\n```\nI hope you didn't get confused.",
            ),
            # Test when there are no backticks at all, also strip whitespace
            (
                """
                Here are the instructions.
                """,
                "Here are the instructions.",
            ),
            # Test stripping <think> tags with backtick-delimited instruction
            (
                "<think>\nLet me analyze the failures...\n</think>\n\n```\nNew instruction here\n```",
                "New instruction here",
            ),
            # Test stripping <think> tags without backticks
            (
                "<think>\nReasoning about the prompt...\n</think>\n\nNew instruction here",
                "New instruction here",
            ),
            # Test stripping multiple <think> blocks (backticks extract inner content)
            (
                "<think>\nFirst thought\n</think>\nSome text\n<think>\nSecond thought\n</think>\n\n```\nFinal instruction\n```",
                "Final instruction",
            ),
        ],
    )
    def test_extract_code_blocks(self, lm_output, expected_instruction):
        """Test extraction of instructions from various code block formats."""
        result = InstructionProposalSignature.output_extractor(lm_output)
        assert result["new_instruction"] == expected_instruction


class TestStripThinkTags:
    """Test the shared <think>-tag stripper."""

    def test_no_tags_passthrough(self):
        assert strip_think_tags("Just an instruction.") == "Just an instruction."

    def test_strips_closed_block(self):
        assert strip_think_tags("<think>reasoning</think>answer") == "answer"

    def test_strips_multiple_blocks(self):
        text = "<think>one</think>a<think>two</think>b"
        assert strip_think_tags(text) == "ab"

    def test_unclosed_tag_truncates_to_end(self):
        # Truncated reasoning: the answer never arrived, so nothing after the
        # dangling <think> should leak through.
        text = "prefix<think>reasoning that never closes"
        assert strip_think_tags(text) == "prefix"

    def test_closed_block_followed_by_unclosed_tag(self):
        text = "<think>done</think>answer<think>truncated"
        assert strip_think_tags(text) == "answer"
