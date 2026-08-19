# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for the atomic edit tools (:mod:`gepa.strategies.edit_tools`).

Exercises apply_edit across the INSERT/DELETE/REPLACE/MOVE tools, checking both the
edited text and the atomic-operation log it reports, plus the EditApplicationError
raised for empty or unmatched arguments. Nothing is mocked; apply_edit is pure.

Expected usage:
```bash
pytest tests/test_edit_tools.py -vv
```
"""

# Third-party imports
import pytest

# Local imports
from gepa.strategies.edit_tools import (
    DeleteTextArgs,
    EditApplicationError,
    EditArgs,
    InsertTextArgs,
    MoveTextArgs,
    ReplaceTextArgs,
    apply_edit,
)


class TestApplyEdit:
    """Test cases for apply_edit."""

    @pytest.mark.parametrize(
        # Parameter names
        [
            "text",
            "edit",
            "expected_text",
            "expected_ops",
            "expected_exception",
        ],
        # Parameter values
        [
            pytest.param(
                "hello world",  # text
                InsertTextArgs(anchor="hello", where="after", text=" big"),  # edit
                "hello big world",  # expected_text
                ["INSERT ' big' after 'hello'"],  # expected_ops
                None,  # expected_exception
                id="insert_after_anchor",
            ),
            pytest.param(
                "hello world",  # text
                InsertTextArgs(anchor="world", where="before", text="big "),  # edit
                "hello big world",  # expected_text
                ["INSERT 'big ' before 'world'"],  # expected_ops
                None,  # expected_exception
                id="insert_before_anchor",
            ),
            pytest.param(
                "hello",  # text
                InsertTextArgs(text="!"),  # edit
                "hello!",  # expected_text
                ["INSERT '!' at end"],  # expected_ops
                None,  # expected_exception
                id="insert_at_end_without_anchor",
            ),
            pytest.param(
                "hello",  # text
                InsertTextArgs(text=""),  # edit
                None,  # expected_text
                None,  # expected_ops
                EditApplicationError,  # expected_exception
                id="insert_empty_text_raises",
            ),
            pytest.param(
                "hello",  # text
                InsertTextArgs(anchor="zzz", text="x"),  # edit
                None,  # expected_text
                None,  # expected_ops
                EditApplicationError,  # expected_exception
                id="insert_missing_anchor_raises",
            ),
            pytest.param(
                "hello world",  # text
                DeleteTextArgs(target=" world"),  # edit
                "hello",  # expected_text
                ["DELETE ' world'"],  # expected_ops
                None,  # expected_exception
                id="delete_target",
            ),
            pytest.param(
                "a a a",  # text
                DeleteTextArgs(target="a "),  # edit
                "a a",  # expected_text
                ["DELETE 'a '"],  # expected_ops
                None,  # expected_exception
                id="delete_only_first_occurrence",
            ),
            pytest.param(
                "hello",  # text
                DeleteTextArgs(target="zzz"),  # edit
                None,  # expected_text
                None,  # expected_ops
                EditApplicationError,  # expected_exception
                id="delete_missing_target_raises",
            ),
            pytest.param(
                "be nice",  # text
                ReplaceTextArgs(target="nice", text="concise"),  # edit
                "be concise",  # expected_text
                ["DELETE 'nice'", "INSERT 'concise'"],  # expected_ops
                None,  # expected_exception
                id="replace_decomposes_to_delete_then_insert",
            ),
            pytest.param(
                "hello",  # text
                ReplaceTextArgs(target="zzz", text="x"),  # edit
                None,  # expected_text
                None,  # expected_ops
                EditApplicationError,  # expected_exception
                id="replace_missing_target_raises",
            ),
            pytest.param(
                "A B C",  # text
                MoveTextArgs(target="A ", anchor="C", where="after"),  # edit
                "B CA ",  # expected_text
                ["DELETE 'A '", "INSERT (moved) 'A ' after 'C'"],  # expected_ops
                None,  # expected_exception
                id="move_decomposes_to_delete_then_insert",
            ),
            pytest.param(
                "A B C",  # text
                MoveTextArgs(target="A ", anchor=""),  # edit
                None,  # expected_text
                None,  # expected_ops
                EditApplicationError,  # expected_exception
                id="move_requires_anchor",
            ),
            pytest.param(
                "A B C",  # text
                MoveTextArgs(target="zzz", anchor="C"),  # edit
                None,  # expected_text
                None,  # expected_ops
                EditApplicationError,  # expected_exception
                id="move_missing_target_raises",
            ),
        ],
    )
    def test_apply_edit(
        self,
        text: str,
        edit: EditArgs,
        expected_text: str | None,
        expected_ops: list[str] | None,
        expected_exception: type[BaseException] | None,
    ) -> None:
        """Test that apply_edit applies each atomic operation and rejects malformed arguments.

        Args:
            text: The document the edit is applied to.
            edit: The typed edit arguments to apply.
            expected_text: The document after the edit, or None when an exception is expected.
            expected_ops: The primitive-operation log the edit must report, or None when an exception is expected.
            expected_exception: The exception apply_edit must raise, or None when the edit must succeed.
        """
        if expected_exception is not None:
            with pytest.raises(expected_exception):
                apply_edit(text, edit)
            return

        new_text, ops = apply_edit(text, edit)
        assert new_text == expected_text
        assert ops == expected_ops
