# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for the atomic edit tools."""

import pytest

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
        [
            "text",
            "edit",
            "expected_text",
            "expected_ops",
            "expected_exception",
        ],
        [
            pytest.param(
                "hello world",
                InsertTextArgs(anchor="hello", where="after", text=" big"),
                "hello big world",
                ["INSERT ' big' after 'hello'"],
                None,
                id="insert_after_anchor",
            ),
            pytest.param(
                "hello world",
                InsertTextArgs(anchor="world", where="before", text="big "),
                "hello big world",
                ["INSERT 'big ' before 'world'"],
                None,
                id="insert_before_anchor",
            ),
            pytest.param(
                "hello",
                InsertTextArgs(text="!"),
                "hello!",
                ["INSERT '!' at end"],
                None,
                id="insert_at_end_without_anchor",
            ),
            pytest.param(
                "hello",
                InsertTextArgs(text=""),
                None,
                None,
                EditApplicationError,
                id="insert_empty_text_raises",
            ),
            pytest.param(
                "hello",
                InsertTextArgs(anchor="zzz", text="x"),
                None,
                None,
                EditApplicationError,
                id="insert_missing_anchor_raises",
            ),
            pytest.param(
                "hello world",
                DeleteTextArgs(target=" world"),
                "hello",
                ["DELETE ' world'"],
                None,
                id="delete_target",
            ),
            pytest.param(
                "a a a",
                DeleteTextArgs(target="a "),
                "a a",
                ["DELETE 'a '"],
                None,
                id="delete_only_first_occurrence",
            ),
            pytest.param(
                "hello",
                DeleteTextArgs(target="zzz"),
                None,
                None,
                EditApplicationError,
                id="delete_missing_target_raises",
            ),
            pytest.param(
                "be nice",
                ReplaceTextArgs(target="nice", text="concise"),
                "be concise",
                ["DELETE 'nice'", "INSERT 'concise'"],
                None,
                id="replace_decomposes_to_delete_then_insert",
            ),
            pytest.param(
                "hello",
                ReplaceTextArgs(target="zzz", text="x"),
                None,
                None,
                EditApplicationError,
                id="replace_missing_target_raises",
            ),
            pytest.param(
                "A B C",
                MoveTextArgs(target="A ", anchor="C", where="after"),
                "B CA ",
                ["DELETE 'A '", "INSERT (moved) 'A ' after 'C'"],
                None,
                id="move_decomposes_to_delete_then_insert",
            ),
            pytest.param(
                "A B C",
                MoveTextArgs(target="A ", anchor=""),
                None,
                None,
                EditApplicationError,
                id="move_requires_anchor",
            ),
            pytest.param(
                "A B C",
                MoveTextArgs(target="zzz", anchor="C"),
                None,
                None,
                EditApplicationError,
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
