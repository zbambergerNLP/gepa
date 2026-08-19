# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for document templates (:mod:`gepa.strategies.document_template`).

Covers EditTarget naming, DocumentTemplate render/parse round-tripping and its
rejection of malformed documents, the TEMPLATES registry, and migrate_document's
skip/restructure/retry/give-up paths. The migration LM is a scripted in-memory fake.

Expected usage:
```bash
pytest tests/test_document_template.py -vv
```
"""

# Third-party imports
import pytest

# Local imports
from gepa.strategies.document_template import (
    TEMPLATES,
    DocumentTemplate,
    EditTarget,
    MalformedDocumentError,
    migrate_document,
)

# ====================== #
# Test Fakes and Helpers #
# ====================== #


PROMPT_TEMPLATE = TEMPLATES["prompt"]
BODIES = {"Role": "you are a helper", "Rules": "- be nice\n- be brief", "Output Format": "one line"}
PROMPT = PROMPT_TEMPLATE.render(BODIES)


class FakeMigrationLM:
    """A fake migration LM: records prompts, replays scripted replies in order."""

    def __init__(self, *outputs: str) -> None:
        self.outputs = list(outputs)
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        """Record ``prompt`` and return the next scripted reply."""
        self.calls.append(prompt)
        return self.outputs.pop(0)


class TestEditTarget:
    """Test cases for EditTarget."""

    @pytest.mark.parametrize(
        # Parameter names
        [
            "component_name",
            "section",
            "expected_name",
            "expected_label",
        ],
        # Parameter values
        [
            pytest.param(
                "sys",  # component_name
                "Rules",  # section
                "Rules",  # expected_name
                "sys:Rules",  # expected_label
                id="named_section",
            ),
            pytest.param(
                "sys",  # component_name
                None,  # section
                "whole",  # expected_name
                "sys:whole",  # expected_label
                id="whole_document",
            ),
        ],
    )
    def test_edit_target_name_and_label(
        self,
        component_name: str,
        section: str | None,
        expected_name: str,
        expected_label: str,
    ) -> None:
        """Test that EditTarget exposes the section name (or "whole") and its component-scoped label.

        Args:
            component_name: The candidate component the target belongs to.
            section: The template section addressed, or None for the whole document.
            expected_name: The short region name EditTarget.name must return.
            expected_label: The "component:name" label EditTarget.label must return.
        """
        target = EditTarget(component_name, section)
        assert target.name == expected_name
        assert target.label == expected_label


class TestDocumentTemplate:
    """Test cases for DocumentTemplate."""

    def test_render_emits_every_section_in_order(self) -> None:
        """Test that render emits one '## <Section>' header per template section, in order."""
        headers = [line for line in PROMPT.splitlines() if line.startswith("## ")]
        assert headers == [f"## {s}" for s in PROMPT_TEMPLATE.sections]

    def test_render_then_parse_round_trips_bodies(self) -> None:
        """Test that parse recovers each rendered body and reports every section, including empty ones."""
        bodies = PROMPT_TEMPLATE.parse(PROMPT)
        assert bodies["Role"] == "you are a helper"
        assert bodies["Rules"] == "- be nice\n- be brief"
        assert bodies["Task"] == ""  # missing at render time -> empty body, header still present
        assert list(bodies) == list(PROMPT_TEMPLATE.sections)

    def test_render_is_canonical_fixed_point(self) -> None:
        """Test that rendering the parse of a canonical document reproduces it exactly."""
        assert PROMPT_TEMPLATE.render(PROMPT_TEMPLATE.parse(PROMPT)) == PROMPT

    def test_parse_tolerates_sub_headers_and_whitespace(self) -> None:
        """Test that parse keeps '### ' sub-headers inside a body and ignores leading whitespace."""
        text = "\n" + PROMPT_TEMPLATE.render({**BODIES, "Examples": "### Example 1\nin -> out"})
        assert PROMPT_TEMPLATE.parse(text)["Examples"] == "### Example 1\nin -> out"

    @pytest.mark.parametrize(
        # Parameter names
        [
            "text",
            "expected_message_substr",
        ],
        # Parameter values
        [
            pytest.param(
                PROMPT.replace("## Reasoning\n", ""),  # text
                "Reasoning",  # expected_message_substr
                id="missing_section",
            ),
            pytest.param(
                "## Task\n\n" + PROMPT.replace("## Task\n", "", 1),  # text
                "in that order",  # expected_message_substr
                id="wrong_order",
            ),
            pytest.param(
                PROMPT + "## Notes\nextra\n",  # text
                "Notes",  # expected_message_substr
                id="extra_header",
            ),
            pytest.param(
                "# Title\n" + PROMPT,  # text
                "before the first",  # expected_message_substr
                id="content_before_first_header",
            ),
            pytest.param(
                "just some unstructured text",  # text
                None,  # expected_message_substr
                id="unstructured_text",
            ),
        ],
    )
    def test_parse_rejects_malformed_document(
        self,
        text: str,
        expected_message_substr: str | None,
    ) -> None:
        """Test that parse raises MalformedDocumentError for documents that break the canonical format.

        Args:
            text: A document that violates the template's section contract.
            expected_message_substr: A substring the error message must contain, or None when only the
                exception type matters.
        """
        with pytest.raises(MalformedDocumentError) as exc_info:
            PROMPT_TEMPLATE.parse(text)
        if expected_message_substr is not None:
            assert expected_message_substr in str(exc_info.value)

    def test_edit_targets_are_every_section_then_whole(self) -> None:
        """Test that edit_targets lists every section in order, then the whole-document target."""
        targets = PROMPT_TEMPLATE.edit_targets("sys")
        assert [t.section for t in targets] == [*PROMPT_TEMPLATE.sections, None]
        assert all(t.component_name == "sys" for t in targets)

    def test_templates_cover_prompt_and_skill(self) -> None:
        """Test that the TEMPLATES registry holds the prompt and skill kinds with the skill's sections."""
        assert set(TEMPLATES) == {"prompt", "skill"}
        assert list(TEMPLATES["skill"].sections) == ["Name", "Description", "Instructions", "Examples"]

    def test_custom_kind_is_just_a_template(self) -> None:
        """Test that an ad-hoc DocumentTemplate round-trips its own bodies."""
        template = DocumentTemplate("note", {"Body": "the text"})
        assert template.parse(template.render({"Body": "hi"})) == {"Body": "hi"}


class TestMigrateDocument:
    """Test cases for migrate_document."""

    def test_already_canonical_text_returned_untouched_without_lm_call(self) -> None:
        """Test that text already in canonical format is returned unchanged and never reaches the LM."""
        lm = FakeMigrationLM()
        assert migrate_document(PROMPT, PROMPT_TEMPLATE, lm) == PROMPT
        assert lm.calls == []

    def test_free_form_text_is_restructured_and_normalized(self) -> None:
        """Test that free-form text is restructured by the LM, re-normalized, and prompted with the section guide."""
        lm = FakeMigrationLM(f"<think>hmm</think>sure:\n<document>\n{PROMPT}\n\n</document>")
        migrated = migrate_document("You help people. Be nice and brief.", PROMPT_TEMPLATE, lm)
        assert migrated == PROMPT
        prompt = lm.calls[0]
        assert "You help people. Be nice and brief." in prompt
        assert "## Rules: Constraints" in prompt  # section guide reaches the LM

    def test_non_conforming_reply_is_retried_with_error(self) -> None:
        """Test that a non-conforming first reply triggers one retry carrying the parse error."""
        lm = FakeMigrationLM("<document>no headers here</document>", f"<document>{PROMPT}</document>")
        assert migrate_document("free text", PROMPT_TEMPLATE, lm) == PROMPT
        assert len(lm.calls) == 2
        assert "Your previous reply was rejected" in lm.calls[1]

    def test_gives_up_after_retry(self) -> None:
        """Test that migrate_document raises once the retry budget is exhausted without a conforming reply."""
        lm = FakeMigrationLM("nope", "still nope")
        with pytest.raises(MalformedDocumentError, match="Could not migrate"):
            migrate_document("free text", PROMPT_TEMPLATE, lm)
        assert len(lm.calls) == 2
