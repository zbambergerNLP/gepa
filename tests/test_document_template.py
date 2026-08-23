# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for document templates."""

import pytest

from gepa.strategies.document_template import (
    TEMPLATE_FAMILIES,
    TEMPLATES,
    DocumentTemplate,
    EditTarget,
    MalformedDocumentError,
    infer_template_family,
    migrate_document,
)

PROMPT_TEMPLATE = TEMPLATES["system_prompt"]
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
        [
            "component_name",
            "section",
            "expected_name",
            "expected_label",
        ],
        [
            pytest.param(
                "sys",
                "Rules",
                "Rules",
                "sys:Rules",
                id="named_section",
            ),
            pytest.param(
                "sys",
                None,
                "whole",
                "sys:whole",
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

    def test_render_emits_only_populated_sections_in_order(self) -> None:
        """Test that render omits empty sections and preserves schema order."""
        headers = [line for line in PROMPT.splitlines() if line.startswith("## ")]
        assert headers == ["## Role", "## Rules", "## Output Format"]

    def test_render_then_parse_round_trips_bodies(self) -> None:
        """Test that parse recovers each rendered body and reports every section, including empty ones."""
        bodies = PROMPT_TEMPLATE.parse(PROMPT)
        assert bodies["Role"] == "you are a helper"
        assert bodies["Rules"] == "- be nice\n- be brief"
        assert bodies["Task"] == ""
        assert list(bodies) == list(PROMPT_TEMPLATE.sections)

    def test_empty_document_represents_an_unpopulated_template(self) -> None:
        """Test that an all-empty schema renders to no task-model text and still parses."""
        assert PROMPT_TEMPLATE.render({}) == ""
        assert PROMPT_TEMPLATE.parse("") == dict.fromkeys(PROMPT_TEMPLATE.sections, "")

    def test_explicit_empty_header_is_dropped_by_canonical_rendering(self) -> None:
        """Test that parsing can recover an empty header but canonical output omits it."""
        text = "## Role\nhelper\n\n## Reasoning\n\n## Output Format\none line\n"
        bodies = PROMPT_TEMPLATE.parse(text)
        assert bodies["Reasoning"] == ""
        assert "## Reasoning" not in PROMPT_TEMPLATE.render(bodies)

    def test_render_is_canonical_fixed_point(self) -> None:
        """Test that rendering the parse of a canonical document reproduces it exactly."""
        assert PROMPT_TEMPLATE.render(PROMPT_TEMPLATE.parse(PROMPT)) == PROMPT

    def test_parse_tolerates_sub_headers_and_whitespace(self) -> None:
        """Test that parse keeps '### ' sub-headers inside a body and ignores leading whitespace."""
        text = "\n" + PROMPT_TEMPLATE.render({**BODIES, "Examples": "### Example 1\nin -> out"})
        assert PROMPT_TEMPLATE.parse(text)["Examples"] == "### Example 1\nin -> out"

    @pytest.mark.parametrize(
        [
            "text",
            "expected_message_substr",
        ],
        [
            pytest.param(
                "## Rules\nbe brief\n\n## Role\nhelper\n",
                "in that order",
                id="wrong_order",
            ),
            pytest.param(
                PROMPT + "## Notes\nextra\n",
                "Notes",
                id="extra_header",
            ),
            pytest.param(
                "## Role\nhelper\n\n## Role\nexpert\n",
                "at most once",
                id="duplicate_header",
            ),
            pytest.param(
                "# Title\n" + PROMPT,
                "before the first",
                id="content_before_first_header",
            ),
            pytest.param(
                "just some unstructured text",
                None,
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

    def test_replace_section_body_adds_and_removes_sparse_sections(self) -> None:
        """Test that an omitted section can be added in order and removed when cleared."""
        with_reasoning = PROMPT_TEMPLATE.replace_section_body(PROMPT, "Reasoning", "Check the answer.")
        headers = [line for line in with_reasoning.splitlines() if line.startswith("## ")]
        assert headers == ["## Role", "## Rules", "## Reasoning", "## Output Format"]
        assert PROMPT_TEMPLATE.parse(with_reasoning)["Reasoning"] == "Check the answer."
        assert PROMPT_TEMPLATE.replace_section_body(with_reasoning, "Reasoning", "") == PROMPT

    def test_replace_section_body_rejects_structural_header_text(self) -> None:
        """Test that a proposer cannot smuggle document structure into a section body."""
        with pytest.raises(MalformedDocumentError, match="cannot contain"):
            PROMPT_TEMPLATE.replace_section_body(PROMPT, "Rules", "be brief\n## Reasoning\nthink")

    def test_rendered_sections_reports_only_present_headers(self) -> None:
        """Test that callers can protect the raw header sequence during whole-document edits."""
        assert PROMPT_TEMPLATE.rendered_sections(PROMPT) == ("Role", "Rules", "Output Format")

    def test_section_span_rejects_an_omitted_section(self) -> None:
        """Test that byte spans are available only for sections present in rendered text."""
        with pytest.raises(MalformedDocumentError, match="omitted"):
            PROMPT_TEMPLATE.section_body_span(PROMPT, "Reasoning")

    def test_templates_cover_system_user_and_skill_components(self) -> None:
        """Test that generic system/user prompts share one shape and skills remain distinct."""
        assert set(TEMPLATES) == {"system_prompt", "user_prompt", "skill"}
        assert TEMPLATES["system_prompt"] is TEMPLATES["user_prompt"]
        assert list(TEMPLATES["skill"].sections) == ["Name", "Description", "Instructions", "Examples"]

    def test_custom_kind_is_just_a_template(self) -> None:
        """Test that an ad-hoc DocumentTemplate round-trips its own bodies."""
        template = DocumentTemplate("note", {"Body": "the text"})
        assert template.parse(template.render({"Body": "hi"})) == {"Body": "hi"}


class TestTemplateFamilies:
    """Test cases for the TEMPLATE_FAMILIES registry."""

    def test_registry_covers_the_known_families(self) -> None:
        """Test that the registry holds the provider families and that generic is the TEMPLATES object itself."""
        assert set(TEMPLATE_FAMILIES) == {"generic", "openai", "anthropic", "google", "alibaba"}
        assert TEMPLATE_FAMILIES["generic"] is TEMPLATES

    def test_skill_kind_is_provider_invariant(self) -> None:
        """Test that every family shares the one skill template object (the skill shape has no provider guide)."""
        assert all(kinds["skill"] is TEMPLATES["skill"] for kinds in TEMPLATE_FAMILIES.values())

    def test_provider_role_templates_are_distinct_when_their_schemas_differ(self) -> None:
        """Test that only the generic family aliases its system and user templates."""
        assert TEMPLATE_FAMILIES["generic"]["system_prompt"] is TEMPLATE_FAMILIES["generic"]["user_prompt"]
        assert all(
            kinds["system_prompt"] is not kinds["user_prompt"]
            for family, kinds in TEMPLATE_FAMILIES.items()
            if family != "generic"
        )

    @pytest.mark.parametrize(
        [
            "family",
            "component_kind",
            "expected_sections",
        ],
        [
            pytest.param(
                "generic",
                "system_prompt",
                ["Role", "Task", "Context", "Rules", "Reasoning", "Examples", "Output Format"],
                id="generic_system_prompt",
            ),
            pytest.param(
                "generic",
                "user_prompt",
                ["Role", "Task", "Context", "Rules", "Reasoning", "Examples", "Output Format"],
                id="generic_user_prompt",
            ),
            pytest.param(
                "openai",
                "system_prompt",
                ["Identity", "Instructions", "Examples", "Context"],
                id="openai_system_prompt",
            ),
            pytest.param(
                "openai",
                "user_prompt",
                ["Input"],
                id="openai_user_prompt",
            ),
            pytest.param(
                "anthropic",
                "system_prompt",
                ["Role", "Instructions"],
                id="anthropic_system_prompt",
            ),
            pytest.param(
                "anthropic",
                "user_prompt",
                ["Context", "Examples", "Output Format", "Reasoning", "Instructions"],
                id="anthropic_user_prompt",
            ),
            pytest.param(
                "google",
                "system_prompt",
                ["Role", "Instructions", "Constraints", "Output Format"],
                id="google_system_prompt",
            ),
            pytest.param(
                "google",
                "user_prompt",
                ["Context", "Task", "Final Instruction"],
                id="google_user_prompt",
            ),
            pytest.param(
                "alibaba",
                "system_prompt",
                ["Objective", "Style", "Tone", "Audience", "Response"],
                id="alibaba_system_prompt",
            ),
            pytest.param(
                "alibaba",
                "user_prompt",
                ["Context", "Objective"],
                id="alibaba_user_prompt",
            ),
        ],
    )
    def test_message_sections_follow_the_provider_guide(
        self,
        family: str,
        component_kind: str,
        expected_sections: list[str],
    ) -> None:
        """Test that each message role has its provider-specific sections and round-trips.

        Args:
            family: The TEMPLATE_FAMILIES key under test.
            component_kind: System or user prompt registry key.
            expected_sections: Exact ordered section names for the role.
        """
        template = TEMPLATE_FAMILIES[family][component_kind]
        assert template.kind == "prompt"
        assert list(template.sections) == expected_sections
        bodies = {section: f"text for {section}" for section in expected_sections}
        assert template.parse(template.render(bodies)) == bodies


class TestInferTemplateFamily:
    """Test cases for infer_template_family."""

    @pytest.mark.parametrize(
        [
            "model",
            "expected_family",
        ],
        [
            pytest.param(
                "anthropic/claude-opus-4",
                "anthropic",
                id="claude_maps_to_anthropic",
            ),
            pytest.param(
                "gemini/gemini-2.5-pro",
                "google",
                id="gemini_maps_to_google",
            ),
            pytest.param(
                "gemma-3-27b-it",
                "google",
                id="gemma_maps_to_google",
            ),
            pytest.param(
                "gpt-4.1-mini",
                "openai",
                id="gpt_maps_to_openai",
            ),
            pytest.param("openai/gpt-5", "openai", id="provider_prefixed_gpt_maps_to_openai"),
            pytest.param(
                "openai/o3",
                "openai",
                id="o_series_with_provider_prefix_maps_to_openai",
            ),
            pytest.param(
                "o4-mini",
                "openai",
                id="bare_o_series_maps_to_openai",
            ),
            pytest.param(
                "azure/o1-preview",
                "openai",
                id="o_series_behind_other_provider_maps_to_openai",
            ),
            pytest.param(
                "openai/some-future-model",  # model
                "openai",  # expected_family
                id="openai_prefix_alone_maps_to_openai",
            ),
            pytest.param(
                "dashscope/qwen-max",  # model
                "alibaba",  # expected_family
                id="qwen_maps_to_alibaba",
            ),
            pytest.param(
                "qwq-32b",  # model
                "alibaba",  # expected_family
                id="qwq_maps_to_alibaba",
            ),
            pytest.param(
                "groq/llama-3.3-70b",  # model
                "generic",  # expected_family (Meta prescribes no prompt structure)
                id="llama_maps_to_generic",
            ),
            pytest.param(
                "meta/muse-spark-1.2",  # model
                "generic",  # expected_family (Meta prescribes no prompt structure)
                id="muse_maps_to_generic",
            ),
            pytest.param(
                "mistral/mistral-large",  # model
                "generic",  # expected_family
                id="unknown_provider_maps_to_generic",
            ),
            pytest.param(
                "solo",  # model
                "generic",  # expected_family (no false positive on the o-series pattern)
                id="lone_o_without_digits_maps_to_generic",
            ),
            pytest.param(
                None,  # model
                "generic",  # expected_family
                id="none_maps_to_generic",
            ),
            pytest.param(
                "",  # model
                "generic",  # expected_family
                id="empty_string_maps_to_generic",
            ),
        ],
    )
    def test_model_name_maps_to_its_provider_family(
        self,
        model: str | None,
        expected_family: str,
    ) -> None:
        """Test that infer_template_family maps a model identifier to its provider's template family.

        Args:
            model: The (LiteLLM-style) model identifier, or None when no task model name is available.
            expected_family: The TEMPLATE_FAMILIES key the identifier must map to.
        """
        assert infer_template_family(model) == expected_family


class TestMigrateDocument:
    """Test cases for migrate_document."""

    def test_already_canonical_text_returned_without_lm_call(self) -> None:
        """Test that canonical text bypasses the LM."""
        lm = FakeMigrationLM()
        assert migrate_document(PROMPT, PROMPT_TEMPLATE, lm) == PROMPT
        assert lm.calls == []

    def test_structured_text_with_empty_header_is_canonicalized_without_lm_call(self) -> None:
        """Test that migration removes an explicit empty section without consulting the LM."""
        lm = FakeMigrationLM()
        text = PROMPT.replace("## Output Format", "## Reasoning\n\n## Output Format")
        assert migrate_document(text, PROMPT_TEMPLATE, lm) == PROMPT
        assert lm.calls == []

    def test_free_form_text_is_restructured_and_normalized(self) -> None:
        """Test that free-form text is restructured by the LM, re-normalized, and prompted with the section guide."""
        lm = FakeMigrationLM(f"<think>hmm</think>sure:\n<document>\n{PROMPT}\n\n</document>")
        migrated = migrate_document("You help people. Be nice and brief.", PROMPT_TEMPLATE, lm)
        assert migrated == PROMPT
        prompt = lm.calls[0]
        assert "You help people. Be nice and brief." in prompt
        assert "## Rules: Constraints" in prompt  # section guide reaches the LM
        assert "Never emit an empty section header" in prompt

    def test_non_conforming_reply_is_retried_with_error(self) -> None:
        """Test that a non-conforming first reply triggers one retry carrying the parse error."""
        lm = FakeMigrationLM("<document>no headers here</document>", f"<document>{PROMPT}</document>")
        assert migrate_document("free text", PROMPT_TEMPLATE, lm) == PROMPT
        assert len(lm.calls) == 2
        assert "Your previous reply was rejected" in lm.calls[1]

    def test_nonempty_source_rejects_an_empty_migration(self) -> None:
        """Test that sparse empty documents cannot silently discard free-form source text."""
        lm = FakeMigrationLM("<document></document>", f"<document>{PROMPT}</document>")
        assert migrate_document("free text", PROMPT_TEMPLATE, lm) == PROMPT
        assert "Migration removed all content" in lm.calls[1]

    def test_gives_up_after_retry(self) -> None:
        """Test that migrate_document raises once the retry budget is exhausted without a conforming reply."""
        lm = FakeMigrationLM("nope", "still nope")
        with pytest.raises(MalformedDocumentError, match="Could not migrate"):
            migrate_document("free text", PROMPT_TEMPLATE, lm)
        assert len(lm.calls) == 2
