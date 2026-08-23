# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for document templates (:mod:`gepa.strategies.document_template`).

Covers EditTarget naming, DocumentTemplate render/parse round-tripping and its
rejection of malformed documents, the TEMPLATES registry, the provider template
families (TEMPLATE_FAMILIES and infer_template_family), and migrate_document's
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
    TEMPLATE_FAMILIES,
    TEMPLATES,
    DocumentTemplate,
    EditTarget,
    MalformedDocumentError,
    infer_template_family,
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


class TestTemplateFamilies:
    """Test cases for the TEMPLATE_FAMILIES registry."""

    def test_registry_covers_the_known_families(self) -> None:
        """Test that the registry holds the provider families and that generic is the TEMPLATES object itself."""
        assert set(TEMPLATE_FAMILIES) == {"generic", "openai", "openai-gpt-5.6", "anthropic", "google", "alibaba"}
        assert TEMPLATE_FAMILIES["generic"] is TEMPLATES

    def test_skill_kind_is_provider_invariant(self) -> None:
        """Test that every family shares the one skill template object (the skill shape has no provider guide)."""
        assert all(kinds["skill"] is TEMPLATES["skill"] for kinds in TEMPLATE_FAMILIES.values())

    @pytest.mark.parametrize(
        # Parameter names
        [
            "family",
            "expected_sections",
        ],
        # Parameter values
        [
            pytest.param(
                "generic",  # family
                ["Role", "Task", "Context", "Rules", "Reasoning", "Examples", "Output Format"],  # expected_sections
                id="generic_papers_grounded_taxonomy",
            ),
            pytest.param(
                "openai",  # family
                [  # expected_sections (the prompt-engineering guide's skeleton: identity first, context last)
                    "Identity",
                    "Instructions",
                    "Examples",
                    "Context",
                ],
                id="openai_identity_first_context_last",
            ),
            pytest.param(
                "openai-gpt-5.6",  # family
                [  # expected_sections (the GPT-5.6 family guide's suggested prompt structure)
                    "Role",
                    "Personality",
                    "Goal",
                    "Success Criteria",
                    "Constraints",
                    "Tools",
                    "Output",
                    "Stop Rules",
                ],
                id="gpt56_role_first_stop_rules_last",
            ),
            pytest.param(
                "anthropic",  # family
                [  # expected_sections (Claude best practices: longform data first, instructions last)
                    "Role",
                    "Context",
                    "Examples",
                    "Reasoning",
                    "Output Format",
                    "Instructions",
                ],
                id="anthropic_data_first_instructions_last",
            ),
            pytest.param(
                "google",  # family
                [  # expected_sections (the Gemini best-practices template: role first, closing reminder last)
                    "Role",
                    "Instructions",
                    "Constraints",
                    "Output Format",
                    "Context",
                    "Task",
                    "Final Instruction",
                ],
                id="google_role_first_final_instruction_last",
            ),
            pytest.param(
                "alibaba",  # family
                [  # expected_sections (Model Studio's six-part framework, known as CO-STAR: context first, response last)
                    "Context",
                    "Objective",
                    "Style",
                    "Tone",
                    "Audience",
                    "Response",
                ],
                id="alibaba_costar_framework",
            ),
        ],
    )
    def test_prompt_sections_follow_the_provider_guide(
        self,
        family: str,
        expected_sections: list[str],
    ) -> None:
        """Test that each family's prompt template has its guide's sections, in order, and round-trips.

        Args:
            family: The TEMPLATE_FAMILIES key under test.
            expected_sections: The exact ordered section names the family's prompt template must define.
        """
        template = TEMPLATE_FAMILIES[family]["prompt"]
        assert template.kind == "prompt"
        assert list(template.sections) == expected_sections
        bodies = {section: f"text for {section}" for section in expected_sections}
        assert template.parse(template.render(bodies)) == bodies


class TestInferTemplateFamily:
    """Test cases for infer_template_family."""

    @pytest.mark.parametrize(
        # Parameter names
        [
            "model",
            "expected_family",
        ],
        # Parameter values
        [
            pytest.param(
                "anthropic/claude-opus-4",  # model
                "anthropic",  # expected_family
                id="claude_maps_to_anthropic",
            ),
            pytest.param(
                "gemini/gemini-2.5-pro",  # model
                "google",  # expected_family
                id="gemini_maps_to_google",
            ),
            pytest.param(
                "gemma-3-27b-it",  # model
                "google",  # expected_family
                id="gemma_maps_to_google",
            ),
            pytest.param(
                "gpt-4.1-mini",  # model
                "openai",  # expected_family
                id="gpt_maps_to_openai",
            ),
            pytest.param(
                "openai/gpt-5.6-sol",  # model
                "openai-gpt-5.6",  # expected_family (the model-specific family wins over the provider one)
                id="gpt56_maps_to_its_model_specific_family",
            ),
            pytest.param(
                "gpt-5.6",  # model
                "openai-gpt-5.6",  # expected_family
                id="bare_gpt56_maps_to_its_model_specific_family",
            ),
            pytest.param(
                "openai/o3",  # model
                "openai",  # expected_family
                id="o_series_with_provider_prefix_maps_to_openai",
            ),
            pytest.param(
                "o4-mini",  # model
                "openai",  # expected_family
                id="bare_o_series_maps_to_openai",
            ),
            pytest.param(
                "azure/o1-preview",  # model
                "openai",  # expected_family
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
