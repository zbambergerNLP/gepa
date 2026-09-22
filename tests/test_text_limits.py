"""Verify unlimited defaults and optional character budgets at model boundaries."""

import json
from types import SimpleNamespace

import pytest

from gepa.adapters.optimize_anything_adapter.optimize_anything_adapter import OptimizeAnythingAdapter
from gepa.gepa_launcher import RefinerConfig
from gepa.proposer.reflective_mutation.react_v2_proposer import ReActV2Proposer
from gepa.proposer.reflective_mutation.reflection_lm import StatelessReflectionLM
from gepa.strategies.action_space import VerbalizedActionSelector
from gepa.strategies.document_template import TEMPLATES, EditTarget
from gepa.strategies.edit_tools import EditTool
from gepa.strategies.intervention import summarize_feedback
from gepa.strategies.text_limits import TextLimitError, TextLimits, clip_text, parse_text_limits


class RecordingLM:
    """Capture requests without contacting a provider."""

    def __init__(self, response: str = "```\nrevised instruction\n```") -> None:
        """Set a deterministic provider response."""
        self.response = response
        self.calls: list = []

    def __call__(self, prompt):
        """Record the unmodified request and return the configured response."""
        self.calls.append(prompt)
        return self.response


@pytest.mark.parametrize("field", TextLimits.__dataclass_fields__)
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "100"])
def test_invalid_limits_fail_before_any_execution(field: str, value: object) -> None:
    """Reject ambiguous, nonpositive, or noninteger character budgets."""
    with pytest.raises(ValueError, match="positive integer or null"):
        parse_text_limits(json.dumps({field: value}))


def test_json_defaults_and_unknown_fields() -> None:
    """Preserve unlimited defaults and reject typos instead of ignoring them."""
    assert all(value is None for value in TextLimits().to_dict().values())
    assert parse_text_limits('{"max_component_chars":10000,"selector_target_chars":null}') == TextLimits(
        max_component_chars=10000
    )
    with pytest.raises(ValueError, match="Unknown text_limits"):
        parse_text_limits('{"max_componnent_chars":10000}')


def test_unicode_excerpts_count_characters_and_preserve_originals() -> None:
    """Keep complete Unicode by default and use explicit markers when capped."""
    text = "אבגדהוזחט🙂"
    assert clip_text(text, None) == text
    assert clip_text(text, 10) == text
    assert clip_text(text, 5, head_and_tail=True) == "אבג\n[... 5 characters omitted ...]\nט🙂"  # noqa: RUF001
    assert clip_text(text, 1, head_and_tail=True).startswith("א\n[")


def test_feedback_is_whole_unless_explicitly_capped() -> None:
    """Keep feedback beyond both former selector cutoffs, including late evidence."""
    feedback = "Evidence. " * 1200 + "LATE_FAILURE"
    entries = [{"Feedback": feedback}]
    assert summarize_feedback(entries) == feedback
    assert StatelessReflectionLM._summarize_feedback({"prompt": entries}) == feedback
    for result in (summarize_feedback(entries, 50), StatelessReflectionLM._summarize_feedback({"prompt": entries}, 50)):
        assert result.startswith(feedback[:50])
        assert "characters omitted" in result
        assert "LATE_FAILURE" not in result


@pytest.mark.parametrize("target", [None, 7000])
def test_selector_target_is_optional_guidance(target: int | None) -> None:
    """Expose a configured size preference without rejecting long components."""
    lm = RecordingLM("<response><candidate><action>edit</action><probability>1</probability></candidate></response>")
    item = SimpleNamespace(menu_id="edit", menu_description="Improve the section")
    selector = VerbalizedActionSelector([item], lm, text_limits=TextLimits(selector_target_chars=target))
    selector.select(1, candidate="x" * 15000, feedback_summary="Improve accuracy")
    assert "x" * 15000 in lm.calls[0]
    assert ("size target: ~7000" in lm.calls[0]) == (target is not None)


@pytest.mark.parametrize("role", ["controller", "stateless", "editor"])
def test_assembled_prompt_limit_prevents_model_calls(role: str) -> None:
    """Check the complete role prompt instead of truncating one input field."""
    lm = RecordingLM()
    limits = TextLimits(max_prompt_chars=10)
    with pytest.raises(TextLimitError, match="max_prompt_chars=10"):
        if role == "controller":
            item = SimpleNamespace(menu_id="edit", menu_description="Improve")
            VerbalizedActionSelector([item], lm, text_limits=limits).select(1, candidate="body", feedback_summary="bad")
        elif role == "stateless":
            StatelessReflectionLM(lm, text_limits=limits).reflect(
                {"prompt": "body"}, {"prompt": [{"Feedback": "bad"}]}, ["prompt"]
            )
        else:
            ReActV2Proposer(lm, TEMPLATES["system_prompt"], [EditTool.REPLACE_TEXT], text_limits=limits).propose(
                "body", EditTarget("sys", "Rules"), EditTool.REPLACE_TEXT, "Improve", "bad", "trace", [], None
            )
    assert lm.calls == []


def test_prompt_budget_includes_native_tools_and_unicode() -> None:
    """Count serialized messages and tool schemas without escaping Unicode."""
    messages = [{"role": "user", "content": "שלום"}]
    tools = [{"name": "edit", "description": "A long tool description"}]
    size = len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")))
    TextLimits(max_prompt_chars=size).check_prompt(messages)
    with pytest.raises(TextLimitError):
        TextLimits(max_prompt_chars=size).check_prompt(messages, tools)


@pytest.mark.parametrize("limits", [TextLimits(max_component_chars=5), TextLimits(max_candidate_chars=25)])
def test_stateless_reconstructed_documents_respect_configured_limits(limits: TextLimits) -> None:
    """Drop oversized rewrites while preserving original documents and evidence."""
    parent = {"prompt": "short", "sibling": "keep exactly"}
    lm = RecordingLM()
    proposal, _ = StatelessReflectionLM(lm, text_limits=limits).reflect(
        parent, {"prompt": [{"Feedback": "fix it"}]}, ["prompt"]
    )
    assert proposal.new_texts == {}
    assert proposal.metadata["length_capped_dropped"] == ["prompt"]
    assert parent == {"prompt": "short", "sibling": "keep exactly"}


@pytest.mark.parametrize("limit", [None, 2000])
def test_refiner_saved_output_has_no_hidden_default_cutoff(limit: int | None) -> None:
    """Keep late parse-error evidence in subsequent refiner prompts unless capped."""
    raw = "x" * 3000 + "LATE_PARSE_EVIDENCE"
    lm = RecordingLM(raw)
    adapter = OptimizeAnythingAdapter(
        evaluator=lambda *_args, **_kwargs: (0.0, None, {}),
        refiner_config=RefinerConfig(refiner_lm=lm, max_refinements=2, text_limits=TextLimits(history_text_chars=limit)),
    )
    _, _, _, attempts = adapter._refine_and_evaluate({"prompt": "seed"}, None, "Improve", 0.0, {})
    assert len(lm.calls) == 2
    if limit is None:
        assert attempts[1]["raw_output"] == raw
        assert "LATE_PARSE_EVIDENCE" in lm.calls[1]
    else:
        assert "characters omitted" in attempts[1]["raw_output"]
        assert "LATE_PARSE_EVIDENCE" not in lm.calls[1]
