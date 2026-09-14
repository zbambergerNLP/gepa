"""Keep repeated evidence intact in stateless and FOREST reflection prompts."""

from copy import deepcopy

import pytest

from gepa.proposer.reflective_mutation.three_role import _summarize_traces
from gepa.strategies.instruction_proposal import InstructionProposalSignature


def _render(records: list[dict], renderer: str) -> str:
    """Render evidence through the production path for the selected optimizer."""
    if renderer == "forest":
        return _summarize_traces(records)
    prompt = InstructionProposalSignature.prompt_renderer(
        {"current_instruction_doc": "Current rules.", "dataset_with_feedback": records}
    )
    assert isinstance(prompt, str)
    return prompt


@pytest.mark.parametrize("renderer", ["stateless", "forest"])
def test_exact_duplicates_remain_in_each_example_without_mutating_records(renderer: str) -> None:
    """Keep every repeated passage and feedback paragraph in its original example."""
    passage = "A substantial retrieved passage with supporting facts. " * 20
    feedback = "A shared explanation of how this component works. " * 10
    records = [
        {"Inputs": {"question": "First?", "passages": [passage, passage]}, "Feedback": feedback + "\n\nFIRST_ERROR"},
        {"Inputs": {"question": "Second?", "passages": [passage]}, "Feedback": feedback + "\n\nSECOND_ERROR"},
    ]
    original = deepcopy(records)
    prompt = _render(records, renderer)
    assert records == original
    assert prompt.count(passage.strip()) == 3
    assert prompt.count(feedback.strip()) == 2
    assert "First?" in prompt and "Second?" in prompt
    assert "FIRST_ERROR" in prompt and "SECOND_ERROR" in prompt
    assert "[Repeated text; see" not in prompt


@pytest.mark.parametrize("renderer", ["stateless", "forest"])
def test_long_evidence_and_every_repeated_line_survive(renderer: str) -> None:
    """Retain late evidence past 8,000 characters and all 500 identical log lines."""
    evidence = "\n".join(f"Step {index}: distinct command and result {index}" for index in range(500))
    repeated = "still waiting for the package download\n" * 500
    prompt = _render([{"Generated Outputs": repeated + "FINAL_ERROR\n" + evidence}], renderer)
    assert repeated + "FINAL_ERROR\n" + evidence in prompt
    assert prompt.count("still waiting") == 500
    assert "additional times" not in prompt


def test_stateless_prompt_retains_the_current_document_in_each_record() -> None:
    """Keep a supplied document body even when it also appears above the examples."""
    instruction = "Use this detailed current instruction when answering. " * 20
    prompt = InstructionProposalSignature.prompt_renderer(
        {
            "current_instruction_doc": instruction,
            "dataset_with_feedback": [{"Document": instruction, "Feedback": "fail"}],
        }
    )
    assert isinstance(prompt, str)
    assert prompt.count(instruction.strip()) == 2
    assert "see Current instruction above" not in prompt


@pytest.mark.parametrize("renderer", ["stateless", "forest"])
def test_independent_prompts_keep_all_their_evidence(renderer: str) -> None:
    """Retain evidence in every request even when earlier requests used the same text."""
    text = "This unique evidence must remain in both independent prompts. " * 20
    records = [{"Inputs": text}]
    assert text.strip() in _render(records, renderer)
    assert text.strip() in _render(records, renderer)
