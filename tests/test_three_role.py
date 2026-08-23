# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Integration tests for Controller -> Manifestor -> ReAct V2 reflection."""

import json
import random
from copy import deepcopy
from typing import Any
from unittest.mock import MagicMock

import pytest

from gepa.lm import TrackingLM
from gepa.proposer.reflective_mutation.reflection_lm import ReflectionProposal, StatelessReflectionLM
from gepa.proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer
from gepa.proposer.reflective_mutation.three_role import ThreeRoleReflectionLM
from gepa.strategies.document_template import TEMPLATE_FAMILIES, TEMPLATES, DocumentTemplate, MalformedDocumentError
from gepa.strategies.edit_tools import EditTool

PROMPT = TEMPLATES["prompt"].render({"Role": "helper", "Rules": "- be nice\n- be brief"})
SKILL = TEMPLATES["skill"].render(
    {"Name": "summarize", "Description": "Summarize text.", "Instructions": "- be nice\n- be brief"}
)
OPENAI_PROMPT = TEMPLATE_FAMILIES["openai"]["prompt"].render({"Identity": "helper", "Instructions": "be accurate"})
MEMO_TEMPLATE = DocumentTemplate("memo", {"Header": "memo header", "Body": "memo body"})
MEMO = MEMO_TEMPLATE.render({"Header": "h", "Body": "b"})


def tool_call(tool: EditTool, **fields: str) -> str:
    """Render one ReAct V2 tool call."""
    children = [f"<tool>{tool.value}</tool>"]
    children.extend(f"<{name}>{value}</{name}>" for name, value in fields.items())
    return f"<tool_call>{''.join(children)}</tool_call>"


BROAD_LEVEL1_REPLIES = [
    tool_call(EditTool.REPLACE_TEXT, target="be nice", text="be kind"),
    "<finish>The edit is complete.</finish>",
]
DIRECT_REPHRASE_REPLIES = [tool_call(EditTool.REPLACE_TEXT, target="be nice", text="be kind")]
MINIMAL_REPHRASE_REPLIES = [
    tool_call(EditTool.DELETE_TEXT, target="be nice"),
    tool_call(EditTool.INSERT_TEXT, anchor="be brief", where="before", text="be kind\n"),
    "<finish>The semantic edit is complete.</finish>",
]


class ThreeRoleLM:
    """Script Controller, Manifestor, and ReAct V2 through their prompt shapes."""

    def __init__(
        self,
        react_replies: list[str],
        *,
        region: str = "Rules",
        semantic_action: str = "rephrase",
        model: str | None = None,
    ):
        """Store the region, semantic action, and ReAct replies."""
        self.react_replies = react_replies
        self.region = region
        self.semantic_action = semantic_action
        self.roles: list[str] = []
        self.string_calls: list[str] = []
        self.react_calls: list[list[dict[str, Any]]] = []
        if model is not None:
            self.model = model

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        """Route a call to its deterministic role reply."""
        if isinstance(prompt, list):
            self.roles.append("react_v2")
            self.react_calls.append(deepcopy(prompt))
            index = len(self.react_calls) - 1
            if index >= len(self.react_replies):
                raise AssertionError(f"Unexpected ReAct V2 turn {index + 1}")
            return self.react_replies[index]

        self.string_calls.append(prompt)
        if "You are selecting which edit action" in prompt:
            self.roles.append("controller")
            semantic_needle = f"{self.semantic_action}@{self.region}/"
            atomic_needle = f"EDIT@{self.region}"
            chosen = None
            for line in prompt.splitlines():
                if line.startswith("- **") and (semantic_needle in line or atomic_needle in line):
                    chosen = line.split("**", 2)[1]
                    if semantic_needle in line:
                        break
            if chosen is None:
                raise AssertionError(f"No Controller option for region {self.region!r}")
            return (
                "<response><candidate>"
                f"<action>{chosen}</action><reasoning>test</reasoning><probability>1.0</probability>"
                "</candidate></response>"
            )
        if "You steer a language model editor" in prompt:
            self.roles.append("manifestor")
            return "The failures expose vague wording, so make that wording exact."
        self.roles.append("vanilla")
        return "```revised free-form instruction```"


class CostTrackingLM(ThreeRoleLM):
    """Expose a fixed cumulative LM cost."""

    def __init__(self, total_cost: float, react_replies: list[str]):
        """Store the fixed cost alongside the normal scripted replies."""
        super().__init__(react_replies)
        self.total_cost = total_cost


class SequenceManifestorLM:
    """Return scripted string replies to Manifestor calls."""

    def __init__(self, replies: list[str]):
        """Store the replies and recorded prompts."""
        self.replies = replies
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        """Record the prompt and return its scripted reply."""
        self.calls.append(prompt)
        return self.replies[len(self.calls) - 1]


class FallbackControllerLM(ThreeRoleLM):
    """Force the Controller's uniform parse fallback."""

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        """Return malformed Controller output and delegate other roles."""
        if isinstance(prompt, str) and "You are selecting which edit action" in prompt:
            self.string_calls.append(prompt)
            self.roles.append("controller")
            return "not a verbalized distribution"
        return super().__call__(prompt)


def reflective_dataset(*components: str) -> dict[str, list[dict[str, str]]]:
    """Build one failure trace per component."""
    return {
        component: [
            {
                "Inputs": "question",
                "Generated Outputs": "vague answer",
                "Feedback": "the answer was too vague",
            }
        ]
        for component in components
    }


def strategy(
    level: int,
    *,
    lm: ThreeRoleLM | None = None,
    react_replies: list[str] | None = None,
    **kwargs: Any,
) -> tuple[ThreeRoleReflectionLM, ThreeRoleLM]:
    """Build a deterministic strategy and return it with its base LM."""
    if lm is None:
        default_replies = DIRECT_REPHRASE_REPLIES if level == 2 else BROAD_LEVEL1_REPLIES
        lm = ThreeRoleLM(list(default_replies if react_replies is None else react_replies))
    instance = ThreeRoleReflectionLM(lm, level=level, rng=random.Random(0), max_menu=999, **kwargs)
    return instance, lm


def make_reflective_proposer(reflection_strategy: ThreeRoleReflectionLM) -> ReflectiveMutationProposer:
    """Build the public reflective-proposer seam around a three-role strategy."""
    adapter = MagicMock()
    adapter.propose_new_texts = None
    return ReflectiveMutationProposer(
        logger=MagicMock(),
        trainset=[{"q": 1}],
        adapter=adapter,
        candidate_selector=MagicMock(),
        module_selector=MagicMock(),
        batch_sampler=MagicMock(),
        perfect_score=None,
        skip_perfect_score=False,
        experiment_tracker=MagicMock(),
        reflection_lm=MagicMock(),
        custom_candidate_proposer=None,
        reflection_strategy=reflection_strategy,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"level": 3}, id="invalid_level"),
        pytest.param({"level": 1, "edit_tool_set": "huge"}, id="invalid_tool_set"),
        pytest.param({"level": 1, "template_family": "meta"}, id="invalid_template_family"),
        pytest.param(
            {"level": 1, "component_kinds": {"sys": "memo"}},
            id="invalid_component_kind",
        ),
    ],
)
def test_construction_rejects_invalid_configuration(kwargs: dict[str, Any]) -> None:
    """Fail early for unknown ablation and document-template settings."""
    with pytest.raises(ValueError):
        ThreeRoleReflectionLM(ThreeRoleLM([]), **kwargs)


def test_validate_candidate_accepts_both_prompt_and_skill_documents() -> None:
    """Support the two document types selected in the design."""
    strat, _ = strategy(1, component_kinds={"skill": "skill"})
    strat.validate_candidate({"sys": PROMPT, "skill": SKILL})


def test_validate_candidate_names_the_migration_path_for_free_form_text() -> None:
    """Explain how to bring legacy text into canonical section form."""
    strat, _ = strategy(1)
    with pytest.raises(MalformedDocumentError, match="migrate_document"):
        strat.validate_candidate({"sys": "You are a helpful assistant."})


@pytest.mark.parametrize(
    ("kwargs", "candidate", "accepted"),
    [
        pytest.param({"template_family": "openai"}, {"sys": OPENAI_PROMPT}, True, id="matching_family"),
        pytest.param({"template_family": "openai"}, {"sys": PROMPT}, False, id="wrong_family"),
        pytest.param(
            {"templates": {"memo": MEMO_TEMPLATE}, "component_kinds": {"sys": "memo"}},
            {"sys": MEMO},
            True,
            id="custom_template_overlay",
        ),
    ],
)
def test_template_family_and_overlay(
    kwargs: dict[str, Any],
    candidate: dict[str, str],
    accepted: bool,
) -> None:
    """Validate documents against the configured provider family or overlay."""
    strat, _ = strategy(1, **kwargs)
    if accepted:
        strat.validate_candidate(candidate)
    else:
        with pytest.raises(MalformedDocumentError):
            strat.validate_candidate(candidate)


def test_level1_selects_a_region_and_runs_react_without_manifestor() -> None:
    """Keep level 1 as the region-plus-atomic-basis ablation."""
    strat, lm = strategy(1)
    proposal, next_strategy = strat.reflect({"sys": PROMPT}, reflective_dataset("sys"), ["sys"])
    assert next_strategy is strat
    assert proposal.new_texts["sys"] != PROMPT
    assert lm.roles.count("controller") == 1
    assert "manifestor" not in lm.roles
    assert lm.roles.count("react_v2") == 2
    assert proposal.metadata["intervention_spec"] is None
    assert proposal.metadata["preferred_edit_tool"] is None


def test_level2_selects_semantic_action_manifests_and_executes_one_direct_call() -> None:
    """Run rephrase through Controller, Manifestor, and one coupled REPLACE call."""
    strat, lm = strategy(2)
    proposal, _ = strat.reflect({"sys": PROMPT}, reflective_dataset("sys"), ["sys"])
    assert proposal.new_texts["sys"] != PROMPT
    assert lm.roles == ["controller", "manifestor", "react_v2"]
    assert proposal.metadata["intervention_spec"] == "rephrase"
    assert proposal.metadata["preferred_edit_tool"] == "REPLACE_TEXT"
    assert proposal.metadata["manifested_intervention"]
    record = proposal.metadata["three_role_actions"][0]
    assert record["react_iterations"] == 1
    assert record["react_tool_calls"] == 1
    assert record["react_steps"][0]["action"] == "REPLACE_TEXT"
    sampling = proposal.metadata["controller_sampling"]
    assert sampling["sampled"] == ["rephrase@Rules/REPLACE_TEXT"]
    assert sampling["probs"] == {"rephrase@Rules/REPLACE_TEXT": 1.0}
    assert sampling["fallback"] is False
    assert sampling["used_full_fallback"] is True
    assert record["controller_sampling"] == sampling


def test_level2_minimal_basis_uses_a_deeper_delete_insert_trajectory() -> None:
    """Keep depth observable for the planned atomic-versus-semantic ablation."""
    lm = ThreeRoleLM(list(MINIMAL_REPHRASE_REPLIES))
    strat, _ = strategy(2, lm=lm, edit_tool_set="minimal")
    proposal, _ = strat.reflect({"sys": PROMPT}, reflective_dataset("sys"), ["sys"])
    record = proposal.metadata["three_role_actions"][0]
    assert record["preferred_edit_tool"] == "REPLACE_TEXT"
    assert record["react_iterations"] == 3
    assert record["react_tool_calls"] == 2
    assert [step["action"] for step in record["react_steps"]] == [
        "DELETE_TEXT",
        "INSERT_TEXT",
        "FINISH",
    ]
    assert record["chat_messages"][-1] == {
        "role": "assistant",
        "content": MINIMAL_REPHRASE_REPLIES[-1],
    }


def test_edit_is_scoped_to_selected_region_and_keeps_canonical_format() -> None:
    """Leave unselected sections unchanged and preserve fixed section headers."""
    strat, _ = strategy(2)
    proposal, _ = strat.reflect({"sys": PROMPT}, reflective_dataset("sys"), ["sys"])
    before = TEMPLATES["prompt"].parse(PROMPT)
    after = TEMPLATES["prompt"].parse(proposal.new_texts["sys"])
    assert after["Role"] == before["Role"]
    assert after["Rules"] != before["Rules"]


def test_skill_component_is_independently_sectioned_and_editable() -> None:
    """Run the same architecture over a declared skill Instructions section."""
    lm = ThreeRoleLM(list(DIRECT_REPHRASE_REPLIES), region="Instructions")
    strat, _ = strategy(2, lm=lm, component_kinds={"skill": "skill"})
    proposal, _ = strat.reflect({"skill": SKILL}, reflective_dataset("skill"), ["skill"])
    before = TEMPLATES["skill"].parse(SKILL)
    after = TEMPLATES["skill"].parse(proposal.new_texts["skill"])
    assert after["Name"] == before["Name"]
    assert after["Instructions"] != before["Instructions"]


@pytest.mark.parametrize(
    ("model", "expected_role"),
    [
        pytest.param("openai/gpt-5.6", "developer", id="openai"),
        pytest.param("anthropic/claude-sonnet-4-5", "user", id="claude"),
    ],
)
def test_manifestor_steering_reaches_react_in_provider_role(model: str, expected_role: str) -> None:
    """Use developer messages for OpenAI and user messages for Claude."""
    lm = ThreeRoleLM(list(DIRECT_REPHRASE_REPLIES), model=model)
    strat, _ = strategy(2, lm=lm)
    proposal, _ = strat.reflect({"sys": PROMPT}, reflective_dataset("sys"), ["sys"])
    record = proposal.metadata["three_role_actions"][0]
    assert record["inject_as"] == expected_role
    first_messages = lm.react_calls[0]
    if expected_role == "developer":
        assert any(message["role"] == "developer" for message in first_messages)
    else:
        assert [message["role"] for message in first_messages] == ["system", "user"]
        assert first_messages[-1]["content"].startswith(record["manifested_intervention"])


def test_tracking_wrapper_preserves_openai_provider_routing() -> None:
    """Keep wrapped callable model metadata visible to the Manifestor router."""
    base = ThreeRoleLM(list(DIRECT_REPHRASE_REPLIES), model="openai/gpt-5.6")
    wrapped = TrackingLM(base)
    strat = ThreeRoleReflectionLM(wrapped, level=2, rng=random.Random(0))

    proposal, _ = strat.reflect({"sys": PROMPT}, reflective_dataset("sys"), ["sys"])

    assert wrapped.model == "openai/gpt-5.6"
    assert proposal.metadata["three_role_actions"][0]["inject_as"] == "developer"
    assert any(message["role"] == "developer" for message in base.react_calls[0])


def test_separate_manifestor_lm_receives_only_manifestation_call() -> None:
    """Allow a deterministic Manifestor model without changing Controller/ReAct routing."""
    base = ThreeRoleLM(list(DIRECT_REPHRASE_REPLIES))
    manifestor = ThreeRoleLM([])
    strat, _ = strategy(2, lm=base, manifestor_lm=manifestor)
    strat.reflect({"sys": PROMPT}, reflective_dataset("sys"), ["sys"])
    assert manifestor.roles == ["manifestor"]
    assert "manifestor" not in base.roles


def test_think_only_manifestation_retries_then_runs_react() -> None:
    """Never pass blank semantic steering into ReAct after hidden-thought removal."""
    base = ThreeRoleLM(list(DIRECT_REPHRASE_REPLIES))
    manifestor = SequenceManifestorLM(["<think>private</think>", "Make the vague rule exact."])
    strat, _ = strategy(2, lm=base, manifestor_lm=manifestor)
    proposal, _ = strat.reflect({"sys": PROMPT}, reflective_dataset("sys"), ["sys"])
    assert proposal.new_texts["sys"] != PROMPT
    assert len(manifestor.calls) == 2
    assert base.roles == ["controller", "react_v2"]
    assert proposal.metadata["manifested_intervention"] == "Make the vague rule exact."


def test_repeated_empty_manifestation_is_recorded_as_a_dropped_attempt() -> None:
    """Drop an unmanifestable semantic action with explicit branch-history evidence."""
    base = ThreeRoleLM([])
    manifestor = SequenceManifestorLM(["<think>private</think>", "   "])
    strat, _ = strategy(2, lm=base, manifestor_lm=manifestor)
    proposal, _ = strat.reflect({"sys": PROMPT}, reflective_dataset("sys"), ["sys"])
    assert proposal.new_texts == {}
    assert len(manifestor.calls) == 2
    assert base.roles == ["controller"]
    assert proposal.metadata["revision_records"] == []
    assert proposal.metadata["attempt_records"] == proposal.metadata["three_role_actions"]
    record = proposal.metadata["attempt_records"][0]
    assert "no visible steering text" in record["manifestor_error"]
    assert record["react_iterations"] == 0
    assert record["react_steps"] == []
    assert proposal.metadata["react_v2_dropped"] == ["sys"]


def test_manifestor_receives_full_candidate_feedback_and_execution_trace() -> None:
    """Ground semantic steering in X and the observed failures."""
    lm = ThreeRoleLM(list(DIRECT_REPHRASE_REPLIES))
    strat, _ = strategy(2, lm=lm)
    strat.reflect({"sys": PROMPT}, reflective_dataset("sys"), ["sys"])
    manifestor_prompt = next(call for call in lm.string_calls if "You steer a language model editor" in call)
    assert "helper" in manifestor_prompt
    assert "the answer was too vague" in manifestor_prompt
    assert "Generated Outputs" not in manifestor_prompt
    assert "Output: vague answer" in manifestor_prompt


def test_branch_history_is_visible_to_react_and_reported_in_metadata() -> None:
    """Replay only the selected parent's user/assistant transcript."""
    lm = ThreeRoleLM(list(DIRECT_REPHRASE_REPLIES))
    strat, _ = strategy(2, lm=lm)
    history = [
        {"role": "assistant", "content": "<tool_call>this-parent-only</tool_call>"},
        {"role": "user", "content": "Optimizer feedback: ACCEPTED."},
    ]
    proposal, _ = strat.reflect(
        {"sys": PROMPT},
        reflective_dataset("sys"),
        ["sys"],
        metadata={"branch_edit_history": history},
    )
    assert proposal.metadata["branch_history_length"] == 2
    assert lm.react_calls[0][1:3] == history


@pytest.mark.parametrize(
    ("history", "error"),
    [
        pytest.param("not-a-sequence-of-records", TypeError, id="string"),
        pytest.param(["not-a-record"], TypeError, id="bad_entry"),
        pytest.param([{"role": "tool", "content": "bad"}], ValueError, id="tool_role"),
        pytest.param([{"role": "user", "content": "ok", "extra": True}], ValueError, id="extra_field"),
        pytest.param([{"role": "assistant", "content": 1}], TypeError, id="non_string_content"),
    ],
)
def test_branch_history_shape_is_validated(history: object, error: type[Exception]) -> None:
    """Reject malformed lineage context instead of treating it as global text."""
    strat, _ = strategy(2)
    with pytest.raises(error, match="branch_edit_history"):
        strat.reflect(
            {"sys": PROMPT},
            reflective_dataset("sys"),
            ["sys"],
            metadata={"branch_edit_history": history},
        )


def test_revision_records_include_only_completed_component_revisions() -> None:
    """Keep incomplete edits out of legacy revisions while retaining attempt evidence."""
    lm = ThreeRoleLM(["no protocol action", "still no protocol action"])
    strat, _ = strategy(2, lm=lm, react_max_iterations=2)
    proposal, _ = strat.reflect({"sys": PROMPT}, reflective_dataset("sys"), ["sys"])
    assert proposal.new_texts == {}
    assert proposal.metadata["revision_records"] == []
    assert proposal.metadata["react_v2_dropped"] == ["sys"]
    assert proposal.metadata["length_capped_dropped"] == ["sys"]
    record = proposal.metadata["three_role_actions"][0]
    assert [step["action"] for step in record["react_steps"]] == ["INVALID", "INVALID"]
    assert proposal.metadata["attempt_records"] == [record]
    assert record["react_steps"][0]["assistant"] == "no protocol action"
    assert record["react_steps"][0]["observation"]
    assert record["chat_messages"][:2] == [
        {"role": "assistant", "content": "no protocol action"},
        {"role": "user", "content": record["react_steps"][0]["observation"]},
    ]


def test_attempt_history_fields_are_bounded_and_json_serializable() -> None:
    """Bound persistent assistant/error/observation detail without losing provenance."""
    lm = ThreeRoleLM(["x" * 5000])
    strat, _ = strategy(2, lm=lm, react_max_iterations=1)
    proposal, _ = strat.reflect({"sys": PROMPT}, reflective_dataset("sys"), ["sys"])
    record = proposal.metadata["attempt_records"][0]
    assert len(record["react_steps"][0]["assistant"]) < 2100
    assert "...(+3000 chars)" in record["react_steps"][0]["assistant"]
    json.dumps(record)


def test_controller_uniform_fallback_provenance_is_persisted() -> None:
    """Expose when malformed verbalized sampling fell back to a uniform menu."""
    lm = FallbackControllerLM(["no protocol action"])
    strat, _ = strategy(1, lm=lm, react_max_iterations=1)
    proposal, _ = strat.reflect({"sys": PROMPT}, reflective_dataset("sys"), ["sys"])
    sampling = proposal.metadata["controller_sampling"]
    assert sampling["fallback"] is True
    assert sampling["used_full_fallback"] is False
    assert sampling["n_parsed_entries"] == len(sampling["probs"])
    assert proposal.metadata["attempt_records"][0]["controller_sampling"] == sampling


def test_reflection_metadata_keeps_legacy_keys_and_react_diagnostics() -> None:
    """Preserve downstream metadata consumers while adding ReAct V2 provenance."""
    strat, _ = strategy(2)
    proposal, _ = strat.reflect({"sys": PROMPT}, reflective_dataset("sys"), ["sys"])
    metadata = proposal.metadata
    for key in (
        "action",
        "reflection_level",
        "proposer_backend",
        "edit_target",
        "edit_tool",
        "preferred_edit_tool",
        "intervention_spec",
        "manifested_intervention",
        "executed_edit",
        "controller_sampling",
        "three_role_actions",
        "attempt_records",
        "revision_records",
    ):
        assert key in metadata
    assert metadata["edit_tool"] == metadata["preferred_edit_tool"] == "REPLACE_TEXT"
    assert metadata["proposer_backend"] == "react_v2"
    assert isinstance(metadata["action"], str)


def test_reflective_proposer_seam_preserves_reflection_metadata() -> None:
    """Return three-role diagnostics through the existing four-value proposal seam."""
    strat, _ = strategy(2)
    proposer = make_reflective_proposer(strat)
    new_texts, prompts, raw_outputs, metadata = proposer.propose_new_texts(
        {"sys": PROMPT},
        reflective_dataset("sys"),
        ["sys"],
        metadata={"branch_edit_history": []},
    )
    assert new_texts["sys"] != PROMPT
    assert prompts["sys"] == metadata["manifested_intervention"]
    assert raw_outputs["sys"]
    assert metadata["proposer_backend"] == "react_v2"
    assert metadata["revision_records"]


def test_level0_remains_identical_to_stateless_vanilla_reflection() -> None:
    """Keep the vanilla GEPA baseline byte-for-byte at ablation level 0."""
    dataset = reflective_dataset("sys")
    baseline_lm = ThreeRoleLM([])
    baseline, _ = StatelessReflectionLM(baseline_lm).reflect({"sys": PROMPT}, dataset, ["sys"])
    level0_lm = ThreeRoleLM([])
    strat = ThreeRoleReflectionLM(level0_lm, level=0, rng=random.Random(0))
    proposal, _ = strat.reflect({"sys": PROMPT}, dataset, ["sys"])
    assert proposal.new_texts == baseline.new_texts
    assert "reflection_level" not in proposal.metadata


def test_reflect_many_aligns_each_job_with_its_own_history() -> None:
    """Keep parallel proposal jobs from sharing branch context."""
    lm = ThreeRoleLM(DIRECT_REPHRASE_REPLIES * 2)
    strat, _ = strategy(2, lm=lm)
    jobs = [
        ({"sys": PROMPT}, reflective_dataset("sys"), ["sys"]),
        ({"sys": PROMPT}, reflective_dataset("sys"), ["sys"]),
    ]
    results = strat.reflect_many(
        jobs,
        metadatas=[
            {"branch_edit_history": [{"role": "user", "content": "left-only"}]},
            {"branch_edit_history": [{"role": "user", "content": "right-only"}]},
        ],
    )
    assert len(results) == 2
    assert all(isinstance(proposal, ReflectionProposal) for proposal, _ in results)
    assert {message["content"] for message in lm.react_calls[0]} >= {"left-only"}
    assert all(message["content"] != "right-only" for message in lm.react_calls[0])
    assert {message["content"] for message in lm.react_calls[1]} >= {"right-only"}
    assert all(message["content"] != "left-only" for message in lm.react_calls[1])


def test_reflect_many_validates_metadata_alignment() -> None:
    """Reject missing per-job context rather than shifting histories across jobs."""
    strat, _ = strategy(2)
    jobs = [({"sys": PROMPT}, reflective_dataset("sys"), ["sys"])]
    with pytest.raises(ValueError, match="Expected 1 metadata records"):
        strat.reflect_many(jobs, metadatas=[])


def test_strategy_hooks_and_cost_tracking_remain_compatible() -> None:
    """Preserve seeded RNG, logger binding, and shared/separate LM accounting."""
    base = CostTrackingLM(1.25, list(DIRECT_REPHRASE_REPLIES))
    manifestor = CostTrackingLM(0.5, [])
    strat = ThreeRoleReflectionLM(base, level=2, manifestor_lm=manifestor)
    rng = random.Random(42)
    logger = MagicMock()
    strat.bind_rng(rng)
    strat.bind_logger(logger)
    assert strat.rng is rng
    assert strat.logger is logger
    assert strat.supports_cost_tracking() is True
    assert strat.total_cost == pytest.approx(1.75)

    shared = ThreeRoleReflectionLM(base, level=2, manifestor_lm=base)
    assert shared.total_cost == pytest.approx(1.25)
