# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Integration tests for Controller -> Manifestor -> ReAct V2 reflection."""

import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from gepa.lm import LM, InlineReasoningLM, TrackingLM
from gepa.proposer.reflective_mutation.reflection_lm import ReflectionProposal, StatelessReflectionLM
from gepa.proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer
from gepa.proposer.reflective_mutation.three_role import ThreeRoleReflectionLM, ensure_reflection_run_contract
from gepa.strategies.document_template import TEMPLATE_FAMILIES, TEMPLATES, DocumentTemplate, MalformedDocumentError
from gepa.strategies.edit_tools import EditTool

PROMPT = TEMPLATES["system_prompt"].render({"Role": "helper", "Rules": "- be nice\n- be brief"})
SKILL = TEMPLATES["skill"].render(
    {"Name": "summarize", "Description": "Summarize text.", "Instructions": "- be nice\n- be brief"}
)
OPENAI_PROMPT = TEMPLATE_FAMILIES["openai"]["system_prompt"].render(
    {"Identity": "helper", "Instructions": "be accurate"}
)
OPENAI_USER_PROMPT = TEMPLATE_FAMILIES["openai"]["user_prompt"].render({"Input": "Answer this question."})
MEMO_TEMPLATE = DocumentTemplate("memo", {"Header": "memo header", "Body": "memo body"})
MEMO = MEMO_TEMPLATE.render({"Header": "h", "Body": "b"})
SYS_REFLECTIVE_DATASET = {
    "sys": [
        {
            "Inputs": "question",
            "Generated Outputs": "vague answer",
            "Feedback": "the answer was too vague",
        }
    ]
}
SKILL_REFLECTIVE_DATASET = {
    "skill": [
        {
            "Inputs": "question",
            "Generated Outputs": "vague answer",
            "Feedback": "the answer was too vague",
        }
    ]
}


def tool_call(tool: EditTool, **fields: str) -> str:
    """Render one ReAct compatibility-protocol tool call.

    Args:
        tool: Edit operator named by the call.
        **fields: Tool-specific child fields.

    Returns:
        XML-like tool-call block.
    """
    children = [f"<tool>{tool.value}</tool>"]
    children.extend(f"<{name}>{value}</{name}>" for name, value in fields.items())
    return f"<tool_call>{''.join(children)}</tool_call>"


BROAD_LEVEL1_REPLIES = [
    tool_call(EditTool.REPLACE_TEXT, target="be nice", text="be kind"),
    "<finish>The edit is complete.</finish>",
]
DIRECT_REEXPRESS_REPLIES = [tool_call(EditTool.REPLACE_TEXT, target="be nice", text="be kind")]
MINIMAL_REEXPRESS_REPLIES = [
    tool_call(EditTool.DELETE_TEXT, target="- be nice\n- be brief"),
    tool_call(EditTool.INSERT_TEXT, anchor="", where="after", text="- be kind\n- be brief"),
    "<finish>The semantic edit is complete.</finish>",
]


class ThreeRoleLM:
    """Script Controller, Manifestor, and ReAct V2 through their prompt shapes."""

    def __init__(
        self,
        react_replies: list[str],
        *,
        region: str = "Rules",
        semantic_action: str = "reexpress",
        model: str | None = None,
    ):
        """Configure deterministic replies for all three roles.

        Args:
            react_replies: Assistant replies consumed by ReAct turns.
            region: Controller section to select.
            semantic_action: Controller semantic action to select.
            model: Optional provider/model identifier exposed by the test LM.
        """
        self.react_replies = react_replies
        self.region = region
        self.semantic_action = semantic_action
        self.roles: list[str] = []
        self.string_calls: list[str] = []
        self.react_calls: list[list[dict[str, Any]]] = []
        if model is not None:
            self.model = model

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        """Route a prompt to its deterministic role reply.

        Args:
            prompt: String Controller or Manifestor prompt, or ReAct messages.

        Returns:
            Verbalized distribution, Manifestor steering, ReAct reply, or
            vanilla fenced revision matching the prompt shape.

        Raises:
            AssertionError: A ReAct turn exceeds scripted replies or the
                Controller menu lacks the requested region and action.
        """
        if isinstance(prompt, list):
            self.roles.append("react_v2")
            self.react_calls.append(deepcopy(prompt))
            index = len(self.react_calls) - 1
            if index >= len(self.react_replies):
                raise AssertionError(f"Unexpected ReAct V2 turn {index + 1}")
            return self.react_replies[index]

        self.string_calls.append(prompt)
        if "Choose edit actions that address" in prompt:
            self.roles.append("controller")
            semantic_needle = f"{self.semantic_action}@{self.region}/"
            atomic_needle = f"EDIT@{self.region}"
            options: list[str] = []
            for line in prompt.splitlines():
                if line.startswith("- ") and ": " in line:
                    options.append(line[2:].split(": ", 1)[0])
            chosen = next((option for option in options if semantic_needle in option), None)
            if chosen is None:
                chosen = next((option for option in options if atomic_needle in option), None)
            if chosen is None:
                raise AssertionError(f"No Controller option for region {self.region!r}")
            requested = int(prompt.rsplit("...repeat for ", 1)[1].split(" candidates", 1)[0])
            if requested != len(options):
                options = [chosen]
            chosen_probability = 1.0 if len(options) == 1 else 0.99
            other_probability = 0.0 if len(options) == 1 else 0.01 / (len(options) - 1)
            candidates = "".join(
                "<candidate>"
                f"<action>{option}</action><reasoning>test</reasoning>"
                f"<probability>{chosen_probability if option == chosen else other_probability}</probability>"
                "</candidate>"
                for option in options
            )
            return f"<response>{candidates}</response>"
        if "Write the next instruction for a language model editor" in prompt:
            self.roles.append("manifestor")
            return "The failures expose vague wording, so make that wording exact."
        self.roles.append("vanilla")
        return "```revised free-form instruction```"


class CostTrackingLM(ThreeRoleLM):
    """Expose a fixed cumulative LM cost."""

    def __init__(self, total_cost: float, react_replies: list[str]):
        """Store fixed provider cost alongside scripted role replies.

        Args:
            total_cost: Cumulative provider spend exposed by the test model.
            react_replies: Assistant replies consumed by ReAct turns.
        """
        super().__init__(react_replies)
        self.total_cost = total_cost


class SequenceManifestorLM:
    """Return scripted string replies to Manifestor calls."""

    def __init__(self, replies: list[str]):
        """Store Manifestor replies and initialize an empty prompt log.

        Args:
            replies: Responses returned on successive calls.
        """
        self.replies = replies
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        """Record one Manifestor prompt and return its scripted reply.

        Args:
            prompt: Manifestor request text.

        Returns:
            Reply at the matching call index.
        """
        self.calls.append(prompt)
        return self.replies[len(self.calls) - 1]


class FallbackControllerLM(ThreeRoleLM):
    """Force the Controller's uniform parse fallback."""

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        """Return malformed Controller output and delegate other roles.

        Args:
            prompt: String role prompt or ReAct message list.

        Returns:
            Malformed distribution for Controller calls, otherwise the base
            three-role response.
        """
        if isinstance(prompt, str) and "Choose edit actions that address" in prompt:
            self.string_calls.append(prompt)
            self.roles.append("controller")
            return "not a verbalized distribution"
        return super().__call__(prompt)


def strategy(
    level: int,
    *,
    lm: ThreeRoleLM | None = None,
    react_replies: list[str] | None = None,
    **kwargs: Any,
) -> tuple[ThreeRoleReflectionLM, ThreeRoleLM]:
    """Build a deterministic strategy and its base test model.

    Args:
        level: Reflection level under test.
        lm: Existing scripted model, or ``None`` to construct one.
        react_replies: ReAct replies used only when constructing ``lm``.
        **kwargs: Additional strategy configuration.

    Returns:
        Three-role strategy and the model shared by its roles.
    """
    if lm is None:
        default_replies = DIRECT_REEXPRESS_REPLIES if level == 2 else BROAD_LEVEL1_REPLIES
        lm = ThreeRoleLM(list(default_replies if react_replies is None else react_replies))
    kwargs.setdefault("base_lm_run_identity", {"test_lm": "ThreeRoleLM"})
    instance = ThreeRoleReflectionLM(lm, level=level, rng=random.Random(0), max_menu=999, **kwargs)
    return instance, lm


def make_reflective_proposer(reflection_strategy: ThreeRoleReflectionLM) -> ReflectiveMutationProposer:
    """Build the public reflective-proposer seam around a three-role strategy.

    Args:
        reflection_strategy: Three-role strategy to inject.

    Returns:
        Reflective proposer with mocked adapter and selection dependencies.
    """
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
    """Fail early for unknown ablation and template settings.

    Args:
        kwargs: Invalid constructor configuration under test.
    """
    with pytest.raises(ValueError):
        ThreeRoleReflectionLM(ThreeRoleLM([]), **kwargs)


def test_validate_candidate_accepts_system_user_and_skill_components() -> None:
    """Resolve the three component roles from their conventional names."""
    strat, _ = strategy(1)
    strat.validate_candidate({"system_prompt": PROMPT, "user_prompt": PROMPT, "skill": SKILL})


def test_arbitrary_component_names_default_to_system_prompt() -> None:
    """Treat an unlisted optimization component as a system prompt."""
    strat, _ = strategy(1)
    assert strat.run_contract({"answer_instructions": PROMPT})["component_kinds"] == {
        "answer_instructions": "system_prompt"
    }


def test_run_contract_keeps_system_user_and_skill_roles_distinct() -> None:
    """Record role-specific template keys even when generic prompt shapes are shared."""
    strat, _ = strategy(2)
    contract = strat.run_contract({"system_prompt": PROMPT, "user_prompt": PROMPT, "skill": SKILL})
    assert contract["component_kinds"] == {
        "system_prompt": "system_prompt",
        "user_prompt": "user_prompt",
        "skill": "skill",
    }
    assert set(contract["templates"]) == {"system_prompt", "user_prompt", "skill"}
    assert contract["semantic_action_spaces"]["user_prompt"]["kind"] == "prompt"


def test_validate_candidate_names_the_migration_path_for_free_form_text() -> None:
    """Explain how to bring legacy text into canonical section form."""
    strat, _ = strategy(1)
    with pytest.raises(MalformedDocumentError, match="migrate_document"):
        strat.validate_candidate({"sys": "You are a helpful assistant."})


def test_validate_candidate_rejects_explicit_empty_sections() -> None:
    """Keep empty section headers out of the text sent to the task model."""
    strat, _ = strategy(1)
    noncanonical = PROMPT + "\n## Reasoning\n"
    with pytest.raises(MalformedDocumentError, match="empty sections must be omitted"):
        strat.validate_candidate({"sys": noncanonical})


def test_validate_candidate_rejects_uncataloged_level2_kind_before_reflection() -> None:
    """Fail before evaluation when a custom document kind has no action catalog."""
    strat, _ = strategy(2, templates={"memo": MEMO_TEMPLATE}, component_kinds={"memo": "memo"})
    with pytest.raises(ValueError, match="has no level-2 semantic catalog"):
        strat.validate_candidate({"memo": MEMO})
    with pytest.raises(ValueError, match="has no level-2 semantic catalog"):
        strat.run_contract({"memo": MEMO})


@pytest.mark.parametrize(
    ("kwargs", "candidate", "accepted"),
    [
        pytest.param({"template_family": "openai"}, {"sys": OPENAI_PROMPT}, True, id="matching_family"),
        pytest.param(
            {"template_family": "openai", "component_kinds": {"turn": "user_prompt"}},
            {"turn": OPENAI_USER_PROMPT},
            True,
            id="matching_user_role",
        ),
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
    """Validate documents against a provider family or custom overlay.

    Args:
        kwargs: Template configuration supplied to the strategy.
        candidate: Structured candidate under test.
        accepted: Whether validation should succeed.
    """
    strat, _ = strategy(1, **kwargs)
    if accepted:
        strat.validate_candidate(candidate)
    else:
        with pytest.raises(MalformedDocumentError):
            strat.validate_candidate(candidate)


def test_level1_selects_a_region_and_runs_react_without_manifestor() -> None:
    """Keep level 1 as the region-plus-atomic-basis ablation."""
    strat, lm = strategy(1)
    proposal, next_strategy = strat.reflect({"sys": PROMPT}, deepcopy(SYS_REFLECTIVE_DATASET), ["sys"])
    assert next_strategy is strat
    assert proposal.new_texts["sys"] != PROMPT
    assert TEMPLATES["system_prompt"].parse(proposal.new_texts["sys"])["Role"] == "helper"
    assert lm.roles.count("controller") == 1
    assert "manifestor" not in lm.roles
    assert lm.roles.count("react_v2") == 2
    assert proposal.metadata["semantic_action"] is None
    assert proposal.metadata["preferred_edit_tool"] is None
    assert proposal.metadata["action_choice"].startswith("EDIT@")
    assert proposal.metadata["action_operator"] is None


def test_three_role_run_contract_blocks_catalog_or_policy_drift(tmp_path: Path) -> None:
    """Make direct API resumes as strict as benchmark harnesses.

    Args:
        tmp_path: Temporary run directory supplied by pytest.
    """
    strat, _ = strategy(2)
    contract = strat.run_contract({"sys": PROMPT})
    assert contract["schema_version"] == 3
    assert contract["component_kinds"] == {"sys": "system_prompt"}
    assert contract["controller"]["version"] == 4
    assert contract["controller"]["factorization"] == "P(region, action)"
    assert len(contract["semantic_action_spaces"]["system_prompt"]["actions"]) == 10
    assert contract["semantic_action_spaces"]["system_prompt"]["kind"] == "prompt"
    assert contract["reflection_prompt_template"] is None
    assert contract["controller_react_lm"]["configuration_source"] == "explicit"
    assert contract["manifestor_lm"]["configuration_source"] == "explicit"

    path = Path(ensure_reflection_run_contract(str(tmp_path), contract))
    assert path.name == "reflection-run-contract.json"
    assert ensure_reflection_run_contract(str(tmp_path), contract) == str(path)
    with pytest.raises(ValueError, match="different reflection strategy contract"):
        ensure_reflection_run_contract(str(tmp_path), {**contract, "reflection_level": 1})
    with pytest.raises(ValueError, match="different reflection strategy contract"):
        ensure_reflection_run_contract(str(tmp_path), {**contract, "schema_version": 2})

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    (legacy_dir / "gepa_state.bin").write_bytes(b"old")
    with pytest.raises(ValueError, match="GEPA state but no reflection-run-contract.json"):
        ensure_reflection_run_contract(str(legacy_dir), contract)


def test_three_role_run_contract_identifies_role_lm_configuration_without_credentials() -> None:
    """Distinguish Controller/ReAct and Manifestor behavior without writing secrets."""
    base_lm = LM(
        "openai/controller-model",
        temperature=0.7,
        max_tokens=123,
        api_base="https://example.test/v1",
        api_key="controller-secret",
        token="generic-secret",
        github_token="github-secret",
        secret_key="key-secret",
        private_key="private-secret",
    )
    manifestor_lm = LM(
        "openai/manifestor-model",
        temperature=0.0,
        api_key="manifestor-secret",
    )
    strat = ThreeRoleReflectionLM(base_lm, 2, manifestor_lm=manifestor_lm)
    contract = strat.run_contract({"sys": PROMPT})

    assert contract["controller_react_lm"]["model"] == "openai/controller-model"
    assert contract["controller_react_lm"]["completion_kwargs"] == {
        "temperature": 0.7,
        "max_tokens": 123,
        "api_base": "https://example.test/v1",
        "api_key": "<redacted>",
        "token": "<redacted>",
        "github_token": "<redacted>",
        "secret_key": "<redacted>",
        "private_key": "<redacted>",
    }
    assert contract["manifestor_lm"]["model"] == "openai/manifestor-model"
    assert contract["manifestor_lm"]["completion_kwargs"]["temperature"] == 0.0
    assert contract["manifestor_lm"]["completion_kwargs"]["api_key"] == "<redacted>"


def test_three_role_run_contract_requires_identity_for_custom_lm() -> None:
    """Refuse resumable state when a custom callable has no stable configuration identity."""
    strat = ThreeRoleReflectionLM(ThreeRoleLM(DIRECT_REEXPRESS_REPLIES), 2)
    with pytest.raises(ValueError, match="stable run identity"):
        strat.run_contract({"sys": PROMPT})


def test_level2_selects_semantic_action_manifests_and_executes_one_direct_call() -> None:
    """Run reexpress through Controller, Manifestor, and one coupled REPLACE call."""
    strat, lm = strategy(2)
    proposal, _ = strat.reflect({"sys": PROMPT}, deepcopy(SYS_REFLECTIVE_DATASET), ["sys"])
    assert proposal.new_texts["sys"] != PROMPT
    assert lm.roles == ["controller", "manifestor", "react_v2"]
    assert proposal.metadata["semantic_action"] == "reexpress"
    assert proposal.metadata["action_choice"] == "reexpress@Rules/REPLACE_TEXT"
    assert proposal.metadata["action_operator"] == "REPLACE_TEXT"
    assert proposal.metadata["action_target_section"] == "Rules"
    assert proposal.metadata["preferred_edit_tool"] == "REPLACE_TEXT"
    assert proposal.metadata["steering_message"]
    record = proposal.metadata["three_role_actions"][0]
    assert record["action_choice"] == proposal.metadata["action_choice"]
    assert record["action_operator"] == proposal.metadata["action_operator"]
    assert record["action_target_section"] == proposal.metadata["action_target_section"]
    assert record["react_iterations"] == 1
    assert record["react_tool_calls"] == 1
    assert record["react_steps"][0]["action"] == "REPLACE_TEXT"
    sampling = proposal.metadata["controller_sampling"]
    assert sampling["sampled"] == ["reexpress@Rules/REPLACE_TEXT"]
    assert len(sampling["probs"]) == 70
    assert sampling["probs"]["reexpress@Rules/REPLACE_TEXT"] == pytest.approx(0.99)
    assert sampling["fallback"] is False
    assert sampling["policy"] == "joint_region_action_v4"
    assert sampling["sampling_policy"] == "positive_support_uniform_mixture"
    assert sampling["exploration_epsilon"] == pytest.approx(0.1)
    assert sampling["joint_sampling_probability"] == pytest.approx(sampling["sampled_probabilities"][0])
    assert sampling["joint_sampling_probability"] > 0
    assert record["controller_sampling"] == sampling


def test_controller_sees_omitted_sections_as_empty_without_rendering_them() -> None:
    """Expose sparse section occupancy only to the Controller selection call."""
    strat, lm = strategy(2)

    strat.reflect({"sys": PROMPT}, deepcopy(SYS_REFLECTIVE_DATASET), ["sys"])

    controller_prompt = next(prompt for prompt in lm.string_calls if "Choose edit actions that address" in prompt)
    assert (
        "An empty region has no target bytes: assign probability 0 to its DELETE_TEXT, REPLACE_TEXT, and "
        "MOVE_TEXT choices. Judge its INSERT_TEXT choices by their semantic fit."
    ) in controller_prompt
    assert "## Role\nhelper" in controller_prompt
    assert "## Rules\n- be nice\n- be brief" in controller_prompt
    for section in ("Task", "Context", "Reasoning", "Examples", "Output Format"):
        assert f"## {section}\n[EMPTY SECTION]" in controller_prompt
        assert f"## {section}" not in PROMPT


def test_level2_minimal_basis_uses_a_deeper_delete_insert_trajectory() -> None:
    """Keep depth observable for the planned atomic-versus-semantic ablation."""
    lm = ThreeRoleLM(list(MINIMAL_REEXPRESS_REPLIES))
    strat, _ = strategy(2, lm=lm, edit_tool_set="minimal")
    proposal, _ = strat.reflect({"sys": PROMPT}, deepcopy(SYS_REFLECTIVE_DATASET), ["sys"])
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
        "content": MINIMAL_REEXPRESS_REPLIES[-1],
    }


def test_edit_is_scoped_to_selected_region_and_keeps_canonical_format() -> None:
    """Leave unselected sections unchanged and preserve fixed section headers."""
    strat, _ = strategy(2)
    proposal, _ = strat.reflect({"sys": PROMPT}, deepcopy(SYS_REFLECTIVE_DATASET), ["sys"])
    before = TEMPLATES["system_prompt"].parse(PROMPT)
    after = TEMPLATES["system_prompt"].parse(proposal.new_texts["sys"])
    assert after["Role"] == before["Role"]
    assert after["Rules"] != before["Rules"]


def test_skill_component_is_independently_sectioned_and_editable() -> None:
    """Run the same architecture over a declared skill Instructions section."""
    lm = ThreeRoleLM(list(DIRECT_REEXPRESS_REPLIES), region="Instructions")
    strat, _ = strategy(2, lm=lm, component_kinds={"skill": "skill"})
    proposal, _ = strat.reflect({"skill": SKILL}, deepcopy(SKILL_REFLECTIVE_DATASET), ["skill"])
    before = TEMPLATES["skill"].parse(SKILL)
    after = TEMPLATES["skill"].parse(proposal.new_texts["skill"])
    assert after["Name"] == before["Name"]
    assert after["Instructions"] != before["Instructions"]
    assert "document component" in lm.string_calls[0]
    assert "improve a prompt" not in lm.string_calls[0]


@pytest.mark.parametrize(
    "model",
    [
        pytest.param("openai/gpt-5", id="openai"),
        pytest.param("anthropic/claude-sonnet-4-5", id="claude"),
    ],
)
def test_manifestor_steering_reaches_react_as_a_user_message(model: str) -> None:
    """Deliver Manifestor guidance in the user message for every provider.

    Args:
        model: Provider/model identifier exposed by the scripted LM.
    """
    lm = ThreeRoleLM(list(DIRECT_REEXPRESS_REPLIES), model=model)
    strat, _ = strategy(2, lm=lm)
    proposal, _ = strat.reflect({"sys": PROMPT}, deepcopy(SYS_REFLECTIVE_DATASET), ["sys"])
    record = proposal.metadata["three_role_actions"][0]
    assert record["manifestor_delivery"] == "user_message"
    first_messages = lm.react_calls[0]
    assert [message["role"] for message in first_messages] == ["system", "user"]
    assert first_messages[-1]["content"].startswith(record["steering_message"])


def test_tracking_wrapper_preserves_manifestor_user_delivery() -> None:
    """Keep user-message guidance when the proposer callable is wrapped."""
    base = ThreeRoleLM(list(DIRECT_REEXPRESS_REPLIES), model="openai/gpt-5")
    wrapped = TrackingLM(base)
    strat = ThreeRoleReflectionLM(wrapped, level=2, rng=random.Random(0))

    proposal, _ = strat.reflect({"sys": PROMPT}, deepcopy(SYS_REFLECTIVE_DATASET), ["sys"])

    assert wrapped.model == "openai/gpt-5"
    assert [message["role"] for message in base.react_calls[0]] == ["system", "user"]


def test_separate_manifestor_lm_receives_only_manifestation_call() -> None:
    """Allow a deterministic Manifestor model without changing Controller/ReAct routing."""
    base = ThreeRoleLM(list(DIRECT_REEXPRESS_REPLIES))
    manifestor = ThreeRoleLM([])
    strat, _ = strategy(2, lm=base, manifestor_lm=manifestor)
    strat.reflect({"sys": PROMPT}, deepcopy(SYS_REFLECTIVE_DATASET), ["sys"])
    assert manifestor.roles == ["manifestor"]
    assert "manifestor" not in base.roles


def test_inline_reasoning_adapter_retries_empty_manifestation_then_runs_react() -> None:
    """Apply inline-reasoning cleanup at the LM boundary before manifestation."""
    base = ThreeRoleLM(list(DIRECT_REEXPRESS_REPLIES))
    raw_manifestor = SequenceManifestorLM(["<think>private</think>", "Make the vague rule exact."])
    manifestor = InlineReasoningLM(raw_manifestor)
    strat, _ = strategy(2, lm=base, manifestor_lm=manifestor)
    proposal, _ = strat.reflect({"sys": PROMPT}, deepcopy(SYS_REFLECTIVE_DATASET), ["sys"])
    assert proposal.new_texts["sys"] != PROMPT
    assert len(raw_manifestor.calls) == 2
    assert base.roles == ["controller", "react_v2"]
    assert proposal.metadata["steering_message"] == "Make the vague rule exact."


def test_repeated_empty_manifestation_is_recorded_as_a_dropped_attempt() -> None:
    """Drop an unmanifestable semantic action with explicit branch-history evidence."""
    base = ThreeRoleLM([])
    manifestor = SequenceManifestorLM(["   ", "\n"])
    strat, _ = strategy(2, lm=base, manifestor_lm=manifestor)
    proposal, _ = strat.reflect({"sys": PROMPT}, deepcopy(SYS_REFLECTIVE_DATASET), ["sys"])
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


def test_manifestor_receives_only_selected_section_feedback_and_trace() -> None:
    """Ground steering in the selected section without exposing siblings."""
    lm = ThreeRoleLM(list(DIRECT_REEXPRESS_REPLIES))
    strat, _ = strategy(2, lm=lm)
    strat.reflect({"sys": PROMPT}, deepcopy(SYS_REFLECTIVE_DATASET), ["sys"])
    manifestor_prompt = next(
        call for call in lm.string_calls if "Write the next instruction for a language model editor" in call
    )
    controller_prompt = next(call for call in lm.string_calls if "Choose edit actions that address" in call)
    react_prompt = json.dumps(lm.react_calls[0], ensure_ascii=False)
    assert "helper" in controller_prompt
    assert "- be nice" in manifestor_prompt
    assert "helper" not in manifestor_prompt
    assert "- be nice" in react_prompt
    assert "helper" not in react_prompt
    assert "the answer was too vague" in manifestor_prompt
    assert "Generated Outputs" not in manifestor_prompt
    assert "Output: vague answer" in manifestor_prompt


def test_reconstructed_component_enforces_the_full_length_cap() -> None:
    """Reject a section body that fits alone but overflows its parent document."""
    lm = ThreeRoleLM([tool_call(EditTool.REPLACE_TEXT, target="be nice", text="be much nicer")])
    strat, _ = strategy(2, lm=lm, max_chars=len(PROMPT))

    proposal, _ = strat.reflect({"sys": PROMPT}, deepcopy(SYS_REFLECTIVE_DATASET), ["sys"])

    assert proposal.new_texts == {}
    record = proposal.metadata["attempt_records"][0]
    assert record["attempt_status"] == "dropped"
    assert "Edited component" in record["dropped_reason"]


def test_branch_history_is_visible_to_react_and_reported_in_metadata() -> None:
    """Replay only the selected target's user/assistant transcript."""
    lm = ThreeRoleLM(list(DIRECT_REEXPRESS_REPLIES))
    strat, _ = strategy(2, lm=lm)
    history = [
        {"role": "assistant", "content": "<tool_call>this-parent-only</tool_call>"},
        {
            "role": "user",
            "content": "Optimizer result: accepted; the branch now contains this edit. Edit target: sys:Rules.",
        },
    ]
    proposal, _ = strat.reflect(
        {"sys": PROMPT},
        deepcopy(SYS_REFLECTIVE_DATASET),
        ["sys"],
        metadata={"branch_edit_history": history},
    )
    assert proposal.metadata["branch_history_length"] == 2
    assert lm.react_calls[0][1:3] == history


def test_branch_history_excludes_sibling_section_messages() -> None:
    """Never replay a Role attempt while editing the Rules section."""
    lm = ThreeRoleLM(list(DIRECT_REEXPRESS_REPLIES))
    strat, _ = strategy(2, lm=lm)
    role_history = [
        {"role": "assistant", "content": "role-only-marker"},
        {
            "role": "user",
            "content": "Optimizer result: accepted; the branch now contains this edit. Edit target: sys:Role.",
        },
    ]
    rules_history = [
        {"role": "assistant", "content": "rules-only-marker"},
        {
            "role": "user",
            "content": "Optimizer result: accepted; the branch now contains this edit. Edit target: sys:Rules.",
        },
    ]

    proposal, _ = strat.reflect(
        {"sys": PROMPT},
        deepcopy(SYS_REFLECTIVE_DATASET),
        ["sys"],
        metadata={"branch_edit_history": role_history + rules_history},
    )

    replayed = {message["content"] for message in lm.react_calls[0]}
    assert "rules-only-marker" in replayed
    assert "role-only-marker" not in replayed
    assert proposal.metadata["branch_history_length"] == len(rules_history)


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
    """Reject malformed lineage context instead of treating it as global text.

    Args:
        history: Invalid branch-history value under test.
        error: Validation exception expected for that value.
    """
    strat, _ = strategy(2)
    with pytest.raises(error, match="branch_edit_history"):
        strat.reflect(
            {"sys": PROMPT},
            deepcopy(SYS_REFLECTIVE_DATASET),
            ["sys"],
            metadata={"branch_edit_history": history},
        )


def test_revision_records_include_only_completed_component_revisions() -> None:
    """Keep incomplete edits out of legacy revisions while retaining attempt evidence."""
    lm = ThreeRoleLM(["no protocol action", "still no protocol action"])
    strat, _ = strategy(2, lm=lm, react_max_iterations=2)
    proposal, _ = strat.reflect({"sys": PROMPT}, deepcopy(SYS_REFLECTIVE_DATASET), ["sys"])
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
    proposal, _ = strat.reflect({"sys": PROMPT}, deepcopy(SYS_REFLECTIVE_DATASET), ["sys"])
    record = proposal.metadata["attempt_records"][0]
    assert len(record["react_steps"][0]["assistant"]) < 2100
    assert "...(+3000 chars)" in record["react_steps"][0]["assistant"]
    json.dumps(record)


def test_controller_uniform_fallback_provenance_is_persisted() -> None:
    """Expose when malformed verbalized sampling fell back to a uniform menu."""
    lm = FallbackControllerLM(["no protocol action"])
    strat, _ = strategy(1, lm=lm, react_max_iterations=1)
    proposal, _ = strat.reflect({"sys": PROMPT}, deepcopy(SYS_REFLECTIVE_DATASET), ["sys"])
    sampling = proposal.metadata["controller_sampling"]
    assert sampling["fallback"] is True
    assert sampling["used_full_fallback"] is False
    assert sampling["n_parsed_entries"] == len(sampling["probs"])
    assert proposal.metadata["attempt_records"][0]["controller_sampling"] == sampling


def test_level2_drops_after_two_incomplete_controller_distributions() -> None:
    """Do not invent a region/action pair when the Controller never scores the menu."""
    lm = FallbackControllerLM([])
    strat, _ = strategy(2, lm=lm)

    proposal, _ = strat.reflect({"sys": PROMPT}, deepcopy(SYS_REFLECTIVE_DATASET), ["sys"])

    assert proposal.new_texts == {}
    assert lm.roles == ["controller", "controller"]
    assert proposal.metadata["react_v2_dropped"] == ["sys"]
    assert proposal.metadata["controller_failures"] == [
        {
            "component": "sys",
            "error": "Controller did not return a complete action distribution after two attempts.",
        }
    ]


def test_reflection_metadata_keeps_action_and_react_diagnostics() -> None:
    """Preserve the documented metadata contract and ReAct V2 provenance."""
    strat, _ = strategy(2)
    proposal, _ = strat.reflect({"sys": PROMPT}, deepcopy(SYS_REFLECTIVE_DATASET), ["sys"])
    metadata = proposal.metadata
    for key in (
        "action",
        "action_choice",
        "action_operator",
        "action_target_section",
        "reflection_level",
        "proposer_backend",
        "edit_target",
        "edit_tool",
        "preferred_edit_tool",
        "semantic_action",
        "steering_message",
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
        deepcopy(SYS_REFLECTIVE_DATASET),
        ["sys"],
        metadata={"branch_edit_history": []},
    )
    assert new_texts["sys"] != PROMPT
    assert prompts["sys"] == metadata["steering_message"]
    assert raw_outputs["sys"]
    assert metadata["proposer_backend"] == "react_v2"
    assert metadata["revision_records"]


def test_level0_remains_identical_to_stateless_vanilla_reflection() -> None:
    """Keep the vanilla GEPA baseline byte-for-byte at ablation level 0."""
    dataset = deepcopy(SYS_REFLECTIVE_DATASET)
    baseline_lm = ThreeRoleLM([])
    baseline, _ = StatelessReflectionLM(baseline_lm).reflect({"sys": PROMPT}, dataset, ["sys"])
    level0_lm = ThreeRoleLM([])
    strat = ThreeRoleReflectionLM(level0_lm, level=0, rng=random.Random(0))
    proposal, _ = strat.reflect({"sys": PROMPT}, dataset, ["sys"])
    assert proposal.new_texts == baseline.new_texts
    assert "reflection_level" not in proposal.metadata


def test_reflect_many_aligns_each_job_with_its_own_history() -> None:
    """Keep parallel proposal jobs from sharing branch context."""
    lm = ThreeRoleLM(DIRECT_REEXPRESS_REPLIES * 2)
    strat, _ = strategy(2, lm=lm)
    jobs = [
        ({"sys": PROMPT}, deepcopy(SYS_REFLECTIVE_DATASET), ["sys"]),
        ({"sys": PROMPT}, deepcopy(SYS_REFLECTIVE_DATASET), ["sys"]),
    ]
    results = strat.reflect_many(
        jobs,
        metadatas=[
            {
                "branch_edit_history": [
                    {"role": "assistant", "content": "left-only"},
                    {
                        "role": "user",
                        "content": (
                            "Optimizer result: accepted; the branch now contains this edit. "
                            "Edit target: sys:Rules."
                        ),
                    },
                ]
            },
            {
                "branch_edit_history": [
                    {"role": "assistant", "content": "right-only"},
                    {
                        "role": "user",
                        "content": (
                            "Optimizer result: accepted; the branch now contains this edit. "
                            "Edit target: sys:Rules."
                        ),
                    },
                ]
            },
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
    jobs = [({"sys": PROMPT}, deepcopy(SYS_REFLECTIVE_DATASET), ["sys"])]
    with pytest.raises(ValueError, match="Expected 1 metadata records"):
        strat.reflect_many(jobs, metadatas=[])


def test_strategy_hooks_and_cost_tracking_remain_compatible() -> None:
    """Preserve seeded RNG, logger binding, and shared/separate LM accounting."""
    base = CostTrackingLM(1.25, list(DIRECT_REEXPRESS_REPLIES))
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


def test_explicit_strategy_rng_remains_independent_of_engine_sampling() -> None:
    """Preserve a caller-supplied Controller RNG when GEPA binds its run RNG."""
    strategy_rng = random.Random(7)
    engine_rng = random.Random(42)
    strat = ThreeRoleReflectionLM(ThreeRoleLM([]), level=2, rng=strategy_rng)

    strat.bind_rng(engine_rng)

    assert strat.rng is strategy_rng


def test_default_strategy_rng_binds_to_engine_sampling() -> None:
    """Keep the engine RNG as the default stream when no strategy RNG is set."""
    engine_rng = random.Random(42)
    strat = ThreeRoleReflectionLM(ThreeRoleLM([]), level=2)

    strat.bind_rng(engine_rng)

    assert strat.rng is engine_rng
