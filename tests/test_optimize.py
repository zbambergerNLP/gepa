import re
from collections.abc import Callable
from unittest.mock import Mock

import pytest

from gepa import optimize
from gepa.strategies.document_template import MalformedDocumentError


def test_reflection_prompt_template():
    """Test that reflection_prompt_template works with optimize()."""
    mock_data = [
        {
            "input": "my_input",
            "answer": "my_answer",
            "additional_context": {"context": "my_context"},
        }
    ]

    # Mock the reflection LM to return improved instructions and track calls
    reflection_calls = []

    task_lm = Mock()
    task_lm.return_value = "test response"

    def mock_reflection_lm(prompt):
        reflection_calls.append(prompt)
        return "```\nimproved instructions\n```"

    custom_template = """Current instructions:
<curr_param>
Inputs, outputs, and feedback:
<side_info>
Please improve the instructions."""

    optimize(
        seed_candidate={"instructions": "initial instructions"},
        trainset=mock_data,
        task_lm=task_lm,
        reflection_lm=mock_reflection_lm,
        reflection_prompt_template=custom_template,
        max_metric_calls=2,
        reflection_minibatch_size=1,
    )

    # Check that the reflection_lm was called with our custom template
    assert len(reflection_calls) > 0
    reflection_prompt = reflection_calls[0]
    assert "initial instructions" in reflection_prompt
    assert "my_input" in reflection_prompt
    assert "Please improve the instructions." in reflection_prompt


def test_reflection_prompt_template_missing_placeholders():
    """Test that reflection_prompt_template fails when placeholders are missing."""
    mock_data = [
        {
            "input": "my_input",
            "answer": "my_answer",
            "additional_context": {"context": "my_context"},
        }
    ]

    # Mock the reflection LM to return improved instructions and track calls
    reflection_calls = []

    task_lm = Mock()
    task_lm.return_value = "test response"

    def mock_reflection_lm(prompt):
        reflection_calls.append(prompt)
        return "```\nimproved instructions\n```"

    custom_template = "Missing both placeholders."

    with pytest.raises(
        ValueError,
        match=re.escape("Missing placeholder(s) in prompt template: <curr_param>, <side_info>"),
    ):
        optimize(
            seed_candidate={"instructions": "initial instructions"},
            trainset=mock_data,
            task_lm=task_lm,
            reflection_lm=mock_reflection_lm,
            reflection_prompt_template=custom_template,
            max_metric_calls=2,
            reflection_minibatch_size=1,
        )


def test_reflection_prompt_template_dict():
    """Test that reflection_prompt_template works with a dict mapping parameter names to templates."""
    mock_data = [
        {
            "input": "my_input",
            "answer": "my_answer",
            "additional_context": {"context": "my_context"},
        }
    ]

    # Track which parameter each reflection call was for
    reflection_calls = {}

    task_lm = Mock()
    task_lm.return_value = "test response"

    def mock_reflection_lm(prompt):
        # Store the prompt to check later
        if "Instructions template:" in prompt:
            reflection_calls["instructions"] = prompt
        elif "Context template:" in prompt:
            reflection_calls["context"] = prompt
        return "```\nimproved text\n```"

    # Create parameter-specific templates
    custom_templates = {
        "instructions": """Instructions template:
<curr_param>
Data:
<side_info>
Make it better.""",
        "context": """Context template:
<curr_param>
Feedback:
<side_info>
Improve context.""",
    }

    optimize(
        seed_candidate={"instructions": "initial instructions", "context": "initial context"},
        trainset=mock_data,
        task_lm=task_lm,
        reflection_lm=mock_reflection_lm,
        reflection_prompt_template=custom_templates,
        max_metric_calls=4,
        reflection_minibatch_size=1,
        module_selector="round_robin",  # Round robin to update each component in turn
    )

    # Check that at least one reflection call was made
    assert len(reflection_calls) > 0

    # Verify that custom templates were used correctly for the parameters that were reflected on
    if "instructions" in reflection_calls:
        instructions_call = reflection_calls["instructions"]
        assert "Instructions template:" in instructions_call
        assert "Make it better." in instructions_call

    if "context" in reflection_calls:
        context_call = reflection_calls["context"]
        assert "Context template:" in context_call
        assert "Improve context." in context_call


def test_empty_seed_candidate():
    """Test that optimize() fails gracefully with empty seed_candidate."""
    mock_data = [
        {
            "input": "my_input",
            "answer": "my_answer",
            "additional_context": {"context": "my_context"},
        }
    ]

    task_lm = Mock()
    task_lm.return_value = "test response"

    def mock_reflection_lm(prompt):
        return "```\nimproved instructions\n```"

    # Test with empty dict
    with pytest.raises(ValueError, match=r"seed_candidate must contain at least one component text\."):
        optimize(
            seed_candidate={},
            trainset=mock_data,
            task_lm=task_lm,
            reflection_lm=mock_reflection_lm,
            max_metric_calls=2,
            reflection_minibatch_size=1,
        )


def test_none_seed_candidate():
    """Test that optimize() fails gracefully with None seed_candidate."""
    mock_data = [
        {
            "input": "my_input",
            "answer": "my_answer",
            "additional_context": {"context": "my_context"},
        }
    ]

    task_lm = Mock()
    task_lm.return_value = "test response"

    def mock_reflection_lm(prompt):
        return "```\nimproved instructions\n```"

    # Test with None - Note: this will be caught by type checker, but we test runtime behavior
    with pytest.raises(ValueError, match=r"seed_candidate must contain at least one component text\."):
        optimize(
            seed_candidate=None,  # type: ignore
            trainset=mock_data,
            task_lm=task_lm,
            reflection_lm=mock_reflection_lm,
            max_metric_calls=2,
            reflection_minibatch_size=1,
        )


def test_three_role_seed_must_be_in_canonical_format() -> None:
    """Test that with reflection_level > 0 a free-form seed fails before any evaluation is spent."""
    mock_data = [{"input": "my_input", "answer": "my_answer", "additional_context": {"context": "my_context"}}]
    task_lm = Mock()
    task_lm.return_value = "test response"

    with pytest.raises(MalformedDocumentError, match=r"'system_prompt'.*migrate_document"):
        optimize(
            seed_candidate={"system_prompt": "You help people. Be nice."},
            trainset=mock_data,
            task_lm=task_lm,
            reflection_lm=lambda prompt: "noop",
            reflection_level=1,
            max_metric_calls=2,
            reflection_minibatch_size=1,
        )
    task_lm.assert_not_called()


class _StrategyCapturedError(Exception):
    """Raised by the stub strategy so optimize() stops right after building it."""


@pytest.mark.parametrize(
    # Parameter names
    [
        "reflection_lm",
        "reflection_level",
        "expected_derived",
    ],
    # Parameter values
    [
        pytest.param(
            "openai/gpt-4o-mini",  # reflection_lm
            2,  # reflection_level
            True,  # expected_derived
            id="model_name_level_2",
        ),
        pytest.param(
            "openai/gpt-4o-mini",  # reflection_lm
            1,  # reflection_level
            False,  # expected_derived
            id="model_name_level_1",
        ),
        pytest.param(
            lambda prompt: "noop",  # reflection_lm
            2,  # reflection_level
            False,  # expected_derived
            id="callable_level_2",
        ),
    ],
)
def test_three_role_level2_derives_a_deterministic_manifestor_lm(
    monkeypatch: pytest.MonkeyPatch,
    reflection_lm: str | Callable[[str], str],
    reflection_level: int,
    expected_derived: bool,
) -> None:
    """Test that at level 2 a model-name reflection_lm yields a temperature-0 Manifestor LM.

    POSIT manifests deterministically, so the derived Manifestor LM must pin its temperature to 0.

    Args:
        reflection_lm: The reflection LM passed to optimize (a model name or a callable).
        reflection_level: The three-role reflection level under test.
        expected_derived: Whether a distinct Manifestor LM should be derived for this case.
    """
    import gepa.api as api_module

    captured: dict = {}

    def stub_strategy(**kwargs):
        captured.update(kwargs)
        raise _StrategyCapturedError

    monkeypatch.setattr(api_module, "ThreeRoleReflectionLM", stub_strategy)
    mock_data = [{"input": "my_input", "answer": "my_answer", "additional_context": {"context": "my_context"}}]
    with pytest.raises(_StrategyCapturedError):
        optimize(
            seed_candidate={"system_prompt": "## Role\nhelper\n## Rules\n- be nice"},
            trainset=mock_data,
            task_lm=Mock(return_value="test response"),
            reflection_lm=reflection_lm,
            reflection_lm_kwargs={"max_tokens": 64, "temperature": 0.9},
            reflection_level=reflection_level,
            max_metric_calls=2,
            reflection_minibatch_size=1,
        )
    assert captured["level"] == reflection_level
    manifestor_lm = captured["manifestor_lm"]
    if not expected_derived:
        assert manifestor_lm is None
        return
    assert manifestor_lm is not captured["base_lm"]
    assert manifestor_lm.model == "openai/gpt-4o-mini"
    # reflection_lm_kwargs carry over, but temperature is pinned to 0.
    assert manifestor_lm.completion_kwargs == {"max_tokens": 64, "temperature": 0.0}


@pytest.mark.parametrize(
    # Parameter names
    [
        "template_family",
        "task_lm",
        "expected_family",
    ],
    # Parameter values
    [
        pytest.param(
            None,  # template_family (omitted -> the 'auto' default)
            "anthropic/claude-3-5-sonnet-20241022",  # task_lm
            "anthropic",  # expected_family
            id="default_auto_infers_from_task_lm_model_name",
        ),
        pytest.param(
            None,  # template_family (omitted -> the 'auto' default)
            "dashscope/qwen-max",  # task_lm
            "alibaba",  # expected_family
            id="default_auto_maps_qwen_to_alibaba",
        ),
        pytest.param(
            None,  # template_family (omitted -> the 'auto' default)
            "openai/gpt-5.6-sol",  # task_lm
            "openai-gpt-5.6",  # expected_family (the model-specific family wins over the provider one)
            id="default_auto_maps_gpt56_to_its_model_specific_family",
        ),
        pytest.param(
            "auto",  # template_family
            None,  # task_lm (a callable stands in; there is no name to infer from)
            "generic",  # expected_family
            id="auto_falls_back_to_generic_for_callable_task_lm",
        ),
        pytest.param(
            "google",  # template_family
            "openai/gpt-4o-mini",  # task_lm
            "google",  # expected_family
            id="explicit_family_wins_over_inference",
        ),
        pytest.param(
            "generic",  # template_family
            "anthropic/claude-3-5-sonnet-20241022",  # task_lm
            "generic",  # expected_family
            id="explicit_generic_opts_out_of_inference",
        ),
    ],
)
def test_template_family_follows_the_task_model(
    monkeypatch: pytest.MonkeyPatch,
    template_family: str | None,
    task_lm: str | None,
    expected_family: str,
) -> None:
    """Test that optimize derives template_family from task_lm by default and honors explicit choices.

    The family must follow the task model (the optimized prompt's consumer), never the reflection
    model; passing a concrete family must opt out of inference entirely.

    Args:
        template_family: The template_family argument passed to optimize, or None to omit it and
            exercise the 'auto' default.
        task_lm: The task model name, or None to pass a callable task LM instead.
        expected_family: The resolved family the ThreeRoleReflectionLM must be constructed with.
    """
    import gepa.api as api_module

    captured: dict = {}

    def stub_strategy(**kwargs):
        captured.update(kwargs)
        raise _StrategyCapturedError

    monkeypatch.setattr(api_module, "ThreeRoleReflectionLM", stub_strategy)
    mock_data = [{"input": "my_input", "answer": "my_answer", "additional_context": {"context": "my_context"}}]
    extra = {} if template_family is None else {"template_family": template_family}
    with pytest.raises(_StrategyCapturedError):
        optimize(
            seed_candidate={"system_prompt": "## Role\nhelper\n## Rules\n- be nice"},
            trainset=mock_data,
            task_lm=task_lm if task_lm is not None else Mock(return_value="test response"),
            reflection_lm=lambda prompt: "noop",
            reflection_level=1,
            max_metric_calls=2,
            reflection_minibatch_size=1,
            **extra,
        )
    assert captured["template_family"] == expected_family


def test_auto_family_mismatch_names_the_inferred_family() -> None:
    """Test that a seed that fails the auto-inferred family raises an error naming family and remedies."""
    mock_data = [{"input": "my_input", "answer": "my_answer", "additional_context": {"context": "my_context"}}]

    with pytest.raises(MalformedDocumentError, match=r"'anthropic'.*auto-inferred.*template_family='generic'"):
        optimize(
            seed_candidate={"system_prompt": "## Role\nhelper\n## Rules\n- be nice"},
            trainset=mock_data,
            task_lm="anthropic/claude-3-5-sonnet-20241022",
            reflection_lm=lambda prompt: "noop",
            reflection_level=1,
            max_metric_calls=2,
            reflection_minibatch_size=1,
        )
