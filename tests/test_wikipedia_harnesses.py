"""Tests for the Wikipedia-backed HotPotQA and HOVER runners."""

import fcntl
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import datasets
import pytest
from litellm.utils import get_optional_params

sys.path.insert(0, str(Path(__file__).parents[1]))

from examples.common.experiment_models import (
    EXPERIMENT_NUM_RETRIES,
    GLM_5_3_FLASH_MODEL,
    GLM_5_3_FLASH_OPENROUTER_MODEL,
    QWEN3_8_27B_MODEL,
    QWEN3_8_27B_OPENROUTER_MODEL,
    experiment_decoding,
    experiment_request_overrides,
)
from examples.common.wikipedia import WikipediaClient, WikipediaPassage
from examples.hotpotqa import main as hotpot_main
from examples.hotpotqa import utils as hotpot_utils
from examples.hover import utils as hover_utils
from examples.hover.main import make_evaluator as make_hover_evaluator

REPO_ROOT = Path(__file__).parents[1]
HOVER_COT_OUTPUTS = (
    "[[ ## reasoning ## ]]\nsummary one reasoning\n\n[[ ## summary ## ]]\nsummary one\n\n[[ ## completed ## ]]",
    "[[ ## reasoning ## ]]\nquery two reasoning\n\n[[ ## query ## ]]\nquery two\n\n[[ ## completed ## ]]",
    "[[ ## reasoning ## ]]\nsummary two reasoning\n\n[[ ## summary ## ]]\nsummary two\n\n[[ ## completed ## ]]",
    "[[ ## reasoning ## ]]\nquery three reasoning\n\n[[ ## query ## ]]\nquery three\n\n[[ ## completed ## ]]",
)
HOTPOT_COT_RESULTS = (
    ("summary one reasoning", "summary one"),
    ("bridge query reasoning", "bridge query"),
    ("summary two reasoning", "summary two"),
    ("answer reasoning", "exact answer"),
)


class FakeRetriever:
    """Return deterministic pages while recording retrieval calls."""

    def __init__(self, pages_by_query: dict[str, list[WikipediaPassage]]) -> None:
        """Initialize fixed results and an empty call log.

        Args:
            pages_by_query: Passage lists keyed by exact retrieval query.
        """
        self.pages_by_query = pages_by_query
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, limit: int) -> list[WikipediaPassage]:
        """Record and return a bounded deterministic result.

        Args:
            query: Exact lookup key.
            limit: Maximum passages to return.

        Returns:
            Configured passage prefix, or an empty list for an unknown query.
        """
        self.calls.append((query, limit))
        return self.pages_by_query.get(query, [])[:limit]


@pytest.mark.parametrize("module", [hotpot_utils, hover_utils])
@pytest.mark.parametrize("model", [QWEN3_8_27B_MODEL, GLM_5_3_FLASH_MODEL])
def test_wikipedia_lm_uses_experiment_model_decoding(monkeypatch, module, model: str) -> None:
    """Keep sparse messages and model-specific decoding in solver calls.

    Args:
        monkeypatch: Pytest fixture used to replace LiteLLM completion.
        module: Parameterized benchmark utility module.
        model: Experiment model whose decoding arguments are inspected.
    """
    calls = []

    def completion(**kwargs):
        """Capture one completion request and return fixed content.

        Args:
            **kwargs: LiteLLM completion arguments under test.

        Returns:
            Minimal response object containing ``answer``.
        """
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))])

    monkeypatch.setattr(module.litellm, "completion", completion)

    assert module._call_lm("", "question", model, None) == "answer"
    assert calls[0]["messages"] == [{"role": "user", "content": "question"}]
    expected_request = {
        "num_retries": EXPERIMENT_NUM_RETRIES,
        **experiment_decoding(model),
        **experiment_request_overrides(model),
    }
    if module is hotpot_utils:
        expected_request["seed"] = hotpot_utils.HOTPOTQA_SCIENTIFIC_REQUEST_SEED
    assert {key: value for key, value in calls[0].items() if key not in {"model", "messages"}} == expected_request
    if module is not hotpot_utils:
        assert "seed" not in calls[0]
    assert calls[0].get("extra_body") == experiment_request_overrides(model).get("extra_body")


@pytest.mark.parametrize("module", [hotpot_utils, hover_utils])
@pytest.mark.parametrize(
    "model",
    [QWEN3_8_27B_OPENROUTER_MODEL, GLM_5_3_FLASH_OPENROUTER_MODEL],
)
def test_wikipedia_lm_keeps_openrouter_routing_on_solver_calls(monkeypatch, module, model: str) -> None:
    """Attach the exact OpenRouter endpoint policy to every solver request.

    Args:
        monkeypatch: Pytest fixture used to replace LiteLLM completion.
        module: Parameterized benchmark utility module.
        model: Effective OpenRouter runtime model under test.
    """
    calls = []

    def completion(**kwargs):
        """Capture one provider request and return fixed answer content.

        Args:
            **kwargs: LiteLLM completion arguments under test.

        Returns:
            Minimal response object containing ``answer``.
        """
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))])

    monkeypatch.setattr(module.litellm, "completion", completion)

    assert module._call_lm("", "question", model, None) == "answer"
    assert calls[0]["model"] == model
    assert calls[0]["extra_body"] == experiment_request_overrides(model)["extra_body"]


@pytest.mark.parametrize(
    "model",
    [QWEN3_8_27B_OPENROUTER_MODEL, GLM_5_3_FLASH_OPENROUTER_MODEL],
)
def test_litellm_preserves_openrouter_routing_and_reasoning(model: str) -> None:
    """Keep the provider and reasoning objects intact through LiteLLM.

    Args:
        model: Effective OpenRouter runtime model under test.
    """
    request_overrides = experiment_request_overrides(model)

    transformed = get_optional_params(
        model=model.removeprefix("openrouter/"),
        custom_llm_provider="openrouter",
        drop_params=True,
        **experiment_decoding(model),
        **request_overrides,
    )

    assert transformed["extra_body"] == request_overrides["extra_body"]


def test_litellm_preserves_local_glm_chat_template_settings() -> None:
    """Keep GLM maximum reasoning and clear-history settings intact."""
    request_overrides = experiment_request_overrides(GLM_5_3_FLASH_MODEL)

    transformed = get_optional_params(
        model="zai-org/GLM-5.3-Flash",
        custom_llm_provider="hosted_vllm",
        drop_params=True,
        **experiment_decoding(GLM_5_3_FLASH_MODEL),
        **request_overrides,
    )

    assert transformed["extra_body"]["chat_template_kwargs"] == {
        "reasoning_effort": "max",
        "clear_thinking": True,
    }
    assert transformed["temperature"] == 1.0
    assert transformed["top_p"] == 0.95


def test_wikipedia_client_orders_and_persists_results(tmp_path) -> None:
    """Preserve MediaWiki rank and reuse results from the SQLite cache.

    Args:
        tmp_path: Pytest directory used for the isolated cache.
    """
    calls = []

    def transport(endpoint, params, timeout, headers):
        """Record request fields and return pages in reverse rank order.

        Args:
            endpoint: MediaWiki endpoint selected by the client.
            params: Generated API query parameters.
            timeout: Configured request timeout.
            headers: Generated request headers.

        Returns:
            MediaWiki-shaped response with explicit rank indices.
        """
        calls.append((endpoint, params, timeout, headers))
        return {
            "query": {
                "pages": [
                    {"pageid": 2, "index": 2, "title": "Second", "extract": "second text"},
                    {"pageid": 1, "index": 1, "title": "First", "extract": "first text"},
                ]
            }
        }

    cache_path = tmp_path / "wikipedia.sqlite3"
    client = WikipediaClient(cache_path=cache_path, transport=transport)
    first = client.search("  multi   hop  ", 2)
    second = client.search("multi hop", 2)

    assert [passage.title for passage in first] == ["First", "Second"]
    assert second == first
    assert len(calls) == 1
    assert calls[0][1]["generator"] == "search"
    assert calls[0][1]["prop"] == "extracts"
    assert "User-Agent" in calls[0][3]

    cached_client = WikipediaClient(
        cache_path=cache_path,
        transport=Mock(side_effect=AssertionError("cache miss")),
    )
    assert cached_client.search("multi hop", 2) == first


def test_hover_chain_of_thought_matches_dspy_field_protocol(monkeypatch) -> None:
    """Format and parse HoVer calls with the artifact's DSPy field protocol.

    Args:
        monkeypatch: Pytest fixture used to replace LiteLLM completion.
    """
    calls = []

    def completion(**kwargs):
        """Capture one completion request and return structured CoT fields.

        Args:
            **kwargs: LiteLLM completion arguments under test.

        Returns:
            Minimal response containing the first artifact-style completion.
        """
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=HOVER_COT_OUTPUTS[0]))])

    monkeypatch.setattr(hover_utils.litellm, "completion", completion)

    reasoning, summary = hover_utils._call_chain_of_thought(
        "Summarize the evidence.",
        {"claim": "claim", "passages": ["Page A | text", "Page B | other"]},
        "summary",
        QWEN3_8_27B_MODEL,
        None,
    )

    assert (reasoning, summary) == ("summary one reasoning", "summary one")
    assert calls[0]["max_tokens"] == 16_384
    assert "seed" not in calls[0]
    assert "extra_body" not in calls[0]
    system, user = [message["content"] for message in calls[0]["messages"]]
    assert "Your input fields are:\n1. `claim` (str): \n2. `passages` (str): " in system
    assert "Your output fields are:\n1. `reasoning` (str): \n2. `summary` (str): " in system
    assert "[[ ## reasoning ## ]]\n{reasoning}" in system
    assert "In adhering to this structure, your objective is: \n        Summarize the evidence." in system
    assert user.startswith(
        "[[ ## claim ## ]]\nclaim\n\n[[ ## passages ## ]]\n[1] «Page A | text»\n[2] «Page B | other»"
    )
    assert user.endswith("then ending with the marker for `[[ ## completed ## ]]`.")

    monkeypatch.setattr(hover_utils, "_call_lm", Mock(return_value="unstructured summary"))
    with pytest.raises(ValueError, match="omitted required fields"):
        hover_utils._call_chain_of_thought(
            "Summarize the evidence.",
            {"claim": "claim", "passages": ["Page A | text"]},
            "summary",
            QWEN3_8_27B_MODEL,
            None,
        )


@pytest.mark.skipif(hotpot_utils.dspy is None, reason="HotPotQA's locked DSPy group is not installed")
def test_hotpot_chat_adapter_repairs_only_expected_malformed_field_headers() -> None:
    """Accept Qwen's missing trailing hashes without weakening field checks."""
    dspy_module = hotpot_utils.dspy
    adapter_class = hotpot_utils._HotPotQAChatAdapter
    assert dspy_module is not None
    assert adapter_class is not None
    signature = dspy_module.ensure_signature("question->reasoning,summary")
    adapter = adapter_class()

    parsed = adapter.parse(
        signature,
        "[[ ## reasoning ## ]]\nBecause evidence.\n\n[[ ## summary ]]\nFinal summary.\n\n[[ ## completed ]]",
    )

    assert parsed == {"reasoning": "Because evidence.", "summary": "Final summary."}
    canonical = adapter.parse(
        signature,
        "[[ ## reasoning ## ]]\nBecause evidence.\n\n[[ ## summary ## ]]\nFinal summary.\n\n[[ ## completed ## ]]",
    )
    assert canonical == parsed
    with pytest.raises(ValueError, match="Expected"):
        adapter.parse(signature, "[[ ## reasoning ]]\nOnly reasoning.")
    with pytest.raises(ValueError, match="Failed to parse response as per signature"):
        adapter.parse(signature, None)


def test_hotpot_evaluator_scores_only_dspy_task_parse_failures_as_zero(monkeypatch) -> None:
    """Reject malformed candidate output without hiding systemic failures.

    Args:
        monkeypatch: Pytest fixture used to isolate task-program execution.
    """
    monkeypatch.setattr(hotpot_main, "build_hotpotqa_task_lm", Mock(return_value=object()))
    evaluator = hotpot_main.make_evaluator(
        QWEN3_8_27B_MODEL,
        FakeRetriever({}),
        program="2stage",
    )
    parse_error = ValueError(
        "Failed to parse response as per signature from original completion with input and num present and expected"
    )
    monkeypatch.setattr(hotpot_main, "run_program", Mock(side_effect=parse_error))

    score, side_info = evaluator(
        {"summarize1": "candidate"},
        {"question": "question", "answer": "answer"},
    )

    assert score == 0.0
    assert side_info == {
        "evaluation_error": {
            "type": "task_output_parse_error",
            "message": "Task-model output omitted DSPy's required structured fields; this example scored 0.",
        }
    }

    for systemic_error in (ValueError("unrelated candidate bug"), RuntimeError("provider unavailable")):
        monkeypatch.setattr(hotpot_main, "run_program", Mock(side_effect=systemic_error))
        with pytest.raises(type(systemic_error), match=str(systemic_error)):
            evaluator(
                {"summarize1": "candidate"},
                {"question": "question", "answer": "answer"},
            )


def test_hotpot_heldout_evaluation_checkpoints_and_reuses_predictions(monkeypatch, tmp_path) -> None:
    """Persist each held-out result and resume without repeating model calls.

    Args:
        monkeypatch: Pytest fixture used to isolate task-program execution.
        tmp_path: Pytest directory receiving held-out checkpoints.
    """
    task_lm = object()
    candidate = {"summarize1": "candidate"}
    dataset = [
        {"id": "first", "question": "first question", "answer": "Alpha"},
        {"id": "second", "question": "second question", "answer": "Beta delta"},
    ]
    run_program = Mock(
        side_effect=[
            ("query one", "Alpha", {}),
            ("query two", "Beta gamma", {}),
        ]
    )
    monkeypatch.setattr(hotpot_main, "build_hotpotqa_task_lm", Mock(return_value=task_lm))
    monkeypatch.setattr(hotpot_main, "run_program", run_program)

    first = hotpot_main.evaluate_on_set(
        candidate,
        dataset,
        QWEN3_8_27B_MODEL,
        FakeRetriever({}),
        None,
        max_workers=1,
        checkpoint_dir=tmp_path,
    )

    assert first == pytest.approx((0.5, 0.75))
    assert run_program.call_count == 2
    checkpoint_roots = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(checkpoint_roots) == 1
    checkpoint_root = checkpoint_roots[0]
    records = sorted(checkpoint_root.glob("*.json"))
    assert [path.name for path in records] == [
        "0000-a7937b64b8ca.json",
        "0001-16367aacb67a.json",
        "summary.json",
    ]
    assert json.loads((checkpoint_root / "summary.json").read_text()) == {
        "candidate_sha256": checkpoint_root.name,
        "example_count": 2,
        "exact_match": 0.5,
        "f1": 0.75,
        "schema_version": 1,
    }

    monkeypatch.setattr(hotpot_main, "run_program", Mock(side_effect=AssertionError("must resume")))
    second = hotpot_main.evaluate_on_set(
        candidate,
        dataset,
        QWEN3_8_27B_MODEL,
        FakeRetriever({}),
        None,
        max_workers=2,
        checkpoint_dir=tmp_path,
    )

    assert second == pytest.approx(first)


def test_hotpot_heldout_evaluation_scores_only_task_parse_errors_as_zero(monkeypatch, tmp_path) -> None:
    """Checkpoint expected DSPy parse failures without hiding other errors.

    Args:
        monkeypatch: Pytest fixture used to isolate task-program execution.
        tmp_path: Pytest directory receiving held-out checkpoints.
    """
    parse_error = ValueError("Failed to parse response as per signature from malformed task output")
    monkeypatch.setattr(hotpot_main, "build_hotpotqa_task_lm", Mock(return_value=object()))
    monkeypatch.setattr(hotpot_main, "run_program", Mock(side_effect=parse_error))

    scores = hotpot_main.evaluate_on_set(
        {"summarize1": "candidate"},
        [{"id": "broken", "question": "question", "answer": "answer"}],
        QWEN3_8_27B_MODEL,
        FakeRetriever({}),
        None,
        max_workers=1,
        checkpoint_dir=tmp_path,
    )

    assert scores == (0.0, 0.0)
    record_paths = [path for path in tmp_path.rglob("*.json") if path.name != "summary.json"]
    assert len(record_paths) == 1
    record = json.loads(record_paths[0].read_text())
    assert record["prediction"] is None
    assert record["task_output_parse_error"] is True

    monkeypatch.setattr(hotpot_main, "run_program", Mock(side_effect=RuntimeError("provider unavailable")))
    with pytest.raises(RuntimeError, match="provider unavailable"):
        hotpot_main.evaluate_on_set(
            {"summarize1": "different candidate"},
            [{"id": "broken", "question": "question", "answer": "answer"}],
            QWEN3_8_27B_MODEL,
            FakeRetriever({}),
            None,
            max_workers=1,
            checkpoint_dir=tmp_path,
        )


@pytest.mark.skipif(hotpot_utils.dspy is None, reason="HotPotQA's locked DSPy group is not installed")
def test_hotpot_chain_of_thought_uses_the_real_dspy_protocol() -> None:
    """Execute the artifact signature through DSPy's real ChatAdapter.

    This no-network integration test verifies that the optimized instruction
    remains the signature objective, passages keep DSPy's list rendering, and
    Chain-of-Thought returns both visible output fields.
    """
    dspy_module = hotpot_utils.dspy
    assert dspy_module is not None
    assert hotpot_utils.validate_hotpotqa_dspy_runtime() == (
        hotpot_utils.HOTPOTQA_DSPY_VERSION,
        hotpot_utils.HOTPOTQA_DSPY_COMMIT,
    )
    task_lm = dspy_module.utils.DummyLM([{"reasoning": "bridge reasoning", "summary": "bridge summary"}])

    reasoning, summary = hotpot_utils._call_chain_of_thought(
        "Summarize the evidence.",
        "question,passages->summary",
        {"question": "question", "passages": ["Page A | text", "Page B | other"]},
        "summary",
        task_lm,
    )

    assert (reasoning, summary) == ("bridge reasoning", "bridge summary")
    messages = task_lm.history[0]["messages"]
    system, user = [message["content"] for message in messages]
    assert "Your output fields are:\n1. `reasoning` (str): \n2. `summary` (str):" in system
    assert "In adhering to this structure, your objective is: \n        Summarize the evidence." in system
    assert "[[ ## passages ## ]]\n[1] «Page A | text»\n[2] «Page B | other»" in user
    assert user.endswith("then ending with the marker for `[[ ## completed ## ]]`.")


@pytest.mark.skipif(hotpot_utils.dspy is None, reason="HotPotQA's locked DSPy group is not installed")
def test_hotpot_four_component_program_runs_real_chain_of_thought_modules() -> None:
    """Execute all four artifact predictors through DSPy without a network.

    The real modules must preserve their distinct output schemas while only
    terminal fields—not their reasoning—flow into the next predictor.
    """
    dspy_module = hotpot_utils.dspy
    assert dspy_module is not None
    task_lm = dspy_module.utils.DummyLM(
        [
            {"reasoning": "summary one reasoning", "summary": "summary one"},
            {"reasoning": "bridge query reasoning", "query": "bridge query"},
            {"reasoning": "summary two reasoning", "summary": "summary two"},
            {"reasoning": "answer reasoning", "answer": "exact answer"},
        ]
    )
    retriever = FakeRetriever(
        {
            "original question": [WikipediaPassage("First page", "first")],
            "bridge query": [WikipediaPassage("Second page", "second")],
        }
    )

    query, answer, trace = hotpot_utils.run_two_stage(
        "summarize one",
        "query two",
        "summarize two",
        "answer",
        "original question",
        retriever,
        task_lm=task_lm,
    )

    assert (query, answer) == ("bridge query", "exact answer")
    assert retriever.calls == [("original question", 7), ("bridge query", 7)]
    assert trace["summary_1_reasoning"] == "summary one reasoning"
    assert trace["query_reasoning"] == "bridge query reasoning"
    assert trace["summary_2_reasoning"] == "summary two reasoning"
    assert trace["answer_reasoning"] == "answer reasoning"
    assert len(task_lm.history) == 4
    expected_terminals = ("summary", "query", "summary", "answer")
    for history, terminal in zip(task_lm.history, expected_terminals, strict=True):
        system = history["messages"][0]["content"]
        assert "1. `reasoning` (str)" in system
        assert f"2. `{terminal}` (str)" in system


def test_hotpot_task_lm_requires_the_locked_dspy_runtime(monkeypatch) -> None:
    """Fail before an experiment when DSPy is missing or has drifted.

    Args:
        monkeypatch: Pytest fixture used to simulate missing and mismatched
            task-program runtimes.
    """
    monkeypatch.setattr(hotpot_utils, "dspy", None)
    with pytest.raises(RuntimeError, match="requires DSPy"):
        hotpot_utils.build_hotpotqa_task_lm(QWEN3_8_27B_MODEL, None)

    monkeypatch.setattr(hotpot_utils, "dspy", SimpleNamespace())
    monkeypatch.setattr(hotpot_utils, "package_version", Mock(return_value="3.3.1"))
    with pytest.raises(RuntimeError, match="requires dspy==2.6.23"):
        hotpot_utils.build_hotpotqa_task_lm(QWEN3_8_27B_MODEL, None)

    monkeypatch.setattr(
        hotpot_utils,
        "package_version",
        Mock(return_value=hotpot_utils.HOTPOTQA_DSPY_VERSION),
    )
    monkeypatch.setattr(
        hotpot_utils,
        "package_distribution",
        Mock(
            return_value=SimpleNamespace(
                read_text=Mock(
                    return_value=json.dumps({"vcs_info": {"commit_id": "0000000000000000000000000000000000000000"}})
                )
            )
        ),
    )
    with pytest.raises(RuntimeError, match="requires DSPy commit"):
        hotpot_utils.build_hotpotqa_task_lm(QWEN3_8_27B_MODEL, None)


@pytest.mark.parametrize("model", [QWEN3_8_27B_MODEL, GLM_5_3_FLASH_MODEL])
def test_hotpot_dspy_lm_uses_the_selected_experiment_profile(monkeypatch, model: str) -> None:
    """Apply the selected solver's exact decoding settings to DSPy.

    Args:
        monkeypatch: Pytest fixture used to replace the DSPy LM constructor.
        model: Homogeneous experiment profile under test.
    """
    task_lm = object()
    lm_constructor = Mock(return_value=task_lm)
    settings = SimpleNamespace(configure=Mock())
    monkeypatch.setattr(hotpot_utils, "dspy", SimpleNamespace(LM=lm_constructor, settings=settings))
    monkeypatch.setattr(
        hotpot_utils,
        "package_version",
        Mock(return_value=hotpot_utils.HOTPOTQA_DSPY_VERSION),
    )
    monkeypatch.setattr(
        hotpot_utils,
        "package_distribution",
        Mock(
            return_value=SimpleNamespace(
                read_text=Mock(return_value=json.dumps({"vcs_info": {"commit_id": hotpot_utils.HOTPOTQA_DSPY_COMMIT}}))
            )
        ),
    )

    result = hotpot_utils.build_hotpotqa_task_lm(model, "http://solver.example/v1")

    assert result is task_lm
    settings.configure.assert_called_once_with(disable_history=True)
    expected_kwargs = {
        "model": model,
        "cache_in_memory": False,
        **hotpot_utils.resolve_hotpotqa_lm_kwargs(model, "http://solver.example/v1"),
    }
    lm_constructor.assert_called_once_with(**expected_kwargs)


@pytest.mark.parametrize(
    "model",
    [QWEN3_8_27B_OPENROUTER_MODEL, GLM_5_3_FLASH_OPENROUTER_MODEL],
)
def test_hotpot_dspy_lm_forwards_openrouter_routing(monkeypatch, model: str) -> None:
    """Keep the provider pin inside DSPy's forwarded request body.

    Args:
        monkeypatch: Pytest fixture used to replace the DSPy LM constructor.
        model: Effective OpenRouter runtime model under test.
    """
    lm_constructor = Mock(return_value=object())
    monkeypatch.setattr(
        hotpot_utils,
        "dspy",
        SimpleNamespace(LM=lm_constructor, settings=SimpleNamespace(configure=Mock())),
    )
    monkeypatch.setattr(hotpot_utils, "package_version", Mock(return_value=hotpot_utils.HOTPOTQA_DSPY_VERSION))
    monkeypatch.setattr(
        hotpot_utils,
        "package_distribution",
        Mock(
            return_value=SimpleNamespace(
                read_text=Mock(return_value=json.dumps({"vcs_info": {"commit_id": hotpot_utils.HOTPOTQA_DSPY_COMMIT}}))
            )
        ),
    )

    hotpot_utils.build_hotpotqa_task_lm(model, None)

    lm_constructor.assert_called_once_with(
        model=model,
        num_retries=EXPERIMENT_NUM_RETRIES,
        cache_in_memory=False,
        **experiment_decoding(model),
        **experiment_request_overrides(model),
    )


def test_hotpot_dspy_lm_accepts_the_resolved_technical_smoke_profile(monkeypatch) -> None:
    """Forward bounded Qwen settings into the pinned DSPy task client.

    Args:
        monkeypatch: Pytest fixture used to replace the DSPy LM constructor.
    """
    lm_constructor = Mock(return_value=object())
    monkeypatch.setattr(
        hotpot_utils,
        "dspy",
        SimpleNamespace(LM=lm_constructor, settings=SimpleNamespace(configure=Mock())),
    )
    monkeypatch.setattr(hotpot_utils, "package_version", Mock(return_value=hotpot_utils.HOTPOTQA_DSPY_VERSION))
    monkeypatch.setattr(
        hotpot_utils,
        "package_distribution",
        Mock(
            return_value=SimpleNamespace(
                read_text=Mock(return_value=json.dumps({"vcs_info": {"commit_id": hotpot_utils.HOTPOTQA_DSPY_COMMIT}}))
            )
        ),
    )
    model = QWEN3_8_27B_OPENROUTER_MODEL
    lm_kwargs = hotpot_utils.resolve_hotpotqa_lm_kwargs(model, None, "technical-smoke")

    hotpot_utils.build_hotpotqa_task_lm(model, None, lm_kwargs)

    assert lm_kwargs["max_tokens"] == 16_384
    assert lm_kwargs["extra_body"]["reasoning"] == {"effort": "low"}
    lm_constructor.assert_called_once_with(model=model, cache_in_memory=False, **lm_kwargs)


def test_hotpot_dspy_lm_uses_the_standard_local_glm_client(monkeypatch) -> None:
    """Use DSPy's normal local OpenAI-compatible client for GLM.

    Args:
        monkeypatch: Pytest fixture used to replace runtime validation and the
            DSPy client constructor.
    """
    lm_constructor = Mock(return_value=object())
    monkeypatch.setattr(
        hotpot_utils,
        "dspy",
        SimpleNamespace(LM=lm_constructor, settings=SimpleNamespace(configure=Mock())),
    )
    monkeypatch.setattr(
        hotpot_utils,
        "validate_hotpotqa_dspy_runtime",
        Mock(return_value=(hotpot_utils.HOTPOTQA_DSPY_VERSION, hotpot_utils.HOTPOTQA_DSPY_COMMIT)),
    )

    result = hotpot_utils.build_hotpotqa_task_lm(GLM_5_3_FLASH_MODEL, "http://127.0.0.1:8000/v1")

    assert result is lm_constructor.return_value
    lm_constructor.assert_called_once()
    assert lm_constructor.call_args.kwargs["api_base"] == "http://127.0.0.1:8000/v1"
    assert lm_constructor.call_args.kwargs["seed"] == hotpot_utils.HOTPOTQA_SCIENTIFIC_REQUEST_SEED


def test_hotpot_openrouter_launcher_mirrors_six_campaign_cells_per_model() -> None:
    """Lock the tiny matrix, data sizes, budgets, and runtime/provider profiles."""
    script = REPO_ROOT / "scripts" / "openrouter" / "run_hotpotqa_tiny.sh"
    environment = dict(os.environ)
    environment.pop("OPENROUTER_SMOKE_START_ARM", None)

    result = subprocess.run(
        [str(script), "--dry-run"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    plans = [line for line in result.stdout.splitlines() if line.startswith("PLAN ")]
    assert len(plans) == 12
    assert all(" --merge" not in line for line in plans)
    assert sum("--max-metric-calls 16" in line for line in plans) == 8
    assert sum("--max-metric-calls 32" in line for line in plans) == 4
    assert sum("--condition vanilla " in line for line in plans) == 4
    assert sum("--condition react_v2 " in line for line in plans) == 4
    assert sum("--condition action " in line for line in plans) == 2
    assert sum("--condition react_v2_random " in line for line in plans) == 2
    assert all("--condition random " not in line for line in plans)
    assert sum("--solver-model hosted_vllm/zai-org/GLM-5.3-Flash" in line for line in plans) == 6
    assert sum("--solver-model hosted_vllm/Qwen/Qwen3.8-27B" in line for line in plans) == 6
    assert all("--api-profile openrouter" in line for line in plans)
    assert all("--runtime-profile technical-smoke" in line for line in plans)
    assert all("--train-limit 6 --val-limit 5 --test-limit 2" in line for line in plans)
    assert all("--max-workers 1" in line for line in plans)
    assert all("--technical-mini-index" in line for line in plans)
    assert all("--technical-mini-index-dir" in line for line in plans)
    assert all("--wiki17-dir" not in line for line in plans)
    assert all("--condition both" not in line for line in plans)
    assert all(line.count("--solver-model") == 1 and line.count("--reflection-model") == 1 for line in plans)
    assert "NON-SCIENTIFIC selected-context technical-mini BM25 index" in result.stdout
    assert "48 GiB" not in script.read_text(encoding="utf-8")


def test_hotpot_openrouter_launcher_resumes_from_an_explicit_arm() -> None:
    """Skip completed paid arms only under an explicit stable run tag."""
    script = REPO_ROOT / "scripts" / "openrouter" / "run_hotpotqa_tiny.sh"
    environment = dict(os.environ)
    environment["OPENROUTER_SMOKE_START_ARM"] = "4"
    environment["SMOKE_TAG"] = "resume-test"

    result = subprocess.run(
        [str(script), "--dry-run"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    plans = [line for line in result.stdout.splitlines() if line.startswith("PLAN ")]
    assert len(plans) == 9
    assert plans[0].startswith("PLAN 4/12 glm-standard-action")
    assert plans[-1].startswith("PLAN 12/12 qwen-expanded-react-v2")
    assert "resume: arms 1-3 are skipped and not revalidated" in result.stdout


@pytest.mark.parametrize("start_arm", ["0", "13", "not-an-arm"])
def test_hotpot_openrouter_launcher_rejects_invalid_resume_arms(start_arm: str) -> None:
    """Reject invalid resume bounds before planning or paid requests.

    Args:
        start_arm: Invalid one-based arm index.
    """
    script = REPO_ROOT / "scripts" / "openrouter" / "run_hotpotqa_tiny.sh"
    environment = dict(os.environ)
    environment["OPENROUTER_SMOKE_START_ARM"] = start_arm
    environment["SMOKE_TAG"] = "resume-test"

    result = subprocess.run(
        [str(script), "--dry-run"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "PLAN " not in result.stdout


def test_hotpot_openrouter_launcher_requires_a_stable_tag_when_resuming() -> None:
    """Prevent a partial resume from silently receiving a new run identity."""
    script = REPO_ROOT / "scripts" / "openrouter" / "run_hotpotqa_tiny.sh"
    environment = dict(os.environ)
    environment["OPENROUTER_SMOKE_START_ARM"] = "4"
    environment.pop("SMOKE_TAG", None)

    result = subprocess.run(
        [str(script), "--dry-run"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "SMOKE_TAG is required" in result.stderr
    assert "PLAN " not in result.stdout


def test_hotpot_openrouter_launcher_refuses_execution_without_a_key() -> None:
    """Stop before endpoint checks or paid calls when credentials are absent."""
    script = REPO_ROOT / "scripts" / "openrouter" / "run_hotpotqa_tiny.sh"
    environment = dict(os.environ)
    environment.pop("OPENROUTER_API_KEY", None)
    environment.pop("OPENROUTER_SMOKE_START_ARM", None)

    result = subprocess.run(
        [str(script), "--execute"],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "OPENROUTER_API_KEY is required" in result.stderr
    assert "RUN 1/8" not in result.stdout


def test_hotpot_smoke_conversion_retains_gold_context_for_feedback() -> None:
    """Retain labeled context without retaining a solver-facing passage field."""
    examples = hotpot_utils._jsonl_to_examples(
        [
            {
                "id": "example",
                "question": "Question?",
                "answer": "Answer",
                "context": {"title": ["Leaked"], "sentences": [["Do not expose"]]},
                "supporting_facts": {"title": ["Leaked"], "sent_id": [0]},
                "passages": [{"title": "Leaked", "text": "Do not expose"}],
            }
        ]
    )

    assert examples == [
        {
            "question": "Question?",
            "answer": "Answer",
            "id": "example",
            "type": "",
            "level": "",
            "context": {"title": ["Leaked"], "sentences": [["Do not expose"]]},
            "supporting_facts": {"title": ["Leaked"], "sent_id": [0]},
        }
    ]
    assert "passages" not in examples[0]


def test_hotpot_production_loader_uses_the_artifact_split_and_retains_labels(monkeypatch) -> None:
    """Load fullwiki using the artifact's ordered pools and seed-one samples.

    Args:
        monkeypatch: Pytest fixture used to replace the Hugging Face loader.
    """
    calls = []
    records = [
        {
            "id": str(index),
            "question": f"Question {index}",
            "answer": f"Answer {index}",
            "context": {"title": [f"Gold {index}"], "sentences": [[f"Evidence {index}"]]},
            "supporting_facts": {"title": ["Gold"], "sent_id": [0]},
        }
        for index in range(1000)
    ]

    def load_dataset(name, config, **kwargs):
        """Capture dataset selection and return deterministic fullwiki splits.

        Args:
            name: Requested Hugging Face dataset name.
            config: Requested dataset configuration.
            **kwargs: Loader options supplied by the production path.

        Returns:
            Raw training and validation records.
        """
        calls.append((name, config, kwargs))
        return {
            "train": records,
            "validation": [{"id": "must-not-be-used"}],
        }

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    train, val, test = hotpot_utils.load_hotpotqa_dataset()

    assert calls == [
        (
            "hotpot_qa",
            "fullwiki",
            {"revision": hotpot_utils.HOTPOTQA_HF_REVISION},
        )
    ]
    assert (len(train), len(val), len(test)) == (150, 300, 300)
    assert [example["id"] for example in train] == [
        str(index) for index in random.Random(1).sample(list(range(800, 1000)), 150)
    ]
    assert [example["id"] for example in val] == [
        str(index) for index in random.Random(1).sample(list(range(400, 800)), 300)
    ]
    assert [example["id"] for example in test] == [
        str(index) for index in random.Random(1).sample(list(range(400)), 300)
    ]
    assert all("context" in example and "supporting_facts" in example for example in train + val + test)
    assert all(example["id"] != "must-not-be-used" for example in train + val + test)


def test_hotpot_nonzero_seed_remixes_only_selected_train_and_validation(monkeypatch) -> None:
    """Keep test fixed while applying artifact seed remixes to optimization data.

    Args:
        monkeypatch: Pytest fixture used to replace the Hugging Face loader.
    """
    records = [
        {
            "id": str(index),
            "question": f"Question {index}",
            "answer": f"Answer {index}",
            "context": {"title": [], "sentences": []},
            "supporting_facts": {"title": [], "sent_id": []},
        }
        for index in range(1000)
    ]
    monkeypatch.setattr(datasets, "load_dataset", lambda *_args, **_kwargs: {"train": records})

    base_train, base_val, base_test = hotpot_utils.load_hotpotqa_dataset(seed=0)
    mixed_train, mixed_val, mixed_test = hotpot_utils.load_hotpotqa_dataset(seed=7)
    expected = base_train + base_val
    random.Random(7).shuffle(expected)

    assert mixed_train == expected[:150]
    assert mixed_val == expected[150:]
    assert mixed_test == base_test


def test_hotpot_production_loader_never_falls_back_implicitly(monkeypatch) -> None:
    """Require explicit smoke selection when fullwiki is unavailable.

    Args:
        monkeypatch: Pytest fixture used to force an offline loader failure.
    """
    monkeypatch.setattr(datasets, "load_dataset", Mock(side_effect=OSError("offline")))

    with pytest.raises(RuntimeError, match="explicit smoke run"):
        hotpot_utils.load_hotpotqa_dataset()


def test_hotpot_program_executes_two_wikipedia_hops(monkeypatch) -> None:
    """Use the generated bridge query for the second Wikipedia retrieval.

    Args:
        monkeypatch: Pytest fixture used to provide deterministic LM outputs.
    """
    chain_of_thought = Mock(side_effect=HOTPOT_COT_RESULTS)
    monkeypatch.setattr(hotpot_utils, "_call_chain_of_thought", chain_of_thought)
    task_lm = object()
    retriever = FakeRetriever(
        {
            "original question": [WikipediaPassage("First page", "first")],
            "bridge query": [WikipediaPassage("Second page", "second")],
        }
    )

    query, answer, trace = hotpot_utils.run_two_stage(
        "summarize one",
        "query two",
        "summarize two",
        "answer",
        "original question",
        retriever,
        retrieval_k=7,
        task_lm=task_lm,
    )

    assert query == "bridge query"
    assert answer == "exact answer"
    assert retriever.calls == [("original question", 7), ("bridge query", 7)]
    assert chain_of_thought.call_args_list == [
        call(
            "summarize one",
            "question,passages->summary",
            {"question": "original question", "passages": ["First page | first"]},
            "summary",
            task_lm,
        ),
        call(
            "query two",
            "question,summary_1->query",
            {"question": "original question", "summary_1": "summary one"},
            "query",
            task_lm,
        ),
        call(
            "summarize two",
            "question,context,passages->summary",
            {
                "question": "original question",
                "context": "summary one",
                "passages": ["Second page | second"],
            },
            "summary",
            task_lm,
        ),
        call(
            "answer",
            "question,summary_1,summary_2->answer",
            {"question": "original question", "summary_1": "summary one", "summary_2": "summary two"},
            "answer",
            task_lm,
        ),
    ]
    assert trace == {
        "hop1_documents": [WikipediaPassage("First page", "first")],
        "summary_1_reasoning": "summary one reasoning",
        "summary_1": "summary one",
        "query_reasoning": "bridge query reasoning",
        "query": "bridge query",
        "hop2_documents": [WikipediaPassage("Second page", "second")],
        "summary_2_reasoning": "summary two reasoning",
        "summary_2": "summary two",
        "answer_reasoning": "answer reasoning",
        "answer": "exact answer",
    }


def test_hotpot_component_feedback_uses_gold_only_after_execution() -> None:
    """Give each component its own oracle feedback without gold-input leakage."""
    example = {
        "question": "Which bridge fact answers this?",
        "answer": "target",
        "context": {
            "title": ["First page", "Missing page"],
            "sentences": [["First supporting sentence."], ["Secret supporting sentence."]],
        },
        "supporting_facts": {"title": ["First page", "Missing page"], "sent_id": [0, 0]},
    }
    trace = {
        "hop1_documents": [WikipediaPassage("First page", "retrieved first abstract")],
        "summary_1_reasoning": "first reasoning",
        "summary_1": "summary one",
        "query_reasoning": "query reasoning",
        "query": "bridge query",
        "hop2_documents": [WikipediaPassage("Other page", "retrieved second abstract")],
        "summary_2_reasoning": "second reasoning",
        "summary_2": "summary two",
        "answer_reasoning": "answer reasoning",
        "answer": "wrong",
    }

    records = hotpot_utils.artifact_component_records(example, trace, 0.0)

    assert set(records) == {"summarize1", "create_query_hop2", "summarize2", "final_answer"}
    assert records["summarize1"]["Inputs"]["passages"] == ["First page | retrieved first abstract"]
    assert "Secret supporting sentence." not in str(records["summarize1"]["Inputs"])
    assert "Secret supporting sentence." in records["summarize1"]["Feedback"]
    assert records["summarize1"]["Generated Outputs"] == {
        "reasoning": "first reasoning",
        "summary": "summary one",
    }
    assert records["create_query_hop2"]["Generated Outputs"] == {
        "reasoning": "query reasoning",
        "query": "bridge query",
    }
    assert records["summarize2"]["Generated Outputs"] == {
        "reasoning": "second reasoning",
        "summary": "summary two",
    }
    assert records["final_answer"]["Generated Outputs"] == {
        "reasoning": "answer reasoning",
        "answer": "wrong",
    }
    assert "correct answer is: target" in records["final_answer"]["Feedback"]


def test_hotpot_metric_uses_exact_match_as_primary_score() -> None:
    """Keep token overlap in feedback without promoting it above exact match."""
    score, feedback = hotpot_utils.hotpotqa_metric("Paris France", "Paris")

    assert score == 0.0
    assert "token-F1" in feedback
    assert "EM=0" in feedback


def test_hotpot_normalization_matches_the_artifact_unicode_behavior() -> None:
    """Normalize canonically equivalent Unicode answers to identical NFD text."""
    composed = hotpot_utils.normalize_answer("The Caf\u00e9")
    decomposed = hotpot_utils.normalize_answer("cafe\u0301")

    assert composed == decomposed
    assert composed == "cafe\u0301"


def test_hotpot_f1_uses_ordinary_token_overlap_for_yes_no_answers() -> None:
    """Avoid adding the non-artifact yes/no exact-match guard to token F1."""
    assert hotpot_utils.f1_score("yes perhaps", "yes") == pytest.approx(2 / 3)
    assert hotpot_utils.f1_score("no", "yes") == 0.0


def test_hover_loader_reproduces_artifact_splits_and_seed_remixing(tmp_path, monkeypatch) -> None:
    """Reproduce the artifact's exact pool split, sampling, and seed remix.

    Args:
        tmp_path: Pytest directory containing the synthetic official release.
        monkeypatch: Pytest fixture used to redirect the pinned data source and
            scale its eligible-record invariant to the synthetic release.
    """
    records = []
    for index in range(1000):
        records.append(
            {
                "uid": str(index),
                "claim": f"Claim {index}",
                "supporting_facts": [
                    {"key": "Page A", "value": 0},
                    {"key": "Page A", "value": 1},
                    {"key": "Page B", "value": 0},
                    {"key": "Page C", "value": 0},
                ],
                "label": "SUPPORTED",
                "num_hops": 3,
            }
        )
    records.append(
        {
            "uid": "excluded",
            "claim": "Only two documents",
            "supporting_facts": [["Page A", 0], ["Page B", 0]],
            "label": "SUPPORTED",
            "num_hops": 3,
        }
    )
    path = tmp_path / hover_utils.HOVER_TRAIN_FILE
    path.write_text(json.dumps(records), encoding="utf-8")
    monkeypatch.setattr(hover_utils, "ensure_data_downloaded", Mock(return_value=path))
    monkeypatch.setattr(hover_utils, "HOVER_ELIGIBLE_COUNT", 1000)

    train, val, test = hover_utils.load_hover_dataset(data_dir=tmp_path)
    shuffled_ids = [str(index) for index in range(1000)]
    random.Random(0).shuffle(shuffled_ids)
    first_boundary = int(0.4 * len(shuffled_ids))
    second_boundary = int(0.8 * len(shuffled_ids))
    expected_test = random.Random(1).sample(shuffled_ids[:first_boundary], 300)
    expected_val = random.Random(1).sample(shuffled_ids[first_boundary:second_boundary], 300)
    expected_train = random.Random(1).sample(shuffled_ids[second_boundary:], 150)

    assert (len(train), len(val), len(test)) == (150, 300, 300)
    assert [example["id"] for example in train] == expected_train
    assert [example["id"] for example in val] == expected_val
    assert [example["id"] for example in test] == expected_test
    assert all(example["gold_titles"] == ["Page A", "Page B", "Page C"] for example in train + val + test)
    assert all(example["id"] != "excluded" for example in train + val + test)

    train_seeded, val_seeded, test_seeded = hover_utils.load_hover_dataset(seed=7, data_dir=tmp_path)
    remixed_ids = expected_train + expected_val
    random.Random(7).shuffle(remixed_ids)

    assert [example["id"] for example in train_seeded] == remixed_ids[:150]
    assert [example["id"] for example in val_seeded] == remixed_ids[150:]
    assert [example["id"] for example in test_seeded] == expected_test


def test_hover_release_identity_is_pinned_to_the_artifact_source() -> None:
    """Lock the official release revision, byte identity, and eligible count."""
    assert hover_utils.HOVER_HF_REVISION == "c0e43052759879b3461642ca6c0dd26658f47691"
    assert hover_utils.HOVER_SOURCE_REVISION == "39b84697f196308f398a251a7aea9b82ae0f0562"
    assert hover_utils.HOVER_SOURCE_REVISION in hover_utils.HOVER_TRAIN_URL
    assert hover_utils.HOVER_TRAIN_SHA256 == "1f1cd57abd616fa00c70bdc575ce77c16fc6cf1a6cffd5ff87c208030a336bb6"
    assert hover_utils.HOVER_TRAIN_SIZE == 9_205_582
    assert hover_utils.HOVER_ELIGIBLE_COUNT == 6_084


def test_hover_release_validation_rejects_corrupt_existing_and_downloaded_bytes(tmp_path, monkeypatch) -> None:
    """Reject corrupt HoVer bytes before they can define an experiment split.

    Args:
        tmp_path: Pytest directory containing isolated existing and download
            destinations.
        monkeypatch: Pytest fixture used to replace the network downloader.
    """
    existing_dir = tmp_path / "existing"
    existing_dir.mkdir()
    (existing_dir / hover_utils.HOVER_TRAIN_FILE).write_bytes(b"corrupt")

    with pytest.raises(RuntimeError, match="does not match the pinned v1.1 artifact"):
        hover_utils.ensure_data_downloaded(existing_dir)

    download_dir = tmp_path / "download"

    def write_corrupt_download(_url, destination) -> None:
        """Write invalid bytes in place of the remote HoVer artifact.

        Args:
            _url: Download URL retained for the ``urlretrieve`` signature.
            destination: Partial-file path that receives invalid bytes.
        """
        Path(destination).write_bytes(b"corrupt")

    monkeypatch.setattr(hover_utils.urllib.request, "urlretrieve", write_corrupt_download)
    with pytest.raises(RuntimeError, match="Could not download the official HoVer v1.1 data"):
        hover_utils.ensure_data_downloaded(download_dir)

    assert not (download_dir / f"{hover_utils.HOVER_TRAIN_FILE}.part").exists()


def test_hover_eligibility_counts_distinct_raw_titles_before_normalization() -> None:
    """Mirror the artifact when distinct raw titles normalize to one title."""
    record = {
        "supporting_facts": [
            ["The Page", 0],
            ["Page", 0],
            ["Other", 0],
            ["The Page", 1],
        ]
    }

    assert hover_utils._supporting_titles(record) == ["The Page", "Page", "Other"]


def test_hover_program_scores_pages_retrieved_across_three_hops(monkeypatch) -> None:
    """Accumulate pages from every HOVER retrieval hop before scoring.

    Args:
        monkeypatch: Pytest fixture used to provide deterministic LM outputs.
    """
    monkeypatch.setattr(
        hover_utils,
        "_call_lm",
        Mock(side_effect=HOVER_COT_OUTPUTS),
    )
    retriever = FakeRetriever(
        {
            "claim": [WikipediaPassage("Page A", "a")],
            "query two": [WikipediaPassage("Page B", "b")],
            "query three": [WikipediaPassage("Page C", "c")],
        }
    )

    queries, passages, trace = hover_utils.run_two_stage(
        "summarize one",
        "query two prompt",
        "summarize two",
        "query three prompt",
        "claim",
        retriever,
    )
    example = {"gold_titles": ["Page A", "Page B", "Page C"]}
    score, feedback = hover_utils.hover_metric(passages, example)

    assert json.loads(queries) == ["query two", "query three"]
    assert retriever.calls == [("claim", 7), ("query two", 7), ("query three", 10)]
    assert score == 1.0
    assert hover_utils.hover_recall(passages, example) == 1.0
    assert "Retrieved 3/3" in feedback
    assert trace["hop1_documents"] == [WikipediaPassage("Page A", "a")]
    assert trace["summary_1_reasoning"] == "summary one reasoning"
    assert trace["hop2_documents"] == [WikipediaPassage("Page B", "b")]
    assert trace["query_2_reasoning"] == "query two reasoning"
    assert trace["hop3_documents"] == [WikipediaPassage("Page C", "c")]
    assert trace["query_3_reasoning"] == "query three reasoning"
    assert trace["retrieved_documents"] == passages

    records = hover_utils.artifact_component_records({"claim": "claim", **example}, trace, score)
    assert list(records) == ["summarize1", "create_query_hop2", "summarize2", "create_query_hop3"]
    assert records["summarize1"]["Inputs"] == {"claim": "claim", "passages": ["Page A | a"]}
    assert records["create_query_hop2"]["Generated Outputs"] == {
        "reasoning": "query two reasoning",
        "query": "query two",
    }
    assert records["summarize2"]["Inputs"]["context"] == "summary one"
    assert records["create_query_hop3"]["Generated Outputs"] == {
        "reasoning": "query three reasoning",
        "query": "query three",
    }
    assert all("gold_titles" not in record["Inputs"] for record in records.values())


def test_hover_component_feedback_attributes_each_retrieval_hop() -> None:
    """Give each HoVer component only its artifact-equivalent diagnosis."""
    trace = {
        "hop1_documents": [WikipediaPassage("Page A", "a")],
        "summary_1": "summary one",
        "query_2": "query two",
        "hop2_documents": [WikipediaPassage("Page B", "b")],
        "summary_2": "summary two",
        "query_3": "query three",
        "hop3_documents": [WikipediaPassage("Page D", "d")],
    }
    records = hover_utils.artifact_component_records(
        {"claim": "claim", "gold_titles": ["Page A", "Page B", "Page C"]},
        trace,
        0.0,
    )

    assert "page b" in records["summarize1"]["Feedback"]
    assert "page b" in records["create_query_hop2"]["Feedback"]
    assert "page c" in records["summarize2"]["Feedback"]
    assert "page c" in records["create_query_hop3"]["Feedback"]
    assert "page b" not in records["create_query_hop3"]["Feedback"]


def test_hover_evaluator_exposes_only_four_component_specific_records(monkeypatch) -> None:
    """Prevent global feedback from leaking into every optimized component.

    Args:
        monkeypatch: Pytest fixture used to provide deterministic LM outputs.
    """
    monkeypatch.setattr(
        hover_utils,
        "_call_lm",
        Mock(side_effect=HOVER_COT_OUTPUTS),
    )
    retriever = FakeRetriever(
        {
            "claim": [WikipediaPassage("Page A", "a")],
            "query two": [WikipediaPassage("Page B", "b")],
            "query three": [WikipediaPassage("Page C", "c")],
        }
    )
    evaluator = make_hover_evaluator(QWEN3_8_27B_MODEL, retriever)
    candidate = {
        "summarize1": "summarize one",
        "create_query_hop2": "query two prompt",
        "summarize2": "summarize two",
        "create_query_hop3": "query three prompt",
    }

    score, side_info = evaluator(
        candidate,
        {"claim": "claim", "prompt": "claim", "gold_titles": ["Page A", "Page B", "Page C"]},
    )

    assert score == 1.0
    assert set(side_info) == {
        "summarize1_specific_info",
        "create_query_hop2_specific_info",
        "summarize2_specific_info",
        "create_query_hop3_specific_info",
    }
    assert all(set(record) == {"Inputs", "Generated Outputs", "Feedback"} for record in side_info.values())


def test_hover_passage_rendering_does_not_truncate_artifact_inputs() -> None:
    """Render every retrieved abstract even when the combined text is large."""
    passages = [
        WikipediaPassage("First", "a" * 12_000),
        WikipediaPassage("Second", "tail"),
    ]

    rendered = hover_utils._render_passages(passages)

    assert rendered == [f"First | {'a' * 12_000}", "Second | tail"]


def test_hover_smoke_mode_is_explicit_and_offline() -> None:
    """Provide three offline HOVER examples only when smoke mode is explicit."""
    train, val, test = hover_utils.load_hover_dataset(smoke=True)

    assert (len(train), len(val), len(test)) == (1, 1, 1)
    assert all(len(example["gold_titles"]) == 3 for example in train + val + test)


def test_hover_sbatch_defaults_are_compatible_with_react_v2() -> None:
    """Keep the HoVer launcher on the artifact substrate and paired defaults."""
    script = (REPO_ROOT / "examples" / "hover" / "run_hover.sbatch").read_text()

    assert 'CONDITION="${CONDITION:-both}"' in script
    assert 'SEED_STYLE="${SEED_STYLE:-structured}"' in script
    assert 'EXPERIMENT_SEED="${EXPERIMENT_SEED:-0}"' in script
    assert 'MAX_WORKERS="${MAX_WORKERS:-32}"' in script
    assert 'RETRIEVAL_K="${RETRIEVAL_K:-7}"' in script
    assert 'FINAL_RETRIEVAL_K="${FINAL_RETRIEVAL_K:-10}"' in script
    assert 'MODEL_PROFILE="${MODEL_PROFILE:-qwen3.8-27b}"' in script
    assert 'MODEL="${MODEL:-Qwen3.8-27B}"' in script
    assert 'SOLVER_MODEL="hosted_vllm/Qwen/Qwen3.8-27B"' in script
    assert "#SBATCH --cpus-per-task=8" in script
    assert "export JAX_PLATFORMS=cpu" in script
    assert '--data-dir "${HOVER_DATA_DIR}"' in script
    assert '--wiki17-dir "${WIKI17_DIR}"' in script
    assert '--max-workers "${MAX_WORKERS}"' in script
    assert '--seed "${EXPERIMENT_SEED}"' in script
    assert "load_hover_dataset(seed=0" in script
    assert "--wikipedia-endpoint" not in script


@pytest.mark.parametrize("benchmark", ["hotpotqa", "hover"])
def test_wikipedia_python_defaults_use_the_qwen_experiment_pair(benchmark: str) -> None:
    """Default both Python model roles to the Qwen3.8-27B condition.

    Args:
        benchmark: Wikipedia benchmark whose Python entrypoint is inspected.
    """
    source = (REPO_ROOT / "examples" / benchmark / "main.py").read_text()

    assert source.count("default=QWEN3_8_27B_MODEL") == 2
    assert "validate_experiment_model_pair(args.solver_model, args.reflection_model)" in source


@pytest.mark.parametrize("benchmark", ["hotpotqa", "hover"])
def test_wikipedia_sbatch_exposes_both_homogeneous_model_profiles(benchmark: str) -> None:
    """Run either experiment model in both roles without mixing providers.

    Args:
        benchmark: Wikipedia benchmark whose batch script is inspected.
    """
    script = (REPO_ROOT / "examples" / benchmark / f"run_{benchmark}.sbatch").read_text()

    assert 'MODEL_PROFILE="${MODEL_PROFILE:-qwen3.8-27b}"' in script
    assert 'SOLVER_MODEL="hosted_vllm/Qwen/Qwen3.8-27B"' in script
    secondary_model = (
        'SOLVER_MODEL="hosted_vllm/zai-org/GLM-5.3-Flash"'
        if benchmark == "hotpotqa"
        else 'SOLVER_MODEL="deepseek/deepseek-v4-flash"'
    )
    assert secondary_model in script
    assert 'REFLECTION_MODEL="${SOLVER_MODEL}"' in script
    if benchmark == "hover":
        assert 'if [[ "${LOCAL_SOLVER}" == "1" ]]' in script
    assert 'SOLVER_API_ARG=(--solver-api-base "${SOLVER_API_BASE}")' in script
    assert 'REFLECTION_API_ARG=(--reflection-api-base "${REFLECTION_API_BASE}")' in script
    if benchmark == "hotpotqa":
        assert 'export OPENAI_API_KEY="EMPTY"' in script
        assert "DEEPSEEK_API_KEY" not in script
        assert 'SERVING_ENGINE="sglang"' in script
    else:
        assert 'export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"' in script
        assert 'export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"' in script


@pytest.mark.parametrize("benchmark", ["hotpotqa", "hover"])
def test_wikipedia_della_submit_scales_resources_by_model_profile(benchmark: str) -> None:
    """Request the model-specific Della resources for each benchmark.

    Args:
        benchmark: Wikipedia benchmark whose Della wrapper is inspected.
    """
    submit = (REPO_ROOT / "scripts" / "della" / f"submit_{benchmark}.sh").read_text()

    assert 'DELLA_GPUS="${DELLA_GPUS:-}"' in submit
    assert 'DELLA_CPUS_PER_TASK="${DELLA_CPUS_PER_TASK:-}"' in submit
    assert 'DELLA_MEMORY="${DELLA_MEMORY:-}"' in submit
    assert 'DELLA_GPUS="${DELLA_GPUS:-8}"' in submit
    assert 'DELLA_CPUS_PER_TASK="${DELLA_CPUS_PER_TASK:-64}"' in submit
    assert 'DELLA_MEMORY="${DELLA_MEMORY:-768G}"' in submit
    assert 'JOB_PARTITION="${GPU_PARTITION}"' in submit
    assert 'MAX_WORKERS="${MAX_WORKERS:-128}"' in submit
    assert 'VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-${DELLA_GPUS}}"' in submit
    assert 'VLLM_API_SERVER_COUNT="${VLLM_API_SERVER_COUNT:-${VLLM_DATA_PARALLEL_SIZE}}"' in submit
    if benchmark == "hotpotqa":
        assert "DELLA_GPUS=0" not in submit
        assert "glm-5.3-flash)" in submit
        assert 'VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-8}"' in submit
        assert 'VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-1}"' in submit
        assert 'VLLM_API_SERVER_COUNT="${VLLM_API_SERVER_COUNT:-1}"' in submit
        assert 'MAX_WORKERS="${MAX_WORKERS:-8}"' in submit
    else:
        assert "DELLA_GPUS=0" in submit
        assert 'DELLA_MEMORY="${DELLA_MEMORY:-128G}"' in submit
        assert 'JOB_PARTITION="${CPU_PARTITION:-}"' in submit
        assert 'MAX_WORKERS="${MAX_WORKERS:-64}"' in submit
        assert "VLLM_DATA_PARALLEL_SIZE=1" in submit
        assert "VLLM_API_SERVER_COUNT=1" in submit
    assert '"--cpus-per-task=${DELLA_CPUS_PER_TASK}"' in submit
    assert '"--mem=${DELLA_MEMORY}"' in submit
    assert 'if [[ -n "${JOB_PARTITION}" ]]; then' in submit
    assert 'SBATCH_RESOURCE_ARGS+=("--partition=${JOB_PARTITION}")' in submit
    assert "if (( DELLA_GPUS > 0 )); then" in submit
    assert 'SBATCH_RESOURCE_ARGS+=("--gres=gpu:${DELLA_GPUS}")' in submit
    if benchmark == "hotpotqa":
        assert r'"\${SBATCH_BIN}"${SBATCH_RESOURCE_COMMAND}' in submit
    else:
        assert "sbatch${SBATCH_RESOURCE_COMMAND}" in submit


@pytest.mark.parametrize("benchmark", ["hotpotqa", "hover"])
def test_wikipedia_sbatch_configures_within_run_vllm_throughput(benchmark: str) -> None:
    """Batch independent examples across data-parallel Qwen replicas.

    Args:
        benchmark: Wikipedia benchmark whose batch script is inspected.
    """
    script = (REPO_ROOT / "examples" / benchmark / f"run_{benchmark}.sbatch").read_text()

    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --mem=128G" in script
    assert "#SBATCH --gres" not in script
    assert 'VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-1}"' in script
    assert (
        'VLLM_API_SERVER_COUNT="${VLLM_API_SERVER_COUNT:-${VLLM_DATA_PARALLEL_SIZE}}"'
        in script
    )
    expected_max_num_seqs = "1" if benchmark == "hotpotqa" else "64"
    assert f'VLLM_MAX_NUM_SEQS="${{VLLM_MAX_NUM_SEQS:-{expected_max_num_seqs}}}"' in script
    assert 'VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"' in script
    assert "--tensor-parallel-size 1" in script
    if benchmark == "hotpotqa":
        assert "--data-parallel-size 8" in script
        assert "--api-server-count 8" in script
        assert "--max-num-seqs 1" in script
    else:
        assert '--data-parallel-size "${VLLM_DATA_PARALLEL_SIZE}"' in script
        assert '--api-server-count "${VLLM_API_SERVER_COUNT}"' in script
        assert '--max-num-seqs "${VLLM_MAX_NUM_SEQS}"' in script
    assert '--max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}"' in script
    if benchmark == "hotpotqa":
        assert "--no-enable-prefix-caching" in script
    else:
        assert "--enable-prefix-caching" in script
    assert "--language-model-only" in script


@pytest.mark.parametrize("benchmark", ["hotpotqa", "hover"])
def test_wikipedia_sbatch_limits_nested_cpu_threads_after_vllm_starts(benchmark: str) -> None:
    """Apply CPU thread caps to evaluators without throttling the vLLM server.

    Args:
        benchmark: Wikipedia benchmark whose batch script is inspected.
    """
    script = (REPO_ROOT / "examples" / benchmark / f"run_{benchmark}.sbatch").read_text()
    vllm_start = script.index('"${VLLM_BIN}" serve "${SOLVER_MODEL_PATH}"')
    evaluator_start = script.index(f'"${{PY}}" -m examples.{benchmark}.main')

    for variable, default in (
        ("OMP_NUM_THREADS", "1"),
        ("MKL_NUM_THREADS", "1"),
        ("OPENBLAS_NUM_THREADS", "1"),
        ("NUMEXPR_NUM_THREADS", "1"),
        ("TOKENIZERS_PARALLELISM", "false"),
    ):
        if benchmark == "hotpotqa":
            export = f"export {variable}={default}"
        else:
            export = f'export {variable}="${{{variable}:-{default}}}"'
        assert f"-u {variable}" in script[vllm_start - 300 : vllm_start]
        assert script.count(export) == 1
        assert vllm_start < script.index(export) < evaluator_start


@pytest.mark.parametrize(
    "script_path",
    [
        "scripts/della/submit_hotpotqa.sh",
        "scripts/della/submit_hover.sh",
        "examples/hotpotqa/run_hotpotqa.sbatch",
        "examples/hover/run_hover.sbatch",
    ],
)
def test_wikipedia_della_launch_scripts_have_valid_bash_syntax(script_path: str) -> None:
    """Parse every Wikipedia Della launcher with Bash.

    Args:
        script_path: Repository-relative launcher path.
    """
    subprocess.run(
        ["bash", "-n", str(REPO_ROOT / script_path)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_hotpotqa_della_launchers_enforce_the_scientific_matrix() -> None:
    """Pin methodology while retaining only quality-neutral throughput knobs."""
    submit = (REPO_ROOT / "scripts" / "della" / "submit_hotpotqa.sh").read_text()
    sbatch = (REPO_ROOT / "examples" / "hotpotqa" / "run_hotpotqa.sbatch").read_text()
    build = (REPO_ROOT / "scripts" / "della" / "build_env.sh").read_text()
    sync = (REPO_ROOT / "scripts" / "della" / "sync_to_della.sh").read_text()

    assert "REFLECTION_MODEL" in submit
    assert "REFLECTION_API_BASE" in submit
    assert 'MODEL_PROFILE="${MODEL_PROFILE:-qwen3.8-27b}"' in submit
    assert 'BUDGET_PROFILE="${BUDGET_PROFILE:-campaign}"' in submit
    assert 'HOTPOTQA_CAMPAIGN_ID="${HOTPOTQA_CAMPAIGN_ID:-hotpotqa-final-v1}"' in submit
    assert "MAX_METRIC_CALLS=6871" in submit
    assert 'STANDARD_TIME="${STANDARD_TIME:-${TIME:-72:00:00}}"' in submit
    assert "MAX_METRIC_CALLS=13742" in submit
    assert 'EXPANDED_TIME="${EXPANDED_TIME:-${TIME:-144:00:00}}"' in submit
    assert 'CONDITION="${CONDITION:-all}"' in submit
    assert 'MAX_WORKERS="${MAX_WORKERS:-}"' in submit
    assert 'MODEL="Qwen3.8-27B"' in submit
    assert 'if [[ "${GPU_PARTITION}" != "ailab" ]]' in submit
    assert "Qwen3.8-27B production runs require GPU_PARTITION=ailab" in submit
    assert "scientific Qwen runs require 8 H200 data-parallel replicas and 8 API servers" in submit
    assert 'SOLVER_MODEL_PATH="${MODEL_STORAGE}/${MODEL}"' in submit
    assert 'SOLVER_MODEL="hosted_vllm/Qwen/Qwen3.8-27B"' in submit
    assert 'SOLVER_MODEL="hosted_vllm/zai-org/GLM-5.3-Flash"' in submit
    assert 'REFLECTION_MODEL="${SOLVER_MODEL}"' in submit
    assert "DEEPSEEK_API_KEY" not in submit
    assert 'GLM_SGLANG_IMAGE="${MODEL_STORAGE}/runtimes/sglang-glm-5.3-flash-x86_64.sif"' in submit
    assert r'"BUDGET_PROFILE=\${run_budget_profile}"' in submit
    assert "HOTPOTQA_CAMPAIGN_ID=${HOTPOTQA_CAMPAIGN_ID}" in submit
    assert 'SBATCH_BIN="\\$(command -v sbatch)"' in submit
    assert 'SBATCH_HELP="\\$("\\${SBATCH_BIN}" --help 2>&1)"' in submit
    assert '[[ "\\${SBATCH_HELP}" != *"--export-file"* ]]' in submit
    assert "umask 077" in submit
    for local_script in (submit, build, sync):
        assert '[[ -L "${ENV_FILE}" || ! -O "${ENV_FILE}" ]]' in local_script
        assert "(8#${ENV_MODE} & 8#077)" in local_script
        assert "chmod 600 ${ENV_FILE}" in local_script
    assert 'SBATCH_EXPORT_FILE="\\$(mktemp)"' in submit
    assert "cleanup_export_file()" in submit
    assert 'rm -f -- "\\${SBATCH_EXPORT_FILE}"' in submit
    assert "printf '%s\\0'" in submit
    assert "env -i" in submit
    assert 'HOME="\\${HOME}"' in submit
    assert 'PATH="\\${PATH}"' in submit
    assert "LANG=C.UTF-8" in submit
    assert "LC_ALL=C.UTF-8" in submit
    assert '"HOME=\\${HOME}"' in submit
    assert '"PATH=\\${PATH}"' in submit
    assert "--export=ALL" in submit
    assert '--export-file="\\${SBATCH_EXPORT_FILE}"' in submit
    assert "HOTPOTQA_PRODUCTION_LAUNCH=1" in submit
    assert 'HOTPOTQA_SOURCE_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"' in submit
    assert 'REMOTE_SOURCE_DIR="${REMOTE_DIR%/}/sources/${HOTPOTQA_SOURCE_COMMIT}"' in submit
    assert 'GEPA_VENV_DIR="${REMOTE_DIR%/}/.venv"' in submit
    assert submit.count('"${GEPA_VENV_DIR}/bin/python"') == 5
    assert 'SYNC_SOURCE_COMMIT="${HOTPOTQA_SOURCE_COMMIT}"' in submit
    assert 'SYNC_REMOTE_DIR="${REMOTE_SOURCE_DIR}"' in submit
    assert 'SYNC_MANIFEST_OUTPUT="${SOURCE_MANIFEST_OUTPUT}"' in submit
    assert 'cd "${REMOTE_SOURCE_DIR}"' in submit
    assert 'export PYTHONPATH="${REMOTE_SOURCE_DIR}/src:${REMOTE_SOURCE_DIR}"' in submit
    assert 'HOTPOTQA_ENV_SPEC_SHA256="\\$(' in submit
    assert "printf 'python=%s\\n' \"\\${HOTPOTQA_PYTHON_VERSION}\"" in submit
    assert "printf 'uv=%s\\n' \"\\${HOTPOTQA_UV_VERSION}\"" in submit
    assert 'export UV_PROJECT_ENVIRONMENT="${GEPA_VENV_DIR}"' in submit
    assert "--frozen --check --no-install-project" in submit
    assert '"${GEPA_VENV_DIR}/.gepa-env-spec.sha256"' in submit
    assert '"${GEPA_VENV_DIR}/.gepa-python-version"' in submit
    assert '"${GEPA_VENV_DIR}/.gepa-uv-version"' in submit
    assert '"${GEPA_VENV_DIR}/.gepa-uv-sha256"' in submit
    assert "HOTPOTQA_ENV_SPEC_SHA256=\\${HOTPOTQA_ENV_SPEC_SHA256}" in submit
    assert "HOTPOTQA_GEPA_ENV_SHA256=\\${HOTPOTQA_GEPA_ENV_SHA256}" in submit
    assert "HOTPOTQA_POSIT_ENV_SHA256=\\${HOTPOTQA_POSIT_ENV_SHA256}" in submit
    assert 'examples.common.python_environment verify --path "\\${GEPA_ENV_MANIFEST}"' in submit
    assert "load_hotpotqa_dataset(seed=0)" in submit
    assert "SUBMIT_BUDGET_PROFILES=(standard standard standard standard expanded expanded)" in submit
    assert "SUBMIT_CONDITIONS=(vanilla react_v2 react_v2_random action vanilla react_v2)" in submit
    assert "SUBMIT_CONDITIONS=(vanilla random" not in submit
    assert "SUBMIT_BUDGET_PROFILES=(standard standard standard standard)" in submit
    assert "SUBMIT_CONDITIONS=(vanilla react_v2 react_v2_random action)" in submit
    assert "SUBMIT_BUDGET_PROFILES=(expanded expanded)" in submit
    assert "SUBMIT_CONDITIONS=(vanilla react_v2)" in submit
    assert r'RUN_BUDGET_PROFILE="\${SUBMIT_BUDGET_PROFILES[\${CELL_INDEX}]}"' in submit
    assert "RUN_MAX_METRIC_CALLS=6871" in submit
    assert "RUN_MAX_METRIC_CALLS=13742" in submit
    assert 'RUN_TIME="${STANDARD_TIME}"' in submit
    assert 'RUN_TIME="${EXPANDED_TIME}"' in submit
    assert 'DEPENDENCY_ARGS+=("--dependency=afterok:\\${PREVIOUS_JOB_ID}")' in submit
    assert "afterany:" not in submit
    assert r'"CONDITION=\${run_condition}"' in submit
    assert r'--job-name="gepa-hp-${MODEL_PROFILE}-\${RUN_BUDGET_PROFILE}-\${RUN_CONDITION}"' in submit
    assert r'--time="\${RUN_TIME}"' in submit
    assert 'sha256sum "\\${POSIT_ENV_MANIFEST}"' in submit
    assert 'pip check --python "\\${VLLM_PY}"' in submit
    assert ".gepa-source-commit" in submit
    assert ".gepa-source-manifest.sha256sums" in submit
    assert "HOTPOTQA_SOURCE_MANIFEST_SHA256=${HOTPOTQA_SOURCE_MANIFEST_SHA256}" in submit
    assert 'if [[ "${MODEL_PROFILE}" == "glm-5.3-flash" ]]' in submit
    assert "commit the complete experiment source" in submit
    assert '"${SCRIPT_DIR}/sync_to_della.sh"' in submit
    assert "NO_SYNC" not in submit
    assert "sshpass" not in submit
    assert "REMOTE_PASSWORD" not in submit
    assert "-o BatchMode=yes -o StrictHostKeyChecking=yes" in submit
    for forbidden_export in (
        "PROGRAM=",
        "SEED_STYLE=",
        "DATA_PATH=",
        "TRAIN_LIMIT=",
        "VAL_LIMIT=",
        "TEST_LIMIT=",
        "MERGE=",
        "EXPERIMENT_SEED=",
        "RETRIEVAL_K=",
    ):
        assert forbidden_export not in submit

    assert "set -euo pipefail" in sbatch
    assert 'if [[ "${HOTPOTQA_PRODUCTION_LAUNCH:-}" != "1" ]]' in sbatch
    assert 'HOTPOTQA_CAMPAIGN_ID="${HOTPOTQA_CAMPAIGN_ID:-}"' in sbatch
    assert "submit this production job through scripts/della/submit_hotpotqa.sh" in sbatch
    assert "#SBATCH --cpus-per-task=8" in sbatch
    assert 'BUDGET_PROFILE="${BUDGET_PROFILE:-standard}"' in sbatch
    assert "EXPECTED_MAX_METRIC_CALLS=6871" in sbatch
    assert "EXPECTED_MAX_METRIC_CALLS=13742" in sbatch
    assert 'MODEL="Qwen3.8-27B"' in sbatch
    assert 'SOLVER_MODEL_PATH="${MODEL_STORAGE}/${MODEL}"' in sbatch
    assert 'QWEN_REVISION="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"' in sbatch
    assert 'GLM_REVISION="04c4e9e95c5da8862dced7e5056455116f83a7e0"' in sbatch
    assert "DeepSeek" not in sbatch
    assert 'HOTPOTQA_MODEL_REVISION="${QWEN_REVISION}"' in sbatch
    assert 'HOTPOTQA_MODEL_REVISION="${GLM_REVISION}"' in sbatch
    assert 'SOLVER_API_BASE="http://127.0.0.1:${GEN_PORT}/v1"' in sbatch
    assert 'HOTPOTQA_MODEL_INTEGRITY_SHA256="$(sha256sum "${SOLVER_MODEL_PATH}/.gepa-model-integrity.json"' in sbatch
    assert "examples.common.model_snapshot verify" in sbatch
    assert '--model-profile "${MODEL_SNAPSHOT_PROFILE}"' in sbatch
    assert 'examples.common.wiki17_bm25 verify --deep --root "${WIKI17_DIR}"' in sbatch
    assert 'HOTPOTQA_VERIFIED_WIKI17_INTEGRITY_SHA256="${WIKI17_INTEGRITY_SHA256}"' in sbatch
    assert "export HOTPOTQA_VERIFIED_WIKI17_INTEGRITY_SHA256" in sbatch
    assert '--wiki17-dir "${WIKI17_DIR}"' in sbatch
    assert '--max-workers "${MAX_WORKERS}"' in sbatch
    assert 'CONDITION="${CONDITION:-}"' in sbatch
    assert "the requested condition and budget are not an approved HotPotQA campaign cell" in sbatch
    assert "standard:vanilla|standard:react_v2|standard:react_v2_random|standard:action" in sbatch
    assert "expanded:vanilla|expanded:react_v2" in sbatch
    for rejected_cell in (
        "standard:random",
        "expanded:random",
        "expanded:action",
        "expanded:react_v2_random",
    ):
        assert rejected_cell not in sbatch
    assert 'RUN_LOCK_PATH="${RUN_LOCK_DIR}/${MODEL_PROFILE}-${BUDGET_PROFILE}-${CONDITION}.lock"' in sbatch
    assert 'if ! flock -n "${RUN_LOCK_FD}"' in sbatch
    assert "another HotPotQA job is already writing" in sbatch
    assert "proxy/default" not in sbatch
    assert "GEN_MAX_LEN=262144" in sbatch
    assert "max_model_len=${GEN_MAX_LEN}" in sbatch
    assert "--dtype bfloat16" in sbatch
    assert "--kv-cache-dtype auto" in sbatch
    assert "--no-enable-prefix-caching" in sbatch
    assert "--enable-prefix-caching" not in sbatch
    assert "--seed 0" in sbatch
    assert "unset VLLM_BATCH_INVARIANT" in sbatch
    assert 'HOTPOTQA_VLLM_BATCH_INVARIANT="false"' in sbatch
    assert 'HOTPOTQA_VLLM_SINGLE_SEQUENCE_REPLICAS="true"' in sbatch
    assert 'GEN_PORT="${GEN_PORT:-}"' in sbatch
    assert 'listener.bind(("127.0.0.1", requested_port))' in sbatch
    assert 'SOLVER_API_BASE="http://127.0.0.1:${GEN_PORT}/v1"' in sbatch
    assert "generator_reports_expected_model()" in sbatch
    assert "ids == [sys.argv[1]]" in sbatch
    assert sbatch.index('if ! kill -0 "${GEN_PID}"') < sbatch.index(
        "if generator_reports_expected_model"
    )
    assert "--reasoning-parser qwen3" in sbatch
    assert "--enable-auto-tool-choice" in sbatch
    assert "--tool-call-parser qwen3_coder" in sbatch
    assert 'Version("0.17.0")' in sbatch
    assert '"${VLLM_BIN}" serve --help=all' in sbatch
    assert '"${VLLM_BIN}" serve --help 2>&1' not in sbatch
    assert 'echo "==> checking native tool-call compatibility"' in sbatch
    assert "from examples.hotpotqa.utils import resolve_hotpotqa_lm_kwargs" in sbatch
    assert 'tool_choice="auto"' in sbatch
    assert 'export DSPY_CACHEDIR="${SCRATCH_BASE}/.cache/dspy/hotpotqa/${RUNTIME_CACHE_KEY}"' in sbatch
    assert "export HOTPOTQA_MODEL_REVISION" in sbatch
    assert "export HOTPOTQA_MODEL_INTEGRITY_SHA256" in sbatch
    assert "export HOTPOTQA_VLLM_VERSION" in sbatch
    assert "export HOTPOTQA_TORCH_VERSION" in sbatch
    assert "export HOTPOTQA_CUDA_VERSION" in sbatch
    assert "export HOTPOTQA_CUDA_MODULE" in sbatch
    assert "export HOTPOTQA_TRANSFORMERS_VERSION" in sbatch
    assert "export HOTPOTQA_POSIT_COMMIT" in sbatch
    assert "export HOTPOTQA_POSIT_ENV_SHA256" in sbatch
    assert "export HOTPOTQA_GPU_RUNTIME" in sbatch
    assert "export HOTPOTQA_SOURCE_COMMIT" in sbatch
    assert "export HOTPOTQA_SOURCE_MANIFEST_SHA256" in sbatch
    assert "export HOTPOTQA_PYTHON_VERSION" in sbatch
    assert "export HOTPOTQA_UV_VERSION" in sbatch
    assert "export HOTPOTQA_UV_SHA256" in sbatch
    assert "export HOTPOTQA_ENV_SPEC_SHA256" in sbatch
    assert "export HOTPOTQA_GEPA_ENV_SHA256" in sbatch
    assert 'CURRENT_ENV_SPEC_SHA256="$(' in sbatch
    assert 'EXPECTED_HOTPOTQA_PYTHON_VERSION="3.11.13"' in sbatch
    assert 'EXPECTED_HOTPOTQA_UV_VERSION="0.9.13"' in sbatch
    assert 'export UV_PROJECT_ENVIRONMENT="${GEPA_VENV_DIR}"' in sbatch
    assert "--frozen --check --no-install-project" in sbatch
    assert 'if ! flock -s -n "${ARTIFACT_LOCK_FD}"' in sbatch
    assert '"${GEPA_VENV_DIR}/.gepa-env-spec.sha256"' in sbatch
    assert 'GEPA_VENV_DIR="${GEPA_VENV_DIR:-}"' in sbatch
    assert 'PY="${GEPA_VENV_DIR}/bin/python"' in sbatch
    assert 'export PYTHONPATH="${PWD}/src:${PWD}"' in sbatch
    assert 'examples.common.python_environment verify --path "${GEPA_ENV_MANIFEST}"' in sbatch
    assert ".gepa-source-commit" in sbatch
    assert "export HOTPOTQA_LITELLM_VERSION" in sbatch
    assert "export HOTPOTQA_WEIGHT_DTYPE" in sbatch
    assert "export HOTPOTQA_KV_CACHE_DTYPE" in sbatch
    assert "export HOTPOTQA_SERVE_ARGUMENTS" in sbatch
    assert "export HOTPOTQA_VLLM_BATCH_INVARIANT" in sbatch
    assert "export HOTPOTQA_VLLM_SINGLE_SEQUENCE_REPLICAS" in sbatch
    assert "batch_invariant=false" in sbatch
    assert "single_sequence_replicas=true" in sbatch
    assert "gpu_memory_utilization=${GEN_GMU}" in sbatch
    assert "max_model_len=${GEN_MAX_LEN}" in sbatch
    assert "rope_scaling=none" in sbatch
    assert "--rope-scaling" not in sbatch
    assert "max_num_seqs=1" in sbatch
    assert "posit_env=${HOTPOTQA_POSIT_ENV_SHA256}" in sbatch
    assert "gpu=${HOTPOTQA_GPU_RUNTIME}" in sbatch
    assert 'examples.common.python_environment verify --path "${POSIT_ENV_MANIFEST}"' in sbatch
    assert 'pip check --python "${VLLM_PY}"' in sbatch
    assert '"H200" not in name.upper() or capability != "9.0"' in sbatch
    assert '"nvidia-smi", "--query-gpu=driver_version"' in sbatch
    assert 'exec {MODEL_LOCK_FD}<"${SOLVER_MODEL_PATH}"' in sbatch
    assert 'if ! flock -s -n "${MODEL_LOCK_FD}"' in sbatch
    assert 'HOTPOTQA_VLLM_VERSION=""' in sbatch
    assert 'HOTPOTQA_SGLANG_VERSION=""' in sbatch
    assert 'HOTPOTQA_SERVING_IMAGE_URI=""' in sbatch
    assert 'HOTPOTQA_SERVING_IMAGE_SHA256=""' in sbatch
    assert 'HOTPOTQA_CUDA_MODULE=""' in sbatch
    assert 'WIKI17_INTEGRITY_SHA256="$(sha256sum "${WIKI17_DIR}/integrity.json"' in sbatch
    assert 'HOTPOTQA_CUDA_MODULE="cudatoolkit/${HOTPOTQA_CUDA_VERSION}"' in sbatch
    assert 'module load "${HOTPOTQA_CUDA_MODULE}"' in sbatch
    assert 'module is-loaded "${HOTPOTQA_CUDA_MODULE}"' in sbatch
    assert "module load cudatoolkit/13.0" not in sbatch
    assert "cuda_module=${HOTPOTQA_CUDA_MODULE}" in sbatch
    assert "RUNTIME_IDENTITY_SHA256=" in sbatch
    assert "CACHE_IDENTITY_SHA256=" in sbatch
    assert sbatch.count("litellm=${HOTPOTQA_LITELLM_VERSION}") >= 2
    assert "model_manifest=${HOTPOTQA_MODEL_INTEGRITY_SHA256}" in sbatch
    assert "source_manifest=${HOTPOTQA_SOURCE_MANIFEST_SHA256}" in sbatch
    assert "python=${HOTPOTQA_PYTHON_VERSION}" in sbatch
    assert "uv=${HOTPOTQA_UV_VERSION}" in sbatch
    assert "uv_sha=${HOTPOTQA_UV_SHA256}" in sbatch
    assert "api_base=${SOLVER_API_BASE}" in sbatch
    assert "engine=${HOTPOTQA_SERVING_ENGINE}" in sbatch
    assert "sglang=${HOTPOTQA_SGLANG_VERSION}" in sbatch
    assert "image_uri=${HOTPOTQA_SERVING_IMAGE_URI}" in sbatch
    assert "image_sha=${HOTPOTQA_SERVING_IMAGE_SHA256}" in sbatch
    assert "HOTPOTQA_DEEPSEEK" not in sbatch
    assert "env_spec=${HOTPOTQA_ENV_SPEC_SHA256}" in sbatch
    assert "gepa_env=${HOTPOTQA_GEPA_ENV_SHA256}" in sbatch
    assert "budget_profile=${BUDGET_PROFILE}" in sbatch
    assert 'RUNTIME_CACHE_KEY="${MODEL_PROFILE}-${CACHE_IDENTITY_SHA256}"' in sbatch
    assert 'CAMPAIGN_LOCK_DIR="${SCRATCH_BASE}/.cache/gepa/hotpotqa-campaign/${HOTPOTQA_CAMPAIGN_ID}"' in sbatch
    assert 'CAMPAIGN_LOCK_PATH="${CAMPAIGN_LOCK_DIR}/${MODEL_PROFILE}.sha256"' in sbatch
    assert 'DATA_CAMPAIGN_LOCK_PATH="${CAMPAIGN_LOCK_DIR}/data-and-source.sha256"' in sbatch
    assert 'ln "${DATA_CAMPAIGN_LOCK_TEMP}" "${DATA_CAMPAIGN_LOCK_PATH}"' in sbatch
    assert 'ln "${CAMPAIGN_LOCK_TEMP}" "${CAMPAIGN_LOCK_PATH}"' in sbatch
    assert "noclobber" not in sbatch
    assert "source, dependency lock, HotPotQA data, DSPy, or Wiki-2017 index differs" in sbatch
    assert "source or quality-relevant runtime differs from the existing campaign lock" in sbatch
    for scientific_argument in (
        "--api-profile direct",
        "--runtime-profile scientific",
        "--enforce-scientific-contract",
        "--program 2stage",
        "--seed-style structured",
        "--seed 0",
        "--retrieval-k 7",
        "--reflection-level 2",
        "--edit-tool-set broad",
        "--template-family auto",
    ):
        assert scientific_argument in sbatch
    assert "--merge" not in sbatch
    assert "--data-path" not in sbatch
    assert "--train-limit" not in sbatch
    assert "--val-limit" not in sbatch
    assert "--test-limit" not in sbatch

    assert "validate_hotpotqa_dspy_runtime" in submit
    assert "sshpass" not in build
    assert "REMOTE_PASSWORD" not in build
    assert "-o BatchMode=yes -o StrictHostKeyChecking=yes" in build
    assert "sshpass" not in sync
    assert "REMOTE_PASSWORD" not in sync
    assert sync.count("-o BatchMode=yes -o StrictHostKeyChecking=yes") == 2
    assert 'HOTPOTQA_PYTHON_VERSION="3.11.13"' in build
    assert 'HOTPOTQA_UV_VERSION="0.9.13"' in build
    assert "UV_UNMANAGED_INSTALL" in build
    assert "--frozen --no-install-project" in build
    assert "--frozen --check --no-install-project" in build
    assert 'if ! flock -n "\\${ARTIFACT_LOCK_FD}"' in build
    assert "validate_hotpotqa_dspy_runtime" in build
    assert 'examples.common.python_environment prepare --path "\\${POSIT_ENV_MANIFEST}"' in build
    assert 'examples.common.python_environment prepare --path "\\${GEPA_ENV_MANIFEST}"' in build
    assert 'pip check --python "\\${VLLM_PY}"' in build
    assert 'git -C "\\${POSIT_DIR}" status --porcelain --untracked-files=normal' in build
    assert 'HOTPOTQA_ENV_SPEC_SHA256="\\$(' in build
    assert ".venv/.gepa-env-spec.sha256" in build
    assert ".venv/.gepa-python-version" in build
    assert ".venv/.gepa-uv-version" in build
    assert ".venv/.gepa-uv-sha256" in build
    assert "examples.common.model_snapshot prepare" in build
    assert "examples.common.model_snapshot verify" in build
    assert '--model-profile qwen3.8-27b --root "${QWEN_MODEL_DIR}"' in build
    assert '--model-profile glm-5.3-flash --root "${GLM_MODEL_DIR}"' in build
    assert 'exec {MODEL_LOCK_FD}<"${QWEN_MODEL_DIR}"' in build
    assert 'if ! flock -n "\\${MODEL_LOCK_FD}"' in build
    assert 'examples.common.wiki17_bm25 verify --deep --root "${WIKI17_DIR}"' in build
    assert "load_hotpotqa_dataset(seed=0)" in build
    assert "load_hover_dataset" not in build
    assert "--exclude '.cache/'" in sync
    assert "--exclude 'logs/'" in sync
    assert "--exclude 'sources/'" in sync
    assert 'git -C "${REPO_ROOT}" archive "${SYNC_SOURCE_COMMIT}"' in sync
    assert 'if [[ -z "${SYNC_SOURCE_COMMIT}" ]]' in sync
    assert '"${RSYNC_EXCLUDES[@]}"' in sync
    assert "--exclude '/outputs/'" in sync
    assert "--exclude '/gepa-hotpotqa-*.log'" in sync
    assert ".gepa-source-manifest.sha256sums" in sync
    assert 'SYNC_MANIFEST_OUTPUT="${SYNC_MANIFEST_OUTPUT:-}"' in sync
    assert 'EXPECTED_REMOTE_SOURCE_DIR="${REMOTE_DIR%/}/sources/${SYNC_SOURCE_COMMIT}"' in sync
    assert 'printf \'%s\\n\' "${SYNC_SOURCE_COMMIT}" > "${SOURCE_STAGE_DIR}/.gepa-source-commit"' in sync
    assert '"mkdir -p -- ${REMOTE_SOURCE_DIR_QUOTED}"' in sync


def test_hotpotqa_execution_lock_rejects_a_duplicate_writer(tmp_path: Path) -> None:
    """Hold one run lock and verify a second non-blocking writer is rejected.

    Args:
        tmp_path: Isolated directory containing the simulated run lock.
    """
    lock_path = tmp_path / "campaign-qwen3.8-27b-standard-react_v2.lock"
    with lock_path.open("w") as owner:
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        contender = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import fcntl, pathlib, sys; "
                    "handle = pathlib.Path(sys.argv[1]).open('w'); "
                    "fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)"
                ),
                str(lock_path),
            ],
            capture_output=True,
            text=True,
        )

    assert contender.returncode != 0


def test_hover_della_submit_preserves_artifact_methodology() -> None:
    """Export every HoVer artifact axis through the Della submission wrapper."""
    submit = (REPO_ROOT / "scripts" / "della" / "submit_hover.sh").read_text()

    assert 'MAX_METRIC_CALLS="${MAX_METRIC_CALLS:-7051}"' in submit
    assert 'EXPERIMENT_SEED="${EXPERIMENT_SEED:-0}"' in submit
    assert 'MAX_WORKERS="${MAX_WORKERS:-}"' in submit
    assert 'RETRIEVAL_K="${RETRIEVAL_K:-7}"' in submit
    assert 'FINAL_RETRIEVAL_K="${FINAL_RETRIEVAL_K:-10}"' in submit
    assert 'MODEL="${MODEL:-Qwen3.8-27B}"' in submit
    assert 'SOLVER_MODEL="hosted_vllm/Qwen/Qwen3.8-27B"' in submit
    assert 'SOLVER_MODEL="deepseek/deepseek-v4-flash"' in submit
    assert 'REFLECTION_MODEL="${SOLVER_MODEL}"' in submit
    assert "DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}" in submit
    assert "examples.common.wiki17_bm25 verify" in submit
    assert "load_hover_dataset(seed=0" in submit
    assert "WIKI17_DIR=${WIKI17_DIR}" in submit
    assert "HOVER_DATA_DIR=${HOVER_DATA_DIR}" in submit
    for variable in (
        "PROGRAM",
        "SEED_STYLE",
        "TAG",
        "TRAIN_LIMIT",
        "VAL_LIMIT",
        "TEST_LIMIT",
        "SMOKE",
        "GEN_GMU",
        "GEN_MAX_LEN",
        "HEALTH_TIMEOUT",
        "POSIT_DIR",
    ):
        assert f"{variable}=${{{variable}}}" in submit
