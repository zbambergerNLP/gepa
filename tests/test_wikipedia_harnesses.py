"""Tests for the Wikipedia-backed HotPotQA runner."""

import asyncio
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import datasets
import litellm
import pytest
from litellm.utils import get_optional_params

sys.path.insert(0, str(Path(__file__).parents[1]))

from examples.common import provider_retries
from examples.common.experiment_models import (
    DEEPSEEK_V4_1_FLASH_MODEL,
    QWEN3_8_27B_MODEL,
    experiment_decoding,
    experiment_request_overrides,
)
from examples.common.model_settings import SCIENTIFIC_REQUEST_SEED
from examples.common.wikipedia import WikipediaPassage
from examples.hotpotqa import utils as hotpot_utils

REPO_ROOT = Path(__file__).parents[1]
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


@pytest.mark.parametrize("model", [QWEN3_8_27B_MODEL, DEEPSEEK_V4_1_FLASH_MODEL])
def test_litellm_preserves_local_thinking_and_effort_settings(model: str) -> None:
    """Keep both providers' explicit thinking controls in the outgoing vLLM request."""
    request_overrides = experiment_request_overrides(model, explicit_reasoning=True)

    transformed = get_optional_params(
        model=model.removeprefix("hosted_vllm/"),
        custom_llm_provider="hosted_vllm",
        drop_params=True,
        **experiment_decoding(model),
        **request_overrides,
    )

    assert transformed["extra_body"]["chat_template_kwargs"] == request_overrides["extra_body"]["chat_template_kwargs"]
    if model == QWEN3_8_27B_MODEL:
        assert transformed["extra_body"]["top_k"] == 20
    assert transformed["temperature"] == 1.0
    assert transformed["top_p"] == 0.95


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


@pytest.mark.skipif(hotpot_utils.dspy is None, reason="HotPotQA's locked DSPy group is not installed")
@pytest.mark.parametrize("model", [QWEN3_8_27B_MODEL, DEEPSEEK_V4_1_FLASH_MODEL])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_real_dspy_provider_requests_share_three_retries_for_transport_and_output(
    tmp_path, monkeypatch, model, asynchronous
):
    """Exercise the pinned DSPy transport with the same bounded retry policy."""
    raw = litellm.ModelResponse(
        model=model,
        choices=[{"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
        usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    )
    unfinished = litellm.ModelResponse(
        model=model,
        choices=[{"message": {"role": "assistant", "content": None}, "finish_reason": "length"}],
        usage={"prompt_tokens": 10, "completion_tokens": 65536, "total_tokens": 65546},
    )
    outcomes = [ConnectionError("temporary"), unfinished, ConnectionError("temporary"), raw]
    provider = AsyncMock(side_effect=outcomes) if asynchronous else Mock(side_effect=outcomes)
    monkeypatch.setattr(litellm, "acompletion" if asynchronous else "completion", provider)
    monkeypatch.setattr(provider_retries.time, "sleep", Mock())
    monkeypatch.setattr(provider_retries.asyncio, "sleep", AsyncMock())
    path = tmp_path / "provider-attempts.jsonl"
    settings = hotpot_utils.resolve_benchmark_lm_kwargs(model, "http://localhost:8000/v1")
    settings.update(provider_retries.provider_retry_kwargs(path, "solver"))
    lm = hotpot_utils.build_hotpotqa_task_lm(model, None, settings)
    result = asyncio.run(lm.aforward(prompt="test")) if asynchronous else lm.forward(prompt="test")
    assert result.choices[0].message.content == "done"
    assert provider.call_count == 4
    assert all(call.kwargs["num_retries"] == call.kwargs["max_retries"] == 0 for call in provider.call_args_list)
    assert all(call.kwargs["max_tokens"] == 65_536 for call in provider.call_args_list)
    assert [call.kwargs["seed"] for call in provider.call_args_list] == [0, 0, 1, 1]
    assert len(path.read_text().splitlines()) == 4


@pytest.mark.skipif(hotpot_utils.dspy is None, reason="HotPotQA's locked DSPy group is not installed")
@pytest.mark.parametrize("model", [QWEN3_8_27B_MODEL, DEEPSEEK_V4_1_FLASH_MODEL])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_repeated_hotpot_dspy_requests_are_fresh(tmp_path, monkeypatch, model, asynchronous):
    """Bypass both DSPy caches and call the provider for identical new requests."""
    responses = [
        litellm.ModelResponse(
            model=model,
            choices=[{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        )
        for text in ("first result", "second result")
    ]
    provider = AsyncMock(side_effect=responses) if asynchronous else Mock(side_effect=responses)
    monkeypatch.setattr(litellm, "acompletion" if asynchronous else "completion", provider)
    monkeypatch.setattr(hotpot_utils.dspy.cache, "get", Mock(side_effect=AssertionError("cache read")))
    monkeypatch.setattr(hotpot_utils.dspy.cache, "put", Mock(side_effect=AssertionError("cache write")))
    path = tmp_path / "provider-attempts.jsonl"
    settings = hotpot_utils.resolve_benchmark_lm_kwargs(model, "http://localhost:8000/v1")
    settings.update(provider_retries.provider_retry_kwargs(path, "solver"))
    settings.update(cache=True, cache_in_memory=True)
    lm = hotpot_utils.build_hotpotqa_task_lm(model, None, settings)
    assert lm.cache is False and lm.cache_in_memory is False
    outputs = []
    for _ in range(2):
        response = asyncio.run(lm.aforward(prompt="same input")) if asynchronous else lm.forward(prompt="same input")
        outputs.append(response.choices[0].message.content)
    assert outputs == ["first result", "second result"]
    assert provider.call_count == 2
    assert all(call.kwargs["cache"] == {"no-cache": True, "no-store": True} for call in provider.call_args_list)
    assert len(path.read_text().splitlines()) == 2


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


@pytest.mark.parametrize("model", [QWEN3_8_27B_MODEL, DEEPSEEK_V4_1_FLASH_MODEL])
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
        "cache": False,
        "cache_in_memory": False,
        **hotpot_utils.resolve_benchmark_lm_kwargs(model, "http://solver.example/v1"),
    }
    lm_constructor.assert_called_once_with(**expected_kwargs)


def test_hotpot_dspy_lm_uses_the_standard_local_deepseek_client(monkeypatch) -> None:
    """Use DSPy's normal local OpenAI-compatible client for DeepSeek.

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

    result = hotpot_utils.build_hotpotqa_task_lm(DEEPSEEK_V4_1_FLASH_MODEL, "http://127.0.0.1:8000/v1")

    assert result is lm_constructor.return_value
    lm_constructor.assert_called_once()
    assert lm_constructor.call_args.kwargs["api_base"] == "http://127.0.0.1:8000/v1"
    assert lm_constructor.call_args.kwargs["seed"] == SCIENTIFIC_REQUEST_SEED


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

    diagnosed = hotpot_utils.artifact_component_records(example, trace, 0.0, include_diagnostics=True)
    for name, record in diagnosed.items():
        assert {k: record[k] for k in records[name]} == records[name]
        assert record["End-to-end Outcome"]["score"] == 0.0
        assert "not a causal score" in record["End-to-end Outcome"]["attribution"]
        assert "Component Context" not in records[name]
        assert "Secret supporting sentence." not in str(record["Inputs"])
    assert "without the retrieved passages" in diagnosed["summarize2"]["Component Context"]["downstream"]


def test_hotpot_metric_uses_exact_match_as_primary_score() -> None:
    """Score partial overlap as incorrect without calculating a supplemental metric."""
    score, feedback = hotpot_utils.hotpotqa_metric("Paris France", "Paris")

    assert score == 0.0
    assert "F1" not in feedback
    assert "EM=0" in feedback


def test_hotpot_normalization_matches_the_artifact_unicode_behavior() -> None:
    """Normalize canonically equivalent Unicode answers to identical NFD text."""
    composed = hotpot_utils.normalize_answer("The Caf\u00e9")
    decomposed = hotpot_utils.normalize_answer("cafe\u0301")

    assert composed == decomposed
    assert composed == "cafe\u0301"


@pytest.mark.parametrize(
    "prediction,gold,expected",
    [
        ("yes perhaps", "yes", 0.0),
        ("no", "yes", 0.0),
        ("The Paris!", "Paris", 1.0),
    ],
)
def test_hotpot_metric_preserves_normalized_em(prediction, gold, expected) -> None:
    """Use exact match even when an incorrect answer partially overlaps."""
    score, feedback = hotpot_utils.hotpotqa_metric(prediction, gold)
    assert score == expected
    assert "F1" not in feedback
