"""Tests for the Wikipedia-backed HotPotQA and HOVER runners."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import datasets
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from examples.common.wikipedia import WikipediaClient, WikipediaPassage
from examples.hotpotqa import utils as hotpot_utils
from examples.hover import utils as hover_utils

REPO_ROOT = Path(__file__).parents[1]


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
def test_wikipedia_lm_omits_an_empty_system_message(monkeypatch, module) -> None:
    """Keep empty system prompts out of solver message lists.

    Args:
        monkeypatch: Pytest fixture used to replace LiteLLM completion.
        module: Parameterized benchmark utility module.
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

    assert module._call_lm("", "question", "provider/model", None) == "answer"
    assert calls[0]["messages"] == [{"role": "user", "content": "question"}]


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


def test_hotpot_smoke_conversion_discards_bundled_context() -> None:
    """Exclude supplied contexts when converting explicit smoke records."""
    examples = hotpot_utils._jsonl_to_examples(
        [
            {
                "id": "example",
                "question": "Question?",
                "answer": "Answer",
                "context": {"title": ["Leaked"], "sentences": [["Do not expose"]]},
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
            "supporting_titles": [],
        }
    ]


def test_hotpot_production_loader_uses_fullwiki_and_discards_context(monkeypatch) -> None:
    """Load fullwiki with artifact splits while excluding dataset passages.

    Args:
        monkeypatch: Pytest fixture used to replace the Hugging Face loader.
    """
    calls = []
    records = [
        {
            "id": str(index),
            "question": f"Question {index}",
            "answer": f"Answer {index}",
            "context": {"title": ["Leaked"], "sentences": [["Do not expose"]]},
            "supporting_facts": {"title": ["Gold"], "sent_id": [0]},
        }
        for index in range(750)
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
            "train": records[:450],
            "validation": records[450:],
        }

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    train, val, test = hotpot_utils.load_hotpotqa_dataset()

    assert calls == [("hotpot_qa", "fullwiki", {"trust_remote_code": True})]
    assert (len(train), len(val), len(test)) == (150, 300, 300)
    assert all("context" not in example for example in train + val + test)


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
    monkeypatch.setattr(
        hotpot_utils,
        "_call_lm",
        Mock(side_effect=["summary one", "bridge query", "summary two", "Final Response: exact answer"]),
    )
    retriever = FakeRetriever(
        {
            "original question": [WikipediaPassage("First page", "first")],
            "bridge query": [WikipediaPassage("Second page", "second")],
        }
    )

    query, answer = hotpot_utils.run_two_stage(
        "summarize one",
        "query two",
        "summarize two",
        "answer",
        "original question",
        retriever,
        retrieval_k=7,
    )

    assert query == "bridge query"
    assert answer == "exact answer"
    assert retriever.calls == [("original question", 7), ("bridge query", 7)]


def test_hotpot_metric_uses_exact_match_as_primary_score() -> None:
    """Keep token overlap in feedback without promoting it above exact match."""
    score, feedback = hotpot_utils.hotpotqa_metric("Paris France", "Paris")

    assert score == 0.0
    assert "token-F1" in feedback
    assert "EM=0" in feedback


def test_hover_loader_parses_key_facts_and_filters_exactly_three_docs(tmp_path) -> None:
    """Deduplicate fact titles and retain only three-document HOVER records.

    Args:
        tmp_path: Pytest directory containing the synthetic official release.
    """
    records = []
    for index in range(750):
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

    train, val, test = hover_utils.load_hover_dataset(data_dir=tmp_path)

    assert (len(train), len(val), len(test)) == (150, 300, 300)
    assert all(example["gold_titles"] == ["Page A", "Page B", "Page C"] for example in train + val + test)
    assert all(example["id"] != "excluded" for example in train + val + test)


def test_hover_program_scores_pages_retrieved_across_three_hops(monkeypatch) -> None:
    """Accumulate pages from every HOVER retrieval hop before scoring.

    Args:
        monkeypatch: Pytest fixture used to provide deterministic LM outputs.
    """
    monkeypatch.setattr(
        hover_utils,
        "_call_lm",
        Mock(side_effect=["summary one", "query two", "summary two", "query three"]),
    )
    retriever = FakeRetriever(
        {
            "claim": [WikipediaPassage("Page A", "a")],
            "query two": [WikipediaPassage("Page B", "b")],
            "query three": [WikipediaPassage("Page C", "c")],
        }
    )

    queries, passages = hover_utils.run_two_stage(
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


def test_hover_smoke_mode_is_explicit_and_offline() -> None:
    """Provide three offline HOVER examples only when smoke mode is explicit."""
    train, val, test = hover_utils.load_hover_dataset(smoke=True)

    assert (len(train), len(val), len(test)) == (1, 1, 1)
    assert all(len(example["gold_titles"]) == 3 for example in train + val + test)


def test_hover_sbatch_defaults_are_compatible_with_react_v2() -> None:
    """Keep the cluster defaults on structured seeds and the primary condition."""
    script = (REPO_ROOT / "examples" / "hover" / "run_hover.sbatch").read_text()

    assert 'CONDITION="${CONDITION:-both}"' in script
    assert 'SEED_STYLE="${SEED_STYLE:-structured}"' in script


@pytest.mark.parametrize("benchmark", ["hotpotqa", "hover"])
def test_wikipedia_sbatch_separates_model_roles_and_preserves_auth(benchmark: str) -> None:
    """Keep solver and proposer endpoints independent in cluster launchers.

    Args:
        benchmark: Wikipedia benchmark whose batch script is inspected.
    """
    script = (REPO_ROOT / "examples" / benchmark / f"run_{benchmark}.sbatch").read_text()

    assert 'SAME_MODEL="${SAME_MODEL:-0}"' in script
    assert 'echo "ERROR: set REFLECTION_MODEL for the proposer, or opt into SAME_MODEL=1"' in script
    assert '--solver-api-base "${SOLVER_API_BASE}"' in script
    assert 'REFLECTION_API_ARG=(--reflection-api-base "${REFLECTION_API_BASE}")' in script
    assert 'REFLECTION_MODEL="hosted_vllm/${MODEL}"' not in script
    assert 'export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"' in script


def test_hotpotqa_della_submit_preserves_separate_model_roles() -> None:
    """Expose independent solver and reflection roles in the Della wrapper."""
    submit = (REPO_ROOT / "scripts" / "della" / "submit_hotpotqa.sh").read_text()
    assert "REFLECTION_MODEL" in submit
    assert "REFLECTION_API_BASE" in submit
    assert "SAME_MODEL" in submit
