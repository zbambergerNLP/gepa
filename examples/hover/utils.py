"""HoVer v1.1 data, Wikipedia-backed retrieval program, and document metric."""

import json
import os
import random
import re
import string
import urllib.request
from pathlib import Path
from typing import Sequence

import litellm

from examples.common.wikipedia import WikipediaPassage, WikipediaRetriever

DATA_DIR = Path(__file__).parent / "data"
HOVER_TRAIN_FILE = "hover_train_release_v1.1.json"
_DATA_BASE_URL = "https://raw.githubusercontent.com/hover-nlp/hover/main/data/hover"
FINAL_RESPONSE_MARKER = "Final Response:"
COT_FORMAT_INSTRUCTION = (
    "\n\nFirst reason step by step about retrieval. Then write the requested value after a line containing exactly "
    f"'{FINAL_RESPONSE_MARKER}'. Only the text after that line is used."
)


def ensure_data_downloaded(data_dir: str | os.PathLike[str] = DATA_DIR) -> Path:
    """Download the official HoVer v1.1 training release when it is missing."""
    destination_dir = Path(data_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / HOVER_TRAIN_FILE
    if destination.exists():
        return destination

    url = f"{_DATA_BASE_URL}/{HOVER_TRAIN_FILE}"
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, partial)
        partial.replace(destination)
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the official HoVer v1.1 data from {url}. "
            "Use --smoke only for an explicit local smoke run."
        ) from exc
    return destination


def _strip_think(text: str) -> str:
    """Remove hidden reasoning blocks emitted by Qwen-family models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_final_response(output: str) -> str:
    """Extract the value after the final response marker."""
    output = _strip_think(output)
    if FINAL_RESPONSE_MARKER in output:
        return output.rsplit(FINAL_RESPONSE_MARKER, 1)[1].strip()
    return output.strip()


def _call_lm(system: str, user: str, model: str, api_base: str | None) -> str:
    """Call the solver with the decoding settings used by the GEPA artifact."""
    kwargs: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": 16384,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    if api_base is not None:
        kwargs["api_base"] = api_base
    response = None
    for max_tokens in (16384, 4096, 1024, 256):
        kwargs["max_tokens"] = max_tokens
        try:
            response = litellm.completion(**kwargs)
            break
        except litellm.exceptions.ContextWindowExceededError:
            continue
    if response is None:
        return ""
    message = response.choices[0].message
    return message.content or getattr(message, "reasoning_content", None) or ""


def _supporting_titles(item: dict) -> list[str]:
    """Extract unique document titles from raw or HuggingFace HoVer facts."""
    facts = item.get("supporting_facts")
    titles: list[str] = []
    if isinstance(facts, dict):
        keys = facts.get("key") or facts.get("title") or []
        if isinstance(keys, str):
            titles.append(keys)
        elif isinstance(keys, list):
            titles.extend(str(key) for key in keys)
    elif isinstance(facts, list):
        for fact in facts:
            if isinstance(fact, list | tuple) and fact:
                titles.append(str(fact[0]))
            elif isinstance(fact, dict):
                title = fact.get("key") or fact.get("title")
                if title:
                    titles.append(str(title))

    unique: list[str] = []
    seen: set[str] = set()
    for title in titles:
        title = title.strip()
        normalized = _normalize_title(title)
        if title and normalized not in seen:
            seen.add(normalized)
            unique.append(title)
    return unique


def _to_example(item: dict) -> dict:
    """Normalize one official HoVer record for optimize_anything."""
    titles = _supporting_titles(item)
    claim = str(item.get("claim", "")).strip()
    return {
        "claim": claim,
        "prompt": claim,
        "question": claim,
        "supporting_facts": item.get("supporting_facts", []),
        "supporting_titles": titles,
        "gold_titles": titles,
        "label": str(item.get("label", "")),
        "id": str(item.get("uid", item.get("id", ""))),
        "num_hops": int(item.get("num_hops", 0) or 0),
        "answer": json.dumps(titles, ensure_ascii=False),
    }


def _smoke_records() -> list[dict]:
    """Return three explicit smoke records with three unique documents each."""
    return [
        {
            "uid": "smoke-train",
            "claim": "Marie Curie was born in the capital of Poland and won the Nobel Prize in Physics.",
            "supporting_facts": [["Marie Curie", 0], ["Warsaw", 0], ["Nobel Prize in Physics", 0]],
            "label": "SUPPORTED",
            "num_hops": 3,
        },
        {
            "uid": "smoke-val",
            "claim": "George Orwell wrote two novels and was born in British India.",
            "supporting_facts": [["George Orwell", 0], ["Animal Farm", 0], ["Nineteen Eighty-Four", 0]],
            "label": "SUPPORTED",
            "num_hops": 3,
        },
        {
            "uid": "smoke-test",
            "claim": "The Eiffel Tower stands in the capital of France beside the Seine.",
            "supporting_facts": [["Eiffel Tower", 0], ["Paris", 0], ["Seine", 0]],
            "label": "SUPPORTED",
            "num_hops": 3,
        },
    ]


def load_hover_dataset(
    seed: int = 0,
    data_dir: str | os.PathLike[str] = DATA_DIR,
    smoke: bool = False,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Load official HoVer v1.1 records with exactly three unique documents."""
    if smoke:
        examples = [_to_example(record) for record in _smoke_records()]
        return examples[:1], examples[1:2], examples[2:]

    path = ensure_data_downloaded(data_dir)
    with path.open(encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError(f"Expected a list in {path}")

    examples = [
        _to_example(record)
        for record in records
        if isinstance(record, dict) and record.get("claim") and len(_supporting_titles(record)) == 3
    ]
    random.Random(seed).shuffle(examples)
    if len(examples) < 750:
        raise ValueError(f"HoVer v1.1 yielded {len(examples)} three-document records; at least 750 are required")
    return examples[:150], examples[150:450], examples[450:750]


def _render_passages(passages: Sequence[WikipediaPassage], max_chars: int = 12000) -> str:
    """Render ranked passages within a bounded solver context."""
    rendered: list[str] = []
    total = 0
    for passage in passages:
        text = passage.render()
        if total + len(text) > max_chars:
            break
        rendered.append(text)
        total += len(text)
    return "\n\n".join(rendered)


def run_two_stage(
    summarize1_prompt: str,
    query2_prompt: str,
    summarize2_prompt: str,
    query3_prompt: str,
    claim: str,
    retriever: WikipediaRetriever,
    model: str = "hosted_vllm/Qwen3.5-9B",
    api_base: str | None = None,
    retrieval_k: int = 7,
    final_retrieval_k: int = 10,
) -> tuple[str, list[WikipediaPassage]]:
    """Run the artifact's three-hop HoVer retrieval program."""
    hop1 = retriever.search(claim, retrieval_k)
    summary_1 = _extract_final_response(
        _call_lm(
            summarize1_prompt + COT_FORMAT_INSTRUCTION,
            f"Claim:\n{claim}\n\nPassages:\n{_render_passages(hop1)}\n\nSummary:",
            model,
            api_base,
        )
    )

    query_2 = _extract_final_response(
        _call_lm(
            query2_prompt + COT_FORMAT_INSTRUCTION,
            f"Claim:\n{claim}\n\nFirst-hop summary:\n{summary_1}\n\nQuery:",
            model,
            api_base,
        )
    )
    hop2 = retriever.search(query_2, retrieval_k)
    summary_2 = _extract_final_response(
        _call_lm(
            summarize2_prompt + COT_FORMAT_INSTRUCTION,
            f"Claim:\n{claim}\n\nContext:\n{summary_1}\n\nPassages:\n{_render_passages(hop2)}\n\nSummary:",
            model,
            api_base,
        )
    )

    query_3 = _extract_final_response(
        _call_lm(
            query3_prompt + COT_FORMAT_INSTRUCTION,
            f"Claim:\n{claim}\n\nFirst-hop summary:\n{summary_1}\n\nSecond-hop summary:\n{summary_2}\n\nQuery:",
            model,
            api_base,
        )
    )
    hop3 = retriever.search(query_3, final_retrieval_k)
    return json.dumps([query_2, query_3], ensure_ascii=False), hop1 + hop2 + hop3


def run_single_stage(
    prompt: str,
    claim: str,
    retriever: WikipediaRetriever,
    model: str = "hosted_vllm/Qwen3.5-9B",
    api_base: str | None = None,
    retrieval_k: int = 10,
) -> list[WikipediaPassage]:
    """Generate one query and retrieve its ranked Wikipedia pages."""
    query = _extract_final_response(
        _call_lm(prompt + COT_FORMAT_INSTRUCTION, f"Claim:\n{claim}\n\nWikipedia query:", model, api_base)
    )
    return retriever.search(query, retrieval_k)


def _normalize_title(title: str) -> str:
    """Normalize a Wikipedia title for retrieval matching."""
    title = title.lower()
    title = "".join(character for character in title if character not in string.punctuation)
    title = re.sub(r"\b(a|an|the)\b", " ", title)
    return " ".join(title.split())


def _retrieved_titles(prediction: Sequence[WikipediaPassage | str]) -> list[str]:
    """Extract page titles from retrieved passage objects or rendered strings."""
    titles: list[str] = []
    for passage in prediction:
        if isinstance(passage, WikipediaPassage):
            titles.append(passage.title)
        else:
            titles.append(str(passage).split(" | ", 1)[0].strip())
    return titles


def hover_metric(prediction: Sequence[WikipediaPassage | str], example: dict) -> tuple[float, str]:
    """Score whether the retrieved pages contain all three gold documents."""
    gold_titles = [str(title) for title in example.get("gold_titles", _supporting_titles(example))]
    gold = {_normalize_title(title) for title in gold_titles}
    predicted_titles = _retrieved_titles(prediction)
    found = {_normalize_title(title) for title in predicted_titles}
    retrieved = gold & found
    missing = gold - found
    recall = len(retrieved) / len(gold) if gold else 0.0
    score = float(bool(gold) and not missing)
    feedback = (
        f"Retrieved {len(retrieved)}/{len(gold)} gold Wikipedia documents (recall={recall:.3f}). "
        f"Found: {sorted(retrieved)}. Missing: {sorted(missing)}. "
        f"Retrieved titles: {predicted_titles}."
    )
    return score, feedback


def hover_recall(prediction: Sequence[WikipediaPassage | str], example: dict) -> float:
    """Return gold-document recall for retrieved Wikipedia pages."""
    gold = {_normalize_title(str(title)) for title in example.get("gold_titles", _supporting_titles(example))}
    found = {_normalize_title(title) for title in _retrieved_titles(prediction)}
    return len(gold & found) / len(gold) if gold else 0.0
