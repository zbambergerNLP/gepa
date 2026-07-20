import json
import os
import re
import string
from collections import Counter

import litellm


DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "hotpotqa_distractor_sample.jsonl",
)


def format_passages(passages: list[dict[str, str]], max_chars: int = 12000) -> str:
    """Format a list of {title, text} passage dicts into a single context string.

    Truncates the total context to max_chars to stay within model context limits.
    """
    parts = []
    total = 0
    for i, p in enumerate(passages, 1):
        part = f"[{i}] {p['title']}: {p['text']}"
        if total + len(part) > max_chars:
            break
        parts.append(part)
        total += len(part)
    return "\n\n".join(parts)


def normalize_answer(text: str) -> str:
    """HotpotQA official answer normalization (ported from hotpot_evaluate_v1.py)."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def f1_score(prediction: str, gold: str) -> float:
    """Token-overlap F1 between prediction and gold (official HotpotQA metric)."""
    pred = normalize_answer(prediction)
    truth = normalize_answer(gold)
    if not truth:
        return 0.0
    if pred in ("yes", "no", "noanswer") and pred != truth:
        return 0.0
    if truth in ("yes", "no", "noanswer") and pred != truth:
        return 0.0
    pred_tokens = pred.split()
    truth_tokens = truth.split()
    if not pred_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def run_llm(
    prompt: str,
    context: str,
    question: str,
    model: str = "hosted_vllm/Qwen3.5-9B",
    api_base: str | None = None,
) -> str:
    """Run the solver LM on a single HotpotQA example."""
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
    ]
    kwargs: dict = {"model": model, "messages": messages, "temperature": 1.0, "max_tokens": 512}
    if api_base is not None:
        kwargs["api_base"] = api_base
    response = litellm.completion(**kwargs)
    return response.choices[0].message.content.strip()


def hotpotqa_metric(prediction: str, gold: str) -> tuple[float, str]:
    """Compute F1 score and feedback for a single HotpotQA prediction."""
    score = f1_score(prediction, gold)

    if score >= 1.0:
        feedback = f"Correct! Your answer '{prediction}' matches the gold answer '{gold}'."
    elif score > 0.0:
        feedback = (
            f"Partially correct. Your answer '{prediction}' has token-overlap F1={score:.2f} "
            f"with the gold answer '{gold}'. Try to be more precise."
        )
    else:
        feedback = (
            f"Incorrect. Your answer '{prediction}' does not match the gold answer '{gold}'. "
            f"Re-read the context passages and chain reasoning across multiple passages."
        )

    return score, feedback


def load_hotpotqa_dataset(
    data_path: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Load the HotpotQA sample and split into train/val sets.

    Returns (trainset, valset) as lists of dicts with keys:
    question, answer, context, id, type.
    Split: 14 train / 6 val with stable ordering.
    """
    if data_path is None:
        data_path = DEFAULT_DATA_PATH

    data_path = os.path.normpath(data_path)

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"HotpotQA data not found at {data_path}. "
            f"Expected the bundled sample at examples/hotpotqa/data/hotpotqa_distractor_sample.jsonl"
        )

    records = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    examples = []
    for record in records:
        examples.append(
            {
                "question": record["question"],
                "answer": record["answer"],
                "context": format_passages(record["passages"]),
                "id": record["id"],
                "type": record["type"],
            }
        )

    trainset = examples[:14]
    valset = examples[14:]

    return trainset, valset
