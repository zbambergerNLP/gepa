"""HotpotQA data, Wikipedia-backed multi-hop program, and official metrics.

The production path mirrors the GEPA artifact: load ``hotpot_qa/fullwiki``,
retrieve twice from Wikipedia, and optimize the two summarizers, second-hop
query generator, and final answer component. Bundled dataset contexts are never
passed to the solver. The committed sample is available only when explicitly
selected for smoke runs.
"""

import json
import os
import random
import re
import string
from collections import Counter

import litellm

from examples.common.wikipedia import WikipediaPassage, WikipediaRetriever

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "hotpotqa_distractor_sample.jsonl",
)

FINAL_RESPONSE_MARKER = "Final Response:"

# Decoding config matches the paper's Qwen3-8B setup (gepa-artifact
# experiment_configs.py: temp=0.6, top_p=0.95, top_k=20; max_tokens=16384
# from run_experiments.py). Shared across ifbench/pupa/hotpotqa.
_HOTPOTQA_DECODING = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "max_tokens": 16384,
}


def _render_passages(passages: list[WikipediaPassage], max_chars: int = 12000) -> str:
    """Render a passage prefix under the configured text-length total.

    Args:
        passages: Ranked passages to render in order.
        max_chars: Maximum sum of rendered passage lengths. Separators added
            between passages are not counted.

    Returns:
        Double-newline-separated passage prefix selected before the first text
        that would exceed ``max_chars``.
    """
    rendered: list[str] = []
    total = 0
    for passage in passages:
        text = f"{passage.title} | {passage.text}"
        if total + len(text) > max_chars:
            break
        rendered.append(text)
        total += len(text)
    return "\n\n".join(rendered)


def normalize_answer(text: str) -> str:
    """Apply the official HotPotQA answer normalization.

    Args:
        text: Prediction or reference answer.

    Returns:
        Lowercase text without punctuation, articles, or repeated whitespace.
    """
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def f1_score(prediction: str, gold: str) -> float:
    """Compute the official token-overlap F1 score.

    Args:
        prediction: Model answer.
        gold: Reference answer.

    Returns:
        Token F1 in the inclusive range from zero to one.
    """
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


def _extract_final_response(output: str) -> str:
    """Extract the last marked final response after removing reasoning blocks.

    Args:
        output: Raw model output.

    Returns:
        Text after the final response marker, or all visible output when the
        marker is absent.
    """
    output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()
    if FINAL_RESPONSE_MARKER in output:
        return output.rsplit(FINAL_RESPONSE_MARKER, 1)[1].strip()
    return output.strip()


def _call_lm(system: str, user: str, model: str, api_base: str | None) -> str:
    """Call the solver with the HotPotQA experiment's decoding settings.

    Args:
        system: Candidate system prompt, omitted when empty.
        user: Example-specific user message.
        model: LiteLLM model identifier.
        api_base: Optional solver API endpoint.

    Returns:
        Raw message content when it contains non-whitespace text, otherwise raw
        reasoning content when available. Returns an empty string when every
        context-window retry fails or the response contains neither field.
    """
    messages = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": _HOTPOTQA_DECODING["temperature"],
        "top_p": _HOTPOTQA_DECODING["top_p"],
        "top_k": _HOTPOTQA_DECODING["top_k"],
        "max_tokens": _HOTPOTQA_DECODING["max_tokens"],
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    if api_base is not None:
        kwargs["api_base"] = api_base
    response = None
    for max_tokens in (kwargs["max_tokens"], 4096, 1024, 256):
        kwargs["max_tokens"] = max_tokens
        try:
            response = litellm.completion(**kwargs)
            break
        except litellm.exceptions.ContextWindowExceededError:
            continue
    if response is None:
        print(f"WARNING: input exceeds model context (prompt {len(system) + len(user)} chars); scoring 0.")
        return ""
    message = response.choices[0].message
    content = message.content or ""
    if not content.strip():
        content = getattr(message, "reasoning_content", None) or ""
    return content


def run_single_stage(
    prompt: str,
    question: str,
    retriever: WikipediaRetriever,
    model: str = "hosted_vllm/Qwen3.5-9B",
    api_base: str | None = None,
    retrieval_k: int = 7,
) -> str:
    """Retrieve once and answer with one optimized prompt.

    Args:
        prompt: Candidate answering instruction.
        question: HotPotQA question.
        retriever: Wikipedia passage retriever.
        model: Solver model identifier.
        api_base: Optional solver API endpoint.
        retrieval_k: Maximum passages requested.

    Returns:
        Extracted final answer.
    """
    passages = _render_passages(retriever.search(question, retrieval_k))
    user = f"Question:\n{question}\n\nRetrieved passages:\n{passages}\n\nAnswer:"
    out = _call_lm(prompt, user, model, api_base)
    return _extract_final_response(out)


def run_two_stage(
    summarize1_prompt: str,
    query_prompt: str,
    summarize2_prompt: str,
    answer_prompt: str,
    question: str,
    retriever: WikipediaRetriever,
    model: str = "hosted_vllm/Qwen3.5-9B",
    api_base: str | None = None,
    retrieval_k: int = 7,
) -> tuple[str, str]:
    """Run the GEPA artifact's two-retrieval-hop HotPotQA program.

    Args:
        summarize1_prompt: Instruction for summarizing first-hop evidence.
        query_prompt: Instruction for generating the second-hop query.
        summarize2_prompt: Instruction for summarizing second-hop evidence.
        answer_prompt: Instruction for producing the final answer.
        question: HotPotQA question.
        retriever: Wikipedia passage retriever.
        model: Solver model identifier.
        api_base: Optional solver API endpoint.
        retrieval_k: Maximum passages requested per hop.

    Returns:
        Bounded second-hop query and extracted final answer.
    """
    hop1_passages = _render_passages(retriever.search(question, retrieval_k))
    summary1_user = f"Question:\n{question}\n\nPassages:\n{hop1_passages}\n\nSummary:"
    summary_1 = _extract_final_response(_call_lm(summarize1_prompt, summary1_user, model, api_base))

    q_user = f"Question:\n{question}\n\nFirst-hop summary:\n{summary_1}\n\nSecond-hop query:"
    q_out = _call_lm(query_prompt, q_user, model, api_base)
    query = _extract_final_response(q_out)
    if len(query) > 2000:
        query = query[:2000] + " [truncated]"

    hop2_passages = _render_passages(retriever.search(query, retrieval_k))
    summary2_user = (
        f"Question:\n{question}\n\nFirst-hop summary:\n{summary_1}\n\nSecond-hop passages:\n{hop2_passages}\n\nSummary:"
    )
    summary_2 = _extract_final_response(_call_lm(summarize2_prompt, summary2_user, model, api_base))

    a_user = f"Question:\n{question}\n\nFirst-hop summary:\n{summary_1}\n\nSecond-hop summary:\n{summary_2}\n\nAnswer:"
    a_out = _call_lm(answer_prompt, a_user, model, api_base)
    answer = _extract_final_response(a_out)
    return query, answer


# Alias for compatibility with generic two-stage naming
run_hotpotqa_two_stage = run_two_stage


def hotpotqa_metric(prediction: str, gold: str) -> tuple[float, str]:
    """Compute exact match and explain token-level agreement for reflection.

    Args:
        prediction: Model answer.
        gold: Reference answer.

    Returns:
        Exact-match score and grounded feedback containing token F1.
    """
    f1 = f1_score(prediction, gold)
    em = float(normalize_answer(prediction) == normalize_answer(gold))

    if em >= 1.0:
        feedback = f"Exact match: prediction='{prediction}', gold='{gold}', token-F1=1.00, EM=1"
    elif f1 > 0.0:
        feedback = (
            f"Partial token overlap: prediction='{prediction}', gold='{gold}', "
            f"token-F1={f1:.2f}, EM={em:.0f}. Check the answer wording."
        )
    else:
        feedback = (
            f"No token overlap: prediction='{prediction}', gold='{gold}', "
            f"token-F1={f1:.2f}, EM={em:.0f}. Check the evidence across retrieved passages."
        )

    return em, feedback


def _load_from_jsonl(path: str) -> list[dict]:
    """Decode each non-empty line without applying schema validation.

    Args:
        path: JSONL source path.

    Returns:
        Decoded JSON values in file order; downstream conversion expects each
        value to be a mapping.
    """
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _jsonl_to_examples(records: list[dict]) -> list[dict]:
    """Convert explicit smoke records without exposing bundled passages.

    Args:
        records: Raw HotPotQA JSONL records.

    Returns:
        Solver examples containing question metadata but no dataset context.
    """
    examples = []
    for record in records:
        examples.append(
            {
                "question": record["question"],
                "answer": record["answer"],
                "id": record["id"],
                "type": record.get("type", ""),
                "level": record.get("level", ""),
                "supporting_titles": record.get("supporting_titles", []),
            }
        )
    return examples


def load_hotpotqa_dataset(
    data_path: str | None = None,
    train_limit: int | None = None,
    val_limit: int | None = None,
    test_limit: int | None = None,
    seed: int = 0,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Load HotpotQA fullwiki with deterministic 150/300/300 splits.

    Default (data_path is None): loads the GEPA artifact dataset via HuggingFace
    ``hotpot_qa/fullwiki``. Dataset contexts are intentionally discarded.
    Splits are deterministic (seed 0): shuffle train with `seed`, then
    train = train[:150], val = train[150:450], test = validation[:300]
    (paper Table 1: 150/300/300). This mirrors IFBench's deterministic slicing
    from `IFBench_train.jsonl`.

    If `data_path` is given, loads that JSONL file (smoke, 20 examples). For
    the 20-example sample, returns a 14/3/3 split (14 train / 3 val / 3 test)
    so the smoke still exercises the 3-way pipeline; the 14 train / 6 val
    legacy is preserved in the counts when val+test are combined.

    Args:
        data_path: Explicit JSONL smoke source. ``None`` selects fullwiki.
        train_limit: Optional prefix limit for the training split.
        val_limit: Optional prefix limit for the validation split.
        test_limit: Optional prefix limit for the test split.
        seed: Shuffle seed used for full-size data.

    Returns:
        Deterministic training, validation, and test examples with bundled
        contexts removed.

    Raises:
        FileNotFoundError: The requested explicit JSONL source does not exist.
        RuntimeError: The production fullwiki dataset cannot be loaded or is too
            small for the required splits.
    """
    # File-path mode (smoke / explicit)
    if data_path is not None:
        data_path = os.path.normpath(data_path)
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"HotpotQA data not found at {data_path}. "
                f"Expected the bundled sample at examples/hotpotqa/data/hotpotqa_distractor_sample.jsonl"
            )
        records = _load_from_jsonl(data_path)
        examples = _jsonl_to_examples(records)
        # Smoke is 20 examples: provide 14/3/3; larger files slice 150/300/300
        if len(examples) >= 750:
            rng = random.Random(seed)
            rng.shuffle(examples)
            trainset = examples[:150]
            valset = examples[150:450]
            testset = examples[450:750]
        elif len(examples) >= 20:
            # Preserve smoke counts: 14 train, rest split val/test
            trainset = examples[:14]
            remainder = examples[14:]
            mid = len(remainder) // 2
            valset = remainder[:mid] if mid else remainder[:3]
            testset = remainder[mid:] if mid else remainder[3:]
            # Pad to at least 3/3 if needed
            if not valset and remainder:
                valset = remainder[:3]
            if not testset and remainder:
                testset = remainder[3:6] if len(remainder) >= 6 else remainder[-3:]
            if not testset:
                testset = valset
        else:
            trainset = examples
            valset = examples
            testset = examples

        if train_limit is not None:
            trainset = trainset[:train_limit]
        if val_limit is not None:
            valset = valset[:val_limit]
        if test_limit is not None:
            testset = testset[:test_limit]
        return trainset, valset, testset

    # Production mode: use fullwiki exactly as the artifact does.
    try:
        from datasets import load_dataset

        ds = load_dataset("hotpot_qa", "fullwiki", trust_remote_code=True)
        train_raw = list(ds["train"])
        val_raw = list(ds["validation"]) if "validation" in ds else []

        rng = random.Random(seed)
        rng.shuffle(train_raw)

        # Paper Table 1: 150 train / 300 val / 300 test
        train_slice = train_raw[:150]
        val_slice = train_raw[150:450]
        if len(val_raw) >= 300:
            test_slice = val_raw[:300]
        else:
            test_slice = train_raw[450:750]
            if len(test_slice) < 300 and val_raw:
                # top up from validation
                need = 300 - len(test_slice)
                test_slice = test_slice + val_raw[:need]

        # Bundled contexts are deliberately discarded; the solver retrieves its
        # evidence from live Wikipedia instead of receiving dataset passages.
        converted_splits: list[list[dict]] = []
        for records in (train_slice, val_slice, test_slice):
            converted_splits.append(
                [
                    {
                        "question": record.get("question", ""),
                        "answer": record.get("answer", ""),
                        "id": record.get("id", ""),
                        "type": record.get("type", ""),
                        "level": record.get("level", ""),
                        "supporting_titles": record.get("supporting_facts", {}).get("title", [])
                        if isinstance(record.get("supporting_facts"), dict)
                        else [],
                    }
                    for record in records
                ]
            )
        trainset, valset, testset = converted_splits

        if len(trainset) < 150 or len(valset) < 300 or len(testset) < 300:
            raise ValueError("HotpotQA fullwiki did not contain enough records for 150/300/300 splits")

        if train_limit is not None:
            trainset = trainset[:train_limit]
        if val_limit is not None:
            valset = valset[:val_limit]
        if test_limit is not None:
            testset = testset[:test_limit]
        return trainset, valset, testset

    except Exception as exc:
        raise RuntimeError(
            "Could not load hotpot_qa/fullwiki. For an explicit smoke run, pass "
            f"--data-path {DEFAULT_DATA_PATH}. Original error: {exc}"
        ) from exc
