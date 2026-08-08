"""AIME Math utilities: dataset loading, single-step CoT program, and metric.

Replicates the AIME 2022-2024 -> 2025 setup (AI-MO/aimo-validation-aime
train split shuffled seed 0 -> 45/45 train/val, MathArena/aime_2025 test
30 problems expanded 5x -> 150 evaluation items) with a single-step
chain-of-thought LM program. Metric is exact integer-match accuracy
with per-example feedback. See README.

The LM helpers (_call_lm, run_math_single_stage) mirror
examples/ifbench/utils.py and examples/pupa/utils.py so solver
behaviour is identical across benchmarks (temp 0.6 / top_p 0.95 /
top_k 20 / max 16384 / enable_thinking False).
"""

from __future__ import annotations

import os
import random
import re

import litellm

# ---------------------------------------------------------------------------
# LM helpers (mirrors ifbench/utils.py)
# ---------------------------------------------------------------------------

FINAL_RESPONSE_MARKER = "Final Answer:"

COT_FORMAT_INSTRUCTION = (
    "\n\nFirst reason step by step about how to solve the problem. "
    f"Then write your final numerical answer after a line containing exactly '{FINAL_RESPONSE_MARKER}'. "
    "Only the text after that line is used as your answer and it must be a single integer."
)


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models like Qwen3."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_final_response(output: str) -> str:
    """Extract text after the last FINAL_RESPONSE_MARKER; fallback to full output."""
    output = _strip_think(output)
    if FINAL_RESPONSE_MARKER in output:
        return output.rsplit(FINAL_RESPONSE_MARKER, 1)[1].strip()
    return output.strip()


def _extract_integer(text: str) -> int | None:
    """Try to parse a single integer from text; returns None on failure."""
    text = text.strip()
    # If the model obeyed the marker, the extracted chunk should be a bare integer.
    # Otherwise search for the last integer in the output (common fallback).
    # Remove commas and whitespace.
    cleaned = text.replace(",", "").strip()
    # Direct int parse
    try:
        # allow surrounding text like "123" possibly with period
        m = re.search(r"-?\d+", cleaned)
        if m:
            # Prefer the last integer (final answer) when multiple present
            last = re.findall(r"-?\d+", cleaned)
            if last:
                return int(last[-1])
        return int(cleaned)
    except (ValueError, TypeError):
        return None
    # unreachable, fallback search
    # m = re.search(r"-?\d+", text)
    # return int(m.group(0)) if m else None


def _call_lm(system: str, user: str, model: str, api_base: str | None) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": 16384,
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
        print(f"WARNING: input alone exceeds model context (prompt {len(system) + len(user)} chars); scoring 0.")
        return ""
    message = response.choices[0].message
    content = message.content or ""
    if not content.strip():
        content = getattr(message, "reasoning_content", None) or ""
    return content


def run_math_single_stage(
    system_prompt: str,
    problem: str,
    model: str = "hosted_vllm/Qwen3.5-9B",
    api_base: str | None = None,
) -> str:
    """Run single-step CoT math program: one system prompt, one LM call.

    Returns the raw LM output (including reasoning); callers extract the
    final answer via _extract_final_response / _extract_integer.
    """
    # Cap problem length so prompt + problem + output budget fits context (32k)
    capped = problem if len(problem) <= 24000 else problem[:24000] + "\n[truncated]"
    out = _call_lm(system_prompt + COT_FORMAT_INSTRUCTION, f"Problem:\n{capped}", model, api_base)
    return out


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def math_metric(example: dict, prediction: str) -> tuple[float, str]:
    """Score a math response with exact integer accuracy and feedback.

    Args:
        example: dict with keys answer (str/int), solution (optional), problem/prompt/input.
        prediction: raw LM output string (from run_math_single_stage).
    Returns:
        (score 0/1, feedback_text)
    """
    # Handle both dict and legacy dspy.Example
    if hasattr(example, "answer"):
        # dspy.Example fallback
        correct_raw = example.answer
        solution = getattr(example, "solution", "")
    else:
        correct_raw = example.get("answer", "")
        solution = example.get("solution", "")

    try:
        correct_answer = int(str(correct_raw).strip().replace(",", ""))
    except (ValueError, TypeError):
        # If gold isn't an int (should not happen for AIME), compare normalized strings
        correct_answer = str(correct_raw).strip()

    solution_suffix = (
        f" Here's the full step-by-step solution:\n{solution}\n\nThink about what takeaways you can learn from this solution to improve your future answers and approach to similar problems"
        if solution
        else ""
    )

    final_text = _extract_final_response(prediction)
    llm_answer = _extract_integer(final_text)
    # Fallback: try raw prediction if final_text parse failed but full output contains int
    if llm_answer is None:
        llm_answer = _extract_integer(prediction)

    if llm_answer is None:
        preview = (final_text or prediction)[:300]
        feedback_text = (
            f"The final answer must be a valid integer and nothing else. You responded with '{preview}', "
            f"which couldn't be parsed as a python integer. Please ensure your answer is a valid integer without any additional text or formatting. "
            f"The correct answer is '{correct_answer}'.{solution_suffix}"
            + (" and ensure your final answer is a valid integer." if solution else "")
        )
        return 0.0, feedback_text

    if isinstance(correct_answer, int):
        score = float(correct_answer == llm_answer)
    else:
        score = float(str(correct_answer).strip() == str(llm_answer).strip())
    status = "correct" if score == 1.0 else "incorrect"
    feedback_text = f"Your answer is {status}. The correct answer is '{correct_answer}'.{solution_suffix}"
    return score, feedback_text


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _load_aimo() -> list[dict]:
    """Load AI-MO/aimo-validation-aime train split as list of dicts."""
    from datasets import load_dataset

    raw = load_dataset("AI-MO/aimo-validation-aime", "default", split="train")
    out: list[dict] = []
    for item in raw:
        question = str(item["problem"])
        solution = str(item.get("solution", ""))
        answer = str(item["answer"])
        out.append(
            {
                "prompt": question,
                "problem": question,
                "input": question,
                "answer": answer,
                "solution": solution,
                # normalized aliases
                "question": question,
            }
        )
    return out


def _load_aime2025() -> list[dict]:
    """Load MathArena/aime_2025 train split as list of dicts (AIME 2025 has 30 problems)."""
    from datasets import load_dataset

    raw = load_dataset("MathArena/aime_2025", "default", split="train")
    out: list[dict] = []
    for item in raw:
        question = str(item["problem"])
        answer = str(item["answer"])
        out.append(
            {
                "prompt": question,
                "problem": question,
                "input": question,
                "answer": answer,
                "solution": "",
                "question": question,
            }
        )
    return out


def load_math_dataset(
    seed: int = 0,
    test_repeats: int = 5,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Load AIME with paper-faithful splits.

    Splits (seed 0):
        AI-MO/aimo-validation-aime covers AIME 2022-2024 (90 problems).
        Shuffled with seed 0, then split 45 / 45 -> train / val.
        MathArena/aime_2025 covers AIME 2025 (30 problems) -> test,
        expanded test_repeats times (default 5 -> 150 items) to reduce
        stochastic decoding variance; each repeat is a distinct evaluation
        item with a repeat id but identical problem/answer.

    Returns (trainset, valset, testset) as lists of dicts with keys
    prompt/problem/input, answer, solution. Each item also carries a
    stable id for caching (problem text hash prefix).

    Args:
        seed: shuffle seed for train/val split (paper uses 0).
        test_repeats: how many times to repeat the 30-problem test set.
            Use 1 to get 30 unique problems; 5 (default) matches the
            45/45/30x5 protocol described in the README.

    Offline fallback: if HF download fails but a local artifact exists at
    data/aime_2022_2024.jsonl and data/aime_2025.jsonl, loads from there.
    """
    # Try HF first; fall back to local jsonl artifact if offline
    train_pool: list[dict] | None = None
    test_pool: list[dict] | None = None

    try:
        train_pool = _load_aimo()
    except Exception as e:
        print(f"[aime] HF load AI-MO failed ({e}); trying local artifact...")
        train_pool = None

    try:
        test_pool = _load_aime2025()
    except Exception as e:
        print(f"[aime] HF load MathArena/aime_2025 failed ({e}); trying local artifact...")
        test_pool = None

    # Local artifact fallback
    if train_pool is None:
        import json as _json

        local = os.path.join(DATA_DIR, "aime_2022_2024.jsonl")
        if os.path.exists(local):
            train_pool = []
            with open(local) as f:
                for line in f:
                    if line.strip():
                        j = _json.loads(line)
                        train_pool.append(
                            {
                                "prompt": str(j.get("problem") or j.get("prompt") or j.get("input", "")),
                                "problem": str(j.get("problem") or j.get("prompt") or j.get("input", "")),
                                "input": str(j.get("problem") or j.get("prompt") or j.get("input", "")),
                                "answer": str(j.get("answer", "")),
                                "solution": str(j.get("solution", "")),
                                "question": str(j.get("problem") or j.get("prompt") or j.get("input", "")),
                            }
                        )
        else:
            raise RuntimeError("Unable to load AI-MO data: no HF access and no local artifact at data/aime_2022_2024.jsonl") from None

    if test_pool is None:
        import json as _json

        local = os.path.join(DATA_DIR, "aime_2025.jsonl")
        if os.path.exists(local):
            test_pool = []
            with open(local) as f:
                for line in f:
                    if line.strip():
                        j = _json.loads(line)
                        test_pool.append(
                            {
                                "prompt": str(j.get("problem") or j.get("prompt") or j.get("input", "")),
                                "problem": str(j.get("problem") or j.get("prompt") or j.get("input", "")),
                                "input": str(j.get("problem") or j.get("prompt") or j.get("input", "")),
                                "answer": str(j.get("answer", "")),
                                "solution": "",
                                "question": str(j.get("problem") or j.get("prompt") or j.get("input", "")),
                            }
                        )
        else:
            raise RuntimeError("Unable to load aime_2025 data: no HF access and no local artifact at data/aime_2025.jsonl") from None

    assert train_pool is not None and test_pool is not None

    # Shuffle train pool with seed and split 45/45 (paper protocol)
    rng = random.Random(seed)
    rng.shuffle(train_pool)
    # Paper: AI-MO has 90 (30 per year 2022-2024). Keep 45/45 exactly when available;
    # otherwise split in half like the legacy loader.
    if len(train_pool) >= 90:
        trainset = train_pool[:45]
        valset = train_pool[45:90]
    else:
        mid = len(train_pool) // 2
        trainset = train_pool[:mid]
        valset = train_pool[mid:]

    # Test expansion 30 x test_repeats (default 150)
    testset: list[dict] = []
    for rep in range(test_repeats):
        for idx, ex in enumerate(test_pool):
            # shallow copy with repeat metadata so GEPA's cache distinguishes repeats
            # but the underlying problem/answer stay identical
            copy = dict(ex)
            copy["repeat_id"] = rep
            copy["aime_year"] = "2025"
            copy["aime_idx"] = idx
            copy["id"] = f"aime2025_{idx}_rep{rep}"
            testset.append(copy)

    # Attach ids to train/val as well for caching
    for i, ex in enumerate(trainset):
        ex.setdefault("id", f"aime_train_{i}")
    for i, ex in enumerate(valset):
        ex.setdefault("id", f"aime_val_{i}")

    return trainset, valset, testset


# Legacy helper for simple baseline scoring (kept for backwards compat)
def evaluate_on_dataset(
    prompt: str | dict,
    dataset: list[dict],
    model: str = "hosted_vllm/Qwen3.5-9B",
    api_base: str | None = None,
    max_workers: int = 16,
) -> float:
    """Evaluate a prompt (str or {instruction: str}) on a dataset, returning mean accuracy."""
    from concurrent.futures import ThreadPoolExecutor

    if isinstance(prompt, dict):
        # support both {"instruction": "..."} and legacy string
        prompt_str = prompt.get("instruction") or prompt.get("system_prompt") or next(iter(prompt.values()))
    else:
        prompt_str = prompt

    def score_one(example: dict) -> float:
        out = run_math_single_stage(prompt_str, example["prompt"] if "prompt" in example else example.get("problem", ""), model=model, api_base=api_base)
        score, _ = math_metric(example, out)
        return score

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        scores = list(pool.map(score_one, dataset))
    return sum(scores) / len(scores) if scores else 0.0
