"""GSM8K utilities: dataset loading, single-step CoT program, and metric.

Replicates a GSM8K (Cobbe et al. 2021, HF ``gsm8k`` / ``openai/gsm8k`` main
config) setup mirroring the AIME and IFBench examples: deterministic
shuffle seed 0 with splits 150/300/300 or 200/300/300 (paper's typical
7.5K train / 1K test noted, headroom for larger val), single-step
chain-of-thought LM program with a single optimized ``instruction`` prompt,
exact-match metric after numeric/string normalization with reflection-ready
feedback, and a defective-seed variant for VISTA-style recovery tests.

LM helpers (_call_lm, run_gsm8k_single_stage, numeric normalization,
boxed extraction, think stripping) are intentionally identical to
examples/aime_math/utils.py and examples/ifbench/utils.py so solver
behaviour is consistent across benchmarks (temp 0.6 / top_p 0.95 /
top_k 20 / max 16384 / enable_thinking False, truncation, context-window
retries, Final Answer marker).
"""

from __future__ import annotations

import json
import os
import random
import re

import litellm

# ---------------------------------------------------------------------------
# LM helpers (mirrors aime_math/utils.py and ifbench/utils.py exactly)
# ---------------------------------------------------------------------------

FINAL_RESPONSE_MARKER = "Final Answer:"

COT_FORMAT_INSTRUCTION = (
    "\n\nFirst reason step by step about how to solve the problem. "
    f"Then write your final numerical answer after a line containing exactly '{FINAL_RESPONSE_MARKER}'. "
    "Only the text after that line is used as your answer."
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models like Qwen3."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_boxed(text: str) -> str | None:
    """Extract content of the last \\boxed{...} handling nested braces."""
    # Find last occurrence of \boxed{
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return text[start : i - 1].strip()
    return None


def _extract_final_response(output: str) -> str:
    """Extract text after the last FINAL_RESPONSE_MARKER; fallback to full output."""
    output = _strip_think(output)
    if FINAL_RESPONSE_MARKER in output:
        return output.rsplit(FINAL_RESPONSE_MARKER, 1)[1].strip()
    return output.strip()


def _normalize_numeric(text: str) -> str:
    """Normalize a numeric answer string for comparison.

    - Strip whitespace, commas, dollar signs, leading +/- noise
    - Remove surrounding \\boxed{} wrapper if present
    - Collapse to a canonical numeric/string representation:
      prefer integer/float numeric equality; otherwise lowered stripped string
    """
    text = text.strip()
    # Unwrap \boxed{} if the whole answer is boxed
    boxed = _extract_boxed(text)
    # Only unwrap if boxed extraction covers most of the string or marker region
    # Heuristic: if boxed exists and raw text doesn't have much outside it,
    # prefer the boxed inner value.
    if boxed is not None:
        # If the stripped text is essentially just the boxed expression, use it.
        # Otherwise still prefer boxed as the intended answer.
        stripped_no_boxed = text.replace(f"\\boxed{{{boxed}}}", "").strip()
        if len(stripped_no_boxed) < len(text) * 0.5 or FINAL_RESPONSE_MARKER not in text:
            # When the model put the answer in a box, treat boxed as the answer.
            # Only override if final marker region also contains boxed.
            pass
        text = boxed.strip() if boxed.strip() else text

    # Remove common decoration
    text = text.replace(",", "").replace("$", "").strip()
    # Keep only the last numeric-looking token span if there's surrounding prose
    # e.g. "The answer is 42." -> "42"
    # We do the prose fallback downstream; here just strip.
    return text


def _numeric_equal(pred: str, gold: str) -> bool:
    """Check numeric equality after normalization.

    Tries integer, then float with tolerance, then exact normalized string match.
    Handles answers like "42", "42.0", "1,000", "$42", "\\boxed{42}".
    """
    p = _normalize_numeric(pred)
    g = _normalize_numeric(gold)

    # Direct case-insensitive string match (covers non-numeric edge cases)
    if p.lower() == g.lower():
        return True

    # Try to extract the last numeric token from each side when prose remains
    def _last_number(s: str) -> str | None:
        # Match integers, decimals, fractions like 3/4, and negatives
        nums = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", s)
        return nums[-1] if nums else None

    p_num = _last_number(p)
    g_num = _last_number(g)
    # If both have a numeric token, compare those; otherwise compare full normalized
    p_cmp = p_num if p_num is not None else p
    g_cmp = g_num if g_num is not None else g

    # Fraction handling: "3/4" -> 0.75
    def _to_float(s: str) -> float | None:
        s = s.strip()
        if "/" in s:
            parts = s.split("/")
            if len(parts) == 2:
                try:
                    return float(parts[0]) / float(parts[1])
                except (ValueError, ZeroDivisionError):
                    return None
            return None
        try:
            return float(s)
        except ValueError:
            return None

    pf = _to_float(p_cmp)
    gf = _to_float(g_cmp)
    if pf is not None and gf is not None:
        # Exact for integers, tolerance for floats
        if pf == gf:
            return True
        # Allow small floating tolerance for GSM8K decimals
        if abs(pf - gf) < 1e-6:
            return True
        # Also handle integer equivalence like 42 vs 42.0
        try:
            if int(pf) == int(gf) and abs(pf - gf) < 1e-9:
                return True
        except (ValueError, OverflowError):
            pass
        return False

    # Fallback string compare on normalized forms
    return p_cmp.strip().lower() == g_cmp.strip().lower()


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


def run_gsm8k_single_stage(
    system_prompt: str,
    problem: str,
    model: str = "hosted_vllm/Qwen3.5-9B",
    api_base: str | None = None,
) -> str:
    """Run single-step CoT GSM8K program: one system prompt, one LM call.

    Returns the raw LM output (including reasoning); callers extract the
    final answer via _extract_final_response / _numeric_equal.
    """
    capped = problem if len(problem) <= 24000 else problem[:24000] + "\n[truncated]"
    out = _call_lm(system_prompt + COT_FORMAT_INSTRUCTION, f"Problem:\n{capped}", model, api_base)
    return out


# Alias for consistency with aime_math naming
run_math_single_stage = run_gsm8k_single_stage


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def _gold_answer(example: dict) -> str:
    """Extract gold answer string from a GSM8K example.

    GSM8K ``answer`` fields contain reasoning + ``#### <number>`` suffix.
    We prefer ``answer_number`` if present, then parse ``####`` suffix,
    otherwise fall back to ``answer``/``target``.
    """
    if hasattr(example, "answer"):
        # dspy.Example fallback not expected for GSM8K but keep for parity
        raw = getattr(example, "answer")
        return str(raw).strip()
    # Prefer explicit numeric field if loader populated it
    if "answer_number" in example and str(example["answer_number"]).strip():
        return str(example["answer_number"]).strip()
    # Common keys
    for key in ("answer", "target", "label", "output"):
        if key in example and str(example[key]).strip():
            raw = str(example[key]).strip()
            # GSM8K format: "reasoning ... #### 42"
            if "####" in raw:
                return raw.rsplit("####", 1)[1].strip()
            return raw
    return ""


def gsm8k_metric(example: dict, prediction: str) -> tuple[float, str]:
    """Score a GSM8K response with exact match after normalization and feedback.

    Normalization handles: stripping, commas/dollars, \\boxed{}, last-number
    extraction, integer/float/fraction equality. Feedback is a reflection-ready
    string: "Your answer is correct/incorrect. The correct answer is '...'."
    When the gold contains a solution trace, it is appended for reflection.

    Args:
        example: dict with question/answer fields (see _gold_answer).
        prediction: raw LM output string (from run_gsm8k_single_stage).
    Returns:
        (score 0/1, feedback_text)
    """
    # Support legacy dspy.Example
    if hasattr(example, "answer") and not isinstance(example, dict):
        correct_raw = str(getattr(example, "answer", "")).strip()
        solution = str(getattr(example, "solution", "")).strip()
    else:
        correct_raw = _gold_answer(example)
        # GSM8K ``answer`` often is the full solution trace; keep it as solution
        solution = str(example.get("answer", "")).strip() if "####" in str(example.get("answer", "")) else str(example.get("solution", "")).strip()

    # Derive the correct numeric/string target (after #### if present)
    if "####" in correct_raw:
        correct_answer = correct_raw.rsplit("####", 1)[1].strip()
    else:
        correct_answer = correct_raw.strip()

    # Some loaders store the numeric separately
    if not correct_answer and "answer_number" in example:
        correct_answer = str(example["answer_number"]).strip()

    solution_suffix = (
        f" Here's the full step-by-step solution:\n{solution}\n\nThink about what takeaways you can learn from this solution to improve your future answers and approach to similar problems"
        if solution and solution != correct_answer
        else ""
    )

    final_text = _extract_final_response(prediction)

    # Try boxed extraction inside final_text first
    boxed = _extract_boxed(final_text)
    candidate_text = boxed if boxed is not None else final_text

    # Fallback: if no final marker region yielded a parseable answer, try full output boxed
    if not candidate_text.strip():
        boxed_full = _extract_boxed(prediction)
        if boxed_full is not None:
            candidate_text = boxed_full

    # Also fallback to raw prediction boxed/full when final_text parse seems empty
    if not candidate_text.strip():
        candidate_text = prediction

    is_correct = _numeric_equal(candidate_text, correct_answer)

    # Additional fallback: try the raw prediction's last number if final_text failed
    if not is_correct and candidate_text is not final_text:
        pass  # already tried boxed
    if not is_correct:
        # Try raw prediction directly against gold (handles models that ignore marker)
        if _numeric_equal(prediction, correct_answer):
            is_correct = True
            candidate_text = prediction

    if not candidate_text.strip():
        preview = (final_text or prediction)[:300]
        feedback_text = (
            f"Your answer is incorrect. You responded with '{preview}', "
            f"which couldn't be parsed as a valid answer. Please ensure your final answer after '{FINAL_RESPONSE_MARKER}' is a single number or expression. "
            f"The correct answer is '{correct_answer}'.{solution_suffix}"
        )
        return 0.0, feedback_text

    score = 1.0 if is_correct else 0.0
    status = "correct" if score == 1.0 else "incorrect"
    feedback_text = f"Your answer is {status}. The correct answer is '{correct_answer}'.{solution_suffix}"
    return score, feedback_text


# Backwards-compatible alias (mirrors aime_math's math_metric name)
math_metric = gsm8k_metric


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _parse_gsm8k_answer_number(answer: str) -> str:
    """Extract the numeric answer after #### from a GSM8K answer string."""
    if "####" in answer:
        return answer.rsplit("####", 1)[1].strip()
    return answer.strip()


def _load_gsm8k_hf(split: str = "train") -> list[dict]:
    """Load GSM8K from HuggingFace (``gsm8k`` or ``openai/gsm8k`` main config)."""
    from datasets import load_dataset

    # Try canonical names in order
    last_err: Exception | None = None
    for ds_name in ("gsm8k", "openai/gsm8k"):
        for config in ("main", "default", None):
            try:
                if config is None:
                    raw = load_dataset(ds_name, split=split)
                else:
                    raw = load_dataset(ds_name, config, split=split)
                out: list[dict] = []
                for item in raw:
                    question = str(item.get("question", "")).strip()
                    answer = str(item.get("answer", "")).strip()
                    answer_number = _parse_gsm8k_answer_number(answer)
                    out.append(
                        {
                            "prompt": question,
                            "problem": question,
                            "input": question,
                            "question": question,
                            "answer": answer,
                            "answer_number": answer_number,
                            "solution": answer,
                        }
                    )
                return out
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(f"Failed to load GSM8K from HF (gsm8k/openai/gsm8k): {last_err}") from last_err


def _load_gsm8k_local(path: str) -> list[dict]:
    """Load GSM8K from a local JSONL file (one JSON per line with question/answer)."""
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            question = str(j.get("question") or j.get("prompt") or j.get("problem") or j.get("input", "")).strip()
            answer = str(j.get("answer", "")).strip()
            answer_number = str(j.get("answer_number") or _parse_gsm8k_answer_number(answer)).strip()
            solution = str(j.get("solution") or answer).strip()
            records.append(
                {
                    "prompt": question,
                    "problem": question,
                    "input": question,
                    "question": question,
                    "answer": answer,
                    "answer_number": answer_number,
                    "solution": solution,
                }
            )
    return records


def load_gsm8k_dataset(
    seed: int = 0,
    data_path: str | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Load GSM8K with deterministic splits.

    Paper notes GSM8K's typical split is ~7.5K train / 1K test (Cobbe et al.
    2021, HF ``main`` config: train 7473, test 1319). For GEPA we use a
    lightweight headroom-friendly split mirroring aime_math/ifbench patterns:

    - Deterministic shuffle with ``seed`` (default 0)
    - Splits: **150 train / 300 val / 300 test** when the pool is large
      enough, or **200/300/300** if headroom requires (both documented;
      loader uses 150/300/300 to leave more held-out examples).
    - When the HF pool is < 750 examples the loader falls back to a
      proportional split so tests still run.

    Offline fallback: if HF download fails but a local artifact exists at
    ``data/gsm8k.jsonl`` (or ``data_path`` if provided), loads from there.
    When ``data_path`` points to a file it is used directly; when it points
    to a directory, ``gsm8k.jsonl`` inside it is used.

    Args:
        seed: shuffle seed (paper uses 0 for determinism).
        data_path: optional local path (file or directory) overriding HF.
    Returns:
        (trainset, valset, testset) as lists of dicts with keys
        prompt/problem/input/question, answer, answer_number, solution.
        Each item also carries a stable ``id`` for caching.
    """
    pool: list[dict] | None = None

    # Explicit data_path takes precedence
    if data_path is not None:
        candidate = data_path
        if os.path.isdir(candidate):
            candidate = os.path.join(candidate, "gsm8k.jsonl")
        if os.path.exists(candidate):
            pool = _load_gsm8k_local(candidate)
        else:
            raise FileNotFoundError(f"--data-path {data_path!r} not found (tried {candidate!r})")

    # Try HF if no explicit pool yet
    if pool is None:
        try:
            train_pool = _load_gsm8k_hf("train")
            # Also load test split and merge so shuffle covers the full
            # distribution (mirrors aime_math's merge of 2022-24 + 2025 idea
            # but GSM8K's canonical split is single-source).
            try:
                test_pool = _load_gsm8k_hf("test")
                pool = train_pool + test_pool
            except Exception:
                pool = train_pool
        except Exception as e:
            print(f"[gsm8k] HF load failed ({e}); trying local artifact...")
            pool = None

    # Local artifact fallback
    if pool is None:
        local = os.path.join(DATA_DIR, "gsm8k.jsonl")
        if os.path.exists(local):
            pool = _load_gsm8k_local(local)
        else:
            raise RuntimeError(
                "Unable to load GSM8K: no HF access and no local artifact at data/gsm8k.jsonl "
                "(or --data-path). Place a JSONL with {question, answer} at examples/gsm8k/data/gsm8k.jsonl "
                "or pass --data-path."
            ) from None

    assert pool is not None

    rng = random.Random(seed)
    rng.shuffle(pool)

    # Deterministic splits: 150 train / 300 val / 300 test (or 200/300/300).
    # Use 150/300/300 to preserve more held-out data; caller can slice via
    # --train-limit etc. For very small pools, fall back to proportional.
    n = len(pool)
    if n >= 750:
        # 150 train / 300 val / 300 test = 750; remaining held out (not returned)
        # This matches the "150 train / 300 val / 300 test (or 200/300/300 if
        # needed for headroom)" spec: 150 is the default, 200 noted as alt.
        trainset = pool[:150]
        valset = pool[150:450]
        testset = pool[450:750]
    elif n >= 600:
        # Minimum for 150/300 fallback proportionally
        trainset = pool[:150]
        valset = pool[150:350] if n >= 350 else pool[150:]
        testset = pool[350:650] if n >= 650 else pool[350:] if len(pool) > 350 else pool[: min(300, n)]
        # Ensure val/test at least 100 each when possible
        if len(valset) < 100 and n >= 300:
            valset = pool[150 : 150 + min(300, n - 150)]
        if len(testset) < 100 and n >= 300:
            testset = pool[-min(300, n) :]
    else:
        # Small pool: third split
        third = n // 3
        trainset = pool[:third]
        valset = pool[third : 2 * third]
        testset = pool[2 * third :]

    # Attach stable ids for caching
    for i, ex in enumerate(trainset):
        ex.setdefault("id", f"gsm8k_train_{i}")
    for i, ex in enumerate(valset):
        ex.setdefault("id", f"gsm8k_val_{i}")
    for i, ex in enumerate(testset):
        ex.setdefault("id", f"gsm8k_test_{i}")

    return trainset, valset, testset


# Alias mirroring aime_math's load_math_dataset name for generic callers
load_math_dataset = load_gsm8k_dataset


# ---------------------------------------------------------------------------
# Defective seed support (VISTA recovery test)
# ---------------------------------------------------------------------------

DEFECTIVE_SEED_CANDIDATE = {
    "instruction": (
        "You are a knowledgeable assistant. Answer concisely and always wrap "
        "your final answer in \\boxed{}. Do not show your work."
    ),
}

# Alternative defective variant: actively misleading instruction
DEFECTIVE_SEED_CANDIDATE_ALT = {
    "instruction": (
        "Solve the problem. Reply with ONLY the number 0 regardless of the question."
    ),
}


def get_defective_seed(variant: str = "default") -> dict:
    """Return a defective seed candidate for VISTA-style recovery tests.

    The default variant is a weak instruction that suppresses chain-of-thought
    (mirroring VISTA's defective-seed analysis on GSM8K where GEPA degraded
    sharply). An ``alt`` variant returns an actively misleading seed.

    Args:
        variant: "default" or "alt".
    """
    if variant == "alt":
        return dict(DEFECTIVE_SEED_CANDIDATE_ALT)
    return dict(DEFECTIVE_SEED_CANDIDATE)


# ---------------------------------------------------------------------------
# Legacy helper for simple baseline scoring (parity with aime_math)
# ---------------------------------------------------------------------------

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
        prompt_str = prompt.get("instruction") or prompt.get("system_prompt") or next(iter(prompt.values()))
    else:
        prompt_str = prompt

    def score_one(example: dict) -> float:
        out = run_gsm8k_single_stage(
            prompt_str, example["prompt"] if "prompt" in example else example.get("problem", ""), model=model, api_base=api_base
        )
        score, _ = gsm8k_metric(example, out)
        return score

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        scores = list(pool.map(score_one, dataset))
    return sum(scores) / len(scores) if scores else 0.0
