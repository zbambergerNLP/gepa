"""LiveBench-Math utilities: dataset loading, single-step CoT program, and metric.

LiveBench-Math (White et al. 2025, https://livebench.ai, arXiv:2408.14596)
collects n=368 fresh competition math problems (AMC, AIME, symbolic
algebra, olympiad) graded by LiveBench scorers. The benchmark is
contamination-limited: problems post-date most model cutoffs and are
released with a timed lock. This module mirrors the Terrarium split
used in the GEPA parallel-proposals release (100/100/168) but adopts
the paper-faithful 122/123/123 seed-0 shuffle for new action-conditioned
experiments (368 / 3 ≈ 122.7). See ATTRIBUTION.md.

The LM helpers (_call_lm, run_livebench_single_stage) and decoding
config mirror examples/ifbench/utils.py and examples/pupa/utils.py so
solver behaviour is identical across benchmarks (temp 0.6 / top_p 0.95 /
top_k 20 / max 16384 / enable_thinking False). Metric is exact-match
accuracy after answer normalization with per-example feedback.
"""

from __future__ import annotations

import json
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
    f"Then write your final answer after a line containing exactly '{FINAL_RESPONSE_MARKER}'. "
    "Only the text after that line is used as your answer."
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
# Optional local artifact: examples/livebench_math/data/livebench_math.jsonl
LOCAL_ARTIFACT = os.path.join(DATA_DIR, "livebench_math.jsonl")
# Also support terrarium artifact location if present
TERRARIUM_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "livebench_math.jsonl"),
]


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_final_response(output: str) -> str:
    output = _strip_think(output)
    if FINAL_RESPONSE_MARKER in output:
        return output.rsplit(FINAL_RESPONSE_MARKER, 1)[1].strip()
    return output.strip()


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


def run_livebench_single_stage(
    system_prompt: str,
    problem: str,
    model: str = "hosted_vllm/Qwen3.5-9B",
    api_base: str | None = None,
) -> str:
    """Run single-step CoT LiveBench-Math program: one system prompt, one LM call.

    Returns the raw LM output (including reasoning).
    """
    capped = problem if len(problem) <= 24000 else problem[:24000] + "\n[truncated]"
    out = _call_lm(system_prompt + COT_FORMAT_INSTRUCTION, f"Problem:\n{capped}", model, api_base)
    return out


# ---------------------------------------------------------------------------
# Answer normalization (LiveBench math answers vary: integers, boxed, latex)
# ---------------------------------------------------------------------------

def _normalize_answer(text: str) -> str:
    """Normalize a math answer for comparison.

    - strips <think> blocks, whitespace, trailing period
    - extracts \\boxed{...} if present (common in math datasets)
    - lowercases, removes commas, collapses whitespace
    - removes surrounding $ and LaTeX wrappers
    """
    text = _strip_think(text)
    # If boxed, take inside
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        text = m.group(1)
    # Also handle \fbox, $...$
    text = text.strip()
    # Remove leading "Answer:" etc
    text = re.sub(r"^(answer|final answer)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)
    # Collapse whitespace, remove commas
    text = text.replace(",", "").strip()
    text = re.sub(r"\s+", " ", text)
    # Strip surrounding dollars
    if text.startswith("$") and text.endswith("$") and len(text) > 1:
        text = text[1:-1].strip()
    # Remove trailing period
    if text.endswith(".") and len(text) > 1:
        text = text.rstrip(".").strip()
    return text.lower()


def _answers_match(pred: str, gold: str) -> bool:
    """Check if normalized prediction matches gold, with numeric tolerance."""
    p = _normalize_answer(pred)
    g = _normalize_answer(gold)
    if p == g:
        return True
    # Numeric tolerance: if both parse as floats, allow small epsilon
    try:
        pf = float(p.replace(" ", ""))
        gf = float(g.replace(" ", ""))
        # exact for integers, tolerance for floats
        if abs(pf - gf) < 1e-6:
            return True
        # also handle fractions like "1/2" vs "0.5"
        # try eval fraction
        if "/" in g:
            import fractions

            try:
                gf_frac = float(fractions.Fraction(g))
                if abs(pf - gf_frac) < 1e-6:
                    return True
            except Exception:
                pass
        if "/" in p:
            import fractions

            try:
                pf_frac = float(fractions.Fraction(p))
                if abs(pf_frac - gf) < 1e-6:
                    return True
            except Exception:
                pass
    except (ValueError, TypeError):
        pass
    return False


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def livebench_metric(response: str, example: dict) -> tuple[float, str]:
    """Score a LiveBench-Math response.

    Accuracy is 1 if the text after FINAL_RESPONSE_MARKER (or the full
    response as fallback) normalizes to the gold answer, else 0. Feedback
    lists correctness and the gold answer for reflection.

    Returns (score 0/1, feedback_text).
    """
    gold = str(example.get("answer") or example.get("ground_truth") or example.get("target") or "").strip()
    raw_pred = _extract_final_response(response)
    # Also fallback to full response if marker missing but answer present
    candidate_text = raw_pred if raw_pred.strip() else response
    # For feedback, keep a preview
    preview = candidate_text[:400].strip().replace("\n", " ")

    is_correct = _answers_match(candidate_text, gold) if gold else False
    score = 1.0 if is_correct else 0.0
    status = "correct" if is_correct else "incorrect"
    # Provide gold for reflection (like AIME solution suffix)
    feedback = f"Your answer is {status}. The correct answer is '{gold}'. You answered '{preview}'."
    if score == 0 and not candidate_text.strip():
        feedback += " Your response was empty or could not be parsed; ensure you write the final answer after 'Final Answer:'."
    return score, feedback


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _try_load_hf_math() -> list[dict] | None:
    """Try HuggingFace LiveBench sources; return list of dicts or None on failure."""
    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"[livebench] datasets import failed: {e}")
        return None

    # Candidate HF specs to try (ordered). LiveBench distribution has shifted;
    # we try the most common current locations first.
    candidates = [
        # (dataset_name, config, split, requires_math_filter)
        ("livebench/livebench", None, "test", True),
        ("livebench/livebench", "math", "test", False),
        ("livebench/livebench_math", None, "test", False),
        ("livebench/math", None, "test", False),
        ("gepa-ai/livebench-math", None, "test", False),
    ]

    for ds_name, config, split, need_filter in candidates:
        try:
            print(f"[livebench] trying HF {ds_name} config={config} split={split} ...")
            ds = load_dataset(ds_name, config, split=split) if config else load_dataset(ds_name, split=split)
        except Exception as e:
            print(f"[livebench] {ds_name} load failed: {e}")
            continue

        # Convert to uniform dicts
        filtered: list[dict] = []
        all_items: list[dict] = []
        for item in ds:
            # LiveBench items vary: look for question/prompt/problem and answer/ground_truth
            question = (
                item.get("question")
                or item.get("prompt")
                or item.get("problem")
                or item.get("input")
                or item.get("turns", [{}])[0].get("content") if isinstance(item.get("turns"), list) else None
                or ""
            )
            # Some LiveBench formats store turns as list of dicts with role/content
            if not question and isinstance(item.get("turns"), list):
                # concatenate user turns
                parts = [t.get("content", "") for t in item["turns"] if t.get("role") == "user"]
                question = "\n".join(parts)
            question = str(question).strip()
            # Answer fields
            answer = (
                item.get("answer")
                or item.get("ground_truth")
                or item.get("ground_truth_answer")
                or item.get("correct_answer")
                or item.get("target")
                or item.get("ideal_response")
                or ""
            )
            # Some items store answer in second turn
            if not answer and isinstance(item.get("turns"), list) and len(item["turns"]) > 1:
                answer = str(item["turns"][-1].get("content", "")).strip()
            answer = str(answer).strip()
            if not question or not answer:
                continue
            entry = {
                "prompt": question,
                "problem": question,
                "input": question,
                "question": question,
                "answer": answer,
                "ground_truth": answer,
                "raw": item,
            }
            # Preserve category for filtering
            cat = str(item.get("category") or item.get("task") or item.get("subset") or item.get("livebench_category") or "").lower()
            # Keep math when filter needed, else keep all
            all_items.append((entry, cat))
            if need_filter:
                if "math" in cat:
                    filtered.append(entry)
                elif not cat:
                    # No category field -> assume math dataset, keep
                    filtered.append(entry)
            else:
                filtered.append(entry)

        # Decide which list to keep
        if need_filter and filtered:
            # If we filtered and got ~368, that's success
            if len(filtered) >= 300:
                print(f"[livebench] got {len(filtered)} math items from {ds_name} (filtered)")
                return filtered
            # If filtered is small but all_items larger, maybe category field is missing; fall back to all
            if len(all_items) >= 300 and len(filtered) < 50:
                print(f"[livebench] filtered {len(filtered)} but all {len(all_items)}; using all")
                return [e for e, _ in all_items]
            if filtered:
                return filtered
        elif filtered:
            print(f"[livebench] got {len(filtered)} items from {ds_name}")
            if len(filtered) >= 300:
                return filtered
            return filtered

    return None


def _load_local_artifact() -> list[dict] | None:
    """Load local jsonl artifact if present; supports multiple locations."""
    paths = [LOCAL_ARTIFACT] + TERRARIUM_CANDIDATES
    # Also respect env override
    env_path = os.environ.get("LIVEBENCH_DATA") or os.environ.get("LIVEBENCH_MATH_DATA")
    if env_path:
        paths.insert(0, env_path)

    for path in paths:
        if not path or not os.path.exists(path):
            continue
        print(f"[livebench] loading local artifact {path} ...")
        items: list[dict] = []
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    j = json.loads(line)
                    q = str(j.get("question") or j.get("prompt") or j.get("problem") or j.get("input") or "")
                    a = str(j.get("answer") or j.get("ground_truth") or j.get("target") or "")
                    if not q or not a:
                        continue
                    items.append(
                        {
                            "prompt": q,
                            "problem": q,
                            "input": q,
                            "question": q,
                            "answer": a,
                            "ground_truth": a,
                            "raw": j,
                        }
                    )
            if items:
                print(f"[livebench] loaded {len(items)} from {path}")
                return items
        except Exception as e:
            print(f"[livebench] failed to load {path}: {e}")
            continue
    return None


def load_livebench_math_dataset(
    seed: int = 0,
    splits: tuple[int, int, int] | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Load LiveBench-Math (White et al. 2025, n=368) with paper-faithful splits.

    Pipeline:
        1. Try HuggingFace (livebench/livebench filtered to math) – contamination-
           limited fresh problems.
        2. Fall back to local artifact at data/livebench_math.jsonl or env
           LIVEBENCH_DATA.
        3. Shuffle with seed 0 and split ~122/123/123 (368 / 3).

    For the GEPA parallel-proposals release the Terrarium split was
    100/100/168; this function defaults to 122/123/123 for new
    action-conditioned experiments. Pass splits=(100,100,168) to
    reproduce Terrarium.

    Returns (trainset, valset, testset) as lists of dicts with keys
    prompt/problem/input, answer/ground_truth.

    When HF is offline and no artifact exists, synthesizes a deterministic
    placeholder of 368 items with dummy math problems so that py_compile
    and shape checks pass (real evaluation requires the artifact or HF).
    """
    pool: list[dict] | None = None

    # Try local first if env forces offline, else HF
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        pool = _load_local_artifact()
        if pool is None:
            pool = _try_load_hf_math()
    else:
        pool = _try_load_hf_math()
        if pool is None:
            pool = _load_local_artifact()

    if pool is None:
        # Deterministic synthetic fallback for CI / offline py_compile checks
        print("[livebench] WARNING: no HF or local data found; synthesizing 368 placeholder items for offline checks.")
        pool = []
        for i in range(368):
            # Simple synthetic arithmetic problem
            a, b = (i * 7) % 50 + 1, (i * 13) % 50 + 1
            ans = a + b
            q = f"What is {a} + {b}? Answer as an integer."
            pool.append(
                {
                    "prompt": q,
                    "problem": q,
                    "input": q,
                    "question": q,
                    "answer": str(ans),
                    "ground_truth": str(ans),
                    "synthetic": True,
                    "id": f"synthetic_{i}",
                }
            )

    # Verify expected size; warn if drift
    if len(pool) != 368:
        print(f"[livebench] WARNING: expected 368 math problems, got {len(pool)}; splitting proportionally.")

    rng = random.Random(seed)
    rng.shuffle(pool)

    if splits is not None:
        n_train, n_val, n_test = splits
    else:
        # 122/123/123 for 368
        n_train = 122
        n_val = 123
        n_test = len(pool) - n_train - n_val  # 123

    trainset = pool[:n_train]
    valset = pool[n_train : n_train + n_val]
    testset = pool[n_train + n_val : n_train + n_val + n_test]

    # Attach stable ids for caching if not already present
    for i, ex in enumerate(trainset):
        ex.setdefault("id", f"livebench_train_{i}")
    for i, ex in enumerate(valset):
        ex.setdefault("id", f"livebench_val_{i}")
    for i, ex in enumerate(testset):
        ex.setdefault("id", f"livebench_test_{i}")

    return trainset, valset, testset


# Legacy alias for older imports
load_dataset = load_livebench_math_dataset
load_livebench_dataset = load_livebench_math_dataset


def evaluate_on_set(
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
        prob = example.get("prompt") or example.get("problem") or example.get("input", "")
        out = run_livebench_single_stage(prompt_str, prob, model=model, api_base=api_base)
        score, _ = livebench_metric(out, example)
        return score

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        scores = list(pool.map(score_one, dataset))
    return sum(scores) / len(scores) if scores else 0.0
