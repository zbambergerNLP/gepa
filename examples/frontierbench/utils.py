"""FrontierBench utilities: dataset loading, 1-stage and 2-stage research programs, and metric.

Replicates the Frontier-Bench setup (https://github.com/laude-institute/frontier-bench):
harder agentic research tasks from the Terminal-Bench authors. Each task is an
end-to-end research assignment (literature + code + analysis) scored by an
LLM judge / test suite pass.

The metric is task success (0-1) via LLM judge (or test-suite substring checks)
with per-criterion feedback. See ATTRIBUTION.md.

Dataset loading mirrors ifbench/pupa/hotpotqa/frontiercs patterns:
HF ``laude/frontier-bench`` or a local ``data/frontierbench.jsonl`` fallback,
plus a synthetic offline fallback so that :func:`load_frontierbench_dataset`
always returns 30/30/30 splits for smoke/tests without network.

Program structure mirrors FrontierCS/IFBench:
- 1-stage: single research execution turn (one optimized prompt, one LM call).
- 2-stage: research plan -> execution (two optimized prompts, two LM calls,
  the second conditioned on the first).

Decoding config is identical to ``examples/ifbench/utils.py``
(temp 0.6, top_p 0.95, top_k 20, max 16384, enable_thinking False, truncation
retries, <think> stripping).
"""

from __future__ import annotations

import json
import os
import random
import re

import litellm

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_DATA_PATH = os.path.join(DATA_DIR, "frontierbench.jsonl")

FINAL_RESPONSE_MARKER = "Final Response:"

COT_FORMAT_INSTRUCTION = (
    "\n\nFirst reason step by step about how to best respond. Then write your "
    f"final response after a line containing exactly '{FINAL_RESPONSE_MARKER}'. "
    "Only the text after that line is used as your response."
)

_SYNTHETIC_CATEGORIES = ["Literature Review", "Experimental Design", "Data Analysis", "System Building", "Evaluation"]
_SYNTHETIC_DIFFICULTIES = ["medium", "hard", "extreme"]


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models like Qwen3."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_final_response(output: str) -> str:
    """Extract the text after the last 'Final Response:' marker.

    Falls back to the full output when the marker is missing, mirroring how
    the artifact's program always scores the final_response field.
    """
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
        # Paper decoding config for Qwen3-8B (gepa-artifact experiment_configs.py:
        # temp=0.6, top-p=0.95, top-k=20; max_tokens=16384 from run_experiments.py).
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": 16384,
        # Disable Qwen's hidden thinking mode: COT_FORMAT_INSTRUCTION already
        # elicits visible reasoning (like dspy ChainOfThought), and hidden
        # <think> blocks can consume the entire token budget, leaving
        # message.content empty (vLLM's qwen3 reasoning parser routes them to
        # reasoning_content).
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    if api_base is not None:
        kwargs["api_base"] = api_base
    # Long inputs (e.g. a rambling stage-1 response, or a candidate prompt that
    # grew huge over many accretive edits) can leave less than max_tokens of
    # context headroom. Step the output budget down before giving up; if the
    # input alone overflows the context, return "" so the rollout scores 0
    # instead of killing the run.
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
        # Fallback: if the model still spent the whole budget thinking, score
        # the reasoning text rather than an empty string.
        content = getattr(message, "reasoning_content", None) or ""
    return content


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _normalize_record(rec: dict, idx: int) -> dict:
    """Normalize a raw Frontier-Bench record to the canonical example schema."""
    # Accept multiple upstream field names (HF vs local jsonl vs synthetic).
    prompt = (
        rec.get("prompt")
        or rec.get("task")
        or rec.get("instruction")
        or rec.get("problem")
        or rec.get("question")
        or rec.get("description")
        or rec.get("goal")
        or ""
    )
    # Tests / success criteria may be string, list, or missing.
    tests = rec.get("tests") or rec.get("test_cases") or rec.get("success_criteria") or rec.get("evaluation") or rec.get("criteria") or ""
    if isinstance(tests, list):
        tests_str = "\n".join(str(t) for t in tests)
        tests_list = [str(t) for t in tests]
    elif isinstance(tests, str):
        tests_str = tests
        tests_list = [t.strip() for t in re.split(r"\n|;\s*", tests) if t.strip()]
    else:
        tests_str = str(tests)
        tests_list = [tests_str] if tests_str else []

    category = rec.get("category") or rec.get("research_area") or rec.get("area") or _SYNTHETIC_CATEGORIES[idx % len(_SYNTHETIC_CATEGORIES)]
    difficulty = rec.get("difficulty") or _SYNTHETIC_DIFFICULTIES[idx % len(_SYNTHETIC_DIFFICULTIES)]
    example_id = str(rec.get("id") or rec.get("task_id") or rec.get("problem_id") or f"frontierbench_{idx}")
    reference = rec.get("reference") or rec.get("gold") or rec.get("answer") or rec.get("solution") or ""
    # Keep tests_list bounded for prompt brevity
    if len(tests_list) > 6:
        tests_list = tests_list[:6]

    return {
        "id": example_id,
        "task": str(prompt),
        "prompt": str(prompt),
        "instruction": str(prompt),
        "tests": tests_str,
        "tests_list": tests_list,
        "category": str(category),
        "difficulty": str(difficulty),
        "success_criteria": tests_str,
        "reference": str(reference),
        "raw": rec,
    }


def _synthetic_examples(n: int = 90) -> list[dict]:
    """Generate deterministic synthetic FrontierBench examples for offline fallback."""
    stems = [
        "Reproduce the scaling-law experiment from the Chinchilla paper on a 1B-param model and report the loss curve.",
        "Build an agent that can autonomously debug a failing CI pipeline by reading logs and patching the Dockerfile.",
        "Conduct a literature review on mechanistic interpretability of induction heads and propose a novel intervention.",
        "Implement a distributed training job with FSDP on 8 GPUs and benchmark throughput vs. ZeRO-3.",
        "Design and run an ablation study for a retrieval-augmented generation pipeline over 100K documents.",
        "Analyze the robustness of LLM watermarking under paraphrase attacks and propose a mitigation.",
        "Create a reproducible benchmark for long-context code generation with execution-based scoring.",
        "Investigate catastrophic forgetting in continual pre-training and propose a replay-free regularizer.",
        "Build a tool-using agent that solves a multi-step data-science task via bash and Python in a sandbox.",
        "Evaluate the faithfulness of chain-of-thought via counterfactual perturbations and causal tracing.",
    ]
    examples: list[dict] = []
    for i in range(n):
        stem = stems[i % len(stems)]
        task = f"[Synthetic {i}] {stem} Provide a complete execution trace, code, and analysis."
        rec = {
            "id": f"frontierbench_synth_{i}",
            "task": task,
            "tests": [
                "Output contains a clear methodology and steps taken.",
                "Output includes concrete results, numbers, or code artifacts.",
                "Output discusses limitations and validates claims.",
            ],
            "category": _SYNTHETIC_CATEGORIES[i % len(_SYNTHETIC_CATEGORIES)],
            "difficulty": _SYNTHETIC_DIFFICULTIES[i % len(_SYNTHETIC_DIFFICULTIES)],
            "reference": f"Reference solution {i}: staged execution with validation checks.",
        }
        examples.append(_normalize_record(rec, i))
    return examples


def _load_from_jsonl(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def load_frontierbench_dataset(
    data_path: str | None = None,
    *,
    seed: int = 0,
    train_limit: int | None = None,
    val_limit: int | None = None,
    test_limit: int | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Load FrontierBench with 30/30/30 splits (paper-scale).

    Load order:
    1. If ``data_path`` is given, load that JSONL file (each line is one task).
       Expected fields: ``task`` / ``prompt`` / ``instruction``, ``tests`` or
       ``success_criteria`` (string or list), ``category``, ``difficulty``.
       A 20-line smoke file is expanded to 30/30/30 by cycling.
    2. Else try HuggingFace ``laude/frontier-bench`` via ``datasets``.
       The repo reports harder agentic research tasks (Terminal-Bench lineage);
       we shuffle (seed 0) and slice 30/30/30 (train 0:30, val 30:60, test 60:90).
    3. Else try bundled ``data/frontierbench.jsonl`` if it exists.
    4. Else fall back to deterministic synthetic 90-example pool (30/30/30).

    Returns (trainset, valset, testset) as lists of dicts with keys:
    id, task, prompt, instruction, tests, tests_list, category, difficulty,
    success_criteria, reference.
    """
    examples: list[dict] | None = None

    # 1) Explicit file path mode
    if data_path is not None:
        data_path = os.path.normpath(data_path)
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"FrontierBench data not found at {data_path}. "
                f"Expected a JSONL file with one task per line, or omit --data-path to try HF 'laude/frontier-bench'."
            )
        raw = _load_from_jsonl(data_path)
        if not raw:
            raise ValueError(f"FrontierBench data at {data_path} is empty.")
        examples = [_normalize_record(r, i) for i, r in enumerate(raw)]
        if len(examples) < 90 and len(examples) >= 5:
            base = list(examples)
            expanded: list[dict] = []
            for i in range(90):
                rec = dict(base[i % len(base)])
                rec["id"] = f"{rec['id']}_cycle{i}"
                expanded.append(rec)
            examples = expanded
        rng = random.Random(seed)
        rng.shuffle(examples)
        trainset = examples[:30]
        valset = examples[30:60]
        testset = examples[60:90]
        if train_limit is not None:
            trainset = trainset[:train_limit]
        if val_limit is not None:
            valset = valset[:val_limit]
        if test_limit is not None:
            testset = testset[:test_limit]
        return trainset, valset, testset

    # 2) HF mode
    try:
        from datasets import load_dataset  # type: ignore

        hf_candidates = [
            ("laude/frontier-bench", None),
            ("laude/frontier-bench", "default"),
        ]
        ds = None
        last_err: Exception | None = None
        for hf_name, config in hf_candidates:
            try:
                if config is None:
                    ds = load_dataset(hf_name)
                else:
                    ds = load_dataset(hf_name, config)
                break
            except Exception as e:
                last_err = e
                continue
        if ds is not None:
            split_name = None
            for cand in ("train", "test", "validation", "default"):
                if cand in ds:
                    split_name = cand
                    break
            if split_name is None:
                split_name = list(ds.keys())[0]
            raw_hf = list(ds[split_name])
            if len(raw_hf) >= 30:
                examples = [_normalize_record(dict(r), i) for i, r in enumerate(raw_hf)]
                rng = random.Random(seed)
                rng.shuffle(examples)
                trainset = examples[:30]
                valset = examples[30:60]
                testset = examples[60:90] if len(examples) >= 90 else examples[60:]
                if len(testset) > 30:
                    testset = testset[:30]
                if len(examples) < 90:
                    base = list(examples)
                    expanded = []
                    for i in range(90):
                        rec = dict(base[i % len(base)])
                        rec["id"] = f"{rec['id']}_cycle{i}"
                        expanded.append(rec)
                    examples = expanded
                    rng = random.Random(seed)
                    rng.shuffle(examples)
                    trainset = examples[:30]
                    valset = examples[30:60]
                    testset = examples[60:90]
                if train_limit is not None:
                    trainset = trainset[:train_limit]
                if val_limit is not None:
                    valset = valset[:val_limit]
                if test_limit is not None:
                    testset = testset[:test_limit]
                return trainset, valset, testset
            else:
                raise ValueError(f"HF FrontierBench split {split_name} has only {len(raw_hf)} examples (<30)")
        else:
            if last_err is not None:
                raise last_err
            raise ValueError("No HF dataset loaded")
    except Exception as e:
        # 3) Bundled local fallback
        if os.path.exists(DEFAULT_DATA_PATH):
            try:
                raw = _load_from_jsonl(DEFAULT_DATA_PATH)
                if raw:
                    examples = [_normalize_record(r, i) for i, r in enumerate(raw)]
                    rng = random.Random(seed)
                    rng.shuffle(examples)
                    if len(examples) >= 90:
                        trainset = examples[:30]
                        valset = examples[30:60]
                        testset = examples[60:90]
                    elif len(examples) >= 30:
                        base = list(examples)
                        expanded = []
                        for i in range(90):
                            rec = dict(base[i % len(base)])
                            rec["id"] = f"{rec['id']}_cycle{i}"
                            expanded.append(rec)
                        rng = random.Random(seed)
                        rng.shuffle(expanded)
                        trainset = expanded[:30]
                        valset = expanded[30:60]
                        testset = expanded[60:90]
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
                    print(f"WARNING: HF FrontierBench load failed ({e}); using bundled {DEFAULT_DATA_PATH}.")
                    return trainset, valset, testset
            except Exception:
                pass
        # 4) Synthetic offline fallback (always 30/30/30)
        print(f"WARNING: HF FrontierBench load failed ({e}); using synthetic offline fallback 30/30/30.")
        examples = _synthetic_examples(90)
        rng = random.Random(seed)
        rng.shuffle(examples)
        trainset = examples[:30]
        valset = examples[30:60]
        testset = examples[60:90]
        if train_limit is not None:
            trainset = trainset[:train_limit]
        if val_limit is not None:
            valset = valset[:val_limit]
        if test_limit is not None:
            testset = testset[:test_limit]
        return trainset, valset, testset


# ---------------------------------------------------------------------------
# Programs
# ---------------------------------------------------------------------------

def run_single_stage(
    prompt: str,
    task: str,
    model: str = "hosted_vllm/Qwen3.5-9B",
    api_base: str | None = None,
) -> str:
    """Run the 1-stage FrontierBench program: one optimized prompt, one LM call.

    The prompt should instruct the model to execute the research task end-to-end.
    Returns the final response (after Final Response marker).
    """
    out = _call_lm(prompt + COT_FORMAT_INSTRUCTION, f"Task:\n{task}", model, api_base)
    return _extract_final_response(out)


def run_two_stage(
    plan_prompt: str,
    execute_prompt: str,
    task: str,
    model: str = "hosted_vllm/Qwen3.5-9B",
    api_base: str | None = None,
) -> tuple[str, str]:
    """Run the 2-stage FrontierBench program: plan -> execute.

    Stage 1 (plan): generate a research plan / literature outline for the task.
    Stage 2 (execute): execute the task conditioned on the task plus the
    stage-1 plan; its output is the final response that gets scored.

    Returns (plan, final_output).
    """
    stage1_out = _call_lm(
        plan_prompt + COT_FORMAT_INSTRUCTION,
        f"Task:\n{task}\n\nFirst, produce a detailed research plan for this task.",
        model,
        api_base,
    )
    plan = _extract_final_response(stage1_out)

    # Cap plan fed into stage 2
    plan_capped = plan if len(plan) <= 12000 else plan[:12000] + "\n[truncated]"
    stage2_user = f"Task:\n{task}\n\nResearch Plan:\n{plan_capped}\n\nNow execute the task and provide the full output."
    stage2_out = _call_lm(execute_prompt + COT_FORMAT_INSTRUCTION, stage2_user, model, api_base)
    final = _extract_final_response(stage2_out)
    return plan, final


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def _heuristic_task_score(response: str, tests_list: list[str]) -> tuple[float, str]:
    """Offline heuristic fallback when no judge model is available.

    For each test criterion, score 1 if keywords from the criterion appear in
    the response or the response is long/structured, else 0. Keeps metric
    usable offline with non-zero variance for GEPA.
    """
    if not tests_list:
        # No explicit tests: score by structure + length
        has_structure = any(kw in response.lower() for kw in ["method", "result", "evaluat", "experiment", "analysis", "conclusion"])
        length_ok = len(response.strip()) > 300
        score = 1.0 if (has_structure and length_ok) else (0.5 if length_ok else 0.0)
        return score, f"Heuristic (no tests): structure={has_structure} length_ok={length_ok} score={score:.2f}"
    scores: list[bool] = []
    fb_lines: list[str] = []
    resp_lower = response.lower()
    for test in tests_list:
        words = [w for w in re.findall(r"[a-z]{4,}", test.lower()) if w not in {"that", "with", "from", "this", "have", "will", "your", "task"}]
        if not words:
            satisfied = len(response.strip()) > 100
        else:
            hits = sum(1 for w in words if w in resp_lower)
            satisfied = (hits / len(words) >= 0.4) or (len(response) > 800 and hits > 0)
        scores.append(satisfied)
        status = "PASS" if satisfied else "FAIL"
        fb_lines.append(f"[{status}] {test} (heuristic)")
    score = sum(scores) / len(scores) if scores else 0.0
    feedback = "\n".join(fb_lines) + f"\nHeuristic task success: {score:.2f} ({sum(scores)}/{len(scores)})"
    return score, feedback


def frontierbench_metric(
    response: str,
    example: dict,
    judge_model: str | None = None,
    judge_api_base: str | None = None,
) -> tuple[float, str]:
    """Score a FrontierBench response.

    Primary path: LLM judge prompted with task, tests/success_criteria, and
    response; asked to give per-test PASS/FAIL and a final SCORE 0-1 (mean pass
    rate). Mirrors frontiercs_metric but for agentic task success.

    Offline fallback: heuristic keyword overlap.

    Returns (score, feedback) where score is 0-1.
    """
    tests_list: list[str] = example.get("tests_list") or []
    if not tests_list:
        # Derive from tests string if tests_list missing
        tests_raw = example.get("tests") or example.get("success_criteria") or ""
        if isinstance(tests_raw, list):
            tests_list = [str(t) for t in tests_raw]
        elif isinstance(tests_raw, str) and tests_raw.strip():
            tests_list = [t.strip() for t in re.split(r"\n|;\s*", tests_raw) if t.strip()]
    task = example.get("task") or example.get("prompt") or example.get("instruction") or ""

    if judge_model is not None:
        tests_text = "\n".join(f"{i+1}. {t}" for i, t in enumerate(tests_list)) if tests_list else "(no explicit tests; judge holistic success)"
        judge_prompt = (
            "You are a strict grader for agentic research tasks (Terminal-Bench / Frontier-Bench style).\n"
            f"Task:\n{task}\n\n"
            f"Success Criteria / Tests:\n{tests_text}\n\n"
            f"Candidate Output (truncated):\n{response[:8000]}\n\n"
            "For each success criterion, say PASS or FAIL with a one-line reason.\n"
            "Then on the last line write SCORE: <float between 0 and 1> as fraction of criteria that PASS.\n"
            "If no explicit criteria, score holistically 0-1 on task completion.\n"
            "Be strict but fair."
        )
        try:
            judge_out = _call_lm("You are a helpful grader.", judge_prompt, judge_model, judge_api_base)
            lines = judge_out.splitlines()
            passes: list[bool] = []
            for line in lines:
                if "PASS" in line.upper() and "FAIL" not in line.upper():
                    if re.search(r"(criterion|criteria|test|#|\d+\.)", line, re.IGNORECASE) or len(tests_list) <= 4:
                        passes.append(True)
                elif "FAIL" in line.upper():
                    if re.search(r"(criterion|criteria|test|#|\d+\.)", line, re.IGNORECASE) or len(tests_list) <= 4:
                        passes.append(False)
            m = re.search(r"SCORE\s*:\s*(0?\.\d+|1\.0|0|1)", judge_out, re.IGNORECASE)
            if m is not None:
                try:
                    parsed = float(m.group(1))
                    parsed = max(0.0, min(1.0, parsed))
                    if len(passes) == len(tests_list) and tests_list:
                        score = sum(passes) / len(tests_list)
                        score = 0.7 * score + 0.3 * parsed
                    else:
                        score = parsed
                    feedback = judge_out.strip()[:2000] + f"\nTask success: {score:.2f}"
                    return score, feedback
                except ValueError:
                    pass
            if len(passes) == len(tests_list) and tests_list:
                score = sum(passes) / len(tests_list)
                feedback = judge_out.strip()[:2000] + f"\nTask success: {score:.2f} ({sum(passes)}/{len(tests_list)})"
                return score, feedback
            if passes:
                score = sum(passes) / len(passes)
                feedback = judge_out.strip()[:2000] + f"\nParsed success: {score:.2f} ({sum(passes)}/{len(passes)})"
                return score, feedback
        except Exception:
            pass

    return _heuristic_task_score(response, tests_list)


# Alias for generic harness compatibility
frontier_metric = frontierbench_metric
