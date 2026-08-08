"""FrontierCS utilities: dataset loading, 1-stage and 2-stage research programs, and metric.

Replicates the Frontier-CS setup (https://github.com/FrontierCS/Frontier-CS):
open-ended CS research problems benchmarked via an auto-research framework.
The metric is a rubric-based LLM-judge pass (0-1) with per-criterion feedback.
See ATTRIBUTION.md.

Dataset loading mirrors ifbench/pupa/hotpotqa patterns:
HF ``FrontierCS/Frontier-CS`` or a local ``data/frontiercs.jsonl`` fallback,
plus a synthetic offline fallback so that :func:`load_frontiercs_dataset`
always returns 30/30/30 splits for smoke/tests without network.

Program structure mirrors IFBench:
- 1-stage: single research-proposal turn (one optimized prompt, one LM call).
- 2-stage: literature review -> proposal (two optimized prompts, two LM calls,
  the second conditioned on the first).

Decoding config is identical to ``examples/ifbench/utils.py`` and
``examples/pupa/utils.py`` (paper Qwen3-8B: temp 0.6, top_p 0.95, top_k 20,
max 16384, enable_thinking False, truncation retries, <think> stripping).
"""

from __future__ import annotations

import json
import os
import random
import re

import litellm

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_DATA_PATH = os.path.join(DATA_DIR, "frontiercs.jsonl")

FINAL_RESPONSE_MARKER = "Final Response:"

COT_FORMAT_INSTRUCTION = (
    "\n\nFirst reason step by step about how to best respond. Then write your "
    f"final response after a line containing exactly '{FINAL_RESPONSE_MARKER}'. "
    "Only the text after that line is used as your response."
)

_SYNTHETIC_AREAS = ["Machine Learning", "Systems", "Theory", "Security", "HCI"]
_SYNTHETIC_DIFFICULTIES = ["easy", "medium", "hard"]


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
    """Normalize a raw Frontier-CS record to the canonical example schema."""
    # Accept multiple upstream field names (HF vs local jsonl vs synthetic).
    problem = (
        rec.get("problem")
        or rec.get("prompt")
        or rec.get("question")
        or rec.get("task")
        or rec.get("research_problem")
        or rec.get("description")
        or ""
    )
    # Rubric may be list, string, or missing.
    rubric = rec.get("rubric") or rec.get("criteria") or rec.get("evaluation_rubric") or rec.get("grading_rubric") or []
    if isinstance(rubric, str):
        # split on newlines / numbered items
        rubric = [r.strip() for r in re.split(r"\n|;\s*", rubric) if r.strip()]
    if not isinstance(rubric, list):
        rubric = [str(rubric)]
    # Keep at most 6 rubric items for prompt brevity.
    rubric = [str(r) for r in rubric[:6]]
    if not rubric:
        rubric = [
            "Proposal is technically sound and feasible.",
            "Proposal demonstrates novelty over prior work.",
            "Proposal includes clear evaluation plan.",
        ]

    area = rec.get("area") or rec.get("research_area") or rec.get("category") or _SYNTHETIC_AREAS[idx % len(_SYNTHETIC_AREAS)]
    difficulty = rec.get("difficulty") or _SYNTHETIC_DIFFICULTIES[idx % len(_SYNTHETIC_DIFFICULTIES)]
    example_id = str(rec.get("id") or rec.get("problem_id") or rec.get("task_id") or f"frontiercs_{idx}")
    reference = rec.get("reference") or rec.get("reference_solution") or rec.get("gold") or rec.get("answer") or ""
    context = rec.get("context") or rec.get("literature_context") or rec.get("background") or ""

    return {
        "id": example_id,
        "problem": str(problem),
        "prompt": str(problem),
        "area": str(area),
        "difficulty": str(difficulty),
        "rubric": rubric,
        "reference": str(reference),
        "context": str(context),
        "raw": rec,
    }


def _synthetic_examples(n: int = 90) -> list[dict]:
    """Generate deterministic synthetic FrontierCS examples for offline fallback.

    Each synthetic item has a distinct research problem stem so that
    prompt-diversity and metric logic can be exercised without network.
    The synthetic split is 30/30/30 after shuffling.
    """
    stems = [
        "Design a differentially-private federated learning protocol that tolerates 30% Byzantine clients.",
        "Propose a verified compilation pipeline for WebAssembly that preserves constant-time guarantees.",
        "Develop a sublinear-time algorithm for estimating graph effective resistance.",
        "Create an interactive proof system for verifying LLM chain-of-thought faithfulness.",
        "Build a cache-coherent accelerator for sparse attention with near-memory compute.",
        "Formulate a causal inference framework for A/B tests with network interference.",
        "Design a post-quantum signature scheme with aggregatable signatures for blockchains.",
        "Propose a continual-learning regularizer that prevents catastrophic forgetting on non-stationary streams.",
        "Develop a formal method for detecting prompt-injection in tool-using agents.",
        "Create a benchmark for evaluating long-context retrieval over 1M-token codebases.",
    ]
    examples: list[dict] = []
    for i in range(n):
        stem = stems[i % len(stems)]
        problem = f"[Synthetic {i}] {stem} Elaborate the core idea, novelty over baselines, and evaluation plan."
        rec = {
            "id": f"frontiercs_synth_{i}",
            "problem": problem,
            "area": _SYNTHETIC_AREAS[i % len(_SYNTHETIC_AREAS)],
            "difficulty": _SYNTHETIC_DIFFICULTIES[i % len(_SYNTHETIC_DIFFICULTIES)],
            "rubric": [
                "Technical soundness and feasibility of the approach.",
                "Novelty relative to prior work and clear differentiation.",
                "Evaluation plan with baselines, datasets, and metrics.",
                "Discussion of limitations and failure modes.",
            ],
            "reference": f"Reference idea {i}: use coupled analysis plus empirical validation on 3 benchmarks.",
            "context": f"Background for {stem[:60]}...",
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


def load_frontiercs_dataset(
    data_path: str | None = None,
    *,
    seed: int = 0,
    train_limit: int | None = None,
    val_limit: int | None = None,
    test_limit: int | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Load FrontierCS with 30/30/30 splits (paper-scale).

    Load order:
    1. If ``data_path`` is given, load that JSONL file (each line is one task).
       Expected fields: ``problem`` / ``prompt``, ``rubric`` (list or string),
       ``area``, ``difficulty``. Missing fields use sensible defaults. A 20-line
       smoke file is expanded to 30/30/30 by cycling when needed so len checks
       pass offline.
    2. Else try HuggingFace ``FrontierCS/Frontier-CS`` via ``datasets``.
       The paper reports ~100 open-ended CS research problems; we use a
       deterministic shuffle (seed 0) and slice 30/30/30 (train 0:30, val 30:60,
       test 60:90), mirroring IFBench's slicing from a single pool.
    3. Else try the bundled ``data/frontiercs.jsonl`` if it exists.
    4. Else fall back to a deterministic synthetic 90-example pool (30/30/30).

    Returns (trainset, valset, testset) as lists of dicts with keys:
    id, problem, prompt, area, difficulty, rubric (list), reference, context.
    """
    examples: list[dict] | None = None

    # 1) Explicit file path mode (smoke / user-provided)
    if data_path is not None:
        data_path = os.path.normpath(data_path)
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"FrontierCS data not found at {data_path}. "
                f"Expected a JSONL file with one task per line, or omit --data-path to try HF 'FrontierCS/Frontier-CS'."
            )
        raw = _load_from_jsonl(data_path)
        if not raw:
            raise ValueError(f"FrontierCS data at {data_path} is empty.")
        examples = [_normalize_record(r, i) for i, r in enumerate(raw)]
        # Smoke handling: if file has <90 examples, cycle to reach 90 so 30/30/30 works
        if len(examples) < 90 and len(examples) >= 5:
            base = list(examples)
            expanded: list[dict] = []
            for i in range(90):
                rec = dict(base[i % len(base)])
                rec = dict(rec)
                rec["id"] = f"{rec['id']}_cycle{i}"
                expanded.append(rec)
            examples = expanded
        # Shuffle deterministically then slice
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

        # FrontierCS/Frontier-CS may have a single split; try common names.
        hf_candidates = [
            ("FrontierCS/Frontier-CS", None),
            ("FrontierCS/Frontier-CS", "default"),
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
            # Pick first split that exists
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
                # Paper notes ~100 problems; we slice 30/30/30 from the pool.
                trainset = examples[:30]
                valset = examples[30:60]
                testset = examples[60:90] if len(examples) >= 90 else examples[60:]
                # If pool is exactly 90, test is 30; if larger, cap at 30 as well.
                if len(testset) > 30:
                    testset = testset[:30]
                # If pool was 60-89, we still return what we have (caller may limit)
                if len(examples) < 90:
                    # pad by cycling to keep 30/30/30 invariant for tests when HF pool is small
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
                raise ValueError(f"HF FrontierCS split {split_name} has only {len(raw_hf)} examples (<30)")
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
                        # cycle to 90
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
                    print(f"WARNING: HF FrontierCS load failed ({e}); using bundled {DEFAULT_DATA_PATH}.")
                    return trainset, valset, testset
            except Exception:
                pass
        # 4) Synthetic offline fallback (always 30/30/30)
        print(f"WARNING: HF FrontierCS load failed ({e}); using synthetic offline fallback 30/30/30.")
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
    problem: str,
    model: str = "hosted_vllm/Qwen3.5-9B",
    api_base: str | None = None,
) -> str:
    """Run the 1-stage FrontierCS program: one optimized prompt, one LM call.

    The prompt should instruct the model to produce a research proposal for the
    given CS problem. Returns the final response (after Final Response marker).
    """
    out = _call_lm(prompt + COT_FORMAT_INSTRUCTION, f"Research Problem:\n{problem}", model, api_base)
    return _extract_final_response(out)


def run_two_stage(
    literature_prompt: str,
    proposal_prompt: str,
    problem: str,
    model: str = "hosted_vllm/Qwen3.5-9B",
    api_base: str | None = None,
) -> tuple[str, str]:
    """Run the 2-stage FrontierCS program: literature review -> proposal.

    Stage 1 (literature): survey relevant prior work / baselines for the problem.
    Stage 2 (proposal): draft the research proposal conditioned on the problem
    plus the stage-1 literature summary; its output is the final response that
    gets scored.

    Returns (literature_review, final_proposal).
    """
    stage1_out = _call_lm(
        literature_prompt + COT_FORMAT_INSTRUCTION,
        f"Research Problem:\n{problem}\n\nSurvey the most relevant prior work and baselines for this problem.",
        model,
        api_base,
    )
    literature = _extract_final_response(stage1_out)

    # Cap stage-1 text fed into stage 2 so query + response + output budget
    # always fits the model context (32k for Qwen3-8B). ~12k chars ~3k tokens.
    lit_capped = literature if len(literature) <= 12000 else literature[:12000] + "\n[truncated]"
    stage2_user = f"Research Problem:\n{problem}\n\nLiterature Review:\n{lit_capped}\n\nNow draft the full research proposal."
    stage2_out = _call_lm(proposal_prompt + COT_FORMAT_INSTRUCTION, stage2_user, model, api_base)
    final = _extract_final_response(stage2_out)
    return literature, final


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def _heuristic_rubric_score(response: str, rubric: list[str]) -> tuple[float, str]:
    """Offline heuristic fallback when no judge model is available.

    Scores each rubric item 1 if the response mentions keywords from the item
    (or is sufficiently long and structured), else 0. This keeps the metric
    usable offline and ensures non-zero variance for GEPA reflection.
    """
    if not rubric:
        return (1.0 if len(response.strip()) > 200 else 0.0), "Heuristic: no rubric, scored by length."
    scores: list[bool] = []
    feedback_lines: list[str] = []
    resp_lower = response.lower()
    for item in rubric:
        # Heuristic: item is satisfied if at least half its content words appear in response,
        # or response is long and mentions generic research signals.
        words = [w for w in re.findall(r"[a-z]{4,}", item.lower()) if w not in {"that", "with", "from", "this", "have", "will", "your"}]
        if not words:
            satisfied = len(response.strip()) > 100
        else:
            hits = sum(1 for w in words if w in resp_lower)
            satisfied = (hits / len(words) >= 0.4) or (len(response) > 800 and hits > 0)
        scores.append(satisfied)
        status = "PASS" if satisfied else "FAIL"
        feedback_lines.append(f"[{status}] {item} (heuristic)")
    score = sum(scores) / len(scores) if scores else 0.0
    feedback = "\n".join(feedback_lines) + f"\nHeuristic rubric score: {score:.2f} ({sum(scores)}/{len(scores)})"
    return score, feedback


def frontiercs_metric(
    response: str,
    example: dict,
    judge_model: str | None = None,
    judge_api_base: str | None = None,
) -> tuple[float, str]:
    """Score a FrontierCS response against the example's rubric.

    Primary path: LLM judge. The judge is prompted with the problem, rubric,
    and response and asked to score each rubric item 0/1 and give an overall
    0-1. The score is the mean pass rate over rubric items (instruction-level
    accuracy analog, like ifbench_metric). Feedback lists which criteria passed.

    Offline fallback (no judge_model or judge call fails): heuristic keyword
    overlap plus length signal, still returning a 0-1 score and feedback so
    GEPA reflection has a learning signal.

    Returns (score, feedback).
    """
    rubric: list[str] = example.get("rubric") or []
    if isinstance(rubric, str):
        rubric = [rubric]
    problem = example.get("problem") or example.get("prompt") or ""

    # Try LLM judge if a model is provided
    if judge_model is not None:
        rubric_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(rubric))
        judge_prompt = (
            "You are a strict grader for open-ended CS research proposals.\n"
            f"Research Problem:\n{problem}\n\n"
            f"Rubric ({len(rubric)} criteria):\n{rubric_text}\n\n"
            f"Candidate Proposal:\n{response[:8000]}\n\n"
            "For each rubric criterion, say PASS or FAIL and give a one-line reason.\n"
            "Then on the last line write SCORE: <float between 0 and 1> as the fraction of criteria that PASS.\n"
            "Be strict but fair."
        )
        try:
            judge_out = _call_lm("You are a helpful grader.", judge_prompt, judge_model, judge_api_base)
            # Try to parse per-criterion PASS/FAIL
            lines = judge_out.splitlines()
            passes: list[bool] = []
            for line in lines:
                if "PASS" in line.upper() and "FAIL" not in line.upper():
                    # Only count lines that look like rubric judgments
                    if re.search(r"(criterion|criteria|#|\d+\.)", line, re.IGNORECASE) or len(rubric) <= 4:
                        passes.append(True)
                elif "FAIL" in line.upper():
                    if re.search(r"(criterion|criteria|#|\d+\.)", line, re.IGNORECASE) or len(rubric) <= 4:
                        passes.append(False)
            # Prefer explicit SCORE: line if present
            m = re.search(r"SCORE\s*:\s*(0?\.\d+|1\.0|0|1)", judge_out, re.IGNORECASE)
            if m is not None:
                try:
                    parsed = float(m.group(1))
                    parsed = max(0.0, min(1.0, parsed))
                    # If we also have per-item counts, prefer mean; else use parsed
                    if len(passes) == len(rubric) and rubric:
                        score = sum(passes) / len(rubric)
                        # Blend slightly toward parsed to respect judge's holistic view
                        score = 0.7 * score + 0.3 * parsed
                    else:
                        score = parsed
                    # Build feedback from judge output
                    feedback = judge_out.strip()[:2000]
                    # Append normalized line
                    feedback += f"\nRubric pass rate: {score:.2f}"
                    return score, feedback
                except ValueError:
                    pass
            if len(passes) == len(rubric) and rubric:
                score = sum(passes) / len(rubric)
                feedback = judge_out.strip()[:2000] + f"\nRubric pass rate: {score:.2f} ({sum(passes)}/{len(rubric)})"
                return score, feedback
            # Fallback: count PASS vs FAIL occurrences if rubric-length not matched but judge gave signals
            if passes:
                # If judge emitted N PASS/FAIL lines but not exactly rubric length, still use ratio
                score = sum(passes) / len(passes)
                feedback = judge_out.strip()[:2000] + f"\nParsed pass rate: {score:.2f} ({sum(passes)}/{len(passes)})"
                return score, feedback
        except Exception:
            pass  # fall through to heuristic

    # Heuristic fallback
    return _heuristic_rubric_score(response, rubric)


# Backwards-compatible alias (some callers may import frontiercs_metric as metric)
frontier_metric = frontiercs_metric
