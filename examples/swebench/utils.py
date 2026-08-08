"""SWE-Bench utilities: dataset loading, 1/2-stage LM program, and metric.

Replicates the GEPA paper pattern (mirroring examples/ifbench) for SWE-Bench
Verified (Jimenez et al. 2024, https://www.swebench.com, ~2294 Python GitHub
issues, Verified 500). Single-stage: code patch generation. Two-stage:
locate-then-fix (identify files/lines, then generate patch). Metric is patch
applies + tests pass proxy (exact patch match / test-feedback string) with
feedback. _call_lm is identical to ifbench (temp 0.6 / top_p 0.95 / top_k 20 /
max 16384 / enable_thinking False, truncation, retries, <think> stripping).
Offline-friendly: HF `princeton-nlp/SWE-bench_Verified` with local
`data/swebench_verified.jsonl` fallback and synthetic fallback. Long code
context is truncated so input + output fits the model window.

See ATTRIBUTION.md.
"""

from __future__ import annotations

import json
import os
import random
import re

import litellm

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_HF_DATASET = "princeton-nlp/SWE-bench_Verified"
_FALLBACK_FILE = "swebench_verified.jsonl"

FINAL_PATCH_MARKER = "Final Patch:"
FINAL_LOCATE_MARKER = "Final Location:"

COT_FORMAT_INSTRUCTION = (
    "\n\nFirst reason step by step about how to best respond. Then write your "
    f"final response after a line containing exactly '{FINAL_PATCH_MARKER}'. "
    "Only the text after that line is used as your response. "
    "For code patches, emit a unified diff (git diff format)."
)

LOCATE_COT_INSTRUCTION = (
    "\n\nFirst reason step by step about the bug location. Then write your "
    f"location after a line containing exactly '{FINAL_LOCATE_MARKER}'. "
    "Only the text after that line is used as your location. "
    "List file paths and line ranges that need changing."
)


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models like Qwen3."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_final_patch(output: str) -> str:
    """Extract text after the last FINAL_PATCH_MARKER, fallback to full output."""
    output = _strip_think(output)
    if FINAL_PATCH_MARKER in output:
        return output.rsplit(FINAL_PATCH_MARKER, 1)[1].strip()
    return output.strip()


def _extract_final_locate(output: str) -> str:
    """Extract text after the last FINAL_LOCATE_MARKER, fallback to full output."""
    output = _strip_think(output)
    if FINAL_LOCATE_MARKER in output:
        return output.rsplit(FINAL_LOCATE_MARKER, 1)[1].strip()
    return output.strip()


def _call_lm(system: str, user: str, model: str, api_base: str | None) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    kwargs: dict = {
        "model": model,
        "messages": messages,
        # Paper decoding config for Qwen3-8B (ifbench experiment_configs.py:
        # temp=0.6, top-p=0.95, top-k=20; max_tokens=16384 from run_experiments.py).
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": 16384,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    if api_base is not None:
        kwargs["api_base"] = api_base
    # Long code context + candidate prompt can overflow the model window.
    # Step the output budget down before giving up; if the input alone
    # overflows the context, return "" so the rollout scores 0.
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


def _read_jsonl(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _normalize_record(raw: dict, idx: int) -> dict:
    """Normalize a raw SWE-Bench record to a common schema.

    Expected normalized keys: instance_id, prompt (problem_statement),
    problem_statement, repo, base_commit, patch (gold), hints_text,
    plus _raw for metric introspection.
    """
    instance_id = str(raw.get("instance_id") or raw.get("id") or raw.get("task_id") or f"swe-{idx}")
    problem_statement = raw.get("problem_statement") or raw.get("issue") or raw.get("prompt") or raw.get("description") or raw.get("text") or ""
    problem_statement = str(problem_statement).strip()
    if not problem_statement:
        for k in ("title", "body", "issue_text"):
            if raw.get(k):
                problem_statement = str(raw[k]).strip()
                break
    if not problem_statement:
        problem_statement = f"Fix the bug in {instance_id}."

    patch = raw.get("patch") or raw.get("gold_patch") or raw.get("solution") or raw.get("diff") or ""
    repo = str(raw.get("repo") or raw.get("repository") or "")
    base_commit = str(raw.get("base_commit") or raw.get("commit") or "")
    hints = raw.get("hints_text") or raw.get("hints") or ""

    # Optionally include code context if present (e.g., SWE-Bench context)
    # Keep it but truncate later at call sites.
    context = raw.get("context") or raw.get("code") or raw.get("file_contents") or ""

    return {
        "instance_id": instance_id,
        "prompt": problem_statement,
        "problem_statement": problem_statement,
        "repo": repo,
        "base_commit": base_commit,
        "patch": str(patch) if patch else "",
        "gold_patch": str(patch) if patch else "",
        "hints_text": str(hints) if hints else "",
        "context": str(context) if context else "",
        "_raw": raw,
    }


def _synthetic_issues(n: int = 90) -> list[dict]:
    """Offline synthetic GitHub issues so GEPA never crashes without data."""
    templates = [
        (
            "Fix off-by-one in `sum_range` — should include end index.\n\n```python\ndef sum_range(a, b):\n    return sum(range(a, b))  # bug: should be b+1\n```",
            "diff --git a/utils.py b/utils.py\n--- a/utils.py\n+++ b/utils.py\n@@ -1,2 +1,2 @@\n-def sum_range(a, b):\n-    return sum(range(a, b))\n+def sum_range(a, b):\n+    return sum(range(a, b+1))\n",
            "utils.py",
        ),
        (
            "Fix `is_valid` to reject empty strings.\n\n```python\ndef is_valid(s):\n    return s is not None  # bug: empty string passes\n```",
            "diff --git a/validate.py b/validate.py\n--- a/validate.py\n+++ b/validate.py\n@@ -1,2 +1,2 @@\n-def is_valid(s):\n-    return s is not None\n+def is_valid(s):\n+    return bool(s)\n",
            "validate.py",
        ),
        (
            "Fix `divide` to handle zero division gracefully.\n\n```python\ndef divide(a, b):\n    return a / b\n```",
            "diff --git a/math.py b/math.py\n--- a/math.py\n+++ b/math.py\n@@ -1,2 +1,4 @@\n def divide(a, b):\n+    if b == 0:\n+        return None\n     return a / b\n",
            "math.py",
        ),
    ]
    out: list[dict] = []
    for i in range(n):
        ps, patch, repo_file = templates[i % len(templates)]
        out.append({
            "instance_id": f"synthetic-{i:04d}",
            "prompt": f"{ps}\n\nInstance {i}",
            "problem_statement": f"{ps}\n\nInstance {i}",
            "repo": f"example/repo-{i % 5}",
            "base_commit": "abc123",
            "patch": patch,
            "gold_patch": patch,
            "hints_text": f"Check file {repo_file}",
            "context": ps,
            "_raw": {"synthetic": True},
        })
    return out


def load_swebench_dataset(
    data_path: str | None = None,
    seed: int = 0,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Load SWE-Bench Verified with ~90-total 30/30/30 splits (train/val/test).

    Priority:
      1. data_path if provided (jsonl: one issue per line).
      2. HF `princeton-nlp/SWE-bench_Verified` via `datasets.load_dataset`.
      3. Local `data/swebench_verified.jsonl` fallback.
      4. Synthetic issues.

    SWE-Bench Verified is 500 test instances; the full SWE-Bench is ~2294.
    This loader shuffles with `seed` and splits into train/val/test for
    optimization. Callers can --train-limit/--val-limit/--test-limit to
    request 100/100/100 or 500-test sweeps (noting full Verified 500).

    Returns (trainset, valset, testset) as lists of normalized dicts.
    """
    records: list[dict] | None = None

    if data_path is not None:
        if os.path.isdir(data_path):
            candidate = os.path.join(data_path, _FALLBACK_FILE)
            if os.path.exists(candidate):
                data_path = candidate
        if os.path.exists(data_path):
            try:
                raw = _read_jsonl(data_path)
                records = [_normalize_record(r, i) for i, r in enumerate(raw) if isinstance(r, dict)]
                print(f"Loaded {len(records)} SWE-Bench issues from {data_path}")
            except Exception as e:
                print(f"WARNING: failed to read {data_path}: {e}; trying fallbacks.")
                records = None
        else:
            print(f"WARNING: --data-path {data_path} not found; trying fallbacks.")
            records = None

    if records is None:
        try:
            from datasets import load_dataset  # type: ignore

            ds = load_dataset(_HF_DATASET)
            split = None
            for candidate in ("test", "train", "validation"):
                if candidate in ds:
                    split = candidate
                    break
            if split is None:
                split = list(ds.keys())[0]
            raw_list = list(ds[split])
            records = [_normalize_record(r, i) for i, r in enumerate(raw_list)]
            print(f"Loaded {len(records)} SWE-Bench issues from HF {_HF_DATASET}/{split}")
        except Exception as e:
            print(f"HF load failed ({e}); trying local fallback.")

    if records is None:
        local = os.path.join(DATA_DIR, _FALLBACK_FILE)
        if os.path.exists(local):
            try:
                raw = _read_jsonl(local)
                records = [_normalize_record(r, i) for i, r in enumerate(raw) if isinstance(r, dict)]
                print(f"Loaded {len(records)} SWE-Bench issues from {local}")
            except Exception as e:
                print(f"Failed to read local {local}: {e}")
                records = None

    if records is None or len(records) == 0:
        print("Using synthetic SWE-Bench issues (offline fallback).")
        records = _synthetic_issues(90)

    records = [r for r in records if r.get("prompt")]
    if not records:
        records = _synthetic_issues(90)

    rng = random.Random(seed)
    rng.shuffle(records)

    n = len(records)
    # Spec: e.g., 30/30/30 or 100/100/100, noting full 500 test vs Verified 500.
    # Default to 90-total 30/30/30 when n≈500 or synthetic 90; scale proportionally.
    if n >= 300:
        # 100/100/100 for richer runs; still leaves remainder for test pool
        trainset = records[:100]
        valset = records[100:200]
        testset = records[200:300] if n >= 300 else records[200:]
        # If caller wants full 500 test, they can --test-limit 500 on remainder
        if n >= 500 and len(testset) < 100:
            testset = records[200:500]
    elif n >= 90:
        trainset = records[:30]
        valset = records[30:60]
        testset = records[60:90]
    elif n >= 30:
        n_train = max(1, n // 3)
        n_val = max(1, n // 3)
        trainset = records[:n_train]
        valset = records[n_train:n_train + n_val]
        testset = records[n_train + n_val:]
        if not testset:
            testset = records[-max(1, n // 3):]
    else:
        mid = n // 2
        trainset = records[:mid] if mid else records[:1]
        valset = records[mid:] if mid < n else records[-1:]
        testset = records[: min(30, n)]

    return trainset, valset, testset


def run_single_stage(
    prompt: str,
    problem_statement: str,
    model: str = "hosted_vllm/Qwen3-8B",
    api_base: str | None = None,
) -> str:
    """Run 1-stage patch generation: one optimized prompt, one LM call.

    The prompt should instruct the model to produce a unified diff.
    Long code context is truncated to fit the model window.
    Returns the extracted final patch.
    """
    # Truncate long problem statements / code context to ~24k chars ≈ 6k tokens
    truncated = problem_statement if len(problem_statement) <= 24000 else problem_statement[:24000] + "\n[truncated]"
    out = _call_lm(prompt + COT_FORMAT_INSTRUCTION, f"Issue:\n{truncated}", model, api_base)
    return _extract_final_patch(out)


def run_two_stage(
    locate_prompt: str,
    fix_prompt: str,
    problem_statement: str,
    model: str = "hosted_vllm/Qwen3-8B",
    api_base: str | None = None,
) -> tuple[str, str]:
    """Run 2-stage locate-then-fix.

    Stage 1 (locate): identify files/lines to change.
    Stage 2 (fix): generate the patch conditioned on the location.

    Returns (location, final_patch).
    """
    truncated_issue = problem_statement if len(problem_statement) <= 24000 else problem_statement[:24000] + "\n[truncated]"
    loc_out = _call_lm(locate_prompt + LOCATE_COT_INSTRUCTION, f"Issue:\n{truncated_issue}", model, api_base)
    location = _extract_final_locate(loc_out)

    loc_trunc = location if len(location) <= 8000 else location[:8000] + "\n[truncated]"
    stage2_user = f"Issue:\n{truncated_issue}\n\nSuspected location:\n{loc_trunc}"
    patch_out = _call_lm(fix_prompt + COT_FORMAT_INSTRUCTION, stage2_user, model, api_base)
    final_patch = _extract_final_patch(patch_out)

    return location, final_patch


def swebench_metric(patch: str, example: dict) -> tuple[float, str]:
    """Score a generated patch (proxy for patch applies + tests pass).

    Checks:
      - Patch is non-empty and looks like a unified diff (diff --git / --- / +++ / @@)
      - If gold patch available: exact match, hunk overlap, file overlap
      - Degeneracy / empty checks

    Returns (score, feedback) where score in [0,1] and feedback is a
    test-feedback string for reflection (lists passes/fails and gold hints).
    """
    raw_patch = patch
    patch = (patch or "").strip()
    gold = (example.get("patch") or example.get("gold_patch") or "").strip()
    instance_id = example.get("instance_id", "?")
    problem_excerpt = (example.get("problem_statement") or example.get("prompt") or "")[:400]

    if not patch:
        return 0.0, (
            f"Instance {instance_id}: Empty patch. Provide a unified diff (git diff format) "
            f"that fixes the issue. Score 0.\nIssue excerpt: {problem_excerpt}"
        )

    # Patch format checks (proxy for `patch applies`)
    has_diff_header = "diff --git" in patch or patch.startswith("diff --")
    has_minus = "--- " in patch or "--- a/" in patch
    has_plus = "+++ " in patch or "+++ b/" in patch
    has_hunk = "@@" in patch
    # Allow minimal patch with at least hunk markers
    format_checks = {
        "diff header (diff --git)": has_diff_header,
        "removed file marker (---)": has_minus,
        "added file marker (+++)": has_plus,
        "hunk header (@@)": has_hunk,
    }
    format_pass = sum(format_checks.values())
    format_score = format_pass / len(format_checks)

    if format_pass == 0:
        return 0.0, (
            f"Instance {instance_id}: Patch has no diff structure (missing diff/---/+++/@@). "
            f"It would fail to apply with `git apply`. Score 0.\n"
            f"Checks: {format_checks}\n"
            f"Your output (first 500 chars):\n{patch[:500]}\n"
            f"Issue excerpt: {problem_excerpt}"
        )

    # If gold available, compare
    if gold:
        if patch.strip() == gold.strip():
            return 1.0, (
                f"Instance {instance_id}: Patch exactly matches gold — proxy for patch applies + all tests pass. Score 1.\n"
                f"Checks: {format_checks}"
            )
        # File overlap: do we touch the same files?
        gold_files = set(re.findall(r"--- a/(\S+)", gold)) | set(re.findall(r"\+\+\+ b/(\S+)", gold))
        pred_files = set(re.findall(r"--- a/(\S+)", patch)) | set(re.findall(r"\+\+\+ b/(\S+)", patch))
        file_overlap = len(gold_files & pred_files) / len(gold_files) if gold_files else (1.0 if not pred_files else 0.0)

        # Hunk content overlap (simple line-level)
        gold_lines = set(l.strip() for l in gold.splitlines() if l.strip().startswith(("+", "-")) and not l.startswith(("+++", "---")))
        pred_lines = set(l.strip() for l in patch.splitlines() if l.strip().startswith(("+", "-")) and not l.startswith(("+++", "---")))
        line_overlap = len(gold_lines & pred_lines) / len(gold_lines) if gold_lines else 0.0

        # Overall: weighted combination with format
        # High overlap -> likely tests would pass
        if line_overlap >= 0.9 and file_overlap >= 1.0 and format_score >= 0.75:
            score = 0.85
            verdict = "Patch closely matches gold (file + hunk overlap high) — proxy for tests pass, minor whitespace/context diff."
        elif line_overlap >= 0.5 and file_overlap >= 0.5:
            score = 0.5
            verdict = "Patch partially matches gold — touches right files/lines but differs — may pass some tests."
        elif file_overlap >= 0.5 and format_score >= 0.75:
            score = 0.25
            verdict = "Patch touches correct files and is well-formed but hunk content diverges — proxy for patch applies but tests likely fail."
        elif format_score >= 0.75:
            score = 0.1
            verdict = "Patch is well-formed and would apply, but targets wrong files/lines — proxy for tests fail."
        else:
            score = 0.0
            verdict = "Patch is malformed or targets wrong area — proxy for patch fails to apply."

        feedback_parts = [
            f"Instance {instance_id}: {verdict}",
            f"Format checks: {format_checks} ({format_pass}/{len(format_checks)})",
            f"File overlap with gold: {file_overlap:.2f} (gold files: {sorted(gold_files)[:5]}, pred files: {sorted(pred_files)[:5]})",
            f"Hunk line overlap: {line_overlap:.2f}",
            f"Proxy score: {score:.2f} (1 = patch applies + tests pass).",
            f"Gold patch (first 600 chars):\n{gold[:600]}",
            f"Your patch (first 600 chars):\n{patch[:600]}",
        ]
        # Degeneracy note: check for markdown fencing that would break `git apply`
        if "```" in raw_patch:
            feedback_parts.append("Note: output contains markdown fencing (```); `git apply` would fail — emit raw diff only.")
        return float(score), "\n".join(feedback_parts)

    # No gold: score on format alone (proxy)
    if format_score >= 0.75:
        feedback = (
            f"Instance {instance_id}: No gold patch available. Patch is well-formed ({format_pass}/{len(format_checks)} checks pass) — "
            f"proxy for patch applies. Score 0.6.\n"
            f"Checks: {format_checks}\n"
            f"Your patch (first 600 chars):\n{patch[:600]}"
        )
        score = 0.6
    elif format_score >= 0.5:
        feedback = (
            f"Instance {instance_id}: No gold patch. Patch partially formed ({format_pass}/{len(format_checks)}). Score 0.3.\n"
            f"Checks: {format_checks}"
        )
        score = 0.3
    else:
        feedback = (
            f"Instance {instance_id}: No gold patch. Patch poorly formed ({format_pass}/{len(format_checks)}). Score 0.\n"
            f"Checks: {format_checks}"
        )
        score = 0.0
    if "```" in raw_patch:
        feedback += "\nNote: markdown fencing (```) would break `git apply`."
    return float(score), feedback
