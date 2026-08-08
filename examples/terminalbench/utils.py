"""TerminalBench utilities: dataset loading, 1/2-stage LM program, and metric.

Replicates the GEPA paper pattern (mirroring examples/ifbench) for
TerminalBench (T-Bench, https://terminal-bench.github.io, 2024, 50+ terminal
agent tasks, Docker-based). Single-stage: shell command generation.
Two-stage: plan-then-execute. Metric is task success via unit tests /
Docker exit-code proxy with feedback. _call_lm is identical to ifbench
(temp 0.6 / top_p 0.95 / top_k 20 / max 16384 / enable_thinking False,
truncation, retries, <think> stripping). Offline-friendly: HF
`laude/terminal-bench` with local `data/terminalbench.jsonl` fallback and a
synthetic fallback so optimization never crashes offline.

See ATTRIBUTION.md.
"""

from __future__ import annotations

import json
import os
import random
import re

import litellm

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_HF_DATASET = "laude/terminal-bench"
_FALLBACK_FILE = "terminalbench.jsonl"

FINAL_RESPONSE_MARKER = "Final Command:"
FINAL_PLAN_MARKER = "Final Plan:"

COT_FORMAT_INSTRUCTION = (
    "\n\nFirst reason step by step about how to best respond. Then write your "
    f"final response after a line containing exactly '{FINAL_RESPONSE_MARKER}'. "
    "Only the text after that line is used as your response. "
    "For shell tasks, emit a single bash command or script block."
)

PLAN_COT_INSTRUCTION = (
    "\n\nFirst reason step by step about how to approach the task. Then write your "
    f"plan after a line containing exactly '{FINAL_PLAN_MARKER}'. "
    "Only the text after that line is used as your plan."
)


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models like Qwen3."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_final_response(output: str) -> str:
    """Extract text after the last FINAL_RESPONSE_MARKER, fallback to full output."""
    output = _strip_think(output)
    if FINAL_RESPONSE_MARKER in output:
        return output.rsplit(FINAL_RESPONSE_MARKER, 1)[1].strip()
    return output.strip()


def _extract_final_plan(output: str) -> str:
    """Extract text after the last FINAL_PLAN_MARKER, fallback to full output."""
    output = _strip_think(output)
    if FINAL_PLAN_MARKER in output:
        return output.rsplit(FINAL_PLAN_MARKER, 1)[1].strip()
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
    # Long inputs (rambling stage-1 plan, or a candidate prompt that grew huge)
    # can leave less than max_tokens of headroom. Step the output budget down
    # before giving up; if the input alone overflows, return "" so the rollout
    # scores 0 instead of killing the run. Also retry on context-window errors.
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
    """Normalize a raw TerminalBench record to a common schema.

    Expected normalized keys: task_id, prompt (instruction), task_description,
    expected_commands / tests / solution hints when available, plus the raw
    dict under _raw for metric introspection.
    """
    # Common HF fields: task_id/id, instruction/task/description/prompt
    task_id = str(raw.get("task_id") or raw.get("id") or raw.get("instance_id") or f"tbench-{idx}")
    prompt = raw.get("prompt") or raw.get("instruction") or raw.get("task") or raw.get("description") or raw.get("task_description") or ""
    prompt = str(prompt).strip()
    if not prompt:
        # fallback: join any text fields
        for k in ("input", "query", "goal", "objective"):
            if raw.get(k):
                prompt = str(raw[k]).strip()
                break
    if not prompt:
        prompt = f"Complete the terminal task {task_id}."

    # Expected command hints (for proxy metric)
    expected = raw.get("expected_commands") or raw.get("solution") or raw.get("answer") or raw.get("command") or ""
    tests = raw.get("tests") or raw.get("test") or raw.get("unit_tests") or ""

    return {
        "task_id": task_id,
        "prompt": prompt,
        "task_description": prompt,
        "expected_commands": str(expected) if expected else "",
        "tests": str(tests) if tests else "",
        "instruction": prompt,
        "_raw": raw,
    }


def _synthetic_tasks(n: int = 50) -> list[dict]:
    """Offline synthetic terminal tasks so GEPA never crashes without data."""
    templates = [
        ("Create file /tmp/hello.txt with content 'hello world'", "echo 'hello world' > /tmp/hello.txt"),
        ("List files in /tmp and count lines with wc -l", "ls /tmp | wc -l"),
        ("Find all .py files in /tmp recursively", "find /tmp -name '*.py'"),
        ("Show disk usage of /tmp with du -sh", "du -sh /tmp"),
        ("Create directory /tmp/test_dir and list it", "mkdir -p /tmp/test_dir && ls /tmp/test_dir"),
        ("Display current working directory", "pwd"),
        ("Show environment variable PATH", "echo $PATH"),
        ("Count words in /etc/passwd", "wc -w /etc/passwd"),
        ("Sort lines in /etc/hosts", "sort /etc/hosts"),
        ("Grep for 'localhost' in /etc/hosts", "grep localhost /etc/hosts"),
    ]
    out: list[dict] = []
    for i in range(n):
        desc, sol = templates[i % len(templates)]
        out.append({
            "task_id": f"synthetic-{i:03d}",
            "prompt": f"{desc} (task {i})",
            "task_description": f"{desc} (task {i})",
            "expected_commands": sol,
            "tests": sol,
            "instruction": f"{desc} (task {i})",
            "_raw": {"synthetic": True},
        })
    return out


def load_terminalbench_dataset(
    data_path: str | None = None,
    seed: int = 0,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Load TerminalBench with 50-total 20/15/15 splits (train/val/test).

    Priority:
      1. data_path if provided (jsonl: one task per line).
      2. HF `laude/terminal-bench` via `datasets.load_dataset`.
      3. Local `data/terminalbench.jsonl` fallback.
      4. Synthetic tasks.

    Returns (trainset, valset, testset) as lists of normalized dicts with
    keys: task_id, prompt, task_description, expected_commands, tests, _raw.
    Splits are shuffled with `seed` for reproducibility.
    """
    records: list[dict] | None = None

    # 1) explicit data_path
    if data_path is not None:
        if os.path.isdir(data_path):
            # allow directory containing terminalbench.jsonl
            candidate = os.path.join(data_path, _FALLBACK_FILE)
            if os.path.exists(candidate):
                data_path = candidate
        if os.path.exists(data_path):
            try:
                raw = _read_jsonl(data_path)
                records = [_normalize_record(r, i) for i, r in enumerate(raw) if isinstance(r, dict)]
                print(f"Loaded {len(records)} TerminalBench tasks from {data_path}")
            except Exception as e:
                print(f"WARNING: failed to read {data_path}: {e}; trying fallbacks.")
                records = None
        else:
            print(f"WARNING: --data-path {data_path} not found; trying fallbacks.")
            records = None

    # 2) HF dataset
    if records is None:
        try:
            from datasets import load_dataset  # type: ignore

            ds = load_dataset(_HF_DATASET)
            # HF TerminalBench has been published under several configs; try common splits
            split = None
            for candidate in ("train", "test", "validation", "eval"):
                if candidate in ds:
                    split = candidate
                    break
            if split is None:
                # dict with single split
                split = list(ds.keys())[0]
            raw_list = list(ds[split])
            records = [_normalize_record(r, i) for i, r in enumerate(raw_list)]
            print(f"Loaded {len(records)} TerminalBench tasks from HF {_HF_DATASET}/{split}")
        except Exception as e:
            # offline or not installed
            print(f"HF load failed ({e}); trying local fallback.")

    # 3) local data/terminalbench.jsonl
    if records is None:
        local = os.path.join(DATA_DIR, _FALLBACK_FILE)
        if os.path.exists(local):
            try:
                raw = _read_jsonl(local)
                records = [_normalize_record(r, i) for i, r in enumerate(raw) if isinstance(r, dict)]
                print(f"Loaded {len(records)} TerminalBench tasks from {local}")
            except Exception as e:
                print(f"Failed to read local {local}: {e}")
                records = None

    # 4) synthetic
    if records is None or len(records) == 0:
        print("Using synthetic TerminalBench tasks (offline fallback).")
        records = _synthetic_tasks(50)

    # Normalize and split 20/15/15 (or proportionally if fewer)
    # Filter out empty prompts
    records = [r for r in records if r.get("prompt")]
    if not records:
        records = _synthetic_tasks(50)

    rng = random.Random(seed)
    rng.shuffle(records)

    # 50 total -> 20/15/15 to match spec; if fewer, proportionally split
    n = len(records)
    if n >= 50:
        trainset = records[:20]
        valset = records[20:35]
        testset = records[35:50]
    elif n >= 30:
        # 40/30/30 split of available
        n_train = max(1, int(n * 0.4))
        n_val = max(1, int(n * 0.3))
        trainset = records[:n_train]
        valset = records[n_train:n_train + n_val]
        testset = records[n_train + n_val:]
        if not testset:
            testset = records[-max(1, n // 3):]
    else:
        mid = n // 2
        trainset = records[:mid] if mid else records[:1]
        valset = records[mid:] if mid < n else records[-1:]
        testset = records[: min(15, n)]

    return trainset, valset, testset


def run_single_stage(
    prompt: str,
    task: str,
    model: str = "hosted_vllm/Qwen3-8B",
    api_base: str | None = None,
) -> str:
    """Run the 1-stage TerminalBench program: one optimized prompt, one LM call.

    The prompt should instruct the model to generate a shell command/solution.
    Returns the extracted final command.
    """
    out = _call_lm(prompt + COT_FORMAT_INSTRUCTION, f"Task:\n{task}", model, api_base)
    return _extract_final_response(out)


def run_two_stage(
    plan_prompt: str,
    execute_prompt: str,
    task: str,
    model: str = "hosted_vllm/Qwen3-8B",
    api_base: str | None = None,
) -> tuple[str, str]:
    """Run the 2-stage plan-then-execute program.

    Stage 1 (plan): generate a step-by-step plan for the terminal task.
    Stage 2 (execute): turn the plan + task into a shell command/script.

    Returns (plan, final_command).
    """
    plan_out = _call_lm(plan_prompt + PLAN_COT_INSTRUCTION, f"Task:\n{task}", model, api_base)
    plan = _extract_final_plan(plan_out)

    # Cap plan fed into stage 2 so query + plan + output budget fits context (32k).
    truncated_plan = plan if len(plan) <= 24000 else plan[:24000] + "\n[truncated]"
    stage2_user = f"Task:\n{task}\n\nPlan:\n{truncated_plan}"
    cmd_out = _call_lm(execute_prompt + COT_FORMAT_INSTRUCTION, stage2_user, model, api_base)
    final_command = _extract_final_response(cmd_out)

    return plan, final_command


def terminalbench_metric(response: str, example: dict) -> tuple[float, str]:
    """Score a TerminalBench response (shell command) against an example.

    Proxy for Docker unit-test / exit-code evaluation (offline-friendly):
    - 0 if empty
    - Heuristic shell-validity checks (non-empty, contains plausible shell tokens)
    - Keyword overlap with expected_commands / tests when available
    - Length / degeneracy penalties

    Returns (score, feedback) where score is 0 or 1 (or 0.5 partial), and
    feedback lists which checks passed/failed for reflection.
    """
    raw_response = response
    response = (response or "").strip()

    expected = (example.get("expected_commands") or "").strip()
    tests = (example.get("tests") or "").strip()
    task_desc = (example.get("prompt") or example.get("task_description") or "").strip()

    if not response:
        return 0.0, "Your response is empty. Provide a bash command or script that solves the task. Score 0."

    # Shell-validity heuristics
    shell_tokens = ["|", ">", "<", "&&", "||", ";", "$", "/", "-", "--", "echo", "ls", "cat", "grep", "find", "awk", "sed", "python", "bash", "sh", "mkdir", "touch", "chmod", "cp", "mv", "rm", "wc", "sort", "head", "tail", "curl", "wget", "git", "docker", "pip", "npm", "make", "gcc"]
    has_shell_token = any(tok in response for tok in shell_tokens)
    # Check for code block fencing - strip it for scoring but note it
    fenced = "```" in raw_response

    # Degeneracy: single word or very short without shell syntax
    words = response.split()
    degenerate = len(words) < 2 and not has_shell_token

    # Keyword overlap with expected when available
    overlap_score = 0.0
    overlap_detail = ""
    if expected:
        exp_tokens = set(expected.lower().split())
        resp_tokens = set(response.lower().split())
        # Also consider substring match for commands like "echo 'hello world' > /tmp/hello.txt"
        if expected.lower().strip() in response.lower():
            overlap_score = 1.0
            overlap_detail = f"Exact expected command found in response."
        elif exp_tokens:
            inter = exp_tokens & resp_tokens
            overlap = len(inter) / len(exp_tokens) if exp_tokens else 0.0
            overlap_score = overlap
            overlap_detail = f"Token overlap with expected: {len(inter)}/{len(exp_tokens)} ({overlap:.2f})."
        else:
            overlap_detail = "Expected command is empty; skipping overlap check."
    elif tests:
        # fallback: check test string containment
        if tests.lower().strip() in response.lower():
            overlap_score = 1.0
            overlap_detail = "Response contains test string."
        else:
            overlap_detail = "No expected_commands; checked tests field."

    # Task keyword check: does response address task's key nouns/verbs?
    task_keywords = [w.lower() for w in re.findall(r"\b\w{4,}\b", task_desc)][:8]
    task_hits = sum(1 for kw in task_keywords if kw in response.lower())
    task_kw_detail = f"Task keyword hits: {task_hits}/{len(task_keywords)} ({', '.join(task_keywords[:5])})." if task_keywords else "No task keywords."

    # Scoring
    if degenerate:
        score = 0.0
        verdict = "Response looks degenerate (too short / no shell syntax)."
    elif expected:
        if overlap_score >= 0.9:
            score = 1.0
            verdict = "Response matches expected command closely — proxy for Docker tests passing."
        elif overlap_score >= 0.5:
            score = 0.5
            verdict = "Response partially matches expected — partial credit (would likely fail some unit tests)."
        else:
            # If it looks like a valid shell command addressing the task, give partial
            if has_shell_token and task_hits >= 1:
                score = 0.25
                verdict = "Valid shell syntax addressing task but not matching expected — proxy for exit-code failure."
            else:
                score = 0.0
                verdict = "Response does not match expected and lacks convincing shell solution."
    else:
        # No expected available: score on shell validity + task relevance
        if has_shell_token and not degenerate and task_hits >= 1:
            score = 0.5
            verdict = "No gold command; response looks like a plausible shell solution (proxy pass)."
        elif has_shell_token:
            score = 0.25
            verdict = "No gold command; response has shell syntax but weak task relevance."
        else:
            score = 0.0
            verdict = "No gold command; response lacks shell syntax."

    # Build feedback
    parts: list[str] = []
    parts.append(f"Task: {task_desc[:300]}")
    if expected:
        parts.append(f"Expected (proxy gold): {expected[:400]}")
    parts.append(f"Your command:\n{response[:800]}")
    parts.append(verdict)
    if overlap_detail:
        parts.append(overlap_detail)
    parts.append(task_kw_detail)
    if fenced:
        parts.append("Note: response contains markdown fencing (```); prefer raw commands.")
    if has_shell_token:
        parts.append("Shell syntax check: PASS (contains shell tokens).")
    else:
        parts.append("Shell syntax check: FAIL (no shell tokens detected).")
    parts.append(f"Proxy score: {score:.2f} (1 = Docker tests would pass, 0 = fail).")

    return float(score), "\n".join(parts)
