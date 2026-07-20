"""IFBench utilities: dataset loading, 2-stage LM program, and metric.

Replicates the GEPA paper's IFBench setup (gepa-ai/gepa-artifact,
gepa_artifact/benchmarks/IFBench): exact data splits, 2-stage program
structure, and the instruction-level accuracy metric with per-constraint
feedback. See ATTRIBUTION.md.
"""

import json
import os
import re
import urllib.request

import litellm

from examples.ifbench.utils_ifbench import instructions_registry

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Data files are not committed (IFBench_train.jsonl is ~16 MB); they are
# fetched on first use from the GEPA paper's artifact repository.
_DATA_BASE_URL = "https://raw.githubusercontent.com/gepa-ai/gepa-artifact/main/gepa_artifact/benchmarks/IFBench/data"
DATA_FILES = ["IFBench_train.jsonl", "IFBench_test.jsonl"]


def ensure_data_downloaded() -> None:
    """Download the IFBench data files into DATA_DIR if they are missing."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for name in DATA_FILES:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            continue
        url = f"{_DATA_BASE_URL}/{name}"
        print(f"Downloading {name} from {url} ...")
        tmp_path = path + ".part"
        urllib.request.urlretrieve(url, tmp_path)
        os.replace(tmp_path, path)

FINAL_RESPONSE_MARKER = "Final Response:"

# Appended to each stage's system prompt to emulate dspy.ChainOfThought:
# reason first, then emit the answer field after a fixed marker.
COT_FORMAT_INSTRUCTION = (
    "\n\nFirst reason step by step about how to best respond. Then write your "
    f"final response after a line containing exactly '{FINAL_RESPONSE_MARKER}'. "
    "Only the text after that line is used as your response."
)


def load_ifbench_dataset() -> tuple[list[dict], list[dict], list[dict]]:
    """Load IFBench with the GEPA paper's exact splits.

    Returns (trainset, valset, testset) as lists of dicts with keys:
    key, prompt, instruction_id_list, kwargs.
    Splits (from the artifact's ifbench_data.py):
    test = all 294 of IFBench_test.jsonl, val = IFBench_train.jsonl[:300],
    train = IFBench_train.jsonl[300:600].
    """
    import nltk

    nltk.download("punkt_tab", quiet=True)
    ensure_data_downloaded()

    def read_jsonl(name: str) -> list[dict]:
        records = []
        with open(os.path.join(DATA_DIR, name)) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    testset = read_jsonl("IFBench_test.jsonl")
    train_val = read_jsonl("IFBench_train.jsonl")
    trainset = train_val[300:600]
    valset = train_val[:300]
    return trainset, valset, testset


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
        # Paper decoding config for Qwen3 (temp=0.6, top-p=0.95).
        "temperature": 0.6,
        "top_p": 0.95,
        # 2048 keeps rollout latency manageable (thinking tokens + CoT + response);
        # IFBench's leniency checks make truncated rambling harmless.
        "max_tokens": 2048,
    }
    if api_base is not None:
        kwargs["api_base"] = api_base
    response = litellm.completion(**kwargs)
    return response.choices[0].message.content or ""


def run_two_stage(
    gen_prompt: str,
    ensure_prompt: str,
    query: str,
    model: str = "hosted_vllm/Qwen3.5-9B",
    api_base: str | None = None,
) -> tuple[str, str]:
    """Run the 2-stage IFBench program with plain LM calls.

    Stage 1 (generate_response): answer the query.
    Stage 2 (ensure_correct_response): rewrite the answer to satisfy the
    constraints; its output is the final response that gets scored.
    Returns (stage1_response, final_response).
    """
    stage1_out = _call_lm(gen_prompt + COT_FORMAT_INSTRUCTION, f"Query:\n{query}", model, api_base)
    response = _extract_final_response(stage1_out)

    stage2_user = f"Query:\n{query}\n\nResponse:\n{response}"
    stage2_out = _call_lm(ensure_prompt + COT_FORMAT_INSTRUCTION, stage2_user, model, api_base)
    final_response = _extract_final_response(stage2_out)

    return response, final_response


def ifbench_metric(response: str, example: dict) -> tuple[float, str]:
    """Score a response against an example's constraints.

    Direct port of the artifact's metric_with_feedback (ifbench_metric.py):
    instruction-level accuracy (fraction of constraints satisfied) with
    leniency over 8 response variants, plus feedback text listing which
    constraints were followed and which were violated.
    """
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]

    instruction_list = example["instruction_id_list"]
    is_following_list = []
    correct_feedbacks = []
    incorrect_feedbacks = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        instruction_kwargs = {k: v for k, v in example["kwargs"][index].items() if v is not None}

        ins_text = instruction.build_description(**instruction_kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            ins_text = instruction.build_description(prompt=example["prompt"])

        is_following = False
        for candidate_response in all_responses:
            if candidate_response.strip() and instruction.check_following(candidate_response):
                is_following = True
                break

        if not is_following:
            incorrect_feedbacks.append(ins_text)
        else:
            correct_feedbacks.append(ins_text)

        is_following_list.append(is_following)

    correct_feedback_text = ""
    if len(correct_feedbacks) > 0:
        correct_feedback_text = "Your response correctly followed the following instructions:\n" + "\n".join(
            correct_feedbacks
        )

    incorrect_feedback_text = ""
    if len(incorrect_feedbacks) > 0 and len(correct_feedbacks) > 0:
        incorrect_feedback_text = (
            "However, your response did not follow the following instructions properly:\n"
            + "\n".join(incorrect_feedbacks)
        )
    elif len(incorrect_feedbacks) > 0:
        incorrect_feedback_text = "Your response did not follow the following instructions properly:\n" + "\n".join(
            incorrect_feedbacks
        )

    feedback_text = (correct_feedback_text + "\n" + incorrect_feedback_text).strip()
    score = sum(is_following_list) / len(is_following_list)
    return score, feedback_text
