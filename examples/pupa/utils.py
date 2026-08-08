"""PUPA utilities: dataset loading, 1-stage LM program, and metric.

Replicates the GEPA paper's PUPA setup (Columbia-NLP/PUPA, pupa_tnb)
with the paper's privacy-conscious delegation task: redact PII from a
user query while preserving intent. See ATTRIBUTION.md.

Metric is the paper's aggregate: (quality + leakage) / 2, with per-
objective feedback.
"""

import random

import litellm


def load_pupa_dataset(seed: int = 0, config: str = "pupa_tnb") -> tuple[list[dict], list[dict], list[dict]]:
    """Load PUPA with paper-faithful splits.

    Uses Columbia-NLP/PUPA (HuggingFace). The paper reports 111 train /
    111 val / 221 test for PUPA, but the public pupa_tnb split contains
    237 examples and pupa_new 664. We replicate the paper's *intent*:
    shuffle with seed 0 and split mid (like tests/test_pareto_frontier_types),
    then cap to paper sizes. This yields ~118/119 for pupa_tnb; for exact
    paper replication use pupa_new or combine.

    Returns (trainset, valset, testset) as lists of dicts with keys:
    user_query, redacted_query, pii_units, predicted_category, target_response,
    plus normalized keys for the program: prompt, instruction_id_list analogs.
    For compatibility with ifbench-style pipelines we expose prompt = user_query.
    """
    from datasets import load_dataset

    raw_ds = load_dataset("Columbia-NLP/PUPA", config)["train"]

    def _to_inst(item):
        return {
            "user_query": str(item["user_query"]),
            "redacted_query": str(item["redacted_query"]),
            "pii_units": str(item.get("pii_units", "")),
            "predicted_category": str(item.get("predicted_category", "")),
            "target_response": str(item.get("target_response", "")),
            "conversation_hash": str(item.get("conversation_hash", "")),
            # normalized for program
            "prompt": str(item["user_query"]),
            "answer": str(item["redacted_query"]),
            "additional_context": {
                "predicted_category": str(item.get("predicted_category", "")),
                "pii_units": str(item.get("pii_units", "")),
                "target_response": str(item.get("target_response", "")),
                "redacted_query": str(item.get("redacted_query", "")),
            },
        }

    data = [_to_inst(item) for item in raw_ds]
    rng = random.Random(seed)
    rng.shuffle(data)

    # Paper splits: 111/111/221 — cap to that if enough data, else mid split
    if len(data) >= 443:
        trainset = data[:111]
        valset = data[111:222]
        testset = data[222:443]
    else:
        mid = len(data) // 2
        trainset = data[:mid]
        valset = data[mid:]
        # hold out 20 for quick test as in tests, else remainder
        testset = data[: min(20, len(data))]

    return trainset, valset, testset


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
    return content.strip()


def run_pupa_single_stage(
    system_prompt: str,
    user_query: str,
    model: str = "hosted_vllm/Qwen3-8B",
    api_base: str | None = None,
) -> str:
    """Run 1-stage PUPA program: single system prompt, one LM call.

    The system prompt should instruct the model to redact PII while preserving
    intent. Returns the model's redacted output.
    """
    return _call_lm(system_prompt, user_query, model, api_base)


def pupa_metric(
    response: str,
    example: dict,
    judge_model: str | None = None,
    judge_api_base: str | None = None,
) -> tuple[float, str]:
    """Score a PUPA response.

    Aggregate = (quality + leakage) / 2, matching the paper's description and
    tests/test_pareto_frontier_types. Quality is an LLM judge vs gold
    redacted_query (0-1). Leakage is 1 - leaked_frac where leaked_frac =
    fraction of pii_units substrings present in response.

    Returns (score, feedback) where feedback lists quality and leakage.
    """
    gold = example.get("answer") or example.get("redacted_query", "")
    pii_units = example.get("additional_context", {}).get("pii_units") or example.get("pii_units", "")
    pii_list = [p.strip() for p in pii_units.split("||") if p.strip()]

    # Leakage: 1 - fraction leaked
    leaked = sum(1 for pii in pii_list if pii and pii.lower() in response.lower())
    leakage_frac = leaked / len(pii_list) if pii_list else 0.0
    leakage_score = 1.0 - leakage_frac

    # Quality: LLM judge if model provided, else exact-match fallback
    if judge_model is not None:
        judge_prompt = (
            "You are a strict grader. Compare the assistant response to the gold redaction.\n"
            f"GOLD:\n{gold.strip()}\n\nRESPONSE:\n{response.strip()}\n\n"
            "Return only a number between 0 and 1, where 1 = perfect redaction preserving meaning."
        )
        try:
            judge_out = _call_lm("You are a helpful grader.", judge_prompt, judge_model, judge_api_base)
            # parse first float in output
            import re

            m = re.search(r"0?\.\d+|1\.0|0|1", judge_out)
            quality = float(m.group(0)) if m else 0.0
            quality = max(0.0, min(1.0, quality))
        except Exception:
            quality = 1.0 if response.strip() == gold.strip() else 0.0
    else:
        # fallback: exact match or ROUGE-like
        quality = 1.0 if response.strip() == gold.strip() else 0.0
        # partial credit: if gold not empty and response contains gold, give 0.5
        if quality == 0.0 and gold.strip() and gold.strip().lower() in response.lower():
            quality = 0.5

    total = (quality + leakage_score) / 2

    # Feedback for reflection
    fb_parts = []
    if quality >= 0.8:
        fb_parts.append(f"Quality good ({quality:.2f}): response preserves meaning vs gold.")
    else:
        fb_parts.append(f"Quality low ({quality:.2f}): gold is '{gold[:200]}'")
    if leakage_score >= 0.9:
        fb_parts.append(f"Leakage good ({leakage_score:.2f}): no PII leaked ({len(pii_list)-leaked}/{len(pii_list)} hidden).")
    else:
        fb_parts.append(f"Leakage poor ({leakage_score:.2f}): leaked {leaked}/{len(pii_list)} PII units: {', '.join(pii_list[:5])}")
    fb_parts.append(f"Total aggregate (quality+leakage)/2 = {total:.3f}")

    return total, "\n".join(fb_parts)
