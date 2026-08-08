"""HoVer utilities: dataset loading, 2-stage LM program, and metric.

Replicates a HoVer-style setup (Jiang et al. 2020, many-hop fact extraction
& claim verification, up to 3 hops; hover-nlp.github.io). The paper's full
system retrieves over a BM25 index of 5.2M 2017 Wikipedia abstracts; here we
provide a lightweight, offline 2-stage LM program (query-writer -> doc-
summarizer) whose output is a list of predicted Wikipedia titles, scored by
gold-doc retrieval F1/recall. See ATTRIBUTION.md.

Metric is gold-doc retrieval: precision/recall/F1 over supporting doc titles.
"""

import json
import os
import random
import re
import urllib.request

import litellm

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# HoVer data is hosted in the official repo; HuggingFace `hover` is primary,
# raw GitHub is fallback (files not committed, like IFBench).
_DATA_BASE_URL = "https://raw.githubusercontent.com/hover-nlp/hover/main/data"
DATA_FILES = ["hover_train.json", "hover_dev.json"]


def ensure_data_downloaded() -> None:
    """Download HoVer data files into DATA_DIR if they are missing."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for name in DATA_FILES:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            continue
        url = f"{_DATA_BASE_URL}/{name}"
        print(f"Downloading {name} from {url} ...")
        tmp_path = path + ".part"
        try:
            urllib.request.urlretrieve(url, tmp_path)
            os.replace(tmp_path, path)
        except Exception as e:
            # Clean up partial and allow caller to fall back to HF datasets
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            print(f"WARNING: could not download {name}: {e}")


FINAL_RESPONSE_MARKER = "Final Response:"

# Appended to each stage's system prompt to emulate dspy.ChainOfThought:
# reason first, then emit the answer field after a fixed marker.
COT_FORMAT_INSTRUCTION = (
    "\n\nFirst reason step by step about how to best retrieve and verify. Then write your "
    f"final response after a line containing exactly '{FINAL_RESPONSE_MARKER}'. "
    "Only the text after that line is used as your response."
)


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
    # Long inputs can leave less than max_tokens of context headroom. Step the
    # output budget down before giving up; if the input alone overflows, return
    # "" so the rollout scores 0 instead of killing the run.
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
# Dataset loading
# ---------------------------------------------------------------------------

def _normalize_claim(item: dict) -> str:
    for key in ("claim", "prompt", "question", "input", "text"):
        if key in item and item[key]:
            return str(item[key])
    return ""


def _get_gold_titles(item: dict) -> list[str]:
    """Extract gold supporting doc titles from a HoVer item robustly."""
    # Direct title list
    for key in ("supporting_titles", "gold_titles", "supporting_titles_list", "titles"):
        if key in item and item[key]:
            val = item[key]
            if isinstance(val, list):
                return [str(t).strip() for t in val if str(t).strip()]
            if isinstance(val, str):
                # maybe JSON list or comma-separated
                try:
                    parsed = json.loads(val)
                    if isinstance(parsed, list):
                        return [str(t).strip() for t in parsed if str(t).strip()]
                except Exception:
                    pass
                # split on separators
                parts = re.split(r"[|,;\n]+", val)
                return [p.strip() for p in parts if p.strip()]

    # supporting_facts is list of [title, sent_id] or dicts
    facts = item.get("supporting_facts") or item.get("supporting_facts_list") or item.get("evidence") or item.get("docs")
    titles: list[str] = []
    if isinstance(facts, list):
        for f in facts:
            if isinstance(f, (list, tuple)) and len(f) >= 1:
                titles.append(str(f[0]).strip())
            elif isinstance(f, dict):
                for k in ("title", "doc_title", "page", "document"):
                    if k in f and f[k]:
                        titles.append(str(f[k]).strip())
                        break
            elif isinstance(f, str):
                # maybe "Title [sent]" format
                m = re.match(r"^(.*?)\s*\[", f)
                titles.append((m.group(1) if m else f).strip())
    # Deduplicate preserving order
    seen = set()
    uniq = []
    for t in titles:
        low = t.lower()
        if low not in seen and t:
            seen.add(low)
            uniq.append(t)
    if uniq:
        return uniq
    # Fallback: single doc field
    for key in ("document", "page", "title"):
        if key in item and item[key]:
            return [str(item[key]).strip()]
    return []


def _to_inst(item: dict) -> dict:
    claim = _normalize_claim(item)
    gold_titles = _get_gold_titles(item)
    # Preserve raw facts for metric debugging
    facts = item.get("supporting_facts") or item.get("evidence") or []
    label = str(item.get("label") or item.get("verification_label") or item.get("answer") or "")
    # Hotpot-style id
    inst_id = str(item.get("id") or item.get("uid") or item.get("claim_id") or "")
    hops = item.get("num_hops") or item.get("hops") or item.get("hop") or ""
    try:
        hops_int = int(hops) if hops != "" else 0
    except Exception:
        hops_int = 0
    return {
        "claim": claim,
        "prompt": claim,
        "question": claim,
        "supporting_facts": facts,
        "supporting_titles": gold_titles,
        "gold_titles": gold_titles,
        "label": label,
        "id": inst_id,
        "num_hops": hops_int,
        "raw": item,
        "answer": json.dumps(gold_titles),
        "additional_context": {
            "gold_titles": gold_titles,
            "supporting_facts": facts,
            "label": label,
            "num_hops": hops_int,
        },
    }


def load_hover_dataset(seed: int = 0) -> tuple[list[dict], list[dict], list[dict]]:
    """Load HoVer with paper-faithful splits 150/300/300.

    Primary source is HuggingFace `hover` (hover-nlp/hover). If unavailable
    (offline or missing), falls back to raw GitHub artifact JSON in DATA_DIR.
    Returns (trainset, valset, testset) shuffled with seed 0 and sliced to
    150 train / 300 val / 300 test (750 total), capping to available size.
    """
    data: list[dict] = []

    # Try HuggingFace datasets first
    try:
        from datasets import load_dataset

        # Try a few common dataset identifiers / configs
        last_err = None
        for ds_args in [
            ("hover", None),
            ("hover", "hover"),
            ("nlp-hover/hover", None),
        ]:
            try:
                name, config = ds_args
                if config is None:
                    raw_ds = load_dataset(name)
                else:
                    raw_ds = load_dataset(name, config)
                # raw_ds may be DatasetDict with train/validation/test or single split
                if hasattr(raw_ds, "keys"):
                    for split in ("train", "validation", "dev", "test"):
                        if split in raw_ds:
                            for item in raw_ds[split]:
                                data.append(item)
                            break
                    # If still empty but raw_ds is DatasetDict with one key
                    if not data:
                        first_key = list(raw_ds.keys())[0]
                        for item in raw_ds[first_key]:
                            data.append(item)
                else:
                    # raw_ds is a Dataset
                    for item in raw_ds:
                        data.append(item)
                if data:
                    break
            except Exception as e:
                last_err = e
                continue
        if not data and last_err is not None:
            raise last_err
    except Exception as e:
        print(f"INFO: HF datasets load failed ({e}); trying raw GitHub artifact...")

    # Fallback: raw GitHub JSON files
    if not data:
        ensure_data_downloaded()
        # Try multiple filename variants that the repo has used
        candidates = [
            os.path.join(DATA_DIR, "hover_train.json"),
            os.path.join(DATA_DIR, "hover_train_release_v1.1.json"),
            os.path.join(DATA_DIR, "train.json"),
        ] + [
            os.path.join(DATA_DIR, "hover_dev.json"),
            os.path.join(DATA_DIR, "hover_dev_release_v1.1.json"),
            os.path.join(DATA_DIR, "dev.json"),
            os.path.join(DATA_DIR, "validation.json"),
        ]
        loaded_any = False
        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                with open(path) as f:
                    content = json.load(f)
                # content may be list or dict with data key
                if isinstance(content, dict) and "data" in content:
                    content = content["data"]
                if isinstance(content, list):
                    data.extend(content)
                    loaded_any = True
                elif isinstance(content, dict):
                    # FEVER-style wrapped
                    for v in content.values():
                        if isinstance(v, list):
                            data.extend(v)
                    loaded_any = True
            except Exception as e:
                print(f"WARNING: failed to parse {path}: {e}")
                continue
        if not loaded_any and not data:
            # Last resort: try to read any jsonl in DATA_DIR
            for name in os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else []:
                if name.endswith(".jsonl"):
                    path = os.path.join(DATA_DIR, name)
                    try:
                        with open(path) as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    data.append(json.loads(line))
                    except Exception:
                        continue
        if not data:
            # Synthetic tiny fallback so offline py_compile / smoke tests still work
            print("WARNING: no HoVer data found; using synthetic fallback (3 examples).")
            data = [
                {"claim": "The Eiffel Tower is in Paris.", "supporting_facts": [["Eiffel Tower", 0]], "label": "SUPPORTED", "num_hops": 1, "id": "syn0"},
                {"claim": "Marie Curie won a Nobel Prize and was born in Warsaw.", "supporting_facts": [["Marie Curie", 0], ["Warsaw", 1]], "label": "SUPPORTED", "num_hops": 2, "id": "syn1"},
                {"claim": "The author of 1984 also wrote Animal Farm and was born in India.", "supporting_facts": [["George Orwell", 0], ["Animal Farm", 1], ["British India", 2]], "label": "SUPPORTED", "num_hops": 3, "id": "syn2"},
            ] * 250  # 750 synthetic

    # Normalize and shuffle
    normalized = [_to_inst(item) for item in data if _normalize_claim(item)]
    # Filter out empty claims
    normalized = [d for d in normalized if d["claim"].strip()]
    if not normalized:
        raise ValueError("No valid HoVer examples found after normalization.")

    rng = random.Random(seed)
    rng.shuffle(normalized)

    # Paper splits intent: 150 train / 300 val / 300 test (up to 3 hops)
    # Cap to available; if <750, split proportionally but keep ratios.
    if len(normalized) >= 750:
        trainset = normalized[:150]
        valset = normalized[150:450]
        testset = normalized[450:750]
    elif len(normalized) >= 450:
        trainset = normalized[:150]
        valset = normalized[150:350] if len(normalized) >= 350 else normalized[150:]
        testset = normalized[350:650] if len(normalized) >= 650 else normalized[350:]
        # Ensure at least some test
        if not testset:
            mid = len(normalized) // 2
            valset = normalized[:mid]
            testset = normalized[mid:]
            trainset = normalized[:150]
    else:
        mid = len(normalized) // 2
        trainset = normalized[: min(150, mid)]
        valset = normalized[mid: mid + 300] if len(normalized) > mid else normalized[: mid]
        # hold out remainder for test, else reuse
        remaining = [d for d in normalized if d not in trainset and d not in valset]
        testset = remaining[:300] if remaining else normalized[: min(300, len(normalized))]

    return trainset, valset, testset


# ---------------------------------------------------------------------------
# 2-stage program: query-writer -> doc-summarizer
# ---------------------------------------------------------------------------

def run_two_stage(
    query_writer_prompt: str,
    doc_summary_prompt: str,
    claim: str,
    model: str = "hosted_vllm/Qwen3.5-9B",
    api_base: str | None = None,
) -> tuple[str, str]:
    """Run the 2-stage HoVer program with plain LM calls.

    Stage 1 (query_writer): generate search queries for the claim.
    Stage 2 (doc_summarizer): given claim + queries, list supporting Wikipedia
    titles. Its output is the final retrieval prediction that gets scored.
    Returns (stage1_queries, final_titles_text).
    """
    stage1_out = _call_lm(query_writer_prompt + COT_FORMAT_INSTRUCTION, f"Claim:\n{claim}", model, api_base)
    queries = _extract_final_response(stage1_out)

    # Cap the stage-1 text fed into stage 2 so query + claim + output budget
    # always fits the model context (32k for Qwen3-8B). ~24k chars ~ 6k tokens.
    queries_capped = queries if len(queries) <= 24000 else queries[:24000] + "\n[truncated]"
    stage2_user = f"Claim:\n{claim}\n\nQueries:\n{queries_capped}"
    stage2_out = _call_lm(doc_summary_prompt + COT_FORMAT_INSTRUCTION, stage2_user, model, api_base)
    final_titles = _extract_final_response(stage2_out)

    return queries, final_titles


def run_single_stage(
    prompt: str,
    claim: str,
    model: str = "hosted_vllm/Qwen3.5-9B",
    api_base: str | None = None,
) -> str:
    """Run the 1-stage HoVer program: one optimized prompt, one LM call.

    Ablation variant of run_two_stage (the 2-stage is the paper protocol).
    Returns the predicted titles text.
    """
    out = _call_lm(prompt + COT_FORMAT_INSTRUCTION, f"Claim:\n{claim}", model, api_base)
    return _extract_final_response(out)


# ---------------------------------------------------------------------------
# Metric: gold-doc retrieval F1 / recall
# ---------------------------------------------------------------------------

def _normalize_title(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def _extract_predicted_titles(text: str) -> list[str]:
    """Extract predicted Wikipedia titles from LM output.

    Tries JSON list, bracketed list, then bullet/numbered lines, then fallback
    to comma/semicolon split. Returns deduplicated titles preserving order.
    """
    text = text.strip()
    if not text:
        return []

    # Try JSON array anywhere in text
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list) and parsed and all(isinstance(x, str) for x in parsed):
                out = [p.strip() for p in parsed if p.strip()]
                if out:
                    return _dedup_titles(out)
        except Exception:
            pass
        # Try single-quoted variant
        try:
            cand = m.group(0).replace("'", '"')
            parsed = json.loads(cand)
            if isinstance(parsed, list) and parsed and all(isinstance(x, str) for x in parsed):
                out = [p.strip() for p in parsed if p.strip()]
                if out:
                    return _dedup_titles(out)
        except Exception:
            pass

    # Bullet / numbered list lines
    lines = text.splitlines()
    bullet_titles = []
    for line in lines:
        s = line.strip()
        # Match "- Title", "* Title", "1. Title", "1) Title", "[1] Title"
        bm = re.match(r"^(?:[-*•]|\d+[\.\)\]]|\(\d+\)|\[\d+\])\s+(.*)", s)
        if bm:
            cand = bm.group(1).strip().rstrip(",;.")
            # Remove markdown bold
            cand = re.sub(r"\*\*(.*?)\*\*", r"\1", cand)
            if cand and len(cand) <= 200:
                bullet_titles.append(cand)
    if len(bullet_titles) >= 1:
        # Filter out obvious non-titles (too long, sentences)
        bullet_titles = [t for t in bullet_titles if 2 <= len(t.split()) <= 8 or len(t) < 80]
        if bullet_titles:
            return _dedup_titles(bullet_titles)

    # Comma / semicolon / newline separated (if text is short list)
    if len(text) < 800 and ("," in text or ";" in text or "\n" in text):
        # Only if looks like a title list (no long sentences)
        avg_len = len(text.split(",")) if "," in text else len(text.split("\n"))
        if avg_len <= 10 and all(len(p.strip().split()) <= 10 for p in re.split(r"[,;\n]+", text) if p.strip()):
            parts = [p.strip().rstrip(".") for p in re.split(r"[,;\n]+", text)]
            parts = [p for p in parts if p and len(p) < 100]
            if 1 <= len(parts) <= 10:
                return _dedup_titles(parts)

    # No structured titles found -> return empty to trigger substring recall fallback
    return []


def _dedup_titles(titles: list[str]) -> list[str]:
    seen = set()
    out = []
    for t in titles:
        low = _normalize_title(t)
        if low and low not in seen:
            seen.add(low)
            out.append(t.strip())
    return out


def _get_gold_titles_for_metric(example: dict) -> list[str]:
    # Prefer normalized gold_titles, fallback to supporting_titles, or parse answer
    for key in ("gold_titles", "supporting_titles", "titles"):
        if key in example and example[key]:
            val = example[key]
            if isinstance(val, list) and val:
                return [str(t).strip() for t in val if str(t).strip()]
    # Try answer JSON
    ans = example.get("answer")
    if ans:
        try:
            parsed = json.loads(ans)
            if isinstance(parsed, list):
                return [str(t).strip() for t in parsed if str(t).strip()]
        except Exception:
            pass
    # Fallback to additional_context
    ctx = example.get("additional_context", {})
    if ctx.get("gold_titles"):
        return [str(t).strip() for t in ctx["gold_titles"]]
    return []


def hover_metric(prediction: str, example: dict) -> tuple[float, str]:
    """Score a response by gold-doc retrieval F1/recall.

    If the prediction yields parseable titles, computes set-based
    precision/recall/F1 (case-insensitive, whitespace-normalized). If no
    parseable titles are found, falls back to substring recall (each gold title
    lower in prediction lower) and scores by recall (0..1).

    Returns (score, feedback) where score is F1 (or recall fallback) and
    feedback lists retrieved / missing and precision/recall/F1.
    """
    gold_titles = _get_gold_titles_for_metric(example)
    if not gold_titles:
        return 0.0, "No gold titles for this example."

    pred_titles = _extract_predicted_titles(prediction)

    if pred_titles:
        gold_set = {_normalize_title(t) for t in gold_titles}
        pred_set = {_normalize_title(t) for t in pred_titles}
        tp = len(gold_set & pred_set)
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(gold_set) if gold_set else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        score = f1
        retrieved = [t for t in gold_titles if _normalize_title(t) in pred_set]
        missing = [t for t in gold_titles if _normalize_title(t) not in pred_set]
        extra = [t for t in pred_titles if _normalize_title(t) not in gold_set]
        fb = (
            f"Gold docs ({len(gold_titles)}): {', '.join(gold_titles)}\n"
            f"Predicted ({len(pred_titles)}): {', '.join(pred_titles)}\n"
            f"Retrieved {len(retrieved)}/{len(gold_titles)}: {', '.join(retrieved) if retrieved else '(none)'}\n"
            f"Missing: {', '.join(missing) if missing else '(none)'}\n"
            f"Extra: {', '.join(extra) if extra else '(none)'}\n"
            f"Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}"
        )
        return score, fb
    else:
        # Substring recall fallback: case-insensitive containment
        pred_low = prediction.lower()
        retrieved = [t for t in gold_titles if _normalize_title(t) in pred_low]
        recall = len(retrieved) / len(gold_titles) if gold_titles else 0.0
        # For fallback, precision is not meaningful; use recall as score
        score = recall
        fb = (
            f"Gold docs ({len(gold_titles)}): {', '.join(gold_titles)}\n"
            f"Prediction (no parseable title list; substring match):\n{prediction[:600]}\n"
            f"Retrieved {len(retrieved)}/{len(gold_titles)} by substring: {', '.join(retrieved) if retrieved else '(none)'}\n"
            f"Missing: {', '.join([t for t in gold_titles if t not in retrieved]) if len(retrieved) < len(gold_titles) else '(none)'}\n"
            f"Recall (substring)={recall:.3f} (used as F1 fallback)"
        )
        return score, fb


def hover_recall(prediction: str, example: dict) -> float:
    """Convenience: recall only."""
    gold_titles = _get_gold_titles_for_metric(example)
    if not gold_titles:
        return 0.0
    pred_titles = _extract_predicted_titles(prediction)
    if pred_titles:
        gold_set = {_normalize_title(t) for t in gold_titles}
        pred_set = {_normalize_title(t) for t in pred_titles}
        tp = len(gold_set & pred_set)
        return tp / len(gold_set) if gold_set else 0.0
    # substring fallback
    pred_low = prediction.lower()
    retrieved = sum(1 for t in gold_titles if _normalize_title(t) in pred_low)
    return retrieved / len(gold_titles) if gold_titles else 0.0
