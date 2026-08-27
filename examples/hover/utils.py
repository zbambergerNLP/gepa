"""HoVer v1.1 data, Wikipedia-backed retrieval program, and document metric."""

import hashlib
import json
import os
import random
import re
import string
import urllib.request
from pathlib import Path
from typing import Sequence, cast

import litellm  # type: ignore[import-not-found]

from examples.common.experiment_models import (
    EXPERIMENT_NUM_RETRIES,
    QWEN3_8_27B_MODEL,
    experiment_decoding,
    experiment_request_overrides,
)
from examples.common.wikipedia import WikipediaPassage, WikipediaRetriever

DATA_DIR = Path(__file__).parent / "data"
HOVER_TRAIN_FILE = "hover_train_release_v1.1.json"
HOVER_HF_REVISION = "c0e43052759879b3461642ca6c0dd26658f47691"
HOVER_SOURCE_REVISION = "39b84697f196308f398a251a7aea9b82ae0f0562"
HOVER_TRAIN_URL = (
    f"https://raw.githubusercontent.com/hover-nlp/hover/{HOVER_SOURCE_REVISION}/data/hover/{HOVER_TRAIN_FILE}"
)
HOVER_TRAIN_SHA256 = "1f1cd57abd616fa00c70bdc575ce77c16fc6cf1a6cffd5ff87c208030a336bb6"
HOVER_TRAIN_SIZE = 9_205_582
HOVER_ELIGIBLE_COUNT = 6_084
FINAL_RESPONSE_MARKER = "Final Response:"
_SMOKE_RECORDS = (
    {
        "uid": "smoke-train",
        "claim": "Marie Curie was born in the capital of Poland and won the Nobel Prize in Physics.",
        "supporting_facts": [["Marie Curie", 0], ["Warsaw", 0], ["Nobel Prize in Physics", 0]],
        "label": "SUPPORTED",
        "num_hops": 3,
    },
    {
        "uid": "smoke-val",
        "claim": "George Orwell wrote two novels and was born in British India.",
        "supporting_facts": [["George Orwell", 0], ["Animal Farm", 0], ["Nineteen Eighty-Four", 0]],
        "label": "SUPPORTED",
        "num_hops": 3,
    },
    {
        "uid": "smoke-test",
        "claim": "The Eiffel Tower stands in the capital of France beside the Seine.",
        "supporting_facts": [["Eiffel Tower", 0], ["Paris", 0], ["Seine", 0]],
        "label": "SUPPORTED",
        "num_hops": 3,
    },
)


def _file_sha256(path: Path) -> str:
    """Hash a local artifact without loading it entirely into memory.

    Args:
        path: File whose bytes should be hashed.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_data_downloaded(data_dir: str | os.PathLike[str] = DATA_DIR) -> Path:
    """Download the official HoVer v1.1 training release when it is missing.

    Args:
        data_dir: Directory containing or receiving the release file.

    Returns:
        Path to the complete local training release.

    Raises:
        RuntimeError: The release cannot be downloaded or installed atomically.
    """
    destination_dir = Path(data_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / HOVER_TRAIN_FILE
    if destination.is_file():
        if destination.stat().st_size != HOVER_TRAIN_SIZE or _file_sha256(destination) != HOVER_TRAIN_SHA256:
            raise RuntimeError(
                f"Existing HoVer release at {destination} does not match the pinned v1.1 artifact. "
                "Remove or replace that file before preparing the benchmark."
            )
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        urllib.request.urlretrieve(HOVER_TRAIN_URL, partial)
        if partial.stat().st_size != HOVER_TRAIN_SIZE or _file_sha256(partial) != HOVER_TRAIN_SHA256:
            raise RuntimeError("Downloaded HoVer v1.1 bytes do not match the pinned artifact")
        partial.replace(destination)
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the official HoVer v1.1 data from {HOVER_TRAIN_URL}. "
            "Use --smoke only for an explicit local smoke run."
        ) from exc
    return destination


def _extract_final_response(output: str) -> str:
    """Extract the last marked final response after removing reasoning blocks.

    Args:
        output: Raw model output.

    Returns:
        Text after the final response marker, or all visible output when the
        marker is absent.
    """
    output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()
    if FINAL_RESPONSE_MARKER in output:
        return output.rsplit(FINAL_RESPONSE_MARKER, 1)[1].strip()
    return output.strip()


def _call_lm(system: str, user: str, model: str, api_base: str | None) -> str:
    """Call the solver with the decoding settings used by the GEPA artifact.

    Args:
        system: Candidate system prompt, omitted when empty.
        user: Example-specific user message.
        model: LiteLLM model identifier.
        api_base: Optional solver API endpoint.

    Returns:
        Raw message content when truthy, otherwise raw reasoning content when
        available. Returns an empty string when the fixed artifact request
        exceeds the model context or the response contains neither field.
    """
    messages = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "num_retries": EXPERIMENT_NUM_RETRIES,
        **experiment_decoding(model),
        **experiment_request_overrides(model),
    }
    if api_base is not None:
        kwargs["api_base"] = api_base
    try:
        response = cast(litellm.ModelResponse, litellm.completion(**kwargs))
    except litellm.exceptions.ContextWindowExceededError:
        print(f"WARNING: input exceeds model context (prompt {len(system) + len(user)} chars); prediction failed.")
        return ""
    message = response.choices[0].message
    return message.content or getattr(message, "reasoning_content", None) or ""


def _call_chain_of_thought(
    instructions: str,
    inputs: dict[str, str | list[str]],
    output_field: str,
    model: str,
    api_base: str | None,
) -> tuple[str, str]:
    """Run one artifact-style DSPy Chain-of-Thought predictor call.

    The GEPA artifact uses DSPy's ChatAdapter around every HoVer module. This
    reproduces its visible ``reasoning`` field, structured input/output markers,
    task-instruction placement, and strict terminal-field parsing without adding
    DSPy as a runtime dependency of the standalone harness.

    Args:
        instructions: Current optimized signature instructions.
        inputs: Ordered fields supplied to the predictor. Passage lists use
            DSPy's numbered blob formatting for string-typed signatures.
        output_field: Terminal output name, either ``summary`` or ``query``.
        model: LiteLLM model identifier.
        api_base: Optional solver API endpoint.

    Returns:
        Parsed visible reasoning and terminal output text.

    Raises:
        ValueError: The completion omits either required structured output.
    """
    output_fields = ["reasoning", output_field]
    input_descriptions = "\n".join(f"{index}. `{name}` (str): " for index, name in enumerate(inputs, 1))
    output_descriptions = "\n".join(f"{index}. `{name}` (str): " for index, name in enumerate(output_fields, 1))
    input_structure = "\n\n".join(f"[[ ## {name} ## ]]\n{{{name}}}" for name in inputs)
    output_structure = "\n\n".join(f"[[ ## {name} ## ]]\n{{{name}}}" for name in output_fields)
    indented_instructions = "\n".join(f"        {line}" for line in instructions.splitlines())
    system = (
        f"Your input fields are:\n{input_descriptions}\n"
        f"Your output fields are:\n{output_descriptions}\n"
        "All interactions will be structured in the following way, with the appropriate values filled in.\n\n"
        f"{input_structure}\n\n{output_structure}\n\n[[ ## completed ## ]]\n\n"
        f"In adhering to this structure, your objective is: \n{indented_instructions}"
    )
    rendered_input_fields = []
    for name, value in inputs.items():
        if isinstance(value, list):
            formatted_items = []
            for item in value:
                if "\n" in item or "«" in item or "»" in item:
                    indented = item.replace("\n", "\n    ")
                    formatted_items.append(f"«««\n    {indented}\n»»»")
                else:
                    formatted_items.append(f"«{item}»")
            if not formatted_items:
                rendered_value = "N/A"
            elif len(formatted_items) == 1:
                rendered_value = formatted_items[0]
            else:
                rendered_value = "\n".join(f"[{index}] {item}" for index, item in enumerate(formatted_items, 1))
        else:
            rendered_value = value
        rendered_input_fields.append(f"[[ ## {name} ## ]]\n{rendered_value}")
    rendered_inputs = "\n\n".join(rendered_input_fields)
    user = (
        f"{rendered_inputs}\n\nRespond with the corresponding output fields, starting with the field "
        f"`[[ ## reasoning ## ]]`, then `[[ ## {output_field} ## ]]`, and then ending with the marker "
        "for `[[ ## completed ## ]]`."
    )
    completion = re.sub(r"<think>.*?</think>", "", _call_lm(system, user, model, api_base), flags=re.DOTALL)
    sections: dict[str, list[str]] = {}
    current_field: str | None = None
    for line in completion.splitlines():
        match = re.match(r"\s*\[\[ ## (\w+) ## \]\]", line)
        if match:
            matched_field = match.group(1)
            current_field = matched_field
            if matched_field not in sections:
                sections[matched_field] = []
                remaining = line[match.end() :].strip()
                if remaining:
                    sections[matched_field].append(remaining)
            else:
                current_field = None
        elif current_field is not None:
            sections[current_field].append(line)

    parsed = {name: "\n".join(sections.get(name, [])).strip() for name in output_fields}
    missing = [name for name in output_fields if name not in sections]
    if missing:
        raise ValueError(f"HoVer Chain-of-Thought completion omitted required fields {missing}: {completion!r}")
    return parsed["reasoning"], parsed[output_field]


def _supporting_titles(item: dict) -> list[str]:
    """Extract unique document titles from raw or Hugging Face HoVer facts.

    Args:
        item: Raw record containing one supported-facts representation.

    Returns:
        First-seen titles deduplicated exactly as in the artifact's raw-title
        document count. Retrieval scoring normalizes the retained titles later.
    """
    facts = item.get("supporting_facts")
    titles: list[str] = []
    if isinstance(facts, dict):
        keys = facts.get("key") or facts.get("title") or []
        if isinstance(keys, str):
            titles.append(keys)
        elif isinstance(keys, list):
            titles.extend(str(key) for key in keys)
    elif isinstance(facts, list):
        for fact in facts:
            if isinstance(fact, list | tuple) and fact:
                titles.append(str(fact[0]))
            elif isinstance(fact, dict):
                title = fact.get("key") or fact.get("title")
                if title:
                    titles.append(str(title))

    unique: list[str] = []
    seen: set[str] = set()
    for title in titles:
        title = title.strip()
        if title and title not in seen:
            seen.add(title)
            unique.append(title)
    return unique


def _to_example(item: dict) -> dict:
    """Normalize one official HoVer record for ``optimize_anything``.

    Args:
        item: Raw release record.

    Returns:
        Claim, label, identity, and supporting-title fields expected by the
        benchmark evaluator.
    """
    titles = _supporting_titles(item)
    claim = str(item.get("claim", "")).strip()
    return {
        "claim": claim,
        "prompt": claim,
        "question": claim,
        "supporting_facts": item.get("supporting_facts", []),
        "supporting_titles": titles,
        "gold_titles": titles,
        "label": str(item.get("label", "")),
        "id": str(item.get("uid", item.get("id", ""))),
        "num_hops": int(item.get("num_hops", 0) or 0),
        "answer": json.dumps(titles, ensure_ascii=False),
    }


def load_hover_dataset(
    seed: int = 0,
    data_dir: str | os.PathLike[str] = DATA_DIR,
    smoke: bool = False,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Load official HoVer v1.1 records with exactly three unique documents.

    Args:
        seed: Experiment seed. Zero preserves the artifact split; a nonzero
            value remixes only the selected training and validation records.
        data_dir: Directory containing or receiving the release file.
        smoke: Use the three-record offline fixture instead of official data.

    Returns:
        Deterministic training, validation, and test records.

    Raises:
        RuntimeError: The official release cannot be downloaded.
        ValueError: The release is not a JSON list or does not contain the
            artifact's exact number of eligible three-document records.
    """
    if smoke:
        examples = [_to_example(record) for record in _SMOKE_RECORDS]
        return examples[:1], examples[1:2], examples[2:]

    path = ensure_data_downloaded(data_dir)
    with path.open(encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError(f"Expected a list in {path}")

    examples = [
        _to_example(record)
        for record in records
        if isinstance(record, dict) and record.get("claim") and len(_supporting_titles(record)) == 3
    ]
    if len(examples) != HOVER_ELIGIBLE_COUNT:
        raise ValueError(
            f"HoVer v1.1 yielded {len(examples)} three-document records; "
            f"the pinned artifact requires exactly {HOVER_ELIGIBLE_COUNT}"
        )
    random.Random(0).shuffle(examples)
    first_boundary = int(0.4 * len(examples))
    second_boundary = int(0.8 * len(examples))
    test_pool = examples[:first_boundary]
    val_pool = examples[first_boundary:second_boundary]
    train_pool = examples[second_boundary:]
    trainset = random.Random(1).sample(train_pool, 150)
    valset = random.Random(1).sample(val_pool, 300)
    testset = random.Random(1).sample(test_pool, 300)
    if seed != 0:
        combined_train_val = trainset + valset
        random.Random(seed).shuffle(combined_train_val)
        trainset = combined_train_val[:150]
        valset = combined_train_val[150:]
    return trainset, valset, testset


def _render_passages(passages: Sequence[WikipediaPassage]) -> list[str]:
    """Render every ranked abstract using the artifact's document format.

    Args:
        passages: Ranked passages to render in order.

    Returns:
        Ordered ``"title | abstract"`` strings supplied to DSPy's adapter.
    """
    return [f"{passage.title} | {passage.text}" for passage in passages]


def run_two_stage(
    summarize1_prompt: str,
    query2_prompt: str,
    summarize2_prompt: str,
    query3_prompt: str,
    claim: str,
    retriever: WikipediaRetriever,
    model: str = QWEN3_8_27B_MODEL,
    api_base: str | None = None,
    retrieval_k: int = 7,
    final_retrieval_k: int = 10,
) -> tuple[str, list[WikipediaPassage], dict[str, object]]:
    """Run the artifact's three-hop HoVer retrieval program.

    Args:
        summarize1_prompt: Instruction for summarizing first-hop evidence.
        query2_prompt: Instruction for generating the second query.
        summarize2_prompt: Instruction for summarizing second-hop evidence.
        query3_prompt: Instruction for generating the final query.
        claim: HoVer claim whose evidence must be retrieved.
        retriever: Wikipedia passage retriever.
        model: Solver model identifier.
        api_base: Optional solver API endpoint.
        retrieval_k: Passages requested for intermediate hops.
        final_retrieval_k: Passages requested by the final hop.

    Returns:
        Serialized generated queries, all passages retrieved across the three
        hops, and the complete component trace used for artifact feedback.
    """
    hop1 = retriever.search(claim, retrieval_k)
    summary_1_reasoning, summary_1 = _call_chain_of_thought(
        summarize1_prompt,
        {"claim": claim, "passages": _render_passages(hop1)},
        "summary",
        model,
        api_base,
    )

    query_2_reasoning, query_2 = _call_chain_of_thought(
        query2_prompt,
        {"claim": claim, "summary_1": summary_1},
        "query",
        model,
        api_base,
    )
    hop2 = retriever.search(query_2, retrieval_k)
    summary_2_reasoning, summary_2 = _call_chain_of_thought(
        summarize2_prompt,
        {"claim": claim, "context": summary_1, "passages": _render_passages(hop2)},
        "summary",
        model,
        api_base,
    )

    query_3_reasoning, query_3 = _call_chain_of_thought(
        query3_prompt,
        {"claim": claim, "summary_1": summary_1, "summary_2": summary_2},
        "query",
        model,
        api_base,
    )
    hop3 = retriever.search(query_3, final_retrieval_k)
    retrieved = hop1 + hop2 + hop3
    trace: dict[str, object] = {
        "hop1_documents": hop1,
        "summary_1_reasoning": summary_1_reasoning,
        "summary_1": summary_1,
        "query_2_reasoning": query_2_reasoning,
        "query_2": query_2,
        "hop2_documents": hop2,
        "summary_2_reasoning": summary_2_reasoning,
        "summary_2": summary_2,
        "query_3_reasoning": query_3_reasoning,
        "query_3": query_3,
        "hop3_documents": hop3,
        "retrieved_documents": retrieved,
    }
    return json.dumps([query_2, query_3], ensure_ascii=False), retrieved, trace


def artifact_component_records(example: dict, trace: dict[str, object], score: float) -> dict[str, dict]:
    """Build the artifact's component-specific HoVer reflection records.

    Summary feedback attributes later retrieval success to the summary that
    informed it. Query feedback isolates documents newly found by that query's
    immediate retrieval hop. Every component keeps the final complete-document
    score while receiving its own inputs, generated output, and diagnosis.

    Args:
        example: HoVer record containing the claim and gold document titles.
        trace: Complete three-hop execution trace from :func:`run_two_stage`.
        score: Final complete-document retrieval score.

    Returns:
        Component names mapped to ``Inputs``, ``Generated Outputs``, and
        artifact-equivalent ``Feedback`` fields.
    """
    claim = str(example["claim"])
    gold_value = example.get("gold_titles")
    gold_titles = [str(title) for title in gold_value] if isinstance(gold_value, list) else _supporting_titles(example)
    gold = {_normalize_title(title) for title in gold_titles}
    hop1_value = trace.get("hop1_documents")
    hop2_value = trace.get("hop2_documents")
    hop3_value = trace.get("hop3_documents")
    if not isinstance(hop1_value, list) or not isinstance(hop2_value, list) or not isinstance(hop3_value, list):
        raise TypeError("HoVer trace document fields must be lists.")
    hop1_documents = cast(list[WikipediaPassage], hop1_value)
    hop2_documents = cast(list[WikipediaPassage], hop2_value)
    hop3_documents = cast(list[WikipediaPassage], hop3_value)
    summary_1 = str(trace.get("summary_1", ""))
    summary_2 = str(trace.get("summary_2", ""))
    query_2 = str(trace.get("query_2", ""))
    query_3 = str(trace.get("query_3", ""))
    summary_1_reasoning = str(trace.get("summary_1_reasoning", ""))
    summary_2_reasoning = str(trace.get("summary_2_reasoning", ""))
    query_2_reasoning = str(trace.get("query_2_reasoning", ""))
    query_3_reasoning = str(trace.get("query_3_reasoning", ""))

    hop1_found = {_normalize_title(passage.title) for passage in hop1_documents}
    through_hop2_found = hop1_found | {_normalize_title(passage.title) for passage in hop2_documents}
    final_found = through_hop2_found | {_normalize_title(passage.title) for passage in hop3_documents}
    missing_after_hop1 = gold - hop1_found
    missing_after_hop2 = gold - through_hop2_found
    missing_at_end = gold - final_found

    if score:
        summary1_feedback = (
            "Your summaries are correct and useful in guiding query generation to retrieve relevant evidence documents."
        )
        summary2_feedback = summary1_feedback
        query2_feedback = "Your queries are correct and useful in retrieving relevant evidence documents."
        query3_feedback = query2_feedback
    else:
        summary1_helped = sorted(missing_after_hop1 - missing_at_end)
        summary2_helped = sorted(missing_after_hop2 - missing_at_end)
        query2_helped = sorted(missing_after_hop1 - missing_after_hop2)
        query3_helped = sorted(missing_after_hop2 - missing_at_end)
        summary1_feedback = (
            "Your summaries are used to generate queries to identify evidence relevant to the claim.\n"
            + (
                "**Successful retrieval:** Your summary correctly helped retrieve the following evidence: "
                f"{', '.join(summary1_helped)}.\n"
                if summary1_helped
                else ""
            )
            + (
                "**Missing evidence:** However, your summary could not help make the connection to these key "
                f"evidence: {', '.join(sorted(missing_at_end))}.\n"
                if missing_at_end
                else ""
            )
            + "Think about how you can make the connection between the provided passages and the missed evidence "
            "relevant to the claim."
        )
        summary2_feedback = (
            "Your summaries are used to generate queries to identify evidence relevant to the claim.\n"
            + (
                "**Successful retrieval:** Your summary correctly helped retrieve the following evidence: "
                f"{', '.join(summary2_helped)}.\n"
                if summary2_helped
                else ""
            )
            + (
                "**Missing evidence:** However, your summary could not help make the connection to these key "
                f"evidence: {', '.join(sorted(missing_at_end))}.\n"
                if missing_at_end
                else ""
            )
            + "Think about how you can make the connection between the provided passages and the missed evidence "
            "relevant to the claim."
        )
        query2_feedback = (
            "Your queries are used to identify evidence relevant to the claim.\n"
            + (
                "**Successful retrieval:** Your query correctly helped retrieve the following evidence: "
                f"{', '.join(query2_helped)}.\n"
                if query2_helped
                else ""
            )
            + (
                "**Missing evidence:** However, your query could not help retrieve these key evidence: "
                f"{', '.join(sorted(missing_after_hop2))}.\n"
                if missing_after_hop2
                else ""
            )
            + "Think about how you can modify your query to make the connection between the provided summary and "
            "the missed evidence relevant to the claim."
        )
        query3_feedback = (
            "Your queries are used to identify evidence relevant to the claim.\n"
            + (
                "**Successful retrieval:** Your query correctly helped retrieve the following evidence: "
                f"{', '.join(query3_helped)}.\n"
                if query3_helped
                else ""
            )
            + (
                "**Missing evidence:** However, your query could not help retrieve these key evidence: "
                f"{', '.join(sorted(missing_at_end))}.\n"
                if missing_at_end
                else ""
            )
            + "Think about how you can modify your query to make the connection between the provided summary and "
            "the missed evidence relevant to the claim."
        )

    hop1_passages = [f"{passage.title} | {passage.text}" for passage in hop1_documents]
    hop2_passages = [f"{passage.title} | {passage.text}" for passage in hop2_documents]
    return {
        "summarize1": {
            "Inputs": {"claim": claim, "passages": hop1_passages},
            "Generated Outputs": {"reasoning": summary_1_reasoning, "summary": summary_1},
            "Feedback": summary1_feedback,
        },
        "create_query_hop2": {
            "Inputs": {"claim": claim, "summary_1": summary_1},
            "Generated Outputs": {"reasoning": query_2_reasoning, "query": query_2},
            "Feedback": query2_feedback,
        },
        "summarize2": {
            "Inputs": {"claim": claim, "context": summary_1, "passages": hop2_passages},
            "Generated Outputs": {"reasoning": summary_2_reasoning, "summary": summary_2},
            "Feedback": summary2_feedback,
        },
        "create_query_hop3": {
            "Inputs": {"claim": claim, "summary_1": summary_1, "summary_2": summary_2},
            "Generated Outputs": {"reasoning": query_3_reasoning, "query": query_3},
            "Feedback": query3_feedback,
        },
    }


def run_single_stage(
    prompt: str,
    claim: str,
    retriever: WikipediaRetriever,
    model: str = QWEN3_8_27B_MODEL,
    api_base: str | None = None,
    retrieval_k: int = 10,
) -> list[WikipediaPassage]:
    """Generate one query and retrieve its ranked Wikipedia pages.

    Args:
        prompt: Candidate query-generation instruction.
        claim: HoVer claim whose evidence must be retrieved.
        retriever: Wikipedia passage retriever.
        model: Solver model identifier.
        api_base: Optional solver API endpoint.
        retrieval_k: Maximum passages requested.

    Returns:
        Ranked passages for the generated query.
    """
    query = _extract_final_response(_call_lm(prompt, f"claim:\n{claim}\n\nquery:", model, api_base))
    return retriever.search(query, retrieval_k)


def _normalize_title(title: str) -> str:
    """Normalize a Wikipedia title for retrieval matching.

    Args:
        title: Raw page title.

    Returns:
        Lowercase title without punctuation, articles, or repeated whitespace.
    """
    title = title.lower()
    title = "".join(character for character in title if character not in string.punctuation)
    title = re.sub(r"\b(a|an|the)\b", " ", title)
    return " ".join(title.split())


def _retrieved_titles(prediction: Sequence[WikipediaPassage | str]) -> list[str]:
    """Extract page titles from passage objects or rendered strings.

    Args:
        prediction: Retrieved passages in object or rendered form.

    Returns:
        Titles in retrieval order.
    """
    titles: list[str] = []
    for passage in prediction:
        if isinstance(passage, WikipediaPassage):
            titles.append(passage.title)
        else:
            titles.append(str(passage).split(" | ", 1)[0].strip())
    return titles


def hover_metric(prediction: Sequence[WikipediaPassage | str], example: dict) -> tuple[float, str]:
    """Score whether retrieved pages contain every gold document.

    Args:
        prediction: Retrieved passages in object or rendered form.
        example: Record containing gold or supporting titles.

    Returns:
        Complete-retrieval score and feedback naming found and missing pages.
    """
    gold_titles = [str(title) for title in example.get("gold_titles", _supporting_titles(example))]
    gold = {_normalize_title(title) for title in gold_titles}
    predicted_titles = _retrieved_titles(prediction)
    found = {_normalize_title(title) for title in predicted_titles}
    retrieved = gold & found
    missing = gold - found
    recall = len(retrieved) / len(gold) if gold else 0.0
    score = float(bool(gold) and not missing)
    feedback = (
        f"Retrieved {len(retrieved)}/{len(gold)} gold Wikipedia documents (recall={recall:.3f}). "
        f"Found: {sorted(retrieved)}. Missing: {sorted(missing)}. "
        f"Retrieved titles: {predicted_titles}."
    )
    return score, feedback


def hover_recall(prediction: Sequence[WikipediaPassage | str], example: dict) -> float:
    """Return gold-document recall for retrieved Wikipedia pages.

    Args:
        prediction: Retrieved passages in object or rendered form.
        example: Record containing gold or supporting titles.

    Returns:
        Fraction of normalized gold titles found, or zero without gold titles.
    """
    gold = {_normalize_title(str(title)) for title in example.get("gold_titles", _supporting_titles(example))}
    found = {_normalize_title(title) for title in _retrieved_titles(prediction)}
    return len(gold & found) / len(gold) if gold else 0.0
