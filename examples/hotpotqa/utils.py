"""Paper-compatible HotPotQA data, multi-hop execution, and feedback."""

import json
import os
import random
import re
import string
import unicodedata
from copy import deepcopy
from importlib.metadata import PackageNotFoundError
from importlib.metadata import distribution as package_distribution
from importlib.metadata import version as package_version

from examples.hotpotqa.benchmark_settings import (
    DATASET_SAMPLE_SEED,
    DATASET_SIZE,
    DEFAULT_DATA_PATH,
    HOTPOTQA_DSPY_COMMIT,
    HOTPOTQA_DSPY_VERSION,
    HOTPOTQA_HF_REVISION,
    RETRIEVAL_K,
    TEST_SIZE,
    TRAIN_SIZE,
    TRAIN_VALIDATION_SIZE,
    VALIDATION_SIZE,
)

try:
    import dspy  # type: ignore[import-not-found]
except ImportError:
    dspy = None  # type: ignore[assignment]

from examples.common.experiment_models import QWEN3_8_27B_MODEL
from examples.common.model_settings import resolve_benchmark_lm_kwargs
from examples.common.wikipedia import WikipediaPassage, WikipediaRetriever

if dspy is not None:

    class _HotPotQAChatAdapter(dspy.ChatAdapter):
        """Parse artifact fields after repairing one observed marker near-miss."""

        def parse(self, signature: type, completion: str | None) -> dict[str, object]:
            """Repair missing trailing hashes in exact output-field headers.

            Qwen occasionally emits ``[[ ## summary ]]`` while otherwise
            following DSPy's protocol exactly. Only a whole-line header for an
            expected output field or ``completed`` is normalized; content and
            canonical headers remain unchanged before the pinned parser runs.

            Args:
                signature: DSPy signature containing the expected output fields.
                completion: Raw assistant completion to parse, or ``None`` when
                    the provider returns no text.

            Returns:
                Parsed output fields produced by the pinned DSPy ChatAdapter.

            Raises:
                ValueError: The provider returns no completion text or the
                    completion remains invalid after the narrow marker repair.
            """
            if completion is None:
                raise ValueError("Failed to parse response as per signature: provider returned no completion text.")
            try:
                return super().parse(signature, completion)
            except ValueError:
                expected_headers = {*signature.output_fields, "completed"}
                normalized_lines = []
                repaired = False
                for line in completion.splitlines():
                    stripped = line.strip()
                    match = re.fullmatch(r"\[\[ ## (\w+) \]\]", stripped)
                    if match is not None and match.group(1) in expected_headers:
                        indentation = line[: len(line) - len(line.lstrip())]
                        line = f"{indentation}[[ ## {match.group(1)} ## ]]"
                        repaired = True
                    normalized_lines.append(line)
                if not repaired:
                    raise
                normalized_completion = "\n".join(normalized_lines)
                return super().parse(signature, normalized_completion)

else:
    _HotPotQAChatAdapter = None


def _render_passages(passages: list[WikipediaPassage]) -> list[str]:
    """Render every ranked abstract using the artifact's document format.

    Args:
        passages: Ranked passages to render in order.

    Returns:
        Ordered ``"title | abstract"`` strings supplied to DSPy's adapter.
    """
    rendered = []
    for passage in passages:
        rendered.append(f"{passage.title} | {passage.text}")
    return rendered


def validate_hotpotqa_dspy_runtime() -> tuple[str, str]:
    """Validate the exact DSPy fork used by the GEPA artifact.

    Returns:
        Installed DSPy version and Git commit.

    Raises:
        RuntimeError: DSPy is missing, has the wrong version, or was not built
            from the artifact fork's pinned commit.
    """
    if dspy is None:
        raise RuntimeError(
            "The two-stage HotPotQA program requires DSPy. "
            "Install the locked benchmark group with `uv sync --group hotpotqa-task-program`."
        )
    try:
        installed_version = package_version("dspy")
    except PackageNotFoundError as exc:
        raise RuntimeError("DSPy is importable but its installed package metadata cannot be determined.") from exc
    if installed_version != HOTPOTQA_DSPY_VERSION:
        raise RuntimeError(
            f"The HotPotQA task program requires dspy=={HOTPOTQA_DSPY_VERSION}; found dspy=={installed_version}."
        )

    try:
        direct_url_text = package_distribution("dspy").read_text("direct_url.json")
    except PackageNotFoundError as exc:
        raise RuntimeError("DSPy is importable but its installed package metadata cannot be determined.") from exc
    try:
        direct_url = json.loads(direct_url_text or "{}")
        installed_commit = str(direct_url.get("vcs_info", {}).get("commit_id", ""))
    except json.JSONDecodeError as exc:
        raise RuntimeError("DSPy's installed source metadata is not valid JSON.") from exc
    if installed_commit != HOTPOTQA_DSPY_COMMIT:
        found = installed_commit or "non-Git or unknown source"
        raise RuntimeError(f"The HotPotQA task program requires DSPy commit {HOTPOTQA_DSPY_COMMIT}; found {found}.")
    return installed_version, installed_commit


def build_hotpotqa_task_lm(
    model: str,
    api_base: str | None,
    lm_kwargs: dict[str, object] | None = None,
) -> object:
    """Build the pinned DSPy task-model client used by HotPotQA.

    Args:
        model: LiteLLM model identifier.
        api_base: Optional solver API endpoint.
        lm_kwargs: Optional fully resolved request settings. When omitted, the
            shared solver profile is used.

    Returns:
        DSPy language-model client configured with the experiment's fixed
        decoding settings.

    Raises:
        RuntimeError: DSPy is missing or does not match the artifact fork's
            pinned version and commit.
    """
    validate_hotpotqa_dspy_runtime()

    if lm_kwargs is None:
        kwargs = resolve_benchmark_lm_kwargs(model, api_base)
    else:
        kwargs = deepcopy(lm_kwargs)
    dspy.settings.configure(disable_history=True)
    kwargs["cache"] = False
    kwargs["cache_in_memory"] = False
    return dspy.LM(model=model, **kwargs)


def _call_chain_of_thought(
    instructions: str,
    signature: str,
    inputs: dict[str, str | list[str]],
    output_field: str,
    task_lm: object,
) -> tuple[str, str]:
    """Execute one real DSPy Chain-of-Thought predictor.

    The candidate text becomes the DSPy signature instruction, matching how
    the GEPA artifact updates each predictor. DSPy's ChatAdapter adds and
    parses the visible ``reasoning`` field before the predictor's terminal
    output field.

    Args:
        instructions: Current optimized signature instructions.
        signature: Artifact predictor signature before Chain-of-Thought adds
            its reasoning output.
        inputs: Ordered predictor inputs. Passage lists retain their list type
            so DSPy renders numbered guillemet-delimited blobs.
        output_field: Terminal output field to return with the reasoning.
        task_lm: Shared DSPy language-model client for the evaluation workers.

    Returns:
        Visible Chain-of-Thought reasoning and terminal output text.

    Raises:
        RuntimeError: DSPy is not installed for the two-stage task program.
        AttributeError: DSPy does not return the required output field.
    """
    if dspy is None:
        raise RuntimeError(
            "The two-stage HotPotQA program requires DSPy. "
            "Install the locked benchmark group with `uv sync --group hotpotqa-task-program`."
        )
    predictor_signature = dspy.ensure_signature(signature).with_instructions(instructions)
    predictor = dspy.ChainOfThought(predictor_signature)
    predictor.set_lm(task_lm)
    assert _HotPotQAChatAdapter is not None
    with dspy.context(adapter=_HotPotQAChatAdapter()):
        prediction = predictor(**inputs)
    return str(prediction.reasoning), str(getattr(prediction, output_field))


def normalize_answer(text: str) -> str:
    """Apply the pinned DSPy artifact's answer normalization.

    Args:
        text: Prediction or reference answer.

    Returns:
        NFD-normalized lowercase text without punctuation, articles, or
        repeated whitespace.
    """
    text = unicodedata.normalize("NFD", text)
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def run_two_stage(
    summarize1_prompt: str,
    query_prompt: str,
    summarize2_prompt: str,
    answer_prompt: str,
    question: str,
    retriever: WikipediaRetriever,
    model: str = QWEN3_8_27B_MODEL,
    api_base: str | None = None,
    retrieval_k: int = RETRIEVAL_K,
    task_lm: object | None = None,
    lm_kwargs: dict[str, object] | None = None,
) -> tuple[str, str, dict[str, object]]:
    """Run the GEPA artifact's two-retrieval-hop HotPotQA program.

    Args:
        summarize1_prompt: Instruction for summarizing first-hop evidence.
        query_prompt: Instruction for generating the second-hop query.
        summarize2_prompt: Instruction for summarizing second-hop evidence.
        answer_prompt: Instruction for producing the final answer.
        question: HotPotQA question.
        retriever: Wikipedia passage retriever.
        model: Solver model identifier.
        api_base: Optional solver API endpoint.
        retrieval_k: Maximum passages requested per hop.
        task_lm: Optional shared DSPy language-model client. A pinned client is
            created when this function is invoked directly without one.
        lm_kwargs: Optional fully resolved solver request settings used when a
            task-model client must be created.

    Returns:
        Generated second-hop query, extracted answer, and the complete
        four-component execution trace used for artifact feedback.
    """
    if task_lm is None:
        task_lm = build_hotpotqa_task_lm(model, api_base, lm_kwargs)

    hop1_documents = retriever.search(question, retrieval_k)
    hop1_passages = _render_passages(hop1_documents)
    summary_1_reasoning, summary_1 = _call_chain_of_thought(
        summarize1_prompt,
        "question,passages->summary",
        {"question": question, "passages": hop1_passages},
        "summary",
        task_lm,
    )

    query_reasoning, query = _call_chain_of_thought(
        query_prompt,
        "question,summary_1->query",
        {"question": question, "summary_1": summary_1},
        "query",
        task_lm,
    )

    hop2_documents = retriever.search(query, retrieval_k)
    hop2_passages = _render_passages(hop2_documents)
    summary_2_reasoning, summary_2 = _call_chain_of_thought(
        summarize2_prompt,
        "question,context,passages->summary",
        {"question": question, "context": summary_1, "passages": hop2_passages},
        "summary",
        task_lm,
    )

    answer_reasoning, answer = _call_chain_of_thought(
        answer_prompt,
        "question,summary_1,summary_2->answer",
        {"question": question, "summary_1": summary_1, "summary_2": summary_2},
        "answer",
        task_lm,
    )
    trace: dict[str, object] = {
        "hop1_documents": hop1_documents,
        "summary_1_reasoning": summary_1_reasoning,
        "summary_1": summary_1,
        "query_reasoning": query_reasoning,
        "query": query,
        "hop2_documents": hop2_documents,
        "summary_2_reasoning": summary_2_reasoning,
        "summary_2": summary_2,
        "answer_reasoning": answer_reasoning,
        "answer": answer,
    }
    return query, answer, trace


def hotpotqa_metric(prediction: str, gold: str) -> tuple[float, str]:
    """Compute artifact-compatible normalized answer exact match.

    Args:
        prediction: Model answer.
        gold: Reference answer.

    Returns:
        Exact-match score and feedback with the predicted and reference answers.
    """
    em = float(normalize_answer(prediction) == normalize_answer(gold))
    outcome = "Exact match" if em else "Incorrect answer"
    feedback = f"{outcome}: prediction='{prediction}', gold='{gold}', EM={em:.0f}"

    return em, feedback


def artifact_component_records(
    example: dict,
    trace: dict[str, object],
    exact_match: float,
    *,
    include_diagnostics: bool = False,
) -> dict[str, dict[str, object]]:
    """Build the paper artifact's predictor-specific reflection records.

    The labeled HotPotQA context is used only here, after the task program has
    completed. It is never included in the solver inputs that produced
    ``trace``.

    Args:
        example: Fullwiki record containing the answer, context, and supporting
            facts.
        trace: Complete two-hop execution trace from :func:`run_two_stage`.
        exact_match: Final-answer exact-match score shared by every component.
        include_diagnostics: Include explicit outcome attribution and component
            boundaries for FOREST's structured reflection evidence.

    Returns:
        Component names mapped to ``Inputs``, ``Generated Outputs``, and
        artifact-style ``Feedback`` fields.
    """
    question = str(example.get("question", ""))
    answer = str(example.get("answer", ""))
    summary_1 = str(trace.get("summary_1", ""))
    summary_1_reasoning = str(trace.get("summary_1_reasoning", ""))
    query = str(trace.get("query", ""))
    query_reasoning = str(trace.get("query_reasoning", ""))
    summary_2 = str(trace.get("summary_2", ""))
    summary_2_reasoning = str(trace.get("summary_2_reasoning", ""))
    prediction = str(trace.get("answer", ""))
    answer_reasoning = str(trace.get("answer_reasoning", ""))
    hop1_value = trace.get("hop1_documents", [])
    hop2_value = trace.get("hop2_documents", [])
    if not isinstance(hop1_value, list) or not isinstance(hop2_value, list):
        raise TypeError("HotPotQA trace document fields must be lists.")
    hop1_documents = hop1_value
    hop2_documents = hop2_value
    hop1_passages = [f"{document.title} | {document.text}" for document in hop1_documents]
    hop2_passages = [f"{document.title} | {document.text}" for document in hop2_documents]

    gold_titles, supporting_sentences, context_by_title = _gold_support(example)
    hop1_titles = {document.title.strip() for document in hop1_documents}
    hop2_titles = hop1_titles | {document.title.strip() for document in hop2_documents}
    relevant_after_hop1 = set(gold_titles) & hop1_titles
    relevant_after_hop2 = set(gold_titles) & hop2_titles
    missing_after_hop1 = set(gold_titles) - hop1_titles
    missing_after_hop2 = set(gold_titles) - hop2_titles
    newly_retrieved_in_hop2 = relevant_after_hop2 - relevant_after_hop1
    missing_documents_after_hop1 = [
        f"{title} | {''.join(context_by_title.get(title, []))}" for title in sorted(missing_after_hop1)
    ]
    missing_after_both_hops = missing_after_hop1 - hop2_titles
    missing_documents_after_both_hops = [
        f"{title} | {''.join(context_by_title.get(title, []))}" for title in sorted(missing_after_both_hops)
    ]
    full_supporting_context = "\n".join(
        f"{title} | {''.join(context_by_title.get(title, []))}" for title in dict.fromkeys(gold_titles)
    )
    ideal_summary = "\n   ".join(supporting_sentences)

    if exact_match >= 1.0:
        answer_feedback = (
            f"The provided answer, '{prediction}' is correct. Here's some additional context behind the answer:\n"
            f"{full_supporting_context}"
        )
    else:
        answer_feedback = (
            f"The provided answer, '{prediction}' is incorrect. The correct answer is: {answer}. "
            f"Here's some context behind the answer, and how you could have reasoned to get the correct answer:\n"
            f"{full_supporting_context}"
        )

    query_feedback = f"""You are optimizing the query generation for the **second hop** of a multi-hop retrieval system. Your goal is to help the system find all relevant documents necessary to answer the following question:

"{question}"

The correct answer is: "{answer}".

**System behavior overview:**
- **First hop:** Documents were retrieved directly using the original question.
- **Second hop (your query):** Your query aims to retrieve additional relevant documents not found in the first hop.

**Analysis:**
- Documents relevant to the answer retrieved in the first hop: {sorted(relevant_after_hop1)}
- Documents still needing retrieval after the first hop: {sorted(missing_after_hop2)}
- New relevant documents your earlier query retrieved in the second hop: {sorted(newly_retrieved_in_hop2)}

**Feedback for improvement:**
Your query successfully retrieved {len(newly_retrieved_in_hop2)} out of {len(missing_after_hop2)} remaining relevant document(s) in the second hop. To improve:
- Analyze the missing documents: {sorted(missing_documents_after_both_hops)}
- How can you rephrase or adjust your query to better target these?

**Tip:** Consider what connections or clues from the retrieved first hop documents could help surface the remaining relevant ones."""

    summary2_feedback = f"""You are the summary generation module in a multi-hop QA system, responsible for producing a high-quality, informative summary from the input question, an intermediate summary (context), and newly retrieved passages. Your summary will be used *directly* by the answer generation module to finalize the answer, which has no access to the underlying passages or full context.

Your goal is to integrate and synthesize information relevant to answering the multi-hop question: "{question}". The correct answer is "{answer}".

An ideal summary to answer this question would have included all of the following information:
   {ideal_summary}

While your input passages may not always contain every necessary detail, you should aim to bridge any gaps by inferring or generalizing, drawing upon information from both the initial summary and new passages. Strive to match the coverage and relevance of the ideal summary, ensuring your output contains all key supporting information needed for accurate answer generation.

Keep your summary precise and well-structured, including all necessary connections and facts that enable the answer module to confidently arrive at the correct answer."""

    summary1_feedback = f"""You are the first-hop **summarization module** in a multi-hop QA system, responsible for distilling the most critical information from the top retrieved passages in response to the initial question:

"{question}"

Your summary must serve two purposes:
1. **Enable the creation of a focused, effective follow-up query** (for the second hop).
2. **Provide a strong foundation for the answer generation module** (later stages depend on what you include here).

**Analysis:**
- Relevant documents retrieved in the first hop: {sorted(relevant_after_hop1)}
- Relevant documents still missing after first hop: {sorted(missing_after_hop1)}

**Ideal summary for this question would include:**
-----
{ideal_summary}
-----

**Feedback:**
- Ensure you cover all necessary facts and clues from the retrieved passages, especially any information that could help generate queries to surface missing supporting facts (such as connections, entities, or bridging concepts).
- Try to represent key details from the cited relevant documents ({sorted(relevant_after_hop1)}), and highlight information that might help hint or bridge to the remaining facts: {sorted(missing_documents_after_hop1)}
- If you missed mentioning or signaling these, it may become impossible for the system to retrieve them in the next hop, or generate the correct answer at the end.

**Tip:** When summarizing, don't just compress; synthesize—include both direct answers and clues required for the system's next steps."""

    records = {
        "summarize1": {
            "Inputs": {"question": question, "passages": hop1_passages},
            "Generated Outputs": {"reasoning": summary_1_reasoning, "summary": summary_1},
            "Feedback": summary1_feedback,
        },
        "create_query_hop2": {
            "Inputs": {"question": question, "summary_1": summary_1},
            "Generated Outputs": {"reasoning": query_reasoning, "query": query},
            "Feedback": query_feedback,
        },
        "summarize2": {
            "Inputs": {"question": question, "context": summary_1, "passages": hop2_passages},
            "Generated Outputs": {"reasoning": summary_2_reasoning, "summary": summary_2},
            "Feedback": summary2_feedback,
        },
        "final_answer": {
            "Inputs": {"question": question, "summary_1": summary_1, "summary_2": summary_2},
            "Generated Outputs": {"reasoning": answer_reasoning, "answer": prediction},
            "Feedback": answer_feedback,
        },
    }
    if include_diagnostics:
        contexts = {
            "summarize1": {
                "role": "Summarize first-hop passages for the question.",
                "downstream": "The summary feeds the second-hop query, second summary, and final answer; "
                "these stages do not receive the original first-hop passages.",
            },
            "create_query_hop2": {
                "role": "Generate the second retrieval query from the question and first summary.",
                "downstream": "The query retrieves passages supplied to the second summarizer.",
            },
            "summarize2": {
                "role": "Synthesize the first summary and second-hop passages for the question.",
                "downstream": "The final answer receives both summaries, without the retrieved passages.",
            },
            "final_answer": {
                "role": "Answer the question using the two summaries.",
                "downstream": "The evaluator scores this answer against the reference.",
            },
        }
        for component, record in records.items():
            record["Example ID"] = example.get("id")
            record["Component Context"] = contexts[component]
            record["End-to-end Outcome"] = {
                "metric": "exact_match",
                "score": exact_match,
                "prediction": prediction,
                "reference_answer": answer,
                "attribution": "Whole-program outcome; not a causal score for this component.",
            }
    return records


def _gold_support(example: dict) -> tuple[list[str], list[str], dict[str, list[str]]]:
    """Extract gold titles, sentences, and context without solver exposure.

    Args:
        example: Canonical HotPotQA record.

    Returns:
        Ordered supporting titles, formatted supporting sentences, and all
        context sentences keyed by document title.
    """
    context = example.get("context", {})
    context_titles = context.get("title", []) if isinstance(context, dict) else []
    context_sentences = context.get("sentences", []) if isinstance(context, dict) else []
    context_by_title = {
        str(title): [str(sentence) for sentence in sentences]
        for title, sentences in zip(context_titles, context_sentences, strict=False)
    }
    supporting_facts = example.get("supporting_facts", {})
    if isinstance(supporting_facts, dict):
        gold_titles = [str(title).strip() for title in supporting_facts.get("title", [])]
        sentence_ids = [int(sentence_id) for sentence_id in supporting_facts.get("sent_id", [])]
    else:
        gold_titles = [str(title).strip() for title in example.get("supporting_titles", [])]
        sentence_ids = [0] * len(gold_titles)
    supporting_sentences: list[str] = []
    for title, sentence_id in zip(gold_titles, sentence_ids, strict=False):
        sentences = context_by_title.get(title, [])
        if 0 <= sentence_id < len(sentences):
            supporting_sentences.append(f"{title} | {sentences[sentence_id]}")
    return gold_titles, supporting_sentences, context_by_title


def _load_from_jsonl(path: str) -> list[dict]:
    """Decode each non-empty line without applying schema validation.

    Args:
        path: JSONL source path.

    Returns:
        Decoded JSON values in file order; downstream conversion expects each
        value to be a mapping.
    """
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _jsonl_to_examples(records: list[dict]) -> list[dict]:
    """Normalize raw or smoke records to the artifact's labeled schema.

    Args:
        records: Raw HotPotQA JSONL records.

    Returns:
        Examples retaining gold context for post-execution feedback. Task
        execution receives only the ``question`` field and retrieved abstracts.
    """
    examples = []
    for record in records:
        context = record.get("context")
        if not isinstance(context, dict):
            passages = record.get("passages", [])
            context = {
                "title": [str(passage.get("title", "")) for passage in passages],
                "sentences": [[str(passage.get("text", ""))] for passage in passages],
            }
        else:
            context = {
                "title": [str(title) for title in context.get("title", [])],
                "sentences": [[str(sentence) for sentence in sentences] for sentences in context.get("sentences", [])],
            }
        supporting_facts = record.get("supporting_facts")
        if not isinstance(supporting_facts, dict):
            supporting_titles = [str(title) for title in record.get("supporting_titles", [])]
            supporting_facts = {
                "title": supporting_titles,
                "sent_id": [0] * len(supporting_titles),
            }
        else:
            supporting_facts = {
                "title": [str(title) for title in supporting_facts.get("title", [])],
                "sent_id": [int(sentence_id) for sentence_id in supporting_facts.get("sent_id", [])],
            }
        examples.append(
            {
                "question": record["question"],
                "answer": record["answer"],
                "id": record["id"],
                "type": record.get("type", ""),
                "level": record.get("level", ""),
                "context": context,
                "supporting_facts": supporting_facts,
            }
        )
    return examples


def load_hotpotqa_dataset(
    data_path: str | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Load HotpotQA fullwiki with deterministic 150/300/300 splits.

    Default (``data_path is None``): load only ``hotpot_qa/fullwiki``'s raw
    training split, preserve its order, allocate the first 40 percent to the
    test pool, the next 40 percent to validation, and the final 20 percent to
    training. As in the paper artifact, each pool is independently sampled
    with ``random.Random(1)`` to 300 test, 300 validation, and 150 training
    examples. Split membership and order are fixed across optimization seeds.

    If `data_path` is given, loads that JSONL file (smoke, 20 examples). For
    the 20-example sample, returns a 14/3/3 split (14 train / 3 val / 3 test)
    so the smoke exercises the three-way evaluation pipeline.

    Args:
        data_path: Explicit JSONL smoke source. ``None`` selects fullwiki.

    Returns:
        Deterministic training, validation, and test examples. Gold context is
        retained only for reflection feedback after task execution.

    Raises:
        FileNotFoundError: The requested explicit JSONL source does not exist.
        RuntimeError: The production fullwiki dataset cannot be loaded or is too
            small for the required splits.
    """
    if data_path is not None:
        data_path = os.path.normpath(data_path)
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"HotpotQA data not found at {data_path}. "
                f"Expected the bundled sample at examples/hotpotqa/data/hotpotqa_distractor_sample.jsonl"
            )
        records = _load_from_jsonl(data_path)
        examples = _jsonl_to_examples(records)
        if len(examples) >= DATASET_SIZE:
            rng = random.Random(0)
            rng.shuffle(examples)
            trainset = examples[:TRAIN_SIZE]
            valset = examples[TRAIN_SIZE:TRAIN_VALIDATION_SIZE]
            testset = examples[TRAIN_VALIDATION_SIZE:DATASET_SIZE]
        elif len(examples) >= 20:
            trainset = examples[:14]
            remainder = examples[14:]
            mid = len(remainder) // 2
            valset = remainder[:mid]
            testset = remainder[mid:]
        else:
            trainset = examples
            valset = examples
            testset = examples

        return trainset, valset, testset

    try:
        from datasets import load_dataset  # type: ignore[import-not-found]

        ds = load_dataset(
            "hotpot_qa",
            "fullwiki",
            revision=HOTPOTQA_HF_REVISION,
        )
        examples = _jsonl_to_examples(list(ds["train"]))
        first_boundary = int(0.4 * len(examples))
        second_boundary = int(0.8 * len(examples))
        test_pool = examples[:first_boundary]
        val_pool = examples[first_boundary:second_boundary]
        train_pool = examples[second_boundary:]
        if len(train_pool) < TRAIN_SIZE or len(val_pool) < VALIDATION_SIZE or len(test_pool) < TEST_SIZE:
            raise ValueError("HotPotQA fullwiki did not contain enough records for the artifact's 150/300/300 split")

        trainset = random.Random(DATASET_SAMPLE_SEED).sample(train_pool, TRAIN_SIZE)
        valset = random.Random(DATASET_SAMPLE_SEED).sample(val_pool, VALIDATION_SIZE)
        testset = random.Random(DATASET_SAMPLE_SEED).sample(test_pool, TEST_SIZE)

        return trainset, valset, testset

    except Exception as exc:
        raise RuntimeError(
            "Could not load hotpot_qa/fullwiki. For an explicit smoke run, pass "
            f"--data-path {DEFAULT_DATA_PATH}. Original error: {exc}"
        ) from exc
