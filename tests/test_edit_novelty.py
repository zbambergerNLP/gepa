"""Check exact-evaluation reuse and conservative changed-span screening."""

import json
from copy import deepcopy

import pytest

from gepa.strategies.edit_novelty import EDIT_NOVELTY_CONTRACT, inspect_edit, verify_edit


def record(before, after, *, minibatch=(1, 2, 3), **fields):
    return {
        "before": before,
        "after": after,
        "minibatch_ids": list(minibatch),
        "attempt_id": "previous",
        "evaluated": True,
        **fields,
    }


def test_exact_edit_blocks_only_identical_ordered_evaluated_evidence():
    previous = record("Answer.", "Answer carefully.")
    same = inspect_edit(previous["before"], previous["after"], [previous], [1, 2, 3])
    assert same["verdict"] == "exact_duplicate"
    assert same["blocked"] is True
    assert same["matched_attempt_id"] == "previous"
    for batch in ([4, 5, 6], [3, 2, 1]):
        different = inspect_edit(previous["before"], previous["after"], [previous], batch)
        assert different["verdict"] == "suspected_similarity"
        assert different["blocked"] is False
        assert "different_ordered_minibatch_allowed" in different["reasons"]


def test_same_batch_pending_reservation_blocks_without_evaluation():
    previous = record("Answer.", "Answer carefully.", evaluated=False, pending=True)
    assert inspect_edit(previous["before"], previous["after"], [previous], [1, 2, 3])["blocked"] is True
    different = inspect_edit(previous["before"], previous["after"], [previous], [4, 5, 6])
    assert different["verdict"] == "novel"


def test_failed_unevaluated_attempt_is_not_a_duplicate():
    previous = record("Answer.", "Answer carefully.", evaluated=False)
    assert inspect_edit(previous["before"], previous["after"], [previous], [1, 2, 3])["verdict"] == "novel"


def test_unchanged_prompt_prefix_does_not_inflate_similarity():
    before = "Unchanged background.\n" * 300
    previous = record(before, before + "Name the country.")
    result = inspect_edit(before, before + "Return a date.", [previous], [1, 2, 3])
    assert result["verdict"] == "novel"
    assert result["blocked"] is False


def test_similar_inserted_text_is_flagged_but_not_blocked():
    before = "Answer.\n"
    previous = record(before, before + "Use the full entity name in the answer.")
    result = inspect_edit(before, before + "Use the full entity name in your answer.", [previous], [1, 2, 3])
    assert result["verdict"] == "suspected_similarity"
    assert result["blocked"] is False
    assert result["score"] >= EDIT_NOVELTY_CONTRACT["similarity_threshold"]
    assert "lexical_flag_only_not_semantic_equivalence" in result["reasons"]


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("Always include the requested supporting evidence.", "Always not include the requested supporting evidence."),
        ("Return at most 100 words of supporting evidence.", "Return at most 101 words of supporting evidence."),
        ("You must include supporting evidence.", "You mustn't include supporting evidence."),
    ],
)
def test_negation_and_numeric_differences_are_not_similarity_matches(old, new):
    before = "Answer.\n"
    result = inspect_edit(before, before + new, [record(before, before + old)], [1, 2, 3])
    assert result["verdict"] == "novel"
    assert result["blocked"] is False


def test_code_whitespace_is_never_normalized_into_exact_match():
    before = "def answer():\n    pass\n"
    previous = record(before, "def answer():\n    return 1\n")
    proposed = "def answer():\n\treturn 1\n"
    result = inspect_edit(before, proposed, [previous], [1, 2, 3])
    assert result["verdict"] != "exact_duplicate"
    assert result["blocked"] is False


def test_same_words_at_different_parent_offsets_are_not_a_match():
    before = "First section.\nSecond section.\n"
    previous = record(before, "First section.\nBe precise.\nSecond section.\n")
    result = inspect_edit(before, before + "Be precise.\n", [previous], [1, 2, 3])
    assert result["verdict"] == "novel"


def test_exact_same_batch_match_wins_over_earlier_other_evidence_match():
    before, after = "Answer.", "Answer carefully."
    previous = [
        record(before, after, minibatch=(4, 5, 6), attempt_id="other-evidence"),
        record(before, after, attempt_id="same-evidence"),
    ]
    result = inspect_edit(before, after, previous, [1, 2, 3])
    assert result["blocked"] is True
    assert result["matched_attempt_id"] == "same-evidence"


def test_feedback_is_bounded_and_inspection_does_not_mutate_inputs():
    before = "Answer.\n"
    previous = [record(before, before + "Detailed supporting context. " * 100)]
    snapshot = deepcopy(previous)
    result = inspect_edit(before, previous[0]["after"], previous, [1, 2, 3])
    assert result == inspect_edit(before, previous[0]["after"], previous, [1, 2, 3])
    assert previous == snapshot
    fragment = result["feedback"]["changes"][0]
    assert len(fragment["after"]) <= 160
    assert fragment["after_truncated"] is True


def test_different_parent_bytes_do_not_block_an_identical_output():
    previous = record("Answer slowly.", "Answer carefully.")
    result = inspect_edit("Answer quickly.", "Answer carefully.", [previous], [1, 2, 3])
    assert result["verdict"] == "novel"


class Verifier:
    def __init__(self, verdict="distinct", matched=None, reason="The operative requirement differs.", raw=None):
        self.raw = (
            raw
            if raw is not None
            else json.dumps({"verdict": verdict, "matched_attempt_id": matched, "reason": reason})
        )
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.raw


def verify(model, previous, *, before="Answer.", after="Answer with evidence.", batch=(1, 2, 3), evidence=None):
    return verify_edit(model, before, after, previous, batch, evidence or [{"question": "Current training question"}])


def test_verifier_cheap_paths_do_not_call_model():
    model = Verifier()
    assert verify(model, [record("Answer.", "Answer with evidence.")])["verdict"] == "exact_duplicate"
    assert verify(model, [])["verdict"] == "novel"
    assert verify(model, [], after="Answer.")["verifier_called"] is False
    assert model.prompts == []


def test_model_can_detect_a_paraphrased_edit_despite_low_lexical_similarity():
    model = Verifier("duplicate", "previous", "Both require evidence supporting the answer.")
    previous = [record("Answer.", "Answer and provide corroboration.")]
    result = verify(model, previous)
    assert result["lexical_verdict"] == "novel"
    assert result["verdict"] == "model_duplicate"
    assert result["blocked"] is True
    assert result["matched_attempt_id"] == "previous"
    assert result["prompt"] == model.prompts[0]
    assert result["raw_output"] == model.raw
    assert result["verifier_called"] is True
    assert result["feedback"]["reason"] == "Both require evidence supporting the answer."
    assert result["feedback"]["attempt_id"] == "previous"
    assert len(model.prompts) == 1


@pytest.mark.parametrize("verdict", ["distinct", "uncertain"])
def test_distinct_and_uncertain_model_decisions_allow_evaluation(verdict):
    result = verify(Verifier(verdict), [record("Answer.", "Answer briefly.")])
    assert result["verdict"] == "model_" + verdict
    assert result["blocked"] is False


@pytest.mark.parametrize(
    "raw",
    [
        "not JSON",
        "[]",
        '{"verdict": "duplicate", "matched_attempt_id": null, "reason": "Same meaning."}',
        '{"verdict": "duplicate", "matched_attempt_id": "unknown", "reason": "Same meaning."}',
        '{"verdict": "duplicate", "matched_attempt_id": "previous", "reason": "  "}',
        '{"verdict": "duplicate", "matched_attempt_id": "previous"}',
        '{"verdict": "duplicate", "matched_attempt_id": "previous", "reason": "Same.", "extra": true}',
        '{"verdict": "similar", "matched_attempt_id": "previous", "reason": "Same meaning."}',
    ],
)
def test_invalid_verifier_output_is_preserved_and_allowed_without_retry(raw):
    model = Verifier(raw=raw)
    result = verify(model, [record("Answer.", "Answer briefly.")])
    assert result["verdict"] == "model_uncertain"
    assert result["blocked"] is False
    assert result["parse_error"]
    assert result["raw_output"] == raw
    assert len(model.prompts) == 1


def test_provider_failure_propagates_without_a_retry_or_allow_verdict():
    calls = []

    def failed_provider(prompt):
        calls.append(prompt)
        raise RuntimeError("provider unavailable")

    with pytest.raises(RuntimeError, match="provider unavailable"):
        verify(failed_provider, [record("Answer.", "Answer briefly.")])
    assert len(calls) == 1


def test_full_parent_edits_training_context_and_outcome_are_visible_without_truncation():
    model = Verifier()
    before = "Keep exact whitespace.\n" * 200
    prior_evidence = [{"question": "Prior question", "answer": "Prior evidence." * 300}]
    current_evidence = [{"question": "New question", "answer": "Current evidence." * 300}]
    previous = [
        record(
            before,
            before + "Return at most 100 words.\n",
            minibatch=(4, 5, 6),
            training_evidence=prior_evidence,
            outcome="loss",
            training_gain=-1.0,
            training_mean_before=1.0,
            training_mean_after=0.0,
        )
    ]
    after = before + "Do not return more than 101 words.\n"
    result = verify(model, previous, before=before, after=after, evidence=current_evidence)
    payload = json.loads(model.prompts[0].split("\nCOMPARISON_DATA\n", 1)[1])
    assert payload["canonical_parent"] == before
    assert payload["proposed_after"] == after
    assert payload["current_training_evidence"] == current_evidence
    assert payload["current_minibatch_ids"] == [1, 2, 3]
    prior = payload["previous_records"][0]
    assert prior["training_evidence"] == prior_evidence
    assert prior["minibatch_ids"] == [4, 5, 6]
    assert prior["outcome"] == "loss"
    assert prior["training_gain"] == -1.0
    assert prior["training_mean_before"] == 1.0
    assert prior["training_mean_after"] == 0.0
    assert "negation, numeric literals, conditions, scope, target context" in result["prompt"]
    assert "A materially stronger requirement can change behavior" in result["prompt"]
    assert "A prior failure, tie or loss is not proof" in result["prompt"]


def test_missing_other_batch_evidence_is_uncertain_without_a_model_call():
    model = Verifier("duplicate", "previous")
    result = verify(model, [record("Answer.", "Answer with evidence.", minibatch=(4, 5, 6))])
    assert result["verdict"] == "model_uncertain"
    assert result["blocked"] is False
    assert result["verifier_called"] is False
    assert result["reasons"] == ["prior_training_evidence_missing_for_different_minibatch"]
    assert model.prompts == []


def test_model_cannot_reject_using_a_contextless_other_batch_record_in_mixed_history():
    model = Verifier("duplicate", "contextless")
    previous = [
        record("Answer.", "Answer clearly.", minibatch=(4, 5, 6), attempt_id="contextless"),
        record("Answer.", "Answer briefly.", attempt_id="same-batch"),
    ]
    result = verify(model, previous)
    assert result["verdict"] == "model_uncertain"
    assert result["blocked"] is False
    assert result["raw_output"] == model.raw
    assert len(model.prompts) == 1


def test_model_compares_only_the_eight_most_recent_eligible_records_once():
    model = Verifier()
    previous = [record("Answer.", f"Answer in style {index}.", attempt_id=str(index)) for index in range(12)]
    previous.append(record("Answer.", "Unevaluated failure.", evaluated=False, attempt_id="failed"))
    verify(model, previous)
    payload = json.loads(model.prompts[0].split("\nCOMPARISON_DATA\n", 1)[1])
    assert [item["attempt_id"] for item in payload["previous_records"]] == [str(index) for index in range(4, 12)]
    assert len(model.prompts) == 1


def test_old_exact_same_evidence_duplicate_still_blocks_beyond_model_history_limit():
    model = Verifier()
    previous = [record("Answer.", "Answer with evidence.", attempt_id="old-exact")]
    previous.extend(record("Answer.", f"Answer in style {index}.", attempt_id=str(index)) for index in range(12))
    result = verify(model, previous)
    assert result["verdict"] == "exact_duplicate"
    assert result["matched_attempt_id"] == "old-exact"
    assert model.prompts == []


def test_model_verification_is_deterministic_and_does_not_mutate_history():
    model = Verifier("duplicate", "previous", "The operative change is identical.")
    previous = [record("Answer.", "Answer with corroboration.", training_evidence=[{"question": "Training only"}])]
    snapshot = deepcopy(previous)
    first = verify(model, previous)
    second = verify(model, previous)
    assert first.pop("elapsed_seconds") >= 0
    assert second.pop("elapsed_seconds") >= 0
    assert first == second
    assert previous == snapshot


def test_structured_backend_receives_the_same_full_evidence_and_metadata_is_copied():
    class StructuredVerifier:
        def __init__(self):
            self.calls = []
            self.last_evidence = {"probabilities": {"distinct": 0.9, "uncertain": 0.1}}

        def classify_edit(self, evidence, instructions):
            self.calls.append((deepcopy(evidence), instructions))
            return json.dumps({"verdict": "distinct", "matched_attempt_id": None, "reason": "Distinct choice."})

        def __call__(self, prompt):
            raise AssertionError("Structured verification must not fall back to generation.")

    model = StructuredVerifier()
    previous = [record("Answer.", "Answer briefly.")]
    result = verify(model, previous)
    payload, instructions = model.calls[0]
    assert payload["previous_records"][0]["attempt_id"] == "previous"
    assert payload["canonical_parent"] == "Answer."
    assert payload["proposed_after"] == "Answer with evidence."
    assert "COMPARISON_DATA" not in instructions
    assert result["verdict"] == "model_distinct"
    assert result["backend_evidence"] == model.last_evidence
    model.last_evidence["probabilities"]["distinct"] = 0.0
    assert result["backend_evidence"]["probabilities"]["distinct"] == 0.9
    assert len(model.calls) == 1


def test_structured_backend_error_propagates_without_generative_fallback():
    class FailedVerifier:
        def classify_edit(self, evidence, instructions):
            raise RuntimeError("Structured provider unavailable")

        def __call__(self, prompt):
            raise AssertionError("No fallback is authorized.")

    with pytest.raises(RuntimeError, match="Structured provider unavailable"):
        verify(FailedVerifier(), [record("Answer.", "Answer briefly.")])


def test_pending_same_parent_edit_on_different_batch_is_compared_with_its_evidence():
    model = Verifier("duplicate", "pending-sibling", "Both changes request evidence for the same uncovered issue.")
    previous = [
        record(
            "Answer.",
            "Answer with corroboration.",
            minibatch=(4, 5, 6),
            attempt_id="pending-sibling",
            evaluated=False,
            pending=True,
            training_evidence=[{"question": "Earlier proposal batch", "feedback": "No supporting evidence."}],
        )
    ]
    result = verify(model, previous)
    assert result["verdict"] == "model_duplicate"
    assert result["blocked"] is True
    payload = json.loads(model.prompts[0].split("\nCOMPARISON_DATA\n", 1)[1])
    prior = payload["previous_records"][0]
    assert prior["pending"] is True
    assert prior["evaluated"] is False
    assert prior["minibatch_ids"] == [4, 5, 6]
    assert prior["training_evidence"] == previous[0]["training_evidence"]


def test_pending_other_batch_without_evidence_is_uncertain_and_allowed():
    model = Verifier("duplicate", "previous")
    previous = [record("Answer.", "Answer with corroboration.", minibatch=(4, 5, 6), evaluated=False, pending=True)]
    result = verify(model, previous)
    assert result["verdict"] == "model_uncertain"
    assert result["blocked"] is False
    assert model.prompts == []


def test_verifier_records_only_per_call_counter_deltas_and_elapsed_time():
    class MeteredVerifier(Verifier):
        def __init__(self):
            super().__init__()
            self.total_cost = 12.5
            self.total_tokens_in = 900
            self.total_tokens_out = 300

        def __call__(self, prompt):
            self.total_cost += 0.125
            self.total_tokens_in += 80
            self.total_tokens_out += 12
            return super().__call__(prompt)

    result = verify(MeteredVerifier(), [record("Answer.", "Answer briefly.")])
    assert result["usage"] == {"cost": 0.125, "tokens_in": 80, "tokens_out": 12}
    assert result["elapsed_seconds"] >= 0


def test_missing_or_nonfinite_usage_counters_remain_unknown():
    model = Verifier()
    model.total_cost = float("nan")
    model.total_tokens_in = float("inf")
    result = verify(model, [record("Answer.", "Answer briefly.")])
    assert result["usage"] == {"cost": None, "tokens_in": None, "tokens_out": None}
    assert result["elapsed_seconds"] >= 0


def test_no_model_call_has_known_zero_usage_and_elapsed_time():
    model = Verifier()
    result = verify(model, [])
    assert result["usage"] == {"cost": 0.0, "tokens_in": 0, "tokens_out": 0}
    assert result["elapsed_seconds"] == 0.0
    assert result["verifier_called"] is False


def test_invalid_verifier_output_keeps_its_measured_cost():
    class InvalidMeteredVerifier(Verifier):
        total_cost = 1.0

        def __call__(self, prompt):
            self.total_cost += 0.125
            return "invalid JSON"

    result = verify(InvalidMeteredVerifier(), [record("Answer.", "Answer briefly.")])
    assert result["verdict"] == "model_uncertain"
    assert result["usage"] == {"cost": 0.125, "tokens_in": None, "tokens_out": None}
