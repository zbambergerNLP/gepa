"""Offline coverage of the stateless action ablation across all prompt and skill text."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from examples.terminalbench.reflection import ComponentActionReflectionLM
from gepa import optimize
from gepa.adapters.terminal_bench_adapter.documents import COMPONENT_KINDS, seed_documents
from gepa.core.adapter import EvaluationBatch
from gepa.strategies.document_template import TEMPLATE_FAMILIES
from gepa.utils.stop_condition import MaxCandidateProposalsStopper


class SelectorLM:
    """Return a valid two-choice distribution for a populated prompt or skill region."""

    def __init__(self, family: str) -> None:
        """Find the provider's populated task section and record selector requests."""
        template = TEMPLATE_FAMILIES[family]["user_prompt"]
        self.prompt_section = next(
            section for section, body in template.parse(seed_documents(family)["instruction_prompt"]).items() if body
        )
        self.calls = []

    def __call__(self, prompt: str) -> str:
        """Choose between two semantic actions without a Manifestor or tool loop."""
        self.calls.append(prompt)
        section = "Instructions" if "- reexpress@Name/REPLACE_TEXT:" in prompt else self.prompt_section
        return (
            "<response>"
            + "".join(
                f"<candidate><action>{action}@{section}/{operator}</action><probability>0.5</probability></candidate>"
                for action, operator in [("contextualize", "INSERT_TEXT"), ("reexpress", "REPLACE_TEXT")]
            )
            + "</response>"
        )


class RewriteLM:
    """Append a visible revision to exactly the requested section body."""

    def __init__(self) -> None:
        """Initialize the list of single-turn rewrite requests."""
        self.calls = []

    def __call__(self, prompt: str) -> str:
        """Return one fenced section body with a deterministic revision marker."""
        self.calls.append(prompt)
        body = prompt.split("```")[1].strip()
        return f"```\n{body}\nverified-rewrite\n```"


def _strategy(family: str, seed: int = 12):
    """Build the production wrapper around local scripted selector and rewrite clients."""
    selector, rewriter = SelectorLM(family), RewriteLM()
    strategy = ComponentActionReflectionLM(
        lm=rewriter,
        selector_lm=selector,
        component_kinds=COMPONENT_KINDS,
        template_family=family,
        rng=random.Random(seed),
    )
    return strategy, selector, rewriter


@pytest.mark.parametrize("family", TEMPLATE_FAMILIES)
def test_stateless_action_edits_all_sixteen_documents_with_the_correct_templates(family: str) -> None:
    """Preserve sibling sections while combining sixteen single-turn edits into one proposal."""
    candidate = seed_documents(family)
    strategy, selector, rewriter = _strategy(family)
    feedback = json.dumps({"reward": 0, "verifier_logs": {"verifier/test-stdout.txt": "FAILED test_output"}})
    dataset = {name: [{"Feedback": feedback}] for name in candidate}
    proposal, next_strategy = strategy.reflect(candidate, dataset, list(candidate))

    assert next_strategy is strategy
    assert candidate == seed_documents(family)
    assert set(proposal.new_texts) == set(proposal.metadata["component_actions"]) == set(COMPONENT_KINDS)
    assert len(selector.calls) == len(rewriter.calls) == 16
    assert all("FAILED test_output" in request for request in selector.calls + rewriter.calls)
    assert all("verified-rewrite" not in request for request in selector.calls)
    for name, revised in proposal.new_texts.items():
        template = TEMPLATE_FAMILIES[family][COMPONENT_KINDS[name]]
        before, after = template.parse(candidate[name]), template.parse(revised)
        action = proposal.metadata["component_actions"][name]
        section = action["action_target_section"]
        assert section == ("Instructions" if COMPONENT_KINDS[name] == "skill" else selector.prompt_section)
        assert action["semantic_action"] in {"contextualize", "reexpress"}
        assert after[section] == before[section] + "\nverified-rewrite"
        assert all(after[sibling] == body for sibling, body in before.items() if sibling != section)


def test_stateless_selector_choices_resume_from_checkpoint_and_retry_snapshot() -> None:
    """Preserve the next sixteen action choices and undo diagnostic history on retry."""
    candidate = seed_documents("generic")
    dataset = {name: [{"Feedback": "Improve verification"}] for name in candidate}
    strategy, _, _ = _strategy("generic")
    strategy.reflect(candidate, dataset, list(candidate))
    checkpoint = strategy.get_state()
    retry_snapshot = strategy.get_batch_retry_state()
    expected, _ = strategy.reflect(candidate, dataset, list(candidate))
    strategy.set_batch_retry_state(retry_snapshot)
    assert strategy.get_batch_retry_state() == retry_snapshot
    retried, _ = strategy.reflect(candidate, dataset, list(candidate))
    assert retried == expected

    resumed, _, _ = _strategy("generic", seed=999)
    resumed.set_state(checkpoint)
    actual, _ = resumed.reflect(candidate, dataset, list(candidate))
    assert actual == expected
    assert resumed.get_state() == strategy.get_state()


def test_real_engine_evaluates_one_combined_action_candidate_per_iteration(tmp_path: Path) -> None:
    """Count one child evaluation, rather than sixteen, for every complete harness proposal."""
    seed = seed_documents("generic")
    strategy, selector, rewriter = _strategy("generic")
    evaluations = []

    class Adapter:
        """Keep task execution local while exercising the production reflection wrapper."""

        propose_new_texts = None

        def evaluate(self, batch, candidate, capture_traces=False):
            """Reward each combined revision and record its complete component set."""
            revisions = {text.count("verified-rewrite") for text in candidate.values()}
            assert len(revisions) == 1
            assert set(candidate) == set(seed)
            evaluations.append(revisions.pop())
            return EvaluationBatch(
                outputs=[{} for _ in batch],
                scores=[evaluations[-1] / 10 for _ in batch],
                trajectories=[{} for _ in batch] if capture_traces else None,
                num_metric_calls=len(batch),
            )

        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            """Expose the same minibatch evidence to every selected document."""
            assert set(components_to_update) == set(seed)
            return {name: [{"Feedback": "Improve verification"}] for name in components_to_update}

    result = optimize(
        seed_candidate=seed,
        trainset=["train"],
        valset=["val"],
        adapter=Adapter(),
        reflection_strategy=strategy,
        module_selector="all",
        reflection_minibatch_size=1,
        stop_callbacks=MaxCandidateProposalsStopper(3),
        use_merge=False,
        run_dir=str(tmp_path),
        display_progress_bar=False,
        raise_on_exception=True,
    )
    assert len(selector.calls) == len(rewriter.calls) == 48
    assert result.total_metric_calls == len(evaluations) == 10
    assert len(result.candidates) == 4
    assert all(text.count("verified-rewrite") == 3 for text in result.best_candidate.values())
