"""Verify exact evaluation recovery without reusing later-iteration measurements."""

import sqlite3

import pytest

from gepa import optimize
from gepa.core.adapter import EvaluationBatch
from gepa.evaluation_journal import EvaluationJournal
from gepa.response_journal import ResponseJournalError
from gepa.utils.stop_condition import MaxCandidateProposalsStopper


def test_interrupted_reflection_keeps_original_feedback_and_adapter_state(tmp_path):
    """Replay the completed parent batch even when new evaluation would differ."""
    physical_calls = []
    feedback = []
    interrupt = [True]

    class Adapter:
        def __init__(self):
            self.logical_calls = 0

        def get_adapter_state(self):
            return {"logical_calls": self.logical_calls}

        def set_adapter_state(self, state):
            self.logical_calls = state.get("logical_calls", self.logical_calls)

        def evaluate(self, batch, candidate, capture_traces=False):
            physical_calls.append(dict(candidate))
            self.logical_calls += 1
            return EvaluationBatch(
                outputs=[f"physical-call-{len(physical_calls)}"] * len(batch),
                scores=[0.5] * len(batch),
                trajectories=[{"observation": len(physical_calls)}] * len(batch),
            )

        def make_reflective_dataset(self, candidate, evaluation, components):
            return {"prompt": evaluation.trajectories}

        def propose_new_texts(self, candidate, reflective_dataset, components):
            feedback.append(reflective_dataset)
            if interrupt[0]:
                interrupt[0] = False
                raise KeyboardInterrupt("allocation ended during reflection")
            return {"prompt": "revised"}

    def run(adapter):
        return optimize(
            seed_candidate={"prompt": "initial"},
            trainset=[1, 2, 3],
            valset=[1, 2, 3],
            adapter=adapter,
            reflection_lm=lambda _: "unused",
            run_dir=str(tmp_path),
            reflection_minibatch_size=3,
            stop_callbacks=MaxCandidateProposalsStopper(1),
            cache_evaluation=False,
            use_merge=False,
        )

    with pytest.raises(KeyboardInterrupt):
        run(Adapter())
    assert len(physical_calls) == 2
    resumed = Adapter()
    run(resumed)
    assert len(physical_calls) == 3
    assert feedback[0] == feedback[1]
    assert physical_calls[-1] == {"prompt": "revised"}
    assert resumed.logical_calls == 3


def test_distinct_iterations_and_phases_are_independent(tmp_path):
    """Keep ordinary repeated evaluations distinct when evaluation caching is off."""
    calls = []

    def execute():
        calls.append(len(calls))
        return [calls[-1]]

    journal = EvaluationJournal(str(tmp_path))
    assert journal.evaluate(0, "parents", [{"prompt": "same"}], object(), execute) == [0]
    assert journal.evaluate(0, "parents", [{"prompt": "same"}], object(), execute) == [0]
    assert journal.evaluate(1, "parents", [{"prompt": "same"}], object(), execute) == [1]
    assert journal.evaluate(1, "children", [{"prompt": "same"}], object(), execute) == [2]
    with pytest.raises(ResponseJournalError, match="request mismatch"):
        journal.evaluate(1, "parents", [{"prompt": "changed"}], object(), execute)
    with sqlite3.connect(journal.path) as connection:
        connection.execute("UPDATE responses SET response_json = '{}' WHERE namespace = 'children'")
    with pytest.raises(ResponseJournalError, match="checksum mismatch"):
        journal.evaluate(1, "children", [{"prompt": "same"}], object(), execute)
    assert len(calls) == 3
