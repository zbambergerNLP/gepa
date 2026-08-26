# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for bounded, guarded RLM execution."""

import threading
from dataclasses import dataclass, field

import pytest

from gepa.proposer.reflective_mutation.rlm_environment import (
    RLM_ENVIRONMENT_CONTRACT,
    RLMBudget,
    RLMEnvironment,
    RLMExecution,
)


@dataclass
class EchoLM:
    """Record leaf calls and echo their prompts in uppercase."""

    calls: list[str] = field(default_factory=list)

    def __call__(self, prompt: str) -> str:
        """Record and uppercase one leaf-model prompt.

        Args:
            prompt: Leaf query text.

        Returns:
            Uppercase prompt.
        """
        self.calls.append(prompt)
        return prompt.upper()


def environment(*, lm=None, budget: RLMBudget | None = None) -> RLMEnvironment:
    """Build a deterministic root environment.

    Args:
        lm: Optional leaf and child model override.
        budget: Optional execution-budget override.

    Returns:
        Root environment with fixed region and feedback context.
    """
    model = lm if lm is not None else EchoLM()
    limits = budget if budget is not None else RLMBudget()
    return RLMEnvironment({"region": "be nice", "feedback": "too vague"}, model, limits)


def test_matched_budget_has_an_eight_model_call_cap_and_is_frozen() -> None:
    """Count root, child, and leaf calls and prevent post-contract drift."""
    budget = RLMBudget(
        max_root_iterations=4,
        max_child_iterations=2,
        max_repl_calls=6,
        max_llm_queries=2,
        max_rlm_queries=1,
        max_recursion_depth=1,
        max_exec_seconds=5,
        max_output_chars=4000,
    )

    assert budget.max_root_iterations + budget.max_rlm_queries * budget.max_child_iterations + budget.max_llm_queries == 8
    with pytest.raises(AttributeError):
        budget.max_root_iterations = 5  # type: ignore[misc]


def test_context_is_repinned_while_non_context_state_persists() -> None:
    """Keep the workspace useful without allowing candidate rebinding."""
    env = environment()

    assert env.execute("region = 'changed'\nnotes = feedback.upper()").error is None
    execution = env.execute("print(region, notes)")

    assert execution.error is None
    assert execution.stdout == "be nice TOO VAGUE\n"
    assert env.context["region"] == "be nice"


@pytest.mark.parametrize(
    ("code", "error"),
    [
        pytest.param("import os", "not allowed", id="os_import"),
        pytest.param("open('secret')", "NameError", id="file_builtin"),
        pytest.param("print((1).__class__)", "private attribute", id="object_graph"),
        pytest.param("print('{0.__class__}'.format(1))", "reflective", id="format_traversal"),
    ],
)
def test_guarded_executor_rejects_io_and_reflective_access(code: str, error: str) -> None:
    """Reject common file and interpreter escape paths before execution.

    Args:
        code: Guarded Python source under test.
        error: Expected diagnostic fragment.
    """
    execution = environment().execute(code)

    assert error in (execution.error or "")


def test_main_thread_infinite_loop_is_stopped_and_environment_recovers() -> None:
    """Enforce a hard per-execution timeout without poisoning later turns."""
    env = environment(budget=RLMBudget(max_exec_seconds=0.05))

    timed_out = env.execute("while True:\n    pass")

    assert "exceeded 0.05 seconds" in (timed_out.error or "")
    assert env.execute("print('alive')").stdout == "alive\n"


def test_worker_thread_execution_fails_closed_when_signal_timeout_is_unavailable() -> None:
    """Never run unbounded model code in a worker that cannot install SIGALRM."""
    env = environment(budget=RLMBudget(max_exec_seconds=0.05))
    outcomes: list[RLMExecution] = []

    def execute_in_worker() -> None:
        """Run guarded execution from the worker and retain its result."""
        outcome = env.execute("while True:\n    pass")
        outcomes.append(outcome)

    worker = threading.Thread(target=execute_in_worker)

    worker.start()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert len(outcomes) == 1
    assert "hard timeout cannot be enforced" in (outcomes[0].error or "")


def test_leaf_batch_is_sequential_and_budgeted_atomically() -> None:
    """Preserve order and reject an overshooting batch without partial calls."""
    lm = EchoLM()
    env = environment(lm=lm, budget=RLMBudget(max_llm_queries=2))

    execution = env.execute("print(llm_query_batched(['first', 'second']))")

    assert execution.error is None
    assert execution.stdout == "['FIRST', 'SECOND']\n"
    assert lm.calls == ["first", "second"]
    assert env.usage.llm_queries == 2

    other_lm = EchoLM()
    other = environment(lm=other_lm, budget=RLMBudget(max_llm_queries=1))
    refused = other.execute("llm_query_batched(['first', 'second'])")
    assert "2 requested" in (refused.error or "")
    assert other_lm.calls == []
    assert other.usage.llm_queries == 0


def test_repl_budget_and_render_output_cap_are_enforced() -> None:
    """Bound both executions and the text fed back to the proposer."""
    env = environment(budget=RLMBudget(max_repl_calls=1))

    assert env.execute("print('first')").error is None
    exhausted = env.execute("print('second')")
    rendered = RLMExecution(stdout="x" * 10, error=None, calls=[]).render(4)

    assert "repl_calls budget exhausted" in (exhausted.error or "")
    assert rendered == "<stdout>xxxx\n...(+6 chars cut; print less at a time)</stdout>"


def test_environment_contract_records_non_security_and_worker_behavior() -> None:
    """Persist execution semantics that materially affect resumability."""
    contract = RLM_ENVIRONMENT_CONTRACT

    assert contract["execution"] == "trusted_model_in_process_guarded"
    assert contract["batched_delegation"] == "sequential"
    assert contract["timeout_off_main_thread"] == "reject_execution"
