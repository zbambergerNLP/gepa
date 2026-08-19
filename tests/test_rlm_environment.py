# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for the RLM environment (:mod:`gepa.proposer.reflective_mutation.rlm_environment`).

Covers the one-action-per-turn protocol parser, the persistent sandbox (bound context, cross-execution
state, blocked imports and builtins, timeouts and budgets) and the ``llm_query``/``rlm_query`` delegation
primitives. The language model is replaced with in-process fakes, so no real model is ever called.

Expected usage:
```bash
pytest tests/test_rlm_environment.py -vv
```
"""

# Standard library imports
import threading

# Third-party imports
import pytest

# Local imports
from gepa.proposer.reflective_mutation.base import LanguageModel
from gepa.proposer.reflective_mutation.rlm_environment import (
    RLMBudget,
    RLMEnvironment,
    RLMExecution,
    RLMProtocolError,
    parse_action,
)

# ====================== #
# Test Fakes and Helpers #
# ====================== #


class FakeScriptedLM:
    """A fake LM: records prompts, replays a fixed list of outputs in order."""

    def __init__(self, *outputs: str) -> None:
        self.outputs = list(outputs)
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        if not self.outputs:
            return "<final>(out of script)</final>"
        return self.outputs.pop(0)


class FakeEchoLM:
    """A fake leaf LM: records prompts (thread-safely) and answers each with its own text, upper-cased."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, prompt: str) -> str:
        with self._lock:
            self.calls.append(prompt)
        return prompt.upper()


class FakeChildLM:
    """A fake LM for child RLMs: inspects `context` in Python, then answers with what it printed.

    Every child prompt is answered by a <python> turn on first sight and a
    <final> turn once its transcript carries a <stdout> block, so it works
    unchanged for concurrent (batched) children.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, prompt: str) -> str:
        with self._lock:
            self.calls.append(prompt)
        if "<stdout>" not in prompt:
            return "<python>print('child saw:', context)</python>"
        stdout = prompt.rsplit("<stdout>", 1)[1].split("</stdout>", 1)[0].strip()
        return f"<final>{stdout}</final>"


def _env(lm: LanguageModel | None = None, budget: RLMBudget | None = None, **context: str) -> RLMEnvironment:
    """Build an RLMEnvironment with a default context, fake LM and budget for tests."""
    return RLMEnvironment(context or {"region": "be nice"}, lm or FakeScriptedLM(), budget or RLMBudget())


class TestParseAction:
    """Test cases for parse_action: exactly one action per turn."""

    @pytest.mark.parametrize(
        # Parameter names
        [
            "reply",
            "terminal",
            "expected_action",
            "expected_payload",
            "expected_exception",
            "expected_message_substr",
        ],
        # Parameter values
        [
            pytest.param(
                "thinking...\n<python>print(1)</python>",  # reply
                "edit",  # terminal
                "python",  # expected_action
                "print(1)",  # expected_payload
                None,  # expected_exception
                None,  # expected_message_substr
                id="python_block",
            ),
            pytest.param(
                "<edit><target>x</target></edit>",  # reply
                "edit",  # terminal
                "edit",  # expected_action
                "<target>x</target>",  # expected_payload
                None,  # expected_exception
                None,  # expected_message_substr
                id="terminal_block",
            ),
            pytest.param(
                "<python>\n```python\n    x = 1\n    print(x)\n```\n</python>",  # reply
                "edit",  # terminal
                "python",  # expected_action
                "x = 1\nprint(x)",  # expected_payload
                None,  # expected_exception
                None,  # expected_message_substr
                id="python_block_markdown_fence_and_indent_removed",
            ),
            pytest.param(
                "I will now think about the region.",  # reply
                "edit",  # terminal
                None,  # expected_action
                None,  # expected_payload
                RLMProtocolError,  # expected_exception
                "No action found",  # expected_message_substr
                id="no_action_rejected",
            ),
            pytest.param(
                "<python>a=1</python><python>b=2</python>",  # reply
                "edit",  # terminal
                None,  # expected_action
                None,  # expected_payload
                RLMProtocolError,  # expected_exception
                "Exactly one action",  # expected_message_substr
                id="two_python_blocks_rejected",
            ),
            pytest.param(
                "<edit>a</edit>\n<edit>b</edit>",  # reply
                "edit",  # terminal
                None,  # expected_action
                None,  # expected_payload
                RLMProtocolError,  # expected_exception
                "Exactly one action",  # expected_message_substr
                id="two_terminal_blocks_rejected",
            ),
            pytest.param(
                "<python>print(region)</python>\n<edit><target>x</target></edit>",  # reply
                "edit",  # terminal
                None,  # expected_action
                None,  # expected_payload
                RLMProtocolError,  # expected_exception
                "1 <python> and 1 <edit>",  # expected_message_substr
                id="python_and_terminal_together_rejected",
            ),
            pytest.param(
                "<python>s = '<edit>not an edit</edit>'\nprint(s)</python>",  # reply
                "edit",  # terminal
                "python",  # expected_action
                "s = '<edit>not an edit</edit>'\nprint(s)",  # expected_payload
                None,  # expected_exception
                None,  # expected_message_substr
                id="terminal_tag_inside_python_is_code_not_an_action",
            ),
        ],
    )
    def test_parse_action(
        self,
        reply: str,
        terminal: str,
        expected_action: str | None,
        expected_payload: str | None,
        expected_exception: type[BaseException] | None,
        expected_message_substr: str | None,
    ) -> None:
        """Test that parse_action returns the single action of a turn or rejects a malformed one.

        Args:
            reply: The model's full reply to parse.
            terminal: Name of the terminating tag, without angle brackets.
            expected_action: The action kind parse_action must return, or None when an exception is expected.
            expected_payload: The action payload parse_action must return, or None when an exception is expected.
            expected_exception: The exception parse_action must raise, or None when it must succeed.
            expected_message_substr: Substring the raised message must contain, or None when no exception is expected.
        """
        if expected_exception is not None:
            with pytest.raises(expected_exception) as exc_info:
                parse_action(reply, terminal)
            assert expected_message_substr in str(exc_info.value)
            return

        action, payload = parse_action(reply, terminal)
        assert action == expected_action
        assert payload == expected_payload


class TestSandbox:
    """Test cases for the persistent RLM sandbox."""

    def test_context_variables_are_bound_and_printed_output_is_captured(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test that context variables are bound and printed output is captured, never reaching the process stdout."""
        env = _env(region="be nice", feedback="too long")
        execution = env.execute("print(region, '|', len(feedback))")
        assert execution.error is None
        assert execution.stdout == "be nice | 8\n"
        assert capsys.readouterr().out == ""  # never reaches the process stdout

    def test_state_persists_between_executions(self) -> None:
        """Test that variables assigned in one execution are visible in the next."""
        env = _env()
        env.execute("found = [w for w in region.split() if w.startswith('n')]")
        execution = env.execute("print(found)")
        assert execution.stdout == "['nice']\n"

    def test_context_variables_are_repinned_each_execution(self) -> None:
        """Test that rebinding a context variable in the sandbox is reverted before the next execution."""
        env = _env(region="be nice")
        env.execute("region = 'hacked'\nprint(region)")
        execution = env.execute("print(region)")
        assert execution.stdout == "be nice\n"
        assert env.context["region"] == "be nice"

    def test_allowed_modules_are_prebound_and_importable(self) -> None:
        """Test that allowed modules are pre-bound and can also be imported."""
        env = _env(traces="a a b")
        execution = env.execute(
            "from collections import Counter\nimport json\n"
            "print(json.dumps(Counter(re.findall(r'\\w', traces)).most_common(1)))"
        )
        assert execution.error is None
        assert execution.stdout == '[["a", 2]]\n'

    @pytest.mark.parametrize(
        # Parameter names
        [
            "code",
            "expected_error_substr",
        ],
        # Parameter values
        [
            pytest.param(
                "import os",  # code
                "not allowed in the RLM sandbox",  # expected_error_substr
                id="import_os_blocked",
            ),
            pytest.param(
                "import subprocess",  # code
                "not allowed in the RLM sandbox",  # expected_error_substr
                id="import_subprocess_blocked",
            ),
            pytest.param(
                "import urllib.request",  # code
                "not allowed in the RLM sandbox",  # expected_error_substr
                id="import_urllib_blocked",
            ),
            pytest.param(
                "open('x')",  # code
                "NameError",  # expected_error_substr
                id="open_builtin_absent",
            ),
            pytest.param(
                "exec('1')",  # code
                "NameError",  # expected_error_substr
                id="exec_builtin_absent",
            ),
            pytest.param(
                "eval('1')",  # code
                "NameError",  # expected_error_substr
                id="eval_builtin_absent",
            ),
            pytest.param(
                "__import__('os')",  # code
                "not allowed in the RLM sandbox",  # expected_error_substr
                id="dunder_import_blocked",
            ),
            pytest.param(
                "globals()",  # code
                "NameError",  # expected_error_substr
                id="globals_builtin_absent",
            ),
        ],
    )
    def test_dangerous_operations_are_denied(self, code: str, expected_error_substr: str) -> None:
        """Test that file, interpreter, OS and network operations are denied and reported as an error.

        Args:
            code: The Python source the sandbox executes.
            expected_error_substr: Substring the reported execution error must contain.
        """
        error = _env().execute(code).error or ""
        assert expected_error_substr in error

    def test_error_traceback_shows_only_sandboxed_frames(self) -> None:
        """Test that an execution error's traceback shows only the sandboxed code's frames."""
        execution = _env().execute("x = 1\ny = x / 0")
        assert execution.error is not None
        assert 'File "<rlm>", line 2' in execution.error
        assert "ZeroDivisionError" in execution.error
        assert "rlm_environment.py" not in execution.error

    def test_execution_timeout_is_reported_not_raised(self) -> None:
        """Test that an execution exceeding the time budget is reported as an error and leaves the env usable."""
        env = _env(budget=RLMBudget(max_exec_seconds=0.1))
        execution = env.execute("while True:\n    pass")
        assert execution.error is not None
        assert "exceeded 0.1 seconds" in execution.error
        assert env.execute("print('still alive')").stdout == "still alive\n"

    def test_repl_call_budget_is_enforced(self) -> None:
        """Test that the REPL-call budget stops further executions and is reported as an error."""
        env = _env(budget=RLMBudget(max_repl_calls=1))
        assert env.execute("print(1)").error is None
        execution = env.execute("print(2)")
        assert execution.stdout == ""
        assert "repl_calls budget exhausted" in (execution.error or "")
        assert env.usage.repl_calls == 1

    @pytest.mark.parametrize(
        # Parameter names
        [
            "stdout",
            "error",
            "max_output_chars",
            "expected_render",
        ],
        # Parameter values
        [
            pytest.param(
                "x" * 20,  # stdout
                None,  # error
                5,  # max_output_chars
                "<stdout>xxxxx\n...(+15 chars cut; print less at a time)</stdout>",  # expected_render
                id="truncates_long_output",
            ),
            pytest.param(
                "",  # stdout
                None,  # error
                5,  # max_output_chars
                "<stdout>(no output; print() what you want to see)</stdout>",  # expected_render
                id="flags_silence",
            ),
            pytest.param(
                "",  # stdout
                "boom",  # error
                5,  # max_output_chars
                "<error>boom</error>",  # expected_render
                id="formats_error_only",
            ),
        ],
    )
    def test_render_truncates_output_and_flags_silence(
        self,
        stdout: str,
        error: str | None,
        max_output_chars: int,
        expected_render: str,
    ) -> None:
        """Test that render truncates long output, flags silence, and formats errors.

        Args:
            stdout: The captured output of the execution.
            error: The execution error, or None on success.
            max_output_chars: The per-execution output cap render applies.
            expected_render: The exact feedback block render must produce.
        """
        rendered = RLMExecution(stdout=stdout, error=error, calls=[]).render(max_output_chars=max_output_chars)
        assert rendered == expected_render


class TestLLMQuery:
    """Test cases for llm_query and llm_query_batched (leaf delegation)."""

    def test_llm_query_calls_the_lm_and_is_recorded(self) -> None:
        """Test that llm_query calls the LM, returns its reply, and records the delegation and usage."""
        lm = FakeEchoLM()
        env = _env(lm, region="be nice")
        execution = env.execute("answer = llm_query('is ' + region + ' short?')\nprint(answer)")
        assert execution.stdout == "IS BE NICE SHORT?\n"
        assert lm.calls == ["is be nice short?"]
        assert [(call.kind, call.prompt, call.response) for call in execution.calls] == [
            ("llm_query", "is be nice short?", "IS BE NICE SHORT?")
        ]
        assert env.usage.llm_queries == 1

    def test_llm_query_batched_returns_answers_in_order(self) -> None:
        """Test that llm_query_batched returns one answer per prompt in order and records each call."""
        lm = FakeEchoLM()
        env = _env(lm)
        execution = env.execute("print(llm_query_batched(['q' + str(i) for i in range(5)]))")
        assert execution.stdout == "['Q0', 'Q1', 'Q2', 'Q3', 'Q4']\n"
        assert sorted(lm.calls) == ["q0", "q1", "q2", "q3", "q4"]
        assert len(execution.calls) == 5
        assert env.usage.llm_queries == 5

    def test_llm_query_budget_is_tree_wide_and_surfaces_as_an_error(self) -> None:
        """Test that the tree-wide llm_query budget surfaces as an error and refuses an overshooting batch whole."""
        env = _env(FakeEchoLM(), budget=RLMBudget(max_llm_queries=2))
        assert env.execute("llm_query('a'); llm_query('b')").error is None
        execution = env.execute("llm_query('c')")
        assert "llm_queries budget exhausted (2/2 used" in (execution.error or "")
        # A batch that would overshoot is refused whole, not partially served.
        env = _env(FakeEchoLM(), budget=RLMBudget(max_llm_queries=2))
        execution = env.execute("llm_query_batched(['a', 'b', 'c'])")
        assert "3 requested" in (execution.error or "")
        assert env.usage.llm_queries == 0


class TestRLMQuery:
    """Test cases for rlm_query and rlm_query_batched (recursive delegation)."""

    def test_child_gets_only_the_explicit_context_and_its_own_environment(self) -> None:
        """Test that a child RLM sees only the explicit context, runs its own turns, and shares the tree's counters."""
        lm = FakeChildLM()
        env = _env(lm, region="be nice", traces="trace-1\ntrace-2")
        execution = env.execute("print(rlm_query(traces.splitlines()[1], 'what is this?'))")
        assert execution.error is None
        assert execution.stdout == "child saw: trace-2\n"
        # The child prompt externalizes its context and states the question; the parent's variables are absent.
        child_prompt = lm.calls[0]
        assert "what is this?" in child_prompt
        assert "`context` (str, 7 chars)" in child_prompt
        assert "trace-2" not in child_prompt
        assert "region" not in child_prompt
        # The delegation is recorded with the child's own turn log.
        (call,) = execution.calls
        assert call.kind == "rlm_query"
        assert [step.action for step in call.steps] == ["python", "final"]
        assert call.steps[0].stdout == "child saw: trace-2\n"
        assert env.usage.rlm_queries == 1
        assert env.usage.repl_calls == 2  # parent's + child's executions share one counter

    def test_child_cannot_see_parent_variables(self) -> None:
        """Test that a child RLM cannot read the parent's context variables."""
        lm = FakeScriptedLM("<python>print(region)</python>", "<final>done</final>")
        env = _env(lm, region="be nice")
        env.execute("rlm_query('ctx', 'q')")
        assert "NameError" in lm.calls[1]  # the child's second prompt carries its failed first turn

    def test_non_string_context_is_json_encoded(self) -> None:
        """Test that a non-string rlm_query context is JSON-encoded for the child."""
        lm = FakeChildLM()
        execution = _env(lm).execute("print(rlm_query({'k': [1, 2]}, 'q'))")
        assert execution.stdout == 'child saw: {\n "k": [\n  1,\n  2\n ]\n}\n'

    def test_rlm_query_batched_runs_each_child_on_its_own_context(self) -> None:
        """Test that rlm_query_batched runs each child on its own context and records every delegation."""
        lm = FakeChildLM()
        execution = _env(lm).execute("print(rlm_query_batched([('ctx-a', 'qa'), ('ctx-b', 'qb')]))")
        assert execution.stdout == "['child saw: ctx-a', 'child saw: ctx-b']\n"
        assert len(execution.calls) == 2

    def test_recursion_depth_limits_where_rlm_query_is_bound(self) -> None:
        """Test that rlm_query is bound only while the recursion depth is below the budget's limit."""
        root = _env(budget=RLMBudget(max_recursion_depth=1))
        assert "rlm_query" in root.namespace
        assert "rlm_query(context: str, prompt: str)" in root.tools_help()
        child = RLMEnvironment({"context": "x"}, root.lm, root.budget, usage=root.usage, depth=1)
        assert "rlm_query" not in child.namespace
        assert "rlm_query" not in child.tools_help()
        flat = _env(budget=RLMBudget(max_recursion_depth=0))
        assert "rlm_query" not in flat.namespace
        assert "NameError" in (flat.execute("rlm_query('c', 'q')").error or "")

    def test_child_that_never_answers_returns_a_fallback(self) -> None:
        """Test that a child which never emits <final> returns a fallback naming its last reply."""
        lm = FakeScriptedLM("<python>print(1)</python>", "<python>print(2)</python>", "still thinking")
        env = _env(lm, budget=RLMBudget(max_child_iterations=3))
        execution = env.execute("print(rlm_query('c', 'q'))")
        assert execution.stdout.startswith("(child RLM gave no <final> answer within 3 turns")
        (call,) = execution.calls
        assert [step.action for step in call.steps] == ["python", "python", "invalid"]
        assert "This is your last turn" in lm.calls[2]

    def test_rlm_query_budget_is_enforced(self) -> None:
        """Test that the tree-wide rlm_query budget stops further children and is reported as an error."""
        env = _env(FakeChildLM(), budget=RLMBudget(max_rlm_queries=1))
        assert env.execute("rlm_query('a', 'q')").error is None
        assert "rlm_queries budget exhausted" in (env.execute("rlm_query('b', 'q')").error or "")
