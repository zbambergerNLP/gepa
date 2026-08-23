# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Provide a persistent Python workspace for RLM proposals.

Context is stored as read-only variables outside the model prompt. The model
can inspect it with guarded Python, retain intermediate variables, and delegate
through ``llm_query`` or ``rlm_query``. :class:`RLMBudget` applies limits across
the delegation tree.

Execution restricts builtins, private attributes, imports, files, networking,
processes, and environment access. It runs in process and provides no security
isolation, so only trusted model output may drive it. Candidate changes leave
the workspace as typed ``<edit>`` blocks and are applied by the proposer.
"""

from __future__ import annotations

import ast
import builtins
import io
import json
import re
import signal
import textwrap
import threading
import traceback
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from gepa.proposer.reflective_mutation.base import LanguageModel
from gepa.utils.text import strip_think_tags

# Modules the guarded executor may import and pre-bind.
ALLOWED_MODULES: tuple[str, ...] = ("re", "json", "collections", "math", "statistics")

# Builtins the guarded executor exposes. Everything with I/O, code loading or interpreter
# access (open, exec, eval, compile, input, globals, ...) is deliberately absent.
_SAFE_BUILTIN_NAMES: tuple[str, ...] = (
    "abs", "all", "any", "bool", "bytes", "callable", "chr", "dict", "divmod", "enumerate",
    "filter", "float", "format", "frozenset", "hash", "int", "isinstance",
    "issubclass", "iter", "len", "list", "map", "max", "min", "next", "ord", "pow", "range", "repr",
    "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip",
    "Exception", "ArithmeticError", "AttributeError", "ImportError", "IndexError", "KeyError",
    "LookupError", "NameError", "RuntimeError", "StopIteration", "TypeError", "ValueError",
    "ZeroDivisionError",
)  # fmt: skip

_PYTHON_BLOCK_RE = re.compile(r"(?is)<python\s*>(.*?)</python>")
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\n(.*?)\n\s*```\s*$", re.DOTALL)

CHILD_RLM_PROMPT = """\
Answer one question about the supplied context.

## Question
{question}

## Your workspace
Read the context from the Python variable `context` (str, {context_chars} chars). It is read-only and available \
in a persistent Python environment. Run code with a <python>...</python> block (one per turn); whatever you \
print() is returned to you next turn, and variables you assign persist across turns. Available: plain Python, \
{modules}, and
{tools}
The executor runs in process without security isolation and accepts only trusted model output. File, network, and \
OS interfaces are unavailable. You have {turns} turns.

## Turn protocol
Reply with exactly one action: a single <python>...</python> block, or, when you know the answer, \
<final>your answer</final>.
"""

LAST_TURN_NOTE = "\n<note>This is your last turn: you must reply with your terminating action now.</note>\n"
RLM_ENVIRONMENT_PROTOCOL_VERSION = 2


def rlm_environment_contract() -> dict[str, Any]:
    """Return the behavior-bearing guarded-execution identity."""
    return {
        "version": RLM_ENVIRONMENT_PROTOCOL_VERSION,
        "execution": "trusted_model_in_process_guarded",
        "allowed_modules": list(ALLOWED_MODULES),
        "allowed_builtins": list(_SAFE_BUILTIN_NAMES),
        "private_names_and_attributes": "rejected_by_ast",
        "batched_delegation": "sequential",
        "timeout_off_main_thread": "reject_execution",
    }


@dataclass(frozen=True)
class RLMBudget:
    """Hard limits of one RLM proposal, covering the root and every child it spawns.

    Turn limits are per RLM (the root gets ``max_root_iterations`` turns, each
    child ``max_child_iterations``); the REPL, ``llm_query`` and ``rlm_query``
    limits are totals over the whole recursion tree, so a bounded root loop
    cannot fan out into an unbounded amount of computation.

    Args:
        max_root_iterations: Model turns of the root Editor.
        max_child_iterations: Model turns of each child RLM spawned by ``rlm_query``.
        max_repl_calls: Total ``<python>`` executions (root + children).
        max_llm_queries: Total leaf ``llm_query`` calls (root + children).
        max_rlm_queries: Total ``rlm_query`` children spawned.
        max_recursion_depth: How many levels of ``rlm_query`` may nest. ``0``
            removes ``rlm_query`` from the environment altogether; ``1`` lets the
            root delegate but its children cannot.
        max_exec_seconds: Wall-clock limit of one ``<python>`` execution,
            excluding time spent inside ``llm_query``/``rlm_query`` (enforced
            on the main thread, where signals are available; execution fails
            closed on other threads). ``None`` disables it.
        max_output_chars: Longest printed output returned to the model per
            execution; the rest is cut with a note.
    """

    max_root_iterations: int = 6
    max_child_iterations: int = 4
    max_repl_calls: int = 12
    max_llm_queries: int = 8
    max_rlm_queries: int = 2
    max_recursion_depth: int = 1
    max_exec_seconds: float | None = 5.0
    max_output_chars: int = 4000

    def __post_init__(self) -> None:
        """Reject limits that cannot produce a bounded, well-defined run."""
        positive = {
            "max_root_iterations": self.max_root_iterations,
            "max_child_iterations": self.max_child_iterations,
            "max_output_chars": self.max_output_chars,
        }
        nonnegative = {
            "max_repl_calls": self.max_repl_calls,
            "max_llm_queries": self.max_llm_queries,
            "max_rlm_queries": self.max_rlm_queries,
            "max_recursion_depth": self.max_recursion_depth,
        }
        for name, value in positive.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer; got {value!r}.")
        for name, value in nonnegative.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer; got {value!r}.")
        seconds = self.max_exec_seconds
        if seconds is not None and (isinstance(seconds, bool) or not isinstance(seconds, int | float) or seconds <= 0):
            raise ValueError(f"max_exec_seconds must be positive or None; got {seconds!r}.")

    @property
    def max_model_calls(self) -> int:
        """Return the tree-wide upper bound on root, child, and leaf LM calls."""
        return (
            self.max_root_iterations
            + self.max_rlm_queries * self.max_child_iterations
            + self.max_llm_queries
        )


class RLMProtocolError(ValueError):
    """The model's reply does not follow the one-action-per-turn protocol.

    Raised by :func:`parse_action` when a turn carries no action or more than
    one; the RLM loop feeds the message back to the model as an ``<error>`` and
    lets it retry rather than guessing which action was meant.
    """


class RLMBudgetError(RuntimeError):
    """A tree-wide budget of :class:`RLMBudget` is used up.

    Raised by :meth:`_Usage.charge` from inside the delegation primitives
    (``llm_query``, ``rlm_query`` and their batched forms) and before each
    ``<python>`` execution; guarded code sees it as an ordinary exception, so
    the message tells the model to finish with its terminating action.
    """


class RLMExecutionTimeout(BaseException):
    """One ``<python>`` execution ran past ``max_exec_seconds``.

    Derives from ``BaseException`` so a bare ``except Exception:`` in the model's
    own code cannot swallow it.
    """


@dataclass
class RLMCall:
    """One delegation made from inside the environment.

    Attributes:
        kind: ``"llm_query"`` for a leaf LM call or ``"rlm_query"`` for a child RLM.
        prompt: The question that was delegated.
        response: The reply returned to the guarded code.
        steps: The child RLM's own turn log for ``rlm_query``; empty for the
            leaf ``llm_query``.
    """

    kind: str
    prompt: str
    response: str
    steps: list[RLMStep] = field(default_factory=list)


@dataclass
class RLMStep:
    """One model turn of an RLM (root or child), for debugging optimization runs.

    Attributes:
        iteration: 1-based turn number within that RLM's loop.
        action: ``"python"``, the terminating action (``"edit"`` for the root,
            ``"final"`` for a child) or ``"invalid"`` (protocol violation).
        code: The executed source; set for ``"python"`` turns only.
        stdout: What the code printed; set for ``"python"`` turns only.
        error: The execution error, the protocol error, or why a terminating
            action was rejected; ``None`` when the turn succeeded.
        child_calls: Delegations made during a ``"python"`` turn.
    """

    iteration: int
    action: str
    code: str | None = None
    stdout: str | None = None
    error: str | None = None
    child_calls: list[RLMCall] = field(default_factory=list)


@dataclass
class RLMExecution:
    """What one ``<python>`` execution produced.

    Attributes:
        stdout: Everything the code printed, unbounded (:meth:`render` applies
            the per-execution cap).
        error: The formatted exception, timeout, or budget message when the
            execution failed; ``None`` on success.
        calls: The delegations (``llm_query``/``rlm_query``) the code made, in
            call order.
    """

    stdout: str
    error: str | None
    calls: list[RLMCall]

    def render(self, max_output_chars: int) -> str:
        """Format the outcome as the feedback block appended to the model's transcript.

        Args:
            max_output_chars: Longest printed output to show; the remainder is
                cut with a note telling the model to print less at a time.

        Returns:
            A ``<stdout>...</stdout>`` block (with a placeholder when nothing was
            printed and nothing failed), followed by an ``<error>...</error>``
            block when the execution failed.
        """
        out = self.stdout
        if len(out) > max_output_chars:
            out = out[:max_output_chars] + f"\n...(+{len(out) - max_output_chars} chars cut; print less at a time)"
        if not out and self.error is None:
            out = "(no output; print() what you want to see)"
        rendered = f"<stdout>{out}</stdout>" if out else ""
        if self.error is not None:
            rendered += f"\n<error>{self.error}</error>"
        return rendered.lstrip("\n")


class _Usage:
    """Tree-wide counters shared by a root environment and all its children.

    One instance is created by the root :class:`RLMEnvironment` and handed to
    every child, so the REPL, ``llm_query`` and ``rlm_query`` limits of
    :class:`RLMBudget` bound the whole recursion tree rather than each node.
    The lock keeps budget charging atomic if callers invoke an environment concurrently.
    """

    def __init__(self):
        """Start every counter at zero."""
        self.repl_calls = 0
        self.llm_queries = 0
        self.rlm_queries = 0
        self._lock = threading.Lock()

    def charge(self, counter: str, limit: int, count: int = 1) -> None:
        """Consume ``count`` units of ``counter`` or raise if that would pass ``limit``.

        Args:
            counter: Name of the counter attribute (``"repl_calls"``,
                ``"llm_queries"`` or ``"rlm_queries"``).
            limit: The budget for that counter.
            count: Units to consume; batched primitives charge their whole batch
                up front so a batch either fits entirely or is rejected.

        Raises:
            RLMBudgetError: Fewer than ``count`` units remain; nothing is charged.
        """
        with self._lock:
            used = getattr(self, counter)
            if used + count > limit:
                raise RLMBudgetError(
                    f"{counter} budget exhausted ({used}/{limit} used, {count} requested); "
                    "finish with your terminating action."
                )
            setattr(self, counter, used + count)


def parse_action(reply: str, terminal: str) -> tuple[str, str]:
    """Read the single action of a model turn.

    A turn is exactly one ``<python>`` block or exactly one ``<terminal>``
    block (``<edit>`` for the root Editor, ``<final>`` for a child). Anything
    else (no block, several blocks, or both kinds) is a protocol violation, so a
    reply that ambiguously runs code *and* commits an edit is rejected instead
    of one action being picked silently. Terminal tags inside a python block are
    code, not actions.

    Args:
        reply: The model's full reply, think-tags already stripped.
        terminal: Name of the terminating tag, without angle brackets.

    Returns:
        ``("python", code)`` with markdown fences removed and dedented, or
        ``(terminal, inner_text)``.

    Raises:
        RLMProtocolError: Zero or more than one action found.
    """
    python_blocks = _PYTHON_BLOCK_RE.findall(reply)
    outside_python = _PYTHON_BLOCK_RE.sub("", reply)
    terminal_blocks = re.findall(rf"(?is)<{terminal}(?:\s[^>]*)?>(.*?)</{terminal}>", outside_python)
    found = len(python_blocks) + len(terminal_blocks)
    if found == 0:
        raise RLMProtocolError(
            f"No action found. Reply with exactly one <python>...</python> block or one <{terminal}>...</{terminal}> block."
        )
    if found > 1:
        raise RLMProtocolError(
            f"Exactly one action per turn is allowed; found {len(python_blocks)} <python> and "
            f"{len(terminal_blocks)} <{terminal}> blocks."
        )
    if python_blocks:
        code = python_blocks[0]
        fenced = _FENCE_RE.match(code)
        if fenced is not None:
            code = fenced.group(1)
        return "python", textwrap.dedent(code).strip("\n")
    return terminal, terminal_blocks[0]


def _format_error(exc: BaseException) -> str:
    """Render an exception raised by guarded code, keeping only the frames of that code.

    Args:
        exc: The exception caught around ``exec`` of a ``<python>`` block.

    Returns:
        A traceback limited to frames from the ``<rlm>`` pseudo-file plus the
        exception line, so the model sees its own code and not the executor
        internals.
    """
    frames = [frame for frame in traceback.extract_tb(exc.__traceback__) if frame.filename == "<rlm>"]
    lines = traceback.format_list(frames) + traceback.format_exception_only(type(exc), exc)
    return "".join(lines).rstrip()


class RLMEnvironment:
    """The persistent guarded in-process Python environment one trusted RLM drives.

    Args:
        context: Read-only context variables (name -> text), e.g. ``region``,
            ``component``, ``feedback``, ``traces`` for the root Editor or
            ``context`` for a child. They are re-pinned before every execution.
        lm: The language model used by ``llm_query`` and by child RLMs.
        budget: Limits shared by the whole recursion tree.
        usage: Tree-wide counters; ``None`` starts a new tree (root).
        depth: Recursion depth of this environment (``0`` for the root).
    """

    def __init__(
        self,
        context: dict[str, str],
        lm: LanguageModel,
        budget: RLMBudget,
        *,
        usage: _Usage | None = None,
        depth: int = 0,
    ):
        """Build the guarded namespace: safe builtins, modules, delegation primitives, and context."""
        self.context = dict(context)
        self.lm = lm
        self.budget = budget
        self.usage = usage if usage is not None else _Usage()
        self.depth = depth
        self._stdout = io.StringIO()
        self._calls: list[RLMCall] = []
        self._timer_active = False
        safe_builtins = {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES}
        safe_builtins["__import__"] = self._restricted_import
        safe_builtins["print"] = self._print
        self.namespace: dict[str, Any] = {"__builtins__": safe_builtins, "__name__": "__rlm__"}
        for module_name in ALLOWED_MODULES:
            self.namespace[module_name] = builtins.__import__(module_name)
        self.namespace["llm_query"] = self.llm_query
        self.namespace["llm_query_batched"] = self.llm_query_batched
        if self.can_recurse:
            self.namespace["rlm_query"] = self.rlm_query
            self.namespace["rlm_query_batched"] = self.rlm_query_batched
        self.namespace.update(self.context)

    @property
    def can_recurse(self) -> bool:
        """Whether ``rlm_query`` is bound in this environment.

        Returns:
            ``True`` while ``depth`` is below ``budget.max_recursion_depth``, so
            the deepest allowed level gets only the leaf ``llm_query`` primitives.
        """
        return self.depth < self.budget.max_recursion_depth

    def tools_help(self) -> str:
        """Describe the delegation primitives bound in this environment, for the prompt.

        Returns:
            A bullet list with one line per primitive: always ``llm_query`` and
            ``llm_query_batched``, plus ``rlm_query`` and ``rlm_query_batched``
            when :attr:`can_recurse` is ``True``.
        """
        lines = [
            "- llm_query(prompt: str) -> str: ask a sub-model one question (a plain LM call, no tools).",
            "- llm_query_batched(prompts: list[str]) -> list[str]: the same for a bounded sequential batch.",
        ]
        if self.can_recurse:
            lines += [
                "- rlm_query(context: str, prompt: str) -> str: delegate one question about one specific piece of "
                "context (pass exactly the text it needs, e.g. a subset of the traces) to a child agent that gets "
                "its own Python environment and turns.",
                "- rlm_query_batched(requests: list[tuple[str, str]]) -> list[str]: many (context, prompt) "
                "delegations in a bounded sequential batch.",
            ]
        return "\n".join(lines)

    def execute(self, code: str) -> RLMExecution:
        """Run one ``<python>`` block in the persistent namespace and report what it did.

        The context variables are re-pinned first, so anything the model
        rebound last turn is restored; everything else it assigned survives.
        Output is whatever the code printed; an exception (including a budget
        exhaustion raised by a delegation primitive, or the execution timeout)
        becomes ``error`` rather than propagating.

        Args:
            code: The Python source to execute.

        Returns:
            The captured output, the error if any, and the delegations made.
        """
        try:
            self.usage.charge("repl_calls", self.budget.max_repl_calls)
        except RLMBudgetError as exc:
            return RLMExecution(stdout="", error=str(exc), calls=[])
        self.namespace.update(self.context)
        self._stdout = io.StringIO()
        self._calls = []
        error: str | None = None
        try:
            _validate_code(code)
            with self._time_limit():
                exec(compile(code, "<rlm>", "exec"), self.namespace)
        except RLMExecutionTimeout as exc:
            error = str(exc)
        except Exception as exc:
            error = _format_error(exc)
        return RLMExecution(stdout=self._stdout.getvalue(), error=error, calls=self._calls)

    def llm_query(self, prompt: str) -> str:
        """Leaf delegation: one plain LM call, bound as ``llm_query`` in the namespace.

        The execution timer is paused for the duration of the call, so LM latency
        does not count against ``max_exec_seconds``.

        Args:
            prompt: The question for the sub-model; coerced to ``str``.

        Returns:
            The sub-model's reply with think-tags stripped.

        Raises:
            RLMBudgetError: The tree's ``max_llm_queries`` budget is exhausted.
        """
        self.usage.charge("llm_queries", self.budget.max_llm_queries)
        with self._timer_paused():
            response = strip_think_tags(self.lm(str(prompt)))
        self._calls.append(RLMCall(kind="llm_query", prompt=str(prompt), response=response))
        return response

    def llm_query_batched(self, prompts: Sequence[str]) -> list[str]:
        """Run a bounded batch of independent ``llm_query`` calls sequentially.

        The whole batch is charged to the budget up front, so it either fits or
        is rejected as a unit.

        Args:
            prompts: The questions; each is coerced to ``str``.

        Returns:
            One reply per prompt, in the same order.

        Raises:
            RLMBudgetError: Fewer than ``len(prompts)`` ``llm_query`` calls remain.
        """
        prompts = [str(prompt) for prompt in prompts]
        self.usage.charge("llm_queries", self.budget.max_llm_queries, count=len(prompts))
        with self._timer_paused():
            responses = [strip_think_tags(self.lm(prompt)) for prompt in prompts]
        for prompt, response in zip(prompts, responses, strict=True):
            self._calls.append(RLMCall(kind="llm_query", prompt=prompt, response=response))
        return responses

    def rlm_query(self, context: Any, prompt: str) -> str:
        """Recursive delegation: run a child RLM over ``context`` and return its final answer.

        The child sees only ``context`` (a string; anything else is JSON-encoded)
        in its own fresh environment, so recursion stays meaningful: the parent
        chooses exactly what the child works on rather than the child inheriting
        the parent's whole workspace. The child shares this tree's budget
        counters and sits one level deeper.

        Args:
            context: The material the child may work on; strings pass through,
                anything else is JSON-encoded.
            prompt: The question the child must answer about ``context``.

        Returns:
            The child's ``<final>`` answer, or an explanatory placeholder when it
            ran out of turns without one.

        Raises:
            RLMBudgetError: The tree's ``max_rlm_queries`` budget is exhausted.
        """
        self.usage.charge("rlm_queries", self.budget.max_rlm_queries)
        with self._timer_paused():
            answer, steps = self._run_child(_as_text(context), str(prompt))
        self._calls.append(RLMCall(kind="rlm_query", prompt=str(prompt), response=answer, steps=steps))
        return answer

    def rlm_query_batched(self, requests: Sequence[tuple[Any, str]]) -> list[str]:
        """Run a bounded batch of independent child RLMs sequentially.

        The whole batch is charged to the budget up front, so it either fits or
        is rejected as a unit.

        Args:
            requests: ``(context, prompt)`` pairs, each handled like one
                :meth:`rlm_query` call.

        Returns:
            One child answer per request, in the same order.

        Raises:
            RLMBudgetError: Fewer than ``len(requests)`` ``rlm_query`` children remain.
        """
        pairs = [(_as_text(context), str(prompt)) for context, prompt in requests]
        self.usage.charge("rlm_queries", self.budget.max_rlm_queries, count=len(pairs))
        with self._timer_paused():
            results = [self._run_child(context, prompt) for context, prompt in pairs]
        for (_, prompt), (answer, steps) in zip(pairs, results, strict=True):
            self._calls.append(RLMCall(kind="rlm_query", prompt=prompt, response=answer, steps=steps))
        return [answer for answer, _ in results]

    def _run_child(self, context: str, question: str) -> tuple[str, list[RLMStep]]:
        """Drive a child RLM to its ``<final>`` answer (or out of turns) and log its steps.

        The child gets a fresh :class:`RLMEnvironment` one level deeper whose
        only context variable is ``context``; protocol violations and execution
        results are fed back into its transcript turn by turn, and the last turn
        carries a note asking it to finish.

        Args:
            context: The text the child works on (already coerced by :func:`_as_text`).
            question: The question the child must answer.

        Returns:
            ``(answer, steps)``: the stripped ``<final>`` payload (or a
            placeholder naming the child's last reply when it never finished)
            and the child's turn log.
        """
        child = RLMEnvironment({"context": context}, self.lm, self.budget, usage=self.usage, depth=self.depth + 1)
        turns = self.budget.max_child_iterations
        base_prompt = CHILD_RLM_PROMPT.format(
            question=question,
            context_chars=len(context),
            modules=", ".join(ALLOWED_MODULES),
            tools=child.tools_help(),
            turns=turns,
        )
        transcript = ""
        steps: list[RLMStep] = []
        raw = ""
        for iteration in range(1, turns + 1):
            note = LAST_TURN_NOTE if iteration == turns else ""
            raw = strip_think_tags(self.lm(base_prompt + transcript + note))
            try:
                action, payload = parse_action(raw, "final")
            except RLMProtocolError as exc:
                steps.append(RLMStep(iteration, "invalid", error=str(exc)))
                transcript += f"\n\n<your-output>{raw}</your-output>\n<error>{exc}</error>\n"
                continue
            if action == "final":
                steps.append(RLMStep(iteration, "final"))
                return payload.strip(), steps
            execution = child.execute(payload)
            steps.append(RLMStep(iteration, "python", payload, execution.stdout, execution.error, execution.calls))
            transcript += f"\n\n<your-output>{raw}</your-output>\n{execution.render(self.budget.max_output_chars)}\n"
        return f"(child RLM gave no <final> answer within {turns} turns; its last reply was: {raw.strip()})", steps

    def _print(self, *args: Any, **kwargs: Any) -> None:
        """Capture guarded ``print`` output without writing to process stdout.

        Args:
            *args: Positional arguments forwarded to :func:`print`.
            **kwargs: Keyword arguments forwarded to :func:`print`; ``file`` is
                overridden with the capture buffer.
        """
        kwargs["file"] = self._stdout
        builtins.print(*args, **kwargs)

    def _restricted_import(
        self,
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        """Allow ``__import__`` only for the already-bound :data:`ALLOWED_MODULES`.

        Args:
            name: Dotted module name being imported.
            globals: Ignored beyond being forwarded to the real ``__import__``.
            locals: Ignored beyond being forwarded to the real ``__import__``.
            fromlist: Names requested by ``from ... import ...``; forwarded.
            level: Relative-import level; anything but ``0`` is refused.

        Returns:
            The imported module, as the real ``__import__`` would return it.

        Raises:
            ImportError: The top-level package is not in :data:`ALLOWED_MODULES`
                or the import is relative.
        """
        if level != 0 or name.split(".")[0] not in ALLOWED_MODULES:
            raise ImportError(
                f"import of {name!r} is not allowed in the guarded RLM executor; available modules: "
                f"{', '.join(ALLOWED_MODULES)}."
            )
        return builtins.__import__(name, globals, locals, fromlist, level)

    @contextmanager
    def _time_limit(self):
        """Bound one execution by ``max_exec_seconds`` or reject it when unavailable.

        Arms an ``ITIMER_REAL`` alarm whose handler raises
        :class:`RLMExecutionTimeout`; the previous handler and timer are restored
        on exit. Skipped when the budget disables the limit, the platform has no
        ``SIGALRM``. When a hard timeout cannot be installed (including worker
        threads), execution is rejected instead of running without a bound.

        Yields:
            Nothing; the body runs under the alarm.
        """
        seconds = self.budget.max_exec_seconds
        if seconds is None:
            yield
            return
        if not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
            raise RLMExecutionTimeout(
                "guarded Python execution is disabled because a hard timeout cannot be enforced on this thread."
            )

        def _on_alarm(signum: int, frame: Any) -> None:
            raise RLMExecutionTimeout(f"execution exceeded {seconds} seconds and was stopped.")

        previous = signal.signal(signal.SIGALRM, _on_alarm)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        self._timer_active = True
        try:
            yield
        finally:
            self._timer_active = False
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)

    @contextmanager
    def _timer_paused(self):
        """Stop the execution timer while waiting on an LM or a child, then resume the remaining time.

        A no-op when no timer is armed (e.g. off the main thread), so the
        delegation primitives can wrap every wait unconditionally.

        Yields:
            Nothing; the body runs with the alarm suspended.
        """
        if not self._timer_active:
            yield
            return
        remaining, _ = signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            yield
        finally:
            if remaining > 0:
                signal.setitimer(signal.ITIMER_REAL, remaining)


def _as_text(context: Any) -> str:
    """Coerce a child's context to text: strings pass through, anything else is JSON-encoded.

    Args:
        context: Whatever the model passed to ``rlm_query`` (a string, or e.g. a
            list or dict it built in the guarded executor).

    Returns:
        ``context`` unchanged if it is a string, otherwise its indented JSON
        rendering (non-serializable values fall back to ``str``).
    """
    if isinstance(context, str):
        return context
    return json.dumps(context, ensure_ascii=False, indent=1, default=str)


def _validate_code(code: str) -> None:
    """Reject reflective access to private interpreter and object internals.

    Args:
        code: Model-generated Python source.

    Raises:
        ValueError: The code references a private name or attribute.
        SyntaxError: The code is not valid Python.
    """
    tree = ast.parse(code, filename="<rlm>", mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ValueError(f"private name access is not allowed: {node.id!r}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError(f"private attribute access is not allowed: {node.attr!r}")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr
            in {
                "format",
                "format_map",
            }
        ):
            raise ValueError("str.format-style reflective field traversal is not allowed.")
