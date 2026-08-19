# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for the RLM Proposer role (:mod:`gepa.proposer.reflective_mutation.rlm_proposer`).

Covers the Editor's externalized context (the prompt names variables, not their
content), its sandboxed REPL turns and delegations, the single forced and
validated ``<edit>`` that terminates the loop, and the budget, trace and safety
guarantees. The Editor LM is a scripted fake; the sandbox, document template and
edit machinery are the real implementations.

Expected usage:
```bash
pytest tests/test_rlm_proposer.py -vv
```
"""

# Third-party imports
import pytest

# Local imports
from gepa.proposer.reflective_mutation.rlm_environment import RLMBudget
from gepa.proposer.reflective_mutation.rlm_proposer import RLMProposer, RLMResult
from gepa.strategies.document_template import TEMPLATES, EditTarget, MalformedDocumentError
from gepa.strategies.edit_tools import EditTool

# ====================== #
# Test Fakes and Helpers #
# ====================== #

TEMPLATE = TEMPLATES["prompt"]
PROMPT = TEMPLATE.render({"Role": "you are a helper", "Rules": "- be nice\n- be brief"})
FEEDBACK = "answers were rude and rambling"
TRACES = "trace-1: input=hi output=whatever\ntrace-2: input=help output=no"
REPLACE_EDIT = "<edit><target>be nice</target><text>be kind</text></edit>"


class FakeEditorLM:
    """Fake Editor LM: records the prompts it receives and replays a fixed list of turn outputs."""

    def __init__(self, *outputs: str) -> None:
        """Store the scripted turn outputs to replay in order.

        Args:
            outputs: The replies to return on successive calls, in order.
        """
        self.outputs = list(outputs)
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        """Record the prompt and return the next scripted output.

        Args:
            prompt: The prompt handed to the Editor for this turn.

        Returns:
            The next scripted output, or a default reply once the script is exhausted.
        """
        self.calls.append(prompt)
        if not self.outputs:
            return "I have nothing more to do."
        return self.outputs.pop(0)


def _target(section: str | None = "Rules") -> EditTarget:
    """Build an edit target for the ``sys`` component's ``section`` (whole document when None)."""
    return EditTarget("sys", section)


def _propose(
    lm: FakeEditorLM,
    tool: EditTool = EditTool.REPLACE_TEXT,
    section: str | None = "Rules",
    *,
    budget: RLMBudget | None = None,
    max_chars: int | None = None,
) -> RLMResult:
    """Run the Editor over the canonical prompt with the shared feedback and traces.

    Args:
        lm: The scripted Editor LM to drive the loop.
        tool: The forced edit tool the terminating edit must satisfy.
        section: The region the edit is scoped to, or None for a whole-document edit.
        budget: The RLM budget to run under, or None for the default budget.
        max_chars: The per-edit character budget, or None for no limit.

    Returns:
        The proposal result produced by the Editor loop.
    """
    rlm = RLMProposer(lm, TEMPLATE, budget=budget)
    return rlm.propose(PROMPT, _target(section), tool, "", FEEDBACK, TRACES, max_chars)


class TestExternalizedContext:
    """Test cases for the Editor's externalized context: the prompt names variables, not their content."""

    def test_prompt_names_variables_but_carries_no_context_text(self) -> None:
        """Test that the prompt names each context variable by type and size but embeds none of its text."""
        lm = FakeEditorLM(REPLACE_EDIT)
        rlm = RLMProposer(lm, TEMPLATE)
        rlm.propose(PROMPT, _target("Rules"), EditTool.REPLACE_TEXT, "shorten rule 2", FEEDBACK, TRACES, None)
        prompt = lm.calls[0]
        assert "You are the Editor" in prompt
        assert "Required operation: REPLACE_TEXT on region 'Rules'" in prompt
        assert "shorten rule 2" in prompt  # planner guidance is in the prompt...
        variables = {"region": "- be nice\n- be brief", "component": PROMPT, "feedback": FEEDBACK, "traces": TRACES}
        for name, value in variables.items():
            assert f"- {name}: str, {len(value):,} chars" in prompt
            assert value not in prompt  # ...but the context itself is not
        assert "<python>" in prompt and "<edit>" in prompt
        assert "llm_query(prompt: str)" in prompt
        assert "rlm_query(context: str, prompt: str)" in prompt

    def test_python_turn_reads_context_and_output_is_fed_back(self) -> None:
        """Test that a Python turn can read the externalized context and its stdout is fed back."""
        lm = FakeEditorLM("<python>print(region.splitlines()[0]); print(len(traces))</python>", REPLACE_EDIT)
        result = _propose(lm)
        assert result.changed is True
        assert result.repl_calls == 1
        assert f"<stdout>- be nice\n{len(TRACES)}\n</stdout>" in lm.calls[1]
        assert "<your-output>" in lm.calls[1]

    def test_repl_state_persists_across_turns(self) -> None:
        """Test that names bound in one Python turn remain available in the next."""
        lm = FakeEditorLM(
            "<python>bad = [l for l in traces.splitlines() if 'whatever' in l]</python>",
            "<python>print(len(bad), bad[0][:7])</python>",
            REPLACE_EDIT,
        )
        result = _propose(lm)
        assert result.changed is True
        assert "<stdout>1 trace-1\n</stdout>" in lm.calls[2]

    def test_python_cannot_mutate_the_candidate(self) -> None:
        """Test that rebinding a context variable in the sandbox cannot change the real edited document."""
        # Rebinding `region` in the sandbox changes nothing: the edit still applies to the real region.
        lm = FakeEditorLM("<python>region = 'be rude'\ncomponent = ''\nprint(region)</python>", REPLACE_EDIT)
        result = _propose(lm)
        assert "<stdout>be rude\n</stdout>" in lm.calls[1]
        assert result.new_text == PROMPT.replace("be nice", "be kind")

    def test_python_error_is_fed_back_and_loop_continues(self) -> None:
        """Test that a Python exception is reported back to the Editor and the loop continues."""
        lm = FakeEditorLM("<python>print(undefined_name)</python>", REPLACE_EDIT)
        result = _propose(lm)
        assert result.changed is True
        assert "<error>" in lm.calls[1] and "NameError" in lm.calls[1]
        assert result.steps[0].error is not None

    def test_llm_query_from_python(self) -> None:
        """Test that llm_query from Python calls a leaf sub-model and returns its answer."""
        lm = FakeEditorLM(
            "<python>print(llm_query('is ' + region.splitlines()[0] + ' rude?'))</python>",
            "no, it is fine",  # the leaf sub-model's answer
            REPLACE_EDIT,
        )
        result = _propose(lm)
        assert lm.calls[1] == "is - be nice rude?"
        assert "<stdout>no, it is fine\n</stdout>" in lm.calls[2]
        assert result.llm_queries == 1
        assert result.steps[0].child_calls[0].kind == "llm_query"

    def test_rlm_query_from_python_runs_a_child_over_explicit_context(self) -> None:
        """Test that rlm_query runs a child Editor over an explicit context it is handed, not the parent's."""
        lm = FakeEditorLM(
            "<python>print(rlm_query(traces.splitlines()[1], 'which input?'))</python>",
            "<python>print(context.split('input=')[1].split()[0])</python>",  # child turn 1
            "<final>help</final>",  # child turn 2
            REPLACE_EDIT,
        )
        result = _propose(lm)
        assert result.changed is True
        assert "which input?" in lm.calls[1]  # the child gets the question...
        assert "trace-2" not in lm.calls[1]  # ...but not the context text
        assert "<stdout>help\n</stdout>" in lm.calls[2]  # the child's own turn feeds back to the child
        assert "<stdout>help\n</stdout>" in lm.calls[3]  # the child's answer feeds back to the root
        assert result.rlm_queries == 1
        assert result.repl_calls == 2
        (call,) = result.steps[0].child_calls
        assert call.kind == "rlm_query"
        assert [step.action for step in call.steps] == ["python", "final"]


class TestForcedEdit:
    """Test cases for the terminating action: one forced, validated edit."""

    def test_replace_edit_applies_to_selected_region(self) -> None:
        """Test that a REPLACE edit rewrites its target inside the selected region and leaves other sections intact."""
        result = _propose(FakeEditorLM("<edit><target>be nice</target><text>be concise</text></edit>"))
        assert result.changed is True
        assert "be concise" in result.new_text
        assert "## Role" in result.new_text  # untouched section survives
        assert result.executed_edit == ["DELETE 'be nice'", "INSERT 'be concise'"]

    def test_insert_edit_at_anchor(self) -> None:
        """Test that an INSERT edit places its text at the given anchor."""
        lm = FakeEditorLM("<edit><anchor>be nice</anchor><where>after</where><text> and be honest</text></edit>")
        result = _propose(lm, EditTool.INSERT_TEXT)
        assert result.changed is True
        assert "be nice and be honest" in result.new_text

    def test_insert_edit_with_empty_anchor_appends(self) -> None:
        """Test that an INSERT edit with an empty anchor appends to the region."""
        lm = FakeEditorLM("<edit><anchor></anchor><where>after</where><text>\n- be honest</text></edit>")
        result = _propose(lm, EditTool.INSERT_TEXT)
        assert TEMPLATE.parse(result.new_text)["Rules"] == "- be nice\n- be brief\n- be honest"

    def test_delete_edit_removes_target(self) -> None:
        """Test that a DELETE edit removes its target from the region."""
        result = _propose(FakeEditorLM("<edit><target>\n- be brief</target></edit>"), EditTool.DELETE_TEXT)
        assert result.changed is True
        assert TEMPLATE.parse(result.new_text)["Rules"] == "- be nice"

    def test_move_edit(self) -> None:
        """Test that a MOVE edit relocates its target before the anchor verbatim."""
        lm = FakeEditorLM("<edit><target>- be brief</target><anchor>- be nice</anchor><where>before</where></edit>")
        result = _propose(lm, EditTool.MOVE_TEXT)
        assert result.executed_edit == ["DELETE '- be brief'", "INSERT (moved) '- be brief' before '- be nice'"]
        assert TEMPLATE.parse(result.new_text)["Rules"] == "- be brief- be nice"  # verbatim: no whitespace repair

    def test_edit_parsed_under_forced_tool_schema_only(self) -> None:
        """Test that a block is parsed under the forced tool's schema only, ignoring fields it does not read."""
        # A REPLACE-shaped block under a forced DELETE tool is parsed as DELETE:
        # only <target> is read, the <text> is ignored, so it is a pure deletion.
        result = _propose(
            FakeEditorLM("<edit><target>be nice</target><text>ignored</text></edit>"), EditTool.DELETE_TEXT
        )
        assert result.executed_edit == ["DELETE 'be nice'"]
        assert "ignored" not in result.new_text

    def test_whole_document_edit(self) -> None:
        """Test that a whole-document edit (no region) rewrites its target anywhere in the document."""
        result = _propose(FakeEditorLM("<edit><target>helper</target><text>assistant</text></edit>"), section=None)
        assert result.new_text == PROMPT.replace("helper", "assistant")


class TestEditValidation:
    """Test cases for validating the forced edit against its tool schema."""

    @pytest.mark.parametrize(
        # Parameter names
        [
            "tool",
            "block",
            "expected_message",
        ],
        # Parameter values
        [
            pytest.param(
                EditTool.INSERT_TEXT,  # tool
                "<edit><anchor>be nice</anchor><where>after</where></edit>",  # block
                "requires a <text> field",  # expected_message
                id="insert_missing_text",
            ),
            pytest.param(
                EditTool.INSERT_TEXT,  # tool
                "<edit><anchor>be nice</anchor><text>x</text></edit>",  # block
                "requires a <where> field",  # expected_message
                id="insert_missing_where",
            ),
            pytest.param(
                EditTool.INSERT_TEXT,  # tool
                "<edit><anchor>be nice</anchor><where>above</where><text>x</text></edit>",  # block
                "Invalid <where> value 'above'",  # expected_message
                id="insert_invalid_where_above",
            ),
            pytest.param(
                EditTool.DELETE_TEXT,  # tool
                "<edit><text>be nice</text></edit>",  # block
                "requires a <target> field",  # expected_message
                id="delete_missing_target",
            ),
            pytest.param(
                EditTool.DELETE_TEXT,  # tool
                "<edit><target>  </target></edit>",  # block
                "requires a non-empty <target>",  # expected_message
                id="delete_empty_target",
            ),
            pytest.param(
                EditTool.REPLACE_TEXT,  # tool
                "<edit><target>be nice</target></edit>",  # block
                "requires a <text> field",  # expected_message
                id="replace_missing_text",
            ),
            pytest.param(
                EditTool.MOVE_TEXT,  # tool
                "<edit><target>be nice</target><where>after</where></edit>",  # block
                "requires a <anchor> field",  # expected_message
                id="move_missing_anchor",
            ),
            pytest.param(
                EditTool.MOVE_TEXT,  # tool
                "<edit><target>be nice</target><anchor>be brief</anchor><where>below</where></edit>",  # block
                "Invalid <where>",  # expected_message
                id="move_invalid_where_below",
            ),
        ],
    )
    def test_missing_or_invalid_fields_are_rejected_and_fed_back(
        self,
        tool: EditTool,
        block: str,
        expected_message: str,
    ) -> None:
        """Test that an edit block missing or misusing a field required by its tool is rejected and fed back.

        Args:
            tool: The forced edit tool whose schema the block must satisfy.
            block: The malformed ``<edit>`` block the Editor emits.
            expected_message: Substring the drop reason and the step error must both contain.
        """
        lm = FakeEditorLM(block)
        result = _propose(lm, tool, budget=RLMBudget(max_root_iterations=1))
        assert result.changed is False
        assert result.new_text == PROMPT
        assert result.dropped_reason is not None and expected_message in result.dropped_reason
        assert result.steps[0].action == "edit" and expected_message in (result.steps[0].error or "")

    def test_replace_with_empty_text_is_allowed(self) -> None:
        """Test that a REPLACE edit with empty replacement text is allowed and deletes the target."""
        result = _propose(FakeEditorLM("<edit><target>\n- be brief</target><text></text></edit>"))
        assert TEMPLATE.parse(result.new_text)["Rules"] == "- be nice"

    def test_rejected_edit_is_retried_next_turn(self) -> None:
        """Test that a rejected edit is fed back with its reason and the Editor is given another turn."""
        lm = FakeEditorLM("<edit><target>be nice</target><text>x</text><where>sideways</where></edit>", REPLACE_EDIT)
        result = _propose(lm, EditTool.INSERT_TEXT, budget=RLMBudget(max_root_iterations=2))
        assert "Invalid <where> value 'sideways'" in lm.calls[1]
        assert "Try again." in lm.calls[1]
        assert result.iterations == 2


class TestOneActionPerTurn:
    """Test cases for the one-action-per-turn protocol."""

    def test_reply_without_an_action_is_a_protocol_error(self) -> None:
        """Test that a reply carrying no action is a protocol error fed back to the Editor."""
        lm = FakeEditorLM("Let me look at the region first.", REPLACE_EDIT)
        result = _propose(lm)
        assert result.changed is True
        assert result.steps[0].action == "invalid"
        assert "No action found" in lm.calls[1]

    def test_python_and_edit_in_one_reply_are_rejected(self) -> None:
        """Test that a reply mixing a Python turn and an edit is rejected without executing either half."""
        lm = FakeEditorLM(f"<python>print(region)</python>\n{REPLACE_EDIT}", REPLACE_EDIT)
        result = _propose(lm)
        assert result.repl_calls == 0  # neither half was executed
        assert "Exactly one action per turn" in lm.calls[1]
        assert result.iterations == 2

    def test_two_edits_in_one_reply_are_rejected(self) -> None:
        """Test that a reply carrying two edits is rejected as a protocol error."""
        lm = FakeEditorLM(f"{REPLACE_EDIT}\n{REPLACE_EDIT}", REPLACE_EDIT)
        result = _propose(lm)
        assert result.steps[0].action == "invalid"
        assert result.iterations == 2


class TestBudgetsAndTrace:
    """Test cases for budgets, tracing and safety guarantees."""

    def test_terminates_and_drops_when_no_edit_emitted(self) -> None:
        """Test that the loop terminates at the iteration budget and drops the proposal when no edit is emitted."""
        lm = FakeEditorLM("just thinking", "<python>print(1)</python>", "no edit here", "nope")
        result = _propose(lm, budget=RLMBudget(max_root_iterations=4))
        assert result.changed is False
        assert result.new_text == PROMPT  # parent text unchanged
        assert result.iterations == 4
        assert result.dropped_reason is not None and "no_valid_edit after 4 iterations" in result.dropped_reason
        assert len(lm.calls) == 4
        assert "This is your last turn" in lm.calls[3] and "This is your last turn" not in lm.calls[2]

    def test_retries_after_unapplicable_edit_then_succeeds(self) -> None:
        """Test that an unapplicable edit is retried and a later valid edit succeeds."""
        lm = FakeEditorLM("<edit><target>does-not-exist</target><text>x</text></edit>", REPLACE_EDIT)
        result = _propose(lm, budget=RLMBudget(max_root_iterations=4))
        assert result.changed is True
        assert result.iterations == 2

    def test_repl_budget_exhaustion_is_fed_back(self) -> None:
        """Test that exhausting the REPL-call budget is fed back to the Editor."""
        lm = FakeEditorLM("<python>print(1)</python>", "<python>print(2)</python>", REPLACE_EDIT)
        result = _propose(lm, budget=RLMBudget(max_repl_calls=1))
        assert "repl_calls budget exhausted" in lm.calls[2]
        assert result.repl_calls == 1
        assert result.changed is True

    def test_steps_trace_every_turn(self) -> None:
        """Test that the step trace records the action, code, stdout and error of every turn."""
        lm = FakeEditorLM(
            "no action",
            "<python>x = 1\nprint(x)</python>",
            "<edit><target>missing</target><text>y</text></edit>",
            REPLACE_EDIT,
        )
        result = _propose(lm)
        actions = [(s.iteration, s.action) for s in result.steps]
        assert actions == [(1, "invalid"), (2, "python"), (3, "edit"), (4, "edit")]
        assert result.steps[1].code == "x = 1\nprint(x)" and result.steps[1].stdout == "1\n"
        assert result.steps[2].error is not None and "missing" in result.steps[2].error
        assert result.steps[3].error is None
        assert result.final_output == REPLACE_EDIT

    def test_over_budget_edit_is_dropped(self) -> None:
        """Test that an edit exceeding the max-chars budget is dropped."""
        lm = FakeEditorLM("<edit><anchor>be nice</anchor><where>after</where><text> extra padding text</text></edit>")
        result = _propose(lm, EditTool.INSERT_TEXT, max_chars=10)
        assert result.changed is False
        assert result.new_text == PROMPT
        assert result.dropped_reason is not None and "over_budget" in result.dropped_reason

    def test_whole_document_edit_may_not_remove_a_header(self) -> None:
        """Test that a whole-document edit may not delete a canonical section header."""
        # Deleting "## Rules" would break the canonical format: the edit is
        # rejected with the parse error and the model gets another turn.
        lm = FakeEditorLM("<edit><target>## Rules\n</target></edit>", "<edit><target>- be brief</target></edit>")
        result = _propose(lm, EditTool.DELETE_TEXT, section=None, budget=RLMBudget(max_root_iterations=3))
        assert result.changed is True
        assert "## Rules" in result.new_text
        assert "be brief" not in result.new_text
        assert "must have exactly the sections" in lm.calls[1]

    def test_section_edit_may_not_inject_a_header(self) -> None:
        """Test that a section edit may not inject a new header into the document."""
        lm = FakeEditorLM("<edit><anchor>be brief</anchor><where>after</where><text>\n## Notes\nnew</text></edit>")
        result = _propose(lm, EditTool.INSERT_TEXT, budget=RLMBudget(max_root_iterations=1))
        assert result.changed is False
        assert result.dropped_reason is not None and "Notes" in result.dropped_reason

    def test_malformed_component_text_is_a_caller_error(self) -> None:
        """Test that proposing against component text that violates the template raises MalformedDocumentError."""
        rlm = RLMProposer(FakeEditorLM(), TEMPLATE)
        with pytest.raises(MalformedDocumentError):
            rlm.propose("free-form text", _target("Rules"), EditTool.REPLACE_TEXT, "", "fb", "traces", None)
