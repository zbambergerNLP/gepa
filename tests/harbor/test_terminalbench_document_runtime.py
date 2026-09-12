"""Exercise real Harbor prompt, parser, skill, and summarization paths offline.

Run in Harbor's Python 3.12 environment with PYTHONPATH=src. Only the model and
terminal are simulated; no credentials, network requests, or paid trials occur.
"""

import asyncio
import copy
import json
import shlex
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

# This test directory can satisfy a top-level namespace import without Harbor installed.
pytest.importorskip("harbor.agents.terminus_2")

from harbor.agents.terminus_2 import Terminus2
from harbor.agents.terminus_2.terminus_2 import Command
from harbor.llms.base import ContextLengthExceededError, LLMResponse, OutputLengthExceededError
from harbor.llms.chat import Chat
from harbor.models.agent.context import AgentContext
from harbor.models.job.config import JobConfig
from harbor.models.trajectories import Step

from examples.terminalbench.terminus_agent import PromptedTerminus
from gepa.adapters.terminal_bench_adapter import HarborCLI, load_terminalbench_manifest
from gepa.adapters.terminal_bench_adapter.documents import (
    render_initial_instructions,
    seed_documents,
    write_document_bundle,
)


@pytest.fixture
def runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    """Create a real Terminus instance with a deterministic model boundary.

    Args:
        tmp_path: Evaluation and trial artifact directory.
        monkeypatch: Fixture replacing model initialization.

    Returns:
        Agent, complete candidate, fake model, and source directory.
    """
    candidate = {name: text + f"\nSENTINEL_{name} {{literal}}" for name, text in seed_documents("generic").items()}
    bundle = write_document_bundle(tmp_path, candidate)
    model = SimpleNamespace(
        _llm_kwargs={},
        call=AsyncMock(), get_model_context_limit=lambda: 32768, get_model_output_limit=lambda: 4096
    )
    model_call = model.call
    monkeypatch.setattr(Terminus2, "_init_llm", Mock(return_value=model))
    root = Path(__file__).parents[2]
    manifest = load_terminalbench_manifest(root / "examples/terminalbench/terminalbench-v2.1-manifest.json")
    runner = HarborCLI(manifest=manifest, student_model="openai/gpt-4o-mini", work_dir=tmp_path, agent_python_path=root)
    config = runner.build_job_config(
        [manifest.tasks("train", 1)[0].task_id],
        prompt_path=tmp_path / "terminus-prompt.txt",
        bundle_path=bundle,
        jobs_dir=tmp_path / "jobs",
        job_name="runtime-test",
    )
    (tmp_path / "job-config.json").write_text(json.dumps(config))
    settings = config["agents"][0]
    agent = PromptedTerminus(
        logs_dir=tmp_path / "logs",
        model_name=settings["model_name"],
        **{**settings["kwargs"], "record_terminal_session": False},
    )
    model.call = model_call
    return agent, candidate, model, tmp_path


def test_summary_questions_answers_and_handoff_reach_the_model(
    runtime: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trigger real summarization near the context limit and verify saved traces.

    Args:
        runtime: Agent fixture with a fake model transport.
        monkeypatch: Fixture controlling the conversation's estimated token count.
    """
    agent, candidate, model, root = runtime
    assert agent._enable_summarize is True
    assert agent._proactive_summarization_threshold == 8_000
    monkeypatch.setattr(agent, "_count_total_tokens", Mock(return_value=25_000))
    calls = []

    async def respond(prompt: str, **kwargs: object) -> LLMResponse:
        """Capture immutable model inputs and return staged summary responses.

        Args:
            prompt: Model input assembled by the agent.
            **kwargs: History and model options supplied by Harbor.

        Returns:
            Next deterministic summary-stage result.
        """
        calls.append((prompt, copy.deepcopy(kwargs)))
        return LLMResponse(content=("SUMMARY", "QUESTIONS", "ANSWERS")[len(calls) - 1])

    model.call.side_effect = respond
    chat = Chat(model)
    chat._messages = [{"role": "user", "content": "TASK_INPUT"}]
    agent._trajectory_steps = [Step(step_id=1, source="user", message="TASK_INPUT")]
    session = SimpleNamespace(capture_pane=AsyncMock(return_value="REAL_STATE"))
    handoff, refs = asyncio.run(agent._check_proactive_summarization(chat, "TASK_INPUT", session))
    for index, name in enumerate(("summary", "summary_questions", "summary_answers")):
        assert candidate[name] in calls[index][0]
    assert "TASK_INPUT" in calls[0][0]
    assert "SUMMARY" in calls[1][0] and "REAL_STATE" in calls[1][0]
    assert calls[1][1]["message_history"] == []
    assert calls[2][1]["message_history"][-2]["content"] == calls[0][0]
    assert candidate["handoff"] in handoff and "ANSWERS" in handoff
    assert chat.messages[1]["content"] == calls[1][0]
    assert len(refs) == 3
    traces = "\n".join(path.read_text() for path in (root / "logs").glob("trajectory*.json"))
    assert "SENTINEL_summary" in traces and "SENTINEL_summary_answers" in traces


@pytest.mark.parametrize("short_fails", [False, True])
def test_context_fallback_prompts_reach_the_model(
    runtime: tuple, monkeypatch: pytest.MonkeyPatch, short_fails: bool
) -> None:
    """Exercise both short-summary and terminal-state-only recovery.

    Args:
        runtime: Agent fixture.
        monkeypatch: Fixture forcing the full summary to fail.
        short_fails: Whether the short summary also fails.
    """
    agent, candidate, model, _ = runtime
    monkeypatch.setattr(agent, "_unwind_messages_to_free_tokens", Mock())
    monkeypatch.setattr(agent, "_summarize", AsyncMock(side_effect=RuntimeError("summary unavailable")))
    model.call.side_effect = [
        ContextLengthExceededError(),
        RuntimeError("short unavailable") if short_fails else LLMResponse("BRIEF_SUMMARY"),
        LLMResponse("RECOVERED"),
    ]
    chat = Chat(model)
    result = asyncio.run(
        agent._query_llm(
            chat, "TASK_INPUT", "TASK_INPUT", SimpleNamespace(capture_pane=AsyncMock(return_value="REAL_STATE"))
        )
    )
    assert result.content == "RECOVERED"
    assert candidate["short_summary"] in model.call.call_args_list[1].kwargs["prompt"]
    recovered_prompt = model.call.call_args_list[2].kwargs["prompt"]
    assert candidate["context_recovery"] in recovered_prompt
    assert "TASK_INPUT" in recovered_prompt and "REAL_STATE" in recovered_prompt


def test_output_limit_uses_candidate_guidance_and_preserves_failed_response(runtime: tuple) -> None:
    """Keep real truncation evidence while substituting the repair instructions.

    Args:
        runtime: Agent fixture.
    """
    agent, candidate, model, _ = runtime
    model.call.side_effect = [
        OutputLengthExceededError("limit", truncated_response="TRUNCATED"),
        LLMResponse("RECOVERED"),
    ]
    chat = Chat(model)
    asyncio.run(agent._query_llm(chat, "TASK_INPUT"))
    retry_prompt = model.call.call_args_list[1].kwargs["prompt"]
    assert candidate["output_limit"] in retry_prompt and "4096 tokens" in retry_prompt
    assert chat.messages[0]["content"] == "TASK_INPUT"
    assert chat.messages[1]["content"] == "TRUNCATED"


def test_timeout_keeps_real_command_and_observation(runtime: tuple) -> None:
    """Exercise the inherited command executor with candidate timeout guidance.

    Args:
        runtime: Agent fixture.
    """
    agent, candidate, _, _ = runtime
    session = SimpleNamespace(
        send_keys=AsyncMock(side_effect=TimeoutError()), get_incremental_output=AsyncMock(return_value="REAL_STATE")
    )
    timed_out, message = asyncio.run(agent._execute_commands([Command("sleep 10\n", 1)], session))
    assert timed_out
    assert candidate["timeout"] in message
    assert "sleep 10" in message and "REAL_STATE" in message


def test_real_harbor_copied_context_keeps_the_original_message(runtime: tuple) -> None:
    """Verify Harbor's native copy marker preserves message content."""
    agent, _, _, _ = runtime
    agent._trajectory_steps = [Step(step_id=1, source="user", message="ORIGINAL_TASK")]
    copied, _ = agent._prepare_copied_trajectory_steps(1)
    assert copied[0].is_copied_context is True
    assert copied[0].message == agent._trajectory_steps[0].message == "ORIGINAL_TASK"
    assert copied[0] is not agent._trajectory_steps[0]


def test_job_config_is_accepted_by_pinned_harbor(runtime: tuple) -> None:
    """Validate the actual adapter output against Harbor's job schema.

    Args:
        runtime: Fixture providing a materialized candidate bundle.
    """
    _, _, _, root = runtime
    config = json.loads((root / "job-config.json").read_text())
    parsed = JobConfig.model_validate(config)
    assert parsed.n_attempts == 1
    assert parsed.retry.max_retries == 0
    assert parsed.agents[0].kwargs["document_bundle_path"] == str(root / "document-bundle.json")
    assert parsed.agents[0].import_path == "examples.terminalbench.terminus_agent:PromptedTerminus"
    assert parsed.agents[0].override_timeout_sec is None
    assert parsed.timeout_multiplier == 1.0
    assert len(parsed.tasks) + len(parsed.datasets) == 1


def test_real_agent_loop_discovers_then_reads_skills_and_repairs_json(
    runtime: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run discovery, parser repair, skill loading, and completion through Terminus.

    Args:
        runtime: Agent fixture.
        monkeypatch: Fixture replacing terminal startup only.
    """
    agent, candidate, model, root = runtime
    skill_root = root / "skills"
    state = {"output": "SHELL_READY"}

    async def execute(command: str, **kwargs: object) -> SimpleNamespace:
        """Read materialized skill files for Harbor's real discovery routine.

        Args:
            command: Discovery find or cat command.
            **kwargs: Harbor execution options.

        Returns:
            File discovery or file content result.
        """
        if command.startswith("find "):
            output = "\n".join(f"/opt/gepa-skills/{p.parent.name}/SKILL.md" for p in skill_root.glob("*/SKILL.md"))
        else:
            path = shlex.split(command)[1].replace("/opt/gepa-skills", str(skill_root))
            output = Path(path).read_text()
        return SimpleNamespace(return_code=0, stdout=output)

    async def send_keys(keys: str, **kwargs: object) -> None:
        """Return the actual skill file through a simulated terminal read.

        Args:
            keys: Model-requested terminal command.
            **kwargs: Fixed tmux timing options.
        """
        state["output"] = (await execute(keys)).stdout

    async def terminal_output(**kwargs: object) -> str:
        """Return the terminal's observed output.

        Args:
            **kwargs: Harbor capture options.

        Returns:
            Last command output.
        """
        return state["output"]

    environment = SimpleNamespace(upload_dir=AsyncMock(), is_dir=AsyncMock(return_value=True), exec=execute)
    monkeypatch.setattr(Terminus2, "setup", AsyncMock())
    asyncio.run(agent.setup(environment))
    environment.upload_dir.assert_awaited_once_with(source_dir=skill_root, target_dir="/opt/gepa-skills")
    agent._session = SimpleNamespace(
        get_incremental_output=terminal_output,
        capture_pane=terminal_output,
        send_keys=send_keys,
        is_session_alive=AsyncMock(return_value=True),
    )
    valid = {"analysis": "check", "plan": "inspect", "commands": []}
    model.call.side_effect = [
        LLMResponse("invalid JSON"),
        LLMResponse(
            json.dumps(
                {
                    **valid,
                    "commands": [{"keystrokes": "cat /opt/gepa-skills/skill_debugging/SKILL.md\n", "duration": 0.1}],
                }
            )
        ),
        LLMResponse(json.dumps({**valid, "task_complete": True})),
        LLMResponse(json.dumps({**valid, "task_complete": True})),
    ]
    asyncio.run(agent.run("TASK_INPUT", environment, AgentContext()))
    prompts = [call.kwargs["prompt"] for call in model.call.call_args_list]
    assert render_initial_instructions(candidate) in prompts[0]
    assert "terminal-debugging" in prompts[0] and "terminal-verification" in prompts[0]
    assert "SENTINEL_skill_debugging" not in prompts[0]
    assert "SENTINEL_parse_error" in prompts[1]
    assert "SENTINEL_skill_debugging" in prompts[2]
    assert "SENTINEL_completion" in prompts[3]
    trace = (root / "logs" / "trajectory.json").read_text()
    assert "SENTINEL_skill_debugging" in trace
