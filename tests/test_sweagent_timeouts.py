"""Exercise pinned native timeout handling without model calls or task containers."""

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from examples.common.benchmark_types import BenchmarkModels
from examples.common.react_v2 import file_sha256
from examples.swebench_live import prepare_swe_agent, swe_agent


@pytest.fixture
def agent_root():
    value = os.environ.get("SWE_AGENT_TEST_ROOT")
    if not value:
        pytest.skip("Set SWE_AGENT_TEST_ROOT to a pinned, patched SWE-agent test root")
    root = Path(value)
    swe_agent.activate(root)
    return root


@pytest.fixture
def native_agent(agent_root, monkeypatch):
    from sweagent.agent.agents import DefaultAgent, DefaultAgentConfig
    from sweagent.environment.swe_env import SWEEnv

    models = BenchmarkModels("openrouter/offline-check", "unused", None, None, {}, {})
    config = DefaultAgentConfig.model_validate(swe_agent.configuration(agent_root, models, max_steps=30))
    agent = DefaultAgent.from_config(config)
    runtime = SimpleNamespace(run_in_session=AsyncMock(return_value=SimpleNamespace(output="ok", exit_code=0)))
    agent._env = SWEEnv(
        deployment=SimpleNamespace(runtime=runtime, is_alive=AsyncMock(return_value=True)),
        repo=SimpleNamespace(repo_name="testbed"),
        post_startup_commands=[],
    )
    agent.model = SimpleNamespace(query=Mock(side_effect=AssertionError("No model call expected")))
    monkeypatch.setattr(agent.tools, "get_state", Mock(return_value={}))
    monkeypatch.setattr(agent, "_get_format_dict", lambda: {})
    monkeypatch.setattr(agent._env, "read_file", Mock(return_value="partial patch"))
    monkeypatch.setattr(agent._env, "execute_command", Mock())
    return agent, runtime


def set_clock(monkeypatch, values):
    from sweagent.agent import agents

    monkeypatch.setattr(agents, "time", SimpleNamespace(perf_counter=Mock(side_effect=values)))


@pytest.mark.parametrize(
    ("per_action", "total", "used", "expected"),
    [(30, 1800, 0, 30), (17, 42, 4, 17), (30, 1800, 1799.75, 0.25)],
)
def test_native_action_passes_effective_budget_to_rex(native_agent, monkeypatch, per_action, total, used, expected):
    from sweagent.agent.agents import StepOutput
    from swerex.runtime.abstract import BashAction

    agent, runtime = native_agent
    agent.tools.config.execution_timeout = per_action
    agent.tools.config.total_execution_timeout = total
    agent._total_execution_time = used
    set_clock(monkeypatch, [10, 10.1])
    result = agent.handle_action(StepOutput(action="pwd"))
    action = runtime.run_in_session.call_args.args[0]
    assert isinstance(action, BashAction)
    assert action.command == "pwd"
    assert action.timeout == expected
    assert result.observation == "ok" and not result.done
    assert agent._total_execution_time == pytest.approx(used + 0.1)
    assert agent.tools.config.execution_timeout == per_action
    agent.model.query.assert_not_called()


@pytest.mark.parametrize(("used", "effective"), [(0, 30), (1799.75, 0.25)])
def test_native_timeout_interrupts_and_reports_effective_budget(native_agent, monkeypatch, used, effective):
    from sweagent.agent.agents import StepOutput
    from swerex.exceptions import CommandTimeoutError
    from swerex.runtime.abstract import BashInterruptAction

    agent, runtime = native_agent
    agent._total_execution_time = used
    runtime.run_in_session.side_effect = [CommandTimeoutError("stalled command"), None]
    set_clock(monkeypatch, [100, 100 + effective])
    result = agent.handle_action(StepOutput(action="sleep 999"))
    assert runtime.run_in_session.call_args_list[0].args[0].timeout == effective
    assert isinstance(runtime.run_in_session.call_args_list[1].args[0], BashInterruptAction)
    assert f"{effective} seconds" in result.observation
    assert agent._n_consecutive_timeouts == 1
    assert agent._total_execution_time == pytest.approx(used + effective)
    assert not result.done


def test_native_timeout_recovers_and_resets_consecutive_count(native_agent, monkeypatch):
    from sweagent.agent.agents import StepOutput
    from swerex.exceptions import CommandTimeoutError

    agent, runtime = native_agent
    runtime.run_in_session.side_effect = [
        CommandTimeoutError("stalled command"),
        None,
        SimpleNamespace(output="ready", exit_code=0),
    ]
    set_clock(monkeypatch, [0, 30.4, 31, 31.5])
    agent.handle_action(StepOutput(action="sleep 999"))
    result = agent.handle_action(StepOutput(action="pwd"))
    assert result.observation == "ready" and not result.done
    assert agent._n_consecutive_timeouts == 0
    assert agent._total_execution_time == pytest.approx(30.9)
    assert runtime.run_in_session.call_count == 3
    agent.model.query.assert_not_called()


def test_final_consecutive_timeout_interrupts_before_native_autosubmission(native_agent, monkeypatch):
    from swerex.exceptions import CommandTimeoutError
    from swerex.runtime.abstract import BashAction, BashInterruptAction

    agent, runtime = native_agent
    agent._n_consecutive_timeouts = agent.tools.config.max_consecutive_execution_timeouts - 1
    events = []

    async def dispatch(action):
        if isinstance(action, BashAction):
            events.append("action")
            raise CommandTimeoutError("stalled command")
        assert isinstance(action, BashInterruptAction)
        events.append("interrupt")

    runtime.run_in_session.side_effect = dispatch
    agent._env.execute_command.side_effect = lambda *_, **__: events.append("autosubmit")
    agent.model.query.side_effect = None
    agent.model.query.return_value = {
        "message": "Run a command",
        "tool_calls": [
            {"id": "probe", "type": "function", "function": {"name": "bash", "arguments": '{"command":"pwd"}'}}
        ],
    }
    set_clock(monkeypatch, [0, 30.2])
    result = agent.forward_with_handling([])
    assert events == ["action", "interrupt", "autosubmit"]
    assert result.done and result.exit_status == "submitted (exit_command_timeout)"
    assert result.submission == "partial patch"
    assert agent._total_execution_time == pytest.approx(30.2)
    agent.model.query.assert_called_once()


@pytest.mark.parametrize("used", [1800, 1800.01])
def test_total_budget_stops_before_query_and_retains_native_autosubmission(native_agent, used):
    agent, runtime = native_agent
    agent._total_execution_time = used
    result = agent.forward_with_handling([])
    assert result.done and result.exit_status == "submitted (exit_total_execution_time)"
    assert result.submission == "partial patch"
    agent.model.query.assert_not_called()
    runtime.run_in_session.assert_not_called()
    agent._env.execute_command.assert_called_once_with(
        "git add -A && git diff --cached > /root/model.patch", check=True, cwd="/testbed"
    )


def test_exhausted_total_budget_never_dispatches_an_action(native_agent):
    from sweagent.agent.agents import StepOutput, _TotalExecutionTimeExceeded

    agent, runtime = native_agent
    agent._total_execution_time = agent.tools.config.total_execution_timeout
    with pytest.raises(_TotalExecutionTimeExceeded):
        agent.handle_action(StepOutput(action="pwd"))
    runtime.run_in_session.assert_not_called()


def test_rollout_closes_environment_after_interrupt_failure(native_agent, agent_root, tmp_path, monkeypatch):
    from sweagent.agent.agents import StepOutput
    from sweagent.environment import swe_env
    from swerex.exceptions import CommandTimeoutError

    from examples.swebench_live import swe_agent_rex

    agent, runtime = native_agent
    runtime.run_in_session.side_effect = [CommandTimeoutError("stalled command"), RuntimeError("interrupt failed")]
    environment = agent._env
    monkeypatch.setattr(environment, "start", Mock())
    monkeypatch.setattr(environment, "close", Mock())
    monkeypatch.setattr(swe_env, "SWEEnv", lambda **_: environment)
    monkeypatch.setattr(swe_agent_rex, "ApptainerDeployment", lambda *_: None)
    monkeypatch.setattr(swe_agent, "create_agent", lambda _: agent)
    monkeypatch.setattr(agent, "run", lambda **_: agent.handle_action(StepOutput(action="sleep 999")))
    set_clock(monkeypatch, [0, 30.5])
    models = BenchmarkModels("openrouter/offline-check", "unused", None, None, {}, {})
    with pytest.raises(RuntimeError, match="interrupt failed"):
        swe_agent.rollout(
            agent_root,
            {"id": "task", "problem_statement": "task"},
            {},
            models,
            "system",
            tmp_path,
            16,
            base_commit="a" * 40,
            max_steps=30,
        )
    assert agent._total_execution_time == pytest.approx(30.5)
    environment.close.assert_called_once_with()
    assert runtime.run_in_session.call_count == 2


def test_native_explicit_submission_still_reads_patch(native_agent, monkeypatch):
    from sweagent.agent.agents import StepOutput

    agent, runtime = native_agent
    runtime.run_in_session.return_value = SimpleNamespace(output="<<SWE_AGENT_SUBMISSION>>", exit_code=0)
    set_clock(monkeypatch, [0, 0.1])
    result = agent.handle_action(StepOutput(action="submit"))
    assert result.done and result.exit_status == "submitted"
    assert result.submission == "partial patch"
    assert runtime.run_in_session.call_args.args[0].timeout == 30
    agent._env.read_file.assert_called_once_with("/root/model.patch", encoding="utf-8", errors="backslashreplace")


def test_runtime_records_patch_and_timeout_protocol(agent_root, monkeypatch):
    monkeypatch.setattr(swe_agent, "verify_payload", lambda _: {"test_payload": True})
    models = BenchmarkModels("openrouter/offline-check", "unused", None, None, {}, {})
    identity = swe_agent.runtime_identity(agent_root, models, max_steps=30)
    assert identity["source"]["patch_sha256"] == file_sha256(swe_agent.AGENT_PATCH)
    assert identity["execution_timeouts"]["protocol"] == "configured-command-and-total-v1"
    assert identity["execution_timeouts"]["command_seconds"] == 30
    assert identity["execution_timeouts"]["total_seconds"] == 1800
    assert identity["step_budget"] == {"max_queries": 30, "enforcement": "before_query_v1"}


def test_preparation_patches_new_source_and_leaves_existing_source_unchanged(agent_root, tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_swe_agent, "AGENT_URL", str(agent_root / "source"))
    fresh = tmp_path / "fresh"
    identity = prepare_swe_agent.prepare_source(fresh)
    assert identity == swe_agent.source_identity(agent_root / "source")
    assert prepare_swe_agent.prepare_source(fresh) == identity
    native_source = fresh / "source/sweagent/agent/agents.py"
    native_source.write_text(native_source.read_text() + "\n# unexpected drift\n")
    changed_hash = file_sha256(native_source)
    with pytest.raises(ValueError, match="runtime patch mismatch"):
        prepare_swe_agent.prepare_source(fresh)
    assert file_sha256(native_source) == changed_hash

    frozen = tmp_path / "frozen"
    subprocess.run(["git", "clone", str(agent_root / "source"), str(frozen / "source")], check=True)
    pristine_source = frozen / "source/sweagent/agent/agents.py"
    pristine_hash = file_sha256(pristine_source)
    with pytest.raises(ValueError, match="runtime patch mismatch"):
        prepare_swe_agent.prepare_source(frozen)
    assert file_sha256(pristine_source) == pristine_hash
    assert subprocess.check_output(["git", "-C", str(frozen / "source"), "status", "--porcelain"]) == b""


def test_preparation_never_replaces_missing_source_of_existing_payload(tmp_path):
    (tmp_path / "payload").mkdir()
    with pytest.raises(ValueError, match="original source"):
        prepare_swe_agent.prepare_source(tmp_path)
    assert not (tmp_path / "source").exists()
