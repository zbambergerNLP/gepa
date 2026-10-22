"""Check the upstream-agent boundary without making model requests."""

import argparse
import asyncio
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import yaml
from benchmark_model_fixtures import BenchmarkProposerLM

from examples.common.benchmark_types import BenchmarkModels
from examples.local_api.profile import API_BASE, role_kwargs
from examples.swebench_live import main as swe_main
from examples.swebench_live import swe_agent
from examples.swebench_live.main import add_arguments, build_benchmark
from gepa.proposer.reflective_mutation.three_role import ThreeRoleReflectionLM
from gepa.strategies.document_template import VerbatimPromptTemplate


@pytest.fixture
def agent_root():
    value = os.environ.get("SWE_AGENT_TEST_ROOT")
    if not value:
        pytest.skip("Set SWE_AGENT_TEST_ROOT to the pinned prepared source for integration tests")
    root = Path(value)
    swe_agent.activate(root)
    return root


def test_default_selection_and_no_custom_turn_cap():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args([])
    assert (args.train_limit, args.val_limit, args.test_limit) == (60, 120, 120)
    assert args.container_backend == "apptainer"
    assert args.agent_max_steps is None
    assert parser.parse_args(["--agent-max-steps", "30"]).agent_max_steps == 30


@pytest.mark.parametrize("limit", ["0", "-1"])
def test_invalid_step_limit_fails_before_loading_assets(limit, monkeypatch):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(["--agent-max-steps", limit])
    args.api_profile, args.max_workers = "explicit.json", 1
    loader = Mock(side_effect=AssertionError("Must not read assets"))
    monkeypatch.setattr(swe_main, "load_data", loader)
    with pytest.raises(ValueError, match="max_steps must be positive"):
        build_benchmark(args, None)
    loader.assert_not_called()


def test_step_limit_changes_only_model_limit_and_runtime_identity(agent_root, monkeypatch):
    from examples.common.artifacts import digest

    models = BenchmarkModels("openrouter/offline-check", "unused", None, None, {}, {})
    original = swe_agent.configuration(agent_root, models)
    limited = swe_agent.configuration(agent_root, models, max_steps=30)
    assert limited["model"].pop("per_instance_call_limit") == 30
    assert original["model"].pop("per_instance_call_limit") == 100
    assert limited == original
    monkeypatch.setattr(swe_agent, "verify_payload", lambda _: {"test_payload": True})
    original_runtime = swe_agent.runtime_identity(agent_root, models)
    limited_runtime = swe_agent.runtime_identity(agent_root, models, max_steps=30)
    assert "step_budget" not in original_runtime
    assert limited_runtime["step_budget"] == {"max_queries": 30, "enforcement": "before_query_v1"}
    assert digest(limited_runtime) != digest(original_runtime)


@pytest.mark.parametrize("limit", [1, 30])
def test_pinned_agent_stops_before_extra_query_and_uses_native_autosubmission(agent_root, monkeypatch, limit):
    """Exercise the pinned parser and stop handler without a provider or container."""
    from sweagent.agent.agents import DefaultAgent, DefaultAgentConfig
    from sweagent.agent.models import PredeterminedTestModel

    response = {
        "message": "Inspect the repository",
        "tool_calls": [
            {"id": "probe", "type": "function", "function": {"name": "bash", "arguments": '{"command":"pwd"}'}}
        ],
    }
    model = PredeterminedTestModel([response] * (limit + 1))
    dispatch = Mock(wraps=model.query)
    monkeypatch.setattr(model, "query", dispatch)
    models = BenchmarkModels("openrouter/offline-check", "unused", None, None, {}, {})
    config = DefaultAgentConfig.model_validate(swe_agent.configuration(agent_root, models, max_steps=limit))
    agent = DefaultAgent.from_config(config)
    agent.model = swe_agent._StepLimitedModel(model, limit)
    agent._env = SimpleNamespace(
        deployment=SimpleNamespace(is_alive=AsyncMock(return_value=True)),
        repo=SimpleNamespace(repo_name="testbed"),
        execute_command=Mock(),
        read_file=Mock(return_value="partial patch"),
    )
    handle_action = Mock(side_effect=lambda step: step)
    autosubmit = Mock(wraps=agent.attempt_autosubmission_after_error)
    monkeypatch.setattr(agent, "handle_action", handle_action)
    monkeypatch.setattr(agent, "attempt_autosubmission_after_error", autosubmit)

    for _ in range(limit):
        assert not agent.forward_with_handling([]).done
    stopped = agent.forward_with_handling([])
    assert dispatch.call_count == handle_action.call_count == limit
    assert stopped.done and stopped.exit_status == "submitted (exit_cost)"
    assert stopped.submission == "partial patch"
    autosubmit.assert_called_once()
    agent._env.execute_command.assert_called_once_with(
        "git add -A && git diff --cached > /root/model.patch", check=True, cwd="/testbed"
    )
    assert agent.model.stats is model.stats


def test_recovery_queries_share_the_same_step_budget(agent_root, monkeypatch):
    from sweagent.agent.agents import DefaultAgent, DefaultAgentConfig
    from sweagent.exceptions import ContentPolicyViolationError

    models = BenchmarkModels("openrouter/offline-check", "unused", None, None, {}, {})
    config = DefaultAgentConfig.model_validate(swe_agent.configuration(agent_root, models, max_steps=1))
    agent = DefaultAgent.from_config(config)
    dispatch = Mock(side_effect=ContentPolicyViolationError("Retry consumes the existing query budget"))
    agent.model = swe_agent._StepLimitedModel(SimpleNamespace(query=dispatch), 1)
    autosubmit = Mock(side_effect=lambda step: step)
    monkeypatch.setattr(agent, "attempt_autosubmission_after_error", autosubmit)
    stopped = agent.forward_with_handling([])
    assert stopped.done and stopped.exit_status == "exit_cost"
    dispatch.assert_called_once()
    autosubmit.assert_called_once()


def test_rollout_installs_a_fresh_budget_and_closes_before_return(agent_root, tmp_path, monkeypatch):
    from sweagent.agent.agents import DefaultAgent
    from sweagent.environment import swe_env
    from sweagent.exceptions import InstanceCallLimitExceededError

    from examples.swebench_live import swe_agent_rex

    dispatch = Mock(return_value={"message": "test"})
    model = SimpleNamespace(query=dispatch)
    agents = []
    info = {"exit_status": "submitted (exit_cost)", "submission": "partial patch"}

    def from_config(_):
        agent = SimpleNamespace(model=model, get_trajectory_data=lambda: {"info": info})

        def run(**_):
            for _ in range(30):
                agent.model.query([])
            with pytest.raises(InstanceCallLimitExceededError):
                agent.model.query([])
            return SimpleNamespace(info=info)

        agent.run = run
        agents.append(agent)
        return agent

    environment = SimpleNamespace(start=Mock(), close=Mock())
    monkeypatch.setattr(DefaultAgent, "from_config", from_config)
    monkeypatch.setattr(swe_env, "SWEEnv", lambda **_: environment)
    monkeypatch.setattr(swe_agent_rex, "ApptainerDeployment", lambda *_: None)
    models = BenchmarkModels("openrouter/offline-check", "unused", None, None, {}, {})
    for index in range(2):
        trial = tmp_path / str(index)
        trial.mkdir()
        result = swe_agent.rollout(
            agent_root, {"id": "task", "problem_statement": "task"}, {}, models, "system", trial, 16, max_steps=30
        )
        assert result["patch"] == "partial patch"
        assert json.loads((trial / "swe-agent-config.json").read_text())["model"]["per_instance_call_limit"] == 30
    assert dispatch.call_count == 60
    assert agents[0].model is not agents[1].model
    assert environment.close.call_count == 2


def test_swe_builder_exposes_only_the_verbatim_system_template(agent_root, tmp_path, monkeypatch):
    """Use the actual upstream seed while isolating native container prerequisites."""
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args([])
    args.api_profile, args.max_workers, args.run_dir = "explicit.json", 1, tmp_path
    args.swe_agent_root = agent_root
    splits = {split: [{"id": split}] for split in ("train", "val", "test")}
    monkeypatch.setattr(swe_main, "load_data", lambda _: (splits, {}))
    for name in ("verify_runtime", "apptainer_runtime", "verify_apptainer_images", "agent_runtime"):
        monkeypatch.setattr(swe_main, name, lambda *_, **__: {})
    adapter = Mock()
    monkeypatch.setattr(swe_main, "SWEBenchLiveAdapter", adapter)
    args.agent_max_steps = 30
    runtime = Mock(return_value={"step_budget": {"max_queries": 30, "enforcement": "before_query_v1"}})
    monkeypatch.setattr(swe_main, "agent_runtime", runtime)
    benchmark = build_benchmark(args, None)
    runtime.assert_called_once_with(agent_root, None, max_steps=30)
    assert adapter.call_args.kwargs["agent_max_steps"] == 30
    assert benchmark.runtime["agent"]["step_budget"]["max_queries"] == 30
    assert benchmark.runtime["reflection_trajectory_format"] == "swe-agent-history-and-step-metadata-v2"
    seed = swe_agent.seed_prompt(agent_root)
    assert benchmark.seed_candidate == {"system_prompt": seed}
    assert benchmark.component_kinds == {"system_prompt": "system_prompt"}
    template = benchmark.templates["system_prompt"]
    assert isinstance(template, VerbatimPromptTemplate)
    assert template.render(template.parse(seed)) == seed


def test_pinned_default_agent_configuration_is_preserved(agent_root):
    from sweagent.agent.agents import DefaultAgent, DefaultAgentConfig

    original = yaml.safe_load((agent_root / "source/config/default.yaml").read_text())["agent"]
    models = BenchmarkModels(
        "openrouter/test", "unused", "https://openrouter.ai/api/v1", None, {"temperature": 0.5, "max_tokens": 1234}, {}
    )
    config = swe_agent.configuration(agent_root, models)
    edited = swe_agent.configuration(agent_root, models, "Changed system instructions")
    assert edited["templates"].pop("system_template") == "Changed system instructions"
    baseline = swe_agent.configuration(agent_root, models)
    baseline["templates"].pop("system_template")
    assert edited == baseline
    assert config["templates"] == original["templates"]
    assert config["history_processors"] == original["history_processors"]
    for key, value in original["tools"].items():
        if key != "bundles":
            assert config["tools"][key] == value
    assert config["model"]["per_instance_call_limit"] == 100
    assert config["model"]["completion_kwargs"] == models.solver_kwargs
    assert config["model"]["retry"]["retries"] == 1
    assert swe_agent.seed_prompt(agent_root) == original["templates"]["system_template"]
    resolved = DefaultAgentConfig.model_validate(config)
    agent = DefaultAgent.from_config(resolved)
    assert type(agent) is DefaultAgent
    assert {tool["function"]["name"] for tool in agent.tools.config.tools} == {"bash", "str_replace_editor", "submit"}
    assert resolved.tools.execution_timeout == 30
    assert resolved.tools.total_execution_timeout == 1800


@pytest.mark.parametrize("endpoint_source", ["duplicate", "dedicated", "kwargs", "default"])
def test_pinned_agent_api_base_reaches_completion_once(agent_root, monkeypatch, endpoint_source):
    """Reach the real upstream dispatch with hosted controls and a fake transport."""
    import litellm
    from sweagent.agent.agents import DefaultAgent, DefaultAgentConfig

    route = {
        "tag": "deepinfra/fp8",
        "pricing": {"prompt": "0.000001", "completion": "0.000002"},
        "parameters": {"max_tokens": 65_536, "reasoning_effort": "low", "seed": 0, "temperature": 1, "top_p": 0.95},
    }
    kwargs = role_kwargs({"roles": {"solver": route}}, "solver")
    if endpoint_source in {"dedicated", "default"}:
        kwargs.pop("api_base")
    models = BenchmarkModels(
        "openrouter/deepseek/deepseek-v4.1-flash-1017",
        "unused",
        API_BASE if endpoint_source in {"duplicate", "dedicated"} else None,
        None,
        kwargs,
        {},
    )
    original_kwargs = deepcopy(models.solver_kwargs)
    response = litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": "ready"}}])
    dispatch = Mock(return_value=response)
    monkeypatch.setattr(litellm, "completion", dispatch)
    # Token counting and price lookup are outside the endpoint-binding boundary.
    monkeypatch.setattr(litellm.utils, "token_counter", lambda **_: 1)
    monkeypatch.setattr(litellm.cost_calculator, "completion_cost", lambda *_, **__: 0.0)

    config = DefaultAgentConfig.model_validate(swe_agent.configuration(agent_root, models, max_steps=30))
    agent = DefaultAgent.from_config(config)
    result = agent.model.query([{"role": "system", "content": "Inspect the training task."}])

    assert result == {"message": "ready", "tool_calls": []}
    dispatch.assert_called_once()
    sent = dispatch.call_args.kwargs
    if endpoint_source == "default":
        assert "api_base" not in sent
    else:
        assert sent["api_base"] == API_BASE
    assert "api_base" not in config.model.completion_kwargs
    assert sent["model"] == models.solver_model
    assert sent["fallbacks"] == []
    for key, value in original_kwargs.items():
        if key != "api_base":
            assert sent[key] == value
    assert config.model.per_instance_call_limit == 30
    assert config.model.retry.retries == 1
    assert models.solver_kwargs == original_kwargs


def test_conflicting_solver_api_bases_fail_before_dispatch(agent_root):
    models = BenchmarkModels(
        "openrouter/test", "unused", API_BASE, None, {"api_base": "https://different.invalid/v1"}, {}
    )
    with pytest.raises(ValueError, match="SWE-agent solver API endpoints disagree"):
        swe_agent.configuration(agent_root, models)


def test_forest_revision_reaches_only_the_real_swe_agent_system_message(agent_root):
    """Exercise FOREST and upstream message construction with no inference transport."""
    from sweagent.agent.agents import DefaultAgent, DefaultAgentConfig
    from sweagent.agent.problem_statement import TextProblemStatement

    seed = swe_agent.seed_prompt(agent_root)
    lm = BenchmarkProposerLM()
    reflection = ThreeRoleReflectionLM(base_lm=lm, level=2, templates={"system_prompt": VerbatimPromptTemplate()})
    candidate = {"system_prompt": seed}
    proposal, _ = reflection.reflect(
        candidate,
        {"system_prompt": [{"Inputs": "A training task", "Feedback": "Clarify the instructions."}]},
        ["system_prompt"],
        metadata={"candidate_idx": 0},
    )
    assert set(proposal.new_texts) == {"system_prompt"}
    edited = proposal.new_texts["system_prompt"]
    assert edited == seed + " improved guidance."
    assert candidate == {"system_prompt": seed}
    assert swe_agent.seed_prompt(agent_root) == seed

    models = BenchmarkModels("openrouter/offline-check", "unused", None, None, {}, {})
    configurations, histories = [], []
    for prompt in (seed, edited):
        config = DefaultAgentConfig.model_validate(swe_agent.configuration(agent_root, models, prompt))
        configurations.append(config.model_dump(mode="json"))
        agent = DefaultAgent.from_config(config)
        agent._problem_statement = TextProblemStatement(id="task", text="Fixed task statement")
        agent._env = SimpleNamespace(repo=SimpleNamespace(repo_name="testbed"))
        agent.add_system_message_to_history()
        agent.add_demonstrations_to_history()
        agent.add_instance_template_to_history(state={"working_dir": "/testbed"})
        histories.append(agent.history)
    assert histories[0][0]["role"] == histories[1][0]["role"] == "system"
    assert histories[0][0]["content"] == seed
    assert histories[1][0]["content"] == edited
    assert histories[0][1:] == histories[1][1:]
    assert histories[0][1]["role"] == "user"
    assert "Fixed task statement" in histories[0][1]["content"]
    for config in configurations:
        config["templates"].pop("system_template")
    assert configurations[0] == configurations[1]


def test_payload_drift_is_not_treated_as_ready(tmp_path):
    (tmp_path / "runtime").write_text("initial")
    (tmp_path / "manifest.json").write_text(json.dumps({"source": {}, "files": {"runtime": "wrong"}}))
    with pytest.raises(ValueError, match="checksum drift"):
        swe_agent.verify_payload(tmp_path)


@pytest.mark.parametrize("reason", ["exit_api", "exit_environment_error", "exit_error"])
@pytest.mark.parametrize("autosubmitted", [False, True])
def test_rollout_preserves_infrastructure_failures_after_autosubmission(
    agent_root, tmp_path, monkeypatch, reason, autosubmitted
):
    """Never return a salvaged patch for grading after a provider or runtime failure."""
    from sweagent.agent.agents import DefaultAgent
    from sweagent.environment import swe_env

    from examples.swebench_live import swe_agent_rex

    status = f"submitted ({reason})" if autosubmitted else reason
    info = {"exit_status": status, "submission": "partial patch"}
    trajectory = {"info": info, "trajectory": [{"observation": "interrupted work"}]}
    agent = SimpleNamespace(run=lambda **_: SimpleNamespace(info=info), get_trajectory_data=lambda: trajectory)
    environment = SimpleNamespace(start=Mock(), close=Mock())
    monkeypatch.setattr(DefaultAgent, "from_config", lambda _: agent)
    monkeypatch.setattr(swe_env, "SWEEnv", lambda **_: environment)
    monkeypatch.setattr(swe_agent_rex, "ApptainerDeployment", lambda *_: None)
    models = BenchmarkModels("openrouter/offline-check", "unused", None, None, {}, {})

    with pytest.raises(RuntimeError, match="preserve the unresolved trial"):
        swe_agent.rollout(agent_root, {"id": "task", "problem_statement": "task"}, {}, models, "system", tmp_path, 16)

    assert json.loads((tmp_path / "trajectory.json").read_text()) == trajectory
    environment.close.assert_called_once_with()


def test_real_swerex_retains_shell_state_and_transfers_errors(agent_root, tmp_path):
    from swerex.runtime.abstract import BashAction, CreateBashSessionRequest, ReadFileRequest

    from examples.swebench_live.swe_agent_rex import PipeRuntime

    worker = Path(swe_agent.__file__).with_name("rex_worker.py")
    env = {"PATH": os.environ["PATH"], "PYTHONPATH": str((agent_root / "source").resolve())}
    with (tmp_path / "rex.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-B", str(worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            env=env,
            cwd=tmp_path,
        )
        runtime = PipeRuntime(process)

        async def check():
            assert await runtime.is_alive()
            await runtime.create_session(CreateBashSessionRequest())
            await runtime.run_in_session(BashAction(command="export SWE_REX_PROBE=retained", timeout=5))
            result = await runtime.run_in_session(BashAction(command='printf %s "$SWE_REX_PROBE"', timeout=5))
            assert result.output == "retained"
            with pytest.raises(FileNotFoundError):
                await runtime.read_file(ReadFileRequest(path=str(tmp_path / "does-not-exist")))
            await runtime.close()

        try:
            asyncio.run(check())
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=10)
