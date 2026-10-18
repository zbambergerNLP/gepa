"""Check the upstream-agent boundary without making model requests."""

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from benchmark_model_fixtures import BenchmarkProposerLM

from examples.common.benchmark_types import BenchmarkModels
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
    assert not hasattr(args, "agent_max_steps")


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
        monkeypatch.setattr(swe_main, name, lambda *_: {})
    monkeypatch.setattr(swe_main, "SWEBenchLiveAdapter", lambda *_, **__: Mock())
    benchmark = build_benchmark(args, None)
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
