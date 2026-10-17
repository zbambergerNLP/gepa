"""Check the upstream-agent boundary without making model requests."""

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from examples.common.benchmark_types import BenchmarkModels
from examples.swebench_live import swe_agent
from examples.swebench_live.main import add_arguments, build_benchmark


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


@pytest.mark.parametrize("condition", ["forest", "both", "all", "random", "action", "react_v2_random"])
def test_structured_optimization_stops_before_runtime_or_model_work(condition):
    args = argparse.Namespace(api_profile="explicit.json", mode="optimize", condition=condition)
    with pytest.raises(ValueError, match="will not be reformatted automatically"):
        build_benchmark(args, None)


def test_pinned_default_agent_configuration_is_preserved(agent_root):
    from sweagent.agent.agents import DefaultAgent, DefaultAgentConfig

    original = yaml.safe_load((agent_root / "source/config/default.yaml").read_text())["agent"]
    models = BenchmarkModels(
        "openrouter/test", "unused", "https://openrouter.ai/api/v1", None, {"temperature": 0.5, "max_tokens": 1234}, {}
    )
    config = swe_agent.configuration(agent_root, models)
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


def test_payload_drift_is_not_treated_as_ready(tmp_path):
    (tmp_path / "runtime").write_text("initial")
    (tmp_path / "manifest.json").write_text(json.dumps({"source": {}, "files": {"runtime": "wrong"}}))
    with pytest.raises(ValueError, match="checksum drift"):
        swe_agent.verify_payload(tmp_path)


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
