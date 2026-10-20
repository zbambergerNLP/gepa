"""Use the SWE-agent implementation linked by SWE-bench Live's evaluation guide."""

import copy
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import yaml

from examples.common.artifacts import atomic_json, digest
from examples.common.react_v2 import file_sha256

AGENT_URL = "https://github.com/njukenanli/SWE-agent-for-eval.git"
AGENT_REVISION = "824224225e1e952b48da91f2d1bde38adb2115f2"


def source_identity(source: Path) -> dict:
    """Reject a different agent, configuration or tool implementation."""

    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], text=True, timeout=30).strip()

    if git("rev-parse", "HEAD") != AGENT_REVISION or git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("SWE-agent source revision or tracked-file drift")
    files = git("ls-files").splitlines()
    return {
        "repository": AGENT_URL,
        "revision": AGENT_REVISION,
        "files_sha256": digest({name: file_sha256(source / name) for name in files if (source / name).is_file()}),
        "config_sha256": file_sha256(source / "config/default.yaml"),
    }


def activate(root: Path) -> dict:
    """Load only the pinned upstream source, retaining its bundled SWE-ReX."""
    source = (root / "source").resolve()
    identity = source_identity(source)
    for name in ("sweagent", "swerex"):
        module = sys.modules.get(name)
        if module is not None and Path(module.__file__).resolve().parent != source / name:
            raise ValueError(f"Already imported a different {name} runtime")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return identity


def configuration(root: Path, models, prompt: str | None = None, *, max_steps: int | None = None) -> dict:
    """Preserve upstream agent behavior; bind the explicitly chosen API model."""
    config = yaml.safe_load((root / "source/config/default.yaml").read_text())["agent"]
    config = copy.deepcopy(config)
    config["model"]["name"] = models.solver_model
    config["model"]["api_base"] = models.solver_api_base
    config["model"]["completion_kwargs"] = copy.deepcopy(models.solver_kwargs)
    # The shared provider wrapper owns retries and durable attempt accounting.
    # Do not retry an exhausted or unknown provider outcome in a second layer.
    config["model"]["retry"] = {"retries": 1, "min_wait": 0, "max_wait": 0}
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError("SWE-agent max_steps must be positive")
        config["model"]["per_instance_call_limit"] = max_steps
    if prompt is not None:
        config["templates"]["system_template"] = prompt
    for bundle in config["tools"]["bundles"]:
        bundle["path"] = str((root / "source" / bundle["path"]).resolve())
    return config


def seed_prompt(root: Path) -> str:
    """Return the upstream system template without adding a custom wrapper."""
    return yaml.safe_load((root / "source/config/default.yaml").read_text())["agent"]["templates"]["system_template"]


def runtime_identity(root: Path, models, *, max_steps: int | None = None) -> dict:
    """Record the exact agent and resolved config in the experiment contract."""
    identity = activate(root)
    from sweagent.agent.agents import DefaultAgentConfig

    config = DefaultAgentConfig.model_validate(configuration(root, models, max_steps=max_steps))
    result = {
        "agent": "SWE-agent",
        "source": identity,
        "config": config.model_dump(mode="json"),
        "transport": "apptainer-swerex-stdio-v1",
        "payload": verify_payload(root / "payload"),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    }
    if max_steps is not None:
        result["step_budget"] = {"max_queries": max_steps, "enforcement": "before_query_v1"}
    return result


class _StepLimitedModel:
    """Count agent queries, including recovery queries, before model dispatch."""

    def __init__(self, model, max_steps: int):
        if max_steps < 1:
            raise ValueError("SWE-agent max_steps must be positive")
        self.model = model
        self.max_steps = max_steps
        self.queries = 0

    def __getattr__(self, name):
        return getattr(self.model, name)

    def query(self, *args, **kwargs):
        from sweagent.exceptions import InstanceCallLimitExceededError

        # The pinned upstream model checks its limit after sending a request.
        if self.queries >= self.max_steps:
            raise InstanceCallLimitExceededError(f"SWE-agent reached its {self.max_steps}-query limit")
        self.queries += 1
        return self.model.query(*args, **kwargs)


def verify_payload(payload: Path) -> dict:
    """Verify the portable in-container runtime and reject missing or changed bytes."""
    saved = json.loads((payload / "manifest.json").read_text())
    actual = {
        str(p.relative_to(payload)): file_sha256(p)
        for p in payload.rglob("*")
        if p.is_file() and p != payload / "manifest.json"
    }
    if actual != saved["files"]:
        raise ValueError("SWE-ReX payload checksum drift")
    return {"files_sha256": digest(actual), "file_count": len(actual), "source": saved["source"]}


def rollout(
    root: Path,
    record: dict,
    image: dict,
    models,
    prompt: str,
    trial: Path,
    memory_gib: int,
    test_model=None,
    *,
    max_steps: int | None = None,
) -> dict:
    """Run DefaultAgent and its standard tools, then close the solver before grading."""
    activate(root)
    from sweagent.agent.agents import DefaultAgent, DefaultAgentConfig
    from sweagent.agent.problem_statement import TextProblemStatement
    from sweagent.environment.repo import PreExistingRepoConfig
    from sweagent.environment.swe_env import SWEEnv

    from examples.swebench_live.swe_agent_rex import ApptainerDeployment

    config = DefaultAgentConfig.model_validate(configuration(root, models, prompt, max_steps=max_steps))
    atomic_json(trial / "swe-agent-config.json", config.model_dump(mode="json"))
    agent = DefaultAgent.from_config(config)
    if test_model is not None:
        agent.model = test_model
    if max_steps is not None:
        agent.model = _StepLimitedModel(agent.model, max_steps)
    env = SWEEnv(
        deployment=ApptainerDeployment(image, memory_gib, root / "payload", trial),
        repo=PreExistingRepoConfig(repo_name="testbed", base_commit="HEAD"),
        post_startup_commands=["export PIP_NO_INDEX=1 PIP_FIND_LINKS=/opt/gepa-swerex/wheels"],
    )
    try:
        env.start()
        result = agent.run(
            env=env,
            problem_statement=TextProblemStatement(id=record["id"], text=record["problem_statement"]),
            output_dir=trial,
        )
        atomic_json(trial / "trajectory.json", agent.get_trajectory_data())
        status = result.info.get("exit_status", "")
        # Upstream salvages a partial patch by wrapping the original failure status.
        exit_reason = status.removeprefix("submitted (").removesuffix(")")
        if exit_reason in {"exit_api", "exit_environment_error", "exit_error"}:
            raise RuntimeError(f"SWE-agent stopped with {status}; preserve the unresolved trial")
        patch = result.info.get("submission") or ""
        return {"patch": patch, "info": result.info, "trajectory": agent.get_trajectory_data()}
    finally:
        env.close()
