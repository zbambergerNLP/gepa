"""Execute upstream text simulations; import this only inside the tau environment."""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import math
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import litellm
import tau2
import tomllib
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.simulation import TextRunConfig
from tau2.data_model.tasks import Task
from tau2.domains.banking_knowledge import environment as banking_environment
from tau2.evaluator.evaluator import EvaluationType
from tau2.registry import registry
from tau2.runner.build import _build_env_kwargs, build_text_orchestrator
from tau2.runner.simulation import run_simulation
from tau2.utils import llm_utils
from tau2.utils.utils import DATA_DIR

from examples.common.provider_retries import provider_retry_kwargs
from examples.taubench.benchmark_settings import (
    DOMAIN,
    MAX_ERRORS,
    MAX_STEPS,
    PYTHON_VERSION,
    RETRIEVAL_CONFIG,
    RETRIEVAL_TOP_K,
    TEST_REPETITIONS,
    TRIAL_SEED,
    UPSTREAM_VERSION,
)
from examples.taubench.model_settings import JUDGE_MODEL, USER_KWARGS, USER_MODEL
from examples.taubench.utils import DATA_PATH, digest, load_data, upstream_system_prompt
from gepa.lm_constants import PROVIDER_RETRY_KEY

AGENT_NAME = "gepa_banking_system_prompt"
_ORIGINAL_KNOWLEDGE_LOADER = banking_environment.get_knowledge_base


class PromptAgent(LLMAgent):
    """Change only the system prompt consumed by the ordinary upstream LLMAgent."""

    def __init__(self, *, prompt: str, **kwargs):
        self.prompt = prompt
        super().__init__(**kwargs)

    @property
    def system_prompt(self) -> str:
        return self.prompt


def ordered_knowledge_base():
    """Fix upstream filesystem iteration order before BM25 indexes the corpus."""
    knowledge = _ORIGINAL_KNOWLEDGE_LOADER()
    knowledge.documents = dict(sorted(knowledge.documents.items()))
    return knowledge


def install_transport(log_path: Path) -> None:
    """Apply shared retries to tau's imported completion binding, including judges."""
    judge_settings = provider_retry_kwargs(log_path=log_path, role="tau_judge")

    def completion(**kwargs):
        if PROVIDER_RETRY_KEY not in kwargs:
            if kwargs.get("model") != JUDGE_MODEL:
                raise ValueError("Unpinned evaluator model")
            kwargs = {**kwargs, **judge_settings}
            kwargs.setdefault("timeout", USER_KWARGS["timeout"])
            kwargs.setdefault("api_base", USER_KWARGS["api_base"])
        return litellm.completion(**kwargs)

    llm_utils.completion = completion


def validate_simulation(simulation, task: Task, seed: int) -> None:
    """Reject absent grades and infrastructure errors, retaining official failed attempts."""
    if simulation.task_id != task.id or simulation.seed != seed or not simulation.messages:
        raise ValueError("Mismatched task, missing seed, or missing conversation")
    if simulation.termination_reason.value in {"infrastructure_error", "unexpected_error"}:
        raise RuntimeError("Official simulation failed to execute")
    reward = simulation.reward_info.reward if simulation.reward_info is not None else None
    if (
        isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or not math.isfinite(reward)
        or reward not in (0, 1)
    ):
        raise ValueError("Missing or invalid official binary reward")
    normal = simulation.termination_reason.value in {"agent_stop", "user_stop"}
    if not normal and reward != 0:
        raise ValueError("An incomplete episode cannot receive positive reward")
    if normal and set(simulation.reward_info.reward_basis or []) != set(task.evaluation_criteria.reward_basis):
        raise ValueError("Official evaluator did not apply every required reward basis")
    if normal and "NL_ASSERTION" in task.evaluation_criteria.reward_basis:
        checks = simulation.reward_info.nl_assertions or []
        expected = task.evaluation_criteria.nl_assertions or []
        if len(checks) != len(expected) or sorted(check.nl_assertion for check in checks) != sorted(expected):
            raise ValueError("Official NL judge returned incomplete or mismatched assertions")


def run_request(request: dict) -> dict:
    """Verify pinned inputs before constructing any model or running an episode."""
    source = Path(request["source"]).resolve()
    if Path(tau2.__file__).resolve().parent != source / "src/tau2" or DATA_DIR.resolve() != source / "data":
        raise ValueError("Imported tau code/data does not belong to the verified checkout")
    if importlib.metadata.version("tau2") != UPSTREAM_VERSION:
        raise ValueError("Installed tau package version drifted")
    if tuple(sys.version_info[:3]) != tuple(map(int, PYTHON_VERSION.split("."))):
        raise ValueError("Python runtime version drifted")
    locked = {
        package["name"]: package["version"] for package in tomllib.loads((source / "uv.lock").read_text())["package"]
    }
    for name in ("litellm", "pydantic", "numpy", "rank-bm25", "websockets"):
        if importlib.metadata.version(name) != locked[name]:
            raise ValueError(f"Installed {name} version differs from the pinned upstream lock")
    splits, manifest = load_data(source)
    tasks = {
        row["task_id"]: Task.model_validate_json((source / DATA_PATH / "tasks" / f"{row['task_id']}.json").read_text())
        for row in manifest["records"]
    }
    banking_environment.get_knowledge_base = ordered_knowledge_base
    if request["mode"] == "inspect":
        env = banking_environment.get_environment(retrieval_variant=RETRIEVAL_CONFIG)
        agent = LLMAgent(tools=env.get_tools(), domain_policy=env.get_policy(), llm="unused")
        if agent.system_prompt != upstream_system_prompt(source):
            raise ValueError("Seed prompt differs from upstream")
        return {
            "manifest_sha256": digest(manifest),
            "seed_sha256": digest(agent.system_prompt),
            "tools": [tool.name for tool in env.get_tools()],
            "task_count": len(manifest["records"]),
            "document_count": manifest["knowledge"]["count"],
            "litellm_version": importlib.metadata.version("litellm"),
        }
    if request["mode"] != "run":
        raise ValueError("Unknown worker mode")
    candidate = request["candidate"]
    if (
        set(candidate) != {"system_prompt"}
        or not isinstance(candidate["system_prompt"], str)
        or not candidate["system_prompt"].strip()
    ):
        raise ValueError("Exactly one nonempty system_prompt is editable")
    trial = request["trial"]
    if type(trial) is not int or not 0 <= trial < TEST_REPETITIONS:
        raise ValueError("Invalid trial index")
    all_records = {row["id"]: row for rows in splits.values() for row in rows}
    records = request["records"]
    if not records or any(all_records.get(row["id"]) != row for row in records):
        raise ValueError("Unknown or modified task record")
    if len({row["split"] for row in records}) != 1:
        raise ValueError("Do not combine optimization and held-out tasks")
    if records[0]["split"] != "test" and trial != 0:
        raise ValueError("Optimization uses one fixed trial")
    artifacts = Path(request["artifacts"])
    artifacts.mkdir(parents=True, exist_ok=True)
    log_path = artifacts / "provider-attempts.jsonl"
    install_transport(log_path)
    prompt = candidate["system_prompt"]

    def factory(*, tools, domain_policy, llm, llm_args, **ignored):
        return PromptAgent(prompt=prompt, tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)

    registry.register_agent_factory(factory, AGENT_NAME)
    solver_kwargs = deepcopy(request["solver_kwargs"])
    if "top_k" in solver_kwargs:
        solver_kwargs["extra_body"] = {**solver_kwargs.get("extra_body", {}), "top_k": solver_kwargs.pop("top_k")}
    solver_kwargs.update(provider_retry_kwargs(log_path=log_path, role="tau_solver"))
    if request["solver_api_base"] is not None:
        solver_kwargs["api_base"] = request["solver_api_base"]
    user_kwargs = {**USER_KWARGS, **provider_retry_kwargs(log_path=log_path, role="tau_user")}
    config = TextRunConfig(
        domain=DOMAIN,
        agent=AGENT_NAME,
        user="user_simulator",
        llm_agent=request["solver_model"],
        llm_args_agent=solver_kwargs,
        llm_user=USER_MODEL,
        llm_args_user=user_kwargs,
        retrieval_config=RETRIEVAL_CONFIG,
        retrieval_config_kwargs={"top_k": RETRIEVAL_TOP_K},
        max_steps=MAX_STEPS,
        max_errors=MAX_ERRORS,
        enforce_communication_protocol=False,
        max_retries=0,
        hallucination_retries=0,
        auto_review=False,
    )
    rng = random.Random(TRIAL_SEED)
    seed = [rng.randint(0, 1000000) for _ in range(TEST_REPETITIONS)][trial]
    outputs = []
    for record in records:
        task = tasks[record["task_id"]].model_copy(deep=True)
        started = time.perf_counter()
        try:
            orchestrator = build_text_orchestrator(config, task, seed=seed)
            simulation = run_simulation(
                orchestrator, evaluation_type=EvaluationType.ALL, env_kwargs=_build_env_kwargs(config, task)
            )
            validate_simulation(simulation, task, seed)
        except Exception as error:
            failure = {
                "id": record["id"],
                "trial": trial,
                "seed": seed,
                "elapsed_seconds": time.perf_counter() - started,
                "error": type(error).__name__,
                "candidate_sha256": digest(candidate),
                "scored": False,
            }
            (artifacts / f"unscored-{task.id}-{uuid4().hex}.json").write_text(json.dumps(failure, allow_nan=False))
            raise
        elapsed = time.perf_counter() - started
        simulation.trial = trial
        simulation.info = {
            **(simulation.info or {}),
            "gepa_candidate": candidate,
            "manifest_sha256": digest(manifest),
            "solver_model": request["solver_model"],
            "user_model": USER_MODEL,
            "judge_model": JUDGE_MODEL,
        }
        filename = f"{record['task_id']}-trial{trial}-{uuid4().hex}.json"
        (artifacts / filename).write_text(simulation.model_dump_json(indent=2))
        reason = simulation.termination_reason.value
        trace = [
            {
                key: message.model_dump(mode="json").get(key)
                for key in ("role", "content", "tool_calls", "name", "tool_call_id", "receiver", "error")
            }
            for message in simulation.messages
        ]
        outputs.append(
            {
                "id": record["id"],
                "task_id": task.id,
                "trial": trial,
                "seed": seed,
                "elapsed_seconds": elapsed,
                "reward": float(simulation.reward_info.reward),
                "termination_reason": reason,
                "error": None if reason in {"agent_stop", "user_stop"} else reason,
                "candidate_sha256": digest(candidate),
                "simulation_path": str(artifacts / filename),
                "messages": trace,
            }
        )
    return {"manifest_sha256": digest(manifest), "outputs": outputs}


def main() -> None:
    """Reserve stdout for one complete JSON document; logs go to stderr."""
    request = json.load(sys.stdin)
    with contextlib.redirect_stdout(sys.stderr):
        result = run_request(request)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
