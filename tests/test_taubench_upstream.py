"""Opt-in offline checks against the real pinned tau runtime, tools, and grader.

Run in the separately locked tau environment; only model API responses are fake.
"""

import json
import math
import os
from pathlib import Path

import pytest
from benchmark_model_fixtures import install_proposer

if os.environ.get("TAU_BANKING_OFFLINE_CHECKS") != "1":
    pytest.skip("Run explicitly inside the pinned tau environment", allow_module_level=True)

import litellm
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.simulation import TextRunConfig
from tau2.data_model.tasks import Task
from tau2.domains.banking_knowledge.environment import get_environment
from tau2.metrics.agent_metrics import pass_hat_k
from tau2.registry import registry
from tau2.runner.build import _build_env_kwargs
from tau2.user.user_simulator_base import STOP

from examples.common.experiment_models import QWEN3_8_27B_MODEL
from examples.taubench.adapter import trial_seed
from examples.taubench.benchmark_settings import DEFAULT_SOURCE
from examples.taubench.model_settings import USER_MODEL
from examples.taubench.utils import DATA_PATH, load_data, upstream_system_prompt
from examples.taubench.worker import AGENT_NAME, run_request, validate_simulation

pytestmark = pytest.mark.smoke


@pytest.fixture
def source():
    return Path(os.environ.get("TAU_BANKING_SOURCE", DEFAULT_SOURCE)).resolve()


@pytest.fixture
def offline_provider(monkeypatch):
    requests = []
    calls = {"user": 0, "solver": 0}
    factories = dict(registry._agent_factories)
    factories.pop(AGENT_NAME, None)
    monkeypatch.setattr(registry, "_agent_factories", factories)

    def completion(**kwargs):
        requests.append(kwargs)
        message = {"role": "assistant", "content": "The task is successfully completed."}
        reason = "stop"
        if kwargs["model"] == USER_MODEL:
            calls["user"] += 1
            message["content"] = "I need help choosing a cash back credit card." if calls["user"] % 2 else STOP
        elif kwargs["model"] == QWEN3_8_27B_MODEL:
            calls["solver"] += 1
            if calls["solver"] % 2:
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "KB_search",
                                "arguments": json.dumps({"query": "credit card cash back"}),
                            },
                        }
                    ],
                }
                reason = "tool_calls"
        else:
            pytest.fail("Unexpected paid or unpinned model")
        return litellm.ModelResponse(
            model=kwargs["model"],
            choices=[{"index": 0, "message": message, "finish_reason": reason}],
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        )

    monkeypatch.setattr(litellm, "completion", completion)
    return requests


def test_pinned_data_and_seed_equal_official_runtime(source):
    splits, manifest = load_data(source)
    env = get_environment(retrieval_variant="bm25")
    agent = LLMAgent(tools=env.get_tools(), domain_policy=env.get_policy(), llm="unused")
    assert upstream_system_prompt(source) == agent.system_prompt
    assert "KB_search" in {tool.name for tool in env.get_tools()}
    assert "shell" not in {tool.name for tool in env.get_tools()}
    assert sum(map(len, splits.values())) == 97 and manifest["knowledge"]["count"] == 698


def test_repeated_training_task_runs_twice_with_distinct_artifacts(source, tmp_path, offline_provider):
    splits, _ = load_data(source)
    record = next(record for record in splits["train"] if record["task_id"] == "task_001")
    candidate = {"system_prompt": "Run every training occurrence"}
    result = run_request(
        {
            "mode": "run",
            "source": str(source),
            "artifacts": str(tmp_path),
            "records": [record, record],
            "candidate": candidate,
            "trial": 0,
            "solver_model": QWEN3_8_27B_MODEL,
            "solver_api_base": None,
            "solver_kwargs": {"timeout": 3600},
        }
    )
    outputs = result["outputs"]
    assert len(outputs) == 2
    assert [output["id"] for output in outputs] == [record["id"], record["id"]]
    assert len({output["simulation_path"] for output in outputs}) == 2
    simulations = [json.loads(Path(output["simulation_path"]).read_text()) for output in outputs]
    assert len({simulation["id"] for simulation in simulations}) == 2
    assert all(simulation["info"]["gepa_candidate"] == candidate for simulation in simulations)
    assert all(output["elapsed_seconds"] > 0 and output["error"] is None for output in outputs)
    assert sum(request["model"] == QWEN3_8_27B_MODEL for request in offline_provider) == 4
    assert sum(request["model"] == USER_MODEL for request in offline_provider) == 4
    attempts = [json.loads(line) for line in (tmp_path / "provider-attempts.jsonl").read_text().splitlines()]
    assert len(attempts) == 8 and len({attempt["request_id"] for attempt in attempts}) == 8


@pytest.mark.parametrize("split", ["val", "test"])
def test_worker_rejects_duplicate_evaluation_records(source, tmp_path, offline_provider, split):
    splits, _ = load_data(source)
    record = splits[split][0]
    with pytest.raises(ValueError, match="Duplicate"):
        run_request(
            {
                "mode": "run",
                "source": str(source),
                "artifacts": str(tmp_path),
                "records": [record, record],
                "candidate": {"system_prompt": "a"},
                "trial": 0,
                "solver_model": QWEN3_8_27B_MODEL,
                "solver_api_base": None,
                "solver_kwargs": {},
            }
        )
    assert offline_provider == []


@pytest.mark.parametrize("prompt", ["First actual candidate system prompt", "Changed candidate system prompt"])
def test_actual_agent_user_bm25_and_end_state_grader(source, tmp_path, offline_provider, prompt):
    splits, _ = load_data(source)
    record = next(r for records in splits.values() for r in records if r["task_id"] == "task_001")
    payload = {
        "mode": "run",
        "source": str(source),
        "artifacts": str(tmp_path),
        "records": [record],
        "candidate": {"system_prompt": prompt},
        "trial": 0,
        "solver_model": QWEN3_8_27B_MODEL,
        "solver_api_base": "http://localhost:9999/v1",
        "solver_kwargs": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 65536,
            "timeout": 3600,
            "extra_body": {"thinking_token_budget": 32768, "chat_template_kwargs": {"enable_thinking": True}},
        },
    }
    result = run_request(payload)
    output = result["outputs"][0]
    assert output["reward"] == 0  # Saying "success" changed no DB state.
    assert output["error"] is None and output["termination_reason"] == "user_stop"
    assert output["elapsed_seconds"] > 0 and output["seed"] == trial_seed(0)
    trace = json.loads(Path(output["simulation_path"]).read_text())
    assert trace["reward_info"]["db_check"]["db_match"] is False
    solver_requests = [r for r in offline_provider if r["model"] == QWEN3_8_27B_MODEL]
    user_requests = [r for r in offline_provider if r["model"] == USER_MODEL]
    assert len(solver_requests) == 2 and len(user_requests) == 2
    assert any(m["role"] == "tool" and "doc_credit_cards" in m["content"] for m in solver_requests[1]["messages"])
    for request in solver_requests:
        assert request["messages"][0]["content"] == prompt
        assert "Sarah Bosch" not in json.dumps(request["messages"])
        assert "evaluation_criteria" not in json.dumps(request["messages"])
        assert request["max_tokens"] == 65536 and request["extra_body"]["thinking_token_budget"] == 32768
        assert request["num_retries"] == 0 and request["api_base"] == "http://localhost:9999/v1"
    assert all(r["temperature"] == 0 and r["seed"] == trial_seed(0) for r in user_requests)
    attempts = [json.loads(line) for line in (tmp_path / "provider-attempts.jsonl").read_text().splitlines()]
    assert len(attempts) == 4 and {r["role"] for r in attempts} == {"tau_solver", "tau_user"}


def test_grading_fixes_and_upstream_read_allowlist_are_preserved(source):
    task = Task.model_validate_json((source / DATA_PATH / "tasks/task_085.json").read_text())
    kwargs = _build_env_kwargs(TextRunConfig(domain="banking_knowledge", retrieval_config="bm25"), task)
    expected = {
        a.arguments["agent_tool_name"]
        for a in task.evaluation_criteria.actions
        if a.name == "call_discoverable_agent_tool"
    }
    assert kwargs["read_log_allowlist"] == expected and expected
    repaired_task = json.loads((source / DATA_PATH / "tasks/task_074.json").read_text())
    assert any("14.5" in json.dumps(action) for action in repaired_task["evaluation_criteria"]["actions"])


def test_pass_hat_k_formula_matches_official_code():
    for n in (1, 4):
        for successes in range(n + 1):
            for k in range(1, n + 1):
                assert pass_hat_k(n, successes, k) == math.comb(successes, k) / math.comb(n, k)


def test_actual_max_steps_is_failure_and_cannot_be_reported_as_success(source, tmp_path, offline_provider, monkeypatch):
    from tau2.data_model.simulation import SimulationRun

    monkeypatch.setattr("examples.taubench.worker.MAX_STEPS", 1)
    splits, _ = load_data(source)
    record = splits["train"][0]
    result = run_request(
        {
            "mode": "run",
            "source": str(source),
            "artifacts": str(tmp_path),
            "records": [record],
            "candidate": {"system_prompt": "A candidate"},
            "trial": 0,
            "solver_model": QWEN3_8_27B_MODEL,
            "solver_api_base": None,
            "solver_kwargs": {"timeout": 3600},
        }
    )
    output = result["outputs"][0]
    assert output["error"] == "max_steps" and output["reward"] == 0
    simulation = SimulationRun.model_validate_json(Path(output["simulation_path"]).read_text())
    simulation.reward_info.reward = 1
    task = Task.model_validate_json((source / DATA_PATH / "tasks" / f"{record['task_id']}.json").read_text())
    with pytest.raises(ValueError, match="incomplete"):
        validate_simulation(simulation, task, trial_seed(0))


def test_required_nl_assertion_uses_fixed_judge_and_rejects_empty_grades(source, tmp_path, monkeypatch):
    factories = dict(registry._agent_factories)
    factories.pop(AGENT_NAME, None)
    monkeypatch.setattr(registry, "_agent_factories", factories)
    judge_calls = []
    user_calls = []

    def completion(**kwargs):
        assert kwargs["model"] == USER_MODEL
        if "expectedOutcomes:" in kwargs["messages"][-1]["content"]:
            judge_calls.append(kwargs)
            content = json.dumps({"results": []})
        else:
            user_calls.append(kwargs)
            content = STOP
        return litellm.ModelResponse(
            model=kwargs["model"],
            choices=[{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        )

    monkeypatch.setattr(litellm, "completion", completion)
    splits, _ = load_data(source)
    record = next(r for records in splits.values() for r in records if r["task_id"] == "task_102")
    with pytest.raises(ValueError, match="incomplete or mismatched assertions"):
        run_request(
            {
                "mode": "run",
                "source": str(source),
                "artifacts": str(tmp_path),
                "records": [record],
                "candidate": {"system_prompt": "A candidate"},
                "trial": 0,
                "solver_model": QWEN3_8_27B_MODEL,
                "solver_api_base": None,
                "solver_kwargs": {"timeout": 3600},
            }
        )
    assert len(judge_calls) == len(user_calls) == 1
    assert judge_calls[0]["api_base"] == user_calls[0]["api_base"] == "https://api.openai.com/v1"
    attempts = [json.loads(line) for line in (tmp_path / "provider-attempts.jsonl").read_text().splitlines()]
    assert {row["role"] for row in attempts} == {"tau_user", "tau_judge"}
    assert not list(tmp_path.glob("task_102*.json"))
    failed = json.loads(next(tmp_path.glob("unscored-task_102-*.json")).read_text())
    assert failed["scored"] is False and failed["elapsed_seconds"] > 0


@pytest.mark.parametrize("condition", ["vanilla", "random", "action", "react_v2_random", "react_v2"])
def test_optimizer_pilot_uses_actual_tau_runtime_and_training_only(
    source, tmp_path, monkeypatch, offline_provider, condition
):
    """Exercise real optimization, simulation and grading with only model responses scripted."""
    from examples.taubench import main
    from examples.taubench.runtime import TauRuntime

    requests = []

    def invoke(runtime, payload):
        requests.append(payload)
        registry._agent_factories.pop(AGENT_NAME, None)
        return run_request({**payload, "source": str(runtime.source), "artifacts": str(runtime.artifacts)})

    monkeypatch.setattr(TauRuntime, "invoke", invoke)
    proposers = install_proposer(monkeypatch)
    root = tmp_path / "optimizer-run"
    argv = [
        "--mode",
        "optimizer-pilot",
        "--condition",
        condition,
        "--pilot-size",
        "1",
        "--pilot-proposals",
        "1",
        "--tau-source",
        str(source),
        "--run-dir",
        str(root),
    ]
    assert main.main(argv) == 0
    directory = root / "optimizer-pilot" / condition
    winner = json.loads((directory / "pilot-winner.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    assert winner["selection_split"] == "train"
    assert winner["training_score"] == 0
    assert not {"test", "baseline"} & summary.keys()
    train_id = load_data(source)[0]["train"][0]["id"]
    assert len(requests) >= 3
    assert all(request["trial"] == 0 for request in requests)
    assert {record["id"] for request in requests for record in request["records"]} == {train_id}
    assert all(record["split"] == "train" for request in requests for record in request["records"])
    seed_prompt = requests[0]["candidate"]["system_prompt"]
    assert [line for line in seed_prompt.splitlines() if line.startswith("## ")] == ["## Objective"]
    assert "### Rho-Bank Customer Service Policy" in seed_prompt
    assert "#### Guidelines" in seed_prompt and "##### Authenticating Users" in seed_prompt
    assert [line.lstrip("#").strip() for line in seed_prompt.splitlines()[1:] if line.strip()] == [
        line.lstrip("#").strip() for line in upstream_system_prompt(source).splitlines() if line.strip()
    ]
    assert any("improved" in request["candidate"]["system_prompt"] for request in requests)
    solver_calls = [request for request in offline_provider if request["model"] == QWEN3_8_27B_MODEL]
    assert any("improved" in request["messages"][0]["content"] for request in solver_calls)
    assert any(instance.calls for instance in proposers)
    simulations = [json.loads(path.read_text()) for path in (root / "tau-episodes").glob("task_*.json")]
    assert len(simulations) == len(requests)
    assert all(simulation["reward_info"]["reward"] == 0 for simulation in simulations)
    assert any(message["role"] == "tool" for simulation in simulations for message in simulation["messages"])
    assert not list(root.rglob("heldout")) and not list(root.rglob("frozen-winner.json"))
    call_count = len(requests)
    assert main.main(argv) == 0
    assert len(requests) == call_count
