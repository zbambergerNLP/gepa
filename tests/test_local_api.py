"""Verify hosted routing, spend boundaries and native harness bindings without paid calls."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from benchmark_model_fixtures import install_proposer
from test_benchmark_runner import definition

from examples.common import benchmark_runner as runner
from examples.common.experiment_models import experiment_model_version
from examples.local_api import cli, profile
from examples.taubench.runtime import TauRuntime
from gepa.strategies.jev_controller import JevController, JevControllerError


def endpoint(model, tag="provider-a", prompt="0.000001", completion="0.000002"):
    return {
        "status": 0,
        "model_id": model,
        "tag": tag,
        "provider_name": tag,
        "name": f"{tag} | {model}-2025-04-14",
        "quantization": "unknown",
        "context_length": 131_072,
        "max_completion_tokens": 65_536,
        "pricing": {"prompt": prompt, "completion": completion},
        "supported_parameters": [
            "tools",
            "tool_choice",
            "max_tokens",
            "temperature",
            "seed",
            "reasoning",
            "response_format",
            "structured_outputs",
        ],
    }


@pytest.fixture
def hosted(tmp_path, monkeypatch):
    monkeypatch.setattr(profile, "endpoints", lambda model: [endpoint(model)])
    record = profile.make_profile("openai/fixture-solver", "anthropic/fixture-proposer")
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(record))
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    return path, record


def test_cost_selection_uses_both_prices_and_rejects_missing_parameters(monkeypatch):
    def options(model):
        return [
            endpoint(model, "input-cheap", "0.000001", "0.000010"),
            endpoint(model, "output-cheap", "0.000005", "0.000001"),
            {**endpoint(model, "unsupported", "0", "0"), "supported_parameters": ["max_tokens"]},
        ]

    monkeypatch.setattr(profile, "endpoints", options)
    assert (
        profile.make_profile("openai/a", "openai/b", input_output_ratio=1)["roles"]["solver"]["tag"] == "output-cheap"
    )
    assert (
        profile.make_profile("openai/a", "openai/b", input_output_ratio=10)["roles"]["solver"]["tag"] == "input-cheap"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", -2),
        ("quantization", "fp4"),
        ("name", "new revision"),
        ("context_length", 65_536),
        ("pricing", {"prompt": "1", "completion": "1"}),
    ],
)
def test_route_drift_never_silently_falls_back(hosted, monkeypatch, field, value):
    _, record = hosted
    monkeypatch.setattr(profile, "endpoints", lambda model: [{**endpoint(model), field: value}])
    with pytest.raises(ValueError, match="Pinned"):
        profile.verify_routes(record)


def test_profile_and_credentials_do_not_import_other_secrets(hosted, monkeypatch, tmp_path):
    path, record = hosted
    env = tmp_path / ".env"
    env.write_text("OPENROUTER_API_KEY=example-key\nUNRELATED_SECRET=never-import\n")
    monkeypatch.delenv("UNRELATED_SECRET", raising=False)
    assert profile.load_credential(env) == "example-key"
    import os

    assert "UNRELATED_SECRET" not in os.environ
    assert profile.load_profile(path) == record
    record["api_key"] = "should-not-be-here"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="credentials"):
        profile.load_profile(path)


def test_no_default_model_and_no_accidental_benchmark_launch(hosted, monkeypatch):
    with pytest.raises(SystemExit):
        cli.main(["plan", "--output", "unused.json"])
    build = Mock()
    verify = Mock()
    monkeypatch.setattr(runner, "verify_routes", verify)
    with pytest.raises(ValueError, match="allow-api-calls"):
        runner.run_cli(
            benchmark_name="local",
            build_benchmark=build,
            add_arguments=lambda _: None,
            argv=["--api-profile", str(hosted[0])],
        )
    build.assert_not_called()
    verify.assert_not_called()


@pytest.mark.parametrize("benchmark", profile.BENCHMARKS)
def test_hosted_identity_reaches_each_shared_benchmark_without_local_model_defaults(hosted, tmp_path, benchmark):
    path, record = hosted
    seen = []

    def build(args, models):
        seen.append(models)
        return definition(args.run_dir)

    run = tmp_path / benchmark
    assert (
        runner.run_cli(
            benchmark_name=benchmark,
            build_benchmark=build,
            add_arguments=lambda _: None,
            argv=["--api-profile", str(path), "--allow-api-calls", "--mode", "pilot", "--run-dir", str(run)],
        )
        == 0
    )
    models = seen[0]
    assert models.solver_model == "openrouter/openai/fixture-solver"
    assert models.proposer_model == "openrouter/anthropic/fixture-proposer"
    assert "temperature" not in models.proposer_kwargs
    assert models.solver_kwargs["extra_body"]["provider"]["allow_fallbacks"] is False
    assert "chat_template_kwargs" not in models.solver_kwargs["extra_body"]
    assert experiment_model_version(models.solver_model) == "provider-managed; checkpoint revision unavailable"
    contracts = list(run.rglob("*.json"))
    assert any(record["roles"]["solver"]["model_id"] in p.read_text() for p in contracts)
    assert all("fixture-key" not in p.read_text() for p in contracts)


def test_reasoning_parameters_are_explicit_and_not_silently_dropped(hosted):
    _, record = hosted
    record["roles"]["proposer"]["parameters"]["reasoning_effort"] = "high"
    record["roles"]["proposer"]["parameters"]["top_k"] = 20
    kwargs = profile.role_kwargs(record, "proposer")
    assert kwargs["extra_body"]["reasoning"] == {"effort": "high"}
    assert "thinking_token_budget" not in kwargs["extra_body"]
    assert "reasoning_effort" not in kwargs
    assert kwargs["extra_body"]["top_k"] == 20
    assert "top_k" not in kwargs


@pytest.mark.parametrize("condition", ["vanilla", "forest"])
def test_hosted_optimizer_builds_real_roles_without_inserting_sampling_defaults(
    hosted, tmp_path, monkeypatch, condition
):
    instances = install_proposer(monkeypatch)
    run = tmp_path / "optimizer"
    assert (
        runner.run_cli(
            benchmark_name="local",
            build_benchmark=lambda args, models: definition(run),
            add_arguments=lambda _: None,
            argv=[
                "--api-profile",
                str(hosted[0]),
                "--allow-api-calls",
                "--mode",
                "optimizer-pilot",
                "--condition",
                condition,
                "--template-family",
                "alibaba",
                "--run-dir",
                str(run),
            ],
        )
        == 0
    )
    assert instances and any(instance.calls for instance in instances)
    assert all(instance.model == "openrouter/anthropic/fixture-proposer" for instance in instances)
    assert all("temperature" not in instance.completion_kwargs for instance in instances)
    assert all(instance.completion_kwargs["extra_body"]["provider"]["only"] == ["provider-a"] for instance in instances)


def test_decisionbench_rejects_a_route_without_structured_outputs_before_build(hosted, monkeypatch):
    path, record = hosted
    record["roles"]["solver"]["supported_parameters"].remove("structured_outputs")
    path.write_text(json.dumps(record))
    build = Mock()
    with pytest.raises(ValueError, match="JSON-schema"):
        runner.run_cli(
            benchmark_name="decisionbench",
            build_benchmark=build,
            add_arguments=lambda _: None,
            argv=["--api-profile", str(path), "--allow-api-calls"],
        )
    build.assert_not_called()


def test_terminalbench_requires_seed_support_for_repetitions(hosted):
    path, record = hosted
    record["roles"]["solver"]["supported_parameters"].remove("seed")
    path.write_text(json.dumps(record))
    build = Mock()
    with pytest.raises(ValueError, match="seed support"):
        runner.run_cli(
            benchmark_name="terminalbench",
            build_benchmark=build,
            add_arguments=lambda _: None,
            argv=["--api-profile", str(path), "--allow-api-calls"],
        )
    build.assert_not_called()


@pytest.mark.parametrize("model", ["openai/gpt-4.1-mini", "openai/gpt-5-nano", "openai/o3"])
def test_hotpotqa_dspy_preserves_exact_hosted_controls(hosted, model):
    pytest.importorskip("dspy")
    from examples.hotpotqa.utils import build_hotpotqa_task_lm

    _, record = hosted
    record["roles"]["solver"]["parameters"] = {"max_tokens": 4096}
    kwargs = profile.role_kwargs(record, "solver")
    client = build_hotpotqa_task_lm("openrouter/" + model, profile.API_BASE, kwargs)
    assert client.model == "openrouter/" + model
    assert client.kwargs == {key: value for key, value in kwargs.items() if key != "num_retries"}
    assert client.num_retries == 0
    assert client.cache is False
    assert client.cache_in_memory is False


def test_check_records_preflight_failure_without_a_model_request(hosted, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "get_json", Mock(side_effect=ValueError("Rejected fixture-key")))
    send = Mock(side_effect=AssertionError("No model calls after failed preflight"))
    monkeypatch.setattr("litellm.completion", send)
    report = cli.connection_check(hosted[1], "fixture-key", tmp_path / "check.json", allow_calls=True)
    assert report["state"] == "preflight_failed"
    assert report["checks"] == []
    assert "fixture-key" not in json.dumps(report)
    send.assert_not_called()


def test_terminalbench_uses_hosted_identity_and_keeps_native_task_requirements(hosted, tmp_path, monkeypatch):
    from examples.terminalbench import main

    parser = runner.build_parser("terminalbench", main.add_arguments)
    args = parser.parse_args(
        [
            "--api-profile",
            str(hosted[0]),
            "--allow-api-calls",
            "--mode",
            "pilot",
            "--task-scope",
            "della88",
            "--run-dir",
            str(tmp_path / "run"),
        ]
    )
    local_runtime = Mock(side_effect=AssertionError("Hosted API must not attest to local GPUs"))
    requirements = Mock()
    monkeypatch.setattr(main, "load_role_runtimes", local_runtime)
    monkeypatch.setattr(main.HarborCLI, "check_requirements", requirements)
    models = main.configure_models(args, runner.resolve_models(args))
    benchmark = main.build_benchmark(args, models)
    assert models.solver_kwargs["max_tokens"] == 32_768
    assert benchmark.runtime["execution_runtime"]["student"]["provider"] == "openrouter"
    assert (len(benchmark.trainset), len(benchmark.valset), len(benchmark.testset)) == (30, 19, 39)
    requirements.assert_called_once()
    local_runtime.assert_not_called()
    assert profile.model_info(hosted[1], "solver")["input_cost_per_token"] > 0


def test_tau_worker_receives_one_explicit_auxiliary_route(hosted, tmp_path, monkeypatch):
    auxiliary = profile.tau_settings(hosted[1])
    seen = []
    monkeypatch.setattr(TauRuntime, "invoke", lambda self, payload: seen.append(payload) or {})
    runtime = TauRuntime(
        tmp_path, tmp_path, "openrouter/openai/fixture-solver", profile.API_BASE, {}, auxiliary=auxiliary
    )
    runtime.run([{"id": "test"}], {"system_prompt": "test"}, 0)
    assert seen[0]["auxiliary"]["model"] == "openrouter/openai/gpt-4.1"
    assert seen[0]["auxiliary"]["kwargs"]["extra_body"]["provider"]["only"] == ["provider-a"]
    assert "fixture-key" not in json.dumps(seen)
    assert hosted[1]["schema_version"] == 2 and "tau_protocol" not in hosted[1]
    assert auxiliary["official_model"] == "gpt-4.1-2025-04-14"
    assert auxiliary["kwargs"]["max_tokens"] == 32768
    assert auxiliary["kwargs"]["temperature"] == 0.0
    assert "reasoning" not in auxiliary["kwargs"]["extra_body"]


def test_custom_tau_cannot_reuse_reference_baseline_or_resume_contract(hosted, tmp_path, monkeypatch):
    from examples.taubench import main

    monkeypatch.setattr(main, "load_data", lambda _: ({"train": [], "val": [], "test": []}, {"records": []}))
    monkeypatch.setattr(main, "upstream_system_prompt", lambda _: "Keep this system instruction")
    args = SimpleNamespace(tau_source=tmp_path, run_dir=tmp_path / "run", template_family="alibaba", max_workers=1)
    models = SimpleNamespace(solver_model="deepseek/fixture", solver_api_base=profile.API_BASE, solver_kwargs={})
    args._api_profile = hosted[1]
    reference = main.build_benchmark(args, models)
    custom = deepcopy(hosted[1])
    custom.update(schema_version=3, tau_protocol=profile.CUSTOM_TAU_PROTOCOL)
    custom["roles"]["tau"]["model_id"] = "deepseek/custom"
    custom["roles"]["tau"]["parameters"]["reasoning_effort"] = "low"
    args._api_profile = custom
    variant = main.build_benchmark(args, models)
    assert reference.seed_candidate == variant.seed_candidate
    assert reference.source == variant.source
    assert "auxiliary_protocol" not in reference.runtime
    assert variant.runtime["auxiliary_protocol"] == profile.CUSTOM_TAU_PROTOCOL
    assert variant.runtime["user_model"] == variant.runtime["judge_model"] == "openrouter/deepseek/custom"
    monkeypatch.setattr(
        runner, "evaluate_candidate", lambda definition, candidate, records, directory, *a, **kw: directory
    )
    old_path = runner._starting_baseline(reference, args.run_dir, reference.runtime, 0)
    new_path = runner._starting_baseline(variant, args.run_dir, variant.runtime, 0)
    assert old_path != new_path
    assert new_path == runner._starting_baseline(variant, tmp_path / "other-method", variant.runtime, 0)
    runner.ensure_contract(args.run_dir, "run-contract.json", reference.runtime)
    with pytest.raises(ValueError, match="configuration or data changed"):
        runner.ensure_contract(args.run_dir, "run-contract.json", variant.runtime)


def test_custom_tau_cli_preserves_low_for_all_roles_and_requires_explicit_opt_in(tmp_path, monkeypatch):
    model = "deepseek/deepseek-v4.1-flash-1017"
    monkeypatch.setattr(profile, "endpoints", lambda name: [endpoint(name, "deepinfra/fp8")])
    path = tmp_path / "custom.json"
    parameters = {"max_tokens": 32768, "reasoning_effort": "low", "seed": 0}
    assert (
        cli.main(
            [
                "plan",
                "--output",
                str(path),
                "--solver-model",
                model,
                "--proposer-model",
                model,
                "--tau-model",
                model,
                "--solver-parameters",
                json.dumps(parameters),
                "--proposer-parameters",
                json.dumps(parameters),
                "--tau-parameters",
                json.dumps(parameters),
            ]
        )
        == 0
    )
    custom = profile.load_profile(path)
    profile.verify_routes(custom)
    assert custom["schema_version"] == 3
    assert custom["tau_protocol"] == profile.CUSTOM_TAU_PROTOCOL
    assert custom["jev"] == profile.JEV_TRANSPORT
    auxiliary = profile.tau_settings(custom)
    assert auxiliary["model"] == "openrouter/" + model
    assert "official_model" not in auxiliary
    for kwargs in (profile.role_kwargs(custom, "solver"), profile.role_kwargs(custom, "proposer"), auxiliary["kwargs"]):
        assert kwargs["extra_body"]["reasoning"] == {"effort": "low"}
        assert kwargs["extra_body"]["provider"]["only"] == ["deepinfra/fp8"]
        assert kwargs["extra_body"]["provider"]["allow_fallbacks"] is False
        assert "reasoning_effort" not in kwargs
    legacy = deepcopy(custom)
    legacy["schema_version"] = 2
    legacy.pop("tau_protocol")
    path.write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match="Incompatible tau"):
        profile.load_profile(path)
    custom["tau_protocol"] = "official"
    path.write_text(json.dumps(custom))
    with pytest.raises(ValueError, match="protocol"):
        profile.load_profile(path)
    with pytest.raises(ValueError, match="explicit tau_model"):
        profile.make_profile(model, model, tau_parameters=parameters)


@pytest.mark.parametrize("control", ["reasoning", "response_format", "structured_outputs"])
def test_custom_tau_rejects_missing_judge_controls_before_calls(tmp_path, monkeypatch, control):
    monkeypatch.setattr(profile, "endpoints", lambda model: [endpoint(model)])
    record = profile.make_profile(
        "openai/a",
        "openai/b",
        tau_model="deepseek/custom",
        tau_parameters={"max_tokens": 32768, "reasoning_effort": "low"},
    )
    broken = endpoint("deepseek/custom")
    broken["supported_parameters"].remove(control)
    monkeypatch.setattr(profile, "endpoints", lambda model: [broken])
    with pytest.raises(ValueError, match="Pinned tau"):
        profile.verify_routes(record, ("tau",))
    record["roles"]["tau"]["supported_parameters"].remove(control)
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Incompatible tau"):
        profile.load_profile(path)


def test_jev_uses_openrouter_key_and_dated_response_identity(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-key")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    observed = {}
    response = {
        "model": profile.JEV_TRANSPORT["response_model"],
        "usage": {"input_tokens": 100, "output_tokens": 10, "cost": 0.0000042},
        "answers": {"edit": {"type": "choice", "choice": "a", "confidence": 1.0, "probabilities": {"a": 1.0}}},
    }

    def client(**kwargs):
        observed.update(kwargs)

        def send(**request):
            observed["request"] = request
            return SimpleNamespace(model_dump=lambda **_: deepcopy(response))

        return SimpleNamespace(system_one=send)

    monkeypatch.setattr("gepa.strategies.jev_controller.typesafe_sdk.TypeSafeClient", client)
    controller = JevController(provider="openrouter")
    request = {
        "model": profile.JEV_TRANSPORT["model"],
        "state": {},
        "questions": {"edit": {"type": "choice", "instructions": "Choose", "criteria": {"a": "A"}}},
    }
    result = controller._live(request, {"a"})
    assert result["physical_attempts"] == 1
    assert observed["base_url"] == "https://openrouter.ai/api"
    assert observed["api_key"] == "fixture-key"
    assert observed["request"]["model"] == "typesafe/jev-1.13"
    assert controller.run_contract() != JevController().run_contract()
    with pytest.raises(JevControllerError, match="different model"):
        controller._validate(
            {**response, "model": "typesafe/jev-1.14"}, {"a"}, expected_model=profile.JEV_TRANSPORT["response_model"]
        )


def test_credential_only_check_does_not_invoke_a_model(hosted, monkeypatch, tmp_path):
    send = Mock(side_effect=AssertionError("No paid checks allowed"))
    monkeypatch.setattr("litellm.completion", send)
    monkeypatch.setattr(cli, "get_json", lambda *args, **kwargs: {"data": {}})
    report = cli.connection_check(hosted[1], "fixture-key", tmp_path / "check.json")
    assert report["checks"] == [] and report["benchmark_runs_started"] == 0
    send.assert_not_called()
