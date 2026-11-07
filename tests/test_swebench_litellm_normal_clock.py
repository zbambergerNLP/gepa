"""Check frozen normal-call cleanup with labeled local dependency stubs."""

import ast
import asyncio
import hashlib
import importlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from test_swebench_litellm_clock_fixture import (
    frozen_limiter as frozen_limiter,
)
from test_swebench_litellm_clock_fixture import (
    recorded_minute_clock,
    rollover_clock,
)
from test_swebench_litellm_evaluation_proposal import record_path as record_path

NORMAL_TEST = "tests/test_litellm/proxy/hooks/test_parallel_request_limiter.py::test_normal_router_call"


def plugin():
    """Allow the same positive regression to run against the exact v3 baseline."""
    version = os.environ.get("SWE_LITELLM_CLOCK_VARIANT", "v4")
    assert version in {"v3", "v4"}
    return importlib.import_module(f"examples.swebench_live.proposals.litellm_13675_{version}.offline_inputs")


@pytest.fixture
def frozen_normal(frozen_limiter, monkeypatch):
    """Use original test and callback bodies with local cache, auth, and delivery stubs."""
    frozen = frozen_limiter
    module = frozen.module
    state = {"deliveries": 0}

    class Response:
        usage = SimpleNamespace(total_tokens=15)

    module.ModelResponse = Response
    module._get_parent_otel_span_from_kwargs = lambda **_: None
    callback_utils = ModuleType("litellm.proxy.common_utils.callback_utils")
    callback_utils.get_model_group_from_litellm_kwargs = lambda _: None
    monkeypatch.setitem(sys.modules, callback_utils.__name__, callback_utils)

    class Logging:
        def __init__(self, user_api_key_cache):
            self.max_parallel_request_limiter = module._PROXY_MaxParallelRequestsHandler(
                frozen.cache(user_api_key_cache)
            )
            state["handler"] = self.max_parallel_request_limiter

        def _init_litellm_callbacks(self):
            module.litellm.callbacks = [self.max_parallel_request_limiter]

    class Router:
        def __init__(self, **kwargs):
            pass

        async def acompletion(self, **kwargs):
            assert kwargs["mock_response"] == "hello" and not kwargs.get("stream")
            response = Response()
            await state["handler"].async_log_success_event(
                {"litellm_params": {"metadata": kwargs["metadata"]}}, response, None, None
            )
            state["deliveries"] += 1
            return response

    tests = ModuleType("frozen_normal_call_tests")
    tests.__file__ = str(frozen.test_source)
    tests.__dict__.update(
        asyncio=asyncio,
        pytest=pytest,
        hash_token=lambda value: hashlib.sha256(value.encode()).hexdigest(),
        UserAPIKeyAuth=frozen.auth,
        DualCache=dict,
        ProxyLogging=Logging,
        Router=Router,
        litellm=module.litellm,
        datetime=datetime,
    )
    native_test = next(
        node
        for node in ast.parse(frozen.test_source.read_text()).body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "test_normal_router_call"
    )
    native_test.decorator_list = []
    exec(compile(ast.Module(body=[native_test], type_ignores=[]), str(frozen.test_source), "exec"), tests.__dict__)
    module.datetime = rollover_clock()
    tests.datetime = recorded_minute_clock()
    return SimpleNamespace(module=module, tests=tests, original_test=tests.test_normal_router_call, state=state)


def test_original_normal_assertion_fails_when_callback_uses_next_minute(frozen_normal):
    """Complete the actual callback, then prove the original lookup reads the old counter."""
    frozen = frozen_normal
    with pytest.raises(AssertionError):
        asyncio.run(frozen.original_test())
    assert frozen.state["deliveries"] == 1
    values = frozen.state["handler"].internal_usage_cache.values
    assert len(values) == 2
    old = next(value for key, value in values.items() if "2025-01-01-12-00" in key)
    new = next(value for key, value in values.items() if "2025-01-01-12-01" in key)
    assert old == {"current_requests": 1, "current_tpm": 0, "current_rpm": 1}
    assert new == {"current_requests": 0, "current_tpm": 15, "current_rpm": 0}


def test_normal_fixture_passes_original_assertions_and_restores_clocks(frozen_normal, tmp_path):
    """Keep both original assertions in one bucket and restore both prior clocks."""
    frozen = frozen_normal
    previous = (frozen.module.datetime, frozen.tests.datetime)
    audit = tmp_path / "audit"
    with pytest.MonkeyPatch.context() as scoped:
        scoped.setenv("SWE_VARIANT_AUDIT_DIR", str(audit))
        fixture = plugin().proposed_offline_inputs.__wrapped__(
            SimpleNamespace(node=SimpleNamespace(nodeid=NORMAL_TEST, module=frozen.tests)), scoped
        )
        next(fixture)
        try:
            asyncio.run(frozen.original_test())
            with pytest.raises(StopIteration):
                next(fixture)
        finally:
            fixture.close()
    assert (frozen.module.datetime, frozen.tests.datetime) == previous
    assert all(clock.calls == 0 for clock in previous)
    assert frozen.state["deliveries"] == 1
    values = list(frozen.state["handler"].internal_usage_cache.values.values())
    assert values == [{"current_requests": 0, "current_tpm": 15, "current_rpm": 1}]
    receipt = json.loads(next(audit.glob("*.json")).read_text())
    assert receipt["test"] == NORMAL_TEST
    assert [item["role"] for item in receipt["inputs"]].count("limiter") == 6
    assert [item["role"] for item in receipt["inputs"]].count("test") == 3


def test_normal_fixture_restores_clocks_after_failure(frozen_normal, tmp_path):
    """Restore native and test clocks when the surrounding test fails."""
    frozen = frozen_normal
    previous = (frozen.module.datetime, frozen.tests.datetime)
    with pytest.MonkeyPatch.context() as scoped:
        scoped.setenv("SWE_VARIANT_AUDIT_DIR", str(tmp_path / "audit"))
        fixture = plugin().proposed_offline_inputs.__wrapped__(
            SimpleNamespace(node=SimpleNamespace(nodeid=NORMAL_TEST, module=frozen.tests)), scoped
        )
        next(fixture)
        frozen.module.datetime.now()
        frozen.tests.datetime.now()
        with pytest.raises(RuntimeError, match="controlled test failure"):
            fixture.throw(RuntimeError("controlled test failure"))
    assert (frozen.module.datetime, frozen.tests.datetime) == previous
    assert len(json.loads(next((tmp_path / "audit").glob("*.json")).read_text())["inputs"]) == 2


@pytest.mark.parametrize("changed", ["module", "tests"])
def test_normal_fixture_rejects_source_drift(frozen_normal, monkeypatch, tmp_path, changed):
    """Reject native or original test drift before changing either clock."""
    frozen = frozen_normal
    path = tmp_path / "changed.py"
    path.write_text("changed")
    getattr(frozen, changed).__file__ = str(path)
    previous = (frozen.module.datetime, frozen.tests.datetime)
    fixture = plugin().proposed_offline_inputs.__wrapped__(
        SimpleNamespace(node=SimpleNamespace(nodeid=NORMAL_TEST, module=frozen.tests)), monkeypatch
    )
    with pytest.raises(ValueError, match="Frozen rate limiter"):
        next(fixture)
    assert (frozen.module.datetime, frozen.tests.datetime) == previous


def test_v4_changes_only_clock_packet_and_resume_identity(record_path, tmp_path):
    """Keep all v3 requirements and reject resuming its evaluator contract as v4."""
    from examples.common.artifacts import digest
    from examples.common.benchmark_runner import ensure_contract
    from examples.swebench_live.proposals.litellm_13675_v3 import prepare as v3
    from examples.swebench_live.proposals.litellm_13675_v4 import prepare as v4

    original = record_path.read_bytes()
    old_manifest = v3.prepare(record_path, tmp_path / "v3")
    manifest = v4.prepare(record_path, tmp_path / "v4")
    old = json.loads((tmp_path / "v3/proposed-record.json").read_text())
    new = json.loads((tmp_path / "v4/proposed-record.json").read_text())
    assert {key for key in old if old[key] != new[key]} == {"test_patch", "evaluation_variant"}
    assert len(new["FAIL_TO_PASS"]) == 6 and len(new["PASS_TO_PASS"]) == 2586
    assert record_path.read_bytes() == original
    assert manifest["status"] == "PROPOSED_NOT_ADOPTED" and not manifest["adoption_authorized"]
    assert manifest["previous_proposed_record_sha256"] == old_manifest["proposed_record_sha256"]
    assert plugin().SELECTED == plugin().previous.SELECTED | {NORMAL_TEST}
    assert len(plugin().SELECTED) == 15 and plugin().SELECTED.issubset(new["PASS_TO_PASS"])
    assert digest(old) != digest(new)
    directory = tmp_path / "resume"
    ensure_contract(
        directory, "contract.json", {"evaluation_variant": old["evaluation_variant"], "evaluator_sha256": digest(old)}
    )
    with pytest.raises(ValueError, match="use a new run directory"):
        ensure_contract(directory, "contract.json", manifest["evaluation_contract"])
    before = {path.name: path.read_bytes() for path in (tmp_path / "v4").iterdir()}
    with pytest.raises(FileExistsError):
        v4.prepare(record_path, tmp_path / "v4")
    assert {path.name: path.read_bytes() for path in (tmp_path / "v4").iterdir()} == before


def test_staged_plugin_loads_exact_v3_and_rejects_changed_dependency(record_path, tmp_path):
    """Exercise the generated dependency path outside the repository package layout."""
    from examples.swebench_live.proposals.litellm_13675_v3 import prepare as v3
    from examples.swebench_live.proposals.litellm_13675_v4 import prepare as v4

    v4.prepare(record_path, tmp_path / "prepared")
    task = tmp_path / "task"
    task.mkdir()
    subprocess.run(
        ["git", "apply", str(tmp_path / "prepared/proposed-test.patch")], cwd=task, check=True, capture_output=True
    )
    support = task / v4.PREVIOUS_FILE
    assert support.read_bytes() == (v3.PACKET / "offline_inputs.py").read_bytes()
    staged = task / v3.OFFLINE_FILE
    spec = importlib.util.spec_from_file_location("staged_v4", staged)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.SELECTED == plugin().SELECTED
    assert Path(module.previous.__file__) == support
    support.write_text("changed")
    with pytest.raises(ValueError, match="Pinned v3 offline input implementation changed"):
        module.load_previous()


@pytest.mark.parametrize("changed", ["offline_inputs.py", "identity.json"])
def test_v4_preparation_rejects_changed_sources_before_writing(record_path, monkeypatch, tmp_path, changed):
    """Reject changed pinned implementation inputs without publishing a proposal."""
    from examples.swebench_live.proposals.litellm_13675_v4 import prepare as v4

    copy = tmp_path / "copy"
    copy.mkdir()
    for name in ("offline_inputs.py", "identity.json"):
        (copy / name).write_bytes((v4.PACKET / name).read_bytes())
    if changed == "identity.json":
        data = json.loads((copy / changed).read_text())
        data["reused_v3_sha256"]["prepare.py"] = "changed"
        (copy / changed).write_text(json.dumps(data))
    else:
        (copy / changed).write_text("changed")
    monkeypatch.setattr(v4, "PACKET", copy)
    with pytest.raises(ValueError, match="Pinned evaluation variant source identity changed"):
        v4.prepare(record_path, tmp_path / "out")
    assert not (tmp_path / "out").exists()
