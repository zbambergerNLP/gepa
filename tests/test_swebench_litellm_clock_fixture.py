"""Exercise frozen limiter branches with explicit local cache and auth stubs."""

import ast
import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from examples.swebench_live.proposals.litellm_13675_v3 import offline_inputs

LIMITER_TEST = (
    "tests/test_litellm/proxy/hooks/test_parallel_request_limiter.py::test_pre_call_hook_rpm_limits_retry_after"
)
STREAMING_TEST = "tests/test_litellm/proxy/hooks/test_parallel_request_limiter.py::test_streaming_router_tpm_limit"
SOURCE_HASH = "d16fd09f3ebe3d062fb63804faf5398fe1ad30441cd1d18b5126e7c5ebd86553"
TEST_HASH = "c9036eaf895ed90183ae8ad809244d508c3e9492aeb9fc2fbc65f20ca22bb206"


class HTTPError(Exception):
    """Supply only FastAPI's error fields for the local frozen-body check."""

    def __init__(self, status_code, detail, headers=None):
        super().__init__(detail)
        self.status_code = status_code
        self.headers = headers


@pytest.fixture
def frozen_limiter(monkeypatch):
    """Bind unchanged frozen class and test bodies to labeled local dependency stubs."""
    folder = os.environ.get("SWE_LITELLM_CLOCK_SOURCE_DIR")
    if folder is None:
        pytest.skip("Set SWE_LITELLM_CLOCK_SOURCE_DIR to the Git-blob-verified source copies")
    source = Path(folder) / "parallel_request_limiter.py"
    test_source = Path(folder) / "test_parallel_request_limiter.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_HASH
    assert hashlib.sha256(test_source.read_bytes()).hexdigest() == TEST_HASH

    class Cache:
        def __init__(self, dual_cache):
            self.values = dual_cache

        async def async_batch_get_cache(self, keys, **kwargs):
            return [self.values.get(key) for key in keys]

        async def async_batch_set_cache(self, cache_list, **kwargs):
            self.values.update(cache_list)

        def get_cache(self, key, **kwargs):
            return self.values.get(key)

        async def async_get_cache(self, key, **kwargs):
            return self.get_cache(key)

    def auth(**kwargs):
        values = {"parent_otel_span": None, "user_id": None, "team_id": None, "end_user_id": None, "rpm_limit": None}
        return SimpleNamespace(**(values | kwargs))

    module = ModuleType("litellm.proxy.hooks.parallel_request_limiter")
    module.__file__ = str(source)
    module.__dict__.update(
        asyncio=asyncio,
        sys=sys,
        datetime=datetime,
        timedelta=timedelta,
        CustomLogger=object,
        HTTPException=HTTPError,
        CacheObject=dict,
        get_key_model_tpm_limit=lambda _: None,
        get_key_model_rpm_limit=lambda _: None,
        verbose_proxy_logger=SimpleNamespace(debug=lambda _: None, info=lambda _: None),
        litellm=SimpleNamespace(set_verbose=False),
        CommonProxyErrors=SimpleNamespace(max_parallel_request_limit_reached=SimpleNamespace(value="limit")),
    )
    native_class = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "_PROXY_MaxParallelRequestsHandler"
    )
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, native_class], type_ignores=[]))
    exec(compile(tree, str(source), "exec"), module.__dict__)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    native_test = next(
        node
        for node in ast.parse(test_source.read_text()).body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "test_pre_call_hook_rpm_limits_retry_after"
    )
    native_test.decorator_list = []
    context = {
        "asyncio": asyncio,
        "pytest": pytest,
        "hash_token": lambda value: hashlib.sha256(value.encode()).hexdigest(),
        "UserAPIKeyAuth": auth,
        "DualCache": dict,
        "InternalUsageCache": Cache,
        "MaxParallelRequestsHandler": module._PROXY_MaxParallelRequestsHandler,
    }
    exec(compile(ast.Module(body=[native_test], type_ignores=[]), str(test_source), "exec"), context)
    return SimpleNamespace(
        module=module, original_test=context[native_test.name], cache=Cache, auth=auth, test_source=test_source
    )


@pytest.fixture
def frozen_streaming_limiter(frozen_limiter, monkeypatch):
    """Run the frozen streaming test with a labeled router delivery stub and real callback body."""
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
            assert kwargs["mock_response"] == "hello" and kwargs["stream"] is True

            async def stream():
                yield "hello"
                await state["handler"].async_log_success_event(
                    {"litellm_params": {"metadata": kwargs["metadata"]}}, Response(), None, None
                )
                state["deliveries"] += 1

            return stream()

    tests = ModuleType("frozen_limiter_tests")
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
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "test_streaming_router_tpm_limit"
    )
    native_test.decorator_list = []
    exec(compile(ast.Module(body=[native_test], type_ignores=[]), str(frozen.test_source), "exec"), tests.__dict__)
    return SimpleNamespace(module=module, tests=tests, original_test=tests.test_streaming_router_tpm_limit, state=state)


def rollover_clock():
    """Put the first native hook before a minute boundary and the second after it."""

    class Clock:
        calls = 0

        @classmethod
        def now(cls):
            cls.calls += 1
            return datetime(2025, 1, 1, 12, 0, 59) if cls.calls <= 3 else datetime(2025, 1, 1, 12, 1, 1)

    return Clock


def test_original_retry_after_assertion_fails_across_minute(frozen_limiter):
    """Reproduce the original assertion failure with a controlled clock rollover."""
    clock = rollover_clock()
    frozen_limiter.module.datetime = clock
    with pytest.raises(pytest.fail.Exception, match="Expected call to fail"):
        asyncio.run(frozen_limiter.original_test())
    assert clock.calls == 6


@pytest.mark.parametrize("cross_boundary", [False, True])
def test_frozen_limiter_distinguishes_same_and_next_minute(frozen_limiter, cross_boundary):
    """Exercise the actual frozen branches without changing the limiter's policy."""

    class Clock:
        current = datetime(2025, 1, 1, 12, 0, 59)

        @classmethod
        def now(cls):
            return cls.current

    frozen_limiter.module.datetime = Clock

    async def exercise():
        cache = {}
        handler = frozen_limiter.module._PROXY_MaxParallelRequestsHandler(frozen_limiter.cache(cache))
        auth = frozen_limiter.auth(api_key="clock-control", max_parallel_requests=1, tpm_limit=9, rpm_limit=1)
        await handler.async_pre_call_hook(auth, cache, {}, "")
        if cross_boundary:
            Clock.current = datetime(2025, 1, 1, 12, 1, 1)
            await handler.async_pre_call_hook(auth, cache, {}, "")
        else:
            with pytest.raises(HTTPError) as error:
                await handler.async_pre_call_hook(auth, cache, {}, "")
            assert error.value.status_code == 429 and "retry-after" in error.value.headers

    asyncio.run(exercise())


def test_retry_after_fixture_pins_window_and_restores_clock(frozen_limiter, tmp_path):
    """Run every original assertion under the fixture, then restore the prior clock."""
    module = frozen_limiter.module
    clock = rollover_clock()
    module.datetime = clock
    audit = tmp_path / "audit"
    with pytest.MonkeyPatch.context() as scoped:
        scoped.setenv("SWE_VARIANT_AUDIT_DIR", str(audit))
        fixture = offline_inputs.proposed_offline_inputs.__wrapped__(
            SimpleNamespace(node=SimpleNamespace(nodeid=LIMITER_TEST)), scoped
        )
        next(fixture)
        try:
            asyncio.run(frozen_limiter.original_test())
            with pytest.raises(StopIteration):
                next(fixture)
        finally:
            fixture.close()
    assert module.datetime is clock and clock.calls == 0
    receipt = json.loads(next(audit.glob("*.json")).read_text())
    assert len(receipt["inputs"]) == 7
    assert all(
        item == {"kind": "deterministic-rate-limit-clock", "logical_time": "2025-01-01T12:00:10", "role": "limiter"}
        for item in receipt["inputs"]
    )


def test_clock_fixture_rejects_changed_native_source(frozen_limiter, monkeypatch, tmp_path):
    """Reject source drift before the fixture changes the clock."""
    changed = tmp_path / "changed.py"
    changed.write_text("changed")
    module = frozen_limiter.module
    module.__file__ = str(changed)
    original = module.datetime
    fixture = offline_inputs.proposed_offline_inputs.__wrapped__(
        SimpleNamespace(node=SimpleNamespace(nodeid=LIMITER_TEST)), monkeypatch
    )
    with pytest.raises(ValueError, match="Frozen rate limiter implementation changed"):
        next(fixture)
    assert module.datetime is original


def test_unrelated_test_keeps_native_clock(frozen_limiter, monkeypatch):
    """Keep all other test IDs outside the deterministic clock scope."""
    module = frozen_limiter.module
    original = module.datetime
    fixture = offline_inputs.proposed_offline_inputs.__wrapped__(
        SimpleNamespace(node=SimpleNamespace(nodeid=LIMITER_TEST + "-other")), monkeypatch
    )
    next(fixture)
    assert module.datetime is original
    with pytest.raises(StopIteration):
        next(fixture)


def recorded_minute_clock():
    """Keep the test's original cache lookup key in its first minute."""

    class Clock:
        calls = 0

        @classmethod
        def now(cls):
            cls.calls += 1
            return datetime(2025, 1, 1, 12, 0, 59)

    return Clock


def test_original_streaming_assertion_fails_when_callback_uses_next_minute(frozen_streaming_limiter):
    """Show the exact frozen callback updates a different bucket after rollover."""
    frozen = frozen_streaming_limiter
    frozen.module.datetime = rollover_clock()
    frozen.tests.datetime = recorded_minute_clock()
    with pytest.raises(AssertionError):
        asyncio.run(frozen.original_test())
    assert frozen.state["deliveries"] == 1
    values = frozen.state["handler"].internal_usage_cache.values
    assert len(values) == 2
    assert next(value for key, value in values.items() if "2025-01-01-12-00" in key)["current_tpm"] == 0
    assert next(value for key, value in values.items() if "2025-01-01-12-01" in key)["current_tpm"] == 15


def test_streaming_fixture_pins_both_clocks_and_restores_them(frozen_streaming_limiter, tmp_path):
    """Keep the original streaming assertions in one bucket and restore both module clocks."""
    frozen = frozen_streaming_limiter
    limiter_clock = rollover_clock()
    test_clock = recorded_minute_clock()
    frozen.module.datetime = limiter_clock
    frozen.tests.datetime = test_clock
    audit = tmp_path / "audit"
    with pytest.MonkeyPatch.context() as scoped:
        scoped.setenv("SWE_VARIANT_AUDIT_DIR", str(audit))
        fixture = offline_inputs.proposed_offline_inputs.__wrapped__(
            SimpleNamespace(node=SimpleNamespace(nodeid=STREAMING_TEST, module=frozen.tests)), scoped
        )
        next(fixture)
        try:
            asyncio.run(frozen.original_test())
            with pytest.raises(StopIteration):
                next(fixture)
        finally:
            fixture.close()
    assert frozen.module.datetime is limiter_clock and limiter_clock.calls == 0
    assert frozen.tests.datetime is test_clock and test_clock.calls == 0
    assert frozen.state["deliveries"] == 1
    receipt = json.loads(next(audit.glob("*.json")).read_text())
    assert [item["role"] for item in receipt["inputs"]].count("limiter") == 6
    assert [item["role"] for item in receipt["inputs"]].count("test") == 3
    assert all(item["logical_time"] == "2025-01-01T12:00:10" for item in receipt["inputs"])


def test_streaming_fixture_rejects_changed_test_source(frozen_streaming_limiter, monkeypatch, tmp_path):
    """Reject changed test source before mutating either module clock."""
    frozen = frozen_streaming_limiter
    changed = tmp_path / "changed-test.py"
    changed.write_text("changed")
    frozen.tests.__file__ = str(changed)
    original = (frozen.module.datetime, frozen.tests.datetime)
    fixture = offline_inputs.proposed_offline_inputs.__wrapped__(
        SimpleNamespace(node=SimpleNamespace(nodeid=STREAMING_TEST, module=frozen.tests)), monkeypatch
    )
    with pytest.raises(ValueError, match="Frozen rate limiter test module changed"):
        next(fixture)
    assert (frozen.module.datetime, frozen.tests.datetime) == original
