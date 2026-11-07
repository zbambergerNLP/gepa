"""Reuse v3 inputs and control the exact normal-router test's clock input."""

import hashlib
import importlib
import importlib.util
from datetime import datetime
from pathlib import Path

import pytest

PREVIOUS_SHA256 = "d35c75f93929bc9c7f48271274fe91381305208b74f3bd9639501c0f1f9646be"
NORMAL_LIMITER_TEST = "tests/test_litellm/proxy/hooks/test_parallel_request_limiter.py::test_normal_router_call"


def load_previous():
    """Load the identical v3 fixture from the staged test patch or repository."""
    path = Path(__file__).with_name("_proposed_v3_offline_inputs.py")
    if not path.exists():
        path = Path(__file__).parent.with_name("litellm_13675_v3") / "offline_inputs.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != PREVIOUS_SHA256:
        raise ValueError("Pinned v3 offline input implementation changed")
    spec = importlib.util.spec_from_file_location("_pinned_v3_offline_inputs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


previous = load_previous()
IMAGE_TESTS = previous.IMAGE_TESTS
IMAGE_URLS = previous.IMAGE_URLS
LIMITER_TEST = previous.LIMITER_TEST
STREAMING_LIMITER_TEST = previous.STREAMING_LIMITER_TEST
CLOCK_TESTS = previous.CLOCK_TESTS | {NORMAL_LIMITER_TEST}
SELECTED = previous.SELECTED | {NORMAL_LIMITER_TEST}


@pytest.fixture(autouse=True)
def proposed_offline_inputs(request, monkeypatch):
    """Delegate all v3 behavior and freeze both clocks for one additional test."""
    if request.node.nodeid != NORMAL_LIMITER_TEST:
        yield from previous.proposed_offline_inputs.__wrapped__(request, monkeypatch)
        return
    limiter = importlib.import_module("litellm.proxy.hooks.parallel_request_limiter")
    if hashlib.sha256(Path(limiter.__file__).read_bytes()).hexdigest() != previous.LIMITER_SOURCE_SHA256:
        raise ValueError("Frozen rate limiter implementation changed")
    if (
        hashlib.sha256(Path(request.node.module.__file__).read_bytes()).hexdigest()
        != previous.LIMITER_TEST_SOURCE_SHA256
    ):
        raise ValueError("Frozen rate limiter test module changed")
    calls = []

    class SameMinuteClock:
        role = "limiter"

        @classmethod
        def now(cls):
            calls.append(
                {"kind": "deterministic-rate-limit-clock", "logical_time": "2025-01-01T12:00:10", "role": cls.role}
            )
            return datetime(2025, 1, 1, 12, 0, 10)

    class TestClock(SameMinuteClock):
        role = "test"

    monkeypatch.setattr(limiter, "datetime", SameMinuteClock)
    monkeypatch.setattr(request.node.module, "datetime", TestClock)
    try:
        yield
    finally:
        previous.record_audit(request.node.nodeid, calls)
    assert {call["role"] for call in calls} == {"test", "limiter"}, "The declared clock inputs were not exercised"
