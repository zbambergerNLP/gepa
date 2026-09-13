"""Tests for the standalone local HotPotQA serving verification."""

import shlex
import sys
from pathlib import Path
from unittest.mock import Mock, call

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from examples.common.experiment_models import DEEPSEEK_V4_1_FLASH_MODEL
from examples.hotpotqa import verify_serving
from gepa.strategies.edit_tools import EDIT_TOOL_SETS

LOCAL_API_BASE = "http://127.0.0.1:8000/v1"


def test_diagnostic_package_inventory_python_executes(capsys) -> None:
    """Execute the embedded inventory snippet that runs before model startup."""
    script = Path(__file__).parents[1] / "scripts/della/verify_deepseek_serving.sh"
    line = next(line for line in script.read_text().splitlines() if "m.distributions()" in line)
    command = shlex.split(line.rstrip().removesuffix("\\"))
    snippet = command[command.index("-c") + 1]

    exec(compile(snippet, str(script), "exec"), {})

    packages = capsys.readouterr().out.splitlines()
    assert packages == sorted(packages)
    assert any(package.startswith("pytest==") for package in packages)


@pytest.mark.parametrize(
    "api_base",
    [
        "http://localhost:8000/v1",
        "http://127.0.0.1:8001/v1/",
        "http://[::1]:8002/v1",
    ],
)
def test_validate_loopback_api_base_accepts_explicit_local_v1_endpoints(api_base: str) -> None:
    """Accept HTTP loopback endpoints with an explicit port and v1 path.

    Args:
        api_base: Valid local endpoint under test.
    """
    verify_serving._validate_loopback_api_base(api_base)


@pytest.mark.parametrize(
    "api_base",
    [
        "https://127.0.0.1:8000/v1",
        "http://127.0.0.1/v1",
        "http://0.0.0.0:8000/v1",
        "http://example.com:8000/v1",
        "http://127.0.0.1:8000/chat/completions",
        "http://127.0.0.1:8000/v1?token=secret",
        "http://127.0.0.1:not-a-port/v1",
    ],
)
def test_validate_loopback_api_base_rejects_nonlocal_or_ambiguous_endpoints(api_base: str) -> None:
    """Reject endpoints that do not match the local serving contract.

    Args:
        api_base: Invalid endpoint under test.
    """
    with pytest.raises(verify_serving.ServingVerificationError, match="local HTTP loopback"):
        verify_serving._validate_loopback_api_base(api_base)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("   ", "empty text"),
        ("<think>private reasoning</think>answer", "leaked inline reasoning"),
        ("!" * 32, "repeated-character degeneration"),
        ("ab" * 32, "repeated-character degeneration"),
    ],
)
def test_require_healthy_text_rejects_degenerate_output(text: str, message: str) -> None:
    """Reject empty, leaked-reasoning, and low-diversity model output.

    Args:
        text: Unhealthy model text under test.
        message: Expected failure description.
    """
    with pytest.raises(verify_serving.ServingVerificationError, match=message):
        verify_serving._require_healthy_text(text, "Test probe")


def test_run_serving_verification_requires_one_attempt_per_tool_before_model_setup(monkeypatch) -> None:
    """Reject too few edit attempts before resolving or constructing the model client.

    Args:
        monkeypatch: Pytest fixture used to guard model setup calls.
    """
    resolve_kwargs = Mock()
    lm_factory = Mock()
    monkeypatch.setattr(verify_serving, "resolve_hotpotqa_lm_kwargs", resolve_kwargs)
    monkeypatch.setattr(verify_serving, "LM", lm_factory)

    with pytest.raises(verify_serving.ServingVerificationError, match="At least 4 edit attempts"):
        verify_serving.run_serving_verification(DEEPSEEK_V4_1_FLASH_MODEL, LOCAL_API_BASE, 3)

    resolve_kwargs.assert_not_called()
    lm_factory.assert_not_called()


def test_run_serving_verification_cycles_every_tool_and_reports_pass(monkeypatch) -> None:
    """Run every probe, cycle the edit attempts over all four tools, and report PASS.

    Args:
        monkeypatch: Pytest fixture used to isolate the orchestration contract.
    """
    lm = object()
    resolve_kwargs = Mock(return_value={"temperature": 1.0})
    lm_factory = Mock(return_value=lm)
    ordinary_probe = Mock()
    continuation_probe = Mock()
    edit_probe = Mock()
    monkeypatch.setattr(verify_serving, "resolve_hotpotqa_lm_kwargs", resolve_kwargs)
    monkeypatch.setattr(verify_serving, "LM", lm_factory)
    monkeypatch.setattr(verify_serving, "_ordinary_completion_probe", ordinary_probe)
    monkeypatch.setattr(verify_serving, "_tool_continuation_probe", continuation_probe)
    monkeypatch.setattr(verify_serving, "_edit_probe", edit_probe)

    report = verify_serving.run_serving_verification(DEEPSEEK_V4_1_FLASH_MODEL, LOCAL_API_BASE, 8)

    resolve_kwargs.assert_called_once_with(DEEPSEEK_V4_1_FLASH_MODEL, LOCAL_API_BASE)
    lm_factory.assert_called_once()
    assert lm_factory.call_args.args == (DEEPSEEK_V4_1_FLASH_MODEL,)
    assert lm_factory.call_args.kwargs == {
        "temperature": 1.0,
        "timeout": 600,
        "num_retries": 0,
        "max_retries": 0,
        "_gepa_provider_retry": {"log_path": None, "role": "serving_verification"},
    }
    ordinary_probe.assert_called_once_with(lm)
    continuation_probe.assert_called_once_with(lm)
    tools = EDIT_TOOL_SETS["broad"]
    assert edit_probe.call_args_list == [call(lm, tools[offset % len(tools)], offset + 1) for offset in range(8)]
    assert report["status"] == "PASS"
    assert report["failures"] == []
    assert report["tool_attempts"] == {tool.value: 2 for tool in sorted(tools, key=lambda item: item.value)}
    checks = report["checks"]
    assert isinstance(checks, dict)
    assert checks["ordinary_completion"] == "PASS"
    assert checks["tool_result_continuation"] == "PASS"
    assert sorted(name for name in checks if name.startswith("edit:")) == sorted(
        f"edit:{tool.value}#{repetition}" for tool in tools for repetition in (1, 2)
    )
    assert set(checks.values()) == {"PASS"}


def test_run_serving_verification_records_failures_and_keeps_checking(monkeypatch) -> None:
    """Report a failed continuation probe while still exercising every edit tool.

    Args:
        monkeypatch: Pytest fixture used to inject a runtime-probe failure.
    """
    lm = object()
    failure = verify_serving.ServingVerificationError("native tool continuation failed")
    ordinary_probe = Mock()
    continuation_probe = Mock(side_effect=failure)
    edit_probe = Mock(side_effect=[None, RuntimeError("provider timeout"), None, None])
    monkeypatch.setattr(verify_serving, "resolve_hotpotqa_lm_kwargs", Mock(return_value={}))
    monkeypatch.setattr(verify_serving, "LM", Mock(return_value=lm))
    monkeypatch.setattr(verify_serving, "_ordinary_completion_probe", ordinary_probe)
    monkeypatch.setattr(verify_serving, "_tool_continuation_probe", continuation_probe)
    monkeypatch.setattr(verify_serving, "_edit_probe", edit_probe)

    report = verify_serving.run_serving_verification(DEEPSEEK_V4_1_FLASH_MODEL, LOCAL_API_BASE, 4)

    tools = EDIT_TOOL_SETS["broad"]
    assert edit_probe.call_count == 4
    assert report["status"] == "FAIL"
    assert report["failures"] == sorted(["tool_result_continuation", f"edit:{tools[1].value}#1"])
    checks = report["checks"]
    assert isinstance(checks, dict)
    assert checks["ordinary_completion"] == "PASS"
    assert checks["tool_result_continuation"] == "FAIL: native tool continuation failed"
    assert checks[f"edit:{tools[1].value}#1"] == "FAIL: provider timeout"


@pytest.mark.parametrize(("status", "exit_code"), [("PASS", 0), ("FAIL", 1)])
def test_main_exits_nonzero_only_when_a_check_failed(monkeypatch, capsys, status: str, exit_code: int) -> None:
    """Print the report and map the overall status to the process exit code.

    Args:
        monkeypatch: Pytest fixture used to replace the verification run and argv.
        capsys: Pytest fixture capturing the printed report.
        status: Overall report status returned by the verification run.
        exit_code: Expected process exit status.
    """
    report = {
        "status": status,
        "model": DEEPSEEK_V4_1_FLASH_MODEL,
        "api_base": LOCAL_API_BASE,
        "attempts": 4,
        "tool_attempts": {},
        "checks": {"ordinary_completion": "PASS" if status == "PASS" else "FAIL: empty text"},
        "failures": [] if status == "PASS" else ["ordinary_completion"],
    }
    run = Mock(return_value=report)
    monkeypatch.setattr(verify_serving, "run_serving_verification", run)
    monkeypatch.setattr(sys, "argv", ["verify_serving", "--api-base", LOCAL_API_BASE])

    with pytest.raises(SystemExit) as exc_info:
        verify_serving.main()

    assert exc_info.value.code == exit_code
    run.assert_called_once_with(DEEPSEEK_V4_1_FLASH_MODEL, LOCAL_API_BASE, 4, 600, None)
    output = capsys.readouterr().out
    assert f"RESULT: {status}" in output
    assert ("PASS  ordinary_completion" if status == "PASS" else "FAIL  ordinary_completion") in output
