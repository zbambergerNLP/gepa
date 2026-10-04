"""Keep a full Jev run on the fixed FOREST contract and resident transport."""

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
from test_della_consolidation import executable
from test_wikipedia_react_v2_config import (
    DEEPSEEK_SCIENTIFIC_RUNTIME,
    QWEN_SCIENTIFIC_RUNTIME,
    _hotpot_args,
    _scientific_data_identity,
)

from examples.common.experiment_models import DEEPSEEK_V4_1_FLASH_MODEL
from examples.hotpotqa.legacy_main import TEACHER_RUNTIME_KEYS, build_run_contract

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("budget", [6871, 13742])
@pytest.mark.parametrize("changes", [{}, {"max_metric_calls": 10000}, {"condition": "vanilla"}, {"seed": 1}])
@pytest.mark.parametrize("module_selector", ["round_robin", "controller"])
def test_jev_full_run_keeps_scientific_guards(monkeypatch, budget, changes, module_selector):
    """Admit both approved budgets while rejecting unapproved budget, method and seed drift."""
    for name, value in QWEN_SCIENTIFIC_RUNTIME.items():
        monkeypatch.setenv(name, value)
    teacher = {
        "HOTPOTQA_" + key: DEEPSEEK_SCIENTIFIC_RUNTIME.get("HOTPOTQA_" + key, "fixture") for key in TEACHER_RUNTIME_KEYS
    }
    monkeypatch.setenv("HOTPOTQA_TEACHER_RUNTIME", json.dumps(teacher))
    args = _hotpot_args(
        **{
            "condition": "react_v2",
            "max_metric_calls": budget,
            "controller_selection": "jev",
            "module_selector": module_selector,
            "enforce_scientific_contract": True,
            "reflection_model": DEEPSEEK_V4_1_FLASH_MODEL,
            "reflection_api_base": "http://127.0.0.1:8201/v1",
            "train_limit": None,
            "val_limit": None,
            "test_limit": None,
            "data_identity": _scientific_data_identity(),
            **changes,
        }
    )
    if changes:
        with pytest.raises(ValueError):
            build_run_contract(args.condition, args)
    else:
        contract = build_run_contract("react_v2", args)
        assert contract["optimizer"]["semantic_controller_policy"]["model"] == "jev-1.13.0"
        assert contract["optimizer"]["budget_stopping"] == "whole_iteration_threshold"
        assert contract["data"]["splits"]["test"]["count"] == 300
        assert contract["optimizer"]["component_selector"] == module_selector


@pytest.mark.parametrize("profile,budget", [("standard", "6871"), ("expanded", "13742")])
@pytest.mark.parametrize("problem", [None, "coordinator", "pilot", "budget", "controller", "profile"])
def test_resident_entrypoint_checks_coordinator_before_models(tmp_path, profile, budget, problem):
    """Refuse invalid exports and stale coordinators before starting GPU processes."""
    source = tmp_path / "sources" / ("a" * 40)
    source.mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "calls"
    executable(bin_dir / "uv", 'printf "ready:%s\\n" "$*" >> "$CALLS"\nexit "$READY_STATUS"\n')
    executable(
        bin_dir / "bash",
        'printf "models:%s:%s:%s\\n" "$GEPA_JEV_HANDOFF_DIR" "$HOTPOTQA_CONTROLLER_SELECTION" "${TYPESAFE_API_KEY:-absent}" >> "$CALLS"\n',
    )
    settings = {
        "HOTPOTQA_PRODUCTION_LAUNCH": "1",
        "HOTPOTQA_PILOT_ONLY": "0",
        "MODEL_PROFILE": "deepseek-teacher-qwen-student",
        "CONDITION": "react_v2",
        "BUDGET_PROFILE": profile,
        "MAX_METRIC_CALLS": budget,
        "HOTPOTQA_CONTROLLER_SELECTION": "jev",
        "SCRATCH_BASE": str(tmp_path),
        "HOTPOTQA_SOURCE_COMMIT": "a" * 40,
        "GEPA_UV_BIN": str(bin_dir / "uv"),
        "GEPA_VENV_DIR": str(tmp_path / "venv"),
    }
    if problem in {"pilot", "budget", "controller", "profile"}:
        key, value = {
            "pilot": ("HOTPOTQA_PILOT_ONLY", "1"),
            "budget": ("MAX_METRIC_CALLS", "13742" if budget == "6871" else "6871"),
            "controller": ("HOTPOTQA_CONTROLLER_SELECTION", "verbalized"),
            "profile": ("BUDGET_PROFILE", "unapproved"),
        }[problem]
        settings[key] = value
    export = tmp_path / "prepared.env"
    export.write_bytes(b"".join(f"{key}={value}\0".encode() for key, value in settings.items()))
    result = subprocess.run(
        [
            "/bin/bash",
            str(ROOT / "scripts/della/remote/run_hotpotqa_resident.sh"),
            str(export),
            str(tmp_path / "mailbox"),
        ],
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "SLURM_JOB_ID": "123",
            "CALLS": str(capture),
            "READY_STATUS": "1" if problem == "coordinator" else "0",
            "TYPESAFE_API_KEY": "secret",
        },
        capture_output=True,
        text=True,
    )
    calls = capture.read_text().splitlines() if capture.exists() else []
    assert (result.returncode == 0) is (problem is None), result.stderr
    if problem is None:
        assert len(calls) == 2 and calls[0].startswith("ready:") and calls[1].endswith(":jev:absent")
    else:
        assert not any(line.startswith("models:") for line in calls)


@pytest.mark.parametrize("probe_status", [0, 1])
@pytest.mark.parametrize("controller", ["jev", "verbalized"])
def test_full_run_requires_the_exact_editor_canary(tmp_path, probe_status, controller):
    """Run the strict canary on a new full-run identity and stop on failure."""
    source = (ROOT / "scripts/della/remote/hotpotqa_workload.sh").read_text()
    start = source.index('if [[ "${REFLECTION_MODEL}" ==')
    block = source[start : source.index('echo "==> checking native tool-call compatibility"', start)]
    calls = tmp_path / "calls"
    python = tmp_path / "python"
    executable(python, 'printf "%s\\n" "$*" >> "$CALLS"\n[[ "$*" != *runtime_canary* ]] || exit "$PROBE_STATUS"\n')
    env = {
        **os.environ,
        **dict.fromkeys(re.findall(r"\$\{([A-Z][A-Z_0-9]*)", block), "fixture"),
        "PY": str(python),
        "CALLS": str(calls),
        "PROBE_STATUS": str(probe_status),
        "SCRATCH_BASE": str(tmp_path),
        "LOG_DIR": str(tmp_path),
        "REFLECTION_MODEL": DEEPSEEK_V4_1_FLASH_MODEL,
        "HOTPOTQA_CANARY_ONLY": "0",
        "HOTPOTQA_PILOT_ONLY": "0",
        "HOTPOTQA_CONTROLLER_SELECTION": controller,
    }
    result = subprocess.run(
        ["bash", "-c", "set -euo pipefail\nGEN_PID=$$\ngenerator_reports_expected_model() { return 0; }\n" + block],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == probe_status, result.stderr
    assert "runtime_canary" in calls.read_text()
    assert bool(list(tmp_path.rglob("*.ok"))) is (probe_status == 0)
