"""Verify diversity pilot dispatch locally without contacting Della or model APIs."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
WRAPPER = ROOT / "scripts/della/remote/run_hotpotqa_interactive.sh"
WORKLOAD = ROOT / "scripts/della/remote/hotpotqa_workload.sh"
START = 'if [[ "${HOTPOTQA_PILOT_ONLY}" == "1" && "${HOTPOTQA_PILOT_STAGE:-}" == "diversity-quality" ]]; then'


def _env():
    """Avoid inheriting an unrelated live job or a real API credential."""
    return {key: value for key, value in os.environ.items() if key not in {"SLURM_JOB_ID", "TYPESAFE_API_KEY"}}


def _prepared(tmp_path, pilot_only="1"):
    """Write a native NUL-separated export and a local stand-in for the batch entrypoint."""
    source = tmp_path / "sources" / ("a" * 40)
    batch = source / "examples/hotpotqa/run_hotpotqa.sbatch"
    batch.parent.mkdir(parents=True)
    batch.write_text('printf "%s\\n" "$HOTPOTQA_PILOT_STAGE" "$PWD" "$GEPA_VENV_DIR" "$HOME"\n')
    bootstrap = tmp_path / "bootstrap.sh"
    bootstrap.write_text('module() { [[ "$*" == "load proxy/default" ]]; }\n')
    uv = tmp_path / "uv"
    uv.write_text('#!/bin/bash\n[[ "$*" == *"examples.hotpotqa.typesafe_preflight"* ]]\n')
    uv.chmod(0o700)
    export_path = tmp_path / "pilot.env"
    export_path.write_bytes(
        b"\0".join(
            value.encode()
            for value in [
                f"HOTPOTQA_PILOT_ONLY={pilot_only}",
                f"SCRATCH_BASE={tmp_path}",
                f"HOTPOTQA_SOURCE_COMMIT={'a' * 40}",
                f"GEPA_VENV_DIR={tmp_path}/.venv-jev",
                f"GEPA_UV_BIN={uv}",
                f"HOTPOTQA_PILOT_ROOT={tmp_path}/pilot-output",
                "HOME=/must-not-replace-home",
            ]
        )
        + b"\0"
    )
    return source, export_path


@pytest.mark.parametrize(
    "stage", ["all", "preliminary", "smoke", "optimizer", "throughput", "full", "generalization", "diversity-quality"]
)
def test_wrapper_preserves_native_entrypoint_and_prepared_environment(tmp_path, stage):
    """Add the pilot stage while preserving all existing modes and the chosen environment."""
    source, export_path = _prepared(tmp_path)
    result = subprocess.run(
        ["bash", str(WRAPPER), str(export_path), stage],
        env={**_env(), "SLURM_JOB_ID": "test-allocation", "BASH_ENV": str(tmp_path / "bootstrap.sh")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [stage, str(source), str(tmp_path / ".venv-jev"), os.environ["HOME"]]


@pytest.mark.parametrize("case", ["outside_slurm", "not_pilot", "unknown_stage"])
def test_wrapper_requires_allocation_prepared_pilot_and_known_stage(tmp_path, case):
    """Reject invalid entrypoints before the batch script can execute."""
    _, export_path = _prepared(tmp_path, pilot_only="0" if case == "not_pilot" else "1")
    env = _env()
    if case != "outside_slurm":
        env["SLURM_JOB_ID"] = "test-allocation"
    stage = "not-supported" if case == "unknown_stage" else "diversity-quality"
    result = subprocess.run(["bash", str(WRAPPER), str(export_path), stage], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and not result.stdout


def _dispatch_fixture(tmp_path, source, *, secret="test-only-credential", failure=None):
    """Capture module arguments and credential presence without writing the credential to logs."""
    block = source[source.index(START) : source.index('\nif [[ "${HOTPOTQA_PILOT_ONLY}" == "1" ]]; then')]
    python = tmp_path / "fake-python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "with Path(os.environ['CALLS']).open('a') as stream:\n"
        "    stream.write(json.dumps({'args': sys.argv[1:], 'key_loaded': os.environ.get('TYPESAFE_API_KEY') "
        "== 'test-only-credential'}) + '\\n')\n"
        f"raise SystemExit(7 if {failure!r} in sys.argv[1:] else 0)\n"
    )
    python.chmod(0o700)
    if secret is not None:
        key = tmp_path / ".secrets/typesafe-api-key"
        key.parent.mkdir(mode=0o700)
        key.write_text(secret)
        key.chmod(0o600)
    env = {
        **_env(),
        **dict.fromkeys(re.findall(r"\$\{([A-Z][A-Z_0-9]*)", source), "fixture"),
        "PY": str(python),
        "CALLS": str(tmp_path / "calls.jsonl"),
        "SCRATCH_BASE": str(tmp_path),
        "HOTPOTQA_PILOT_ROOT": str(tmp_path / "pilot outputs"),
        "LOG_DIR": str(tmp_path),
        "HOTPOTQA_PILOT_ONLY": "1",
        "HOTPOTQA_PILOT_STAGE": "diversity-quality",
        "HOTPOTQA_CANARY_ONLY": "0",
        "HOTPOTQA_EDITOR_MODE": "single_call",
        "GEPA_JEV_HANDOFF_DIR": "",
    }
    return block, env


def test_dispatch_loads_secret_only_for_new_mode_and_preserves_model_arguments(tmp_path):
    """Run the separate pilot with prepared endpoints and an isolated output directory."""
    block, env = _dispatch_fixture(tmp_path, WORKLOAD.read_text())
    result = subprocess.run(["bash", "-c", "set -euo pipefail\n" + block], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "test-only-credential" not in result.stdout + result.stderr
    rows = [json.loads(line) for line in Path(env["CALLS"]).read_text().splitlines()]
    assert rows == [
        {
            "args": [
                "-m",
                "examples.hotpotqa.diversity_quality_pilot",
                "--model",
                "fixture",
                "--api-base",
                "fixture",
                "--reflection-model",
                "fixture",
                "--reflection-api-base",
                "fixture",
                "--wiki17-dir",
                "fixture",
                "--workers",
                "fixture",
                "--output-dir",
                str(tmp_path / "pilot outputs/diversity-quality"),
            ],
            "key_loaded": True,
        }
    ]


@pytest.mark.parametrize("secret", [None, ""])
def test_dispatch_missing_or_empty_secret_fails_before_pilot(tmp_path, secret):
    """Do not mask credential read failures through the shell export builtin."""
    block, env = _dispatch_fixture(tmp_path, WORKLOAD.read_text(), secret=secret)
    result = subprocess.run(["bash", "-c", "set -euo pipefail\n" + block], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert not Path(env["CALLS"]).exists()


def test_offline_dispatch_does_not_require_compute_node_credentials(tmp_path):
    block, env = _dispatch_fixture(tmp_path, WORKLOAD.read_text(), secret=None)
    env["GEPA_JEV_HANDOFF_DIR"] = str(tmp_path / "handoff")
    result = subprocess.run(["bash", "-c", "set -euo pipefail\n" + block], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in Path(env["CALLS"]).read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["key_loaded"] is False


@pytest.mark.parametrize("pilot,stage", [("0", "diversity-quality"), ("1", "smoke")])
def test_other_modes_do_not_read_secret_or_dispatch_new_pilot(tmp_path, pilot, stage):
    """Keep production and existing qualification modes on their original paths."""
    block, env = _dispatch_fixture(tmp_path, WORKLOAD.read_text(), secret=None)
    env.update(HOTPOTQA_PILOT_ONLY=pilot, HOTPOTQA_PILOT_STAGE=stage)
    result = subprocess.run(["bash", "-c", "set -euo pipefail\n" + block], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not Path(env["CALLS"]).exists()


@pytest.mark.parametrize("failure", ["examples.hotpotqa.verify_thinking_budget", "-"])
def test_existing_boundary_and_native_tool_probes_still_gate_dispatch(tmp_path, failure):
    """Fail the unchanged readiness path and prove no credential-backed pilot starts."""
    source = WORKLOAD.read_text()
    _, env = _dispatch_fixture(tmp_path, source, failure=failure)
    block = source[
        source.index("THINKING_PROBE_PARENT=") : source.index('\nif [[ "${HOTPOTQA_PILOT_ONLY}" == "1" ]]; then')
    ]
    result = subprocess.run(["bash", "-c", "set -euo pipefail\n" + block], env=env, capture_output=True, text=True)
    assert result.returncode == 7, result.stderr
    rows = [json.loads(line) for line in Path(env["CALLS"]).read_text().splitlines()]
    assert not any("examples.hotpotqa.diversity_quality_pilot" in row["args"] for row in rows)
    assert not any(row["key_loaded"] for row in rows)
