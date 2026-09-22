"""Exercise the consolidated laptop launchers without contacting Della."""

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]


def executable(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\nset -eu\n" + body)
    path.chmod(0o700)


@pytest.mark.parametrize(
    "script", ["examples/hotpotqa/run_hotpotqa.sbatch", "scripts/della/verify_deepseek_serving.sh"]
)
@pytest.mark.parametrize("language", ["c", "c++"])
def test_cuda_headers_match_nvcc_with_wheel_fallback(tmp_path, script, language):
    """Resolve compiler runtime headers before wheels while retaining missing cuBLAS headers."""
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("A C preprocessor is required")
    cuda = tmp_path / "cuda"
    wheel = tmp_path / "site" / "nvidia" / "cu13"
    for root in (cuda, wheel):
        (root / "include").mkdir(parents=True)
    (cuda / "include" / "cuda_runtime.h").write_text("compiler_runtime\n")
    (wheel / "include" / "cuda_runtime.h").write_text("incompatible_wheel_runtime\n")
    (wheel / "include" / "cublas.h").write_text("wheel_cublas\n")
    python = tmp_path / "python"
    executable(python, f"printf '%s\\n' {shlex.quote(str(tmp_path / 'site'))}\n")
    source = (ROOT / script).read_text()
    start = source.index("SERVING_CUDA_ROOT=")
    block = source[start : source.index("\nfi\n", start) + 4]
    result = subprocess.run(
        [
            "bash",
            "-c",
            block + f"\n{shlex.quote(compiler)} -isystem {shlex.quote(str(cuda / 'include'))} -E -P -x {language} -",
        ],
        input="#include <cuda_runtime.h>\n#include <cublas.h>\n",
        env={**os.environ, "VLLM_PY": str(python)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["compiler_runtime", "wheel_cublas"]


@pytest.mark.parametrize("case", ["current", "stale_commit", "other_branch", "dirty"])
def test_preflight_uses_clean_consolidated_head_before_any_ssh(tmp_path, case):
    """Default to HEAD and reject source drift before contacting either host."""
    script_dir = tmp_path / "scripts" / "della"
    script_dir.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/della/preflight_hotpotqa.sh", script_dir)
    config = script_dir / ".env"
    config.write_text(
        "REMOTE_USER=testuser\nREMOTE_HOST=login.example\nREMOTE_VIS_HOST=vis.example\n"
        "REMOTE_DIR=/scratch/test/gepa\nSCRATCH_BASE=/scratch/test/gepa\n"
        "MODEL_STORAGE=/projects/test/models\nGPU_PARTITION=ailab\n"
    )
    config.chmod(0o600)
    serving = tmp_path / "examples" / "hotpotqa" / "serving"
    serving.mkdir(parents=True)
    for name in ("requirements-x86_64-linux-py312.txt", "requirements-deepseek-v4.1-flash-x86_64-linux-py312.txt"):
        (serving / name).write_text("fixture\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable(
        bin_dir / "git",
        """case "$*" in
        *--show-current*) printf '%s\\n' "$TEST_BRANCH" ;;
        *rev-parse*) printf '%s\\n' "$TEST_COMMIT" ;;
        *status*) printf '%s' "$TEST_DIRTY" ;;
        *) exit 1 ;;
    esac
    """,
    )
    executable(bin_dir / "ssh", 'printf "%s\\n" "$*" >> "$SSH_CAPTURE"\ncat >/dev/null\n')
    for command in ("rsync", "sha256sum", "uv"):
        executable(bin_dir / command, 'printf "%064d\\n" 1\n')
    capture = tmp_path / "ssh-calls"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "TEST_COMMIT": "a" * 40,
        "TEST_BRANCH": "other" if case == "other_branch" else "main",
        "TEST_DIRTY": " M file" if case == "dirty" else "",
        "SSH_CAPTURE": str(capture),
    }
    env.pop("HOTPOTQA_SOURCE_COMMIT", None)
    if case == "stale_commit":
        env["HOTPOTQA_SOURCE_COMMIT"] = "b" * 40
    result = subprocess.run(
        ["bash", str(script_dir / "preflight_hotpotqa.sh")], env=env, input="", capture_output=True, text=True
    )
    if case == "current":
        assert result.returncode == 0, result.stderr
        assert "a" * 40 in result.stdout
        assert len(capture.read_text().splitlines()) == 3
    else:
        assert result.returncode != 0
        assert not capture.exists()


@pytest.mark.parametrize("verification_status", [0, 1])
def test_existing_shared_model_is_verified_without_preparing_it(tmp_path, verification_status):
    """Reuse Zach's pinned bytes read-only and never repair a failed shared checkpoint silently."""
    model = tmp_path / "models" / "Qwen3.8-27B"
    model.mkdir(parents=True)
    (model / ".gepa-model-integrity.json").write_text("{}")
    python = tmp_path / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    calls = tmp_path / "calls"
    executable(python, f'printf "%s\\n" "$*" >> "${{CALLS}}"\nexit {verification_status}\n')
    source = (ROOT / "scripts/della/remote/download_model.sh").read_text()
    # Linux-only flock setup is unchanged; exercise the new read-only branch on macOS too.
    block = source[source.index('echo "==> ${MODEL} into') :]
    result = subprocess.run(
        ["bash", "-c", "set -eu\n" + block],
        cwd=tmp_path,
        env={**os.environ, "CALLS": str(calls), "MODEL": "qwen3.8-27b", "MODEL_DIR": str(model)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == verification_status
    assert len(calls.read_text().splitlines()) == 1
    assert "model_snapshot verify" in calls.read_text() and "prepare" not in calls.read_text()


@pytest.mark.parametrize(
    "profile,workers", [("qwen3.8-27b", 12), ("deepseek-v4.1-flash", 4), ("deepseek-teacher-qwen-student", 12)]
)
@pytest.mark.parametrize("kind", ["experiment", "pilot", "prepare"])
@pytest.mark.parametrize("invalid_setting", [None, "DELLA_GPUS", "DELLA_CPUS_PER_TASK", "VLLM_TENSOR_PARALLEL_SIZE"])
def test_submit_expands_the_remote_script_without_running_jobs(tmp_path, profile, workers, kind, invalid_setting):
    """Catch laptop-side heredoc expansion errors before any real submission."""
    script_dir = tmp_path / "scripts" / "della"
    script_dir.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/della/submit_hotpotqa.sh", script_dir)
    config = script_dir / ".env"
    config.write_text(
        "REMOTE_USER=testuser\nREMOTE_HOST=login.example\nREMOTE_VIS_HOST=vis.example\n"
        "REMOTE_DIR=/scratch/test/gepa\nSCRATCH_BASE=/scratch/test/gepa\n"
        "MODEL_STORAGE=/projects/test/models\nGPU_PARTITION=ailab\n"
    )
    config.chmod(0o600)
    executable(script_dir / "sync_to_della.sh", 'printf "%064d\\n" 1 > "${SYNC_MANIFEST_OUTPUT}"\n')
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable(bin_dir / "git", '[[ "$*" == *status* ]] || printf "%040d\\n" 2\n')
    executable(bin_dir / "ssh", 'cat > "${CAPTURE_REMOTE_SCRIPT}"\n')
    capture = tmp_path / "remote.sh"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "MODEL_PROFILE": profile,
        "HOTPOTQA_JOB_KIND": "pilot" if kind == "prepare" else kind,
        "HOTPOTQA_PREPARE_ONLY": str(int(kind == "prepare")),
        "CAPTURE_REMOTE_SCRIPT": str(capture),
        "HOTPOTQA_CAMPAIGN_ID": "integration-test",
        "HOTPOTQA_TEXT_LIMITS_JSON": '{"component_chars":12345}',
    }
    for key in (
        "MAX_WORKERS",
        "HOTPOTQA_LOG_DIR",
        "BUDGET_PROFILE",
        "CONDITION",
        "DELLA_GPUS",
        "DELLA_CPUS_PER_TASK",
        "DELLA_MEMORY",
        "VLLM_TENSOR_PARALLEL_SIZE",
        "VLLM_DATA_PARALLEL_SIZE",
        "VLLM_API_SERVER_COUNT",
        "VLLM_MAX_NUM_SEQS",
    ):
        env.pop(key, None)
    if invalid_setting:
        env[invalid_setting] = "64"
    result = subprocess.run(["bash", str(script_dir / "submit_hotpotqa.sh")], env=env, capture_output=True, text=True)
    if invalid_setting:
        assert result.returncode != 0
        assert "ERROR:" in result.stderr
        assert not capture.exists()
        return
    assert result.returncode == 0, result.stderr
    remote = capture.read_text()
    assert 'local run_condition="$3"' in remote
    assert 'local canary_only="$4"' in remote
    assert f'"MAX_WORKERS={workers}"' in remote
    gpus, cpus, memory, tp = (1, 8, "128G", 1) if profile == "qwen3.8-27b" else (4, 32, "768G", 4)
    if profile == "deepseek-teacher-qwen-student":
        gpus, cpus, memory, tp = 5, 40, "896G", 1
    for resource in (f"--gres=gpu:h200:{gpus}", f"--cpus-per-task={cpus}", f"--mem={memory}"):
        assert resource in remote
    for setting in (f"VLLM_TENSOR_PARALLEL_SIZE={tp}", "VLLM_DATA_PARALLEL_SIZE=1", "VLLM_API_SERVER_COUNT=1"):
        assert f'"{setting}"' in remote
    assert f'"HOTPOTQA_PILOT_ONLY={int(kind != "experiment")}"' in remote
    assert "examples.common.slurm_continuation add" in remote
    assert "examples.common.slurm_continuation start" in remote
    assert '"HOTPOTQA_TEXT_LIMITS_JSON=${HOTPOTQA_TEXT_LIMITS_JSON}"' in remote
    suffix = "-deepseek-v4.1-flash" if profile == "deepseek-v4.1-flash" else ""
    assert f'"SERVING_VENV_DIR=/scratch/test/gepa/.serving-venv{suffix}"' in remote
    assert "POSIT" not in remote
    assert "logs/hotpotqa/integration-test/" in remote
    subprocess.run(["bash", "-n", str(capture)], check=True, capture_output=True)
    if kind == "prepare":
        block = remote[remote.index("write_sbatch_export_file()") : remote.index('for _ in "${SUBMIT_CONDITIONS[@]}"')]
        prepared = subprocess.run(
            ["bash", "-c", "set -euo pipefail\n" + block],
            env={
                **env,
                **dict.fromkeys(re.findall(r"\$\{([A-Z][A-Z_0-9]*)", block), "fixture"),
                "CONTINUATION_DIR": str(tmp_path),
                "HOME": os.environ["HOME"],
                "PATH": os.environ["PATH"],
            },
            capture_output=True,
            text=True,
        )
        assert prepared.returncode == 0, prepared.stderr
        export_path = Path(prepared.stdout.strip().removeprefix("INTERACTIVE_EXPORT_FILE="))
        entries = export_path.read_bytes().split(b"\0")
        assert b"HOTPOTQA_PILOT_ONLY=1" in entries
        assert f"VLLM_TENSOR_PARALLEL_SIZE={tp}".encode() in entries
        assert not (tmp_path / "plan.json").exists()


@pytest.mark.parametrize("probe_status", [0, 1])
def test_qwen_tool_verification_gates_pilot_and_resumes(tmp_path, probe_status):
    """Start a pilot only after all tool probes pass, reusing only its exact attestation."""
    source = (ROOT / "examples/hotpotqa/run_hotpotqa.sbatch").read_text() + (
        ROOT / "scripts/della/remote/hotpotqa_workload.sh"
    ).read_text()
    start = source.index('if [[ "${HOTPOTQA_PILOT_ONLY}" == "1" ]]; then\n')
    block = source[start : source.index("CAMPAIGN_LOCK_DIR=", start)]
    calls = tmp_path / "calls"
    python = tmp_path / "python"
    executable(
        python,
        f'printf "%s\\n" "$*" >> "${{CALLS}}"\nif [[ "$*" == *verify_serving* ]]; then exit {probe_status}; fi\n',
    )
    env = {
        **os.environ,
        **dict.fromkeys(re.findall(r"\$\{([A-Z][A-Z_0-9]*)", block), "fixture"),
        "PY": str(python),
        "CALLS": str(calls),
        "SCRATCH_BASE": str(tmp_path),
        "LOG_DIR": str(tmp_path),
        "MODEL_PROFILE": "qwen3.8-27b",
        "HOTPOTQA_PILOT_ONLY": "1",
    }
    command = [
        "bash",
        "-c",
        "set -euo pipefail\nGEN_PID=$$\ngenerator_reports_expected_model() { return 0; }\n" + block,
    ]
    first = subprocess.run(command, env=env, capture_output=True, text=True)
    assert first.returncode == probe_status, first.stderr
    assert ("examples.hotpotqa.pilot" in calls.read_text()) is (probe_status == 0)
    if probe_status == 0:
        subprocess.run(command, env=env, check=True, capture_output=True)
        assert calls.read_text().count("examples.hotpotqa.verify_serving") == 1
        assert calls.read_text().count("examples.hotpotqa.pilot") == 2


@pytest.mark.parametrize(
    "profile,canary_only,probe_status,qualified",
    [
        ("deepseek-v4.1-flash", "1", 1, False),
        ("deepseek-v4.1-flash", "1", 0, True),
        ("qwen3.8-27b", "0", 1, False),
        ("qwen3.8-27b", "0", 0, True),
    ],
)
def test_failed_probes_cannot_freeze_campaign(tmp_path, profile, canary_only, probe_status, qualified):
    """Execute the actual gate/lock block with a controlled local model process."""
    source = (ROOT / "examples/hotpotqa/run_hotpotqa.sbatch").read_text() + (
        ROOT / "scripts/della/remote/hotpotqa_workload.sh"
    ).read_text()
    block = source[source.index("CAMPAIGN_IDENTITY_SHA256=") : source.index('echo "==> running GEPA experiment:')]
    env = {**os.environ, **dict.fromkeys(re.findall(r"\$\{([A-Z][A-Z_0-9]*)", block), "fixture")}
    fake_python = tmp_path / "fake-python"
    executable(fake_python, f"exit {probe_status}\n")
    env.update(
        SCRATCH_BASE=str(tmp_path),
        MODEL_PROFILE=profile,
        REFLECTION_MODEL="hosted_vllm/deepseek-ai/DeepSeek-V4.1-Flash"
        if profile == "deepseek-v4.1-flash"
        else "fixture",
        HOTPOTQA_CANARY_ONLY=canary_only,
        HOTPOTQA_CAMPAIGN_ID="fixture",
        PY=str(fake_python),
        LOG_DIR=str(tmp_path),
    )
    result = subprocess.run(
        ["bash", "-c", "set -euo pipefail\nGEN_PID=$$\ngenerator_reports_expected_model() { return 0; }\n" + block],
        env=env,
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) is qualified, result.stderr
    locks = list((tmp_path / ".cache/gepa/hotpotqa-campaign").rglob("*.sha256"))
    markers = list((tmp_path / ".cache/gepa/hotpotqa-canaries").rglob("*.ok"))
    if canary_only == "1":
        assert not locks
        assert bool(markers) is qualified
    else:
        assert len(locks) == (2 if qualified else 0)


def test_deepseek_smoke_uses_the_campaign_serving_arguments():
    """Compare parsed invocations so a diagnostic cannot qualify different flags."""
    campaign = (ROOT / "examples/hotpotqa/run_hotpotqa.sbatch").read_text()
    smoke = (ROOT / "scripts/della/verify_deepseek_serving.sh").read_text()

    def arguments(script):
        start = script.index('echo "==> serving DeepSeek') if 'echo "==> serving DeepSeek' in script else 0
        start = script.index('"${VLLM_BIN}" serve "${SOLVER_MODEL_PATH}"', start)
        end = script.index('> "${GEN_LOG}"', start)
        tokens = shlex.split(script[start:end].replace("\\\n", " "))
        flags = {}
        for token in tokens[3:]:
            if token.startswith("--"):
                assert token not in flags
                key = token
                flags[key] = True
            else:
                flags[key] = token
        return flags

    assert arguments(campaign) == arguments(smoke)
    assert arguments(campaign)["--tensor-parallel-size"] == "4"
    assert json.loads(arguments(campaign)["--engram-config"]) == {"cpu_offload": True}
    for setting in ("GEN_MAX_LEN=262144", "GEN_GMU=0.92"):
        assert setting in campaign and setting in smoke


@pytest.mark.parametrize("expected,visible", [(1, 1), (4, 4), (1, 8), (4, 2)])
def test_gpu_inventory_uses_only_allocated_devices(tmp_path, monkeypatch, capsys, expected, visible):
    """Reject mismatched allocations and keep physical UUIDs out of the resumable identity."""
    script = (ROOT / "examples/hotpotqa/run_hotpotqa.sbatch").read_text()
    start = script.index('HOTPOTQA_GPU_RUNTIME="$(')
    start = script.index("<<'PY'\n", start) + len("<<'PY'\n")
    block = script[start : script.index("\nPY\n", start)]
    inventory = tmp_path / "inventory.json"
    ids = tmp_path / "ids.txt"
    monkeypatch.setattr(sys, "argv", ["inventory", str(expected), str(inventory), str(ids)])
    cuda = SimpleNamespace(
        device_count=lambda: visible,
        get_device_name=lambda _: "NVIDIA H200",
        get_device_capability=lambda _: (9, 0),
        get_device_properties=lambda index: SimpleNamespace(uuid=f"allocated-{index}", total_memory=140_000_000_000),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))

    def driver_query(command, **kwargs):
        assert command[command.index("--id") + 1] == ",".join(f"GPU-allocated-{i}" for i in range(visible))
        return SimpleNamespace(stdout="590.00\n" * visible)

    monkeypatch.setattr(subprocess, "run", driver_query)
    if visible != expected:
        with pytest.raises(SystemExit, match=f"exactly {expected} visible H200 GPUs"):
            exec(compile(block, "gpu_inventory", "exec"), {})
        assert not inventory.exists() and not ids.exists()
        return
    exec(compile(block, "gpu_inventory", "exec"), {})
    runtime = json.loads(capsys.readouterr().out)
    assert runtime["count"] == expected
    assert "allocated-" not in json.dumps(runtime)
    devices = json.loads(inventory.read_text())["devices"]
    assert len(devices) == expected
    assert devices[0]["memory_total_bytes"] == 140_000_000_000
    assert ids.read_text().strip() == ",".join(device["uuid"] for device in devices)
    assert script.index("GPU_MONITOR_PID=$!") < script.index('"${VLLM_BIN}" serve "${SOLVER_MODEL_PATH}"')


@pytest.mark.parametrize("reply,success", [("31415;cluster", True), ("submission failed", False)])
def test_smoke_submission_forwards_custom_paths(tmp_path, reply, success):
    """Expand the real SSH command and preserve paths containing spaces."""
    script_dir = tmp_path / "scripts" / "della"
    script_dir.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/della/submit_deepseek_smoke.sh", script_dir)
    config = script_dir / ".env"
    config.write_text(
        "REMOTE_USER=testuser\nREMOTE_HOST=login.example\nREMOTE_DIR='/scratch/source checkout'\n"
        "SCRATCH_BASE='/scratch/custom cache'\nMODEL_STORAGE='/projects/custom models'\n"
    )
    config.chmod(0o600)
    executable(script_dir / "sync_to_della.sh", "exit 0\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable(bin_dir / "ssh", 'printf "%s\\n" "$@" > "${CAPTURE_COMMAND}"\nprintf "%s\\n" "${SUBMIT_REPLY}"\n')
    capture = tmp_path / "command.txt"
    result = subprocess.run(
        ["bash", str(script_dir / "submit_deepseek_smoke.sh"), "submit"],
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "CAPTURE_COMMAND": str(capture),
            "SUBMIT_REPLY": reply,
        },
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) is success, result.stderr
    command = shlex.split(capture.read_text().splitlines()[-1])
    assert "SCRATCH_BASE=/scratch/custom cache" in command
    assert "MODEL_STORAGE=/projects/custom models" in command
    assert "GEPA_VENV_DIR=/scratch/source checkout/.venv" in command
    assert "SERVING_VENV_DIR=/scratch/source checkout/.serving-venv-deepseek-v4.1-flash" in command
    assert "--output=/scratch/custom cache/logs/hotpotqa/verify/smoke-%j.out" in command
    if success:
        assert "job 31415" in result.stdout
    else:
        assert "invalid job id" in result.stderr


@pytest.mark.parametrize(
    "path", sorted((ROOT / "scripts/della").rglob("*.sh")) + [ROOT / "scripts/della/smoke_deepseek_serving.sbatch"]
)
def test_della_scripts_parse(path):
    subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True)


@pytest.mark.parametrize(
    "enabled,budget,condition,pilot,expected",
    [
        ("1", "standard", "vanilla", "0", True),
        ("0", "standard", "vanilla", "0", False),
        ("1", "expanded", "vanilla", "0", False),
        ("1", "standard", "react_v2", "0", False),
        ("1", "standard", "vanilla", "1", False),
    ],
)
def test_initial_throughput_precedes_only_first_production_cell(tmp_path, enabled, budget, condition, pilot, expected):
    """Measure training throughput in the existing allocation without running full150."""
    source = (ROOT / "examples/hotpotqa/run_hotpotqa.sbatch").read_text() + (
        ROOT / "scripts/della/remote/hotpotqa_workload.sh"
    ).read_text()
    start = source.index('if [[ "${HOTPOTQA_INITIAL_THROUGHPUT:-0}"')
    block = source[start : source.index('\nif [[ "${HOTPOTQA_PILOT_ONLY}" == "1" ]]; then', start)]
    calls = tmp_path / "calls"
    python = tmp_path / "python"
    executable(python, 'printf "%s\\n" "$*" >> "$CALLS"\nexit "${PROBE_STATUS:-0}"\n')
    env = {
        **os.environ,
        **dict.fromkeys(re.findall(r"\$\{([A-Z][A-Z_0-9]*)", block), "fixture"),
        "PY": str(python),
        "CALLS": str(calls),
        "HOTPOTQA_INITIAL_THROUGHPUT": enabled,
        "BUDGET_PROFILE": budget,
        "CONDITION": condition,
        "HOTPOTQA_PILOT_ONLY": pilot,
    }
    result = subprocess.run(["bash", "-c", "set -euo pipefail\n" + block], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert calls.exists() is expected
    if expected:
        assert "--stage throughput" in calls.read_text()
        failed = subprocess.run(
            ["bash", "-c", "set -euo pipefail\n" + block], env={**env, "PROBE_STATUS": "7"}, capture_output=True
        )
        assert failed.returncode == 7
