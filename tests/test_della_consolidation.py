"""Exercise the consolidated laptop launchers without contacting Della."""

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def executable(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\nset -eu\n" + body)
    path.chmod(0o700)


@pytest.mark.parametrize("script", ["scripts/della/verify_deepseek_serving.sh"])
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
