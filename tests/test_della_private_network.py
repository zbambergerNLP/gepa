"""Check the private-network launcher without changing the host network."""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/della/remote/with_private_network.sh"


def executable(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\nset -eu\n" + body)
    path.chmod(0o700)


@pytest.fixture
def network_commands(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable(bin_dir / "uname", 'printf "%s\\n" "$TEST_PLATFORM"\n')
    executable(
        bin_dir / "unshare",
        'printf "%s\\0" "$@" > "$PRIVATE_NETWORK_UNSHARE_ARGS"\n'
        '[[ "${TEST_UNSHARE_STATUS:-0}" == 0 ]] || exit "$TEST_UNSHARE_STATUS"\n'
        '[[ "$1" == --user && "$2" == --map-root-user && "$3" == --net && "$4" == -- ]]\n'
        'shift 4\nexec "$@"\n',
    )
    executable(
        bin_dir / "ip",
        'printf "%s\\n" "$*" > "$PRIVATE_NETWORK_IP_ARGS"\nexit "${TEST_IP_STATUS:-0}"\n',
    )
    return bin_dir, {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "TEST_PLATFORM": "Linux",
        "PRIVATE_NETWORK_UNSHARE_ARGS": str(tmp_path / "unshare-args"),
        "PRIVATE_NETWORK_IP_ARGS": str(tmp_path / "ip-args"),
        "VLLM_HOST_IP": "host-network-address",
        "NCCL_SOCKET_IFNAME": "host-interface",
        "GLOO_SOCKET_IFNAME": "host-interface",
    }


def test_private_network_preserves_arguments_environment_and_exit_status(network_commands):
    _, env = network_commands
    args = ["", "two words", 'literal $HOME `id` $(id) "quotes"', "line1\nline2", "עברית"]
    result = subprocess.run(
        [
            "/bin/bash",
            str(SCRIPT),
            "/bin/bash",
            "-c",
            'printf "%s\\0" "$VLLM_HOST_IP" "$NCCL_SOCKET_IFNAME" "$GLOO_SOCKET_IFNAME" "$@"; exit 37',
            "child",
            *args,
        ],
        env=env,
        capture_output=True,
    )
    assert result.returncode == 37, result.stderr
    assert result.stdout.decode().split("\0") == ["127.0.0.1", "=lo", "lo", *args, ""]
    assert Path(env["PRIVATE_NETWORK_IP_ARGS"]).read_text() == "link set lo up\n"
    namespace_args = Path(env["PRIVATE_NETWORK_UNSHARE_ARGS"]).read_bytes().decode().split("\0")
    assert namespace_args[:4] == ["--user", "--map-root-user", "--net", "--"]
    assert namespace_args[-len(args) - 1 : -1] == args


@pytest.mark.parametrize("failed_step", ["unshare", "ip"])
def test_private_network_setup_failure_does_not_run_command(network_commands, failed_step):
    _, env = network_commands
    env[f"TEST_{failed_step.upper()}_STATUS"] = "29"
    result = subprocess.run(
        ["/bin/bash", str(SCRIPT), "/bin/bash", "-c", "echo command-started"], env=env, capture_output=True
    )
    assert result.returncode == 29
    assert result.stdout == b""
    assert Path(env["PRIVATE_NETWORK_IP_ARGS"]).exists() == (failed_step == "ip")


@pytest.mark.parametrize("invalid", ["no_command", "not_linux", "unshare", "ip"])
def test_private_network_rejects_missing_prerequisites(network_commands, invalid):
    bin_dir, env = network_commands
    args = ["/bin/bash", "-c", "echo command-started"]
    if invalid == "no_command":
        args = []
        expected = "Usage:"
    elif invalid == "not_linux":
        env["TEST_PLATFORM"] = "Darwin"
        expected = "requires Linux"
    else:
        (bin_dir / invalid).unlink()
        env["PATH"] = str(bin_dir)
        expected = f"Missing required command: {invalid}"
    result = subprocess.run(["/bin/bash", str(SCRIPT), *args], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert expected in result.stderr
    assert result.stdout == ""
    assert not Path(env["PRIVATE_NETWORK_UNSHARE_ARGS"]).exists()
