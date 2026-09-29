"""Keep a denied external API from starting expensive local model servers."""

import json
import os
import subprocess
from pathlib import Path

import httpx2
import pytest

from examples.hotpotqa.typesafe_preflight import API_BASE, check_connectivity
from gepa.strategies.jev_controller import JEV_API_BASE


@pytest.mark.parametrize("status", [200, 401, 404, 405])
def test_preflight_reaches_origin_without_credentials_or_inference(status):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx2.Response(status)

    with httpx2.Client(transport=httpx2.MockTransport(handle)) as client:
        result = check_connectivity(client)
    assert API_BASE == JEV_API_BASE
    assert result == {"status": "PASS", "http_status": status}
    assert len(requests) == 1
    assert requests[0].method == "HEAD"
    assert requests[0].url.host == "api.typesafe.ai"
    assert "authorization" not in requests[0].headers


def test_preflight_proxy_denial_is_sanitized_and_not_retried():
    requests = []

    def handle(request):
        requests.append(request)
        raise httpx2.ProxyError("403 proxy denied; sensitive proxy details")

    with httpx2.Client(transport=httpx2.MockTransport(handle)) as client:
        result = check_connectivity(client)
    assert result == {"status": "FAIL", "error_type": "ProxyError", "http_status": None}
    assert len(requests) == 1
    assert "sensitive" not in json.dumps(result)


@pytest.mark.parametrize("status", [403, 407, 429, 500, 503])
def test_preflight_rejects_unavailable_origin(status):
    with httpx2.Client(transport=httpx2.MockTransport(lambda request: httpx2.Response(status))) as client:
        assert check_connectivity(client)["status"] == "FAIL"


@pytest.mark.parametrize("exit_code", [0, 1])
def test_interactive_launcher_checks_network_before_starting_models(tmp_path, exit_code):
    root = tmp_path / "sources" / "test-source"
    (root / "examples/hotpotqa").mkdir(parents=True)
    marker = tmp_path / "models-started"
    (root / "examples/hotpotqa/run_hotpotqa.sbatch").write_text('touch "$MODEL_MARKER"\n')
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        '#!/bin/bash\n[[ "$no_proxy" == *"127.0.0.1"* ]] || exit 90\n'
        '[[ "$no_proxy" == *"localhost"* ]] || exit 91\n'
        '[[ "$no_proxy" == *"::1"* ]] || exit 92\n'
        '[[ "$no_proxy" == *"existing.internal"* ]] || exit 93\n'
        '[[ "$HTTPS_PROXY" == "http://approved-proxy" ]] || exit 94\n'
        '[[ "$*" == *"examples.hotpotqa.typesafe_preflight"* ]] || exit 95\n'
        'exit "$PREFLIGHT_EXIT"\n'
    )
    fake_uv.chmod(0o700)
    bootstrap = tmp_path / "bootstrap.sh"
    bootstrap.write_text(
        'module() { [[ "$*" == "load proxy/default" ]] && export HTTPS_PROXY=http://approved-proxy; }\n'
    )
    export = tmp_path / "interactive.env"
    export.write_bytes(
        b"\0".join(
            f"{key}={value}".encode()
            for key, value in {
                "HOTPOTQA_PILOT_ONLY": "1",
                "SCRATCH_BASE": tmp_path,
                "HOTPOTQA_SOURCE_COMMIT": "test-source",
                "HOTPOTQA_PILOT_ROOT": tmp_path / "output",
                "GEPA_UV_BIN": fake_uv,
                "GEPA_VENV_DIR": tmp_path / "venv",
            }.items()
        )
        + b"\0"
    )
    script = Path(__file__).resolve().parents[1] / "scripts/della/remote/run_hotpotqa_interactive.sh"
    stage = "jev-quality" if "jev-quality)" in script.read_text() else "diversity-quality"
    result = subprocess.run(
        ["bash", str(script), str(export), stage],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "SLURM_JOB_ID": "test",
            "BASH_ENV": str(bootstrap),
            "MODEL_MARKER": str(marker),
            "PREFLIGHT_EXIT": str(exit_code),
            "NO_PROXY": "existing.internal",
        },
    )
    assert result.returncode == exit_code, result.stderr
    assert marker.exists() is (exit_code == 0)
