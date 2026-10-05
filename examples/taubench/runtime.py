"""Run the official, separately locked tau runtime across a process boundary."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any
from uuid import uuid4

from examples.taubench.benchmark_settings import PYTHON_VERSION, RUNTIME_SUPPLEMENTS

REPO_ROOT = Path(__file__).resolve().parents[2]


def worker_command(source: Path) -> list[str]:
    """Keep tau's LiteLLM pin independent of the optimizer environment."""
    command = ["uv", "run", "--directory", str(source.resolve()), "--frozen", "--extra", "knowledge"]
    for requirement in RUNTIME_SUPPLEMENTS:
        command.extend(["--with", requirement])
    return command + ["--python", PYTHON_VERSION, "python", "-m", "examples.taubench.worker"]


class TauRuntime:
    """Use upstream dependencies without mixing incompatible LiteLLM versions."""

    def __init__(
        self, source: Path, artifacts: Path, solver_model: str, solver_api_base: str | None, solver_kwargs: dict
    ):
        self.source = source.resolve()
        self.artifacts = artifacts.resolve()
        self.solver_model = solver_model
        self.solver_api_base = solver_api_base
        self.solver_kwargs = solver_kwargs

    def invoke(self, payload: dict[str, Any]) -> dict:
        """Send requests through stdin and reject interrupted or malformed worker output."""
        env = dict(os.environ)
        env.pop("UV_PROJECT_ENVIRONMENT", None)
        env.pop("VIRTUAL_ENV", None)
        env.update(
            {
                "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), str(REPO_ROOT / "src")]),
                "PYTHONHASHSEED": "0",
                "TAU2_DATA_DIR": str(self.source / "data"),
                "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            }
        )
        request = {**payload, "source": str(self.source), "artifacts": str(self.artifacts)}
        completed = subprocess.run(
            worker_command(self.source),
            input=json.dumps(request, allow_nan=False),
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )
        if completed.returncode:
            self.artifacts.mkdir(parents=True, exist_ok=True)
            error_log = self.artifacts / f"worker-error-{uuid4().hex}.log"
            error_log.write_text(completed.stderr)
            raise RuntimeError(f"tau worker failed ({completed.returncode}); see {error_log}")
        try:
            result = json.loads(completed.stdout)
        except (ValueError, TypeError) as error:
            raise ValueError("tau worker produced incomplete or malformed JSON") from error
        if not isinstance(result, dict):
            raise ValueError("tau worker did not return an object")
        return result

    def run(self, records: list[dict], candidate: dict[str, str], trial: int) -> dict:
        """Run every episode with the same fixed simulator and candidate system text."""
        return self.invoke(
            {
                "mode": "run",
                "records": records,
                "candidate": candidate,
                "trial": trial,
                "solver_model": self.solver_model,
                "solver_api_base": self.solver_api_base,
                "solver_kwargs": self.solver_kwargs,
            }
        )
