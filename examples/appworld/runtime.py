"""Bound a private subprocess around the official AppWorld environment."""

from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from examples.appworld.benchmark_settings import (
    APPWORLD_VERSION,
    CODE_TIMEOUT_SECONDS,
    ENVIRONMENT_SEED,
    PYTHON_VERSION,
    RPC_TIMEOUT_SECONDS,
)
from examples.appworld.utils import DATA_PIN_PATH

WORKER_PATH = Path(__file__).with_name("worker.py")


class AppWorldRuntimeError(RuntimeError):
    """Abort on missing or invalid environment/evaluator evidence."""


class OfficialAppWorld:
    """Keep Pydantic 1, the Python shell, DB globals, and signal timeouts isolated."""

    def __init__(self, root: Path, python: Path, *, rpc_timeout: float = RPC_TIMEOUT_SECONDS):
        self.root = root.resolve()
        self.python = python.absolute()
        self.rpc_timeout = rpc_timeout
        self._process: subprocess.Popen | None = None
        self._stderr: Any = None
        self._buffer = b""

    def __enter__(self) -> OfficialAppWorld:
        log_dir = self.root / "experiments" / "outputs" / "gepa-runtime"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{uuid.uuid4().hex}.log"
        self._stderr = log_path.open("wb")
        log_path.chmod(0o600)
        environment = {
            **os.environ,
            "APPWORLD_ROOT": str(self.root),
            "APPWORLD_CACHE": str(self.root / ".cache"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        try:
            self._process = subprocess.Popen(
                ["uv", "run", "--no-project", "--python", str(self.python), str(WORKER_PATH)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                env=environment,
                cwd=self.root,
                start_new_session=True,
            )
        except BaseException:
            self._stderr.close()
            raise
        return self

    def request(self, operation: str, **payload: Any) -> dict[str, Any]:
        """Read exactly one complete JSON response within a real wall-clock deadline."""
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            raise AppWorldRuntimeError("AppWorld worker is not running.")
        deadline = time.monotonic() + self.rpc_timeout
        try:
            process.stdin.write(json.dumps({"operation": operation, **payload}, allow_nan=False).encode() + b"\n")
            process.stdin.flush()
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while b"\n" not in self._buffer:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise AppWorldRuntimeError(f"AppWorld {operation} timed out.")
                    block = os.read(process.stdout.fileno(), 65536)
                    if not block:
                        raise AppWorldRuntimeError(f"AppWorld worker exited during {operation}.")
                    self._buffer += block
                    if len(self._buffer) > 32 * 1024 * 1024:
                        raise AppWorldRuntimeError("AppWorld worker returned an oversized response.")
            line, self._buffer = self._buffer.split(b"\n", 1)
            response = json.loads(line)
        except (OSError, ValueError) as error:
            raise AppWorldRuntimeError(f"Invalid AppWorld {operation} response.") from error
        if (
            not isinstance(response, dict)
            or response.get("ok") is not True
            or not isinstance(response.get("result"), dict)
        ):
            raise AppWorldRuntimeError(f"AppWorld {operation} failed; inspect the private runtime log.")
        return response["result"]

    def initialize(self, record: dict[str, Any]) -> dict[str, Any]:
        """Reset one official task and expose only its public solver context."""
        return self.request(
            "initialize",
            config={
                "task_id": record["task_id"],
                "experiment_name": "gepa-appworld-" + uuid.uuid4().hex,
                "random_seed": ENVIRONMENT_SEED,
                "timeout_seconds": CODE_TIMEOUT_SECONDS,
                "max_interactions": 1000,
                "max_api_calls_per_interaction": 1000,
                "load_ground_truth": True,
                "ground_truth_mode": "minimal",
                "raise_on_unsafe_syntax": True,
                "null_patch_unsafe_execution": True,
            },
        )

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        process = self._process
        if process is not None:
            try:
                if process.stdin is not None:
                    process.stdin.close()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            finally:
                if process.stdout is not None:
                    process.stdout.close()
                self._stderr.close()


def inspect_runtime(root: Path, python: Path) -> dict[str, Any]:
    """Reject unsupported versions, dependency drift, and modified engine code."""
    if not python.is_file():
        raise ValueError("Missing AppWorld Python runtime; run examples.appworld.prepare first.")
    with OfficialAppWorld(root, python) as runtime:
        identity = runtime.request("inspect")
    pin = json.loads(DATA_PIN_PATH.read_text())
    if identity["python_version"] != PYTHON_VERSION or identity["appworld_version"] != APPWORLD_VERSION:
        raise ValueError("AppWorld requires its pinned Python and package versions.")
    if identity != pin["runtime"]:
        raise ValueError("AppWorld runtime drift; recreate the environment from runtime-requirements.txt.")
    return identity
