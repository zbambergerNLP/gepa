"""Adapt SWE-ReX transport to isolated Apptainer instances without replacing the agent."""

import base64
import io
import json
import logging
import select
import shutil
import subprocess
import zipfile
from pathlib import Path

from pydantic import TypeAdapter
from swerex.deployment.abstract import AbstractDeployment
from swerex.deployment.hooks.abstract import CombinedDeploymentHook
from swerex.runtime.abstract import IsAliveResponse, UploadResponse, _ExceptionTransfer
from swerex.runtime.remote import RemoteRuntime

from examples.swebench_live.apptainer import ApptainerTask
from examples.swebench_live.swe_agent import verify_payload


class PipeRuntime(RemoteRuntime):
    """Reuse the upstream runtime API and exception types over a local pipe."""

    def __init__(self, process):
        super().__init__(auth_token="")
        self.process = process
        self.failed = False

    async def _request(self, endpoint, payload, output_class, num_retries=0, timeout=60):
        request = {"method": endpoint, "payload": payload.model_dump() if hasattr(payload, "model_dump") else payload}
        if self.failed:
            raise RuntimeError("SWE-ReX transport has an unresolved request")
        try:
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            if not select.select([self.process.stdout], [], [], timeout + 30)[0]:
                raise RuntimeError("SWE-ReX transport outcome unknown; request will not be replayed")
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("SWE-ReX pipe closed; inspect rex.log")
            response = json.loads(line)
        except BaseException:
            self.failed = True
            raise
        if "error" in response:
            self._handle_transfer_exception(_ExceptionTransfer(**response["error"]))
        return TypeAdapter(output_class).validate_python(response["result"])

    async def is_alive(self, *, timeout=None):
        return await self._request("is_alive", None, IsAliveResponse, timeout=timeout or 60)

    async def upload(self, request):
        source = Path(request.source_path)
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(source))
        return await self._request(
            "upload",
            {"target_path": request.target_path, "zip": base64.b64encode(data.getvalue()).decode()},
            UploadResponse,
        )


class ApptainerDeployment(AbstractDeployment):
    """Own one task and run upstream SWE-ReX inside its private namespaces."""

    def __init__(self, image, memory_gib, payload, trial):
        self.task = ApptainerTask(image, memory_gib)
        self.payload, self.trial = payload, trial
        self.logger = logging.getLogger(__name__)
        self._hooks = CombinedDeploymentHook()
        self._runtime = self.process = self.log = None

    def add_hook(self, hook):
        self._hooks.add_hook(hook)

    @property
    def runtime(self):
        if self._runtime is None:
            raise RuntimeError("SWE-ReX deployment has not started")
        return self._runtime

    async def is_alive(self, *, timeout=None):
        return await self.runtime.is_alive(timeout=timeout)

    async def start(self):
        verify_payload(self.payload)
        self.task.__enter__()
        destination = self.task.sandbox / "opt/gepa-swerex"
        if destination.exists() or not destination.parent.resolve().is_relative_to(self.task.sandbox.resolve()):
            raise ValueError("Unexpected runtime payload destination")
        shutil.copytree(self.payload, destination, symlinks=True)
        self.log = (self.trial / "rex.log").open("w")
        self.process = subprocess.Popen(
            [
                "apptainer",
                "exec",
                "--userns",
                "--cleanenv",
                "--pwd",
                "/testbed",
                "instance://" + self.task.name,
                "/opt/gepa-swerex/python/bin/python3",
                "-B",
                "/opt/gepa-swerex/rex_worker.py",
            ],
            cwd=self.task.directory,
            env=self.task.env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self._runtime = PipeRuntime(self.process)
        if not await self.is_alive():
            raise RuntimeError("SWE-ReX did not become ready")

    async def stop(self):
        try:
            if self._runtime is not None and not self._runtime.failed and self.process.poll() is None:
                await self._runtime.close()
        finally:
            self._runtime = None
            try:
                if self.process is not None:
                    if self.process.poll() is None:
                        self.process.terminate()
                    try:
                        self.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=10)
                    self.process = None
            finally:
                if self.log is not None:
                    self.log.close()
                    self.log = None
                self.task.__exit__()
