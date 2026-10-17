"""Run isolated, writable task filesystems inside a bounded Slurm allocation."""

import hashlib
import os
import platform
import shlex
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

ISOLATION = [
    "--fakeroot",
    "--ignore-fakeroot-command",
    "--writable",
    "--containall",
    "--cleanenv",
    "--no-home",
    "--no-mount",
    "hostfs,bind-paths,cwd",
    "--net",
    "--network",
    "none",
    "--pwd",
    "/testbed",
]


def sha256(path: Path) -> str:
    """Hash large images without loading their contents into memory."""
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def allocation_limits(memory_gib: int) -> dict:
    """Require the scheduler to enforce the same four-CPU task memory budget."""
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit() or platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Apptainer task execution requires an AMD64 Linux Slurm allocation")
    result = subprocess.check_output(["scontrol", "show", "job", "-o", job], text=True, timeout=30)
    fields = dict(item.split("=", 1) for item in result.split() if "=" in item)
    if (
        fields.get("JobId") != job
        or fields.get("JobState") != "RUNNING"
        or fields.get("NumCPUs") != "4"
        or fields.get("NumNodes") != "1"
        or fields.get("MinMemoryNode") not in {f"{memory_gib}G", f"{memory_gib * 1024}M"}
        or "gres/gpu" in fields.get("AllocTRES", "")
    ):
        raise RuntimeError(f"Use a single-node CPU allocation with exactly 4 CPUs and --mem={memory_gib}G")
    return {"cpus": 4, "memory_gib": memory_gib, "enforcement": "Slurm allocation", "gpu_count": 0}


def runtime_identity(memory_gib: int) -> dict:
    """Freeze container transport and isolation independently of the Docker backend."""
    executable = shutil.which("apptainer")
    if executable is None:
        raise RuntimeError("Apptainer is not installed")
    return {
        "backend": "apptainer-sandbox-v1",
        "version": subprocess.check_output([executable, "version"], text=True, timeout=30).strip(),
        "executable_sha256": sha256(Path(executable).resolve()),
        "isolation": ISOLATION,
        "resources": allocation_limits(memory_gib),
        "writable_state": "fresh sandbox and private temporary directories per task",
    }


class ApptainerTask:
    """Keep task writes across shell calls while excluding host files and credentials."""

    def __init__(self, image: dict, memory_gib: int):
        self.image, self.memory_gib = image, memory_gib
        self.temporary = None

    def __enter__(self):
        """Verify the image and unpack a fresh root without running its tests."""
        allocation_limits(self.memory_gib)
        if self.image.get("backend") != "apptainer" or sha256(Path(self.image["image_id"])) != self.image["sif_sha256"]:
            raise ValueError("Apptainer task image checksum drift")
        self.temporary = tempfile.TemporaryDirectory(prefix="swebench-task-")
        self.directory = Path(self.temporary.name)
        self.sandbox = self.directory / "sandbox"
        self.work = self.directory / "work"
        self.work.mkdir()
        self.home = self.directory / "control-home"
        self.home.mkdir()
        self.env = {
            "PATH": os.environ["PATH"],
            "HOME": str(self.home),
            "LANG": "C.UTF-8",
            "APPTAINER_TMPDIR": str(self.directory),
        }
        try:
            result = self._run(
                [
                    "apptainer",
                    "build",
                    "--sandbox",
                    "--fix-perms",
                    "--disable-cache",
                    "--no-test",
                    str(self.sandbox),
                    self.image["image_id"],
                ],
                timeout=600,
            )
            if result["exit_code"]:
                raise RuntimeError(f"Cannot unpack task image: {result['stderr']}")
            result = self.execute("test -d /testbed/.git && test -w /testbed && test $(id -u) = 0")
            if result["exit_code"]:
                raise RuntimeError(f"Cannot start isolated task root: {result['stderr']}")
        except BaseException:
            self.__exit__()
            raise
        return self

    def _run(self, argv, timeout, stdin=None, merge_output=False):
        """Terminate the owned process group on timeout and retain complete output."""
        process = subprocess.Popen(
            argv,
            cwd=self.directory,
            env=self.env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if merge_output else subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(stdin, timeout=timeout)
        except BaseException as error:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
            if isinstance(error, subprocess.TimeoutExpired):
                error.output, error.stderr = stdout, stderr
            raise
        return {"exit_code": process.returncode, "stdout": stdout, "stderr": stderr or ""}

    def execute(self, command: str, timeout: int = 120, stdin: str | None = None, merge_output: bool = False) -> dict:
        """Execute in the task root with private PID, IPC and offline network namespaces."""
        return self._run(
            [
                "apptainer",
                "exec",
                *ISOLATION,
                "--workdir",
                str(self.work),
                str(self.sandbox),
                "/bin/bash",
                "-lc",
                command,
            ],
            timeout,
            stdin,
            merge_output,
        )

    def write_text(self, path: str, text: str) -> None:
        """Copy evaluator inputs through stdin without binding their host directory."""
        result = self.execute("cat > " + shlex.quote(path), stdin=text)
        if result["exit_code"]:
            raise RuntimeError(f"Cannot copy evaluator input: {result['stderr']}")

    def __exit__(self, *_):
        """Remove only this task's unpacked filesystem and private temporary files."""
        if self.temporary is not None:
            self.temporary.cleanup()
            self.temporary = None
