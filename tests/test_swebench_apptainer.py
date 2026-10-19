"""Verify immutable images, allocation limits and host isolation without model calls."""

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.swebench_live import apptainer, apptainer_images


@pytest.mark.parametrize("explicit_tmpdir", [False, True])
def test_staging_extracts_locally_and_publishes_atomically(monkeypatch, tmp_path, explicit_tmpdir):
    assets = tmp_path / "shared-assets"
    local = tmp_path / "local-tmp"
    local.mkdir()
    tag = "starryzhang/sweb.eval.x86_64.example:latest"
    pin = tag.rsplit(":", 1)[0] + "@sha256:" + "a" * 64
    monkeypatch.setattr(apptainer_images, "load_data", lambda _: ({"train": [{"id": "task", "image": tag}]}, {}))
    monkeypatch.setattr(
        apptainer_images,
        "resolve_image",
        lambda _: {
            "tag": tag,
            "immutable": pin,
            "architecture": "amd64",
            "compressed_bytes": 10,
        },
    )
    monkeypatch.setattr(apptainer_images.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3))
    monkeypatch.setattr(apptainer_images.tempfile, "gettempdir", lambda: str(local))
    monkeypatch.delenv("APPTAINER_TMPDIR", raising=False)
    if explicit_tmpdir:
        monkeypatch.setenv("APPTAINER_TMPDIR", str(local))
        monkeypatch.setattr(apptainer_images.tempfile, "gettempdir", lambda: pytest.fail("Explicit root ignored"))
    calls = []

    def build(argv, *, env, check, timeout):
        extraction = Path(env["APPTAINER_TMPDIR"])
        output = Path(argv[-2])
        assert extraction.parent == local
        assert output.parent.parent == assets / "apptainer-images"
        assert extraction != output.parent
        assert argv[-1] == "docker://" + pin
        assert check and timeout == 3600
        assert not (assets / "apptainer-images" / ("a" * 64 + ".sif")).exists()
        output.write_bytes(b"verified image bytes")
        calls.append(argv)

    monkeypatch.setattr(apptainer_images.subprocess, "run", build)
    assert apptainer_images.stage_images(assets, "train", ["task"]) == {"requested": 1, "verified": 1}
    entry = json.loads((assets / "apptainer-images/manifest.json").read_text())["task"]
    apptainer_images.verify_image(entry, tag)
    assert Path(entry["image_id"]).stat().st_mode & 0o777 == 0o444
    apptainer_images.stage_images(assets, "train", ["task"])
    assert len(calls) == 1


def test_staging_rejects_insufficient_local_build_space(monkeypatch, tmp_path):
    local = tmp_path / "local-tmp"
    local.mkdir()
    tag = "starryzhang/sweb.eval.x86_64.example:latest"
    monkeypatch.setenv("APPTAINER_TMPDIR", str(local))
    monkeypatch.setattr(apptainer_images, "load_data", lambda _: ({"train": [{"id": "task", "image": tag}]}, {}))
    monkeypatch.setattr(
        apptainer_images,
        "resolve_image",
        lambda _: {
            "tag": tag,
            "immutable": "repo@sha256:" + "a" * 64,
            "compressed_bytes": 10,
        },
    )
    monkeypatch.setattr(
        apptainer_images.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(
            free=0 if Path(path) == local else 100 * 1024**3,
        ),
    )
    monkeypatch.setattr(apptainer_images.subprocess, "run", lambda *a, **k: pytest.fail("Build must not start"))
    with pytest.raises(RuntimeError, match="temporary build space"):
        apptainer_images.stage_images(tmp_path / "assets", "train")


def test_registry_resolution_pins_config_and_manifest(monkeypatch):
    config = json.dumps({"architecture": "amd64", "os": "linux"}).encode()
    config_digest = "sha256:" + hashlib.sha256(config).hexdigest()
    manifest = json.dumps({"config": {"digest": config_digest}, "layers": [{"size": 123}]}).encode()

    def fetch(request, **kwargs):
        url = request if isinstance(request, str) else request.full_url
        raw = (
            json.dumps({"token": "private-registry-token"}).encode()
            if "auth.docker.io" in url
            else (config if "/blobs/" in url else manifest)
        )
        result = io.BytesIO(raw)
        result.headers = {}
        return result

    monkeypatch.setattr(apptainer_images.urllib.request, "urlopen", fetch)
    result = apptainer_images.resolve_image("starryzhang/sweb.eval.x86_64.example:latest")
    assert result["immutable"].endswith("@sha256:" + hashlib.sha256(manifest).hexdigest())
    assert result["compressed_bytes"] == 123
    assert "private-registry-token" not in json.dumps(result)
    config = b'{"architecture":"arm64","os":"linux"}'
    with pytest.raises(ValueError, match="descriptor checksum"):
        apptainer_images.resolve_image("starryzhang/sweb.eval.x86_64.example:latest")


def test_sif_drift_and_missing_tasks_fail_closed(tmp_path):
    image = tmp_path / "task.sif"
    image.write_bytes(b"pinned image")
    entry = {
        "tag": "repo:latest",
        "immutable": "repo@sha256:pin",
        "architecture": "amd64",
        "state": "verified",
        "backend": "apptainer",
        "image_id": str(image),
        "sif_sha256": apptainer.sha256(image),
    }
    apptainer_images.verify_image(entry, "repo:latest")
    image.write_bytes(b"changed image")
    with pytest.raises(ValueError, match="checksum drift"):
        apptainer_images.verify_image(entry, "repo:latest")
    (tmp_path / "apptainer-images").mkdir()
    (tmp_path / "apptainer-images/manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="before running task"):
        apptainer_images.verify_images(tmp_path, [{"id": "unprepared", "image": "repo:latest"}])


@pytest.mark.parametrize(
    "allocation",
    [
        "JobId=42 JobState=PENDING NumCPUs=4 NumNodes=1 MinMemoryNode=16G",
        "JobId=42 JobState=RUNNING NumCPUs=8 NumNodes=1 MinMemoryNode=16G",
        "JobId=42 JobState=RUNNING NumCPUs=4 NumNodes=1 MinMemoryNode=64G",
        "JobId=42 JobState=RUNNING NumCPUs=4 NumNodes=1 MinMemoryNode=16G AllocTRES=gres/gpu=1",
    ],
)
def test_wrong_allocation_cannot_silently_change_task_resources(monkeypatch, allocation):
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    monkeypatch.setattr(apptainer.platform, "system", lambda: "Linux")
    monkeypatch.setattr(apptainer.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(apptainer.subprocess, "check_output", lambda *a, **kw: allocation)
    with pytest.raises(RuntimeError, match="exactly 4 CPUs"):
        apptainer.allocation_limits(16)


def test_task_preserves_files_without_host_mounts_or_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-reach-task")
    monkeypatch.setenv("APPTAINER_BIND", "/private:/leaked")
    monkeypatch.setenv("APPTAINERENV_SECRET", "must-not-reach-task")
    monkeypatch.setattr(apptainer, "allocation_limits", lambda _: {})
    image = tmp_path / "task.sif"
    image.write_bytes(b"pinned image")
    calls = []
    writes = []

    def run(self, argv, timeout, stdin=None, merge_output=False):
        calls.append((argv, dict(self.env)))
        if stdin is not None:
            writes.append((argv[-1], stdin))
        stdout = json.dumps({"instances": [{"instance": self.name}]}) if "list" in argv else ""
        return {"exit_code": 0, "stdout": stdout, "stderr": ""}

    monkeypatch.setattr(apptainer.ApptainerTask, "_run", run)
    with apptainer.ApptainerTask(
        {"backend": "apptainer", "image_id": str(image), "sif_sha256": apptainer.sha256(image)}, 16
    ) as task:
        directory = task.directory
        task.execute("touch /testbed/state")
        task.execute("test -f /testbed/state")
        assert calls[-1][0][-4] == calls[-2][0][-4] == "instance://" + task.name
        startup = calls[1][0]
        assert writes == [("cat > /etc/hosts", apptainer.LOOPBACK_HOSTS)]
        assert "--containall" in startup and "--no-home" in startup
        assert startup[startup.index("--network") + 1] == "none"
        assert "--bind" not in calls[-1][0]
        assert "--userns" in calls[-1][0]
        assert not any(k in calls[-1][1] for k in ("OPENROUTER_API_KEY", "APPTAINER_BIND", "APPTAINERENV_SECRET"))
    assert not directory.exists()
    assert calls[-1][0] == ["apptainer", "instance", "stop", "--force", task.name]


def test_loopback_configuration_failure_prevents_task_execution(monkeypatch, tmp_path):
    monkeypatch.setattr(apptainer, "allocation_limits", lambda _: {})
    image = tmp_path / "task.sif"
    image.write_bytes(b"pinned image")
    calls = []

    def run(self, argv, *args, **kwargs):
        calls.append(argv)
        return {
            "exit_code": 1 if argv[-1] == "cat > /etc/hosts" else 0,
            "stdout": '{"instances": [{}]}' if "list" in argv else "",
            "stderr": "read-only hosts file" if argv[-1] == "cat > /etc/hosts" else "",
        }

    monkeypatch.setattr(apptainer.ApptainerTask, "_run", run)
    task = apptainer.ApptainerTask(
        {"backend": "apptainer", "image_id": str(image), "sif_sha256": apptainer.sha256(image)}, 16
    )
    with pytest.raises(RuntimeError, match="Cannot copy evaluator input: read-only hosts file"):
        with task:
            pytest.fail("An unconfigured task must not run")
    assert not any(argv[-1].startswith("test -d") for argv in calls)
    assert calls[-1] == ["apptainer", "instance", "stop", "--force", task.name]
    assert not task.directory.exists()


def test_loopback_configuration_is_part_of_runtime_identity(monkeypatch):
    monkeypatch.setattr(apptainer.shutil, "which", lambda _: "/usr/bin/apptainer")
    monkeypatch.setattr(apptainer, "sha256", lambda _: "binary-hash")
    monkeypatch.setattr(apptainer, "allocation_limits", lambda _: {"cpus": 4})
    monkeypatch.setattr(apptainer.subprocess, "check_output", lambda *a, **kw: "1.5.0\n")
    before = apptainer.runtime_identity(16)
    assert before["hosts"] == {"path": "/etc/hosts", "content": apptainer.LOOPBACK_HOSTS}
    monkeypatch.setattr(apptainer, "LOOPBACK_HOSTS", "")
    assert apptainer.runtime_identity(16) != before


def test_failed_instance_start_preserves_error_and_removes_unused_sandbox(monkeypatch, tmp_path):
    monkeypatch.setattr(apptainer, "allocation_limits", lambda _: {})
    image = tmp_path / "task.sif"
    image.write_bytes(b"pinned image")
    calls = []

    def run(self, argv, *args, **kwargs):
        calls.append(argv)
        return {
            "exit_code": 1 if "start" in argv else 0,
            "stdout": '{"instances": []}' if "list" in argv else "",
            "stderr": "native start failure" if "start" in argv else "",
        }

    monkeypatch.setattr(apptainer.ApptainerTask, "_run", run)
    task = apptainer.ApptainerTask(
        {"backend": "apptainer", "image_id": str(image), "sif_sha256": apptainer.sha256(image)}, 16
    )
    with pytest.raises(RuntimeError, match="Cannot start task instance: native start failure"):
        with task:
            pytest.fail("Failed startup must not execute a task")
    assert not task.directory.exists()
    assert not any("stop" in argv for argv in calls)


def test_registry_only_accepts_official_task_images():
    with pytest.raises(ValueError, match="official"):
        apptainer_images.resolve_image("somebody/foreign:latest")
