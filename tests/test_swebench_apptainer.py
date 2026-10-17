"""Verify immutable images, allocation limits and host isolation without model calls."""

import hashlib
import io
import json

import pytest

from examples.swebench_live import apptainer, apptainer_images


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

    def run(self, argv, timeout, stdin=None, merge_output=False):
        calls.append((argv, dict(self.env)))
        return {"exit_code": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(apptainer.ApptainerTask, "_run", run)
    with apptainer.ApptainerTask(
        {"backend": "apptainer", "image_id": str(image), "sif_sha256": apptainer.sha256(image)}, 16
    ) as task:
        directory = task.directory
        task.execute("touch /testbed/state")
        task.execute("test -f /testbed/state")
        assert calls[-1][0][-4] == calls[-2][0][-4] == str(task.sandbox)
        assert "--containall" in calls[-1][0] and "--no-home" in calls[-1][0]
        assert calls[-1][0][calls[-1][0].index("--network") + 1] == "none"
        assert "--bind" not in calls[-1][0]
        assert not any(k in calls[-1][1] for k in ("OPENROUTER_API_KEY", "APPTAINER_BIND", "APPTAINERENV_SECRET"))
    assert not directory.exists()


def test_registry_only_accepts_official_task_images():
    with pytest.raises(ValueError, match="official"):
        apptainer_images.resolve_image("somebody/foreign:latest")
