"""Plan, prepare, launch, inspect, and download isolated Modal benchmark runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

from examples.common.remote_runtime import request_json
from examples.modal.config import BENCHMARKS, MODELS, ROOT, SDK_VERSION, roles, validate_plan
from examples.modal.gateway import atomic_json
from examples.modal.images import benchmark_image, serving_image
from examples.modal.worker import data_key

APP_NAME = "gepa-benchmarks"


def sdk():
    """Load the tested SDK without requiring Modal for ordinary benchmark use."""
    import modal

    if modal.__version__ != SDK_VERSION:
        raise RuntimeError(f"Use modal=={SDK_VERSION}; SDK APIs and VM semantics are version-sensitive")
    return modal


def frozen_source(destination: Path) -> dict:
    """Export one clean commit; exclude every ignored or untracked credential/cache."""
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if dirty:
        raise ValueError("Commit source changes first; Modal launches require a clean, reviewable Git revision")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    with destination.open("xb") as stream:
        subprocess.run(["git", "archive", "--format=tar", revision], cwd=ROOT, stdout=stream, check=True)
    return {"revision": revision, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}


def save_receipt(path: Path, receipt: dict) -> None:
    """Publish every known allocation ID immediately after its creation returns."""
    atomic_json(path, receipt)


def read_volume_json(volume, path: str) -> dict:
    """Read committed evidence without opening another compute allocation."""
    return json.loads(b"".join(volume.read_file(path)))


def resources(plan: dict) -> dict:
    """Allocate CPU VM for Docker, or an encoder GPU only when OBLIQ requests CUDA."""
    return {
        "runtime": "vm" if plan["benchmark"] == "terminalbench" else "gvisor",
        "cpu": 16 if plan["benchmark"] == "terminalbench" else 8,
        "memory": 131072,
        **({"gpu": "H200"} if plan["benchmark"] == "obliqbench" and plan["embedding"]["device"] == "cuda" else {}),
    }


def create_once(modal, receipt: dict, path: Path, label: str, **kwargs):
    """Preserve an unknown creation outcome; never automatically retry it."""
    if label in receipt["allocations"]:
        raise ValueError(f"Allocation {label} already has a creation intent")
    entry = {"name": kwargs["name"], "state": "creation_started", "started_unix": time.time()}
    receipt["allocations"][label] = entry
    save_receipt(path, receipt)
    command = kwargs.pop("args")
    sandbox = modal.Sandbox.create(*command, **kwargs)
    entry.update(id=sandbox.object_id, state="created")
    save_receipt(path, receipt)
    return sandbox


def wait_for_runtime(sandbox, endpoint: str, key: str, source: dict, image_id: str, *, timeout: float = 3600) -> dict:
    """Poll authenticated metadata; Modal's five-minute probes cannot cover large-model startup."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if sandbox.poll() is not None:
            raise RuntimeError("Model allocation exited during startup; inspect its saved logs")
        try:
            record = request_json(endpoint, "/runtime", timeout=15, api_key=key)
        except (OSError, RuntimeError, ValueError):
            time.sleep(5)
            continue
        if record.get("source_sha256") != source["sha256"] or record.get("image_id") != image_id:
            raise ValueError("Ready server does not match the frozen source and image")
        return record
    raise TimeoutError("Model startup timed out; its allocation ID is retained in the receipt")


def launch(plan_path: Path, receipt_path: Path, prepared_path: Path | None) -> None:
    """Create a preparation allocation or an evaluation with exclusively owned workers."""
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"schema_version": 1, "phase": "run" if prepared_path else "prepare", "plan": plan, "allocations": {}}
    # The exclusive local intent is deliberately retained even if authentication,
    # image creation, or a later network operation fails.
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream)
    try:
        launch_claimed(plan_path, receipt_path, prepared_path, plan, receipt)
    except BaseException as error:
        receipt.update(state="launch_failed", error_type=type(error).__name__)
        save_receipt(receipt_path, receipt)
        cleanup_failed_launch(receipt, receipt_path)
        raise


def cleanup_failed_launch(receipt: dict, path: Path) -> None:
    """Stop known workers only when no driver creation has been attempted."""
    if "driver" in receipt["allocations"] or not receipt["allocations"]:
        return
    modal = sdk()
    for entry in receipt["allocations"].values():
        if "id" not in entry:
            continue
        entry["cleanup"] = "termination_started"
        save_receipt(path, receipt)
        try:
            sandbox = modal.Sandbox.from_id(entry["id"])
            sandbox.terminate()
            entry["cleanup"] = {"state": "termination_returned", "exit_code": sandbox.poll()}
        except Exception as error:
            entry["cleanup"] = {"state": "termination_outcome_unknown", "error_type": type(error).__name__}
        save_receipt(path, receipt)


def launch_claimed(plan_path: Path, receipt_path: Path, prepared_path: Path | None, plan: dict, receipt: dict) -> None:
    """Perform the explicit launch after its exclusive local intent is saved."""
    archive = receipt_path.with_suffix(".source.tar")
    source = frozen_source(archive)
    plan["source_sha256"] = source["sha256"]
    receipt.update(source=source, source_archive=str(archive.resolve()))
    save_receipt(receipt_path, receipt)
    modal = sdk()
    app = modal.App.lookup(APP_NAME, create_if_missing=True)
    claims = modal.Dict.from_name(APP_NAME + "-claims", create_if_missing=True)
    if not claims.put(receipt["phase"] + ":" + plan["run_id"], {"source": source, "plan": plan}, skip_if_exists=True):
        raise RuntimeError(
            "This run ID is already claimed in Modal; inspect the original receipt instead of resubmitting"
        )
    credential_secrets = [modal.Secret.from_name(name) for name in plan["secrets"]]
    receipt["results_volume"] = f"gepa-{plan['run_id']}-{receipt['phase']}"
    results = modal.Volume.from_name(receipt["results_volume"], create_if_missing=True, version=2)
    with results.batch_upload() as upload:
        upload.put_file(archive, "/source.tar")
        upload.put_file(plan_path, "/requested-plan.json")
    save_receipt(receipt_path, receipt)
    common = {"app": app, "workdir": "/opt/gepa", "region": plan.get("region"), "timeout": plan["timeout"]}
    command = ["uv", "run", "--no-project", "python", "-m", "examples.modal.worker", receipt["phase"]]
    with tempfile.TemporaryDirectory(prefix="gepa-modal-source-") as temporary:
        source_dir = Path(temporary)
        with tarfile.open(archive) as bundle:
            bundle.extractall(source_dir, filter="data")
        if prepared_path is None:
            image = benchmark_image(modal, source_dir, plan["benchmark"]).build(app)
            receipt["image_id"] = image.object_id
            receipt["data_volume"] = f"gepa-{plan['run_id']}-data"
            receipt["models_volume"] = f"gepa-{plan['run_id']}-models"
            data = modal.Volume.from_name(receipt["data_volume"], create_if_missing=True, version=2)
            models = modal.Volume.from_name(receipt["models_volume"], create_if_missing=True, version=2)
            create_once(
                modal,
                receipt,
                receipt_path,
                "prepare",
                args=command,
                name=plan["run_id"] + "-prepare",
                image=image,
                secrets=credential_secrets,
                volumes={"/results": results, "/data": data, "/models": models},
                env={"GEPA_PLAN": json.dumps(plan), "HF_HOME": "/data/huggingface"},
                **resources(plan),
                **common,
            )
        else:
            prepared = json.loads(prepared_path.read_text())
            if prepared["phase"] != "prepare" or data_key(prepared["plan"]) != data_key(plan):
                raise ValueError("The preparation receipt must match source, benchmark, and data/runtime profile")
            preparation = modal.Sandbox.from_id(prepared["allocations"]["prepare"]["id"])
            proof = read_volume_json(modal.Volume.from_name(prepared["results_volume"]), "/prepare-status.json")
            if preparation.poll() != 0 or proof["state"] != "completed":
                raise ValueError("Preparation must finish successfully before reserving model GPUs")
            data = modal.Volume.from_name(prepared["data_volume"])
            models = modal.Volume.from_name(prepared["models_volume"])
            if read_volume_json(data, "/ready.json")["key"] != data_key(plan):
                raise ValueError("Preparation seal does not match this plan")
            if read_volume_json(models, "/ready.json")["models"] != MODELS:
                raise ValueError("Prepared model profiles changed")
            receipt["preparation_receipt"] = str(prepared_path.resolve())
            receipt["image_id"] = prepared["image_id"]
            gateway_key = secrets.token_urlsafe(48)
            private_secret = modal.Secret.from_dict({"HOSTED_VLLM_API_KEY": gateway_key})
            endpoints = {}
            # Finish all image builds before the first GPU reservation.
            serving = {role: serving_image(modal, source_dir, role).build(app) for role in roles(plan)}
            for role in roles(plan):
                profile, image = MODELS[role], serving[role]
                name = plan["run_id"] + "-" + role
                worker = create_once(
                    modal,
                    receipt,
                    receipt_path,
                    role,
                    args=["uv", "run", "--no-project", "python", "-m", "examples.modal.gateway", "--role", role],
                    name=name,
                    image=image,
                    secrets=[private_secret],
                    env={
                        "GEPA_SOURCE_SHA256": source["sha256"],
                        "GEPA_IMAGE_ID": image.object_id,
                        "GEPA_ALLOCATION_NAME": name,
                    },
                    gpu=profile["gpu"],
                    cpu=profile["cpu"],
                    memory=profile["memory"],
                    runtime="gvisor",
                    volumes={"/results": results, "/models": models.with_mount_options(read_only=True)},
                    encrypted_ports=[8080],
                    **common,
                )
                endpoints[role] = {"api_base": worker.tunnels()[8080].url + "/v1"}
                wait_for_runtime(worker, endpoints[role]["api_base"], gateway_key, source, image.object_id)
                receipt["allocations"][role].update(endpoint=endpoints[role], state="ready")
                save_receipt(receipt_path, receipt)
            remaining = min(
                int(record["started_unix"] + plan["timeout"] - time.time() - 120)
                for record in receipt["allocations"].values()
            )
            if remaining < 300:
                raise RuntimeError("No useful allocation time remains after startup; inspect and stop owned workers")
            driver = {"image_id": prepared["image_id"], "resources": resources(plan), "region": plan.get("region")}
            create_once(
                modal,
                receipt,
                receipt_path,
                "driver",
                args=command,
                name=plan["run_id"] + "-driver",
                image=modal.Image.from_id(prepared["image_id"]),
                secrets=[*credential_secrets, private_secret],
                env={
                    "GEPA_PLAN": json.dumps(plan),
                    "GEPA_ENDPOINTS": json.dumps(endpoints),
                    "GEPA_DRIVER_IDENTITY": json.dumps(driver),
                    "HF_HOME": "/data/huggingface",
                },
                volumes={"/results": results, "/data": data.with_mount_options(read_only=True)},
                **resources(plan),
                **{**common, "timeout": remaining},
            )
    receipt["state"] = "submitted"
    save_receipt(receipt_path, receipt)
    print(json.dumps({"receipt": str(receipt_path), "allocations": receipt["allocations"]}, indent=2))


def inspect_receipt(path: Path, *, stop: bool = False) -> dict:
    """Read owned allocation states, or explicitly terminate only their recorded IDs."""
    modal = sdk()
    receipt = json.loads(path.read_text())
    report = {"allocations": {}}
    for label, entry in receipt["allocations"].items():
        if "id" not in entry:
            report["allocations"][label] = {"state": "creation_outcome_unknown", "name": entry["name"]}
            continue
        sandbox = modal.Sandbox.from_id(entry["id"])
        if stop:
            sandbox.terminate()
        code = sandbox.poll()
        report["allocations"][label] = {
            "id": entry["id"],
            "exit_code": code,
            "state": "running" if code is None else "terminal",
        }
    try:
        if "results_volume" not in receipt:
            return {**report, "worker": {"state": "launch_did_not_create_results_volume"}}
        report["worker"] = read_volume_json(
            modal.Volume.from_name(receipt["results_volume"]), f"/{receipt['phase']}-status.json"
        )
    except FileNotFoundError:
        report["worker"] = {"state": "no_committed_status_yet"}
    return report


def download(path: Path, destination: Path) -> dict:
    """Download terminal evidence and verify every available sealed file digest."""
    modal = sdk()
    report = inspect_receipt(path)
    if any(row["state"] != "terminal" for row in report["allocations"].values()):
        raise ValueError("Wait for every owned allocation to be terminal before taking the final evidence archive")
    receipt = json.loads(path.read_text())
    destination.mkdir(parents=True, exist_ok=False)
    volume = modal.Volume.from_name(receipt["results_volume"])
    hashes = {}
    for entry in volume.iterdir("/", recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        relative = Path(entry.path.lstrip("/"))
        if ".." in relative.parts:
            raise ValueError("Invalid artifact path")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        checksum = hashlib.sha256()
        with target.open("xb") as stream:
            for chunk in volume.read_file(entry.path):
                stream.write(chunk)
                checksum.update(chunk)
        hashes[str(relative)] = checksum.hexdigest()
    verified = 0
    for manifest in destination.rglob("*-files.json"):
        for relative, expected in json.loads(manifest.read_text())["files"].items():
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError("Invalid sealed artifact path")
            key = str((manifest.parent / relative).relative_to(destination))
            if hashes.get(key) != expected:
                raise ValueError(f"Sealed artifact checksum mismatch: {key}")
            verified += 1
    if hashes.get("source.tar") != receipt["source"]["sha256"]:
        raise ValueError("Frozen source archive checksum mismatch")
    expected_seals = [receipt["phase"] + "-files.json"] + [
        role + "/serving-files.json" for role in MODELS if role in receipt["allocations"]
    ]
    result = {
        "downloaded_files": len(hashes),
        "verified_sealed_entries": verified,
        "sha256": hashes,
        "scheduler": report,
        "missing_seals": [name for name in expected_seals if name not in hashes],
        "all_expected_seals_present": all(name in hashes for name in expected_seals),
    }
    atomic_json(destination / "download-verification.json", result)
    return {key: value for key, value in result.items() if key != "sha256"}


def main() -> None:
    """Keep billable operations explicit and ordinary planning entirely offline."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    planning = sub.add_parser("plan")
    planning.add_argument("benchmark", choices=BENCHMARKS)
    planning.add_argument("--run-id", required=True)
    planning.add_argument("--output", type=Path, required=True)
    planning.add_argument("--mode", choices=("pilot", "optimizer-pilot", "optimize", "baseline"), default="pilot")
    planning.add_argument("--timeout", type=int, default=43200)
    planning.add_argument("--region")
    planning.add_argument("--secret", action="append", default=[])
    planning.add_argument("--terminalbench-scope", choices=("della88", "all89"), default="della88")
    planning.add_argument("--embedding-device", choices=("cpu", "cuda"), default="cuda")
    planning.add_argument("--embedding-attention", choices=("eager", "sdpa"), default="sdpa")
    planning.add_argument("--embedding-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    planning.add_argument("--embedding-batch-size", type=int, default=1)
    for name in ("prepare", "launch"):
        command = sub.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--receipt", type=Path, required=True)
        if name == "launch":
            command.add_argument("--prepared", type=Path, required=True)
    for name in ("status", "stop", "download"):
        command = sub.add_parser(name)
        command.add_argument("--receipt", type=Path, required=True)
        if name == "download":
            command.add_argument("--output", type=Path, required=True)
    args, extra = parser.parse_known_args()
    if args.command != "plan" and extra:
        parser.error(f"Unexpected arguments: {extra}")
    if args.command == "plan":
        plan = {
            "schema_version": 1,
            "benchmark": args.benchmark,
            "run_id": args.run_id,
            "mode": args.mode,
            "timeout": args.timeout,
            "region": args.region,
            "secrets": args.secret,
            "terminalbench_scope": args.terminalbench_scope,
            "embedding": {
                "device": args.embedding_device,
                "attention": args.embedding_attention,
                "dtype": args.embedding_dtype,
                "batch_size": args.embedding_batch_size,
            },
            "arguments": extra[1:] if extra[:1] == ["--"] else extra,
        }
        validate_plan(plan)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(plan, stream, indent=2)
        print(
            json.dumps(
                {
                    "plan": str(args.output),
                    "driver": resources(plan),
                    "models": {role: MODELS[role]["gpu"] for role in roles(plan)},
                    "billable_actions": False,
                },
                indent=2,
            )
        )
    elif args.command in {"prepare", "launch"}:
        launch(args.plan, args.receipt, getattr(args, "prepared", None))
    elif args.command == "download":
        print(json.dumps(download(args.receipt, args.output), indent=2))
    else:
        print(json.dumps(inspect_receipt(args.receipt, stop=args.command == "stop"), indent=2))


if __name__ == "__main__":
    main()
