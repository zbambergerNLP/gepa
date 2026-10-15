"""Expose one process-verified vLLM server through an authenticated Modal tunnel."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import http.client
import json
import os
import re
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from examples.common.remote_runtime import REQUEST_HEADER, RUNTIME_HEADER
from examples.modal.config import MODELS
from examples.terminalbench.runtime import load_runtime_record


def atomic_json(path: Path, value: object) -> None:
    """Publish complete evidence before another reader can observe it."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


class UncertainRequestError(RuntimeError):
    """An earlier provider attempt started without a sealed response."""


class DeliveryJournal:
    """Recover delivery without confusing transport retries with physical inference."""

    def __init__(self, root: Path, sync):
        self.root, self.sync = root, sync
        root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def complete(self, logical_id: str, body: bytes, send) -> tuple[int, bytes, bool]:
        """Never reroll an unknown start; replay identical sealed successful responses."""
        if not re.fullmatch(r"[a-f0-9-]{36}", logical_id):
            raise ValueError("A provider-generated logical request ID is required")
        with self.lock:
            directory = self.root / logical_id
            directory.mkdir(exist_ok=True)
            previous = sorted(directory.glob("attempt-*.json"))
            checksum = hashlib.sha256(body).hexdigest()
            if previous:
                last = json.loads(previous[-1].read_text())
                if last["state"] != "sealed":
                    raise UncertainRequestError("Previous physical attempt is unresolved; inspect its claim")
                if last["status"] == 200 and last["request_sha256"] == checksum:
                    response = (directory / last["response_file"]).read_bytes()
                    if hashlib.sha256(response).hexdigest() != last["response_sha256"]:
                        raise UncertainRequestError("Sealed response checksum changed")
                    return last["status"], response, True
            number = len(previous) + 1
            record_path = directory / f"attempt-{number:06}.json"
            record = {
                "logical_id": logical_id,
                "physical_attempt": number,
                "request_sha256": checksum,
                "state": "started",
                "started_unix": time.time(),
            }
            atomic_json(record_path, record)
            self.sync()
            # There is deliberately no retry around this boundary. Even a local
            # socket failure after send leaves the durable started claim intact.
            status, response = send(body)
            response_file = f"response-{number:06}.bin"
            (directory / response_file).write_bytes(response)
            atomic_json(
                record_path,
                {
                    **record,
                    "state": "sealed",
                    "status": status,
                    "finished_unix": time.time(),
                    "response_file": response_file,
                    "response_sha256": hashlib.sha256(response).hexdigest(),
                },
            )
            self.sync()
            return status, response, False


def verify_model_ready(model: str) -> None:
    """Require the pinned model to be advertised by the live inference server."""
    connection = http.client.HTTPConnection("127.0.0.1", 8000, timeout=10)
    try:
        connection.request("GET", "/v1/models")
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError("Model API is not ready")
        advertised = json.loads(response.read(1024 * 1024))
        if model.removeprefix("hosted_vllm/") not in [entry["id"] for entry in advertised.get("data", [])]:
            raise ValueError("Model API does not advertise the pinned model")
    finally:
        connection.close()


def main() -> None:
    """Run a bounded, non-restarting model allocation and retain its full evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=tuple(MODELS), required=True)
    args = parser.parse_args()
    role = MODELS[args.role]
    directory = Path("/results") / args.role
    directory.mkdir(parents=True, exist_ok=True)
    record_path = directory / "local-runtime.json"
    key = os.environ["HOSTED_VLLM_API_KEY"]
    sync = lambda: subprocess.run(["sync", "/results"], check=True)
    journal = DeliveryJournal(directory / "deliveries", sync)
    log = (directory / "server.log").open("ab", buffering=0)
    command = [
        "uv",
        "run",
        "--no-project",
        "--python",
        "/opt/serving/bin/python",
        "python",
        "-m",
        "examples.terminalbench.runtime",
        "--model",
        role["model"],
        "--model-path",
        "/models/" + role["profile"],
        "--runtime-record",
        str(record_path),
        "--port",
        "8000",
        "--",
        *role["arguments"],
    ]
    model = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    server = None

    def interrupted(signum, _frame):
        raise InterruptedError(f"Serving allocation received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        deadline = time.monotonic() + 3600
        while True:
            if model.poll() is not None:
                raise RuntimeError("Model server exited during startup; inspect server.log")
            try:
                identity = load_runtime_record(record_path, role["model"], "http://127.0.0.1:8000/v1")
                verify_model_ready(role["model"])
                break
            except (OSError, ValueError, http.client.HTTPException):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Model startup deadline exceeded")
                time.sleep(2)
        evidence = {
            "schema_version": 2,
            "provider": "modal",
            "identity": identity,
            "source_sha256": os.environ["GEPA_SOURCE_SHA256"],
            "image_id": os.environ["GEPA_IMAGE_ID"],
            "allocation": os.environ["GEPA_ALLOCATION_NAME"],
        }
        atomic_json(directory / "runtime.json", evidence)
        sync()

        class Handler(BaseHTTPRequestHandler):
            """Authenticate every route and forward bytes once without provider retries."""

            def log_message(self, *_args):
                """Avoid copying prompts or credentials into HTTP access logs."""

            def reply(self, status: int, body: bytes, *, replay: bool = False):
                """Return one complete, checksum-sealed response."""
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Gepa-Replayed", str(replay).lower())
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # The response is already sealed for an identical delivery retry.

            def handle_request(self):
                """Reject stale runtime identities and unresolved inference claims."""
                if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + key):
                    return self.reply(401, b'{"error":{"message":"Unauthorized","type":"authentication_error"}}')
                if self.command == "POST" and self.path == "/shutdown":
                    self.reply(200, b"{}")
                    threading.Thread(target=server.shutdown, daemon=True).start()
                    return
                try:
                    live = load_runtime_record(record_path, role["model"], "http://127.0.0.1:8000/v1")
                    if live != identity:
                        raise UncertainRequestError("Serving runtime changed")
                    if self.command == "GET" and self.path == "/runtime":
                        return self.reply(200, json.dumps(evidence).encode())
                    if self.command != "POST" or self.path != "/v1/chat/completions":
                        return self.reply(404, b"{}")
                    if self.headers.get(RUNTIME_HEADER) != identity["sha256"]:
                        raise ValueError("Request does not identify the verified serving runtime")
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 64 * 1024 * 1024:
                        raise ValueError("Invalid request size")
                    body = self.rfile.read(length)
                    payload = json.loads(body)
                    if payload.get("stream") or payload.get("model") != role["model"].removeprefix("hosted_vllm/"):
                        raise ValueError("Only non-streaming requests for the pinned model are supported")

                    def send(data):
                        connection = http.client.HTTPConnection("127.0.0.1", 8000, timeout=3600)
                        try:
                            connection.request(
                                "POST", self.path, body=data, headers={"Content-Type": "application/json"}
                            )
                            response = connection.getresponse()
                            return response.status, response.read()
                        finally:
                            connection.close()

                    status, response, replay = journal.complete(self.headers.get(REQUEST_HEADER, ""), body, send)
                    self.reply(status, response, replay=replay)
                except (OSError, ValueError, RuntimeError, http.client.HTTPException) as error:
                    # 409 is outside the fixed provider retry policy. The durable
                    # claim survives for inspection instead of creating another call.
                    self.reply(
                        409,
                        json.dumps(
                            {"error": {"message": type(error).__name__, "type": "gepa_infrastructure_error"}}
                        ).encode(),
                    )

            do_GET = handle_request  # noqa: N815 - BaseHTTPRequestHandler protocol
            do_POST = handle_request  # noqa: N815 - BaseHTTPRequestHandler protocol

        server = ThreadingHTTPServer(("0.0.0.0", 8080), Handler)
        server.serve_forever()
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if server is not None:
            server.server_close()
        if model.poll() is None:
            os.killpg(model.pid, signal.SIGTERM)
            try:
                model.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(model.pid, signal.SIGKILL)
                model.wait()
        log.close()
        from examples.modal.worker import seal_files

        seal_files(directory, directory / "serving-files.json")
        sync()


if __name__ == "__main__":
    main()
