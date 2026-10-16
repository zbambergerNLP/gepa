"""Verify authenticated Modal serving evidence without relaxing local checks."""

from __future__ import annotations

import http.client
import json
import os
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlsplit

from gepa.response_journal import register_verified_remote_endpoint

RUNTIME_HEADER = "X-Gepa-Runtime-Sha256"
REQUEST_HEADER = "X-Gepa-Logical-Request-Id"


def request_json(
    api_base: str, route: str, *, method: str = "GET", timeout: float = 30, api_key: str | None = None
) -> dict:
    """Call one authenticated HTTPS endpoint without redirects or automatic retries."""
    endpoint = urlsplit(api_base)
    if (
        endpoint.scheme != "https"
        or not endpoint.hostname
        or endpoint.path != "/v1"
        or endpoint.username
        or endpoint.password
        or endpoint.query
        or endpoint.fragment
    ):
        raise ValueError("Remote serving requires a credential-free HTTPS /v1 endpoint")
    key = api_key or os.environ.get("HOSTED_VLLM_API_KEY")
    if not key:
        raise ValueError("HOSTED_VLLM_API_KEY is required for the private Modal model gateway")
    connection = http.client.HTTPSConnection(endpoint.hostname, endpoint.port or 443, timeout=timeout)
    try:
        connection.request(method, route, headers={"Authorization": "Bearer " + key})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"Remote serving check failed with HTTP {response.status}")
        return json.loads(response.read(1024 * 1024))
    finally:
        connection.close()


def load_remote_runtime(record: dict, model: str, api_base: str | None) -> dict:
    """Match fresh process-checked evidence to the frozen endpoint and model identity."""
    from examples.terminalbench.runtime import validate_identity

    if record.get("schema_version") != 2 or record.get("provider") != "modal" or record.get("api_base") != api_base:
        raise ValueError("Remote runtime record does not match the Modal endpoint")
    identity = validate_identity(record.get("identity"), model)
    fresh = request_json(api_base, "/runtime")
    if fresh != {key: value for key, value in record.items() if key != "api_base"}:
        raise ValueError("Modal model runtime changed or belongs to a different serving allocation")
    register_verified_remote_endpoint(
        api_base,
        {
            "identity": identity,
            "source_sha256": record.get("source_sha256"),
            "image_id": record.get("image_id"),
        },
    )
    return deepcopy(identity)


def load_execution_runtime(path: Path, models, *, include_proposer: bool) -> dict:
    """Verify the remote roles before any benchmark evaluations are allowed."""
    record = json.loads(path.read_text())
    if record.get("schema_version") != 1 or record.get("provider") != "modal":
        raise ValueError("Unsupported execution runtime manifest")
    result = {"provider": "modal", "source_sha256": record["source_sha256"], "driver": record["driver"]}
    for role in ["solver", "proposer"] if include_proposer else ["solver"]:
        role_record = record["roles"][role]
        if role_record.get("source_sha256") != record["source_sha256"]:
            raise ValueError("Model worker and benchmark driver use different source archives")
        model = getattr(models, role + "_model")
        endpoint = getattr(models, role + "_api_base")
        identity = load_remote_runtime(role_record, model, endpoint)
        result[role] = {
            "identity": identity,
            "image_id": role_record["image_id"],
            "source_sha256": role_record["source_sha256"],
        }
    return result
