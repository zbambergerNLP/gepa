"""Check TypeSafe HTTPS reachability before allocating memory to local models."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx2

API_BASE = "https://api.typesafe.ai"


def check_connectivity(client: httpx2.Client) -> dict:
    """Probe the API without credentials, inference, redirects or retries."""
    try:
        response = client.head(API_BASE)
    except httpx2.HTTPError as error:
        return {"status": "FAIL", "error_type": type(error).__name__, "http_status": None}
    # A missing root route or an authentication challenge still proves HTTPS reachability.
    reachable = 200 <= response.status_code < 400 or response.status_code in {401, 404, 405}
    return {"status": "PASS" if reachable else "FAIL", "http_status": response.status_code}


def main() -> None:
    """Save a sanitized connectivity result and stop before model loading on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with httpx2.Client(timeout=10.0, follow_redirects=False) as client:
        result = check_connectivity(client)
    result.update(
        observed_at=datetime.now(timezone.utc).isoformat(),
        allocation_job_id=os.environ.get("SLURM_JOB_ID"),
        api_base=API_BASE,
        proxy_configured=any(os.environ.get(key) for key in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")),
        model_requests=0,
        authentication_verified=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if result["status"] != "PASS":
        raise SystemExit(
            "TypeSafe network preflight failed; no local models were loaded. "
            "Princeton proxy/default must permit api.typesafe.ai:443 before this pilot can run."
        )


if __name__ == "__main__":
    main()
