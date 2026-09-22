"""Preserve actual revisions while narrowly qualifying reviewed source handoffs."""

import hashlib
import json
import os
import re


def comparison_runtime(contract: dict) -> dict:
    """Normalize only the two reviewed source fields; retain every runtime setting."""
    runtime = dict(contract["execution_runtime"])
    review = contract.get("source_compatibility")
    if review is None:
        return runtime
    cell = (contract.get("condition"), contract.get("optimizer", {}).get("max_metric_calls"))
    if not isinstance(review, dict):
        raise ValueError("Source compatibility requires an explicit review record.")
    if review.get("schema_version") == 1:
        allowed = cell == ("random", 6_871)
    elif review.get("schema_version") == 2:
        allowed = review.get("kind") == "single_call_editor_tracking_handoff" and cell in {
            ("react_v2", 6_871), ("react_v2_random", 6_871), ("action", 6_871), ("random", 6_871),
            ("vanilla", 13_742), ("react_v2", 13_742),
        }
        optimizer = contract.get("optimizer", {})
        if "rendered_seed" in optimizer:
            digest = hashlib.sha256(json.dumps(optimizer, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
            allowed = allowed and digest == review.get("optimizer_sha256")
    else:
        allowed = False
    if (
        not allowed
        or review.get("campaign_id") != runtime.get("campaign_id")
        or review.get("source_commit") != runtime.get("source_commit")
        or review.get("source_manifest_sha256") != runtime.get("source_manifest_sha256")
    ):
        raise ValueError("Source compatibility is restricted to the exact reviewed cell and optimizer contract.")
    for key, length in (
        ("source_commit", 40),
        ("base_source_commit", 40),
        ("source_manifest_sha256", 64),
        ("base_source_manifest_sha256", 64),
        ("review_sha256", 64),
    ):
        if not re.fullmatch(rf"[0-9a-f]{{{length}}}", str(review.get(key, ""))):
            raise ValueError(f"Source compatibility lacks an exact {key}.")
    if review["source_commit"] == review["base_source_commit"]:
        raise ValueError("A source compatibility review must identify two distinct revisions.")
    runtime["source_commit"] = review["base_source_commit"]
    runtime["source_manifest_sha256"] = review["base_source_manifest_sha256"]
    return runtime


def compatibility_contract(condition: str, max_metric_calls: int) -> dict:
    """Embed the operator's pinned review without changing ordinary run contracts."""
    raw = os.environ.get("HOTPOTQA_SOURCE_COMPATIBILITY_JSON")
    if not raw:
        return {}
    record = {"source_compatibility": json.loads(raw)}
    comparison_runtime(
        {
            **record,
            "condition": condition,
            "optimizer": {"max_metric_calls": max_metric_calls},
            "execution_runtime": {
                "campaign_id": os.environ.get("HOTPOTQA_CAMPAIGN_ID"),
                "source_commit": os.environ.get("HOTPOTQA_SOURCE_COMMIT"),
                "source_manifest_sha256": os.environ.get("HOTPOTQA_SOURCE_MANIFEST_SHA256"),
            },
        }
    )
    return record


def main() -> None:
    """Print the reviewed identity for campaign locks, never worker source verification."""
    condition = os.environ["CONDITION"]
    budget = int(os.environ["MAX_METRIC_CALLS"])
    runtime = comparison_runtime(
        {
            **compatibility_contract(condition, budget),
            "condition": condition,
            "optimizer": {"max_metric_calls": budget},
            "execution_runtime": {
                "campaign_id": os.environ["HOTPOTQA_CAMPAIGN_ID"],
                "source_commit": os.environ["HOTPOTQA_SOURCE_COMMIT"],
                "source_manifest_sha256": os.environ["HOTPOTQA_SOURCE_MANIFEST_SHA256"],
            },
        }
    )
    print(runtime["source_commit"], runtime["source_manifest_sha256"])


if __name__ == "__main__":
    main()
