"""Keep additive source reviews narrower than experiment compatibility."""

import hashlib
import json
from copy import deepcopy

import pytest

from examples.hotpotqa.source_compatibility import comparison_runtime


@pytest.fixture
def reviewed_contract():
    runtime = {
        "campaign_id": "campaign",
        "source_commit": "b" * 40,
        "source_manifest_sha256": "2" * 64,
        "serving_env_sha256": "runtime must not change",
    }
    return {
        "condition": "random",
        "optimizer": {"max_metric_calls": 6871},
        "execution_runtime": runtime,
        "source_compatibility": {
            **{key: runtime[key] for key in ("campaign_id", "source_commit", "source_manifest_sha256")},
            "schema_version": 1,
            "base_source_commit": "a" * 40,
            "base_source_manifest_sha256": "1" * 64,
            "review_sha256": "3" * 64,
        },
    }


def test_review_only_normalizes_source_provenance(reviewed_contract):
    original = deepcopy(reviewed_contract)
    result = comparison_runtime(reviewed_contract)
    assert result == {
        **original["execution_runtime"],
        "source_commit": "a" * 40,
        "source_manifest_sha256": "1" * 64,
    }
    assert reviewed_contract == original


@pytest.mark.parametrize("change", ["condition", "budget", "campaign", "source", "manifest", "review"])
def test_review_cannot_authorize_another_experiment(reviewed_contract, change):
    if change == "condition":
        reviewed_contract["condition"] = "action"
    elif change == "budget":
        reviewed_contract["optimizer"]["max_metric_calls"] = 13742
    elif change == "review":
        reviewed_contract["source_compatibility"]["review_sha256"] = "missing"
    else:
        key = {"campaign": "campaign_id", "source": "source_commit", "manifest": "source_manifest_sha256"}[change]
        reviewed_contract["execution_runtime"][key] = "changed"
    with pytest.raises(ValueError):
        comparison_runtime(reviewed_contract)


def test_unreviewed_sources_remain_distinct(reviewed_contract):
    reviewed_contract.pop("source_compatibility")
    assert comparison_runtime(reviewed_contract) == reviewed_contract["execution_runtime"]


def test_editor_handoff_requires_its_own_review_and_exact_optimizer(reviewed_contract):
    contract = reviewed_contract
    contract["condition"] = "react_v2"
    contract["optimizer"].update(rendered_seed={"sys": "original"}, react_execution={"completion": "single_response_ordered_tool_batch"})
    review = contract["source_compatibility"]
    review.update(schema_version=2, kind="single_call_editor_tracking_handoff",
                  optimizer_sha256=hashlib.sha256(json.dumps(contract["optimizer"], sort_keys=True, separators=(",", ":")).encode()).hexdigest())
    assert comparison_runtime(contract)["source_commit"] == "a" * 40
    contract["optimizer"]["react_execution"]["completion"] = "different_algorithm"
    with pytest.raises(ValueError):
        comparison_runtime(contract)
    contract["condition"] = "vanilla"
    with pytest.raises(ValueError):
        comparison_runtime(contract)
