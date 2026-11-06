"""Verify the approved Tau exclusions without episodes, model calls, or task-content review."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.common.benchmark_runner import ensure_contract
from examples.common.benchmark_types import BenchmarkModels
from examples.common.react_v2 import benchmark_data_identity
from examples.taubench import benchmark_settings as settings
from examples.taubench import main, utils
from examples.taubench.adapter import TauBankingAdapter, trial_seed

V4_ADDED = {
    "task_020",
    "task_026",
    "task_027",
    "task_062",
    "task_065",
    "task_066",
    "task_067",
    "task_068",
    "task_069",
    "task_082",
    "task_086",
}
EXCLUDED = {"task_017", "task_024", "task_075", "task_084", "task_094"} | V4_ADDED
V1_SHA256 = "2586a7f390abfe96977583029b58e9cbaa0c0767fbb8e5ec92222d4c8699e6f1"
V2_SHA256 = "7186155ddb09c126376361cc7d4aec8b66e413ae0aa37afad7629765831738b2"
V3_SHA256 = "e2b4c6f3fbd40338292b5064612a8d38393b6d0d3ba0251399201d0ce89e2fd7"


@pytest.fixture
def source_manifest(monkeypatch):
    manifest = json.loads(settings.MANIFEST_PATH.read_text())
    monkeypatch.setattr(utils, "inspect_source", lambda source: deepcopy(manifest))
    return manifest


@pytest.fixture
def historical_v1_manifest(source_manifest, monkeypatch):
    path = settings.MANIFEST_PATH.with_name("banking-exclusions-v1.json")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == V1_SHA256
    with monkeypatch.context() as previous:
        previous.setattr(utils, "DATASET_VERSION", "tau-banking-ambiguity-exclusions-v1")
        previous.setattr(utils, "EXCLUSIONS_MANIFEST_PATH", path)
        previous.setattr(utils, "EXCLUSIONS_MANIFEST_SHA256", V1_SHA256)
        _, manifest = utils.load_data(Path("unused"))
    assert len(manifest["records"]) == 95
    assert {row["task_id"] for row in manifest["records"]} == {
        row["task_id"] for row in source_manifest["records"] if row["task_id"] not in {"task_075", "task_084"}
    }
    return manifest


@pytest.fixture
def historical_v2_manifest(source_manifest, monkeypatch):
    path = settings.MANIFEST_PATH.with_name("banking-exclusions-v2.json")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == V2_SHA256
    with monkeypatch.context() as previous:
        previous.setattr(utils, "DATASET_VERSION", "tau-banking-ambiguity-exclusions-v2")
        previous.setattr(utils, "EXCLUSIONS_MANIFEST_PATH", path)
        previous.setattr(utils, "EXCLUSIONS_MANIFEST_SHA256", V2_SHA256)
        _, manifest = utils.load_data(Path("unused"))
    assert len(manifest["records"]) == 93
    return manifest


@pytest.fixture
def historical_v3_manifest(source_manifest, monkeypatch):
    path = settings.MANIFEST_PATH.with_name("banking-exclusions-v3.json")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == V3_SHA256
    with monkeypatch.context() as previous:
        previous.setattr(utils, "DATASET_VERSION", "tau-banking-ambiguity-exclusions-v3")
        previous.setattr(utils, "EXCLUSIONS_MANIFEST_PATH", path)
        previous.setattr(utils, "EXCLUSIONS_MANIFEST_SHA256", V3_SHA256)
        _, manifest = utils.load_data(Path("unused"))
    assert len(manifest["records"]) == 92
    return manifest


def test_only_approved_tasks_are_removed_without_resplitting(source_manifest):
    before = deepcopy(source_manifest)
    splits, manifest = utils.load_data(Path("unused"))
    retained = [record for record in before["records"] if record["task_id"] not in EXCLUDED]

    assert len(manifest["records"]) == 81
    assert manifest["records"] == retained
    assert {name: len(rows) for name, rows in splits.items()} == {"train": 41, "val": 19, "test": 21}
    for name, rows in splits.items():
        assert rows == [record for record in retained if record["split"] == name]
    assert source_manifest == before == json.loads(settings.MANIFEST_PATH.read_text())


def test_v2_extends_v1_without_changing_surviving_records(historical_v1_manifest, historical_v2_manifest):
    current = historical_v2_manifest
    retained = [row for row in historical_v1_manifest["records"] if row["task_id"] not in {"task_024", "task_094"}]
    assert current["records"] == retained
    old_exclusions = historical_v1_manifest["exclusion_manifest"]["excluded_records"]
    assert [
        item for item in current["exclusion_manifest"]["excluded_records"] if item in old_exclusions
    ] == old_exclusions
    for split in ("val", "test"):
        assert [row for row in current["records"] if row["split"] == split] == [
            row for row in historical_v1_manifest["records"] if row["split"] == split
        ]


def test_v3_extends_v2_without_changing_surviving_records(historical_v2_manifest, historical_v3_manifest):
    current = historical_v3_manifest
    assert current["records"] == [row for row in historical_v2_manifest["records"] if row["task_id"] != "task_017"]
    old_exclusions = historical_v2_manifest["exclusion_manifest"]["excluded_records"]
    assert [
        item for item in current["exclusion_manifest"]["excluded_records"] if item in old_exclusions
    ] == old_exclusions
    for split in ("val", "test"):
        assert [row for row in current["records"] if row["split"] == split] == [
            row for row in historical_v2_manifest["records"] if row["split"] == split
        ]
    assert [row["task_id"] for row in current["records"] if row["split"] == "train"][:3] == [
        "task_005",
        "task_004",
        "task_065",
    ]


def test_v4_extends_v3_without_changing_surviving_records(historical_v3_manifest):
    splits, current = utils.load_data(Path("unused"))
    assert current["records"] == [row for row in historical_v3_manifest["records"] if row["task_id"] not in V4_ADDED]
    old_exclusions = historical_v3_manifest["exclusion_manifest"]["excluded_records"]
    assert [
        item for item in current["exclusion_manifest"]["excluded_records"] if item in old_exclusions
    ] == old_exclusions
    assert all(item["record"]["split"] == "train" for item in current["exclusion_manifest"]["excluded_records"])
    for split in ("val", "test"):
        assert splits[split] == [row for row in historical_v3_manifest["records"] if row["split"] == split]
    assert [row["task_id"] for row in splits["train"][:3]] == ["task_005", "task_004", "task_059"]


@pytest.mark.parametrize("task_id", sorted(EXCLUDED))
def test_removed_task_drift_is_still_rejected_before_filtering(source_manifest, monkeypatch, task_id):
    changed = deepcopy(source_manifest)
    next(record for record in changed["records"] if record["task_id"] == task_id)["task_sha256"] = "changed"
    monkeypatch.setattr(utils, "inspect_source", lambda source: changed)
    with pytest.raises(ValueError, match="identity changed"):
        utils.load_data(Path("unused"))


def test_original_manifest_bytes_are_pinned_even_when_parsed_content_matches(source_manifest, monkeypatch, tmp_path):
    alternate = tmp_path / "same-content.json"
    alternate.write_text(json.dumps(source_manifest))
    monkeypatch.setattr(utils, "MANIFEST_PATH", alternate)
    with pytest.raises(ValueError, match="Original tau manifest"):
        utils.load_data(Path("unused"))


@pytest.mark.parametrize("change", ["task", "fingerprint", "reason", "count", "order"])
def test_exclusion_manifest_drift_fails_closed(source_manifest, monkeypatch, tmp_path, change):
    manifest = json.loads(settings.EXCLUSIONS_MANIFEST_PATH.read_text())
    if change == "task":
        manifest["excluded_records"][0]["record"]["task_id"] = "task_024"
    elif change == "fingerprint":
        manifest["excluded_records"][0]["record"]["task_sha256"] = "changed"
    elif change == "reason":
        manifest["excluded_records"][0]["reason"] = "changed"
    elif change == "count":
        manifest["retained_counts"]["train"] += 1
    else:
        manifest["excluded_records"].reverse()
    path = tmp_path / "exclusions.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(utils, "EXCLUSIONS_MANIFEST_PATH", path)
    with pytest.raises(ValueError, match="exclusion manifest"):
        utils.load_data(Path("unused"))


@pytest.mark.parametrize("task_id", sorted(EXCLUDED))
def test_removed_pilot_record_cannot_reach_runtime(source_manifest, task_id):
    _, manifest = utils.load_data(Path("unused"))

    def reject_run(*args, **kwargs):
        pytest.fail("Excluded task reached the runtime")

    adapter = TauBankingAdapter(SimpleNamespace(run=reject_run), manifest)
    removed = next(record for record in source_manifest["records"] if record["task_id"] == task_id)
    with pytest.raises(ValueError, match="Unknown or changed tau record"):
        adapter.evaluate([removed], {"system_prompt": "Frozen pilot prompt"})


@pytest.mark.parametrize("worker_dataset", ["unfiltered", "v1", "v2", "v3", "v4"])
def test_retained_record_requires_matching_worker_manifest(
    source_manifest, historical_v1_manifest, historical_v2_manifest, historical_v3_manifest, worker_dataset
):
    splits, manifest = utils.load_data(Path("unused"))
    selected = splits["train"][:1]
    calls = []

    def run(records, candidate, trial):
        calls.append(deepcopy(records))
        return {
            "manifest_sha256": utils.digest(
                {
                    "unfiltered": source_manifest,
                    "v1": historical_v1_manifest,
                    "v2": historical_v2_manifest,
                    "v3": historical_v3_manifest,
                    "v4": manifest,
                }[worker_dataset]
            ),
            "outputs": [
                {
                    "id": record["id"],
                    "task_id": record["task_id"],
                    "trial": trial,
                    "seed": trial_seed(trial),
                    "candidate_sha256": utils.digest(candidate),
                    "elapsed_seconds": 1.0,
                    "reward": 0.0,
                    "termination_reason": "user_stop",
                    "error": None,
                    "messages": [{"role": "assistant", "content": "Synthetic boundary response"}],
                }
                for record in records
            ],
        }

    adapter = TauBankingAdapter(SimpleNamespace(run=run), manifest)
    if worker_dataset != "v4":
        with pytest.raises(ValueError, match="Worker ran another tau source or data revision"):
            adapter.evaluate(selected, {"system_prompt": "Fixed test prompt"})
    else:
        result = adapter.evaluate(selected, {"system_prompt": "Fixed test prompt"})
        assert result.scores == [0.0]
    assert calls == [selected]


@pytest.mark.parametrize("previous_dataset", ["unfiltered", "v1", "v2", "v3"])
def test_both_methods_share_retained_data_and_reject_legacy_resume(
    source_manifest,
    historical_v1_manifest,
    historical_v2_manifest,
    historical_v3_manifest,
    monkeypatch,
    tmp_path,
    previous_dataset,
):
    monkeypatch.setattr(main, "upstream_system_prompt", lambda source: "Fixed upstream system instructions")
    models = BenchmarkModels("unused/solver", "unused/proposer", None, None, {}, {})
    definitions = []
    for condition in ("vanilla", "forest"):
        args = SimpleNamespace(
            tau_source=tmp_path / "unused-source",
            run_dir=tmp_path / condition,
            template_family="alibaba",
            max_workers=1,
            condition=condition,
        )
        definition = main.build_benchmark(args, models)
        definitions.append(definition)
        assert definition.runtime["dataset_version"] == "tau-banking-ambiguity-exclusions-v4"
        assert definition.source["dataset_version"] == definition.runtime["dataset_version"]
        assert definition.runtime["manifest_sha256"] == utils.digest(definition.source)
        assert definition.runtime["exclusion_manifest_sha256"] == settings.EXCLUSIONS_MANIFEST_SHA256

        # Identical surviving prefixes must still reject the old dataset contract.
        subsets = {
            "trainset": definition.trainset[:1],
            "valset": definition.valset[:1],
            "testset": definition.testset[:1],
        }
        old_source = {
            "unfiltered": source_manifest,
            "v1": historical_v1_manifest,
            "v2": historical_v2_manifest,
            "v3": historical_v3_manifest,
        }[previous_dataset]
        previous = benchmark_data_identity(source=old_source, **subsets)
        current = benchmark_data_identity(source=definition.source, **subsets)
        assert previous["splits"] == current["splits"]
        assert previous != current
        old_runtime = {
            key: value
            for key, value in definition.runtime.items()
            if key not in {"dataset_version", "exclusion_manifest_sha256"}
        }
        old_runtime["manifest_sha256"] = utils.digest(old_source)
        if previous_dataset != "unfiltered":
            old_runtime["dataset_version"] = old_source["dataset_version"]
            old_runtime["exclusion_manifest_sha256"] = old_source["exclusion_manifest_sha256"]
        directory = tmp_path / condition
        ensure_contract(directory, "benchmark-run-contract.json", {"data": previous, "runtime": old_runtime})
        original = (directory / "benchmark-run-contract.json").read_bytes()
        with pytest.raises(ValueError, match="use a new run directory"):
            ensure_contract(directory, "benchmark-run-contract.json", {"data": current, "runtime": definition.runtime})
        assert (directory / "benchmark-run-contract.json").read_bytes() == original
    first, second = definitions
    assert first.source == second.source and first.runtime == second.runtime
    assert first.trainset == second.trainset and first.valset == second.valset and first.testset == second.testset
