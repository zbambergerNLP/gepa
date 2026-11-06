"""Verify the approved Tau exclusions without episodes, model calls, or task-content review."""

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

EXCLUDED = {"task_075", "task_084"}


@pytest.fixture
def source_manifest(monkeypatch):
    manifest = json.loads(settings.MANIFEST_PATH.read_text())
    monkeypatch.setattr(utils, "inspect_source", lambda source: deepcopy(manifest))
    return manifest


def test_only_approved_tasks_are_removed_without_resplitting(source_manifest):
    before = deepcopy(source_manifest)
    splits, manifest = utils.load_data(Path("unused"))
    retained = [record for record in before["records"] if record["task_id"] not in EXCLUDED]

    assert len(manifest["records"]) == 95
    assert manifest["records"] == retained
    assert {name: len(rows) for name, rows in splits.items()} == {"train": 55, "val": 19, "test": 21}
    for name, rows in splits.items():
        assert rows == [record for record in retained if record["split"] == name]
    assert source_manifest == before == json.loads(settings.MANIFEST_PATH.read_text())


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


@pytest.mark.parametrize("legacy_worker", [False, True])
def test_retained_record_requires_matching_worker_manifest(source_manifest, legacy_worker):
    splits, manifest = utils.load_data(Path("unused"))
    selected = splits["train"][:1]
    calls = []

    def run(records, candidate, trial):
        calls.append(deepcopy(records))
        return {
            "manifest_sha256": utils.digest(source_manifest if legacy_worker else manifest),
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
    if legacy_worker:
        with pytest.raises(ValueError, match="Worker ran another tau source or data revision"):
            adapter.evaluate(selected, {"system_prompt": "Fixed test prompt"})
    else:
        result = adapter.evaluate(selected, {"system_prompt": "Fixed test prompt"})
        assert result.scores == [0.0]
    assert calls == [selected]


def test_both_methods_share_retained_data_and_reject_legacy_resume(source_manifest, monkeypatch, tmp_path):
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
        assert definition.runtime["dataset_version"] == "tau-banking-ambiguity-exclusions-v1"
        assert definition.source["dataset_version"] == definition.runtime["dataset_version"]
        assert definition.runtime["manifest_sha256"] == utils.digest(definition.source)
        assert definition.runtime["exclusion_manifest_sha256"] == settings.EXCLUSIONS_MANIFEST_SHA256

        # Identical surviving prefixes must still reject the old dataset contract.
        subsets = {
            "trainset": definition.trainset[:1],
            "valset": definition.valset[:1],
            "testset": definition.testset[:1],
        }
        previous = benchmark_data_identity(source=source_manifest, **subsets)
        current = benchmark_data_identity(source=definition.source, **subsets)
        assert previous["splits"] == current["splits"]
        assert previous != current
        old_runtime = {
            key: value
            for key, value in definition.runtime.items()
            if key not in {"dataset_version", "exclusion_manifest_sha256"}
        }
        old_runtime["manifest_sha256"] = utils.digest(source_manifest)
        directory = tmp_path / condition
        ensure_contract(directory, "benchmark-run-contract.json", {"data": previous, "runtime": old_runtime})
        original = (directory / "benchmark-run-contract.json").read_bytes()
        with pytest.raises(ValueError, match="use a new run directory"):
            ensure_contract(directory, "benchmark-run-contract.json", {"data": current, "runtime": definition.runtime})
        assert (directory / "benchmark-run-contract.json").read_bytes() == original
    first, second = definitions
    assert first.source == second.source and first.runtime == second.runtime
    assert first.trainset == second.trainset and first.valset == second.valset and first.testset == second.testset
