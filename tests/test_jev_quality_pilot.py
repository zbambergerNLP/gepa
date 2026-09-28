"""Verify the paired pilot cannot expose transfer answers to proposal roles."""

from argparse import Namespace
from types import SimpleNamespace

from examples.hotpotqa import jev_quality_pilot as pilot


def test_matched_proposals_keep_transfer_out_of_reflection(tmp_path, monkeypatch):
    """Give both Controllers identical evidence and score every generated edit."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-only")
    train = [{"id": str(i), "answer": "a"} for i in range(150)]
    monkeypatch.setattr(pilot, "load_hotpotqa_dataset", lambda seed: (train, [], []))
    for name in ["_validate_scientific_data_identity", "_verify_scientific_retriever_integrity"]:
        monkeypatch.setattr(pilot, name, lambda *args: None)
    monkeypatch.setattr(pilot, "benchmark_data_identity", lambda **kw: {})
    monkeypatch.setattr(pilot, "source_identity", lambda path: {})
    monkeypatch.setattr(pilot, "build_run_contract", lambda *args: {})
    monkeypatch.setattr(pilot, "seed_candidate", lambda *args: dict.fromkeys(pilot.COMPONENTS, "original"))
    monkeypatch.setattr(pilot, "Wiki17BM25Retriever", lambda path: SimpleNamespace(provenance=lambda: {}))
    monkeypatch.setattr(pilot, "observed_kwargs", lambda *args: {})
    monkeypatch.setattr(pilot, "make_evaluator", lambda *args, **kw: None)
    calls = []
    evaluations = []

    def evaluate(directory, candidate, examples, evaluator, workers, *, compute_f1):
        assert compute_f1 is False
        evaluations.append([int(e["id"]) for e in examples])
        return [
            {
                "id": e["id"],
                "score": int(int(e["id"]) < 3),
                "feedback": {c + "_specific_info": {"id": e["id"]} for c in pilot.COMPONENTS},
            }
            for e in examples
        ]

    def config(condition, settings, kwargs, directory):
        arm = settings.controller_selection

        def reflect(parent, records, components):
            calls.append((arm, records))
            return SimpleNamespace(
                new_texts={components[0]: "changed"}, metadata={}, prompts=[], raw_lm_outputs=[]
            ), None

        return SimpleNamespace(reflection=SimpleNamespace(reflection_strategy=SimpleNamespace(reflect=reflect))), None

    monkeypatch.setattr(pilot, "evaluate_records", evaluate)
    monkeypatch.setattr(pilot, "build_config", config)
    result = pilot.run(
        Namespace(
            model="hosted_vllm/Qwen/Qwen3.8-27B",
            reflection_model="hosted_vllm/deepseek-ai/DeepSeek-V4.1-Flash",
            api_base="http://solver/v1",
            reflection_api_base="http://teacher/v1",
            wiki17_dir=tmp_path,
            output_dir=tmp_path / "run",
            workers=1,
        )
    )
    assert len(calls) == 14
    assert result["comparisons"][0]["perfect_batch_skip"]
    for first, second in zip(calls[::2], calls[1::2], strict=True):
        assert {first[0], second[0]} == {"jev", "verbalized"}
        assert first[1] == second[1]
        assert all(int(e["id"]) < 24 for records in first[1].values() for e in records)
    assert evaluations[0] == list(range(36))
    assert all(len(ids) == 15 and ids[3:] == list(range(24, 36)) for ids in evaluations[1:])
    assert result["execution_completed"] and not result["heldout_complete"]
