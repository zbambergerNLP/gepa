"""Keep HotPotQA stopping at complete GEPA iterations."""

from collections import Counter

from examples.hotpotqa.main import EngineConfig, GEPAConfig, ReflectionConfig, run_condition


def test_crossing_budget_finishes_validation_and_preserves_the_winner(tmp_path):
    """A threshold inside validation must not discard the last complete proposal."""
    calls = []

    def evaluate(candidate, example):
        calls.append((candidate, example))
        return (0.25 if candidate == "seed" else 0.75), {"feedback": "Improve the answer."}

    def propose(_messages):
        return "```\nimproved candidate\n```"

    result = run_condition(
        "budget boundary regression",
        "seed",
        ["train-a", "train-b", "train-c"],
        ["val-a", "val-b", "val-c", "val-d"],
        GEPAConfig(
            engine=EngineConfig(max_metric_calls=12, run_dir=str(tmp_path), parallel=True, max_workers=4),
            reflection=ReflectionConfig(reflection_lm=propose, reflection_minibatch_size=3),
        ),
        evaluate,
    )

    assert result.total_metric_calls == 14
    assert len(calls) == 14
    assert result.val_aggregate_scores == [0.25, 0.75]
    assert result.best_candidate == "improved candidate"
    assert Counter(example for candidate, example in calls if candidate != "seed" and example.startswith("val")) == {
        "val-a": 1, "val-b": 1, "val-c": 1, "val-d": 1,
    }
