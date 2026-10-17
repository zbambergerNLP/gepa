"""Exercise the real SWE-agent tools on one training task without a paid model."""

import argparse
import json
import os
import socket
from pathlib import Path

import litellm

from examples.common.artifacts import atomic_json
from examples.common.benchmark_types import BenchmarkModels
from examples.swebench_live.adapter import grade
from examples.swebench_live.apptainer_images import verify_images
from examples.swebench_live.data import load_data
from examples.swebench_live.swe_agent import activate, rollout, runtime_identity, seed_prompt


def main():
    """Use fixed synthetic tool calls, then run official negative/reference grading."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--agent-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    activate(args.agent_root)
    from sweagent.agent.models import PredeterminedTestModel

    def forbid_provider(*_, **__):
        raise AssertionError("Native readiness must never call a model provider")

    litellm.completion = forbid_provider
    splits, _ = load_data(args.assets)
    record = next(row for row in splits["train"] if row["id"] == "patroni__patroni-2998")
    image = verify_images(args.assets, [record])[record["id"]]
    models = BenchmarkModels("openrouter/offline-check", "unused", None, None, {}, {})
    atomic_json(args.output / "configuration.json", runtime_identity(args.agent_root, models))

    def call(name, arguments):
        return {
            "message": "Offline infrastructure check",
            "tool_calls": [
                {
                    "id": "probe-" + str(len(calls)),
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        }

    calls = []
    calls.append(
        call(
            "bash",
            {
                "command": 'export PERSISTENT_PROBE=preserved; test ! -d /scratch/gpfs/BSTEWART; test -z "$OPENROUTER_API_KEY"'
            },
        )
    )
    calls.append(call("bash", {"command": "test \"$PERSISTENT_PROBE\" = preserved && printf 'persistent-shell-ok\\n'"}))
    calls.append(
        call(
            "str_replace_editor", {"command": "create", "path": "/testbed/swe_agent_probe.txt", "file_text": "before\n"}
        )
    )
    calls.append(
        call(
            "str_replace_editor",
            {"command": "str_replace", "path": "/testbed/swe_agent_probe.txt", "old_str": "before", "new_str": "after"},
        )
    )
    calls.append(call("submit", {}))
    calls.append(call("submit", {}))
    trial = args.output / "solver"
    trial.mkdir()
    result = rollout(
        args.agent_root,
        record,
        image,
        models,
        seed_prompt(args.agent_root),
        trial,
        16,
        test_model=PredeterminedTestModel(calls),
    )
    assert result["info"]["exit_status"] == "submitted", result["info"]
    assert "+after" in result["patch"] and "swe_agent_probe.txt" in result["patch"], result["patch"]
    trajectory = result["trajectory"]["trajectory"]
    assert len(trajectory) == 6, trajectory
    assert "persistent-shell-ok" in trajectory[1]["observation"]
    assert "Thank you for your work" in trajectory[4]["observation"]
    atomic_json(
        trial / "prediction.json",
        {"instance_id": record["id"], "model_name_or_path": "offline-native-check", "model_patch": result["patch"]},
    )
    raw = next(
        json.loads(line)
        for line in (args.assets / "data/train.jsonl").read_text().splitlines()
        if json.loads(line)["instance_id"] == record["id"]
    )
    atomic_json(trial / "evaluator-record.json", raw)
    negative = grade(args.assets, trial, record, image, 16, 1800)
    assert negative[record["id"]]["resolved"] is False
    reference = args.output / "reference"
    reference.mkdir()
    atomic_json(reference / "evaluator-record.json", raw)
    atomic_json(
        reference / "prediction.json",
        {"instance_id": record["id"], "model_name_or_path": "training-reference", "model_patch": raw["patch"]},
    )
    positive = grade(args.assets, reference, record, image, 16, 1800)
    assert positive[record["id"]]["resolved"] is True
    atomic_json(
        args.output / "complete.json",
        {
            "job_id": os.environ["SLURM_JOB_ID"],
            "node": socket.gethostname(),
            "instance_id": record["id"],
            "real_agent": "SWE-agent DefaultAgent",
            "scripted_model_steps": 6,
            "paid_model_calls": 0,
            "negative_resolved": False,
            "reference_resolved": True,
            "all_task_readiness": False,
            "optimization_started": False,
        },
    )


if __name__ == "__main__":
    main()
