"""Run one upstream SWE-agent rollout, then score its frozen patch with upstream tests."""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from pathlib import Path

from examples.common.artifacts import atomic_json, digest
from examples.common.react_v2 import file_sha256
from examples.swebench_live.settings import COMPONENT
from examples.swebench_live.swe_agent import rollout
from gepa.core.adapter import EvaluationBatch


def grade(root: Path, trial: Path, record: dict, image: dict, memory_gib: int, timeout: int) -> dict:
    """Pass sealed predictions to the official evaluator in a separate fresh container."""
    request = {
        "evaluator_record": str(trial / "evaluator-record.json"),
        "prediction": str(trial / "prediction.json"),
        "image_id": image["image_id"],
        "immutable": image["immutable"],
        "memory_gib": memory_gib,
        "timeout_seconds": timeout,
    }
    apptainer = image.get("backend") == "apptainer"
    worker = [str(Path(__file__).with_name("grader_worker.py"))]
    env = None
    if apptainer:
        request["image"] = image
        worker = ["-m", "examples.swebench_live.apptainer_grader"]
        env = {
            key: value
            for key, value in os.environ.items()
            if key in {"PATH", "HOME", "TMPDIR", "LANG"} or key.startswith("SLURM_")
        }
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    atomic_json(trial / "grader-request.json", request)
    with (trial / "grader.log").open("w") as log:
        subprocess.run(
            [
                "uv",
                "--no-cache",
                "run",
                "--no-project",
                "--python",
                str(root / "venv" / "bin" / "python"),
                "python",
                *worker,
                str(trial / "grader-request.json"),
            ],
            cwd=trial,
            env=env,
            check=True,
            stdout=log,
            stderr=log,
            timeout=timeout + 600,
        )
    report = json.loads((trial / "official-result.json").read_text())
    if set(report) != {record["id"]} or type(report[record["id"]].get("resolved")) is not bool:
        raise ValueError("Official SWE-bench Live report identity/score mismatch")
    return report


class SWEBenchLiveAdapter:
    """Expose a shared prompt to GEPA using official test resolution as its metric."""

    propose_new_texts = None

    def __init__(self, models, root, run_dir, records, images, *, agent_root=None, memory_gib=16, test_timeout=1800):
        self.models = models
        self.root = root.resolve()
        self.run_dir = run_dir.resolve()
        self.records = {record["id"]: record for record in records}
        self.images = images
        self.agent_root = agent_root or root / "swe-agent"
        self.memory_gib = memory_gib
        self.test_timeout = test_timeout
        self.rollout = rollout
        self.grader = grade
        self.evaluator_rows = {}
        for split in ("train", "val", "test"):
            for line in (root / "data" / f"{split}.jsonl").read_text().splitlines():
                row = json.loads(line)
                self.evaluator_rows[row["instance_id"]] = row

    def _one(self, record, candidate):
        started = time.perf_counter()
        if (
            self.records.get(record["id"]) != record
            or digest(self.evaluator_rows[record["id"]]) != record["evaluator_sha256"]
        ):
            raise ValueError("SWE-bench Live task identity drift")
        trial = self.run_dir / "native-trials" / uuid.uuid4().hex
        trial.mkdir(parents=True, exist_ok=False)
        messages = {}
        output = {"id": record["id"], "prompt_sha256": digest(candidate), "error": None, "trial_dir": str(trial)}
        atomic_json(trial / "intent.json", {**output, "state": "started", "image": self.images[record["id"]]})
        try:
            result = self.rollout(
                self.agent_root,
                record,
                self.images[record["id"]],
                self.models,
                candidate[COMPONENT],
                trial,
                self.memory_gib,
            )
            messages = result["trajectory"]
            output["swe_agent_info"] = result["info"]
            patch = result["patch"]
            prediction = {"instance_id": record["id"], "model_name_or_path": "solver", "model_patch": patch}
            atomic_json(trial / "prediction.json", prediction)
            output["prediction_sha256"] = file_sha256(trial / "prediction.json")
            if output["error"]:
                output["resolved"] = False
            elif not patch.strip():
                output.update(resolved=False, error="empty_patch")
            else:
                # The solver container is gone before any gold test data is made available.
                atomic_json(trial / "evaluator-record.json", self.evaluator_rows[record["id"]])
                report = self.grader(
                    self.root, trial, record, self.images[record["id"]], self.memory_gib, self.test_timeout
                )
                output.update(
                    resolved=report[record["id"]]["resolved"],
                    report=report,
                    report_sha256=file_sha256(trial / "official-result.json"),
                )
            output["elapsed_seconds"] = time.perf_counter() - started
            atomic_json(trial / "result.json", output)
            return output
        finally:
            if messages or not (trial / "trajectory.json").exists():
                atomic_json(trial / "trajectory.json", messages)

    def evaluate(self, batch, candidate, capture_traces=False):
        """Run fresh tasks sequentially; propagate infrastructure failures without fabricated scores."""
        if (
            set(candidate) != {COMPONENT}
            or not isinstance(candidate[COMPONENT], str)
            or not candidate[COMPONENT].strip()
        ):
            raise ValueError("SWE-bench Live requires one nonempty system_prompt")
        outputs = [self._one(record, candidate) for record in batch]
        return EvaluationBatch(
            outputs=outputs,
            scores=[float(row["resolved"]) for row in outputs],
            trajectories=[{"record": record, "output": output} for record, output in zip(batch, outputs, strict=True)]
            if capture_traces
            else None,
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        """Provide training-only rollout feedback; never supply held-out tasks to the proposer."""
        if components_to_update != [COMPONENT] or eval_batch.trajectories is None:
            raise ValueError("Reflection requires system_prompt training traces")
        feedback = []
        for trajectory in eval_batch.trajectories:
            record, output = trajectory["record"], trajectory["output"]
            if (
                self.records.get(record["id"]) != record
                or record["split"] != "train"
                or output["prompt_sha256"] != digest(candidate)
            ):
                raise ValueError("Reflection requires matching training records and prompt")
            feedback.append(
                {
                    "Inputs": record["problem_statement"],
                    "Generated Outputs": json.loads((Path(output["trial_dir"]) / "trajectory.json").read_text()),
                    "Feedback": {"resolved": output["resolved"], "error": output["error"]},
                }
            )
        return {COMPONENT: feedback}
