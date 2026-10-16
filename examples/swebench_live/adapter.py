"""Run one isolated shell-agent rollout, then score its frozen patch with upstream tests."""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from copy import deepcopy
from pathlib import Path

import litellm
from litellm.exceptions import ContextWindowExceededError

from examples.common.artifacts import atomic_json, digest
from examples.common.react_v2 import file_sha256
from examples.swebench_live.settings import COMPONENT
from gepa.core.adapter import EvaluationBatch

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a shell command inside the task repository's isolated Linux container.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    }
]


class DockerTask:
    """Own one disposable container, without host mounts, credentials or networking."""

    def __init__(self, image_id: str, memory_gib: int):
        self.image_id = image_id
        self.memory_gib = memory_gib
        self.name = "gepa-swe-live-" + uuid.uuid4().hex

    def __enter__(self):
        subprocess.run(
            [
                "docker",
                "create",
                "--pull",
                "never",
                "--name",
                self.name,
                "--platform",
                "linux/amd64",
                "--network",
                "none",
                "--cpus",
                "4",
                "--memory",
                f"{self.memory_gib}g",
                "--workdir",
                "/testbed",
                self.image_id,
                "tail",
                "-f",
                "/dev/null",
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        try:
            subprocess.run(["docker", "start", self.name], check=True, capture_output=True, timeout=120)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def execute(self, command: str, timeout: int = 120) -> dict:
        """Execute only inside this task container and retain complete output."""
        result = subprocess.run(
            ["docker", "exec", "--workdir", "/testbed", self.name, "bash", "-lc", command],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
        )
        return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}

    def __exit__(self, *_):
        subprocess.run(["docker", "rm", "--force", self.name], check=True, capture_output=True, timeout=120)


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
                str(Path(__file__).with_name("grader_worker.py")),
                str(trial / "grader-request.json"),
            ],
            cwd=trial,
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

    def __init__(self, models, root, run_dir, records, images, *, max_steps=100, memory_gib=16, test_timeout=1800):
        self.models = models
        self.root = root.resolve()
        self.run_dir = run_dir.resolve()
        self.records = {record["id"]: record for record in records}
        self.images = images
        self.max_steps = max_steps
        self.memory_gib = memory_gib
        self.test_timeout = test_timeout
        self.completion = None
        self.task_factory = DockerTask
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
        messages = [
            {"role": "system", "content": candidate[COMPONENT]},
            {"role": "user", "content": record["problem_statement"]},
        ]
        output = {"id": record["id"], "prompt_sha256": digest(candidate), "error": None, "trial_dir": str(trial)}
        atomic_json(trial / "intent.json", {**output, "state": "started", "image": self.images[record["id"]]})
        try:
            with self.task_factory(self.images[record["id"]]["image_id"], self.memory_gib) as task:
                for _ in range(self.max_steps):
                    atomic_json(trial / "trajectory.json", messages)
                    request = {
                        **deepcopy(self.models.solver_kwargs),
                        "model": self.models.solver_model,
                        "messages": deepcopy(messages),
                        "tools": TOOLS,
                        "drop_params": False,
                    }
                    if self.models.solver_api_base:
                        request["api_base"] = self.models.solver_api_base
                    try:
                        response = (self.completion or litellm.completion)(**request)
                    except ContextWindowExceededError:
                        output["error"] = "context_limit_exceeded_no_truncation"
                        break
                    message = response.choices[0].message.model_dump(exclude_none=True)
                    messages.append(message)
                    calls = message.get("tool_calls", [])
                    if not calls:
                        break
                    for call in calls:
                        try:
                            arguments = json.loads(call["function"]["arguments"])
                            if (
                                call["function"]["name"] != "shell"
                                or set(arguments) != {"command"}
                                or not isinstance(arguments["command"], str)
                            ):
                                raise ValueError("Expected shell(command: string)")
                        except (ValueError, TypeError) as error:
                            result = {"error": str(error)}
                        else:
                            result = task.execute(arguments["command"])
                        messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
                        atomic_json(trial / "trajectory.json", messages)
                patch = task.execute("git --no-pager diff HEAD --text")
                if patch["exit_code"] != 0:
                    raise RuntimeError("Could not extract the solver patch")
            prediction = {"instance_id": record["id"], "model_name_or_path": "solver", "model_patch": patch["stdout"]}
            atomic_json(trial / "prediction.json", prediction)
            output["prediction_sha256"] = file_sha256(trial / "prediction.json")
            if output["error"]:
                output["resolved"] = False
            elif not patch["stdout"].strip():
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
