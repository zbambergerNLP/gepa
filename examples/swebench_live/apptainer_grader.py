"""Execute official SWE-bench scripts and scoring using the Apptainer transport."""

import argparse
import json
import subprocess
from pathlib import Path

from swebench.harness.constants import APPLY_PATCH_FAIL, LOG_TEST_OUTPUT, TESTS_TIMEOUT
from swebench.harness.grading import get_eval_report
from swebench.harness.run_evaluation import GIT_APPLY_CMDS
from swebench.harness.test_spec.test_spec import make_test_spec

from examples.swebench_live.apptainer import ApptainerTask


def grade(request: dict) -> dict:
    """Apply a frozen patch in a fresh task, then use the unchanged official scorer."""
    record = json.loads(Path(request["evaluator_record"]).read_text())
    prediction = json.loads(Path(request["prediction"]).read_text())
    if record["instance_id"] != prediction["instance_id"]:
        raise ValueError("Evaluator/prediction task mismatch")
    spec = make_test_spec(record, namespace="starryzhang")
    Path("eval.sh").write_text(spec.eval_script)
    with ApptainerTask(request["image"], request["memory_gib"]) as task:
        task.write_text("/tmp/patch.diff", prediction["model_patch"] or "")
        attempts = []
        for command in GIT_APPLY_CMDS:
            applied = task.execute(command + " /tmp/patch.diff")
            attempts.append({"command": command, **applied})
            if applied["exit_code"] == 0:
                break
        Path("patch-application.json").write_text(json.dumps(attempts, indent=2) + "\n")
        if applied["exit_code"]:
            raise RuntimeError(f"{APPLY_PATCH_FAIL}; inspect preserved patch-application.json")
        task.write_text("/eval.sh", spec.eval_script)
        try:
            result = task.execute("/bin/bash /eval.sh", timeout=request["timeout_seconds"], merge_output=True)
        except subprocess.TimeoutExpired as error:
            Path(LOG_TEST_OUTPUT).write_text(str(error.output or "") + "\n" + TESTS_TIMEOUT)
            raise RuntimeError("Official tests timed out; preserve this unresolved native attempt") from error
        Path(LOG_TEST_OUTPUT).write_text(result["stdout"] + result["stderr"])
        Path("test-exit.json").write_text(json.dumps({"exit_code": result["exit_code"]}) + "\n")
    report = get_eval_report(
        test_spec=spec, prediction=prediction, test_log_path=Path(LOG_TEST_OUTPUT), include_tests_status=True
    )
    if type(report[record["instance_id"]].get("resolved")) is not bool:
        raise ValueError("Malformed official resolution result")
    return report


def main():
    """Publish one official report without importing a model client or loading a key."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    report = grade(json.loads(args.request.read_text()))
    Path("official-result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
