"""Execute the official grader in its isolated runtime after a patch is frozen."""

import argparse
import json
from pathlib import Path

import docker
from swebench.harness.run_evaluation import run_instance
from swebench.harness.test_spec.test_spec import make_test_spec


class PinnedSpec:
    """Keep official test scripts while selecting an already verified local image ID."""

    def __init__(self, spec, image_id):
        self.spec = spec
        self.instance_image_key = image_id

    def __getattr__(self, name):
        return getattr(self.spec, name)


class LocalImages:
    """Refuse implicit downloads during a recorded evaluation."""

    def __init__(self, images):
        self.images = images

    def get(self, image):
        return self.images.get(image)

    def pull(self, *args, **kwargs):
        raise RuntimeError("Task image disappeared; prepare it before starting another evaluation")


class BoundedContainers:
    """Apply the same local resources and network isolation as the solver container."""

    def __init__(self, containers, memory_gib):
        self.containers = containers
        self.memory_gib = memory_gib

    def create(self, **kwargs):
        return self.containers.create(
            **kwargs, nano_cpus=4_000_000_000, mem_limit=f"{self.memory_gib}g", network_disabled=True
        )


class LocalClient:
    """Restrict only image creation/download behavior, leaving official grading unchanged."""

    def __init__(self, client, memory_gib):
        self.client = client
        self.images = LocalImages(client.images)
        self.containers = BoundedContainers(client.containers, memory_gib)

    def __getattr__(self, name):
        return getattr(self.client, name)


def main():
    """Run one fresh official patch evaluation; never load or contact a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    record = json.loads(Path(request["evaluator_record"]).read_text())
    prediction = json.loads(Path(request["prediction"]).read_text())
    if record["instance_id"] != prediction["instance_id"]:
        raise ValueError("Evaluator/prediction task mismatch")
    client = docker.from_env()
    try:
        image = client.images.get(request["image_id"])
        if request["immutable"] not in image.attrs.get("RepoDigests", []) or image.attrs["Architecture"] != "amd64":
            raise ValueError("Grader image digest or architecture drift")
        spec = PinnedSpec(make_test_spec(record, namespace="starryzhang"), request["image_id"])
        result = run_instance(
            spec,
            prediction,
            False,
            False,
            LocalClient(client, request["memory_gib"]),
            "official",
            timeout=request["timeout_seconds"],
        )
        if result is None:
            raise RuntimeError("Official grader did not publish a complete result; inspect preserved native logs")
        report = result[1]
        if type(report[record["instance_id"]].get("resolved")) is not bool:
            raise ValueError("Malformed official resolution result")
        Path("official-result.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        client.close()


if __name__ == "__main__":
    main()
