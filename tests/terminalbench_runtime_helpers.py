"""Build deterministic serving identities for Terminal-Bench runtime tests."""

from examples.common.experiment_models import DEEPSEEK_V4_1_FLASH_MODEL, experiment_model_version
from examples.terminalbench import runtime


def runtime_fixture(model: str):
    """Describe an offline-only server without depending on the test machine's GPUs."""
    identity = {
        "policy": runtime.RUNTIME_POLICY,
        "model": model,
        "model_revision": experiment_model_version(model),
        "model_integrity_sha256": "a" * 64,
        "software": {
            "python": "3.12.8",
            "vllm": "0.1.1.dev5+ge77daef89" if model == DEEPSEEK_V4_1_FLASH_MODEL else "0.25.1",
            "torch": "2.10.0",
            "cuda": "13.0",
            "transformers": "5.0.0",
            "packages_sha256": "b" * 64,
        },
        "gpus": [
            {
                "name": "NVIDIA H200",
                "compute_capability": [9, 0],
                "memory_bytes": 141_000_000_000,
                "driver_version": "580",
            }
        ]
        * 8,
        "launch_arguments": [
            "--dtype",
            "auto",
            "--kv-cache-dtype",
            "fp8",
            "--tensor-parallel-size",
            "8",
            "--data-parallel-size",
            "1",
        ],
        "environment": {},
        "precision": {"dtype": "auto", "kv_cache_dtype": "fp8"},
        "parallelism": {"tensor": 8, "data": 1},
    }
    identity["sha256"] = runtime._digest(identity)
    return identity
