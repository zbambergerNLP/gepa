"""Pin the text banking benchmark independently of optimizer settings."""

from pathlib import Path

UPSTREAM_REPOSITORY = "https://github.com/sierra-research/tau2-bench"
UPSTREAM_REVISION = "5bfa7e37b36656b37dc6d022156be6563c1007f3"
UPSTREAM_VERSION = "1.0.1"
DOMAIN = "banking_knowledge"
RETRIEVAL_CONFIG = "bm25"
RETRIEVAL_TOP_K = 10
MAX_STEPS = 200
MAX_ERRORS = 10
TEST_REPETITIONS = 4
TRIAL_SEED = 300
SPLIT_SALT = "gepa-tau-banking-group-split-v1"
MANIFEST_PATH = Path(__file__).with_name("banking-manifest.json")
DEFAULT_SOURCE = Path(__file__).parent / ".cache" / "tau2-bench"
# This upstream text import reaches a voice module that imports websockets.
RUNTIME_SUPPLEMENTS = ("websockets==15.0.1",)
PYTHON_VERSION = "3.12.11"
