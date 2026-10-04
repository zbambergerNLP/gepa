"""Keep Terminal-Bench's fixed protocol separate from shared experiment control."""

from pathlib import Path

MANIFEST_PATH = Path(__file__).with_name("terminalbench-v2.1-manifest.json")
TRAINING_EPOCHS_BY_BUDGET = {"standard": 4, "double": 8}
TEST_REPETITIONS = 3
