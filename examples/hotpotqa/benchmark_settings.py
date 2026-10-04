"""Fixed HotPotQA evaluation budgets, ordered split sizes, and retrieval settings."""

from pathlib import Path

from examples.common.benchmark_settings import DEFAULT_MAX_METRIC_CALLS as STANDARD_METRIC_CALLS

EXPANDED_METRIC_CALLS = 2 * STANDARD_METRIC_CALLS
TRAIN_SIZE = 150
VALIDATION_SIZE = 300
TEST_SIZE = 300
TRAIN_VALIDATION_SIZE = TRAIN_SIZE + VALIDATION_SIZE
DATASET_SIZE = TRAIN_VALIDATION_SIZE + TEST_SIZE
SPLIT_COUNTS = {"train": TRAIN_SIZE, "val": VALIDATION_SIZE, "test": TEST_SIZE}
DATASET_SAMPLE_SEED = 1
RETRIEVAL_K = 7
DEFAULT_MAX_WORKERS = 32
HOTPOTQA_DSPY_VERSION = "2.6.23"
HOTPOTQA_DSPY_COMMIT = "62dc3b634d7dc0c4889abcf905cb4c391ea6b396"
HOTPOTQA_HF_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
HOTPOTQA_SCIENTIFIC_SPLIT_SHA256 = {
    "train": "0287a62f31caa939df13d9e176436293a5c071164728bcb3cfcbf8dd40a7918e",
    "val": "c6d794b172724eb87e8087d671b74e94725040b79c9a63980cb45dcb53146408",
    "test": "55cd1c7a999476ea4c7ec67f964ad4fa0ae662a2b9f7ade59c64108e659add31",
}
DEFAULT_DATA_PATH = str(Path(__file__).with_name("data") / "hotpotqa_distractor_sample.jsonl")
FINAL_RESPONSE_MARKER = "Final Response:"
HOTPOTQA_SHARED_HARNESS = "artifact-two-hop-four-component-shared-v1"
TEST_REPETITIONS = 1
SEED_CANDIDATE = {
    "summarize1": "Given the fields `question`, `passages`, produce the fields `summary`.",
    "create_query_hop2": "Given the fields `question`, `summary_1`, produce the fields `query`.",
    "summarize2": "Given the fields `question`, `context`, `passages`, produce the fields `summary`.",
    "final_answer": "Given the fields `question`, `summary_1`, `summary_2`, produce the fields `answer`.",
}
SEED_CANDIDATE_1STAGE = {
    "answer_question": (
        "Answer the question from the retrieved passages. Ignore irrelevant passages and "
        "combine evidence across pages. Return a concise answer."
    ),
}
