"""Immutable source and evaluation contracts for DecisionBench."""

DATASET_REPO = "Hanno-Labs/decision-bench"
DATASET_REVISION = "071b7b2d2e1504c89e1e5a811a3f82e1bfe3aedb"
DATASET_FILE = "data/eval-00000-of-00001.parquet"
DATASET_SHA256 = "d9dd5be8e0944a7ceadfcac6cf8f9d17aab256093acff23c6de6a037ad6e55f6"
DATASET_ROWS = 23_900
UPSTREAM_REPO = "https://github.com/Hanno-Labs/decision-bench"
UPSTREAM_REVISION = "9a6328ee066a3dec0ed94ea9e57ddd3f63c57aa9"
UPSTREAM_FILES = {
    "data.py": "99384819a4a0c6c564f8aab35ce99560c84d16b445ef61afb7e2d1d89160128a",
    "schemas.py": "697abc646c1298c5c4a401e14808f42be863371dd999f3269ce4409c25b53d2b",
    "prompt.py": "a32fd1b95c66ae27c67650612c0ff9a7e7e5cc6e869ece7adcdc3a7bfa3c0b87",
    "scoring.py": "332c916dbae89f13d43469e8a9a3e179039ea82f6f041e085f7243239a486b40",
    "evaluate.py": "76e50c5c65630797647f23aef01e97c73e7333efe838fa624f203d054f8aad93",
}
SPLIT_POLICY = "source-connected-sha256-60-20-20-task-round-robin-v1"
SPLIT_SEED = 1
PINNED_SPLITS = {
    "train": {"count": 14331, "sha256": "ea9273d1fda0bb83b5e055d9eff86a3191ab4227b268a3676aee1ca187057aeb"},
    "val": {"count": 4752, "sha256": "fb34b5cd9971179166d2a4b3a90da248a94726df7580bb65244ca1f4d6c34a27"},
    "test": {"count": 4817, "sha256": "9c6af66c5ce38780e6963e6ba08dbea42b19128b41cb47c6791e5686b21ddd8b"},
}
ADAPTER_VERSION = "decisionbench-structured-chat-gepa-v1"
PROBABILITY_SOURCE = "schema_constrained_generated_probability_vector"
COMPONENT = "system_prompt"
