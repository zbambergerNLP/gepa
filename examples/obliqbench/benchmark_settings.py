"""Pin the data, retrieval program, and evaluation semantics independently of GEPA."""

DATASET = "dianetc/OBLIQ-Bench"
DATASET_REVISION = "4ebee29f68ceeb62ca00bd73d5478acdd3cd7764"
HARNESS_VERSION = "obliq-single-rewrite-v2"
SUBSETS = {
    "math": "analogues/math",
    "writing": "analogues/writing",
    "twitter": "descriptive/twitter",
    "wildchat": "descriptive/wildchat",
    "congress": "tip-of-tongue/congress",
}
QUERY_COUNTS = {"math": 151, "writing": 512, "twitter": 281, "wildchat": 40, "congress": 254}
CORPUS_COUNTS = {"math": 3508, "writing": 10389, "twitter": 72122, "wildchat": 507729, "congress": 213650}
POOLED_SUBSETS = frozenset({"math", "twitter", "wildchat"})
EXCLUSION_SUBSETS = frozenset({"math", "writing"})
SPLIT_POLICY = "sha256-v1-connected-positive-and-excluded-documents-60-20-20"
SPLIT_SEED = 0
RETRIEVAL_K = 1000
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
EMBEDDING_DIMENSION = 1024
EMBEDDING_MAX_LENGTH = 32768
RUNTIME_PINS = {
    "pytrec-eval-terrier": "0.5.10",
    "numpy": "2.2.6",
    "rank-bm25": "0.2.2",
    "sentence-transformers": "6.1.0",
    "transformers": "5.17.0",
    "torch": "2.9.0",
}
TASK_DESCRIPTIONS = {
    "math": "Retrieve mathematical problems that share the query's underlying reasoning technique, across topics.",
    "writing": "Retrieve prose by the same author as the supplied snippet, based on writing style across topics.",
    "twitter": "Retrieve tweets whose implicit stance matches the requested description.",
    "wildchat": "Retrieve user-assistant conversations exhibiting the requested behavioral failure.",
    "congress": "Retrieve the congressional hearing passage matching the user's imperfect recollection.",
}
SEED_INSTRUCTION = (
    "Rewrite the supplied retrieval query once to help a fixed retriever find relevant documents. "
    "Use the task description to preserve the intended relevance relation. "
    "Return only a JSON object with exactly one field, query, containing the rewritten search text. "
    "Do not answer the query, invent document identifiers, or output multiple search queries."
)
METRIC_NAME = "gold_ndcg_at_10"
