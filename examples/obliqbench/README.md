# OBLIQ-Bench: one query rewrite and a fixed retriever

This example runs real retrieval over every official OBLIQ-Bench corpus. The
solver receives the query and a fixed task description, rewrites it once, and
the retriever returns 1,000 documents. GEPA and FOREST optimize the single
`query_rewriter_system_prompt` used in that solver call. The evaluator reads the
released relevance judgments after retrieval. It never supplies gold documents,
judgments, excluded IDs, or math solutions to the solver or retriever.

## Baseline choice and deviations

[The paper, version 2, §5](https://arxiv.org/html/2605.06235v2#S5) evaluates a
single GPT-5.2 query rewrite followed by Gemini-Embedding-2 cosine retrieval of
1,000 documents. We implement that simple architecture, without multiple
queries, extra hops, reranking, or oracle insertion. This is a local
reimplementation, **not an official upstream harness**: the paper and released
dataset do not provide executable baseline/evaluator code.

The following deliberate differences are fixed in the run identity:

- The solver is the shared `QWEN3_8_27B_MODEL`; the proposer is
  `DEEPSEEK_V4_1_FLASH_MODEL`. Their exact revisions, thinking budgets,
  sampling, and provider retries come from the shared experiment settings.
- The fixed retriever defaults to **Qwen3-Embedding-0.6B**, one of the paper's
  evaluated dense retrievers, rather than the Gemini service. This makes its
  weights independently reproducible. It is a separate embedding model, not a
  replacement solver or proposer. Revision:
  `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`.
- The seed is a short, general retrieval-rewriting instruction, rendered with
  the shared prompt template. It uses a strict single-field JSON output
  contract. Fixed task descriptions identify the five relevance relations.
  It is not a verbatim transcription of the paper's task-specific prompts.
- Dense search uses normalized float32 embeddings, exact cosine search, the
  checkpoint's fixed query prompt, no document prompt, left padding, and a
  32,768-token maximum (longer documents are truncated by the embedding
  tokenizer). No title/metadata or solution text is embedded. Corpus order
  breaks similarity ties. The model card documents the
  [query/document encoding distinction](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3).
- `--retriever bm25` provides the paper's `rank-bm25` lexical baseline with its
  default BM25Okapi parameters. The paper does not specify its tokenization;
  this implementation explicitly uses lowercase Unicode word tokens. The
  retriever remains fixed for both optimizer arms and their starting baseline.

## Data and metrics

The source is [dianetc/OBLIQ-Bench at the pinned revision](https://huggingface.co/datasets/dianetc/OBLIQ-Bench/tree/4ebee29f68ceeb62ca00bd73d5478acdd3cd7764).
`sources.json` records immutable upstream file hashes and lengths; Git files
use Git blob SHA-1, while LFS content uses SHA-256. `records.jsonl` records each
ordered query's full-content SHA-256, group, and partition. A record fingerprint
includes its query, complete gold and pooled judgments, and exclusion list.
Corpus bytes are verified before parsing, so their content and order are fixed.
Changed files, missing judgments, unknown document IDs, and changed manifests
fail before inference.

There are no official train/validation/test partitions. Our published local
partition assigns approximately 60/20/20 percent **within each subset**.
Queries sharing any positive gold/pooled document, excluded document, or exact
query text form connected groups. Whole groups are assigned largest-first to
the partition with the largest remaining target count; a SHA-256 of the group
IDs and fixed split seed 0 breaks ties. Query order within a partition follows
the original file. This prevents related authorship/problem/label groups from
crossing partitions. Optimization settings and optimizer random seeds never
resample the data. In particular, Twitter has one 183-query group, explaining
its departure from 60 percent training.

| Subset | Corpus documents | Queries | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: | ---: |
| Math | 3,508 | 151 | 91 | 30 | 30 |
| Writing | 10,389 | 512 | 306 | 103 | 103 |
| Twitter | 72,122 | 281 | 183 | 49 | 49 |
| WildChat | 507,729 | 40 | 24 | 8 | 8 |
| Congress | 213,650 | 254 | 152 | 51 | 51 |

The [official dataset card](https://huggingface.co/datasets/dianetc/OBLIQ-Bench/blob/4ebee29f68ceeb62ca00bd73d5478acdd3cd7764/README.md)
specifies NDCG@10/50 and Recall@10/50/100, separately for gold and, where
available, pooled judgments. This example uses the pinned
[`pytrec-eval-terrier`](https://github.com/terrierteam/pytrec_eval) implementation
of standard trec_eval metrics: linear graded gain for NDCG and positive
relevance for recall. Released grades 1 and 2 are preserved. Pooled judgments
are available only for Math, Twitter, and WildChat; they are not substituted
for missing annotations in the other subsets. The released per-query Math
and Writing masks are applied **before** taking the top 1,000 results.

The optimizer's scalar is gold NDCG@10, averaged equally over queries. Reports
also include all metrics per subset and the equal-subset macro gold NDCG@10.
The shared runner freezes the validation-selected winner before held-out
evaluation and reuses the unoptimized seed baseline only when the complete
data/runtime identity matches. There is one attempt per query, with no
best-of sampling. Test traces are rejected by the reflection interface.

## Run

Run from the repository root using Python 3.12. The `obliqbench` extra pins the
retriever/evaluator dependencies; exact versions are checked at startup.
The lexical reference below needs no embedding weights or model endpoint:

```bash
uv sync --locked --python 3.12 --extra dev --extra obliqbench
uv run --no-project python -m examples.obliqbench.prepare --subsets math
uv run --no-sync python -m examples.obliqbench.baseline \
  --subsets math --retriever bm25 --original-query-reference \
  --run-dir outputs/obliqbench-original-math
```

This optional original-query reference performs no rewrite and is separate
from the shared unoptimized **rewrite** baseline. The flag is accepted only
in baseline mode. It is never an optimization arm.

Prepare the full suite explicitly (about 2.9 GB of corpus JSONL; no weights
are downloaded by this command):

```bash
uv run --no-project python -m examples.obliqbench.prepare
# For only the small query/qrel/exclusion files:
uv run --no-project python -m examples.obliqbench.prepare --metadata-only
```

After installing the pinned dependencies and starting the exact shared solver
and proposer models, a training-only latency pilot and a comparison use:

```bash
uv run --no-sync python -m examples.obliqbench.pilot \
  --solver-api-base http://localhost:8000/v1 \
  --reflection-api-base http://localhost:8001/v1 \
  --embedding-device cuda --run-dir outputs/obliqbench
uv run --no-sync python -m examples.obliqbench.main \
  --condition both --solver-api-base http://localhost:8000/v1 \
  --reflection-api-base http://localhost:8001/v1 \
  --embedding-device cuda --run-dir outputs/obliqbench
```

`--subsets math` limits a smoke experiment to that official subset. Omitting
the argument includes all five. The shared `--train-limit`, `--val-limit`, and
`--test-limit` flags take prefixes only after the full dataset is frozen.
Episodes are sequential (`--max-workers 1`) so per-task latency is directly
interpretable. Each episode records wall time across rewriting, query
embedding, search, and metrics. The shared runner reports mean, median, p95,
count, failures, and throughput separately. Corpus indexing is offline
preprocessing and excluded from episode latency. Its hash-verified, memory-
mapped cache rejects incomplete indexes and runtime changes.

## Verification and limits

```bash
uv run --no-sync python -m pytest -q tests/test_obliqbench.py tests/test_benchmark_runner.py
```

Contract tests execute the real lexical/dense-ranking and trec_eval code,
substituting only model inference at external boundaries. They cover prompt
edits reaching retrieval, graded/pooled metrics, pre-ranking masks, group
separation, changed files, ordered-record drift, corrupt/incomplete embedding
caches, malformed rewrites, and held-out reflection rejection. The optional
local-data smoke test verifies the entire real Math corpus when prepared.

The initial integration was verified against all five subsets' query/qrel/mask
files and the complete Math corpus, including a no-model Math lexical
reference. No generative model calls, full-suite runs, embedding-weight
downloads, or Slurm jobs were launched. Actual Qwen embedding inference and
large-corpus indexing still require the pinned packages, weights, and adequate
hardware. The original-query reference and this adapted dense baseline should
not be presented as reproductions of the paper's GPT-5.2/Gemini scores.

Data credit: Diane Tchuindjo, Devavrat Shah, and Omar Khattab, *OBLIQ-Bench*,
2026, [arXiv:2605.06235v2](https://arxiv.org/abs/2605.06235v2). The dataset is
released under CC BY 4.0. This repository does not redistribute its corpus.
