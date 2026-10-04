# HotPotQA

`python -m examples.hotpotqa.main` uses the shared benchmark runner for vanilla
GEPA and FOREST. It preserves the existing GEPA artifact task program: two
Wiki-2017 BM25 retrieval hops, seven passages per hop, and four real DSPy
Chain-of-Thought predictors (`summarize1`, `create_query_hop2`, `summarize2`,
`final_answer`). Every rollout executes the current instructions for all four
components. Normalized answer exact match is the objective.

The shared defaults are the pinned Qwen3.8-27B solver and DeepSeek-V4.1-Flash
proposer, including their shared role budgets and request settings. The same
structured seed, retrieved evidence, scoring and component feedback apply to
both methods. Gold answers and supporting context enter scoring and training
feedback only; the solver receives the question, retrieved passages and its
own intermediate outputs.

## Frozen data and evaluation

The canonical data is `hotpot_qa/fullwiki` at revision
`1908d6afbbead072334abe2965f91bd2709910ab`. The loader retains the artifact's
ordered 40% test, 40% validation, 20% training partition of the original training
pool, followed by independent seed-1 sampling of 150 training, 300 validation and
300 held-out test questions. It verifies the existing ordered split hashes and
rejects duplicate question identities. The optimization seed does not resample
these splits. Full identities are recorded before shared prefix limits apply.

Training-only pilots measure complete episode wall time, including both
retrieval hops, all four DSPy calls and scoring. Optimization selects by
validation, freezes the winner, then evaluates held-out test once per question
and the shared unoptimized seed baseline. Only training trajectories can enter
reflection. Malformed task output scores zero; exhausted provider or runtime
failures abort instead of fabricating a complete result.

`--data-path` explicitly selects the existing JSONL smoke-data loader. Its file
hash is recorded separately from the canonical dataset. The shipped 20-row
sample gives 14/3/3 disjoint splits; smaller fixtures that reuse questions across
splits are rejected by the shared benchmark.

## Run

Use Python 3.12 or 3.13 and the pinned DSPy artifact fork. No model weights are
downloaded by the benchmark command.

```bash
uv sync --locked --extra dev --extra wiki17 --group hotpotqa-task-program --python 3.13
uv run --no-sync python -m examples.hotpotqa.main --help
```

A verified Wiki-2017 corpus/index is required. To prepare it separately on an
Internet-enabled machine (a large download):

```bash
uv run --no-sync python -m examples.common.wiki17_bm25 prepare --root /path/to/wiki17
```

The benchmark checks the corpus/index integrity manifest and preflights retrieval
with a training question before inference. A training-only pilot against the
configured local solver server is:

```bash
uv run --no-sync python -m examples.hotpotqa.main \
  --mode pilot --wiki17-dir /path/to/wiki17 \
  --solver-api-base http://localhost:8000/v1 \
  --reflection-api-base http://localhost:8001/v1 \
  --run-dir outputs/hotpotqa
```

Use `--mode optimize --condition both` for the paired methods or `--mode baseline`
for the shared seed baseline. Offline tests exercise the actual pinned DSPy
program with fixture completions and retrieval; model quality, server transport
and the full Wiki-2017 index still require a live pilot.

## Sources

- [GEPA artifact task program](https://github.com/gepa-ai/gepa-artifact/blob/a924c2045b6f000d2d23ea3b8f8f16b2c08d9e88/gepa_artifact/benchmarks/hotpotQA/hotpot_program.py)
- [GEPA artifact data split](https://github.com/gepa-ai/gepa-artifact/blob/a924c2045b6f000d2d23ea3b8f8f16b2c08d9e88/gepa_artifact/benchmarks/hotpotQA/hotpot_data.py)
- [Pinned DSPy fork](https://github.com/gepa-ai/dspy/tree/62dc3b634d7dc0c4889abcf905cb4c391ea6b396)
- [Pinned HotPotQA dataset](https://huggingface.co/datasets/hotpotqa/hotpot_qa/tree/1908d6afbbead072334abe2965f91bd2709910ab)
