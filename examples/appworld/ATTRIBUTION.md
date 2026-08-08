# Attribution

## Dataset: AppWorld

- **Paper**: Harsh Trivedi, Tushar Khot, Mareike Hartmann, Ruskin Manku, Vinty Dong, Edward Li, Shashank Gupta, Ashish Sabharwal, Niranjan Balasubramanian. *AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents*. ACL 2024 (arXiv:2407.18901). Project site: https://appworld.dev.
- **Benchmark**: 9 everyday apps (email, calendar, contacts, banking, shopping, etc.), 168 tool APIs, 750 tasks. Each task provides a natural-language instruction/goal and evaluation code (supervisor) that checks whether all subtasks pass — task goal completion (TGC) is 1 only when every check succeeds.
- **Data source**:
  - Primary: HuggingFace `appworld/appworld` (or mirrors under the `appworld-appworld` org), loaded via `datasets.load_dataset("appworld/appworld")` with pooling and deterministic seed-0 splits (60 train / 75 val / remaining test; 50/50/remaining via CLI limits). The 750 tasks are sliced deterministically; total budget and MIPRO-style harness follow GEPA paper conventions.
  - Fallback: local `examples/appworld/data/*.jsonl` or `*.json` (any `*.jsonl`/`*.json` files are pooled; for offline compute nodes). If neither HF nor local data is available, a small synthetic placeholder (40 tasks) keeps the harness importable/runnable for infra testing — real evaluation requires the AppWorld data.
- **Metric provenance**: `appworld_metric` in `utils.py` implements task success rate (all subtasks pass → 1, else 0), matching AppWorld's TGC definition. Offline, subtasks are read from `tests`/`eval`/`subtasks`/`checks`/`evaluation` lists or from `supervisor` presence, with non-empty fallback. Feedback lists passed/failed subtasks and the subtask pass rate for reflection. The evaluator shape (`SideInfo` with `score`, `query`, `output`, `execution_feedback`) mirrors `examples/ifbench/utils.py` and `examples/pupa/utils.py`.

## Code provenance

- `utils.py` `_call_lm` and decoding config (`temperature=0.6`, `top_p=0.95`, `top_k=20`, `max_tokens=16384`, `enable_thinking: False`, truncation retries on `ContextWindowExceededError`, `<think>` stripping, `reasoning_content` fallback) are copied from `examples/ifbench/utils.py` to keep solver behavior identical across benchmarks.
- `main.py` replicates the experiment harness pattern of `examples/ifbench/main.py` and `examples/pupa/main.py`: conditions `vanilla`/`random`/`action` (+`all`), programs `1stage`/`2stage` with `seed_style plain`/`structured`, actions `default`/`structured` via `DEFAULT_ACTIONS`/`build_structured_actions`, `ActionDiversityCallback`, `EngineConfig` + `GEPAConfig` + `ReflectionConfig` with `action_selector` branching (vanilla None, random `RandomActionSelector`, action `VerbalizedActionSelector` over `LM(args.reflection_model)`), `max_metric_calls`, `solver-model` via `hosted_vllm`, `dump candidates.json`/`action_summary.json`/`run_log.txt`/`candidate_tree.html`-compatible artifacts (`candidates.json` + `action_summary.json` written explicitly; `run_log.txt`/`candidate_tree.html` via `EngineConfig.run_dir` like the other runners), seed markdown skeleton, `prompt_diversity` (Jaccard), and CLI flags `--data-path` / `--train-limit` / `--val-limit` / `--test-limit`.
- `run_appworld.sbatch` mirrors `examples/ifbench/run_ifbench.sbatch` and `examples/pupa/run_pupa.sbatch`: vLLM serve + GEPA run, 48h, env vars `MODEL`/`CONDITION`/`PROGRAM`/`SEED_STYLE`/`ACTIONS`/`MAX_METRIC_CALLS`/`TAG`, plus `DATA_PATH` for AppWorld. Pool-aware port, scratch caching, and health-check logic are preserved.

## License

- AppWorld data and code are released under the licenses documented at https://appworld.dev and the AppWorld GitHub repository (https://github.com/stonybrooknlp/appworld). Consult those sources for the governing dataset and code licenses before redistributing data.
- GEPA (this repository): see root `LICENSE`.
- Citations:
  - Trivedi et al. 2024. *AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents*. https://arxiv.org/abs/2407.18901
  - GEPA paper (Agrawal et al., arXiv:2507.19457) for the MIPRO-style optimization harness and action-conditioned reflection design.
