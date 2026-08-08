# Attribution

## Dataset: GSM8K

- **GSM8K (Grade-School Math 8K)** — Cobbe et al., 2021. 8.5K grade-school math word problems requiring multi-step reasoning. Paper: [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168).
- **HuggingFace**: [`gsm8k`](https://huggingface.co/datasets/gsm8k) (original) / [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) (mirror), config `main`. Canonical split: train 7,473 / test 1,319 (paper notes ~7.5K train / ~1K test). Each example has `question` (word problem) and `answer` (step-by-step solution ending with `#### <number>`).
- **License**: MIT (via the HuggingFace dataset card; see the original GSM8K release).
- **Usage here**: loaded via `datasets.load_dataset("gsm8k", "main", split=...)` with fallback to `openai/gsm8k`; offline fallback at `examples/gsm8k/data/gsm8k.jsonl`. Deterministic shuffle seed 0 → 150 train / 300 val / 300 test (or 200/300/300 for headroom), mirroring the lightweight AIME/IFBench split pattern. No data files are committed.

## Papers referenced by this benchmark

- **GEPA** — Agrawal et al., 2025. *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning.* arXiv:2507.19457 (ICLR 2026 Oral). The harness in `main.py`/`utils.py` replicates GEPA's optimization loop (seed candidate, evaluator with `SideInfo`/`execution_feedback`, `EngineConfig`/`GEPAConfig`/`ReflectionConfig`, `ActionDiversityCallback`, parallel evaluation, candidate lineage dumps) as used in `examples/ifbench`, `examples/aime_math`, and `examples/pupa`.
- **CANTANTE** — *Optimizing Agentic Systems via Contrastive Credit Attribution*, arXiv:2605.13295 (May 2026). Reports GSM8K 61.27 → **82.33** (**+21.06 pp** vs GEPA, +12.53 vs MIPROv2 69.80) and MBPP 22.96 → **41.89** (+18.93 pp vs GEPA) with contrastive credit attribution. Cited in `docs/BENCHMARKS_GEPA_CITATIONS.md` as the largest GSM8K margin over GEPA and motivator for this scaffold.
- **VISTA** — *Reflection in the Dark: Exposing and Escaping the Black Box in Reflective Prompt Optimization → VISTA*, arXiv:2603.18388 (Mar 2026). Systematic analysis of GEPA/black-box reflective optimization on GSM8K (defective & clean seeds) and AIME20: GEPA on a defective seed **23.81% → 13.50%** (degradation), **VISTA recovers to 87.57%** (+74 pp vs degraded GEPA, +63 vs seed). The `DEFECTIVE_SEED_CANDIDATE` / `--defective-seed` flag and recovery-test framing in `utils.py`/`main.py` directly support VISTA-style experiments.

## Code provenance

- `utils.py` `_call_lm`, `run_gsm8k_single_stage`, `_strip_think`, `_extract_final_response`, and decoding config (temp 0.6, top_p 0.95, top_k 20, max 16384, `enable_thinking: False`, truncation to 24000 chars, `ContextWindowExceededError` retries stepping 16384→4096→1024→256, `<think>` stripping, `\boxed{}` extraction) are copied from `examples/aime_math/utils.py` and `examples/ifbench/utils.py` to keep solver behavior identical across benchmarks.
- `main.py` mirrors `examples/aime_math/main.py` and `examples/ifbench/main.py`: conditions `vanilla`/`random`/`action`/`all`, `seed_style` plain/structured, `actions` default/structured (`DEFAULT_ACTIONS` vs `build_structured_actions`), `ActionDiversityCallback`, `RandomActionSelector` / `VerbalizedActionSelector` branching, `--data-path` + `--train`/`--val`/`--test-limit`, dumps `candidates.json`/`action_summary.json`/`run_log.txt`/`candidate_tree.html`, final report with test accuracy and diversity.
- `run_gsm8k.sbatch` mirrors `examples/ifbench/run_ifbench.sbatch` and `examples/pupa/run_pupa.sbatch` (48h, vLLM serve + GEPA run, env vars `MODEL`/`CONDITION`/`SEED_STYLE`/`ACTIONS`/`MAX_METRIC_CALLS`/`TAG`, plus `DATA_PATH`/`DEFECTIVE_SEED` for GSM8K).
