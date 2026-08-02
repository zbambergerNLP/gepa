# HotpotQA: Action-Conditioned Reflection Smoke Experiment

Small-scale evaluation (20 HotpotQA distractor examples, 14 train / 6 val) comparing vanilla GEPA against action-conditioned reflection with `VerbalizedActionSelector`. This was the first end-to-end exercise of the action-conditioned machinery (Rev 1); the IFBench example (`examples/ifbench/`) is the maintained, paper-faithful benchmark and supersedes this one for measuring effects.

```bash
uv run python examples/hotpotqa/main.py \
    --condition both --max-metric-calls 200 \
    --solver-model hosted_vllm/Qwen3.5-9B --api-base http://localhost:8000/v1
```

On della: `scripts/della/submit_hotpotqa.sh` (knobs: `MODEL`, `MAX_METRIC_CALLS`, `CONDITION`, `TIME`).

Metric is official HotpotQA token-F1 (`utils.py`). The action condition reports per-action proposal/acceptance stats via `ActionDiversityCallback`.
