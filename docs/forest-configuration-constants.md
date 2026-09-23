# FOREST configuration constants

Repeated configuration values have named owners shared by FOREST's roles and
benchmark entry points. CLI and environment overrides retain their existing
precedence. This refactor does not change any default or scientific setting.

| Configuration | Owner |
| --- | --- |
| Provider retries, backoff, response-error codes, usage artifact names | `src/gepa/lm_constants.py` |
| Controller selection, Editor modes, tool sets, role names, reflection defaults | `src/gepa/strategies/forest_constants.py` |
| Action sampling support and verbalized candidate count | `src/gepa/strategies/action_space.py` |
| Model identities, revisions, decoding, context capacity and provider effort | `examples/common/experiment_models.py` |
| HotPotQA role output/thinking budgets and request timeout | `examples/hotpotqa/model_settings.py` |
| HotPotQA optimization budgets, ordered split sizes and retrieval defaults | `examples/hotpotqa/benchmark_settings.py` |
| Terminal-Bench role output budget | `examples/terminalbench/model_settings.py` |
| Shared Bash launcher defaults | `examples/common/launcher_constants.py` |

Settings with different meanings remain independent, even when their values
match. In particular, thinking tokens, combined output tokens, server batch
tokens, metric evaluations and physical request attempts are separate limits.
Terminal-Bench's output budget is independent of HotPotQA's role budgets.
Defaults specific to one runtime, adapter or pilot stay near that implementation.

The shell launchers source `scripts/della/runtime_constants.sh`. Generate it
from its Python owner; do not edit the generated file:

```bash
uv run python -m examples.common.launcher_constants > scripts/della/runtime_constants.sh
```

A test checks both generated-file parity and actual Bash values. Existing
protocol tests keep explicit expected values to detect unintended behavior
changes. Runtime identities include the constants module when an isolated
qualification worker shares LM dispatch with a different strategy revision.

These files belong to each immutable source revision. Changing a constant does
not update an existing allocation, sealed export, checkpoint or qualification.
