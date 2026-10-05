---
name: della
description: >-
  Work with the repository's Della preparation and serving tools: SSH, setup,
  pinned checkpoints, data preparation, independent serving diagnostics, and
  shared benchmark execution. Read before remote Della operations.
---

# Working with Della

Use `scripts/della/README.md` for preparation and `examples/BENCHMARKS.md` for
benchmark execution. All six benchmark entrypoints use the shared runner.

## Scope

- Local implementation and review do not authorize remote operations. Before
  SSH, sync, downloads, builds, or job submissions, check the user's existing
  authorization; ask with concrete commands if that authorization is missing.
- Use the existing SSH configuration and repository preparation scripts.
  `scripts/della/.env` is ignored, user-owned, mode 600, and must stay out of logs.
- Use the login host for brief checks and the visualization host for downloads
  and builds. Run model serving and evaluation only on allocated compute nodes.
- Preserve remote outputs, caches, environments, checkpoint bytes, and running
  jobs. Changes to source or material runtime settings require a fresh run
  directory; never reinterpret an incompatible checkpoint.

## Preparation

1. Inspect the source branch and working tree, then check shared SSH sessions
   with `scripts/della/della_session.sh status` when remote work is authorized.
2. Run `scripts/della/preflight_hotpotqa.sh` for source, SSH, CUDA, storage, and
   serving-lock checks. Its clean-source policy uses the current committed HEAD.
3. Use `scripts/della/build_env.sh` for environments, data, and pinned models.
   Detached downloads must reach `MODELS_DONE`; verify model manifests before
   serving. Check current storage availability before large downloads.
4. Use the independent `submit_deepseek_smoke.sh`/`verify_deepseek_serving.sh`
   diagnostic when serving verification is requested. A submitted job is not a
   passed diagnostic; inspect its completion and recorded result.

## Evaluation

Use `uv` and the benchmark's documented runtime prerequisites. Default roles are
Qwen3.8-27B solver and DeepSeek-V4.1-Flash proposer. Terminal-Bench also requires
pinned Harbor and current records from `examples.terminalbench.runtime`. Princeton
does not allow Docker on its clusters; use the official Harbor `singularity`
backend with Apptainer on Della. Prepare it on the visualization host with
`scripts/della/remote/setup_terminalbench.sh`, then use
`examples.terminalbench.prepare_offline` on that host to stage the supported
training tasks, container images, and in-container dependencies. Pass its sealed
`bundle.json` to the shared runner with `--offline-task-bundle`. See the
Terminal-Bench README for supported task coverage; the runner rejects missing
tasks or changed bytes. A reference-solution trial has verified the offline
bootstrap and official verifier on allocated Della CPU resources. Model pilots
still require their own completion evidence. Docker remains available on
separate Docker-capable hosts.

Measure latency with the shared `--mode pilot`, which uses training data only.
Run vanilla GEPA and FOREST through `--condition both`. Keep the ordered data,
model settings, initial prompts, and benchmark budgets matched. Validation
selects and freezes the winner before held-out testing; test results never
change prompts or selection. Preserve provider-attempt and timing evidence.

The repository has no separate campaign runner or automatic Slurm continuation
workflow. Use the shared entrypoints inside the authorized allocation and follow
their exact resume contracts. Do not invent successful benchmark results from
passing offline tests or an accepted scheduler submission.
