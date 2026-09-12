---
name: della
description: >-
  Work with this repository's Della GPU/Slurm experiments from a laptop: SSH,
  setup, pinned checkpoints, HotPotQA launchers, verification, monitoring,
  result retrieval, and storage. Read before Della operations.
---

# Working with Della

This skill incorporates Zach's PR #62 tooling and the reviewed decisions. Use
`examples/hotpotqa/DELLA_CAMPAIGN.md` as the current runbook and
`docs/experiment-consolidation.md` for integration decisions. Do not launch the
old `169ddda` commit or switch to a separate tooling branch.

## Scope and source

- Start with HotPotQA. Terminal-Bench's Apptainer integration is paused.
- Code consolidation or review does not submit a campaign.
- Use the repository scripts for connections, setup, sync, submission, and
  fetching. Do not invent alternate submission paths.
- Launch from the reviewed clean source. Preflight requires explicit
  `HOTPOTQA_SOURCE_COMMIT`; production stages `git archive HEAD` into
  `$REMOTE_DIR/sources/<sha>` and records the source manifest.
- Source, runtime, checkpoint, or experiment changes require a fresh campaign.
  Keep the same configuration for resume.

## Connection and machines

The ignored `scripts/della/.env` supplies the user, two hosts, scratch/remote
directories, model storage, and partition. Keep it mode 600, owned by the user,
and outside Git. Reuse working SSH configuration; never print credentials.

Operations use `BatchMode=yes` and `StrictHostKeyChecking=yes`. Check existing
masters with `scripts/della/della_session.sh status`. If interactive authentication
is needed, `open` authenticates once per host. It expects `ControlMaster auto`,
`ControlPath ~/.ssh/cm/%r@%h:%p`, and `ControlPersist yes` in the SSH config.
Host keys must already be verified. Close sessions only when requested.

- Login: brief operations, sync, submission, and Slurm status.
- `della-vis1`: internet-dependent builds, datasets, model downloads.
- Allocated `ailab` nodes: eight H200s, model serving and optimization, with
  `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
- Put environments, caches, logs, and outputs on configured `SCRATCH_BASE`.
  Check current quota before large operations. Scratch is not backed up.
- Checkpoints use shared `MODEL_STORAGE`; download through the verified scripts.

References: https://researchcomputing.princeton.edu/systems/della and
https://researchcomputing.princeton.edu/support/knowledge-base/data-storage.

## Scripts

- `preflight_hotpotqa.sh`: read-only source, permission, SSH, CUDA, storage, lock checks.
- `build_env.sh [model ...]`: sync, environments, datasets, detached downloads.
  Its exit proves downloads started. Require `MODELS_DONE` in the printed log
  and the byte-verified checkpoint manifests before proceeding.
- `remote/setup_env.sh`: GEPA Python 3.11.13 / uv 0.9.13 / pinned DSPy, and
  serving Python 3.12.7 from committed requirements hashes.
- `remote/download_dataset.sh`: Wiki-2017 BM25 and 150/300/300 HotPotQA.
- `remote/download_model.sh qwen3.8-27b|deepseek-v4.1-flash`: pinned model bytes.
- `submit_deepseek_smoke.sh submit|fetch <id>`: independent transcript and
  four-tool diagnostic; no campaign qualification marker.
- `submit_hotpotqa.sh`: exact-source production submission and `afterok` chain.
- `fetch_hotpotqa_results.sh`: validated local results under
  `outputs/hotpotqa-campaigns/<campaign>/<commit>/`.
- `sync_to_della.sh`: preserves environments, tools, caches, snapshots, outputs.

Run remote setup stages from the synced checkout with `SCRATCH_BASE`,
`MODEL_STORAGE`, and any custom `WIKI17_DIR` exported. Respect artifact and
model-directory locks. Do not attach long downloads to a laptop SSH session.
Environment installation, including ordered CUTLASS reinstalls, uses lock hashes.

## Approved HotPotQA experiment

Use Qwen3.8-27B and DeepSeek-V4.1-Flash, each homogeneous across all roles.
Qwen uses `.serving-venv` (vLLM 0.25.1 / Torch 2.11); DeepSeek V4.1 uses
`.serving-venv-deepseek-v4.1-flash`, the exact hash-locked `e77daef89` vLLM wheel
(`0.1.1.dev5+ge77daef89`), Torch 2.13, and prebuilt FlashInfer kernel wheels.
HotPotQA does not depend on POSIT.

- Qwen: TP1/DP8, eight API servers, one sequence/replica, context 262,144,
  thinking `xhigh`.
- DeepSeek V4.1: TP8/EP8/DP1, one API server and one active sequence, context
  262,144, numeric thinking effort 100, native `deepseek_v41` parsers, FP8 KV,
  automatic block size (64 on SM90), original weight formats, no speculation.
  `FLASHINFER_NO_DOWNLOAD=1`; use the serving wheels' CUDA headers first.
- Temperature 1.0 / top-p 0.95 for every role. Output caps: 16,384 HotPotQA,
  32,768 Terminal-Bench. Server context is the Della setting, not the provider's
  maximum. See `examples/common/temperature_policy.md`.
- Initial workers: 12 Qwen / 4 DeepSeek. Calibrate on training and freeze across
  methods/budgets; defaults are not evidence of completed calibration.
- Logical request deadline: 3,600 seconds shared by at most three transient
  attempts, 1/2-second backoff, SDK retries zero, and attempt logs.
- Character limits remain configurable and unlimited by default. Multiple
  selected-section edits are allowed; the editor must explicitly finish.
- Evaluation/response caching stays off; checkpoint/journal recovery remains.

The exact DeepSeek runtime must pass its 20-attempt campaign canary before the
six optimization cells. A failed canary or native-tool preflight cannot freeze
campaign identity. The independent smoke does not replace that gate.

Per model: standard `vanilla`, `react_v2`, `react_v2_random`, `action` at 6,871
metric calls, then independent expanded `vanilla`, `react_v2` at 13,742. Inspect
failed parents and orphaned `afterok` dependencies before resubmitting. Exclude
failed or unverified runs.

Jobs request one node, eight H200s, 64 CPUs, 768G on `ailab`; caps are 72 hours
for Qwen standard and 144 hours for expanded/DeepSeek. They are not estimates.
Use Slurm accounting and output artifacts to prove completion, not submission IDs.
