---
name: della
description: >-
  Run GEPA experiments on Princeton's della GPU/SLURM cluster from a laptop:
  SSH setup (key + password, ControlMaster), scripts/della/*.sh launchers,
  the pinned HotPotQA campaign runbook, monitoring/resuming Slurm jobs, and
  storage/quota rules. Use whenever a task mentions della, Slurm (sbatch,
  squeue, sacct), Della logs/results, POSIT/vLLM serving on della, or
  "disk quota exceeded" there.
---

# Working with della (Princeton Research Computing)

Della is a SLURM GPU cluster. This repo drives it from a laptop through
`scripts/della/*.sh`, which read all connection and cluster config from the
gitignored `scripts/della/.env` (template: `scripts/della/.env.example`).
Refs: https://researchcomputing.princeton.edu/systems/della and
https://researchcomputing.princeton.edu/support/knowledge-base/data-storage

## The scripts (use these; never hand-roll ssh/rsync/sbatch)

Local launchers (laptop, repo root):
- `scripts/della/della_session.sh open|status|close`: open the persistent SSH
  master connections (see SSH below). Run `open` once per laptop session
  before anything else; every other script fails with "Permission denied
  (keyboard-interactive)" without it.
- `scripts/della/preflight_hotpotqa.sh`: runbook steps 1-4 (prereqs, exact
  commit + clean tree, `.env`, BatchMode SSH, POSIT/vLLM >= 0.17.0, apptainer,
  writable `MODEL_STORAGE`, home quota). Read-only.
- `scripts/della/build_env.sh`: one-time on `della-vis1` (internet). Builds
  the frozen GEPA venv at `$REMOTE_DIR/.venv`, freezes the POSIT serving env,
  builds/verifies the Wiki-2017 BM25 index, caches the HotPotQA split,
  **downloads and byte-verifies both pinned checkpoints into
  `$MODEL_STORAGE`**, and builds the GLM SGLang Apptainer image. Hours.
- `scripts/della/submit_hotpotqa.sh`: stages `git archive HEAD` under
  `$REMOTE_DIR/sources/<commit>`, verifies every artifact, then submits the
  `afterok` chain. Refuses a dirty tree; records HEAD as the source commit.
- `scripts/della/sync_to_della.sh`: code sync (called by the two above).
- `scripts/della/fetch_hotpotqa_results.sh`: pull runs/logs/locks/analysis
  into `outputs/hotpotqa-campaigns/<campaign>/<commit>/`.

Remote pieces (never call directly): `examples/hotpotqa/run_hotpotqa.sbatch`
serves the model (POSIT vLLM for Qwen, SGLang Apptainer for GLM), waits for
health, runs GEPA, tears down.

## HotPotQA campaign (Gilad's runbook)

Full text: `examples/hotpotqa/DELLA_CAMPAIGN.md`. Essentials:
- Launch from **exactly** the pinned commit, detached
  (`git switch --detach <commit>`), with a clean tree. Tooling lives on a
  separate branch; switching back and forth is fine because `.env` is ignored.
- Order per model arm: standard `vanilla`, `react_v2`, `react_v2_random`,
  `action` (6,871 calls) then expanded `vanilla`, `react_v2` (13,742 calls).
  Qwen submits 6 jobs; GLM submits a 4 h canary + 6 jobs. Arms are independent.
- Job names: `gepa-hp-<profile>-<standard|expanded>-<condition>`,
  `gepa-hp-glm-canary`. Logs:
  `$SCRATCH_BASE/logs/hotpotqa/<campaign>/<commit>/hotpotqa-<job>-<id>.log`
  plus `gen-<id>.log` for the model server.
- Resume = resubmit the same commit/campaign/model with
  `BUDGET_PROFILE=standard|expanded CONDITION=<cell>`; GEPA reuses saved state.
  Cancel orphaned `afterok` dependents of a failed parent first.
- Never include a run that failed a preflight/integrity check.

## SSH (learned the hard way)

- Della requires **publickey AND keyboard-interactive** (password; Duo when
  off-campus). A key alone yields "Authenticated using publickey with partial
  success" then "Permission denied (keyboard-interactive)".
- Gilad's scripts use `BatchMode=yes` + `StrictHostKeyChecking=yes`, so they
  rely on **ControlMaster multiplexing**: `~/.ssh/config` sets
  `ControlMaster auto`, `ControlPath ~/.ssh/cm/%r@%h:%p`, `ControlPersist yes`
  for both hosts, and `della_session.sh open` authenticates once per host.
  `ssh -O check <host>` shows whether a master is alive.
- Both host keys must already be in `~/.ssh/known_hosts`.
- The POSIT repo's scripts use `sshpass` + password instead; do not mix the
  two styles in this repo.

## Node types

- Login (`REMOTE_HOST`, della.princeton.edu): brief ops only (rsync, sbatch,
  squeue, scontrol, sacct, checkquota).
- Vis (`REMOTE_VIS_HOST`, della-vis1): internet + CPU/RAM; builds, downloads,
  apptainer builds, large file moves.
- GPU compute (`ailab` = H200 141 GB, 8 per node): **no internet**; the sbatch
  forces `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

## SLURM specifics

- Never `--partition=gpu` (rejected by the submit filter). Use
  `GPU_PARTITION=ailab`; the campaign requires it and 8 GPUs per job.
- Inspect: `squeue -u $USER -o "%.18i %.60j %.2t %.12M %.30R"`,
  `scontrol show job <id> | tr ' ' '\n' | grep -E '^(JobName|JobState|Reason|Dependency)='`,
  `sacct -u $USER --starttime YYYY-MM-DD --format=JobIDRaw,JobName%64,State,ExitCode,Elapsed,Timelimit`.
- Time limits are caps, not estimates: Qwen standard 72 h, Qwen expanded
  144 h, GLM jobs 144 h, GLM canary 4 h. 144 h is Della's maximum.

## Storage and quota (the #1 source of failures)

- `/home` quota ~48.8 GiB and ~1.9M files; check with `checkquota`. Keep all
  caches, venvs, outputs on scratch: `/scratch/gpfs/BSTEWART/<netid>/gepa`
  (`REMOTE_DIR` = `SCRATCH_BASE`). The scripts already export
  `XDG_CACHE_HOME`, `HF_HOME`, `UV_CACHE_DIR`, `DSPY_CACHEDIR`,
  `APPTAINER_CACHEDIR` under scratch.
- Checkpoints live in the shared, group-writable
  `/projects/BSTEWART/model_storage` (`MODEL_STORAGE`); reference them as
  `${MODEL_STORAGE}/<name>`. Only `build_env.sh` may populate it (it writes the
  `.gepa-model-integrity.json` manifests the launcher demands).
- Scratch is not backed up and is purged periodically.

## POSIT dependency

The Qwen arm serves through the POSIT vLLM venv at `$POSIT_DIR`
(default `/home/<netid>/posit`, `src/.venv/bin/vllm`). `build_env.sh` and the
launcher require `$POSIT_DIR` to be a **clean git checkout** (they record its
commit) with vLLM >= 0.17.0 and data-parallel / multi-API-server / native
tool-call support. The POSIT repo's own sync script rsyncs `src/` without
`.git`, which does not satisfy this; the directory must be a real clone.
