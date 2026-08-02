---
name: della
description: >-
  Interact with Princeton's della GPU/SLURM cluster from this repo: sync code,
  build the venv, submit/monitor sbatch jobs, fetch results, and manage the
  tight /home quota (keep caches + outputs on scratch). Use whenever the task
  involves running on della, SSH/rsync to della, SLURM (sbatch/squeue/sinfo),
  vLLM serving on della, or "disk quota exceeded" / storage errors there. For
  other SLURM clusters, add a sibling skill folder under .claude/skills/.
---

# Working with della (Princeton Research Computing)

della is a SLURM GPU cluster. This repo drives it from a laptop via the scripts
below, which read all connection + cluster config from `scripts/della/.env`
(gitignored; copy from a teammate or posit's `src/.env` pattern). Refs:
- Systems overview: https://researchcomputing.princeton.edu/systems/della
- Data storage & quotas: https://researchcomputing.princeton.edu/support/knowledge-base/data-storage

## Connectivity first

della is only reachable on the Princeton network or the GlobalProtect VPN.
Before any remote step, check `nc -z -G 8 della.princeton.edu 22`; if it times
out (or `della-vis1` fails DNS), ask the user to connect the VPN - do not retry
in a loop.

## The scripts (use these, don't hand-roll ssh/sbatch)

Local launchers (run from the repo root on your laptop; all use sshpass +
`scripts/della/.env`):
- **`scripts/della/sync_to_della.sh`** rsyncs the repo → `${REMOTE_DIR}/`
  (excludes `.venv/`, `outputs/`, caches, `.git/`, and `.env` itself). Run
  after every change.
- **`scripts/della/build_env.sh`** syncs, then builds `.venv` on the **vis
  node** (`uv sync --extra dev`). Run once / after deps change.
- **`scripts/della/submit_hotpotqa.sh`** / **`submit_ifbench.sh`**: sync +
  sbatch an experiment. Knobs via env: `MODEL=`, `MAX_METRIC_CALLS=`,
  `CONDITION=`, `TIME=`, `NO_SYNC=1`. `submit_ifbench.sh` also takes
  `TRAIN_LIMIT=/VAL_LIMIT=/TEST_LIMIT=` for mini runs and `SETUP=1` to install
  the `ifbench` extra + nltk data (to `${SCRATCH_BASE}/nltk_data`) + spacy
  `en_core_web_sm` on the vis node first.
- **`scripts/della/fetch_results.sh`** rsyncs experiment results/logs back.

Remote pieces (run on della; you normally don't call these directly):
- **`examples/hotpotqa/run_hotpotqa.sbatch`** / **`examples/ifbench/run_ifbench.sbatch`**
  are the SLURM jobs: serve the model via vLLM (from the POSIT venv at
  `${POSIT_DIR}/src/.venv/bin/vllm`, default `/home/${USER}/posit`), wait for
  health, run `examples.<name>.main` with GEPA's own `.venv`, tear down. They
  export all cache dirs (and `NLTK_DATA` for ifbench) to scratch.

## Conventions (do not hardcode cluster specifics)

- **Partition comes from `GPU_PARTITION` in `scripts/della/.env`** (currently
  `ailab`). Always submit with `--partition=$GPU_PARTITION`; never hardcode.
- **Models come from `MODEL_STORAGE`** (`/projects/BSTEWART/model_storage`).
  Reference checkpoints as `${MODEL_STORAGE}/<name>`, never an absolute path.
- **Caches + outputs** go under `SCRATCH_BASE`
  (`/scratch/gpfs/${RESEARCH_GROUP}/${USER}/gepa`) on scratch, never `/home`.
- `scripts/della/.env` also holds `REMOTE_PASSWORD`. Never echo it; edit one
  key with `sed -i '' 's#^KEY=.*#KEY=val#' scripts/della/.env` rather than
  reading the whole file.

## SLURM specifics (learned the hard way)

- **Do NOT pass `--partition=gpu`**. della's submit filter rejects it
  ("You specified a partition of gpu. This is not allowed"). Use the named
  partition from `GPU_PARTITION`. The default partition is `cpu*` (no GPUs), so
  a GPU job must request `--gres=gpu:N`.
- **Prefer `ailab` (H200, 141 GB)**. Large models fit with ample KV-cache room.
  On `a100` (80 GB) a 35B model nearly fills the card at
  `--gpu-memory-utilization 0.85` ("No available memory for the cache blocks");
  bump GMU to ~0.95 (the sbatch's `GEN_GMU` knob) there.
- The GEPA jobs need **1 GPU** (single vLLM endpoint serves solver + reflection).
- Inspect: `sinfo -o "%20P %.6a %.14l %.6D %G"`, `squeue -u $USER`,
  `sacct -j <id> --format=JobID,Partition,State,Elapsed,ExitCode`,
  `sacctmgr -nP show assoc user=$USER format=account,qos`.

## Node types

- **Login** (`REMOTE_HOST`): brief ops only (rsync, sbatch, squeue, checkquota).
- **Vis** (`REMOTE_VIS_HOST`, `della-vis1/2`): has internet + CPU/RAM; use for
  `build_env.sh`, `SETUP=1` data downloads, and large file moves.
- **GPU compute**: **no internet**. The sbatch scripts force offline
  (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`); anything needing a download
  (uv sync, nltk, spacy models) must happen on the vis node beforehand.

## Storage & quota (the #1 source of failures)

- **`/home` has a tight ~50 GiB quota** (and ~2M file limit); check with
  `checkquota`. vLLM/HF/torch caches + result traces are GB-scale and overflow
  it, producing **`[Errno 122] Disk quota exceeded`**, which crashes
  `vllm serve` mid-load (often masked by a later "Error in sys.excepthook").
- **Keep all large I/O on scratch**: the sbatch scripts already export
  `XDG_CACHE_HOME`, `HF_HOME`, `VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR`,
  `TRITON_CACHE_DIR` (and `NLTK_DATA`) under `SCRATCH_BASE`.
- If `~/.cache` has already filled `/home`, relocate it to scratch and symlink
  back (run on the vis node): for each of `uv pip vllm huggingface flashinfer
  torch_extensions triton`, `rsync -a --remove-source-files ~/.cache/$d/
  /scratch/gpfs/$RESEARCH_GROUP/$USER/.cache/$d/ && rm -rf ~/.cache/$d && ln -s
  /scratch/gpfs/$RESEARCH_GROUP/$USER/.cache/$d ~/.cache/$d`.
- scratch is **not backed up** and is periodically purged, so keep only caches +
  regenerable artifacts there.

## The venvs

- GEPA's venv is built by `build_env.sh` at `${REMOTE_DIR}/.venv` (uv; use
  `uv pip` or `python -m`, there is no `pip` script). It is **not relocatable**;
  if `REMOTE_DIR` changes, re-run `build_env.sh` rather than moving `.venv`.
- vLLM comes from the **POSIT venv** (`${POSIT_DIR}/src/.venv`), which must be
  built separately (see posit's own della tooling). The sbatch scripts fail
  fast with a clear error if either venv is missing.

## vLLM serving notes

- The sbatch scripts serve with `--reasoning-parser qwen3` (Qwen "thinking"
  models) and `--max-num-seqs 16`; litellm clients talk to it via the
  `hosted_vllm/<name>` model prefix + `api_base=http://localhost:8000/v1` and
  `OPENAI_API_KEY=EMPTY`.
- After launching `vllm serve` in the background, poll
  `curl -sf http://localhost:<port>/v1/models` until it answers before running
  a client; if it never comes up, read the serve log
  (`${SCRATCH_BASE}/logs/gen-<jobid>.log`) for the real error.
