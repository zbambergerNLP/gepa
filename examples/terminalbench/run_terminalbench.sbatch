#!/bin/bash
# Run the GEPA TerminalBench evaluation on della.
#
# Serves a generator model via vLLM (from the POSIT venv, which has vLLM
# installed), then runs the GEPA TerminalBench optimization experiment against
# it. Mirrors examples/ifbench/run_ifbench.sbatch for the TerminalBench
# benchmark (T-Bench, https://terminal-bench.github.io, 50+ tasks, Docker-based).
#
# Prerequisites:
#   1. POSIT repo built on della at /home/${USER}/posit (with vLLM in its venv)
#   2. GEPA repo synced and venv built via scripts/della/build_env.sh
#      (uv sync --extra dev; datasets + litellm already in full)
#   3. HF cache on scratch for first-time T-Bench fetch (laude/terminal-bench)
#      — run once on a login node with internet, or pre-seed
#        data/terminalbench.jsonl fallback.
#
# Usage (full run, all three conditions, 50 tasks 20/15/15):
#   sbatch examples/terminalbench/run_terminalbench.sbatch
#
# Mini test run (10 train / 5 val / 5 test, small budget):
#   sbatch --export=ALL,TRAIN_LIMIT=10,VAL_LIMIT=5,TEST_LIMIT=5,MAX_METRIC_CALLS=150 \
#       --time=02:00:00 examples/terminalbench/run_terminalbench.sbatch
#
# Override defaults:
#   sbatch --export=ALL,MODEL=Qwen3-8B,CONDITION=action,PROGRAM=1stage \
#       examples/terminalbench/run_terminalbench.sbatch
#
# With local data:
#   sbatch --export=ALL,DATA_PATH=/path/to/terminalbench.jsonl \
#       examples/terminalbench/run_terminalbench.sbatch

#SBATCH --job-name=gepa-terminalbench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=gepa-terminalbench-%j.log

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1

# --- Knobs (override with sbatch --export=ALL,VAR=...) --------------------
MODEL_STORAGE="${MODEL_STORAGE:-/projects/BSTEWART/model_storage}"
MODEL="${MODEL:-Qwen3-8B}"
# Unique per-job port: della nodes host 8 GPUs, so multiple single-GPU jobs can
# share a node; a fixed port makes co-located jobs cross-talk and cross-kill.
GEN_PORT="${GEN_PORT:-$((10000 + ${SLURM_JOB_ID:-0} % 5000))}"
MAX_METRIC_CALLS="${MAX_METRIC_CALLS:-3000}"
CONDITION="${CONDITION:-all}"
PROGRAM="${PROGRAM:-2stage}"
SEED_STYLE="${SEED_STYLE:-plain}"
ACTIONS="${ACTIONS:-default}"
TAG="${TAG:-}"
DATA_PATH="${DATA_PATH:-}"
TRAIN_LIMIT="${TRAIN_LIMIT:-}"
VAL_LIMIT="${VAL_LIMIT:-}"
TEST_LIMIT="${TEST_LIMIT:-}"
GEN_GMU="${GEN_GMU:-0.90}"
GEN_MAX_LEN="${GEN_MAX_LEN:-32768}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"
# POSIT repo location (for its vLLM install).
POSIT_DIR="${POSIT_DIR:-/home/${USER}/posit}"

# --- Keep all large I/O on scratch ----------------------------------------
SCRATCH_BASE="${SCRATCH_BASE:-/scratch/gpfs/BSTEWART/${USER}/gepa}"
export XDG_CACHE_HOME="${SCRATCH_BASE}/.cache"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export VLLM_CACHE_ROOT="${SCRATCH_BASE}/.cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_BASE}/.cache/torchinductor"
export TRITON_CACHE_DIR="${SCRATCH_BASE}/.cache/triton"
export HF_HUB_OFFLINE=0
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_USE_FLASHINFER_SAMPLER=0

LOG_DIR="${SCRATCH_BASE}/logs"
mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}" "${VLLM_CACHE_ROOT}" \
    "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${LOG_DIR}"

# --- Verify both venvs exist ----------------------------------------------
PY=".venv/bin/python"
VLLM_BIN="${POSIT_DIR}/src/.venv/bin/vllm"
[[ -x "${PY}" ]] || { echo "ERROR: missing ${PY} -- run scripts/della/build_env.sh" >&2; exit 1; }
[[ -x "${VLLM_BIN}" ]] || { echo "ERROR: missing ${VLLM_BIN} -- build the POSIT venv first" >&2; exit 1; }

# --- Serve the generator (using POSIT's vLLM) -----------------------------
GEN_LOG="${LOG_DIR}/gen-${SLURM_JOB_ID:-local}.log"
echo "==> serving generator ${MODEL} on :${GEN_PORT} (GPU 0, util ${GEN_GMU}); log: ${GEN_LOG}"

# Load CUDA toolkit for JIT compilation (needed by some models).
export PATH="${POSIT_DIR}/src/.venv/bin:${PATH}"
CUDA_TAG="$(${POSIT_DIR}/src/.venv/bin/python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 13.0)"
source /usr/share/Modules/init/bash 2>/dev/null || true
module load "cudatoolkit/${CUDA_TAG}" 2>/dev/null \
    || module load cudatoolkit/13.0 2>/dev/null || true

env CUDA_VISIBLE_DEVICES=0 "${VLLM_BIN}" serve "${MODEL_STORAGE}/${MODEL}" \
    --served-model-name "${MODEL}" \
    --reasoning-parser qwen3 \
    --host 0.0.0.0 \
    --port "${GEN_PORT}" \
    --gpu-memory-utilization "${GEN_GMU}" \
    --max-model-len "${GEN_MAX_LEN}" \
    --max-num-seqs 32 \
    > "${GEN_LOG}" 2>&1 &
GEN_PID=$!

cleanup() {
    echo "==> tearing down endpoint"
    kill "${GEN_PID}" 2>/dev/null || true
    wait "${GEN_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# --- Wait for endpoint health ---------------------------------------------
deadline=$((SECONDS + HEALTH_TIMEOUT))
until curl -sf "http://localhost:${GEN_PORT}/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "${GEN_PID}" 2>/dev/null; then
        echo "ERROR: generator exited before becoming healthy; see ${GEN_LOG}" >&2
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo "ERROR: generator not healthy within ${HEALTH_TIMEOUT}s; see ${GEN_LOG}" >&2
        exit 1
    fi
    sleep 5
done
echo "==> generator up on :${GEN_PORT}"

# --- Run the GEPA experiment (using GEPA's own venv) ---------------------
# litellm talks to vLLM via the hosted_vllm/ prefix + api_base.
# Set OPENAI_API_KEY=EMPTY so litellm doesn't reject the request.
export OPENAI_API_KEY="EMPTY"
# Flush python output line-by-line so the SLURM log shows live progress.
export PYTHONUNBUFFERED=1
SOLVER_MODEL="hosted_vllm/${MODEL}"
REFLECTION_MODEL="hosted_vllm/${MODEL}"
API_BASE="http://localhost:${GEN_PORT}/v1"

LIMIT_ARGS=()
[[ -n "${TRAIN_LIMIT}" ]] && LIMIT_ARGS+=(--train-limit "${TRAIN_LIMIT}")
[[ -n "${VAL_LIMIT}" ]] && LIMIT_ARGS+=(--val-limit "${VAL_LIMIT}")
[[ -n "${TEST_LIMIT}" ]] && LIMIT_ARGS+=(--test-limit "${TEST_LIMIT}")
[[ -n "${TAG}" ]] && LIMIT_ARGS+=(--tag "${TAG}")
[[ -n "${DATA_PATH}" ]] && LIMIT_ARGS+=(--data-path "${DATA_PATH}")

echo "==> running GEPA TerminalBench experiment: condition=${CONDITION} program=${PROGRAM} max_metric_calls=${MAX_METRIC_CALLS}"
echo "==> limits: train=${TRAIN_LIMIT:-full} val=${VAL_LIMIT:-full} test=${TEST_LIMIT:-full} data_path=${DATA_PATH:-HF/local}"
echo "==> solver=${SOLVER_MODEL} reflection=${REFLECTION_MODEL} api_base=${API_BASE}"

"${PY}" -m examples.terminalbench.main \
    --solver-model "${SOLVER_MODEL}" \
    --reflection-model "${REFLECTION_MODEL}" \
    --api-base "${API_BASE}" \
    --max-metric-calls "${MAX_METRIC_CALLS}" \
    --condition "${CONDITION}" \
    --program "${PROGRAM}" \
    --seed-style "${SEED_STYLE}" \
    --actions "${ACTIONS}" \
    "${LIMIT_ARGS[@]}"
