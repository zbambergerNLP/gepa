#!/bin/bash
# Run the GEPA FrontierCS evaluation on della.
#
# Serves a generator model via vLLM (from the POSIT venv, which has vLLM
# installed), then runs the GEPA FrontierCS optimization experiment against it.
# Mirrors examples/ifbench/run_ifbench.sbatch and examples/pupa/run_pupa.sbatch
# but for the FrontierCS open-ended CS research benchmark
# (https://github.com/FrontierCS/Frontier-CS, HF FrontierCS/Frontier-CS).
#
# Prerequisites: same as ifbench (POSIT repo, GEPA venv, HF cache on scratch).
# FrontierCS dataset is pulled via HuggingFace datasets on first use
# (FrontierCS/Frontier-CS, ~100 problems) or from a local data/frontiercs.jsonl
# fallback. First run downloads the dataset; subsequent runs use HF cache.
#
# Usage (full stretch run, all three conditions, 30/30/30 splits):
#   sbatch examples/frontiercs/run_frontiercs.sbatch
#
# Mini test run (15/15/15, small budget):
#   sbatch --export=ALL,TRAIN_LIMIT=15,VAL_LIMIT=15,TEST_LIMIT=15,MAX_METRIC_CALLS=150 \
#       --time=02:00:00 examples/frontiercs/run_frontiercs.sbatch
#
# Override defaults:
#   sbatch --export=ALL,MODEL=Qwen3.5-9B,CONDITION=action,PROGRAM=1stage \
#       examples/frontiercs/run_frontiercs.sbatch
#
# Data-path mode (offline / smoke JSONL, 20 lines expanded to 30/30/30):
#   sbatch --export=ALL,DATA_PATH=examples/frontiercs/data/frontiercs.jsonl \
#       examples/frontiercs/run_frontiercs.sbatch

#SBATCH --job-name=gepa-frontiercs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=gepa-frontiercs-%j.log

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1

# --- Knobs (override with sbatch --export=ALL,VAR=...) --------------------
MODEL_STORAGE="${MODEL_STORAGE:-/projects/BSTEWART/model_storage}"
MODEL="${MODEL:-Qwen3.5-9B}"
# Unique per-job port: della nodes host 8 GPUs, so multiple single-GPU jobs can
# share a node; a fixed port makes co-located jobs cross-talk and cross-kill.
GEN_PORT="${GEN_PORT:-$((10000 + ${SLURM_JOB_ID:-0} % 5000))}"
MAX_METRIC_CALLS="${MAX_METRIC_CALLS:-4000}"
CONDITION="${CONDITION:-all}"
PROGRAM="${PROGRAM:-2stage}"
SEED_STYLE="${SEED_STYLE:-plain}"
ACTIONS="${ACTIONS:-default}"
DATA_PATH="${DATA_PATH:-}"
TAG="${TAG:-}"
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
export NLTK_DATA="${NLTK_DATA:-${SCRATCH_BASE}/nltk_data}"
export HF_HUB_OFFLINE=0
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_USE_FLASHINFER_SAMPLER=0

LOG_DIR="${SCRATCH_BASE}/logs"
mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}" "${VLLM_CACHE_ROOT}" \
    "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${LOG_DIR}" "${NLTK_DATA}"

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

echo "==> running GEPA FrontierCS experiment: condition=${CONDITION} program=${PROGRAM} max_metric_calls=${MAX_METRIC_CALLS}"
echo "==> limits: train=${TRAIN_LIMIT:-30} val=${VAL_LIMIT:-30} test=${TEST_LIMIT:-30}"
echo "==> solver=${SOLVER_MODEL} reflection=${REFLECTION_MODEL} api_base=${API_BASE}"
echo "==> data_path=${DATA_PATH:-HF FrontierCS/Frontier-CS}"

"${PY}" -m examples.frontiercs.main \
    --solver-model "${SOLVER_MODEL}" \
    --reflection-model "${REFLECTION_MODEL}" \
    --api-base "${API_BASE}" \
    --max-metric-calls "${MAX_METRIC_CALLS}" \
    --condition "${CONDITION}" \
    --program "${PROGRAM}" \
    --seed-style "${SEED_STYLE}" \
    --actions "${ACTIONS}" \
    "${LIMIT_ARGS[@]}"
