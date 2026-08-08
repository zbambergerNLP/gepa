#!/bin/bash
# Run the GEPA MBPP evaluation on della.
#
# Serves a generator model via vLLM (from the POSIT venv, which has vLLM
# installed), then runs the GEPA MBPP optimization experiment against it.
# Mirrors examples/gsm8k/run_gsm8k.sbatch and examples/ifbench/run_ifbench.sbatch
# but for MBPP (Austin et al. 2021, HF mbpp sanitized).
#
# Prerequisites: same as gsm8k/ifbench/pupa (POSIT repo, GEPA venv).
# MBPP dataset is pulled via HuggingFace datasets on first use
# (mbpp sanitized, 974 problems; fallback to local data/*.jsonl).
#
# Usage (full run, all three conditions):
#   sbatch examples/mbpp/run_mbpp.sbatch
#
# Mini test run (20/30/30, small budget):
#   sbatch --export=ALL,TRAIN_LIMIT=20,VAL_LIMIT=30,TEST_LIMIT=30,MAX_METRIC_CALLS=100 \
#       --time=02:00:00 examples/mbpp/run_mbpp.sbatch
#
# Override defaults:
#   sbatch --export=ALL,MODEL=Qwen3-8B,CONDITION=action,SEED_STYLE=structured \
#       examples/mbpp/run_mbpp.sbatch

#SBATCH --job-name=gepa-mbpp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=gepa-mbpp-%j.log

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1

# --- Knobs (override with sbatch --export=ALL,VAR=...) --------------------
MODEL_STORAGE="${MODEL_STORAGE:-/projects/BSTEWART/model_storage}"
MODEL="${MODEL:-Qwen3.5-9B}"
GEN_PORT="${GEN_PORT:-$((10000 + ${SLURM_JOB_ID:-0} % 5000))}"
MAX_METRIC_CALLS="${MAX_METRIC_CALLS:-5000}"
CONDITION="${CONDITION:-all}"
SEED_STYLE="${SEED_STYLE:-plain}"
ACTIONS="${ACTIONS:-default}"
TAG="${TAG:-}"
TRAIN_LIMIT="${TRAIN_LIMIT:-}"
VAL_LIMIT="${VAL_LIMIT:-}"
TEST_LIMIT="${TEST_LIMIT:-}"
DATA_PATH="${DATA_PATH:-}"
GEN_GMU="${GEN_GMU:-0.90}"
GEN_MAX_LEN="${GEN_MAX_LEN:-32768}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"
POSIT_DIR="${POSIT_DIR:-/home/${USER}/posit}"

# --- Keep all large I/O on scratch ----------------------------------------
GEPA_SCRATCH_ROOT="${GEPA_SCRATCH_ROOT:-/scratch/gpfs/${USER}/gepa-mbpp-${SLURM_JOB_ID:-$$}}"
export HF_HOME="${HF_HOME:-${GEPA_SCRATCH_ROOT}/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
mkdir -p "${GEPA_SCRATCH_ROOT}" "${HF_HOME}"

echo "[mbpp] Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
echo "[mbpp] MODEL=${MODEL} CONDITION=${CONDITION} SEED_STYLE=${SEED_STYLE} ACTIONS=${ACTIONS} MAX_METRIC_CALLS=${MAX_METRIC_CALLS} TAG=${TAG}"
echo "[mbpp] HF_HOME=${HF_HOME}"

# --- GEPA venv ------------------------------------------------------------
if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
elif [ -f "../.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source ../.venv/bin/activate
fi

# --- Serve generator via vLLM (POSIT) -------------------------------------
GEN_API_BASE="http://localhost:${GEN_PORT}/v1"
VLLM_MODEL_PATH="${MODEL_STORAGE}/${MODEL}"
if [ ! -d "${VLLM_MODEL_PATH}" ] && [ -f "${MODEL_STORAGE}/${MODEL}.tar" ]; then
    echo "[mbpp] Extracting ${MODEL} to scratch..."
    mkdir -p "${GEPA_SCRATCH_ROOT}/models"
    tar -xf "${MODEL_STORAGE}/${MODEL}.tar" -C "${GEPA_SCRATCH_ROOT}/models"
    VLLM_MODEL_PATH="${GEPA_SCRATCH_ROOT}/models/${MODEL}"
fi

if [ -d "${POSIT_DIR}" ] && [ -f "${POSIT_DIR}/scripts/serve_vllm.sh" ]; then
    echo "[mbpp] Starting vLLM server for ${MODEL} on port ${GEN_PORT}..."
    bash "${POSIT_DIR}/scripts/serve_vllm.sh" "${VLLM_MODEL_PATH}" "${GEN_PORT}" &
    VLLM_PID=$!
    # Wait for health
    echo "[mbpp] Waiting for vLLM health (timeout ${HEALTH_TIMEOUT}s)..."
    for i in $(seq 1 "${HEALTH_TIMEOUT}"); do
        if curl -sf "http://localhost:${GEN_PORT}/health" >/dev/null 2>&1 || curl -sf "http://localhost:${GEN_PORT}/v1/models" >/dev/null 2>&1; then
            echo "[mbpp] vLLM ready after ${i}s"
            break
        fi
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "[mbpp] vLLM server died!" >&2
            exit 1
        fi
        sleep 1
    done
else
    echo "[mbpp] No POSIT/vLLM found; assuming external solver at \${API_BASE:-${GEN_API_BASE}}"
    VLLM_PID=""
fi

# --- Build GEPA args ------------------------------------------------------
GEPA_ARGS=(
    --max-metric-calls "${MAX_METRIC_CALLS}"
    --condition "${CONDITION}"
    --seed-style "${SEED_STYLE}"
    --actions "${ACTIONS}"
    --solver-model "hosted_vllm/${MODEL}"
    --api-base "${GEN_API_BASE}"
)
if [ -n "${TAG}" ]; then
    GEPA_ARGS+=(--tag "${TAG}")
fi
if [ -n "${TRAIN_LIMIT}" ]; then
    GEPA_ARGS+=(--train-limit "${TRAIN_LIMIT}")
fi
if [ -n "${VAL_LIMIT}" ]; then
    GEPA_ARGS+=(--val-limit "${VAL_LIMIT}")
fi
if [ -n "${TEST_LIMIT}" ]; then
    GEPA_ARGS+=(--test-limit "${TEST_LIMIT}")
fi
if [ -n "${DATA_PATH}" ]; then
    GEPA_ARGS+=(--data-path "${DATA_PATH}")
fi

echo "[mbpp] Running: uv run python -m examples.mbpp.main ${GEPA_ARGS[*]}"
uv run python -m examples.mbpp.main "${GEPA_ARGS[@]}"
EXIT_CODE=$?

# --- Cleanup --------------------------------------------------------------
if [ -n "${VLLM_PID:-}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[mbpp] Stopping vLLM server (pid ${VLLM_PID})..."
    kill "${VLLM_PID}" || true
    wait "${VLLM_PID}" 2>/dev/null || true
fi

echo "[mbpp] Done with exit code ${EXIT_CODE} at $(date)"
exit "${EXIT_CODE}"
