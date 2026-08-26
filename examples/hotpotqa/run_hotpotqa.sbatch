#!/bin/bash
# Run the GEPA HotpotQA evaluation on della.
#
# Serves a generator model via vLLM (from the POSIT venv, which has vLLM
# installed), then runs the GEPA optimization experiment against it.
#
# Prerequisites:
#   1. POSIT repo built on della at /home/${USER}/posit (with vLLM in its venv)
#   2. GEPA repo synced and venv built via scripts/della/build_env.sh
#
# Usage:
#   sbatch --export=ALL,MODEL=<local-solver-dir>,REFLECTION_MODEL=<provider/model> \
#       examples/hotpotqa/run_hotpotqa.sbatch
#
# Separate solver/proposer roles:
#   sbatch --export=ALL,MODEL=<local-solver-dir>,REFLECTION_MODEL=<provider/model>,MAX_METRIC_CALLS=100 \
#       examples/hotpotqa/run_hotpotqa.sbatch
# Same-model smoke mode must be requested explicitly with SAME_MODEL=1.

#SBATCH --job-name=gepa-hotpotqa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=gepa-hotpotqa-%j.log

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1

# --- Knobs (override with sbatch --export=ALL,VAR=...) --------------------
MODEL_STORAGE="${MODEL_STORAGE:-/projects/BSTEWART/model_storage}"
MODEL="${MODEL:-}"
SOLVER_MODEL_PATH="${SOLVER_MODEL_PATH:-}"
SOLVER_SERVED_NAME="${SOLVER_SERVED_NAME:-${MODEL}}"
SOLVER_MODEL="${SOLVER_MODEL:-}"
REFLECTION_MODEL="${REFLECTION_MODEL:-}"
REFLECTION_API_BASE="${REFLECTION_API_BASE:-}"
SAME_MODEL="${SAME_MODEL:-0}"
# Unique per-job port: della nodes host 8 GPUs, so multiple single-GPU jobs can
# share a node; a fixed port makes co-located jobs cross-talk and cross-kill.
GEN_PORT="${GEN_PORT:-$((10000 + ${SLURM_JOB_ID:-0} % 5000))}"
MAX_METRIC_CALLS="${MAX_METRIC_CALLS:-200}"
CONDITION="${CONDITION:-both}"
GEN_GMU="${GEN_GMU:-0.90}"
GEN_MAX_LEN="${GEN_MAX_LEN:-32768}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"
# POSIT repo location (for its vLLM install).
POSIT_DIR="${POSIT_DIR:-/home/${USER}/posit}"
DATA_PATH="${DATA_PATH:-}"
WIKIPEDIA_ENDPOINT="${WIKIPEDIA_ENDPOINT:-https://en.wikipedia.org/w/api.php}"
RETRIEVAL_K="${RETRIEVAL_K:-7}"

if [[ -z "${SOLVER_MODEL_PATH}" ]]; then
    [[ -n "${MODEL}" ]] || {
        echo "ERROR: set MODEL (under MODEL_STORAGE) or an absolute SOLVER_MODEL_PATH" >&2
        exit 1
    }
    SOLVER_MODEL_PATH="${MODEL_STORAGE}/${MODEL}"
fi
[[ -n "${SOLVER_SERVED_NAME}" ]] || {
    echo "ERROR: set SOLVER_SERVED_NAME when SOLVER_MODEL_PATH is used without MODEL" >&2
    exit 1
}
SOLVER_MODEL="${SOLVER_MODEL:-hosted_vllm/${SOLVER_SERVED_NAME}}"
SOLVER_API_BASE="${SOLVER_API_BASE:-http://localhost:${GEN_PORT}/v1}"
if [[ "${SAME_MODEL}" == "1" ]]; then
    [[ -z "${REFLECTION_MODEL}" || "${REFLECTION_MODEL}" == "${SOLVER_MODEL}" ]] || {
        echo "ERROR: SAME_MODEL=1 conflicts with REFLECTION_MODEL=${REFLECTION_MODEL}" >&2
        exit 1
    }
    REFLECTION_MODEL="${SOLVER_MODEL}"
    REFLECTION_API_BASE="${SOLVER_API_BASE}"
elif [[ -z "${REFLECTION_MODEL}" ]]; then
    echo "ERROR: set REFLECTION_MODEL for the proposer, or opt into SAME_MODEL=1" >&2
    exit 1
fi

# --- Keep all large I/O on scratch ----------------------------------------
SCRATCH_BASE="${SCRATCH_BASE:-/scratch/gpfs/BSTEWART/${USER}/gepa}"
WIKIPEDIA_CACHE="${WIKIPEDIA_CACHE:-${SCRATCH_BASE}/.cache/gepa/wikipedia.sqlite3}"
export XDG_CACHE_HOME="${SCRATCH_BASE}/.cache"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export VLLM_CACHE_ROOT="${SCRATCH_BASE}/.cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_BASE}/.cache/torchinductor"
export TRITON_CACHE_DIR="${SCRATCH_BASE}/.cache/triton"
# Fullwiki must be downloadable on the first production run; callers with a
# warm cache can still opt into offline mode through the environment.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
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
echo "==> serving solver ${SOLVER_MODEL_PATH} as ${SOLVER_SERVED_NAME} on :${GEN_PORT} (GPU 0, util ${GEN_GMU}); log: ${GEN_LOG}"

# Load CUDA toolkit for JIT compilation (needed by some models).
export PATH="${POSIT_DIR}/src/.venv/bin:${PATH}"
CUDA_TAG="$(${POSIT_DIR}/src/.venv/bin/python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 13.0)"
source /usr/share/Modules/init/bash 2>/dev/null || true
module load "cudatoolkit/${CUDA_TAG}" 2>/dev/null \
    || module load cudatoolkit/13.0 2>/dev/null || true

env CUDA_VISIBLE_DEVICES=0 "${VLLM_BIN}" serve "${SOLVER_MODEL_PATH}" \
    --served-model-name "${SOLVER_SERVED_NAME}" \
    --reasoning-parser qwen3 \
    --host 0.0.0.0 \
    --port "${GEN_PORT}" \
    --gpu-memory-utilization "${GEN_GMU}" \
    --max-model-len "${GEN_MAX_LEN}" \
    --max-num-seqs 16 \
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
# Preserve remote-provider credentials; EMPTY is sufficient for local vLLM.
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
DATA_PATH_ARG=()
[[ -n "${DATA_PATH}" ]] && DATA_PATH_ARG=(--data-path "${DATA_PATH}")
REFLECTION_API_ARG=()
[[ -n "${REFLECTION_API_BASE}" ]] && REFLECTION_API_ARG=(--reflection-api-base "${REFLECTION_API_BASE}")

echo "==> running GEPA experiment: condition=${CONDITION} max_metric_calls=${MAX_METRIC_CALLS}"
echo "==> solver=${SOLVER_MODEL} solver_api_base=${SOLVER_API_BASE}"
echo "==> reflection=${REFLECTION_MODEL} reflection_api_base=${REFLECTION_API_BASE:-provider-default}"

"${PY}" -m examples.hotpotqa.main \
    --solver-model "${SOLVER_MODEL}" \
    --reflection-model "${REFLECTION_MODEL}" \
    --solver-api-base "${SOLVER_API_BASE}" \
    "${REFLECTION_API_ARG[@]}" \
    --max-metric-calls "${MAX_METRIC_CALLS}" \
    --condition "${CONDITION}" \
    --wikipedia-endpoint "${WIKIPEDIA_ENDPOINT}" \
    --wikipedia-cache "${WIKIPEDIA_CACHE}" \
    --retrieval-k "${RETRIEVAL_K}" \
    "${DATA_PATH_ARG[@]}"
