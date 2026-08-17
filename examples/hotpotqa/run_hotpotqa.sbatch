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
#   sbatch examples/hotpotqa/run_hotpotqa.sbatch
#
# Override defaults:
#   sbatch --export=ALL,MODEL=Qwen3.6-35B-A3B,MAX_METRIC_CALLS=100 \
#       examples/hotpotqa/run_hotpotqa.sbatch

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
MODEL="${MODEL:-Qwen3.5-9B}"
# Unique per-job port: della nodes host 8 GPUs, so multiple single-GPU jobs can
# share a node; a fixed port makes co-located jobs cross-talk and cross-kill.
GEN_PORT="${GEN_PORT:-$((10000 + ${SLURM_JOB_ID:-0} % 5000))}"
SEED="${SEED:-0}"
NUM_SEEDS="${NUM_SEEDS:-1}"
HOLDOUT_RATIO="${HOLDOUT_RATIO:-0.0}"
TAU="${TAU:-}"
K="${K:-5}"
COMPUTE_VENDI="${COMPUTE_VENDI:-0}"
COMPUTE_BEHAVIORAL="${COMPUTE_BEHAVIORAL:-0}"
MAX_METRIC_CALLS="${MAX_METRIC_CALLS:-200}"
CONDITION="${CONDITION:-both}"
GEN_GMU="${GEN_GMU:-0.90}"
GEN_MAX_LEN="${GEN_MAX_LEN:-32768}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"
# POSIT repo location (for its vLLM install).
POSIT_DIR="${POSIT_DIR:-/home/${USER}/posit}"
DATA_PATH="${DATA_PATH:-}"

# --- Keep all large I/O on scratch ----------------------------------------
SCRATCH_BASE="${SCRATCH_BASE:-/scratch/gpfs/BSTEWART/${USER}/gepa}"
export XDG_CACHE_HOME="${SCRATCH_BASE}/.cache"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export VLLM_CACHE_ROOT="${SCRATCH_BASE}/.cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_BASE}/.cache/torchinductor"
export TRITON_CACHE_DIR="${SCRATCH_BASE}/.cache/triton"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
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


# --- Model routing: openrouter/* via OpenRouter API, hosted_vllm/* via local vLLM ---
if [[ "${MODEL}" == ${OPENROUTER_PREFIX}* ]]; then
    SOLVER_MODEL="${MODEL}"
    REFLECTION_MODEL="${MODEL}"
    echo "[openrouter] Using OpenRouter models: solver=${SOLVER_MODEL} reflection=${REFLECTION_MODEL}"
    if [[ -z "${OPENROUTER_API_KEY}" ]]; then
        for keyfile in "${HOME}/.config/openrouter/key" "${HOME}/.openrouter_key" "./.openrouter_key"; do
            if [[ -f "${keyfile}" ]]; then
                export OPENROUTER_API_KEY="$(cat "${keyfile}" | tr -d ' \n')"
                echo "[openrouter] Loaded API key from ${keyfile}"
                break
            fi
        done
    fi
    if [[ -z "${OPENROUTER_API_KEY}" ]]; then
        echo "ERROR: OPENROUTER_API_KEY not set for openrouter/* model ${MODEL}. Set via: export OPENROUTER_API_KEY=sk-or-v1-... (see docs/OPENROUTER_SETUP.md)" >&2
        exit 1
    fi
    export OPENROUTER_API_KEY OPENROUTER_REFERER OPENROUTER_TITLE
    USE_OPENROUTER=1
    API_BASE=""  # not used for OpenRouter
    GEN_PORT=""
else
    # hosted_vllm path: will start local vLLM server below
    SOLVER_MODEL="${HOSTED_VLLM_PREFIX}${MODEL}"
    REFLECTION_MODEL="${HOSTED_VLLM_PREFIX}${MODEL}"
    # Allow explicit override if SOLVER_MODEL already set to openrouter/*
    if [[ "${SOLVER_MODEL}" == ${OPENROUTER_PREFIX}* ]]; then
        USE_OPENROUTER=1
        API_BASE=""
    else
        USE_OPENROUTER=0
        API_BASE="http://localhost:${GEN_PORT}/v1"
    fi
fi

if [[ "${USE_OPENROUTER:-0}" != "1" ]]; then
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

fi  # end USE_OPENROUTER check (skip vLLM when using OpenRouter)

# --- Run the GEPA experiment (using GEPA's own venv) ---------------------
# litellm talks to vLLM via the openai/ prefix + api_base.
# Set OPENAI_API_KEY=EMPTY so litellm doesn't reject the request.
export OPENAI_API_KEY="EMPTY"
SOLVER_MODEL="${HOSTED_VLLM_PREFIX}${MODEL}"
REFLECTION_MODEL="${HOSTED_VLLM_PREFIX}${MODEL}"
API_BASE="http://localhost:${GEN_PORT}/v1"

DATA_PATH_ARG=()
[[ -n "${DATA_PATH}" ]] && DATA_PATH_ARG=(--data-path "${DATA_PATH}")

if [[ "${USE_OPENROUTER:-0}" == "1" ]]; then
    API_BASE_ARGS=()
    echo "[openrouter] Running via OpenRouter API (no local api_base)"
else
    API_BASE_ARGS=(--api-base "${API_BASE}")
    echo "[vllm] Running via local vLLM at ${API_BASE}"
fi

echo "==> running GEPA experiment: condition=${CONDITION} max_metric_calls=${MAX_METRIC_CALLS}"
echo "==> solver=${SOLVER_MODEL} reflection=${REFLECTION_MODEL} api_base=${API_BASE}"

"${PY}" -m examples.hotpotqa.main \
    --solver-model "${SOLVER_MODEL}" \
    --reflection-model "${REFLECTION_MODEL}" \
    "${API_BASE_ARGS[@]}" \
    --max-metric-calls "${MAX_METRIC_CALLS}" \
    --condition "${CONDITION}" \
    "${DATA_PATH_ARG[@]}"
