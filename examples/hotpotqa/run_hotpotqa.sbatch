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
# Qwen3.8-27B four-condition run:
#   sbatch --export=ALL,MODEL_PROFILE=qwen3.8-27b examples/hotpotqa/run_hotpotqa.sbatch
# DeepSeek V4 Flash four-condition run:
#   sbatch --export=ALL,MODEL_PROFILE=deepseek-v4-flash examples/hotpotqa/run_hotpotqa.sbatch
#
# The default run uses the paper's 6,871-call budget, two-hop program, seven
# documents per hop, seed 0, 32 evaluator workers, and merge disabled.

#SBATCH --job-name=gepa-hotpotqa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=gepa-hotpotqa-%j.log

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1

# --- Knobs (override with sbatch --export=ALL,VAR=...) --------------------
MODEL_STORAGE="${MODEL_STORAGE:-/projects/BSTEWART/model_storage}"
MODEL_PROFILE="${MODEL_PROFILE:-qwen3.8-27b}"
MODEL="${MODEL:-}"
SOLVER_MODEL_PATH="${SOLVER_MODEL_PATH:-}"
# Unique per-job port: della nodes host 8 GPUs, so multiple single-GPU jobs can
# share a node; a fixed port makes co-located jobs cross-talk and cross-kill.
GEN_PORT="${GEN_PORT:-$((10000 + ${SLURM_JOB_ID:-0} % 5000))}"
MAX_METRIC_CALLS="${MAX_METRIC_CALLS:-6871}"
CONDITION="${CONDITION:-all}"
MERGE="${MERGE:-0}"
PROGRAM="${PROGRAM:-2stage}"
SEED_STYLE="${SEED_STYLE:-structured}"
EXPERIMENT_SEED="${EXPERIMENT_SEED:-0}"
MAX_WORKERS="${MAX_WORKERS:-32}"
RETRIEVAL_K="${RETRIEVAL_K:-7}"
GEN_GMU="${GEN_GMU:-0.90}"
GEN_MAX_LEN="${GEN_MAX_LEN:-32768}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"
# POSIT repo location (for its vLLM install).
POSIT_DIR="${POSIT_DIR:-/home/${USER}/posit}"
DATA_PATH="${DATA_PATH:-}"

LOCAL_SOLVER=0
case "${MODEL_PROFILE}" in
    qwen3.8-27b)
        MODEL="${MODEL:-Qwen3.8-27B}"
        SOLVER_SERVED_NAME="Qwen/Qwen3.8-27B"
        SOLVER_MODEL="hosted_vllm/Qwen/Qwen3.8-27B"
        if [[ -z "${SOLVER_MODEL_PATH}" ]]; then
            SOLVER_MODEL_PATH="${MODEL_STORAGE}/${MODEL}"
        fi
        SOLVER_API_BASE="http://localhost:${GEN_PORT}/v1"
        LOCAL_SOLVER=1
        ;;
    deepseek-v4-flash)
        MODEL=""
        SOLVER_MODEL_PATH=""
        SOLVER_SERVED_NAME=""
        SOLVER_MODEL="deepseek/deepseek-v4-flash"
        SOLVER_API_BASE=""
        REFLECTION_API_BASE=""
        ;;
    *)
        echo "ERROR: MODEL_PROFILE must be qwen3.8-27b or deepseek-v4-flash" >&2
        exit 1
        ;;
esac
REFLECTION_MODEL="${SOLVER_MODEL}"
REFLECTION_API_BASE="${SOLVER_API_BASE}"

# --- Keep all large I/O on scratch ----------------------------------------
SCRATCH_BASE="${SCRATCH_BASE:-/scratch/gpfs/BSTEWART/${USER}/gepa}"
WIKI17_DIR="${WIKI17_DIR:-${SCRATCH_BASE}/.cache/gepa/wiki17}"
export XDG_CACHE_HOME="${SCRATCH_BASE}/.cache"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export DSPY_CACHEDIR="${SCRATCH_BASE}/.cache/dspy"
export VLLM_CACHE_ROOT="${SCRATCH_BASE}/.cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_BASE}/.cache/torchinductor"
export TRITON_CACHE_DIR="${SCRATCH_BASE}/.cache/triton"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export JAX_PLATFORMS=cpu
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_USE_FLASHINFER_SAMPLER=0

LOG_DIR="${SCRATCH_BASE}/logs"
mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}" "${DSPY_CACHEDIR}" "${VLLM_CACHE_ROOT}" \
    "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${LOG_DIR}"

# --- Verify both venvs exist ----------------------------------------------
PY=".venv/bin/python"
VLLM_BIN="${POSIT_DIR}/src/.venv/bin/vllm"
[[ -x "${PY}" ]] || { echo "ERROR: missing ${PY} -- run scripts/della/build_env.sh" >&2; exit 1; }
if [[ "${LOCAL_SOLVER}" == "1" && ! -x "${VLLM_BIN}" ]]; then
    echo "ERROR: missing ${VLLM_BIN} -- build the POSIT venv first" >&2
    exit 1
fi

if ! "${PY}" -m examples.common.wiki17_bm25 verify --root "${WIKI17_DIR}" >/dev/null; then
    echo "ERROR: Wiki-2017 is not prepared at ${WIKI17_DIR}; run scripts/della/build_env.sh first" >&2
    exit 1
fi

if [[ "${LOCAL_SOLVER}" == "1" ]]; then
    GEN_LOG="${LOG_DIR}/gen-${SLURM_JOB_ID:-local}.log"
    echo "==> serving solver ${SOLVER_MODEL_PATH} as ${SOLVER_SERVED_NAME} on :${GEN_PORT} (GPU 0, util ${GEN_GMU}); log: ${GEN_LOG}"

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
fi

# --- Run the GEPA experiment (using GEPA's own venv) ---------------------
if [[ "${MODEL_PROFILE}" == "deepseek-v4-flash" && -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "ERROR: set DEEPSEEK_API_KEY for the DeepSeek V4 Flash run" >&2
    exit 1
fi
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
DATA_PATH_ARG=()
[[ -n "${DATA_PATH}" ]] && DATA_PATH_ARG=(--data-path "${DATA_PATH}")
SOLVER_API_ARG=()
[[ -n "${SOLVER_API_BASE}" ]] && SOLVER_API_ARG=(--solver-api-base "${SOLVER_API_BASE}")
REFLECTION_API_ARG=()
[[ -n "${REFLECTION_API_BASE}" ]] && REFLECTION_API_ARG=(--reflection-api-base "${REFLECTION_API_BASE}")
MERGE_ARG=()
case "${MERGE}" in
    0) ;;
    1) MERGE_ARG=(--merge) ;;
    *)
        echo "ERROR: MERGE must be 0 or 1" >&2
        exit 1
        ;;
esac

echo "==> running GEPA experiment: condition=${CONDITION} merge=${MERGE} max_metric_calls=${MAX_METRIC_CALLS} seed=${EXPERIMENT_SEED}"
echo "==> solver=${SOLVER_MODEL} solver_api_base=${SOLVER_API_BASE:-provider-default}"
echo "==> reflection=${REFLECTION_MODEL} reflection_api_base=${REFLECTION_API_BASE:-provider-default}"
echo "==> retrieval=Wiki-2017/BM25 k=${RETRIEVAL_K} workers=${MAX_WORKERS} root=${WIKI17_DIR}"

"${PY}" -m examples.hotpotqa.main \
    --solver-model "${SOLVER_MODEL}" \
    --reflection-model "${REFLECTION_MODEL}" \
    "${SOLVER_API_ARG[@]}" \
    "${REFLECTION_API_ARG[@]}" \
    --max-metric-calls "${MAX_METRIC_CALLS}" \
    --condition "${CONDITION}" \
    "${MERGE_ARG[@]}" \
    --program "${PROGRAM}" \
    --seed-style "${SEED_STYLE}" \
    --seed "${EXPERIMENT_SEED}" \
    --max-workers "${MAX_WORKERS}" \
    --wiki17-dir "${WIKI17_DIR}" \
    --retrieval-k "${RETRIEVAL_K}" \
    "${DATA_PATH_ARG[@]}"
