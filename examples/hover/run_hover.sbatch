#!/bin/bash
# Run the GEPA HoVer evaluation on della.
#
# Serves a generator model via vLLM (from the POSIT venv, which has vLLM
# installed), then runs the GEPA optimization experiment against it.
# Uses the GEPA artifact's official HoVer v1.1 split, frozen Wiki-2017 BM25
# retrieval, three-hop program, and 7/7/10 retrieval depths.
#
# Prerequisites: POSIT repo, GEPA venv, HoVer release, and Wiki-2017 index
# prepared on scratch by scripts/della/build_env.sh.
#
# Qwen3.8-27B run:
#   sbatch --export=ALL,MODEL_PROFILE=qwen3.8-27b examples/hover/run_hover.sbatch
# DeepSeek V4 Flash run:
#   sbatch --export=ALL,MODEL_PROFILE=deepseek-v4-flash examples/hover/run_hover.sbatch
#
# Scaled run (48h, like ifbench scaled):
#   sbatch --time=48:00:00 --export=ALL,MAX_METRIC_CALLS=7051 examples/hover/run_hover.sbatch
#   # or via TIME env: TIME=48:00:00 sbatch examples/hover/run_hover.sbatch
#
# Mini test run (20/20/20, small budget):
#   sbatch --export=ALL,TRAIN_LIMIT=20,VAL_LIMIT=20,TEST_LIMIT=20,MAX_METRIC_CALLS=150 \
#       --time=02:00:00 examples/hover/run_hover.sbatch
#

#SBATCH --job-name=gepa-hover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=gepa-hover-%j.log

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
MAX_METRIC_CALLS="${MAX_METRIC_CALLS:-7051}"
CONDITION="${CONDITION:-both}"
PROGRAM="${PROGRAM:-2stage}"
SEED_STYLE="${SEED_STYLE:-structured}"
TAG="${TAG:-}"
TRAIN_LIMIT="${TRAIN_LIMIT:-}"
VAL_LIMIT="${VAL_LIMIT:-}"
TEST_LIMIT="${TEST_LIMIT:-}"
SMOKE="${SMOKE:-0}"
EXPERIMENT_SEED="${EXPERIMENT_SEED:-0}"
MAX_WORKERS="${MAX_WORKERS:-32}"
RETRIEVAL_K="${RETRIEVAL_K:-7}"
FINAL_RETRIEVAL_K="${FINAL_RETRIEVAL_K:-10}"
GEN_GMU="${GEN_GMU:-0.90}"
GEN_MAX_LEN="${GEN_MAX_LEN:-32768}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"
# POSIT repo location (for its vLLM install).
POSIT_DIR="${POSIT_DIR:-/home/${USER}/posit}"

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
HOVER_DATA_DIR="${HOVER_DATA_DIR:-${SCRATCH_BASE}/.cache/gepa/hover}"
export XDG_CACHE_HOME="${SCRATCH_BASE}/.cache"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export VLLM_CACHE_ROOT="${SCRATCH_BASE}/.cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_BASE}/.cache/torchinductor"
export TRITON_CACHE_DIR="${SCRATCH_BASE}/.cache/triton"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export JAX_PLATFORMS=cpu
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_USE_FLASHINFER_SAMPLER=0

LOG_DIR="${SCRATCH_BASE}/logs"
mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}" "${VLLM_CACHE_ROOT}" \
    "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${LOG_DIR}" "${HOVER_DATA_DIR}"

# --- Verify both venvs exist ----------------------------------------------
PY=".venv/bin/python"
VLLM_BIN="${POSIT_DIR}/src/.venv/bin/vllm"
[[ -x "${PY}" ]] || { echo "ERROR: missing ${PY} -- run scripts/della/build_env.sh" >&2; exit 1; }
if [[ "${LOCAL_SOLVER}" == "1" && ! -x "${VLLM_BIN}" ]]; then
    echo "ERROR: missing ${VLLM_BIN} -- build the POSIT venv first" >&2
    exit 1
fi

echo "==> verifying the pinned HoVer release"
if ! HOVER_DATA_DIR="${HOVER_DATA_DIR}" "${PY}" - <<'PY'
import os

from examples.hover.utils import load_hover_dataset

load_hover_dataset(seed=0, data_dir=os.environ["HOVER_DATA_DIR"])
PY
then
    echo "ERROR: HoVer v1.1 is not prepared at ${HOVER_DATA_DIR}; run scripts/della/build_env.sh first" >&2
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
        --max-num-seqs 32 \
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
# Flush python output line-by-line so the SLURM log shows live progress.
export PYTHONUNBUFFERED=1
LIMIT_ARGS=()
[[ -n "${TRAIN_LIMIT}" ]] && LIMIT_ARGS+=(--train-limit "${TRAIN_LIMIT}")
[[ -n "${VAL_LIMIT}" ]] && LIMIT_ARGS+=(--val-limit "${VAL_LIMIT}")
[[ -n "${TEST_LIMIT}" ]] && LIMIT_ARGS+=(--test-limit "${TEST_LIMIT}")
[[ -n "${TAG}" ]] && LIMIT_ARGS+=(--tag "${TAG}")
[[ "${SMOKE}" == "1" ]] && LIMIT_ARGS+=(--smoke)
SOLVER_API_ARG=()
[[ -n "${SOLVER_API_BASE}" ]] && SOLVER_API_ARG=(--solver-api-base "${SOLVER_API_BASE}")
REFLECTION_API_ARG=()
[[ -n "${REFLECTION_API_BASE}" ]] && REFLECTION_API_ARG=(--reflection-api-base "${REFLECTION_API_BASE}")

echo "==> running GEPA HoVer experiment: condition=${CONDITION} program=${PROGRAM} max_metric_calls=${MAX_METRIC_CALLS} seed=${EXPERIMENT_SEED}"
echo "==> limits: train=${TRAIN_LIMIT:-full} val=${VAL_LIMIT:-full} test=${TEST_LIMIT:-full}"
echo "==> solver=${SOLVER_MODEL} solver_api_base=${SOLVER_API_BASE:-provider-default}"
echo "==> reflection=${REFLECTION_MODEL} reflection_api_base=${REFLECTION_API_BASE:-provider-default}"
echo "==> retrieval=Wiki-2017/BM25 k=${RETRIEVAL_K}/${RETRIEVAL_K}/${FINAL_RETRIEVAL_K} workers=${MAX_WORKERS} root=${WIKI17_DIR}"

"${PY}" -m examples.hover.main \
    --solver-model "${SOLVER_MODEL}" \
    --reflection-model "${REFLECTION_MODEL}" \
    "${SOLVER_API_ARG[@]}" \
    "${REFLECTION_API_ARG[@]}" \
    --max-metric-calls "${MAX_METRIC_CALLS}" \
    --condition "${CONDITION}" \
    --program "${PROGRAM}" \
    --seed-style "${SEED_STYLE}" \
    --data-dir "${HOVER_DATA_DIR}" \
    --seed "${EXPERIMENT_SEED}" \
    --max-workers "${MAX_WORKERS}" \
    --wiki17-dir "${WIKI17_DIR}" \
    --retrieval-k "${RETRIEVAL_K}" \
    --final-retrieval-k "${FINAL_RETRIEVAL_K}" \
    "${LIMIT_ARGS[@]}"
