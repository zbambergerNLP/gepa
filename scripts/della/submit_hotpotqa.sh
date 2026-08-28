#!/bin/bash
# Submit the GEPA HotpotQA experiment to della from your laptop.
#
# Usage:
#   scripts/della/submit_hotpotqa.sh
#
# Use MODEL_PROFILE=qwen3.8-27b or MODEL_PROFILE=deepseek-v4-flash. Each
# profile uses the same model for the student and proposer. The default runs
# vanilla, uniform stateless, verbalized stateless, and ReAct V2 conditions.
# Set MERGE=1 to run the same four conditions with merge enabled.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "ERROR: ${ENV_FILE} not found." >&2
    exit 1
fi

set -a; source "${ENV_FILE}"; set +a

# Tunable knobs (env overrides).
MODEL_PROFILE="${MODEL_PROFILE:-qwen3.8-27b}"
MODEL="${MODEL:-}"
SOLVER_MODEL_PATH="${SOLVER_MODEL_PATH:-}"
DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
MAX_METRIC_CALLS="${MAX_METRIC_CALLS:-6871}"
CONDITION="${CONDITION:-all}"
MERGE="${MERGE:-0}"
EXPERIMENT_SEED="${EXPERIMENT_SEED:-0}"
MAX_WORKERS="${MAX_WORKERS:-32}"
RETRIEVAL_K="${RETRIEVAL_K:-7}"
WIKI17_DIR="${WIKI17_DIR:-${SCRATCH_BASE}/.cache/gepa/wiki17}"
TIME="${TIME:-24:00:00}"
NO_SYNC="${NO_SYNC:-}"

if [[ "${MERGE}" != "0" && "${MERGE}" != "1" ]]; then
    echo "ERROR: MERGE must be 0 or 1" >&2
    exit 1
fi

case "${MODEL_PROFILE}" in
    qwen3.8-27b)
        MODEL="${MODEL:-Qwen3.8-27B}"
        SOLVER_SERVED_NAME="Qwen/Qwen3.8-27B"
        SOLVER_MODEL="hosted_vllm/Qwen/Qwen3.8-27B"
        SOLVER_API_BASE=""
        REFLECTION_API_BASE=""
        ;;
    deepseek-v4-flash)
        MODEL=""
        SOLVER_MODEL_PATH=""
        SOLVER_SERVED_NAME=""
        SOLVER_API_BASE=""
        REFLECTION_API_BASE=""
        [[ -n "${DEEPSEEK_API_KEY}" ]] || {
            echo "ERROR: set DEEPSEEK_API_KEY for the DeepSeek V4 Flash run" >&2
            exit 1
        }
        SOLVER_MODEL="deepseek/deepseek-v4-flash"
        ;;
    *)
        echo "ERROR: MODEL_PROFILE must be qwen3.8-27b or deepseek-v4-flash" >&2
        exit 1
        ;;
esac
REFLECTION_MODEL="${SOLVER_MODEL}"
REFLECTION_API_BASE="${SOLVER_API_BASE}"

# Step 1: sync code (unless NO_SYNC=1).
if [[ -z "${NO_SYNC}" ]]; then
    echo "==> syncing code"
    "${SCRIPT_DIR}/sync_to_della.sh"
fi

# Step 2: submit sbatch on della login node.
echo "==> submitting HotpotQA: profile=${MODEL_PROFILE} solver=${SOLVER_MODEL} reflection=${REFLECTION_MODEL}"
echo "==> method: Wiki-2017/BM25 k=${RETRIEVAL_K} seed=${EXPERIMENT_SEED} workers=${MAX_WORKERS} budget=${MAX_METRIC_CALLS} condition=${CONDITION} merge=${MERGE}"

sshpass -p "${REMOTE_PASSWORD}" ssh -o StrictHostKeyChecking=no \
    "${REMOTE_USER}@${REMOTE_HOST}" bash -l <<REMOTE_SCRIPT
set -euo pipefail
cd "${REMOTE_DIR}"

export DSPY_CACHEDIR="${SCRATCH_BASE}/.cache/dspy"
mkdir -p "\${DSPY_CACHEDIR}"
.venv/bin/python - <<'PY'
from examples.hotpotqa.utils import validate_hotpotqa_dspy_runtime

try:
    validate_hotpotqa_dspy_runtime()
except RuntimeError as exc:
    raise SystemExit(f"{exc} Run scripts/della/build_env.sh first.") from exc
PY

if ! .venv/bin/python -m examples.common.wiki17_bm25 verify --root "${WIKI17_DIR}" >/dev/null; then
    echo "ERROR: Wiki-2017 is not prepared at ${WIKI17_DIR}; run scripts/della/build_env.sh first" >&2
    exit 1
fi

sbatch \
    --partition="${GPU_PARTITION}" \
    --time="${TIME}" \
    --export="ALL,MODEL_PROFILE=${MODEL_PROFILE},MODEL=${MODEL},SOLVER_MODEL_PATH=${SOLVER_MODEL_PATH},SOLVER_SERVED_NAME=${SOLVER_SERVED_NAME},SOLVER_MODEL=${SOLVER_MODEL},SOLVER_API_BASE=${SOLVER_API_BASE},REFLECTION_MODEL=${REFLECTION_MODEL},REFLECTION_API_BASE=${REFLECTION_API_BASE},DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY},MAX_METRIC_CALLS=${MAX_METRIC_CALLS},CONDITION=${CONDITION},MERGE=${MERGE},EXPERIMENT_SEED=${EXPERIMENT_SEED},MAX_WORKERS=${MAX_WORKERS},RETRIEVAL_K=${RETRIEVAL_K},WIKI17_DIR=${WIKI17_DIR},MODEL_STORAGE=${MODEL_STORAGE},SCRATCH_BASE=${SCRATCH_BASE}" \
    examples/hotpotqa/run_hotpotqa.sbatch

echo "==> job submitted. Check status with: squeue -u ${REMOTE_USER}"
REMOTE_SCRIPT
