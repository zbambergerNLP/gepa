#!/bin/bash
# Submit the GEPA HotpotQA experiment to della from your laptop.
#
# Usage:
#   scripts/della/submit_hotpotqa.sh
#
# Configure separate solver/proposer roles via env:
#   MODEL=<local-solver-dir> REFLECTION_MODEL=<provider/model> MAX_METRIC_CALLS=100 CONDITION=both \
#       scripts/della/submit_hotpotqa.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "ERROR: ${ENV_FILE} not found." >&2
    exit 1
fi

set -a; source "${ENV_FILE}"; set +a

# Tunable knobs (env overrides).
MODEL="${MODEL:-}"
SOLVER_MODEL_PATH="${SOLVER_MODEL_PATH:-}"
SOLVER_SERVED_NAME="${SOLVER_SERVED_NAME:-${MODEL}}"
SOLVER_MODEL="${SOLVER_MODEL:-}"
SOLVER_API_BASE="${SOLVER_API_BASE:-}"
REFLECTION_MODEL="${REFLECTION_MODEL:-}"
REFLECTION_API_BASE="${REFLECTION_API_BASE:-}"
SAME_MODEL="${SAME_MODEL:-0}"
MAX_METRIC_CALLS="${MAX_METRIC_CALLS:-200}"
CONDITION="${CONDITION:-both}"
TIME="${TIME:-02:00:00}"
NO_SYNC="${NO_SYNC:-}"

[[ -n "${MODEL}" || -n "${SOLVER_MODEL_PATH}" ]] || {
    echo "ERROR: set MODEL or SOLVER_MODEL_PATH for the local solver" >&2
    exit 1
}
[[ -n "${SOLVER_SERVED_NAME}" ]] || {
    echo "ERROR: set SOLVER_SERVED_NAME when SOLVER_MODEL_PATH is used without MODEL" >&2
    exit 1
}
[[ "${SAME_MODEL}" == "0" || "${SAME_MODEL}" == "1" ]] || {
    echo "ERROR: SAME_MODEL must be 0 or 1" >&2
    exit 1
}
if [[ "${SAME_MODEL}" != "1" && -z "${REFLECTION_MODEL}" ]]; then
    echo "ERROR: set REFLECTION_MODEL for the proposer, or opt into SAME_MODEL=1" >&2
    exit 1
fi

# Step 1: sync code (unless NO_SYNC=1).
if [[ -z "${NO_SYNC}" ]]; then
    echo "==> syncing code"
    "${SCRIPT_DIR}/sync_to_della.sh"
fi

# Step 2: submit sbatch on della login node.
echo "==> submitting job: solver=${SOLVER_MODEL:-hosted_vllm/${SOLVER_SERVED_NAME}} reflection=${REFLECTION_MODEL:-same-model} max_metric_calls=${MAX_METRIC_CALLS} condition=${CONDITION}"

sshpass -p "${REMOTE_PASSWORD}" ssh -o StrictHostKeyChecking=no \
    "${REMOTE_USER}@${REMOTE_HOST}" bash -l <<REMOTE_SCRIPT
set -euo pipefail
cd "${REMOTE_DIR}"

sbatch \
    --partition="${GPU_PARTITION}" \
    --time="${TIME}" \
    --export="ALL,MODEL=${MODEL},SOLVER_MODEL_PATH=${SOLVER_MODEL_PATH},SOLVER_SERVED_NAME=${SOLVER_SERVED_NAME},SOLVER_MODEL=${SOLVER_MODEL},SOLVER_API_BASE=${SOLVER_API_BASE},REFLECTION_MODEL=${REFLECTION_MODEL},REFLECTION_API_BASE=${REFLECTION_API_BASE},SAME_MODEL=${SAME_MODEL},MAX_METRIC_CALLS=${MAX_METRIC_CALLS},CONDITION=${CONDITION},MODEL_STORAGE=${MODEL_STORAGE},SCRATCH_BASE=${SCRATCH_BASE}" \
    examples/hotpotqa/run_hotpotqa.sbatch

echo "==> job submitted. Check status with: squeue -u ${REMOTE_USER}"
REMOTE_SCRIPT
