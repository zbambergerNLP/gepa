#!/bin/bash
# Submit the GEPA HotpotQA experiment to della from your laptop.
#
# Usage:
#   scripts/della/submit_hotpotqa.sh
#
# Override defaults via env:
#   MODEL=Qwen3.6-35B-A3B MAX_METRIC_CALLS=100 CONDITION=action \
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
MODEL="${MODEL:-Qwen3.5-9B}"
MAX_METRIC_CALLS="${MAX_METRIC_CALLS:-200}"
CONDITION="${CONDITION:-both}"
TIME="${TIME:-02:00:00}"
NO_SYNC="${NO_SYNC:-}"

# Step 1: sync code (unless NO_SYNC=1).
if [[ -z "${NO_SYNC}" ]]; then
    echo "==> syncing code"
    "${SCRIPT_DIR}/sync_to_della.sh"
fi

# Step 2: submit sbatch on della login node.
echo "==> submitting job: model=${MODEL} max_metric_calls=${MAX_METRIC_CALLS} condition=${CONDITION}"

sshpass -p "${REMOTE_PASSWORD}" ssh -o StrictHostKeyChecking=no \
    "${REMOTE_USER}@${REMOTE_HOST}" bash -l <<REMOTE_SCRIPT
set -euo pipefail
cd "${REMOTE_DIR}"

sbatch \
    --partition="${GPU_PARTITION}" \
    --time="${TIME}" \
    --export=ALL,MODEL=${MODEL},MAX_METRIC_CALLS=${MAX_METRIC_CALLS},CONDITION=${CONDITION},MODEL_STORAGE=${MODEL_STORAGE},SCRATCH_BASE=${SCRATCH_BASE} \
    examples/hotpotqa/run_hotpotqa.sbatch

echo "==> job submitted. Check status with: squeue -u ${REMOTE_USER}"
REMOTE_SCRIPT
