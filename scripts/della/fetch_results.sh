#!/bin/bash
# Pull GEPA experiment results and logs from della to your laptop.
#
# Usage:
#   scripts/della/fetch_results.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "ERROR: ${ENV_FILE} not found." >&2
    exit 1
fi

set -a; source "${ENV_FILE}"; set +a

mkdir -p "${REPO_ROOT}/outputs"

# Fetch outputs from the remote REMOTE_DIR (where GEPA writes run_dir outputs).
echo "==> fetching outputs from ${REMOTE_HOST}:${REMOTE_DIR}/outputs/"
sshpass -p "${REMOTE_PASSWORD}" rsync -avz \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/outputs/" \
    "${REPO_ROOT}/outputs/"

# Also fetch any SLURM logs from the submit dir.
echo "==> fetching SLURM logs"
sshpass -p "${REMOTE_PASSWORD}" rsync -avz \
    --include='gepa-hotpotqa-*.log' \
    --exclude='*' \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/" \
    "${REPO_ROOT}/outputs/"

echo "==> done. Results in ${REPO_ROOT}/outputs/"
