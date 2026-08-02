#!/bin/bash
# Sync the gepa repo to della. Run from anywhere; paths are absolute.
#
# Usage:
#   scripts/della/sync_to_della.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "ERROR: ${ENV_FILE} not found. Copy from .env.example and fill in." >&2
    exit 1
fi

set -a; source "${ENV_FILE}"; set +a

echo "==> syncing ${REPO_ROOT}/ -> ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

sshpass -p "${REMOTE_PASSWORD}" rsync -avz --delete \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude '.mypy_cache/' \
    --exclude '.pytest_cache/' \
    --exclude '.ruff_cache/' \
    --exclude '.git/' \
    --exclude 'outputs/' \
    --exclude 'gepa-*.log' \
    --exclude '*.egg-info/' \
    --exclude 'scripts/della/.env' \
    "${REPO_ROOT}/" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

echo "==> done"
