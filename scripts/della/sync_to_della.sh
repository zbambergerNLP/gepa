#!/bin/bash
# Sync the local checkout to Della.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "ERROR: ${ENV_FILE} not found. Copy .env.example and fill it in." >&2
    exit 1
fi
if [[ -L "${ENV_FILE}" || ! -O "${ENV_FILE}" ]]; then
    echo "ERROR: ${ENV_FILE} must be a regular file owned by the current user" >&2
    exit 1
fi
if ENV_MODE="$(stat -f '%Lp' "${ENV_FILE}" 2>/dev/null)"; then
    :
elif ENV_MODE="$(stat -c '%a' "${ENV_FILE}" 2>/dev/null)"; then
    :
else
    echo "ERROR: could not verify permissions for ${ENV_FILE}" >&2
    exit 1
fi
if [[ ! "${ENV_MODE}" =~ ^[0-7]{3,4}$ ]] || (( (8#${ENV_MODE} & 8#077) != 0 )); then
    echo "ERROR: ${ENV_FILE} contains credentials and must not grant group or other access; run chmod 600 ${ENV_FILE}" >&2
    exit 1
fi

source "${ENV_FILE}"
unset DEEPSEEK_API_KEY

SYNC_SOURCE_COMMIT="${SYNC_SOURCE_COMMIT:-}"
SYNC_REMOTE_DIR="${SYNC_REMOTE_DIR:-${REMOTE_DIR}}"
SYNC_MANIFEST_OUTPUT="${SYNC_MANIFEST_OUTPUT:-}"
SOURCE_ROOT="${REPO_ROOT}"
SOURCE_STAGE_DIR=""
SOURCE_MANIFEST_SHA256=""

if [[ -n "${SYNC_MANIFEST_OUTPUT}" && ( -z "${SYNC_SOURCE_COMMIT}" || "${SYNC_MANIFEST_OUTPUT}" != /* ) ]]; then
    echo "ERROR: SYNC_MANIFEST_OUTPUT requires a commit staging run and an absolute output path" >&2
    exit 1
fi

cleanup() {
    if [[ -n "${SOURCE_STAGE_DIR}" && -d "${SOURCE_STAGE_DIR}" ]]; then
        rm -rf -- "${SOURCE_STAGE_DIR}"
    fi
}
trap cleanup EXIT

if [[ -n "${SYNC_SOURCE_COMMIT}" ]]; then
    if [[ ! "${SYNC_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
        echo "ERROR: SYNC_SOURCE_COMMIT must be a full Git commit" >&2
        exit 1
    fi
    if [[ "${SYNC_SOURCE_COMMIT}" != "$(git -C "${REPO_ROOT}" rev-parse HEAD)" ]]; then
        echo "ERROR: production source staging accepts only the current clean HEAD" >&2
        exit 1
    fi
    EXPECTED_REMOTE_SOURCE_DIR="${REMOTE_DIR%/}/sources/${SYNC_SOURCE_COMMIT}"
    if [[ "${SYNC_REMOTE_DIR%/}" != "${EXPECTED_REMOTE_SOURCE_DIR}" ]]; then
        echo "ERROR: commit source must be staged under ${EXPECTED_REMOTE_SOURCE_DIR}" >&2
        exit 1
    fi
    SOURCE_STAGE_DIR="$(mktemp -d)"
    git -C "${REPO_ROOT}" archive "${SYNC_SOURCE_COMMIT}" | tar -xf - -C "${SOURCE_STAGE_DIR}"
    printf '%s\n' "${SYNC_SOURCE_COMMIT}" > "${SOURCE_STAGE_DIR}/.gepa-source-commit"
    (
        cd "${SOURCE_STAGE_DIR}"
        find . -type f \
            ! -name '.gepa-source-manifest.sha256sums' \
            ! -name '.gepa-source-manifest.sha256' \
            -print0 \
            | LC_ALL=C sort -z \
            | xargs -0 sha256sum > .gepa-source-manifest.sha256sums
    )
    SOURCE_MANIFEST_SHA256="$(sha256sum "${SOURCE_STAGE_DIR}/.gepa-source-manifest.sha256sums" | cut -d' ' -f1)"
    printf '%s\n' "${SOURCE_MANIFEST_SHA256}" > "${SOURCE_STAGE_DIR}/.gepa-source-manifest.sha256"
    SOURCE_ROOT="${SOURCE_STAGE_DIR}"
fi

printf -v REMOTE_SOURCE_DIR_QUOTED '%q' "${SYNC_REMOTE_DIR}"
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes \
    "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p -- ${REMOTE_SOURCE_DIR_QUOTED}"

echo "==> syncing ${SOURCE_ROOT}/ to ${REMOTE_USER}@${REMOTE_HOST}:${SYNC_REMOTE_DIR}/"

RSYNC_EXCLUDES=()
if [[ -z "${SYNC_SOURCE_COMMIT}" ]]; then
    RSYNC_EXCLUDES=(
        --exclude '.venv/'
        --exclude '__pycache__/'
        --exclude '.mypy_cache/'
        --exclude '.pytest_cache/'
        --exclude '.ruff_cache/'
        --exclude '.cache/'
        --exclude '.git/'
        --exclude 'logs/'
        --exclude 'outputs/'
        --exclude 'sources/'
        --exclude 'gepa-*.log'
        --exclude '*.egg-info/'
        --exclude 'scripts/della/.env'
    )
else
    RSYNC_EXCLUDES=(
        --exclude '/outputs/'
        --exclude '/gepa-hotpotqa-*.log'
    )
fi

rsync -avz --delete \
    -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=yes" \
    "${RSYNC_EXCLUDES[@]}" \
    "${SOURCE_ROOT}/" \
    "${REMOTE_USER}@${REMOTE_HOST}:${SYNC_REMOTE_DIR}/"

if [[ -n "${SYNC_MANIFEST_OUTPUT}" ]]; then
    printf '%s\n' "${SOURCE_MANIFEST_SHA256}" > "${SYNC_MANIFEST_OUTPUT}"
fi

echo "==> sync complete"
