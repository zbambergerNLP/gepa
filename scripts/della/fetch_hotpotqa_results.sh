#!/bin/bash
# Fetch one commit-scoped HotPotQA campaign and analyze its completed runs.
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

HOTPOTQA_SOURCE_COMMIT="${HOTPOTQA_SOURCE_COMMIT:-$(git -C "${REPO_ROOT}" rev-parse HEAD)}"
HOTPOTQA_CAMPAIGN_ID="${HOTPOTQA_CAMPAIGN_ID:-hotpotqa-final-v1}"
ANALYSIS_SOURCE_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
if [[ ! "${HOTPOTQA_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: HOTPOTQA_SOURCE_COMMIT must be a full lowercase Git commit" >&2
    exit 1
fi
if [[ ! "${HOTPOTQA_CAMPAIGN_ID}" =~ ^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?$ ]]; then
    echo "ERROR: HOTPOTQA_CAMPAIGN_ID must start and end with a letter or number" >&2
    exit 1
fi
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=no)" ]]; then
    echo "ERROR: commit tracked worktree changes before publishing campaign analysis" >&2
    exit 1
fi
for REQUIRED_PATH in "${REMOTE_DIR}" "${SCRATCH_BASE}"; do
    if [[ ! "${REQUIRED_PATH}" =~ ^/[A-Za-z0-9._/-]+$ ]]; then
        echo "ERROR: Della result paths must be absolute and contain no shell metacharacters" >&2
        exit 1
    fi
done
if [[ ! "${REMOTE_USER}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "ERROR: REMOTE_USER is not a valid SSH user" >&2
    exit 1
fi
if [[ ! "${REMOTE_HOST}" =~ ^[A-Za-z0-9][A-Za-z0-9.-]*$ ]]; then
    echo "ERROR: REMOTE_HOST is not a valid SSH host name" >&2
    exit 1
fi

REMOTE_SOURCE_DIR="${REMOTE_DIR%/}/sources/${HOTPOTQA_SOURCE_COMMIT}"
REMOTE_LOG_DIR="${SCRATCH_BASE%/}/logs/hotpotqa/${HOTPOTQA_CAMPAIGN_ID}/${HOTPOTQA_SOURCE_COMMIT}"
REMOTE_LOCK_DIR="${SCRATCH_BASE%/}/.cache/gepa/hotpotqa-campaign/${HOTPOTQA_CAMPAIGN_ID}"
LOCAL_PARENT="${REPO_ROOT}/outputs/hotpotqa-campaigns/${HOTPOTQA_CAMPAIGN_ID}"
LOCAL_ROOT="${LOCAL_PARENT}/${HOTPOTQA_SOURCE_COMMIT}"
SSH_TARGET="${REMOTE_USER}@${REMOTE_HOST}"
SSH_OPTIONS="ssh -o BatchMode=yes -o StrictHostKeyChecking=yes"

REMOTE_COMMIT="$(
    ssh -o BatchMode=yes -o StrictHostKeyChecking=yes \
        "${SSH_TARGET}" "tr -d '\n' < '${REMOTE_SOURCE_DIR}/.gepa-source-commit'"
)"
if [[ "${REMOTE_COMMIT}" != "${HOTPOTQA_SOURCE_COMMIT}" ]]; then
    echo "ERROR: remote source does not match ${HOTPOTQA_SOURCE_COMMIT}" >&2
    exit 1
fi

mkdir -p "${LOCAL_PARENT}"
FETCH_ROOT="$(mktemp -d "${LOCAL_PARENT}/.${HOTPOTQA_SOURCE_COMMIT}.fetch.XXXXXX")"
PREVIOUS_ROOT=""
cleanup() {
    if [[ -n "${PREVIOUS_ROOT}" && -e "${PREVIOUS_ROOT}" && ! -e "${LOCAL_ROOT}" ]]; then
        mv -- "${PREVIOUS_ROOT}" "${LOCAL_ROOT}" \
            || echo "ERROR: previous fetch remains at ${PREVIOUS_ROOT}" >&2
    fi
    if [[ -n "${FETCH_ROOT}" && -d "${FETCH_ROOT}" ]]; then
        rm -rf -- "${FETCH_ROOT}"
    fi
}
trap cleanup EXIT
mkdir -p "${FETCH_ROOT}/runs" "${FETCH_ROOT}/logs" "${FETCH_ROOT}/campaign-locks"

echo "==> fetching run artifacts for ${HOTPOTQA_SOURCE_COMMIT}"
rsync -avz --partial \
    -e "${SSH_OPTIONS}" \
    "${SSH_TARGET}:${REMOTE_SOURCE_DIR}/outputs/" \
    "${FETCH_ROOT}/runs/"

echo "==> fetching campaign-scoped Slurm and model-server logs"
rsync -avz --partial \
    -e "${SSH_OPTIONS}" \
    "${SSH_TARGET}:${REMOTE_LOG_DIR}/" \
    "${FETCH_ROOT}/logs/"

echo "==> fetching campaign identity locks"
rsync -avz --partial \
    -e "${SSH_OPTIONS}" \
    "${SSH_TARGET}:${REMOTE_LOCK_DIR}/" \
    "${FETCH_ROOT}/campaign-locks/"

echo "==> analyzing completed runs"
(
    cd "${REPO_ROOT}"
    uv run python -m examples.hotpotqa.analyze_results \
        "${FETCH_ROOT}/runs" \
        --output "${FETCH_ROOT}/hotpotqa_analysis.json" \
        --campaign-id "${HOTPOTQA_CAMPAIGN_ID}" \
        --source-commit "${HOTPOTQA_SOURCE_COMMIT}" \
        --analysis-source-commit "${ANALYSIS_SOURCE_COMMIT}"
)

if [[ -e "${LOCAL_ROOT}" ]]; then
    PREVIOUS_ROOT="${LOCAL_ROOT}.previous.$(date -u +%Y%m%dT%H%M%SZ).$$"
    mv -- "${LOCAL_ROOT}" "${PREVIOUS_ROOT}"
    echo "==> previous fetch retained at ${PREVIOUS_ROOT}"
fi
if ! mv -- "${FETCH_ROOT}" "${LOCAL_ROOT}"; then
    if [[ -n "${PREVIOUS_ROOT}" && -e "${PREVIOUS_ROOT}" ]]; then
        mv -- "${PREVIOUS_ROOT}" "${LOCAL_ROOT}"
    fi
    echo "ERROR: could not promote the analyzed fetch" >&2
    exit 1
fi
FETCH_ROOT=""
PREVIOUS_ROOT=""

echo "==> results available at ${LOCAL_ROOT}"
