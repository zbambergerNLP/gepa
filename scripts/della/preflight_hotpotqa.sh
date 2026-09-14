#!/bin/bash
# Steps 1-4 of examples/hotpotqa/DELLA_CAMPAIGN.md as one check: local
# prerequisites, exact source commit and clean tree, scripts/della/.env,
# non-interactive SSH to both Della hosts, and the serving prerequisites
# (CUDA module, model storage, serving venv vs. lock) on the visualization
# node. Read-only; run before build_env.sh.
#
# Usage:
#   scripts/della/preflight_hotpotqa.sh
# Uses the current clean consolidated branch tip. An optional
# HOTPOTQA_SOURCE_COMMIT additionally checks a specific expected revision.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
EXPECTED_BRANCH="codex/consolidated-della-experiments"

fail() { echo "FAIL: $*" >&2; exit 1; }

echo "== 1. local prerequisites"
for c in git ssh rsync sha256sum uv; do
    command -v "$c" >/dev/null || fail "missing local command: $c"
done
echo "ok"

echo "== 2. source commit and worktree"
CURRENT_BRANCH="$(git -C "${REPO_ROOT}" branch --show-current)"
[[ "${CURRENT_BRANCH}" == "${EXPECTED_BRANCH}" ]] \
    || fail "use the consolidated branch ${EXPECTED_BRANCH}; current branch is ${CURRENT_BRANCH:-detached HEAD}"
HEAD_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
EXPECTED_COMMIT="${HOTPOTQA_SOURCE_COMMIT:-${HEAD_COMMIT}}"
[[ "${HEAD_COMMIT}" == "${EXPECTED_COMMIT}" ]] \
    || fail "HEAD is ${HEAD_COMMIT}, expected ${EXPECTED_COMMIT}; review the current branch tip"
[[ -z "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=normal)" ]] \
    || fail "worktree is dirty; the launcher rejects it"
echo "Source is exact and clean: ${CURRENT_BRANCH} at ${HEAD_COMMIT}."

echo "== 3. scripts/della/.env"
[[ -f "${ENV_FILE}" ]] || fail "${ENV_FILE} missing (copy .env.example)"
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

! grep -qiE 'YOUR_NETID|your_princeton_netid|your_allocation' "${ENV_FILE}" \
    || fail "placeholders remain in ${ENV_FILE}"
source "${ENV_FILE}"
for v in REMOTE_USER REMOTE_HOST REMOTE_VIS_HOST REMOTE_DIR SCRATCH_BASE MODEL_STORAGE GPU_PARTITION; do
    [[ -n "${!v:-}" ]] || fail "${v} is unset in ${ENV_FILE}"
done
echo "user=${REMOTE_USER} remote_dir=${REMOTE_DIR} model_storage=${MODEL_STORAGE} partition=${GPU_PARTITION}"

echo "== 3b. non-interactive SSH (BatchMode) to both hosts"
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20)
ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" 'echo "login node ok: $(hostname)"' \
    || fail "BatchMode ssh to ${REMOTE_HOST} failed; run scripts/della/della_session.sh open"
ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_VIS_HOST}" 'echo "vis node ok: $(hostname)"' \
    || fail "BatchMode ssh to ${REMOTE_VIS_HOST} failed; run scripts/della/della_session.sh open"

echo "== 4. serving prerequisites on ${REMOTE_VIS_HOST}"
# One hash-locked serving environment per model profile: "<venv dir> <lock file>".
SERVING_ENVIRONMENTS=(
    "${REMOTE_DIR%/}/.serving-venv examples/hotpotqa/serving/requirements-x86_64-linux-py312.txt"
    "${REMOTE_DIR%/}/.serving-venv-deepseek-v4.1-flash examples/hotpotqa/serving/requirements-deepseek-v4.1-flash-x86_64-linux-py312.txt"
)
SERVING_ARGS=()
for environment in "${SERVING_ENVIRONMENTS[@]}"; do
    read -r serving_venv serving_lock <<< "${environment}"
    [[ -f "${REPO_ROOT}/${serving_lock}" ]] || fail "missing ${serving_lock}; run scripts/della/lock_serving_env.sh"
    lock_sha="$(sha256sum "${REPO_ROOT}/${serving_lock}" | cut -d' ' -f1)"
    echo "serving lock ${serving_lock}: ${lock_sha}"
    SERVING_ARGS+=("${serving_venv}" "${lock_sha}")
done
ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_VIS_HOST}" bash -s -- \
    "${MODEL_STORAGE}" "${SERVING_ARGS[@]}" <<'REMOTE'
set -euo pipefail
model_storage="$1"; shift
source /usr/share/Modules/init/bash 2>/dev/null || true
if module avail cudatoolkit/13.0 2>&1 | grep -q 'cudatoolkit/13.0'; then
    echo "cudatoolkit/13.0 module ok"
else
    echo "FAIL: cudatoolkit/13.0 module missing"; exit 1
fi
test -w "${model_storage}" && echo "${model_storage} writable" || { echo "FAIL: ${model_storage} not writable"; exit 1; }
echo "home quota:"; checkquota 2>/dev/null | awk '/Della home/ {print "  " $0}'
while (( $# >= 2 )); do
    serving_venv="$1"; lock_sha="$2"; shift 2
    if [[ -x "${serving_venv}/bin/python" ]]; then
        marker="${serving_venv}/.gepa-serving-lock.sha256"
        if [[ -f "${marker}" && "$(tr -d '\n' < "${marker}")" == "${lock_sha}" ]]; then
            echo "${serving_venv}: present and matches the lock"
        else
            echo "${serving_venv}: built from a different lock; build_env.sh will rebuild it"
        fi
        echo "  vLLM $("${serving_venv}/bin/python" -c 'from importlib.metadata import version; print(version("vllm"))')"
    else
        echo "${serving_venv}: not built yet (build_env.sh creates it)"
    fi
done
REMOTE

echo "== preflight passed; next: scripts/della/build_env.sh"
