#!/bin/bash
# Steps 1-4 of examples/hotpotqa/DELLA_CAMPAIGN.md as one check: local
# prerequisites, exact source commit and clean tree, scripts/della/.env,
# non-interactive SSH to both Della hosts, and the serving prerequisites
# (apptainer, CUDA module, model storage, serving venv vs. lock) on the
# visualization node. Read-only; run before build_env.sh.
#
# Usage:
#   HOTPOTQA_SOURCE_COMMIT=<sha> scripts/della/preflight_hotpotqa.sh
# (defaults to the commit pinned in Gilad's runbook)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
EXPECTED_COMMIT="${HOTPOTQA_SOURCE_COMMIT:-169ddda125b1abe305c7714bbb5b3fc38b21b587}"
MIN_VLLM="0.17.0"

fail() { echo "FAIL: $*" >&2; exit 1; }

echo "== 1. local prerequisites"
for c in git ssh rsync sha256sum uv; do
    command -v "$c" >/dev/null || fail "missing local command: $c"
done
echo "ok"

echo "== 2. source commit and worktree"
HEAD_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
[[ "${HEAD_COMMIT}" == "${EXPECTED_COMMIT}" ]] \
    || fail "HEAD is ${HEAD_COMMIT}, expected ${EXPECTED_COMMIT} (git switch --detach ${EXPECTED_COMMIT})"
[[ -z "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=normal)" ]] \
    || fail "worktree is dirty; the launcher rejects it"
echo "Source is exact and clean."

echo "== 3. scripts/della/.env"
[[ -f "${ENV_FILE}" ]] || fail "${ENV_FILE} missing (copy .env.example)"
ENV_MODE="$(stat -f '%Lp' "${ENV_FILE}" 2>/dev/null || stat -c '%a' "${ENV_FILE}")"
[[ "${ENV_MODE}" == "600" ]] || fail "${ENV_FILE} mode is ${ENV_MODE}; run chmod 600"
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
SERVING_VENV_DIR="${SERVING_VENV_DIR:-${REMOTE_DIR%/}/.serving-venv}"
SERVING_LOCK="${REPO_ROOT}/examples/hotpotqa/serving/requirements-x86_64-linux-py312.txt"
[[ -f "${SERVING_LOCK}" ]] || fail "missing ${SERVING_LOCK}; run scripts/della/lock_serving_env.sh"
SERVING_LOCK_SHA256="$(sha256sum "${SERVING_LOCK}" | cut -d' ' -f1)"
echo "serving lock ${SERVING_LOCK_SHA256}"
ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_VIS_HOST}" bash -s -- \
    "${SERVING_VENV_DIR}" "${SERVING_LOCK_SHA256}" "${MIN_VLLM}" "${MODEL_STORAGE}" <<'REMOTE'
set -euo pipefail
serving_venv="$1"; lock_sha="$2"; min_vllm="$3"; model_storage="$4"
command -v apptainer >/dev/null && echo "apptainer $(apptainer --version | awk '{print $NF}')" \
    || { echo "FAIL: apptainer missing on vis node"; exit 1; }
source /usr/share/Modules/init/bash 2>/dev/null || true
if module avail cudatoolkit/13.0 2>&1 | grep -q 'cudatoolkit/13.0'; then
    echo "cudatoolkit/13.0 module ok"
else
    echo "FAIL: cudatoolkit/13.0 module missing"; exit 1
fi
test -w "${model_storage}" && echo "${model_storage} writable" || { echo "FAIL: ${model_storage} not writable"; exit 1; }
echo "home quota:"; checkquota 2>/dev/null | awk '/Della home/ {print "  " $0}'
if [[ -x "${serving_venv}/bin/python" ]]; then
    marker="${serving_venv}/.gepa-serving-lock.sha256"
    if [[ -f "${marker}" && "$(tr -d '\n' < "${marker}")" == "${lock_sha}" ]]; then
        echo "serving venv present and matches the lock"
    else
        echo "serving venv present but built from a different lock; build_env.sh will rebuild it"
    fi
    vllm_version="$("${serving_venv}/bin/python" -c 'from importlib.metadata import version; print(version("vllm"))')"
    echo "vLLM ${vllm_version}"
    "${serving_venv}/bin/python" - "${vllm_version}" "${min_vllm}" <<'PY'
import sys
from packaging.version import Version
have, need = sys.argv[1], sys.argv[2]
if Version(have) < Version(need):
    raise SystemExit(f"FAIL: vLLM {have} < required {need}")
PY
else
    echo "serving venv not built yet (build_env.sh creates ${serving_venv})"
fi
REMOTE

echo "== preflight passed; next: scripts/della/build_env.sh"
