#!/bin/bash
# Steps 1-4 of examples/hotpotqa/DELLA_CAMPAIGN.md as one check: local
# prerequisites, exact source commit and clean tree, scripts/della/.env,
# non-interactive SSH to both Della hosts, and the POSIT/vLLM serving
# environment on the visualization node. Read-only; run before build_env.sh.
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

echo "== 4. POSIT/vLLM on ${REMOTE_VIS_HOST}"
POSIT_DIR="${POSIT_DIR:-/home/${REMOTE_USER}/posit}"
ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_VIS_HOST}" bash -s -- "${POSIT_DIR}" "${MIN_VLLM}" "${MODEL_STORAGE}" <<'REMOTE'
set -euo pipefail
posit_dir="$1"; min_vllm="$2"; model_storage="$3"
test -x "${posit_dir}/src/.venv/bin/python" || { echo "FAIL: missing ${posit_dir}/src/.venv/bin/python"; exit 1; }
test -x "${posit_dir}/src/.venv/bin/vllm" || { echo "FAIL: missing ${posit_dir}/src/.venv/bin/vllm"; exit 1; }
if ! git -C "${posit_dir}" rev-parse --verify HEAD >/dev/null 2>&1; then
    echo "FAIL: ${posit_dir} is not a git checkout; build_env.sh needs an exact POSIT commit"; exit 1
fi
if [[ -n "$(git -C "${posit_dir}" status --porcelain --untracked-files=normal)" ]]; then
    echo "FAIL: POSIT checkout at ${posit_dir} is dirty"; git -C "${posit_dir}" status --short | head -20; exit 1
fi
echo "POSIT commit $(git -C "${posit_dir}" rev-parse HEAD)"
vllm_version="$("${posit_dir}/src/.venv/bin/python" -c 'from importlib.metadata import version; print(version("vllm"))')"
echo "vLLM ${vllm_version}"
"${posit_dir}/src/.venv/bin/python" - "${vllm_version}" "${min_vllm}" <<'PY'
import sys
from packaging.version import Version
have, need = sys.argv[1], sys.argv[2]
if Version(have) < Version(need):
    raise SystemExit(f"FAIL: vLLM {have} < required {need}; ask the lab for its Della POSIT setup")
PY
command -v apptainer >/dev/null && echo "apptainer $(apptainer --version | awk '{print $NF}')" || { echo "FAIL: apptainer missing on vis node"; exit 1; }
test -w "${model_storage}" && echo "${model_storage} writable" || { echo "FAIL: ${model_storage} not writable"; exit 1; }
echo "home quota:"; checkquota 2>/dev/null | awk '/Della home/ {print "  " $0}'
REMOTE

echo "== preflight passed; next: scripts/della/build_env.sh"
