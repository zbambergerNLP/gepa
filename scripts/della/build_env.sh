#!/bin/bash
# Build everything the HotPotQA jobs need on Della, from the laptop. Syncs the checkout,
# then runs these steps on della-vis1 (the only node with internet). Each one also runs
# on its own from the synced checkout:
#   scripts/della/remote/setup_env.sh         GEPA venv and one vLLM serving venv per model
#   scripts/della/remote/download_dataset.sh  Wiki-2017 BM25 index and the HotpotQA split
#   scripts/della/remote/download_model.sh    one pinned checkpoint, byte-verified
# The model downloads run detached (DeepSeek-V4.1-Flash alone is 510 GB), so a dropped
# laptop connection cannot stop them; the script prints their log.
#
# Usage: scripts/della/build_env.sh [model ...]   (default: qwen3.8-27b deepseek-v4.1-flash)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
# .env holds cluster credentials and is sourced as code, so it must be yours alone.
if [[ -L "${ENV_FILE}" || ! -O "${ENV_FILE}" ]]; then
    echo "ERROR: ${ENV_FILE} must be a regular file you own" >&2; exit 1
fi
ENV_MODE="$(stat -f '%Lp' "${ENV_FILE}" 2>/dev/null || stat -c '%a' "${ENV_FILE}")"
(( (8#${ENV_MODE} & 8#077) == 0 )) || { echo "ERROR: run chmod 600 ${ENV_FILE}" >&2; exit 1; }
source "${ENV_FILE}"
MODEL_STORAGE="${MODEL_STORAGE:-/projects/BSTEWART/model_storage}"
WIKI17_DIR="${WIKI17_DIR:-${SCRATCH_BASE}/.cache/gepa/wiki17}"
(( $# )) || set -- qwen3.8-27b deepseek-v4.1-flash
for model in "$@"; do
    case "${model}" in
        qwen3.8-27b|deepseek-v4.1-flash) ;;
        *) echo "ERROR: unsupported model profile: ${model}" >&2; exit 1 ;;
    esac
done
VIS=(ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "${REMOTE_USER}@${REMOTE_VIS_HOST}")
IN_CHECKOUT="cd '${REMOTE_DIR}' && export SCRATCH_BASE='${SCRATCH_BASE}' MODEL_STORAGE='${MODEL_STORAGE}' WIKI17_DIR='${WIKI17_DIR}'"
LOG="${SCRATCH_BASE}/logs/build/download-models-$(date +%Y%m%dT%H%M%S).log"

"${SCRIPT_DIR}/sync_to_della.sh"
"${VIS[@]}" "${IN_CHECKOUT} && bash scripts/della/remote/setup_env.sh && bash scripts/della/remote/download_dataset.sh"
"${VIS[@]}" "${IN_CHECKOUT} && mkdir -p '${LOG%/*}' && nohup setsid bash -c \
    'for model in $*; do bash scripts/della/remote/download_model.sh \${model} || exit 1; done; echo MODELS_DONE' \
    > '${LOG}' 2>&1 < /dev/null &"
echo "==> venvs and datasets ready; downloading $* on ${REMOTE_VIS_HOST} (prints MODELS_DONE when finished)"
echo "    log: ssh ${REMOTE_USER}@${REMOTE_VIS_HOST} tail -f ${LOG}"
