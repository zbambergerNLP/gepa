#!/bin/bash
# Sync code and build the venv on della's visualization node.
# The vis node has internet access (GPU nodes do not).
#
# Usage:
#   scripts/della/build_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "ERROR: ${ENV_FILE} not found." >&2
    exit 1
fi

set -a; source "${ENV_FILE}"; set +a

# Step 1: sync code
echo "==> syncing code to della"
"${SCRIPT_DIR}/sync_to_della.sh"

# Step 2: build venv on vis node
echo "==> building venv on ${REMOTE_VIS_HOST} (vis node, has internet)"
sshpass -p "${REMOTE_PASSWORD}" ssh -o StrictHostKeyChecking=no \
    "${REMOTE_USER}@${REMOTE_VIS_HOST}" bash -l <<REMOTE_SCRIPT
set -euo pipefail
cd "${REMOTE_DIR}"

# Install uv if not present.
if ! command -v uv &>/dev/null; then
    echo "==> installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="\${HOME}/.local/bin:\${PATH}"
fi

# Keep caches on scratch (off /home quota).
SCRATCH_BASE="${SCRATCH_BASE}"
export XDG_CACHE_HOME="\${SCRATCH_BASE}/.cache"
export UV_CACHE_DIR="\${SCRATCH_BASE}/.cache/uv"
mkdir -p "\${UV_CACHE_DIR}"

echo "==> uv sync --extra dev"
uv sync --extra dev

echo "==> verifying install"
.venv/bin/python -c "
import gepa; print('gepa OK')
import litellm; print('litellm OK')
from examples.hotpotqa.utils import load_hotpotqa_dataset
train, val = load_hotpotqa_dataset()
print(f'HotpotQA data: {len(train)} train / {len(val)} val')
"

echo "==> venv ready at ${REMOTE_DIR}/.venv"
REMOTE_SCRIPT

echo "==> done"
