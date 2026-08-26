#!/bin/bash
# Build the Della environment and frozen Wiki-2017 index on the visualization node.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "ERROR: ${ENV_FILE} not found." >&2
    exit 1
fi

set -a
source "${ENV_FILE}"
set +a

WIKI17_DIR="${WIKI17_DIR:-${SCRATCH_BASE}/.cache/gepa/wiki17}"
HOVER_DATA_DIR="${HOVER_DATA_DIR:-${SCRATCH_BASE}/.cache/gepa/hover}"

echo "==> syncing code to Della"
"${SCRIPT_DIR}/sync_to_della.sh"

echo "==> building the environment on ${REMOTE_VIS_HOST}"
sshpass -p "${REMOTE_PASSWORD}" ssh -o StrictHostKeyChecking=no \
    "${REMOTE_USER}@${REMOTE_VIS_HOST}" bash -l <<REMOTE_SCRIPT
set -euo pipefail
cd "${REMOTE_DIR}"

if ! command -v uv &>/dev/null; then
    echo "==> installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="\${HOME}/.local/bin:\${PATH}"
fi

export XDG_CACHE_HOME="${SCRATCH_BASE}/.cache"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export UV_CACHE_DIR="${SCRATCH_BASE}/.cache/uv"
export DSPY_CACHEDIR="${SCRATCH_BASE}/.cache/dspy"
export HOVER_DATA_DIR="${HOVER_DATA_DIR}"
mkdir -p "\${XDG_CACHE_HOME}" "\${HF_HOME}" "\${UV_CACHE_DIR}" "\${DSPY_CACHEDIR}" \
    "${WIKI17_DIR}" "\${HOVER_DATA_DIR}"

echo "==> installing GEPA, development, and Wiki-2017 dependencies"
uv sync --extra dev --extra wiki17 --group hotpotqa-task-program

echo "==> verifying the artifact-compatible DSPy task-program runtime"
.venv/bin/python - <<'PY'
from examples.hotpotqa.utils import validate_hotpotqa_dspy_runtime

version, commit = validate_hotpotqa_dspy_runtime()
print(f"DSPy task-program runtime: {version} ({commit[:8]})")
PY

echo "==> preparing the frozen Wiki-2017 BM25 index"
.venv/bin/python -m examples.common.wiki17_bm25 prepare --root "${WIKI17_DIR}"
.venv/bin/python -m examples.common.wiki17_bm25 verify --root "${WIKI17_DIR}"

echo "==> caching the HotpotQA fullwiki split"
.venv/bin/python - <<'PY'
from examples.hotpotqa.utils import load_hotpotqa_dataset

train, val, test = load_hotpotqa_dataset(seed=0)
print(f"HotpotQA data: {len(train)} train / {len(val)} val / {len(test)} test")
PY

echo "==> caching the official HoVer release"
.venv/bin/python - <<'PY'
import os

from examples.hover.utils import load_hover_dataset

train, val, test = load_hover_dataset(seed=0, data_dir=os.environ["HOVER_DATA_DIR"])
print(f"HoVer data: {len(train)} train / {len(val)} val / {len(test)} test")
PY

echo "==> environment ready at ${REMOTE_DIR}/.venv"
REMOTE_SCRIPT

echo "==> build complete"
