#!/bin/bash
# Build the Della environment and frozen Wiki-2017 index on the visualization node.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "ERROR: ${ENV_FILE} not found." >&2
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

WIKI17_DIR="${WIKI17_DIR:-${SCRATCH_BASE}/.cache/gepa/wiki17}"
MODEL_STORAGE="${MODEL_STORAGE:-/projects/BSTEWART/model_storage}"
QWEN_MODEL_DIR="${MODEL_STORAGE}/Qwen3.8-27B"
POSIT_DIR="${POSIT_DIR:-/home/${REMOTE_USER}/posit}"
HOTPOTQA_PYTHON_VERSION="3.11.13"
HOTPOTQA_UV_VERSION="0.9.13"
GEPA_UV_DIR="${REMOTE_DIR%/}/.tools/uv-${HOTPOTQA_UV_VERSION}"

echo "==> syncing code to Della"
"${SCRIPT_DIR}/sync_to_della.sh"

echo "==> building the environment on ${REMOTE_VIS_HOST}"
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes \
    "${REMOTE_USER}@${REMOTE_VIS_HOST}" bash -l <<REMOTE_SCRIPT
set -euo pipefail
cd "${REMOTE_DIR}"

export XDG_CACHE_HOME="${SCRATCH_BASE}/.cache"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export UV_CACHE_DIR="${SCRATCH_BASE}/.cache/uv"
export DSPY_CACHEDIR="${SCRATCH_BASE}/.cache/dspy"
HOTPOTQA_PYTHON_VERSION="${HOTPOTQA_PYTHON_VERSION}"
HOTPOTQA_UV_VERSION="${HOTPOTQA_UV_VERSION}"
GEPA_UV_DIR="${GEPA_UV_DIR}"
GEPA_UV_BIN="\${GEPA_UV_DIR}/uv"
export UV_PROJECT_ENVIRONMENT="${REMOTE_DIR%/}/.venv"
POSIT_DIR="${POSIT_DIR}"
mkdir -p "\${XDG_CACHE_HOME}" "\${HF_HOME}" "\${UV_CACHE_DIR}" "\${DSPY_CACHEDIR}" \
    "${WIKI17_DIR}" "${QWEN_MODEL_DIR}"

ARTIFACT_LOCK_PATH="${SCRATCH_BASE}/.cache/gepa/hotpotqa-artifacts.lock"
mkdir -p "\$(dirname "\${ARTIFACT_LOCK_PATH}")"
exec {ARTIFACT_LOCK_FD}>"\${ARTIFACT_LOCK_PATH}"
if ! flock -n "\${ARTIFACT_LOCK_FD}"; then
    echo "ERROR: HotPotQA artifacts are in use by a running job; rebuild after it finishes" >&2
    exit 1
fi

echo "==> installing GEPA, development, and Wiki-2017 dependencies"
if [[ ! -x "\${GEPA_UV_BIN}" || "\$("\${GEPA_UV_BIN}" --version 2>/dev/null)" != "uv \${HOTPOTQA_UV_VERSION}"* ]]; then
    mkdir -p "\${GEPA_UV_DIR}"
    curl -LsSf "https://astral.sh/uv/\${HOTPOTQA_UV_VERSION}/install.sh" \
        | env UV_UNMANAGED_INSTALL="\${GEPA_UV_DIR}" sh
fi
if [[ "\$("\${GEPA_UV_BIN}" --version)" != "uv \${HOTPOTQA_UV_VERSION}"* ]]; then
    echo "ERROR: exact uv \${HOTPOTQA_UV_VERSION} is unavailable at \${GEPA_UV_BIN}" >&2
    exit 1
fi
HOTPOTQA_UV_SHA256="\$(sha256sum "\${GEPA_UV_BIN}" | cut -d' ' -f1)"
"\${GEPA_UV_BIN}" python install "\${HOTPOTQA_PYTHON_VERSION}"
"\${GEPA_UV_BIN}" sync --python "\${HOTPOTQA_PYTHON_VERSION}" --frozen --no-install-project \
    --extra dev --extra wiki17 --group hotpotqa-task-program
"\${GEPA_UV_BIN}" sync --python "\${HOTPOTQA_PYTHON_VERSION}" --frozen --check --no-install-project \
    --extra dev --extra wiki17 --group hotpotqa-task-program
ACTUAL_PYTHON_VERSION="\$(.venv/bin/python -c 'import platform; print(platform.python_version())')"
if [[ "\${ACTUAL_PYTHON_VERSION}" != "\${HOTPOTQA_PYTHON_VERSION}" ]]; then
    echo "ERROR: expected Python \${HOTPOTQA_PYTHON_VERSION}, found \${ACTUAL_PYTHON_VERSION}" >&2
    exit 1
fi
HOTPOTQA_ENV_SPEC_SHA256="\$(
    {
        sha256sum pyproject.toml uv.lock
        printf 'python=%s\n' "\${HOTPOTQA_PYTHON_VERSION}"
        printf 'uv=%s\n' "\${HOTPOTQA_UV_VERSION}"
    } | sha256sum | cut -d' ' -f1
)"
printf '%s\n' "\${HOTPOTQA_ENV_SPEC_SHA256}" > .venv/.gepa-env-spec.sha256
printf '%s\n' "\${ACTUAL_PYTHON_VERSION}" > .venv/.gepa-python-version
printf '%s\n' "\${HOTPOTQA_UV_VERSION}" > .venv/.gepa-uv-version
printf '%s\n' "\${HOTPOTQA_UV_SHA256}" > .venv/.gepa-uv-sha256
echo "==> environment specification: \${HOTPOTQA_ENV_SPEC_SHA256}"

echo "==> freezing the realized GEPA task environment"
GEPA_ENV_MANIFEST_DIR="${SCRATCH_BASE}/.cache/gepa/python-environments"
GEPA_ENV_MANIFEST="\${GEPA_ENV_MANIFEST_DIR}/gepa-\${HOTPOTQA_ENV_SPEC_SHA256}.json"
mkdir -p "\${GEPA_ENV_MANIFEST_DIR}"
HOTPOTQA_GEPA_ENV_SHA256="\$(
    .venv/bin/python -m examples.common.python_environment prepare --path "\${GEPA_ENV_MANIFEST}"
)"
if [[ ! "\${HOTPOTQA_GEPA_ENV_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "ERROR: GEPA environment freeze did not produce a valid digest" >&2
    exit 1
fi
echo "==> realized GEPA environment: \${HOTPOTQA_GEPA_ENV_SHA256}"

echo "==> verifying the artifact-compatible DSPy task-program runtime"
.venv/bin/python - <<'PY'
from examples.hotpotqa.utils import validate_hotpotqa_dspy_runtime

version, commit = validate_hotpotqa_dspy_runtime()
print(f"DSPy task-program runtime: {version} ({commit[:8]})")
PY

echo "==> freezing the realized POSIT serving environment"
VLLM_PY="\${POSIT_DIR}/src/.venv/bin/python"
if [[ ! -x "\${VLLM_PY}" ]]; then
    echo "ERROR: missing \${VLLM_PY} -- build the POSIT venv first" >&2
    exit 1
fi
if [[ -n "\$(git -C "\${POSIT_DIR}" status --porcelain --untracked-files=normal)" ]]; then
    echo "ERROR: commit or remove local POSIT changes before preparing production artifacts" >&2
    exit 1
fi
HOTPOTQA_POSIT_COMMIT="\$(git -C "\${POSIT_DIR}" rev-parse HEAD)"
if [[ ! "\${HOTPOTQA_POSIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: POSIT source does not resolve to an exact Git commit" >&2
    exit 1
fi
if ! "\${GEPA_UV_BIN}" pip check --python "\${VLLM_PY}"; then
    echo "ERROR: POSIT serving environment has inconsistent dependencies" >&2
    exit 1
fi
POSIT_ENV_MANIFEST="${SCRATCH_BASE}/.cache/gepa/posit-environments/\${HOTPOTQA_POSIT_COMMIT}.json"
HOTPOTQA_POSIT_ENV_SHA256="\$(
    "\${VLLM_PY}" -m examples.common.python_environment prepare --path "\${POSIT_ENV_MANIFEST}"
)"
if [[ ! "\${HOTPOTQA_POSIT_ENV_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "ERROR: POSIT environment freeze did not produce a valid digest" >&2
    exit 1
fi
echo "==> POSIT environment: \${HOTPOTQA_POSIT_COMMIT}/\${HOTPOTQA_POSIT_ENV_SHA256}"

echo "==> preparing the frozen Wiki-2017 BM25 index"
.venv/bin/python -m examples.common.wiki17_bm25 prepare --root "${WIKI17_DIR}"
.venv/bin/python -m examples.common.wiki17_bm25 verify --deep --root "${WIKI17_DIR}"

echo "==> preparing the pinned Qwen3.8-27B checkpoint"
(
    exec {MODEL_LOCK_FD}<"${QWEN_MODEL_DIR}"
    if ! flock -n "\${MODEL_LOCK_FD}"; then
        echo "ERROR: another user is preparing or serving the shared Qwen3.8-27B checkpoint" >&2
        exit 1
    fi
    .venv/bin/python -m examples.common.model_snapshot prepare --root "${QWEN_MODEL_DIR}"
    .venv/bin/python -m examples.common.model_snapshot verify --root "${QWEN_MODEL_DIR}"
)

echo "==> caching the HotpotQA fullwiki split"
.venv/bin/python - <<'PY'
from examples.hotpotqa.utils import load_hotpotqa_dataset

train, val, test = load_hotpotqa_dataset(seed=0)
print(f"HotpotQA data: {len(train)} train / {len(val)} val / {len(test)} test")
PY

echo "==> environment ready at ${REMOTE_DIR}/.venv"
REMOTE_SCRIPT

echo "==> build complete"
