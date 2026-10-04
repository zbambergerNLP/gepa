#!/bin/bash
# Run on della-vis1; keep Harbor separate from GEPA and the model-serving environments.
set -euo pipefail

: "${SCRATCH_BASE:?Set SCRATCH_BASE to the existing Della GEPA directory}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
UV="${SCRATCH_BASE}/.tools/uv-0.9.13/uv"
export UV_CACHE_DIR="${SCRATCH_BASE}/.cache/uv"
[[ "$(uname -s)" == Linux ]] || { echo "Run this setup on the Della Linux visualization host" >&2; exit 1; }
[[ -x "${UV}" ]] || { echo "Prepare the existing Della uv installation first" >&2; exit 1; }

if ! command -v singularity >/dev/null && ! command -v apptainer >/dev/null; then
    module load apptainer
fi
CONTAINER_BIN="$(command -v singularity || command -v apptainer)"
HARBOR_VERSION="$("${UV}" run --no-project --python 3.12.11 python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["dataset"]["harbor_version"])' \
    "${REPO_ROOT}/examples/terminalbench/terminalbench-v2.1-manifest.json")"
RUNTIME_DIR="${SCRATCH_BASE}/.tools/terminalbench-harbor-${HARBOR_VERSION}"
export APPTAINER_CACHEDIR="${SCRATCH_BASE}/.cache/terminalbench/apptainer"
mkdir -p "${RUNTIME_DIR}/bin" "${SCRATCH_BASE}/.cache/terminalbench/sif" "${APPTAINER_CACHEDIR}"

if [[ ! -x "${RUNTIME_DIR}/venv/bin/harbor" ]]; then
    if [[ ! -x "${RUNTIME_DIR}/venv/bin/python" ]]; then
        "${UV}" venv --python 3.12.11 "${RUNTIME_DIR}/venv"
    fi
    "${UV}" pip install --python "${RUNTIME_DIR}/venv/bin/python" "harbor==${HARBOR_VERSION}"
fi

# Harbor invokes the compatibility command by name, including with Apptainer.
if [[ ! -e "${RUNTIME_DIR}/bin/singularity" ]]; then
    ln -s "${CONTAINER_BIN}" "${RUNTIME_DIR}/bin/singularity"
fi
export PATH="${RUNTIME_DIR}/bin:${RUNTIME_DIR}/venv/bin:${PATH}"
[[ "$(harbor --version)" == "${HARBOR_VERSION}" ]] || { echo "Unexpected Harbor version" >&2; exit 1; }
singularity --version
"${UV}" pip check --python "${RUNTIME_DIR}/venv/bin/python"
"${UV}" pip freeze --python "${RUNTIME_DIR}/venv/bin/python" > "${RUNTIME_DIR}/installed-requirements.txt"
printf 'Harbor %s CLI is installed; container execution is not yet verified.\n' "${HARBOR_VERSION}"
printf '%s\n' 'Before a compute trial, verify image and bootstrap dependency availability; see examples/terminalbench/README.md.'
printf '%s\n' 'Use on allocated compute nodes:'
printf 'export PATH=%q:%q:$PATH\n' "${RUNTIME_DIR}/bin" "${RUNTIME_DIR}/venv/bin"
printf 'export APPTAINER_CACHEDIR=%q\n' "${APPTAINER_CACHEDIR}"
printf '%s\n' '--container-runtime singularity'
printf '%s %q\n' '--singularity-image-cache' "${SCRATCH_BASE}/.cache/terminalbench/sif"
