#!/bin/bash
# Build the GEPA venv and one hash-locked vLLM serving venv per model on della-vis1.
# Run from the synced checkout with SCRATCH_BASE set (scripts/della/build_env.sh does
# both). Each venv is frozen into a manifest that submit_hotpotqa.sh and the sbatch verify.
set -euo pipefail
export PYTHONPATH="${PWD}/src:${PWD}"

: "${SCRATCH_BASE:?}"
PYTHON_VERSION="3.11.13"
SERVING_PYTHON_VERSION="3.12.7"
UV_VERSION="0.9.13"
UV_DIR="${PWD}/.tools/uv-${UV_VERSION}"
UV="${UV_DIR}/uv"
MANIFESTS="${SCRATCH_BASE}/.cache/gepa"
# One serving venv per model: "<venv> <hash-locked requirements>".
SERVING_ENVS=(
    ".serving-venv examples/hotpotqa/serving/requirements-x86_64-linux-py312.txt"
    ".serving-venv-deepseek-v4.1-flash examples/hotpotqa/serving/requirements-deepseek-v4.1-flash-x86_64-linux-py312.txt"
)
export XDG_CACHE_HOME="${SCRATCH_BASE}/.cache" UV_CACHE_DIR="${SCRATCH_BASE}/.cache/uv"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface" DSPY_CACHEDIR="${SCRATCH_BASE}/.cache/dspy"
export UV_PROJECT_ENVIRONMENT="${PWD}/.venv"
mkdir -p "${MANIFESTS}/python-environments" "${MANIFESTS}/serving-environments"

# Jobs hold this lock shared while they run; never rebuild under them.
exec {ARTIFACT_LOCK_FD}>"${MANIFESTS}/hotpotqa-artifacts.lock"
flock -n "${ARTIFACT_LOCK_FD}" || { echo "ERROR: HotPotQA artifacts are in use by a running job" >&2; exit 1; }

echo "==> uv ${UV_VERSION} and Python ${PYTHON_VERSION} / ${SERVING_PYTHON_VERSION}"
if [[ "$("${UV}" --version 2>/dev/null)" != "uv ${UV_VERSION}"* ]]; then
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | env UV_UNMANAGED_INSTALL="${UV_DIR}" sh
fi
"${UV}" python install "${PYTHON_VERSION}" "${SERVING_PYTHON_VERSION}"

echo "==> GEPA venv"
SYNC_ARGS=(--python "${PYTHON_VERSION}" --frozen --no-install-project --extra dev --extra wiki17 --group hotpotqa-task-program)
"${UV}" sync "${SYNC_ARGS[@]}"
"${UV}" sync "${SYNC_ARGS[@]}" --check
# The launchers recompute this spec and compare it against these markers.
ENV_SPEC="$({ sha256sum pyproject.toml uv.lock; printf 'python=%s\nuv=%s\n' "${PYTHON_VERSION}" "${UV_VERSION}"; } | sha256sum | cut -d' ' -f1)"
echo "${ENV_SPEC}" > .venv/.gepa-env-spec.sha256
.venv/bin/python -c 'import platform; print(platform.python_version())' > .venv/.gepa-python-version
echo "${UV_VERSION}" > .venv/.gepa-uv-version
sha256sum "${UV}" | cut -d' ' -f1 > .venv/.gepa-uv-sha256
.venv/bin/python -m examples.common.python_environment prepare \
    --path "${MANIFESTS}/python-environments/gepa-${ENV_SPEC}.json"
.venv/bin/python -c 'from examples.hotpotqa.utils import validate_hotpotqa_dspy_runtime as v; print("DSPy %s (%s)" % v())'

for serving_env in "${SERVING_ENVS[@]}"; do
    read -r VENV REQUIREMENTS <<< "${serving_env}"
    VENV="${PWD}/${VENV}"
    LOCK_SHA="$(sha256sum "${REQUIREMENTS}" | cut -d' ' -f1)"
    echo "==> ${VENV} from ${REQUIREMENTS}"
    if [[ "$(cat "${VENV}/.gepa-serving-lock.sha256" 2>/dev/null)" != "${LOCK_SHA}" ]]; then
        rm -rf -- "${VENV}"
        "${UV}" venv --python "${SERVING_PYTHON_VERSION}" "${VENV}"
        "${UV}" pip sync --python "${VENV}/bin/python" --require-hashes "${REQUIREMENTS}"
        # nvidia-cutlass-dsl-libs-cu13 and -libs-base overwrite each other's copies of the
        # same files; only "cu13 first, base last" leaves a cutlass.cute vLLM can import.
        CUTLASS="$(grep -oE '^nvidia-cutlass-dsl==[0-9.]+' "${REQUIREMENTS}" | cut -d= -f3)"
        for part in ${CUTLASS:+cu13 base}; do
            "${UV}" pip sync --python "${VENV}/bin/python" --require-hashes \
                --reinstall-package "nvidia-cutlass-dsl-libs-${part}" "${REQUIREMENTS}"
        done
        echo "${LOCK_SHA}" > "${VENV}/.gepa-serving-lock.sha256"
    fi
    "${UV}" pip check --python "${VENV}/bin/python"
    "${VENV}/bin/python" -m examples.common.python_environment prepare \
        --path "${MANIFESTS}/serving-environments/${LOCK_SHA}.json"
done
