#!/bin/bash
# Submit the GEPA HotpotQA experiment to della from your laptop.
#
# Usage:
#   scripts/della/submit_hotpotqa.sh
#
# Use MODEL_PROFILE=qwen3.8-27b or MODEL_PROFILE=glm-5.3-flash. Each
# profile uses the same model for the student and proposer. The default
# BUDGET_PROFILE=campaign submits exactly six serial jobs: vanilla, ReAct V2,
# random-Controller ReAct V2, and selected-action GEPA at 6,871 calls, followed
# by vanilla and ReAct V2 at 13,742 calls. BUDGET_PROFILE=standard or expanded
# resubmits only the approved cells at one budget.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
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

if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=normal)" ]]; then
    echo "ERROR: commit the complete experiment source before a production submission" >&2
    exit 1
fi
HOTPOTQA_SOURCE_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
REMOTE_SOURCE_DIR="${REMOTE_DIR%/}/sources/${HOTPOTQA_SOURCE_COMMIT}"
GEPA_VENV_DIR="${REMOTE_DIR%/}/.venv"
HOTPOTQA_UV_VERSION="0.9.13"
GEPA_UV_BIN="${REMOTE_DIR%/}/.tools/uv-${HOTPOTQA_UV_VERSION}/uv"
SOURCE_MANIFEST_OUTPUT="$(mktemp)"

cleanup_local_files() {
    rm -f -- "${SOURCE_MANIFEST_OUTPUT}"
}
trap cleanup_local_files EXIT

# Tunable knobs (env overrides).
MODEL_PROFILE="${MODEL_PROFILE:-qwen3.8-27b}"
BUDGET_PROFILE="${BUDGET_PROFILE:-campaign}"
CONDITION="${CONDITION:-all}"
HOTPOTQA_CAMPAIGN_ID="${HOTPOTQA_CAMPAIGN_ID:-hotpotqa-final-v1}"
MAX_WORKERS="${MAX_WORKERS:-}"
WIKI17_DIR="${WIKI17_DIR:-${SCRATCH_BASE}/.cache/gepa/wiki17}"
GEN_GMU=0.92
GEN_MAX_LEN=262144
VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-}"
VLLM_API_SERVER_COUNT="${VLLM_API_SERVER_COUNT:-}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"
POSIT_DIR="${POSIT_DIR:-/home/${REMOTE_USER}/posit}"
DELLA_GPUS="${DELLA_GPUS:-}"
DELLA_CPUS_PER_TASK="${DELLA_CPUS_PER_TASK:-}"
DELLA_MEMORY="${DELLA_MEMORY:-}"
TIME="${TIME:-}"
STANDARD_TIME="${STANDARD_TIME:-}"
EXPANDED_TIME="${EXPANDED_TIME:-}"
MODEL_STORAGE="${MODEL_STORAGE:-/projects/BSTEWART/model_storage}"
GLM_SGLANG_IMAGE_DIGEST="sha256:0836f0160fa785e424e68d13ef88ddd548f87e6e11ad9f0e4de982e4f9188aaf"
GLM_SGLANG_IMAGE_URI="docker://lmsysorg/sglang@${GLM_SGLANG_IMAGE_DIGEST}"
GLM_SGLANG_IMAGE="${MODEL_STORAGE}/runtimes/sglang-glm-5.3-flash-x86_64.sif"
HOTPOTQA_PYTHON_VERSION="3.11.13"

if [[ ! "${HOTPOTQA_CAMPAIGN_ID}" =~ ^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?$ ]]; then
    echo "ERROR: HOTPOTQA_CAMPAIGN_ID must start and end with a letter or number" >&2
    exit 1
fi

case "${BUDGET_PROFILE}" in
    campaign)
        CAMPAIGN_BUDGET_LABEL="6871+13742"
        if [[ "${CONDITION}" != "all" ]]; then
            echo "ERROR: BUDGET_PROFILE=campaign requires CONDITION=all" >&2
            exit 1
        fi
        ;;
    standard)
        CAMPAIGN_BUDGET_LABEL="6871"
        case "${CONDITION}" in
            vanilla|react_v2|react_v2_random|action|all) ;;
            *)
                echo "ERROR: standard production runs allow vanilla, react_v2, react_v2_random, action, or all" >&2
                exit 1
                ;;
        esac
        ;;
    expanded)
        CAMPAIGN_BUDGET_LABEL="13742"
        case "${CONDITION}" in
            vanilla|react_v2|all) ;;
            *)
                echo "ERROR: expanded production runs allow only vanilla, react_v2, or all" >&2
                exit 1
                ;;
        esac
        ;;
    *)
        echo "ERROR: BUDGET_PROFILE must be campaign, standard, or expanded" >&2
        exit 1
        ;;
esac

case "${MODEL_PROFILE}" in
    qwen3.8-27b)
        if [[ "${GPU_PARTITION}" != "ailab" ]]; then
            echo "ERROR: Qwen3.8-27B production runs require GPU_PARTITION=ailab" >&2
            exit 1
        fi
        DELLA_GPUS="${DELLA_GPUS:-8}"
        DELLA_CPUS_PER_TASK="${DELLA_CPUS_PER_TASK:-64}"
        DELLA_MEMORY="${DELLA_MEMORY:-768G}"
        JOB_PARTITION="${GPU_PARTITION}"
        MAX_WORKERS="${MAX_WORKERS:-128}"
        VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
        VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-${DELLA_GPUS}}"
        VLLM_API_SERVER_COUNT="${VLLM_API_SERVER_COUNT:-${VLLM_DATA_PARALLEL_SIZE}}"
        VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
        if [[ "${DELLA_GPUS}" != "8" \
            || "${VLLM_DATA_PARALLEL_SIZE}" != "8" \
            || "${VLLM_API_SERVER_COUNT}" != "8" ]]; then
            echo "ERROR: scientific Qwen runs require 8 H200 data-parallel replicas and 8 API servers" >&2
            exit 1
        fi
        if [[ "${VLLM_MAX_NUM_SEQS}" != "1" ]]; then
            echo "ERROR: scientific Qwen runs require VLLM_MAX_NUM_SEQS=1" >&2
            exit 1
        fi
        MODEL="Qwen3.8-27B"
        SOLVER_MODEL_PATH="${MODEL_STORAGE}/${MODEL}"
        MODEL_SNAPSHOT_PROFILE="qwen3.8-27b"
        SOLVER_SERVED_NAME="Qwen/Qwen3.8-27B"
        SOLVER_MODEL="hosted_vllm/Qwen/Qwen3.8-27B"
        SOLVER_API_BASE=""
        REFLECTION_API_BASE=""
        STANDARD_TIME="${STANDARD_TIME:-${TIME:-72:00:00}}"
        EXPANDED_TIME="${EXPANDED_TIME:-${TIME:-144:00:00}}"
        ;;
    glm-5.3-flash)
        if [[ "${GPU_PARTITION}" != "ailab" ]]; then
            echo "ERROR: GLM-5.3-Flash production runs require GPU_PARTITION=ailab" >&2
            exit 1
        fi
        DELLA_GPUS="${DELLA_GPUS:-8}"
        DELLA_CPUS_PER_TASK="${DELLA_CPUS_PER_TASK:-64}"
        DELLA_MEMORY="${DELLA_MEMORY:-768G}"
        JOB_PARTITION="${GPU_PARTITION}"
        MAX_WORKERS="${MAX_WORKERS:-8}"
        VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-8}"
        VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-1}"
        VLLM_API_SERVER_COUNT="${VLLM_API_SERVER_COUNT:-1}"
        VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
        MODEL="GLM-5.3-Flash"
        SOLVER_MODEL_PATH="${MODEL_STORAGE}/${MODEL}"
        MODEL_SNAPSHOT_PROFILE="glm-5.3-flash"
        SOLVER_SERVED_NAME="zai-org/GLM-5.3-Flash"
        SOLVER_MODEL="hosted_vllm/zai-org/GLM-5.3-Flash"
        SOLVER_API_BASE=""
        REFLECTION_API_BASE=""
        if [[ "${DELLA_GPUS}" != "8" \
            || "${VLLM_TENSOR_PARALLEL_SIZE}" != "8" \
            || "${VLLM_DATA_PARALLEL_SIZE}" != "1" \
            || "${VLLM_API_SERVER_COUNT}" != "1" \
            || "${VLLM_MAX_NUM_SEQS}" != "8" ]]; then
            echo "ERROR: scientific GLM runs require one TP8 replica on one eight-H200 node" >&2
            exit 1
        fi
        STANDARD_TIME="${STANDARD_TIME:-${TIME:-144:00:00}}"
        EXPANDED_TIME="${EXPANDED_TIME:-${TIME:-144:00:00}}"
        ;;
    *)
        echo "ERROR: MODEL_PROFILE must be qwen3.8-27b or glm-5.3-flash" >&2
        exit 1
        ;;
esac
REFLECTION_MODEL="${SOLVER_MODEL}"
REFLECTION_API_BASE="${SOLVER_API_BASE}"

validate_della_wall_time() {
    local wall_time="$1"
    local label="$2"
    local days=0
    local hours=0
    local minutes=0
    local seconds=0
    local total_seconds=0

    if [[ "${wall_time}" =~ ^([0-9]+)-([0-9]{1,2}):([0-9]{2}):([0-9]{2})$ ]]; then
        days=$((10#${BASH_REMATCH[1]}))
        hours=$((10#${BASH_REMATCH[2]}))
        minutes=$((10#${BASH_REMATCH[3]}))
        seconds=$((10#${BASH_REMATCH[4]}))
        if (( hours > 23 )); then
            echo "ERROR: ${label} has an invalid hour field: ${wall_time}" >&2
            exit 1
        fi
    elif [[ "${wall_time}" =~ ^([0-9]+):([0-9]{2}):([0-9]{2})$ ]]; then
        hours=$((10#${BASH_REMATCH[1]}))
        minutes=$((10#${BASH_REMATCH[2]}))
        seconds=$((10#${BASH_REMATCH[3]}))
    else
        echo "ERROR: ${label} must use Slurm HH:MM:SS or D-HH:MM:SS format" >&2
        exit 1
    fi
    if (( minutes > 59 || seconds > 59 )); then
        echo "ERROR: ${label} has an invalid minute or second field: ${wall_time}" >&2
        exit 1
    fi
    total_seconds=$((days * 86400 + hours * 3600 + minutes * 60 + seconds))
    if (( total_seconds < 1 || total_seconds > 518400 )); then
        echo "ERROR: ${label} must be positive and no longer than Della's 144-hour limit" >&2
        exit 1
    fi
}

validate_della_wall_time "${STANDARD_TIME}" "STANDARD_TIME"
validate_della_wall_time "${EXPANDED_TIME}" "EXPANDED_TIME"

for positive_integer in \
    "${DELLA_CPUS_PER_TASK}" \
    "${MAX_WORKERS}" \
    "${VLLM_TENSOR_PARALLEL_SIZE}" \
    "${VLLM_DATA_PARALLEL_SIZE}" \
    "${VLLM_API_SERVER_COUNT}" \
    "${VLLM_MAX_NUM_SEQS}" \
    "${VLLM_MAX_NUM_BATCHED_TOKENS}"
do
    if [[ ! "${positive_integer}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: concurrency and CPU settings must be positive integers" >&2
        exit 1
    fi
done
if [[ "${DELLA_GPUS}" != "8" ]]; then
    echo "ERROR: DELLA_GPUS must be 8 for either scientific model profile" >&2
    exit 1
fi
if (( DELLA_CPUS_PER_TASK > DELLA_GPUS * 8 )); then
    echo "ERROR: Della permits at most 8 CPU cores per AI Lab H200" >&2
    exit 1
fi
if [[ "${MODEL_PROFILE}" == "qwen3.8-27b" ]]; then
    if (( VLLM_DATA_PARALLEL_SIZE > DELLA_GPUS )); then
        echo "ERROR: VLLM_DATA_PARALLEL_SIZE cannot exceed DELLA_GPUS" >&2
        exit 1
    fi
fi
if (( VLLM_TENSOR_PARALLEL_SIZE * VLLM_DATA_PARALLEL_SIZE != DELLA_GPUS )); then
    echo "ERROR: tensor-parallel size times data-parallel size must use all eight allocated GPUs" >&2
    exit 1
fi

SBATCH_RESOURCE_ARGS=(
    "--nodes=1"
    "--ntasks=1"
    "--cpus-per-task=${DELLA_CPUS_PER_TASK}"
    "--mem=${DELLA_MEMORY}"
)
if [[ -n "${JOB_PARTITION}" ]]; then
    SBATCH_RESOURCE_ARGS+=("--partition=${JOB_PARTITION}")
fi
if (( DELLA_GPUS > 0 )); then
    SBATCH_RESOURCE_ARGS+=("--gres=gpu:${DELLA_GPUS}")
fi
printf -v SBATCH_RESOURCE_COMMAND ' %q' "${SBATCH_RESOURCE_ARGS[@]}"

# Step 1: sync the clean local commit whose identity is recorded in every run.
echo "==> syncing code"
SYNC_SOURCE_COMMIT="${HOTPOTQA_SOURCE_COMMIT}" \
SYNC_REMOTE_DIR="${REMOTE_SOURCE_DIR}" \
SYNC_MANIFEST_OUTPUT="${SOURCE_MANIFEST_OUTPUT}" \
    "${SCRIPT_DIR}/sync_to_della.sh"
HOTPOTQA_SOURCE_MANIFEST_SHA256="$(tr -d '\n' < "${SOURCE_MANIFEST_OUTPUT}")"
if [[ ! "${HOTPOTQA_SOURCE_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "ERROR: source staging did not return a valid manifest digest" >&2
    exit 1
fi

# Step 2: submit sbatch on della login node.
echo "==> submitting HotpotQA: profile=${MODEL_PROFILE} solver=${SOLVER_MODEL} reflection=${REFLECTION_MODEL}"
echo "==> scientific contract: budget_profile=${BUDGET_PROFILE} budget=${CAMPAIGN_BUDGET_LABEL} condition=${CONDITION} merge=off"
echo "==> method: frozen Wiki-2017/BM25 k=7 seed=0 workers=${MAX_WORKERS} two-stage structured prompts"
echo "==> Della resources: partition=${JOB_PARTITION:-cluster-default} gpus=${DELLA_GPUS} cpus=${DELLA_CPUS_PER_TASK} memory=${DELLA_MEMORY}"
if [[ "${MODEL_PROFILE}" == "qwen3.8-27b" ]]; then
    echo "==> Qwen POSIT/vLLM: tp=1 dp=8 api_servers=8 max_num_seqs=1/replica max_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS}"
else
    echo "==> GLM SGLang: tp=8 ep=8 max_running_requests=8 BF16-KV TileLang-DSA deep_gemm no-speculation no-DP-attention"
fi

ssh -o BatchMode=yes -o StrictHostKeyChecking=yes \
    "${REMOTE_USER}@${REMOTE_HOST}" bash -l <<REMOTE_SCRIPT
set -euo pipefail
cd "${REMOTE_SOURCE_DIR}"

if [[ "\$(tr -d '\n' < .gepa-source-commit)" != "${HOTPOTQA_SOURCE_COMMIT}" ]]; then
    echo "ERROR: staged source does not match ${HOTPOTQA_SOURCE_COMMIT}" >&2
    exit 1
fi
if [[ "\$(tr -d '\n' < .gepa-source-manifest.sha256)" != "${HOTPOTQA_SOURCE_MANIFEST_SHA256}" ]] \
    || [[ "\$(sha256sum .gepa-source-manifest.sha256sums | cut -d' ' -f1)" != "${HOTPOTQA_SOURCE_MANIFEST_SHA256}" ]] \
    || ! sha256sum --check .gepa-source-manifest.sha256sums >/dev/null; then
    echo "ERROR: staged source bytes do not match ${HOTPOTQA_SOURCE_MANIFEST_SHA256}" >&2
    exit 1
fi
export PYTHONPATH="${REMOTE_SOURCE_DIR}/src:${REMOTE_SOURCE_DIR}"
HOTPOTQA_PYTHON_VERSION="${HOTPOTQA_PYTHON_VERSION}"
HOTPOTQA_UV_VERSION="${HOTPOTQA_UV_VERSION}"
GEPA_UV_BIN="${GEPA_UV_BIN}"
export UV_PROJECT_ENVIRONMENT="${GEPA_VENV_DIR}"
HOTPOTQA_ENV_SPEC_SHA256="\$(
    {
        sha256sum pyproject.toml uv.lock
        printf 'python=%s\n' "\${HOTPOTQA_PYTHON_VERSION}"
        printf 'uv=%s\n' "\${HOTPOTQA_UV_VERSION}"
    } | sha256sum | cut -d' ' -f1
)"
if [[ ! -f "${GEPA_VENV_DIR}/.gepa-env-spec.sha256" \
    || "\$(tr -d '\n' < "${GEPA_VENV_DIR}/.gepa-env-spec.sha256")" != "\${HOTPOTQA_ENV_SPEC_SHA256}" \
    || ! -f "${GEPA_VENV_DIR}/.gepa-python-version" \
    || "\$(tr -d '\n' < "${GEPA_VENV_DIR}/.gepa-python-version")" != "\${HOTPOTQA_PYTHON_VERSION}" \
    || "\$("${GEPA_VENV_DIR}/bin/python" -c 'import platform; print(platform.python_version())')" != "\${HOTPOTQA_PYTHON_VERSION}" \
    || ! -x "\${GEPA_UV_BIN}" \
    || ! -f "${GEPA_VENV_DIR}/.gepa-uv-version" \
    || "\$(tr -d '\n' < "${GEPA_VENV_DIR}/.gepa-uv-version")" != "\${HOTPOTQA_UV_VERSION}" \
    || "\$("\${GEPA_UV_BIN}" --version)" != "uv \${HOTPOTQA_UV_VERSION}"* \
    || ! -f "${GEPA_VENV_DIR}/.gepa-uv-sha256" \
    || "\$(sha256sum "\${GEPA_UV_BIN}" | cut -d' ' -f1)" != "\$(tr -d '\n' < "${GEPA_VENV_DIR}/.gepa-uv-sha256")" ]]; then
    echo "ERROR: shared GEPA environment does not match the staged dependency lock; run scripts/della/build_env.sh" >&2
    exit 1
fi
if ! "\${GEPA_UV_BIN}" sync --python "\${HOTPOTQA_PYTHON_VERSION}" --frozen --check --no-install-project \
    --extra dev --extra wiki17 --group hotpotqa-task-program; then
    echo "ERROR: shared GEPA environment has drifted from uv.lock; run scripts/della/build_env.sh" >&2
    exit 1
fi
HOTPOTQA_UV_SHA256="\$(sha256sum "\${GEPA_UV_BIN}" | cut -d' ' -f1)"

export DSPY_CACHEDIR="${SCRATCH_BASE}/.cache/dspy"
mkdir -p "\${DSPY_CACHEDIR}"
"${GEPA_VENV_DIR}/bin/python" - <<'PY'
from examples.hotpotqa.utils import validate_hotpotqa_dspy_runtime

try:
    validate_hotpotqa_dspy_runtime()
except RuntimeError as exc:
    raise SystemExit(f"{exc} Run scripts/della/build_env.sh first.") from exc
PY

if ! "${GEPA_VENV_DIR}/bin/python" -m examples.common.wiki17_bm25 verify --root "${WIKI17_DIR}" >/dev/null; then
    echo "ERROR: Wiki-2017 is not prepared at ${WIKI17_DIR}; run scripts/della/build_env.sh first" >&2
    exit 1
fi
MODEL_INTEGRITY_MANIFEST="${SOLVER_MODEL_PATH}/.gepa-model-integrity.json"
if [[ ! -d "${SOLVER_MODEL_PATH}" || ! -s "\${MODEL_INTEGRITY_MANIFEST}" ]]; then
    echo "ERROR: pinned ${MODEL} checkpoint is not staged at ${SOLVER_MODEL_PATH}; run scripts/della/build_env.sh first" >&2
    exit 1
fi
echo "==> found staged local ${MODEL_SNAPSHOT_PROFILE} checkpoint at ${SOLVER_MODEL_PATH}"
"${GEPA_VENV_DIR}/bin/python" - <<'PY'
from examples.hotpotqa.utils import load_hotpotqa_dataset

train, val, test = load_hotpotqa_dataset(seed=0)
if (len(train), len(val), len(test)) != (150, 300, 300):
    raise SystemExit(
        "Pinned HotPotQA data did not produce the required 150/300/300 scientific splits."
    )
PY

GEPA_ENV_MANIFEST="${SCRATCH_BASE}/.cache/gepa/python-environments/gepa-\${HOTPOTQA_ENV_SPEC_SHA256}.json"
HOTPOTQA_GEPA_ENV_SHA256="\$(
    "${GEPA_VENV_DIR}/bin/python" -m examples.common.python_environment verify --path "\${GEPA_ENV_MANIFEST}"
)"
if [[ ! "\${HOTPOTQA_GEPA_ENV_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "ERROR: realized GEPA environment differs from the frozen production environment" >&2
    exit 1
fi

HOTPOTQA_POSIT_ENV_SHA256=""
HOTPOTQA_SGLANG_IMAGE_SHA256=""
if [[ "${MODEL_PROFILE}" == "qwen3.8-27b" ]]; then
    VLLM_PY="${POSIT_DIR}/src/.venv/bin/python"
    if [[ ! -x "\${VLLM_PY}" ]]; then
        echo "ERROR: missing \${VLLM_PY}; run scripts/della/build_env.sh" >&2
        exit 1
    fi
    if [[ -n "\$(git -C "${POSIT_DIR}" status --porcelain --untracked-files=normal)" ]]; then
        echo "ERROR: commit or remove local POSIT changes before a production submission" >&2
        exit 1
    fi
    HOTPOTQA_POSIT_COMMIT="\$(git -C "${POSIT_DIR}" rev-parse HEAD)"
    POSIT_ENV_MANIFEST="${SCRATCH_BASE}/.cache/gepa/posit-environments/\${HOTPOTQA_POSIT_COMMIT}.json"
    if ! "\${GEPA_UV_BIN}" pip check --python "\${VLLM_PY}"; then
        echo "ERROR: POSIT serving environment has inconsistent dependencies" >&2
        exit 1
    fi
    if [[ ! -f "\${POSIT_ENV_MANIFEST}" ]]; then
        echo "ERROR: POSIT serving environment is not frozen; run scripts/della/build_env.sh" >&2
        exit 1
    fi
    HOTPOTQA_POSIT_ENV_SHA256="\$(sha256sum "\${POSIT_ENV_MANIFEST}" | cut -d' ' -f1)"
    if [[ ! "\${HOTPOTQA_POSIT_ENV_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
        echo "ERROR: POSIT serving environment is not frozen; run scripts/della/build_env.sh" >&2
        exit 1
    fi
else
    APPTAINER_BIN="\$(command -v apptainer || true)"
    GLM_IMAGE_SOURCE_PATH="${GLM_SGLANG_IMAGE}.source"
    GLM_IMAGE_SHA_PATH="${GLM_SGLANG_IMAGE}.sha256"
    if [[ -z "\${APPTAINER_BIN}" \
        || ! -f "${GLM_SGLANG_IMAGE}" \
        || ! -f "\${GLM_IMAGE_SOURCE_PATH}" \
        || "\$(tr -d '\n' < "\${GLM_IMAGE_SOURCE_PATH}")" != "${GLM_SGLANG_IMAGE_URI}" \
        || ! -f "\${GLM_IMAGE_SHA_PATH}" ]]; then
        echo "ERROR: pinned GLM SGLang runtime is absent; run scripts/della/build_env.sh" >&2
        exit 1
    fi
    HOTPOTQA_SGLANG_IMAGE_SHA256="\$(sha256sum "${GLM_SGLANG_IMAGE}" | cut -d' ' -f1)"
    if [[ "\${HOTPOTQA_SGLANG_IMAGE_SHA256}" != "\$(tr -d '\n' < "\${GLM_IMAGE_SHA_PATH}")" ]]; then
        echo "ERROR: GLM SGLang image differs from its frozen digest" >&2
        exit 1
    fi
    "\${APPTAINER_BIN}" inspect "${GLM_SGLANG_IMAGE}" >/dev/null
fi

SBATCH_BIN="\$(command -v sbatch)"
SBATCH_HELP="\$("\${SBATCH_BIN}" --help 2>&1)"
if [[ "\${SBATCH_HELP}" != *"--export-file"* ]]; then
    echo "ERROR: this Slurm installation does not support secure --export-file submission" >&2
    exit 1
fi

if [[ "${BUDGET_PROFILE}" == "campaign" ]]; then
    SUBMIT_BUDGET_PROFILES=(standard standard standard standard expanded expanded)
    SUBMIT_CONDITIONS=(vanilla react_v2 react_v2_random action vanilla react_v2)
elif [[ "${BUDGET_PROFILE}" == "standard" && "${CONDITION}" == "all" ]]; then
    SUBMIT_BUDGET_PROFILES=(standard standard standard standard)
    SUBMIT_CONDITIONS=(vanilla react_v2 react_v2_random action)
elif [[ "${BUDGET_PROFILE}" == "expanded" && "${CONDITION}" == "all" ]]; then
    SUBMIT_BUDGET_PROFILES=(expanded expanded)
    SUBMIT_CONDITIONS=(vanilla react_v2)
else
    SUBMIT_BUDGET_PROFILES=("${BUDGET_PROFILE}")
    SUBMIT_CONDITIONS=("${CONDITION}")
fi
if [[ "\${#SUBMIT_BUDGET_PROFILES[@]}" != "\${#SUBMIT_CONDITIONS[@]}" ]]; then
    echo "ERROR: HotPotQA campaign budget and condition lists differ in length" >&2
    exit 1
fi

HOTPOTQA_LOG_DIR="${SCRATCH_BASE}/logs/hotpotqa/${HOTPOTQA_CAMPAIGN_ID}/${HOTPOTQA_SOURCE_COMMIT}"
mkdir -p "${HOTPOTQA_LOG_DIR}"
umask 077
SBATCH_EXPORT_FILE=""
cleanup_export_file() {
    if [[ -n "\${SBATCH_EXPORT_FILE}" ]]; then
        rm -f -- "\${SBATCH_EXPORT_FILE}"
    fi
}
trap cleanup_export_file EXIT

write_sbatch_export_file() {
    local run_budget_profile="$1"
    local run_max_metric_calls="$2"
    local run_condition="$3"
    local canary_only="$4"

    SBATCH_EXPORT_FILE="\$(mktemp)"
    printf '%s\0' \
        "MODEL_PROFILE=${MODEL_PROFILE}" \
        "BUDGET_PROFILE=\${run_budget_profile}" \
        "MAX_METRIC_CALLS=\${run_max_metric_calls}" \
        "CONDITION=\${run_condition}" \
        "HOTPOTQA_CANARY_ONLY=\${canary_only}" \
        "HOTPOTQA_CAMPAIGN_ID=${HOTPOTQA_CAMPAIGN_ID}" \
        "MAX_WORKERS=${MAX_WORKERS}" \
        "WIKI17_DIR=${WIKI17_DIR}" \
        "GEN_GMU=${GEN_GMU}" \
        "GEN_MAX_LEN=${GEN_MAX_LEN}" \
        "VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE}" \
        "VLLM_DATA_PARALLEL_SIZE=${VLLM_DATA_PARALLEL_SIZE}" \
        "VLLM_API_SERVER_COUNT=${VLLM_API_SERVER_COUNT}" \
        "VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS}" \
        "VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS}" \
        "HEALTH_TIMEOUT=${HEALTH_TIMEOUT}" \
        "POSIT_DIR=${POSIT_DIR}" \
        "MODEL_STORAGE=${MODEL_STORAGE}" \
        "GLM_SGLANG_IMAGE=${GLM_SGLANG_IMAGE}" \
        "GLM_SGLANG_IMAGE_URI=${GLM_SGLANG_IMAGE_URI}" \
        "HOTPOTQA_SGLANG_IMAGE_SHA256=\${HOTPOTQA_SGLANG_IMAGE_SHA256}" \
        "SCRATCH_BASE=${SCRATCH_BASE}" \
        "GEPA_VENV_DIR=${GEPA_VENV_DIR}" \
        "HOME=\${HOME}" \
        "PATH=\${PATH}" \
        "LANG=C.UTF-8" \
        "LC_ALL=C.UTF-8" \
        "USER=${REMOTE_USER}" \
        "HOTPOTQA_SOURCE_COMMIT=${HOTPOTQA_SOURCE_COMMIT}" \
        "HOTPOTQA_SOURCE_MANIFEST_SHA256=${HOTPOTQA_SOURCE_MANIFEST_SHA256}" \
        "HOTPOTQA_PYTHON_VERSION=${HOTPOTQA_PYTHON_VERSION}" \
        "HOTPOTQA_UV_VERSION=${HOTPOTQA_UV_VERSION}" \
        "HOTPOTQA_UV_SHA256=\${HOTPOTQA_UV_SHA256}" \
        "GEPA_UV_BIN=${GEPA_UV_BIN}" \
        "HOTPOTQA_ENV_SPEC_SHA256=\${HOTPOTQA_ENV_SPEC_SHA256}" \
        "HOTPOTQA_GEPA_ENV_SHA256=\${HOTPOTQA_GEPA_ENV_SHA256}" \
        "HOTPOTQA_POSIT_ENV_SHA256=\${HOTPOTQA_POSIT_ENV_SHA256}" \
        "HOTPOTQA_PRODUCTION_LAUNCH=1" \
        > "\${SBATCH_EXPORT_FILE}"
}

PREVIOUS_JOB_ID=""
if [[ "${MODEL_PROFILE}" == "glm-5.3-flash" ]]; then
    write_sbatch_export_file standard 6871 react_v2 1
    SBATCH_OUTPUT="\$(
        env -i \
            HOME="\${HOME}" \
            USER="${REMOTE_USER}" \
            PATH="\${PATH}" \
            LANG=C.UTF-8 \
            LC_ALL=C.UTF-8 \
            "\${SBATCH_BIN}"${SBATCH_RESOURCE_COMMAND} \
            --parsable \
            --job-name="gepa-hp-glm-canary" \
            --output="${HOTPOTQA_LOG_DIR}/hotpotqa-%x-%j.log" \
            --time="04:00:00" \
            --export=ALL \
            --export-file="\${SBATCH_EXPORT_FILE}" \
            examples/hotpotqa/run_hotpotqa.sbatch
    )"
    PREVIOUS_JOB_ID="\${SBATCH_OUTPUT%%;*}"
    if [[ ! "\${PREVIOUS_JOB_ID}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: sbatch returned an invalid GLM canary job id: \${SBATCH_OUTPUT}" >&2
        exit 1
    fi
    rm -f -- "\${SBATCH_EXPORT_FILE}"
    SBATCH_EXPORT_FILE=""
    echo "==> submitted pinned GLM four-tool canary: job \${PREVIOUS_JOB_ID}"
fi

for CELL_INDEX in "\${!SUBMIT_CONDITIONS[@]}"; do
    RUN_BUDGET_PROFILE="\${SUBMIT_BUDGET_PROFILES[\${CELL_INDEX}]}"
    RUN_CONDITION="\${SUBMIT_CONDITIONS[\${CELL_INDEX}]}"
    case "\${RUN_BUDGET_PROFILE}" in
        standard)
            RUN_MAX_METRIC_CALLS=6871
            RUN_TIME="${STANDARD_TIME}"
            ;;
        expanded)
            RUN_MAX_METRIC_CALLS=13742
            RUN_TIME="${EXPANDED_TIME}"
            ;;
        *)
            echo "ERROR: generated an unsupported HotPotQA budget profile: \${RUN_BUDGET_PROFILE}" >&2
            exit 1
            ;;
    esac
    write_sbatch_export_file "\${RUN_BUDGET_PROFILE}" "\${RUN_MAX_METRIC_CALLS}" "\${RUN_CONDITION}" 0

    DEPENDENCY_ARGS=()
    if [[ -n "\${PREVIOUS_JOB_ID}" ]]; then
        DEPENDENCY_ARGS+=("--dependency=afterok:\${PREVIOUS_JOB_ID}")
    fi
    SBATCH_OUTPUT="\$(
        env -i \
            HOME="\${HOME}" \
            USER="${REMOTE_USER}" \
            PATH="\${PATH}" \
            LANG=C.UTF-8 \
            LC_ALL=C.UTF-8 \
            "\${SBATCH_BIN}"${SBATCH_RESOURCE_COMMAND} \
            --parsable \
            --job-name="gepa-hp-${MODEL_PROFILE}-\${RUN_BUDGET_PROFILE}-\${RUN_CONDITION}" \
            --output="${HOTPOTQA_LOG_DIR}/hotpotqa-%x-%j.log" \
            --time="\${RUN_TIME}" \
            --export=ALL \
            --export-file="\${SBATCH_EXPORT_FILE}" \
            "\${DEPENDENCY_ARGS[@]}" \
            examples/hotpotqa/run_hotpotqa.sbatch
    )"
    PREVIOUS_JOB_ID="\${SBATCH_OUTPUT%%;*}"
    if [[ ! "\${PREVIOUS_JOB_ID}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: sbatch returned an invalid job id: \${SBATCH_OUTPUT}" >&2
        exit 1
    fi
    rm -f -- "\${SBATCH_EXPORT_FILE}"
    SBATCH_EXPORT_FILE=""
    echo "==> submitted \${RUN_BUDGET_PROFILE}/\${RUN_CONDITION}: job \${PREVIOUS_JOB_ID}"
done

echo "==> approved HotPotQA campaign chain submitted. Check status with: squeue -u ${REMOTE_USER}"
REMOTE_SCRIPT
