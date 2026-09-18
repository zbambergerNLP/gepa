#!/bin/bash
# Execute qualification or one cell against the already verified local endpoints.
set -euo pipefail

python() { "${GEPA_UV_BIN}" run --no-project --python "${GEPA_VENV_DIR}/bin/python" python "$@"; }
PY=python

generator_reports_expected_model() {
    curl --max-time 10 -fsS "${REFLECTION_API_BASE}/models" | "${PY}" -c \
        'import json, sys; assert [m["id"] for m in json.load(sys.stdin)["data"]] == [sys.argv[1]]' \
        "${REFLECTION_MODEL#hosted_vllm/}"
}

RUNTIME_IDENTITY="teacher=${HOTPOTQA_TEACHER_RUNTIME:-none};python=${HOTPOTQA_PYTHON_VERSION};uv=${HOTPOTQA_UV_VERSION};uv_sha=${HOTPOTQA_UV_SHA256};gepa_env=${HOTPOTQA_GEPA_ENV_SHA256};model_manifest=${HOTPOTQA_MODEL_INTEGRITY_SHA256};engine=${HOTPOTQA_SERVING_ENGINE};vllm=${HOTPOTQA_VLLM_VERSION};torch=${HOTPOTQA_TORCH_VERSION};cuda=${HOTPOTQA_CUDA_VERSION};cuda_module=${HOTPOTQA_CUDA_MODULE};transformers=${HOTPOTQA_TRANSFORMERS_VERSION};serving_lock=${HOTPOTQA_SERVING_LOCK_SHA256};serving_env=${HOTPOTQA_SERVING_ENV_SHA256};gpu=${HOTPOTQA_GPU_RUNTIME};${HOTPOTQA_SERVE_ARGUMENTS}"
# Only an explicitly reviewed additive cell can share the existing qualification
# and baseline identity. Its source verification, logs and recovery remain distinct.
export CONDITION MAX_METRIC_CALLS
COMPARISON_SOURCE_IDENTITY="$("${PY}" -m examples.hotpotqa.source_compatibility)"
read -r CAMPAIGN_SOURCE_COMMIT CAMPAIGN_SOURCE_MANIFEST_SHA256 <<< "${COMPARISON_SOURCE_IDENTITY}"
CAMPAIGN_QUALITY_IDENTITY="teacher=${HOTPOTQA_TEACHER_RUNTIME:-none};workers=${MAX_WORKERS};source=${CAMPAIGN_SOURCE_COMMIT};source_manifest=${CAMPAIGN_SOURCE_MANIFEST_SHA256};python=${HOTPOTQA_PYTHON_VERSION};uv=${HOTPOTQA_UV_VERSION};uv_sha=${HOTPOTQA_UV_SHA256};env_spec=${HOTPOTQA_ENV_SPEC_SHA256};gepa_env=${HOTPOTQA_GEPA_ENV_SHA256};model=${HOTPOTQA_MODEL_REVISION};model_manifest=${HOTPOTQA_MODEL_INTEGRITY_SHA256};engine=${HOTPOTQA_SERVING_ENGINE};vllm=${HOTPOTQA_VLLM_VERSION};torch=${HOTPOTQA_TORCH_VERSION};cuda=${HOTPOTQA_CUDA_VERSION};cuda_module=${HOTPOTQA_CUDA_MODULE};transformers=${HOTPOTQA_TRANSFORMERS_VERSION};litellm=${HOTPOTQA_LITELLM_VERSION};serving_lock=${HOTPOTQA_SERVING_LOCK_SHA256};serving_env=${HOTPOTQA_SERVING_ENV_SHA256};gpu=${HOTPOTQA_GPU_RUNTIME};serve=${HOTPOTQA_SERVE_ARGUMENTS};request_seed=0;dspy=${HOTPOTQA_DSPY_COMMIT};hotpot=${HOTPOTQA_DATA_REVISION};wiki17=${WIKI17_REVISION};index=${WIKI17_INTEGRITY_SHA256}"
CAMPAIGN_IDENTITY_SHA256="$(printf '%s' "${CAMPAIGN_QUALITY_IDENTITY}" | sha256sum | cut -d' ' -f1)"
# DSPy initializes storage on import even though task LMs disable response
# caching. Keep that initialization off the home filesystem.
RUNTIME_IDENTITY_SHA256="$(printf '%s' "${RUNTIME_IDENTITY}" | sha256sum | cut -d' ' -f1)"
CACHE_IDENTITY="source=${HOTPOTQA_SOURCE_COMMIT};source_manifest=${HOTPOTQA_SOURCE_MANIFEST_SHA256};python=${HOTPOTQA_PYTHON_VERSION};uv=${HOTPOTQA_UV_VERSION};uv_sha=${HOTPOTQA_UV_SHA256};env_spec=${HOTPOTQA_ENV_SPEC_SHA256};gepa_env=${HOTPOTQA_GEPA_ENV_SHA256};model=${HOTPOTQA_MODEL_REVISION};model_manifest=${HOTPOTQA_MODEL_INTEGRITY_SHA256};budget_profile=${BUDGET_PROFILE};runtime=${RUNTIME_IDENTITY_SHA256};litellm=${HOTPOTQA_LITELLM_VERSION};dspy=${HOTPOTQA_DSPY_COMMIT};hotpot=${HOTPOTQA_DATA_REVISION};wiki17=${WIKI17_REVISION};index=${WIKI17_INTEGRITY_SHA256}"
CACHE_IDENTITY_SHA256="$(printf '%s' "${CACHE_IDENTITY}" | sha256sum | cut -d' ' -f1)"
RUNTIME_CACHE_KEY="${MODEL_PROFILE}-${CACHE_IDENTITY_SHA256}"
export DSPY_CACHEDIR="${SCRATCH_BASE}/.cache/dspy/hotpotqa/${RUNTIME_CACHE_KEY}"
mkdir -p "${DSPY_CACHEDIR}"
echo "==> HotPotQA evaluation and model-response caching: disabled"
# The serving process is already running, so these limits apply only to the
# evaluator and prevent its per-example threads from spawning nested CPU pools.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
SOLVER_API_ARG=()
[[ -n "${SOLVER_API_BASE}" ]] && SOLVER_API_ARG=(--solver-api-base "${SOLVER_API_BASE}")
REFLECTION_API_ARG=()
[[ -n "${REFLECTION_API_BASE}" ]] && REFLECTION_API_ARG=(--reflection-api-base "${REFLECTION_API_BASE}")

THINKING_PROBE_PARENT="$(mktemp -d "${LOG_DIR}/thinking-budget-${MODEL_PROFILE}-${SLURM_JOB_ID:-local}.XXXXXX")"
"${PY}" -m examples.hotpotqa.verify_thinking_budget \
    --model "${SOLVER_MODEL}" --api-base "${SOLVER_API_BASE}" \
    --output-dir "${THINKING_PROBE_PARENT}/exchange"
if [[ "${REFLECTION_MODEL}" == "hosted_vllm/deepseek-ai/DeepSeek-V4.1-Flash" ]]; then
    "${PY}" -m examples.hotpotqa.verify_thinking_budget \
        --model "${REFLECTION_MODEL}" --api-base "${REFLECTION_API_BASE}" \
        --output-dir "${THINKING_PROBE_PARENT}/teacher"
    DEEPSEEK_CANARY_DIR="${SCRATCH_BASE}/.cache/gepa/hotpotqa-canaries/${HOTPOTQA_CAMPAIGN_ID}"
    CANARY_IDENTITY_SHA256="${CAMPAIGN_IDENTITY_SHA256}"
    CANARY_WORKERS=1
    if [[ "${MODEL_PROFILE}" == "deepseek-teacher-qwen-student" ]]; then
        CANARY_IDENTITY_SHA256="$(printf '%s' "${HOTPOTQA_SOURCE_COMMIT};${HOTPOTQA_SOURCE_MANIFEST_SHA256};${HOTPOTQA_GEPA_ENV_SHA256};${HOTPOTQA_TEACHER_RUNTIME}" | sha256sum | cut -d' ' -f1)"
        CANARY_WORKERS="${TEACHER_MAX_NUM_SEQS}"
    fi
    DEEPSEEK_CANARY_MARKER="${DEEPSEEK_CANARY_DIR}/${CANARY_IDENTITY_SHA256}.ok"
    mkdir -p "${DEEPSEEK_CANARY_DIR}"
    if [[ "${HOTPOTQA_CANARY_ONLY}" == "1" \
        || ( "${HOTPOTQA_PILOT_ONLY}" == "1" && ! -f "${DEEPSEEK_CANARY_MARKER}" ) ]]; then
        echo "==> running the fail-closed DeepSeek multi-tool canary"
        "${PY}" -m examples.hotpotqa.runtime_canary \
            --model "${REFLECTION_MODEL}" \
            --api-base "${REFLECTION_API_BASE}" \
            --attempts 20 --workers "${CANARY_WORKERS}" \
            --attempt-log "${LOG_DIR}/provider-attempts-${SLURM_JOB_ID:-local}.jsonl"
        if ! kill -0 "${GEN_PID}" 2>/dev/null || ! generator_reports_expected_model; then
            echo "ERROR: DeepSeek endpoint failed after the runtime canary" >&2
            exit 1
        fi
        DEEPSEEK_CANARY_TEMP="$(mktemp "${DEEPSEEK_CANARY_DIR}/.canary.XXXXXX")"
        printf '%s\n' "${CANARY_IDENTITY_SHA256}" > "${DEEPSEEK_CANARY_TEMP}"
        mv "${DEEPSEEK_CANARY_TEMP}" "${DEEPSEEK_CANARY_MARKER}"
        echo "==> DeepSeek runtime canary passed: ${CANARY_IDENTITY_SHA256}"
        if [[ "${HOTPOTQA_CANARY_ONLY}" == "1" ]]; then
            exit 0
        fi
    fi
    if [[ ! -f "${DEEPSEEK_CANARY_MARKER}" \
        || "$(tr -d '\n' < "${DEEPSEEK_CANARY_MARKER}")" != "${CANARY_IDENTITY_SHA256}" ]]; then
        echo "ERROR: the exact DeepSeek runtime has not passed its fail-closed canary" >&2
        exit 1
    fi
fi

echo "==> checking native tool-call compatibility"
"${PY}" - "${REFLECTION_MODEL}" "${REFLECTION_API_BASE}" <<'PY'
import json
import sys

from examples.hotpotqa.utils import resolve_hotpotqa_lm_kwargs
from gepa.lm import LM

model = sys.argv[1]
api_base = sys.argv[2] or None
kwargs = resolve_hotpotqa_lm_kwargs(model, api_base, role="optimizer")
completion = LM(model, **kwargs).complete_with_tools(
    [{"role": "user", "content": "Call the echo function exactly once with value ready."}],
    [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Return a supplied readiness value.",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
        }
    ],
    tool_choice="auto",
)
if len(completion.tool_calls) != 1 or completion.tool_calls[0].name != "echo":
    raise SystemExit("Native tool-call preflight did not return the required echo call.")
arguments = json.loads(completion.tool_calls[0].arguments)
if arguments.get("value") != "ready":
    raise SystemExit(f"Native tool-call preflight returned unexpected arguments: {arguments!r}")
print("Native tool-call preflight passed.")
PY

# Measure a changed reasoning profile on training data before its first ablation,
# sharing the allocated server and preserving the existing resumable pilot records.
if [[ "${HOTPOTQA_INITIAL_THROUGHPUT:-0}" == "1" && "${HOTPOTQA_PILOT_ONLY}" == "0" \
    && "${BUDGET_PROFILE}" == "standard" && "${CONDITION}" == "vanilla" ]]; then
    "${PY}" -m examples.hotpotqa.pilot \
        --model "${SOLVER_MODEL}" --api-base "${SOLVER_API_BASE}" \
        --wiki17-dir "${WIKI17_DIR}" --workers "${MAX_WORKERS}" \
        --output-dir "${HOTPOTQA_PILOT_ROOT:?pilot output root required}" \
        --text-limits "${HOTPOTQA_TEXT_LIMITS_JSON:-null}" --stage throughput
fi

if [[ "${HOTPOTQA_PILOT_ONLY}" == "1" ]]; then
    if [[ "${MODEL_PROFILE}" == "deepseek-teacher-qwen-student" ]]; then
        "${PY}" -m examples.hotpotqa.pilot \
            --model "${SOLVER_MODEL}" --api-base "${SOLVER_API_BASE}" \
            --reflection-model "${REFLECTION_MODEL}" --reflection-api-base "${REFLECTION_API_BASE}" \
            --wiki17-dir "${WIKI17_DIR}" --workers "${MAX_WORKERS}" \
            --output-dir "${HOTPOTQA_PILOT_ROOT:?pilot output root required}" \
            --throughput-questions "${HOTPOTQA_THROUGHPUT_QUESTIONS:-12}" \
            --text-limits "${HOTPOTQA_TEXT_LIMITS_JSON:-null}" --stage "${HOTPOTQA_PILOT_STAGE:-preliminary}"
        if [[ -n "${HOTPOTQA_BATCHING_WORKERS:-}" ]]; then
            for calibration_workers in ${HOTPOTQA_BATCHING_WORKERS}; do
                [[ "${calibration_workers}" =~ ^(4|8|12|16|24|32)$ ]] || { echo "ERROR: unsupported calibration workers" >&2; exit 1; }
                "${PY}" -m examples.hotpotqa.pilot \
                    --model "${SOLVER_MODEL}" --api-base "${SOLVER_API_BASE}" \
                    --reflection-model "${REFLECTION_MODEL}" --reflection-api-base "${REFLECTION_API_BASE}" \
                    --wiki17-dir "${WIKI17_DIR}" --workers "${calibration_workers}" \
                    --output-dir "${HOTPOTQA_PILOT_ROOT}/batching-w${calibration_workers}" \
                    --throughput-questions "${HOTPOTQA_THROUGHPUT_QUESTIONS:-48}" \
                    --text-limits "${HOTPOTQA_TEXT_LIMITS_JSON:-null}" --stage throughput
            done
        fi
        exit 0
    fi
    if [[ "${MODEL_PROFILE}" == "qwen3.8-27b" ]]; then
        QWEN_VERIFY_DIR="${SCRATCH_BASE}/.cache/gepa/hotpotqa-tool-verification/${HOTPOTQA_CAMPAIGN_ID}"
        QWEN_VERIFY_MARKER="${QWEN_VERIFY_DIR}/${CAMPAIGN_IDENTITY_SHA256}.ok"
        mkdir -p "${QWEN_VERIFY_DIR}"
        if [[ ! -f "${QWEN_VERIFY_MARKER}" \
            || "$(tr -d '\n' < "${QWEN_VERIFY_MARKER}")" != "${CAMPAIGN_IDENTITY_SHA256}" ]]; then
            "${PY}" -m examples.hotpotqa.verify_serving \
                --model "${SOLVER_MODEL}" --api-base "${SOLVER_API_BASE}" --attempts 4 \
                --attempt-log "${LOG_DIR}/tool-verification-${SLURM_JOB_ID:-local}.jsonl" \
                | tee "${LOG_DIR}/tool-verification-${SLURM_JOB_ID:-local}.log"
            if ! kill -0 "${GEN_PID}" 2>/dev/null || ! generator_reports_expected_model; then
                echo "ERROR: Qwen endpoint failed after tool verification" >&2
                exit 1
            fi
            QWEN_VERIFY_TEMP="$(mktemp "${QWEN_VERIFY_DIR}/.verification.XXXXXX")"
            printf '%s\n' "${CAMPAIGN_IDENTITY_SHA256}" > "${QWEN_VERIFY_TEMP}"
            mv "${QWEN_VERIFY_TEMP}" "${QWEN_VERIFY_MARKER}"
        fi
    fi
    if [[ "${HOTPOTQA_PILOT_STAGE:-all}" == "all" ]]; then
        "${PY}" -m examples.hotpotqa.error_recovery_probe \
            --model "${SOLVER_MODEL}" --api-base "${SOLVER_API_BASE}" \
            --attempt-log "${LOG_DIR}/error-recovery-provider-${MODEL_PROFILE}-${SLURM_JOB_ID:-local}.jsonl" \
            --output "${LOG_DIR}/error-recovery-${MODEL_PROFILE}-${SLURM_JOB_ID:-local}.json"
    fi
    "${PY}" -m examples.hotpotqa.pilot \
        --model "${SOLVER_MODEL}" --api-base "${SOLVER_API_BASE}" \
        --wiki17-dir "${WIKI17_DIR}" --workers "${MAX_WORKERS}" \
        --output-dir "${HOTPOTQA_PILOT_ROOT:?pilot output root required}" \
        --text-limits "${HOTPOTQA_TEXT_LIMITS_JSON:-null}" --stage "${HOTPOTQA_PILOT_STAGE:-all}"
    exit 0
fi

CAMPAIGN_LOCK_DIR="${SCRATCH_BASE}/.cache/gepa/hotpotqa-campaign/${HOTPOTQA_CAMPAIGN_ID}"
CAMPAIGN_LOCK_PATH="${CAMPAIGN_LOCK_DIR}/${MODEL_PROFILE}.sha256"
mkdir -p "${CAMPAIGN_LOCK_DIR}"
DATA_CAMPAIGN_IDENTITY="source=${CAMPAIGN_SOURCE_COMMIT};source_manifest=${CAMPAIGN_SOURCE_MANIFEST_SHA256};python=${HOTPOTQA_PYTHON_VERSION};uv=${HOTPOTQA_UV_VERSION};uv_sha=${HOTPOTQA_UV_SHA256};env_spec=${HOTPOTQA_ENV_SPEC_SHA256};gepa_env=${HOTPOTQA_GEPA_ENV_SHA256};dspy=${HOTPOTQA_DSPY_COMMIT};hotpot=${HOTPOTQA_DATA_REVISION};wiki17=${WIKI17_REVISION};index=${WIKI17_INTEGRITY_SHA256}"
DATA_CAMPAIGN_SHA256="$(printf '%s' "${DATA_CAMPAIGN_IDENTITY}" | sha256sum | cut -d' ' -f1)"
DATA_CAMPAIGN_LOCK_PATH="${CAMPAIGN_LOCK_DIR}/data-and-source.sha256"
DATA_CAMPAIGN_LOCK_TEMP="$(mktemp "${CAMPAIGN_LOCK_DIR}/.data-and-source.XXXXXX")"
printf '%s\n' "${DATA_CAMPAIGN_SHA256}" > "${DATA_CAMPAIGN_LOCK_TEMP}"
if ln "${DATA_CAMPAIGN_LOCK_TEMP}" "${DATA_CAMPAIGN_LOCK_PATH}" 2>/dev/null; then
    echo "==> froze shared HotPotQA data and source: ${DATA_CAMPAIGN_SHA256}"
elif [[ "$(tr -d '\n' < "${DATA_CAMPAIGN_LOCK_PATH}")" != "${DATA_CAMPAIGN_SHA256}" ]]; then
    echo "ERROR: source, dependency lock, HotPotQA data, DSPy, or Wiki-2017 index differs from the existing campaign lock" >&2
    rm -f -- "${DATA_CAMPAIGN_LOCK_TEMP}"
    exit 1
fi
rm -f -- "${DATA_CAMPAIGN_LOCK_TEMP}"
CAMPAIGN_LOCK_TEMP="$(mktemp "${CAMPAIGN_LOCK_DIR}/.${MODEL_PROFILE}.XXXXXX")"
printf '%s\n' "${CAMPAIGN_IDENTITY_SHA256}" > "${CAMPAIGN_LOCK_TEMP}"
if ln "${CAMPAIGN_LOCK_TEMP}" "${CAMPAIGN_LOCK_PATH}" 2>/dev/null; then
    echo "==> froze ${MODEL_PROFILE} campaign runtime: ${CAMPAIGN_IDENTITY_SHA256}"
elif [[ "$(tr -d '\n' < "${CAMPAIGN_LOCK_PATH}")" != "${CAMPAIGN_IDENTITY_SHA256}" ]]; then
    echo "ERROR: ${MODEL_PROFILE} source or quality-relevant runtime differs from the existing campaign lock" >&2
    rm -f -- "${CAMPAIGN_LOCK_TEMP}"
    exit 1
fi
rm -f -- "${CAMPAIGN_LOCK_TEMP}"

echo "==> running GEPA experiment: condition=${CONDITION} budget_profile=${BUDGET_PROFILE} max_metric_calls=${MAX_METRIC_CALLS}"
echo "==> solver=${SOLVER_MODEL} local_api_base=${SOLVER_API_BASE}"
echo "==> reflection=${REFLECTION_MODEL} local_api_base=${REFLECTION_API_BASE}"
echo "==> retrieval=Wiki-2017/BM25 k=7 concurrent_examples=${MAX_WORKERS} root=${WIKI17_DIR}"

"${PY}" -m examples.hotpotqa.main \
    --solver-model "${SOLVER_MODEL}" \
    --reflection-model "${REFLECTION_MODEL}" \
    "${SOLVER_API_ARG[@]}" \
    "${REFLECTION_API_ARG[@]}" \
    --max-metric-calls "${MAX_METRIC_CALLS}" \
    --condition "${CONDITION}" \
    --text-limits "${HOTPOTQA_TEXT_LIMITS_JSON:-null}" \
    --program 2stage \
    --seed-style structured \
    --seed 0 \
    --max-workers "${MAX_WORKERS}" \
    --wiki17-dir "${WIKI17_DIR}" \
    --retrieval-k 7 \
    --reflection-level 2 \
    --edit-tool-set broad \
    --template-family auto \
    --enforce-scientific-contract \
    --tag "${BUDGET_PROFILE}"
