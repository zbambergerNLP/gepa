#!/bin/bash
# One-off, manual check that the frozen vLLM serving environment serves
# DeepSeek-V4.1-Flash exactly the way examples/hotpotqa/run_hotpotqa.sbatch
# does, and that the served model handles the HotPotQA ReAct V2 tool protocol.
#
# Run it yourself, once, with four H200s allocated on one node from the synced checkout
# (REMOTE_DIR) after scripts/della/build_env.sh has built both venvs and staged the
# checkpoint:
#
#   salloc --partition=ailab --nodes=1 --gres=gpu:4 --cpus-per-task=32 --mem=512G --time=02:00:00
#   cd /scratch/gpfs/BSTEWART/$USER/gepa     # REMOTE_DIR
#   scripts/della/verify_deepseek_serving.sh
#
# It is deliberately not wired into scripts/della/submit_hotpotqa.sh, is not a
# dependency of any campaign job, and writes no campaign marker. Its outputs
# are the PASS/FAIL report on stdout, a non-zero exit status on failure, and one
# results directory ($SCRATCH_BASE/logs/hotpotqa/verify/<job id>) holding the vLLM
# log, the installed serving packages, the GPU inventory, the tool-probe report,
# and transcript.md/.json from examples/hotpotqa/smoke_serving.py: one simple prompt
# with the campaign's request settings, the prompt text the server rendered from
# it, and the full response including reasoning. scripts/della/submit_deepseek_smoke.sh
# runs it as a batch job and fetches that directory.
#
# Checks, in order: the frozen vLLM registers DeepseekV41ForCausalLM; vLLM starts
# with the campaign's TP4/EP4 single-sequence invocation and reports the served
# name; the smoke exchange returns reasoning and content; then
# examples/hotpotqa/verify_serving.py exercises an ordinary completion,
# a native tool call plus its tool-result continuation, and one ReAct V2 proposal
# per broad edit tool (DELETE_TEXT, INSERT_TEXT, MOVE_TEXT, REPLACE_TEXT).
#
# Environment overrides: MODEL_STORAGE, SERVING_VENV_DIR, GEPA_VENV_DIR,
# SCRATCH_BASE, VERIFY_PORT, HEALTH_TIMEOUT, VERIFY_ATTEMPTS (default 4, one per
# tool; raise it for a longer soak), VERIFY_TIMEOUT (per-request seconds).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MODEL_STORAGE="${MODEL_STORAGE:-/projects/BSTEWART/model_storage}"
MODEL="DeepSeek-V4.1-Flash"
SOLVER_MODEL_PATH="${MODEL_STORAGE}/${MODEL}"
SOLVER_SERVED_NAME="deepseek-ai/DeepSeek-V4.1-Flash"
SOLVER_MODEL="hosted_vllm/deepseek-ai/DeepSeek-V4.1-Flash"
SERVING_VENV_DIR="${SERVING_VENV_DIR:-${REPO_ROOT}/.serving-venv-deepseek-v4.1-flash}"
GEPA_VENV_DIR="${GEPA_VENV_DIR:-${REPO_ROOT}/.venv}"
SCRATCH_BASE="${SCRATCH_BASE:-/scratch/gpfs/BSTEWART/${USER}/gepa}"
VERIFY_PORT="${VERIFY_PORT:-}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-3600}"
VERIFY_ATTEMPTS="${VERIFY_ATTEMPTS:-4}"
VERIFY_TIMEOUT="${VERIFY_TIMEOUT:-3600}"
# Same serving values as run_hotpotqa.sbatch.
GEN_GMU=0.92
GEN_MAX_LEN=262144
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"

VLLM_PY="${SERVING_VENV_DIR}/bin/python"
VLLM_BIN="${SERVING_VENV_DIR}/bin/vllm"
PY="${GEPA_VENV_DIR}/bin/python"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"
mkdir -p "${SCRATCH_BASE}/.cache/gepa"
exec {ARTIFACT_LOCK_FD}>"${SCRATCH_BASE}/.cache/gepa/hotpotqa-artifacts.lock"
flock -s -n "${ARTIFACT_LOCK_FD}" || { echo "ERROR: artifacts are being rebuilt" >&2; exit 1; }
for required in "${VLLM_PY}" "${VLLM_BIN}" "${PY}"; do
    if [[ ! -x "${required}" ]]; then
        echo "ERROR: missing ${required}; run scripts/della/build_env.sh first" >&2
        exit 1
    fi
done
if [[ ! -f "${SOLVER_MODEL_PATH}/config.json" ]]; then
    echo "ERROR: ${SOLVER_MODEL_PATH} is not staged; run scripts/della/build_env.sh first" >&2
    exit 1
fi
# build_env.sh writes the integrity manifest only after every shard is downloaded and
# hashed, so its absence means the checkpoint may still be incomplete.
if [[ ! -s "${SOLVER_MODEL_PATH}/.gepa-model-integrity.json" ]]; then
    echo "ERROR: ${SOLVER_MODEL_PATH} has no .gepa-model-integrity.json; wait for build_env.sh to finish" >&2
    exit 1
fi
exec {MODEL_LOCK_FD}<"${SOLVER_MODEL_PATH}"
flock -s -n "${MODEL_LOCK_FD}" || { echo "ERROR: checkpoint is being prepared" >&2; exit 1; }
"${PY}" -m examples.common.model_snapshot verify \
    --model-profile deepseek-v4.1-flash --root "${SOLVER_MODEL_PATH}" >/dev/null
SERVING_LOCK_SHA256="$(sha256sum examples/hotpotqa/serving/requirements-deepseek-v4.1-flash-x86_64-linux-py312.txt | cut -d' ' -f1)"
if [[ "$(cat "${SERVING_VENV_DIR}/.gepa-serving-lock.sha256")" != "${SERVING_LOCK_SHA256}" ]]; then
    echo "ERROR: serving environment was built from a different lock" >&2
    exit 1
fi
"${VLLM_PY}" -m examples.common.python_environment verify \
    --path "${SCRATCH_BASE}/.cache/gepa/serving-environments/${SERVING_LOCK_SHA256}.json" >/dev/null
if ! command -v nvidia-smi >/dev/null 2>&1 || ! "${GEPA_UV_BIN:-uv}" run --no-project --python "${VLLM_PY}" python -c \
    'import torch; raise SystemExit(0 if torch.cuda.device_count() == 4 and all("H200" in torch.cuda.get_device_name(i) for i in range(4)) else 1)'; then
    echo "ERROR: run this with four CUDA-visible H200 GPUs (salloc --partition=ailab --gres=gpu:4)" >&2
    exit 1
fi

# --- Same cache, offline, and logging environment as the sbatch -------------
export XDG_CACHE_HOME="${SCRATCH_BASE}/.cache"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export DSPY_CACHEDIR="${SCRATCH_BASE}/.cache/dspy"
export VLLM_CACHE_ROOT="${SCRATCH_BASE}/.cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_BASE}/.cache/torchinductor"
export TRITON_CACHE_DIR="${SCRATCH_BASE}/.cache/triton"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export JAX_PLATFORMS=cpu
export VLLM_LOGGING_LEVEL=INFO
export VLLM_USE_FLASHINFER_SAMPLER=0
export FLASHINFER_WORKSPACE_BASE="${SCRATCH_BASE}"
export FLASHINFER_NO_DOWNLOAD=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
LOG_DIR="${SCRATCH_BASE}/logs/hotpotqa/verify"
# Everything this run produces lands in one directory named after the Slurm job.
RUN_DIR="${LOG_DIR}/${SLURM_JOB_ID:-$(date +%Y%m%dT%H%M%S)-$$}"
mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}" "${VLLM_CACHE_ROOT}" \
    "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${DSPY_CACHEDIR}" "${RUN_DIR}"
GEN_LOG="${RUN_DIR}/vllm.log"

export PATH="${SERVING_VENV_DIR}/bin:${PATH}"
HOTPOTQA_VLLM_VERSION="$("${VLLM_PY}" -c 'from importlib.metadata import version; print(version("vllm"))')"
HOTPOTQA_CUDA_VERSION="$("${VLLM_PY}" -c 'import torch; print(torch.version.cuda)')"
HOTPOTQA_CUDA_MODULE="cudatoolkit/${HOTPOTQA_CUDA_VERSION}"
source /usr/share/Modules/init/bash
if ! module load "${HOTPOTQA_CUDA_MODULE}" || ! module is-loaded "${HOTPOTQA_CUDA_MODULE}"; then
    echo "ERROR: exact CUDA module ${HOTPOTQA_CUDA_MODULE} is unavailable" >&2
    exit 1
fi
# Same header/library precedence as run_hotpotqa.sbatch: FlashInfer's first-load JIT
# builds must find cuBLAS headers even on nodes whose local CUDA install lacks them.
SERVING_CUDA_ROOT="$("${VLLM_PY}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/nvidia/cu13"
if [[ -d "${SERVING_CUDA_ROOT}/include" ]]; then
    export CPATH="${SERVING_CUDA_ROOT}/include${CPATH:+:${CPATH}}"
    export LIBRARY_PATH="${SERVING_CUDA_ROOT}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
fi
echo "==> vLLM ${HOTPOTQA_VLLM_VERSION} from ${SERVING_VENV_DIR} (CUDA ${HOTPOTQA_CUDA_VERSION}); results in ${RUN_DIR}"
"${VLLM_PY}" -c 'import importlib.metadata as m; print("\n".join(sorted(f"{d.metadata[\"Name\"]}=={d.version}" for d in m.distributions())))' \
    > "${RUN_DIR}/serving-packages.txt"
nvidia-smi > "${RUN_DIR}/gpus.txt"

echo "==> checking that the frozen vLLM registers the checkpoint architecture"
"${VLLM_PY}" - "${SOLVER_MODEL_PATH}" "${HOTPOTQA_VLLM_VERSION}" <<'PY'
import json
import sys
from pathlib import Path

from vllm import ModelRegistry

config = json.loads((Path(sys.argv[1]) / "config.json").read_text())
architectures = config.get("architectures") or []
supported = set(ModelRegistry.get_supported_archs())
unsupported = [architecture for architecture in architectures if architecture not in supported]
if not architectures or unsupported:
    raise SystemExit(
        f"vLLM {sys.argv[2]} does not register {unsupported or architectures!r}; "
        "the pinned serving environment cannot serve this checkpoint."
    )
print(f"vLLM {sys.argv[2]} registers {architectures}")
PY

GEN_PORT="$("${PY}" - "${VERIFY_PORT}" <<'PY'
import socket
import sys

requested_port = int(sys.argv[1]) if sys.argv[1] else 0
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
    listener.bind(("127.0.0.1", requested_port))
    print(listener.getsockname()[1])
PY
)"

# --- The campaign's DeepSeek invocation, verbatim ----------------------------
echo "==> serving ${MODEL} with vLLM TP4/EP4 on :${GEN_PORT}; log: ${GEN_LOG}"
env \
    -u OMP_NUM_THREADS \
    -u MKL_NUM_THREADS \
    -u OPENBLAS_NUM_THREADS \
    -u NUMEXPR_NUM_THREADS \
    -u TOKENIZERS_PARALLELISM \
    "${VLLM_BIN}" serve "${SOLVER_MODEL_PATH}" \
    --served-model-name "${SOLVER_SERVED_NAME}" \
    --enable-auto-tool-choice \
    --host 127.0.0.1 \
    --port "${GEN_PORT}" \
    --gpu-memory-utilization "${GEN_GMU}" \
    --max-model-len "${GEN_MAX_LEN}" \
    --max-num-seqs 1 \
    --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}" \
    --dtype bfloat16 \
    --seed 0 \
    --no-enable-prefix-caching \
    --language-model-only \
    --tokenizer-mode deepseek_v41 \
    --reasoning-parser deepseek_v41 \
    --tool-call-parser deepseek_v41 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --data-parallel-size 1 \
    --api-server-count 1 \
    --kv-cache-dtype fp8 \
    --engram-config '{"cpu_offload":true}' \
    > "${GEN_LOG}" 2>&1 &
GEN_PID=$!

cleanup() {
    echo "==> tearing down the verification vLLM endpoint"
    kill "${GEN_PID}" 2>/dev/null || true
    wait "${GEN_PID}" 2>/dev/null || true
}
trap cleanup EXIT

generator_reports_expected_model() {
    curl -sf "http://127.0.0.1:${GEN_PORT}/v1/models" \
        | "${PY}" -c 'import json, sys; payload = json.load(sys.stdin); ids = [item.get("id") for item in payload.get("data", [])]; raise SystemExit(0 if ids == [sys.argv[1]] else 1)' "${SOLVER_SERVED_NAME}" \
            2>/dev/null
}

deadline=$((SECONDS + HEALTH_TIMEOUT))
while true; do
    if ! kill -0 "${GEN_PID}" 2>/dev/null; then
        echo "FAIL: vLLM exited before becoming healthy; see ${GEN_LOG}" >&2
        exit 1
    fi
    if generator_reports_expected_model && kill -0 "${GEN_PID}" 2>/dev/null; then
        break
    fi
    if (( SECONDS >= deadline )); then
        echo "FAIL: vLLM did not report ${SOLVER_SERVED_NAME} within ${HEALTH_TIMEOUT}s; see ${GEN_LOG}" >&2
        exit 1
    fi
    sleep 5
done
echo "==> vLLM endpoint ready on :${GEN_PORT} after $((SECONDS))s"

export OPENAI_API_KEY="EMPTY"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"
echo "==> recording one full exchange (request, rendered prompt, reasoning, content)"
SMOKE_STATUS=0
"${PY}" -m examples.hotpotqa.smoke_serving \
    --model "${SOLVER_MODEL}" \
    --served-name "${SOLVER_SERVED_NAME}" \
    --api-base "http://127.0.0.1:${GEN_PORT}/v1" \
    --output-dir "${RUN_DIR}" || SMOKE_STATUS=$?

echo "==> verifying ordinary completion, tool-result continuation, and the four ReAct V2 edit tools"
set +e
"${PY}" -m examples.hotpotqa.verify_serving \
    --model "${SOLVER_MODEL}" \
    --api-base "http://127.0.0.1:${GEN_PORT}/v1" \
    --attempts "${VERIFY_ATTEMPTS}" \
    --timeout "${VERIFY_TIMEOUT}" \
    --attempt-log "${RUN_DIR}/provider-attempts.jsonl" 2>&1 | tee "${RUN_DIR}/verify_report.txt"
VERIFY_STATUS="${PIPESTATUS[0]}"
set -e
if [[ "${SMOKE_STATUS}" == "0" && "${VERIFY_STATUS}" == "0" ]]; then
    echo "==> PASS: ${SOLVER_SERVED_NAME} served by vLLM ${HOTPOTQA_VLLM_VERSION} passed every check; results: ${RUN_DIR}"
else
    echo "==> FAIL: smoke exchange exit ${SMOKE_STATUS}, tool probes exit ${VERIFY_STATUS}; results: ${RUN_DIR}" >&2
    exit 1
fi
