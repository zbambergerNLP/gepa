#!/bin/bash
# Submit the DeepSeek-V4.1-Flash serving smoke test from the laptop, or fetch its results.
#
# Usage:
#   scripts/della/submit_deepseek_smoke.sh submit          # sync the checkout, submit, print the job id
#   scripts/della/submit_deepseek_smoke.sh fetch <job-id>  # copy the run directory and Slurm log here
#
# Requires scripts/della/build_env.sh to have built the DeepSeek serving environment and
# staged the checkpoint. Results land in outputs/deepseek-smoke/<job-id>/: transcript.md
# (request, server-rendered prompt, reasoning, and content), transcript.json,
# verify_report.txt, serving-packages.txt, gpus.txt, vllm.log, and the Slurm log.
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

SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=yes)
VERIFY_LOG_DIR="${SCRATCH_BASE}/logs/hotpotqa/verify"

case "${1:-}" in
    submit)
        echo "==> syncing the checkout to ${REMOTE_DIR}"
        "${SCRIPT_DIR}/sync_to_della.sh"
        printf -v SUBMIT_COMMAND \
            'mkdir -p %q && cd %q && env SCRATCH_BASE=%q MODEL_STORAGE=%q GEPA_VENV_DIR=%q SERVING_VENV_DIR=%q sbatch --parsable --output=%q scripts/della/smoke_deepseek_serving.sbatch' \
            "${VERIFY_LOG_DIR}" "${REMOTE_DIR}" "${SCRATCH_BASE}" \
            "${MODEL_STORAGE:-/projects/BSTEWART/model_storage}" \
            "${REMOTE_DIR%/}/.venv" "${REMOTE_DIR%/}/.serving-venv-deepseek-v4.1-flash" \
            "${VERIFY_LOG_DIR}/smoke-%j.out"
        JOB_ID="$(
            ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
                "${SUBMIT_COMMAND}"
        )"
        JOB_ID="${JOB_ID%%;*}"
        if [[ ! "${JOB_ID}" =~ ^[0-9]+$ ]]; then
            echo "ERROR: sbatch returned an invalid job id: ${JOB_ID}" >&2
            exit 1
        fi
        echo "==> submitted DeepSeek-V4.1-Flash smoke test: job ${JOB_ID}"
        echo "    fetch when it finishes: scripts/della/submit_deepseek_smoke.sh fetch ${JOB_ID}"
        ;;
    fetch)
        JOB_ID="${2:?usage: $0 fetch <job-id>}"
        if [[ ! "${JOB_ID}" =~ ^[0-9]+$ ]]; then
            echo "ERROR: job ID must contain only digits" >&2
            exit 1
        fi
        DESTINATION="${REPO_ROOT}/outputs/deepseek-smoke/${JOB_ID}"
        mkdir -p "${DESTINATION}"
        rsync -a -e "ssh ${SSH_OPTS[*]}" \
            "${REMOTE_USER}@${REMOTE_HOST}:${VERIFY_LOG_DIR}/${JOB_ID}/" "${DESTINATION}/"
        rsync -a -e "ssh ${SSH_OPTS[*]}" \
            "${REMOTE_USER}@${REMOTE_HOST}:${VERIFY_LOG_DIR}/smoke-${JOB_ID}.out" "${DESTINATION}/"
        echo "==> fetched into ${DESTINATION}"
        ls -la "${DESTINATION}"
        ;;
    *)
        echo "usage: $0 submit | fetch <job-id>" >&2
        exit 2
        ;;
esac
