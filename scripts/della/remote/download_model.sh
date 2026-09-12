#!/bin/bash
# Download one pinned checkpoint into MODEL_STORAGE and verify every byte against the
# Hugging Face revision; the launchers refuse a checkpoint without the resulting
# .gepa-model-integrity.json. Run on della-vis1 from the synced checkout, after
# scripts/della/remote/setup_env.sh, with SCRATCH_BASE and MODEL_STORAGE set:
#   bash scripts/della/remote/download_model.sh qwen3.8-27b|deepseek-v4.1-flash
# DeepSeek-V4.1-Flash is 48 shards / 510 GB; run it detached (build_env.sh does).
set -euo pipefail
export PYTHONPATH="${PWD}/src:${PWD}"

MODEL="${1:?usage: download_model.sh qwen3.8-27b|deepseek-v4.1-flash}"
: "${SCRATCH_BASE:?}" "${MODEL_STORAGE:?}"
case "${MODEL}" in
    qwen3.8-27b) MODEL_DIR="${MODEL_STORAGE}/Qwen3.8-27B" ;;
    deepseek-v4.1-flash) MODEL_DIR="${MODEL_STORAGE}/DeepSeek-V4.1-Flash" ;;
    *) echo "ERROR: unknown model ${MODEL}" >&2; exit 1 ;;
esac
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"

mkdir -p "${MODEL_DIR}"
# Jobs hold this lock shared while they serve the checkpoint.
exec {MODEL_LOCK_FD}<"${MODEL_DIR}"
flock -n "${MODEL_LOCK_FD}" || { echo "ERROR: ${MODEL_DIR} is being downloaded or served" >&2; exit 1; }

echo "==> ${MODEL} into ${MODEL_DIR} ($(date))"
.venv/bin/python -m examples.common.model_snapshot prepare --model-profile "${MODEL}" --root "${MODEL_DIR}" > /dev/null
.venv/bin/python -m examples.common.model_snapshot verify --model-profile "${MODEL}" --root "${MODEL_DIR}" > /dev/null
echo "==> ${MODEL} verified ($(date))"
