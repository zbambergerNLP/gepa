#!/bin/bash
# Regenerate the hash-locked serving environment for one HotPotQA model arm.
#
# Each arm has its own requirements file and lock under examples/hotpotqa/serving/
# (qwen3.8-27b: requirements.in; deepseek-v4.1-flash:
# requirements-deepseek-v4.1-flash.in). The lock resolves for Linux x86_64 / Python
# 3.12 (Della's GPU nodes) from any machine, with every transitive package hashed so
# build_env.sh can install it with --require-hashes on the visualization node.
#
# Usage:
#   scripts/della/lock_serving_env.sh                                   # Qwen arm
#   MODEL_PROFILE=deepseek-v4.1-flash scripts/della/lock_serving_env.sh # DeepSeek arm
#   scripts/della/lock_serving_env.sh freeze.txt                        # constrain to a freeze
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SERVING_DIR="${REPO_ROOT}/examples/hotpotqa/serving"
MODEL_PROFILE="${MODEL_PROFILE:-qwen3.8-27b}"
CONSTRAINTS="${1:-}"

INDEX_ARGS=()
case "${MODEL_PROFILE}" in
    qwen3.8-27b)
        REQUIREMENTS_IN="${SERVING_DIR}/requirements.in"
        LOCK_FILE="${SERVING_DIR}/requirements-x86_64-linux-py312.txt"
        ;;
    deepseek-v4.1-flash)
        REQUIREMENTS_IN="${SERVING_DIR}/requirements-deepseek-v4.1-flash.in"
        LOCK_FILE="${SERVING_DIR}/requirements-deepseek-v4.1-flash-x86_64-linux-py312.txt"
        # flashinfer's prebuilt kernel wheels are published only on flashinfer.ai, as
        # package indexes whose links carry each wheel's sha256. --emit-index-url writes
        # the indexes into the lock so build_env.sh's `uv pip sync` finds the same wheels.
        INDEX_ARGS=(
            --extra-index-url https://flashinfer.ai/whl/
            --extra-index-url https://flashinfer.ai/whl/cu130/
            --emit-index-url
        )
        ;;
    *)
        echo "ERROR: MODEL_PROFILE must be qwen3.8-27b or deepseek-v4.1-flash" >&2
        exit 1
        ;;
esac

CONSTRAINT_ARGS=()
if [[ -n "${CONSTRAINTS}" ]]; then
    CONSTRAINT_ARGS=(--constraint "${CONSTRAINTS}")
fi

uv pip compile "${REQUIREMENTS_IN}" \
    --output-file "${LOCK_FILE}" \
    --python-version 3.12 \
    --python-platform x86_64-manylinux_2_28 \
    --generate-hashes \
    --no-header \
    --annotation-style line \
    ${INDEX_ARGS[@]+"${INDEX_ARGS[@]}"} \
    ${CONSTRAINT_ARGS[@]+"${CONSTRAINT_ARGS[@]}"}

echo "==> wrote ${LOCK_FILE} (sha256 $(sha256sum "${LOCK_FILE}" | cut -d' ' -f1))"
