#!/bin/bash
# Regenerate the hash-locked serving environment for the Qwen arm.
#
# Resolves examples/hotpotqa/serving/requirements.in for Linux x86_64 / Python 3.12
# (Della's GPU nodes) from any machine, with every transitive package hashed so
# build_env.sh can install it with --require-hashes on the visualization node.
#
# Usage:
#   scripts/della/lock_serving_env.sh            # resolve freely
#   scripts/della/lock_serving_env.sh freeze.txt # constrain to an existing freeze
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SERVING_DIR="${REPO_ROOT}/examples/hotpotqa/serving"
REQUIREMENTS_IN="${SERVING_DIR}/requirements.in"
LOCK_FILE="${SERVING_DIR}/requirements-x86_64-linux-py312.txt"
CONSTRAINTS="${1:-}"

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
    "${CONSTRAINT_ARGS[@]}"

echo "==> wrote ${LOCK_FILE} (sha256 $(sha256sum "${LOCK_FILE}" | cut -d' ' -f1))"
