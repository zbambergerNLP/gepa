#!/bin/bash
# Download and verify the HotPotQA datasets on della-vis1: the frozen Wiki-2017 BM25
# index and the pinned HotpotQA fullwiki split (150/300/300). Run from the synced
# checkout with SCRATCH_BASE set, after scripts/della/remote/setup_env.sh.
set -euo pipefail
export PYTHONPATH="${PWD}/src:${PWD}"

: "${SCRATCH_BASE:?}"
WIKI17_DIR="${WIKI17_DIR:-${SCRATCH_BASE}/.cache/gepa/wiki17}"
export XDG_CACHE_HOME="${SCRATCH_BASE}/.cache" HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export DSPY_CACHEDIR="${SCRATCH_BASE}/.cache/dspy"

# Jobs hold this lock shared while they run; never rebuild under them.
exec {ARTIFACT_LOCK_FD}>"${SCRATCH_BASE}/.cache/gepa/hotpotqa-artifacts.lock"
flock -n "${ARTIFACT_LOCK_FD}" || { echo "ERROR: HotPotQA artifacts are in use by a running job" >&2; exit 1; }

echo "==> Wiki-2017 BM25 index at ${WIKI17_DIR}"
.venv/bin/python -m examples.common.wiki17_bm25 prepare --root "${WIKI17_DIR}"
.venv/bin/python -m examples.common.wiki17_bm25 verify --deep --root "${WIKI17_DIR}"
echo "==> HotpotQA fullwiki split"
.venv/bin/python -c 'from examples.hotpotqa.utils import load_hotpotqa_dataset as load
print("%d train / %d val / %d test" % tuple(map(len, load(seed=0))))'
