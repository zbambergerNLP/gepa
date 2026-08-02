#!/bin/bash
# Submit the GEPA IFBench experiment to della from your laptop.
#
# Usage:
#   scripts/della/submit_ifbench.sh
#
# First-time setup (installs the ifbench extra + nltk/spacy data on the vis
# node, which has internet):
#   SETUP=1 scripts/della/submit_ifbench.sh
#
# Mini test run (50 train / 50 val / 50 test):
#   TRAIN_LIMIT=50 VAL_LIMIT=50 TEST_LIMIT=50 MAX_METRIC_CALLS=150 TIME=02:00:00 \
#       scripts/della/submit_ifbench.sh
#
# Override defaults via env:
#   MODEL=Qwen3.6-35B-A3B MAX_METRIC_CALLS=1000 CONDITION=action \
#       scripts/della/submit_ifbench.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "ERROR: ${ENV_FILE} not found." >&2
    exit 1
fi

set -a; source "${ENV_FILE}"; set +a

# Tunable knobs (env overrides).
MODEL="${MODEL:-Qwen3.5-9B}"
MAX_METRIC_CALLS="${MAX_METRIC_CALLS:-3593}"
CONDITION="${CONDITION:-all}"
PROGRAM="${PROGRAM:-2stage}"
SEED_STYLE="${SEED_STYLE:-plain}"
ACTIONS="${ACTIONS:-default}"
TAG="${TAG:-}"
TRAIN_LIMIT="${TRAIN_LIMIT:-}"
VAL_LIMIT="${VAL_LIMIT:-}"
TEST_LIMIT="${TEST_LIMIT:-}"
TIME="${TIME:-12:00:00}"
NO_SYNC="${NO_SYNC:-}"
SETUP="${SETUP:-}"

# Step 1: sync code (unless NO_SYNC=1).
if [[ -z "${NO_SYNC}" ]]; then
    echo "==> syncing code"
    "${SCRIPT_DIR}/sync_to_della.sh"
fi

# Step 2 (optional, SETUP=1): install ifbench extra + nltk/spacy data on the
# vis node (has internet; GPU nodes do not).
if [[ -n "${SETUP}" ]]; then
    echo "==> setting up ifbench env on ${REMOTE_VIS_HOST}"
    sshpass -p "${REMOTE_PASSWORD}" ssh -o StrictHostKeyChecking=no \
        "${REMOTE_USER}@${REMOTE_VIS_HOST}" bash -l <<REMOTE_SETUP
set -euo pipefail
cd "${REMOTE_DIR}"
export PATH="\${HOME}/.local/bin:\${PATH}"
export XDG_CACHE_HOME="${SCRATCH_BASE}/.cache"
export UV_CACHE_DIR="${SCRATCH_BASE}/.cache/uv"
export NLTK_DATA="${SCRATCH_BASE}/nltk_data"
mkdir -p "\${UV_CACHE_DIR}" "\${NLTK_DATA}"

echo "==> uv sync --extra dev --extra ifbench"
uv sync --extra dev --extra ifbench

echo "==> downloading nltk data to \${NLTK_DATA}"
.venv/bin/python -m nltk.downloader -d "\${NLTK_DATA}" punkt punkt_tab stopwords

echo "==> downloading spacy en_core_web_sm"
.venv/bin/python -m spacy download en_core_web_sm

echo "==> verifying ifbench setup"
NLTK_DATA="\${NLTK_DATA}" .venv/bin/python -c "
from examples.ifbench.utils import load_ifbench_dataset
t, v, te = load_ifbench_dataset()
print(f'IFBench data: {len(t)} train / {len(v)} val / {len(te)} test')
import spacy; spacy.load('en_core_web_sm'); print('spacy OK')
"
REMOTE_SETUP
fi

# Step 3: submit sbatch on della login node.
echo "==> submitting job: model=${MODEL} max_metric_calls=${MAX_METRIC_CALLS} condition=${CONDITION} program=${PROGRAM}"
echo "==> limits: train=${TRAIN_LIMIT:-full} val=${VAL_LIMIT:-full} test=${TEST_LIMIT:-full}"

sshpass -p "${REMOTE_PASSWORD}" ssh -o StrictHostKeyChecking=no \
    "${REMOTE_USER}@${REMOTE_HOST}" bash -l <<REMOTE_SCRIPT
set -euo pipefail
cd "${REMOTE_DIR}"

sbatch \
    --partition="${GPU_PARTITION}" \
    --time="${TIME}" \
    --export=ALL,MODEL=${MODEL},MAX_METRIC_CALLS=${MAX_METRIC_CALLS},CONDITION=${CONDITION},PROGRAM=${PROGRAM},SEED_STYLE=${SEED_STYLE},ACTIONS=${ACTIONS},TAG=${TAG},TRAIN_LIMIT=${TRAIN_LIMIT},VAL_LIMIT=${VAL_LIMIT},TEST_LIMIT=${TEST_LIMIT},MODEL_STORAGE=${MODEL_STORAGE},SCRATCH_BASE=${SCRATCH_BASE} \
    examples/ifbench/run_ifbench.sbatch

echo "==> job submitted. Check status with: squeue -u ${REMOTE_USER}"
REMOTE_SCRIPT
