#!/bin/bash
# Check the external coordinator before loading the paired GPU servers.
set -euo pipefail
if [[ $# != 2 || -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Usage inside sbatch: $0 <prepared-export-file> <jev-mailbox-directory>" >&2
    exit 1
fi
while IFS= read -r -d '' entry; do
    case "${entry}" in HOME=*) continue ;; esac
    export "${entry}"
done < "$1"
unset entry
if [[ "${HOTPOTQA_PRODUCTION_LAUNCH:-}" != 1 || "${HOTPOTQA_PILOT_ONLY:-}" != 0 \
    || "${MODEL_PROFILE:-}" != deepseek-teacher-qwen-student \
    || "${CONDITION:-}" != react_v2 || "${HOTPOTQA_CONTROLLER_SELECTION:-}" != jev ]]; then
    echo "ERROR: resident full run requires a prepared Jev FOREST cell" >&2
    exit 1
fi
case "${BUDGET_PROFILE:-}:${MAX_METRIC_CALLS:-}" in
    standard:6871|expanded:13742) ;;
    *) echo "ERROR: resident budget must match the standard or expanded profile" >&2; exit 1 ;;
esac
export GEPA_JEV_HANDOFF_DIR="$2"
unset TYPESAFE_API_KEY
export SLURM_SUBMIT_DIR="${SCRATCH_BASE:?}/sources/${HOTPOTQA_SOURCE_COMMIT:?}"
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${PWD}/src:${PWD}"
export DSPY_CACHEDIR="${SCRATCH_BASE}/.cache/dspy/paired-bootstrap"
"${GEPA_UV_BIN:?}" run --no-project --python "${GEPA_VENV_DIR:?}/bin/python" \
    python -m examples.hotpotqa.jev_mailbox check-ready "$GEPA_JEV_HANDOFF_DIR" \
    --job "$SLURM_JOB_ID" --source "$HOTPOTQA_SOURCE_COMMIT"
exec bash examples/hotpotqa/run_hotpotqa.sbatch
