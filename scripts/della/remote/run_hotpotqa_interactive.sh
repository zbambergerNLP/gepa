#!/bin/bash
# Reuse the verified launch environment inside an existing salloc/srun step.
set -euo pipefail
if [[ $# != 2 || -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Usage inside salloc: $0 <prepared-export-file> <all|preliminary|smoke|optimizer|throughput|full|generalization|diversity-quality>" >&2
    exit 1
fi
EXPORT_FILE="$1"
PILOT_STAGE="$2"
case "${PILOT_STAGE}" in
    all|preliminary|smoke|optimizer|throughput|full|generalization|diversity-quality) ;;
    *) echo "ERROR: unsupported pilot stage" >&2; exit 1 ;;
esac
while IFS= read -r -d '' entry; do
    case "${entry}" in HOME=*) continue ;; esac
    export "${entry}"
done < "${EXPORT_FILE}"
unset entry
if [[ "${HOTPOTQA_PILOT_ONLY:-}" != 1 ]]; then
    echo "ERROR: interactive qualification requires a prepared pilot environment" >&2
    exit 1
fi
export HOTPOTQA_PILOT_STAGE="${PILOT_STAGE}"
export SLURM_SUBMIT_DIR="${SCRATCH_BASE}/sources/${HOTPOTQA_SOURCE_COMMIT}"
cd "${SLURM_SUBMIT_DIR}"
case "${PILOT_STAGE}" in
    jev-quality|diversity-quality)
        module load proxy/default
        # The external API needs the proxy; local GPU servers must bypass it.
        export NO_PROXY="localhost,127.0.0.1,::1,${NO_PROXY:-},${no_proxy:-}"
        export no_proxy="${NO_PROXY}"
        export PYTHONPATH="${PWD}/src:${PWD}"
        "${GEPA_UV_BIN:?pinned uv required}" run --no-project --python "${GEPA_VENV_DIR:?}/bin/python" \
            python -m examples.hotpotqa.typesafe_preflight \
            --output "${HOTPOTQA_PILOT_ROOT:?}/network-preflight-${SLURM_JOB_ID}.json"
        ;;
esac
exec bash examples/hotpotqa/run_hotpotqa.sbatch
