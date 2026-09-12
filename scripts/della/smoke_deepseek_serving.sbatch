#!/bin/bash
# Batch wrapper for scripts/della/verify_deepseek_serving.sh: serve DeepSeek-V4.1-Flash
# with the campaign's exact vLLM invocation on one eight-H200 node, record one full
# exchange (request, server-rendered prompt, reasoning, content), and run the ReAct V2
# tool probes. Not part of any campaign chain.
#
# Submit it with scripts/della/submit_deepseek_smoke.sh, which syncs the checkout first
# and fetches the results afterwards.

#SBATCH --job-name=gepa-hp-deepseek-v4.1-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ailab
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=768G
#SBATCH --time=03:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the synced checkout}"
exec scripts/della/verify_deepseek_serving.sh
