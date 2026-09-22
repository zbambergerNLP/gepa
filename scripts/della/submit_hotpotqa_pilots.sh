#!/bin/bash
# Submit independent model pilots using the pinned serving and recovery path.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ "${1:-}" == "--dry-run" ]]; then
    for profile in qwen3.8-27b deepseek-v4.1-flash; do
        printf 'HOTPOTQA_JOB_KIND=pilot MODEL_PROFILE=%s HOTPOTQA_CAMPAIGN_ID=<pilot-id> bash %q\n' "$profile" "${SCRIPT_DIR}/submit_hotpotqa.sh"
    done
    echo "Each arm runs the 3-question smoke, four optimizer cycles, 12-question throughput check, and 150-question training calibration. DeepSeek first runs its serving canary. No validation or test execution."
    exit 0
fi
if [[ $# != 0 ]]; then
    echo "Usage: HOTPOTQA_CAMPAIGN_ID=<pilot-id> $0 [--dry-run]" >&2
    exit 1
fi
: "${HOTPOTQA_CAMPAIGN_ID:?set a fresh pilot campaign ID}"
for profile in qwen3.8-27b deepseek-v4.1-flash; do
    HOTPOTQA_JOB_KIND=pilot MODEL_PROFILE="$profile" bash "${SCRIPT_DIR}/submit_hotpotqa.sh"
done
