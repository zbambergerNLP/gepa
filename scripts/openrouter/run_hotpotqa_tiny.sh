#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

mode="${1:---dry-run}"
if [[ "$mode" != "--dry-run" && "$mode" != "--execute" ]]; then
    echo "Usage: $0 [--dry-run|--execute]" >&2
    exit 2
fi
if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [--dry-run|--execute]" >&2
    exit 2
fi

cache_base="${XDG_CACHE_HOME:-${HOME}/.cache}"
technical_mini_index_dir="${HOTPOTQA_TECHNICAL_MINI_INDEX_DIR:-${cache_base}/gepa/hotpotqa-technical-mini}"
configured_smoke_tag="${SMOKE_TAG:-}"
smoke_tag="${configured_smoke_tag:-openrouter-tiny-$(date -u +%Y%m%dT%H%M%SZ)}"
smoke_key_limit_usd="${OPENROUTER_SMOKE_KEY_LIMIT_USD:-25}"
smoke_start_arm="${OPENROUTER_SMOKE_START_ARM:-1}"

arm_names=(
    deepseek-vanilla-no-merge
    deepseek-random-stateless-no-merge
    deepseek-verbalized-stateless-no-merge
    deepseek-react-v2-no-merge
    qwen-vanilla-no-merge
    qwen-random-stateless-no-merge
    qwen-verbalized-stateless-no-merge
    qwen-react-v2-no-merge
    deepseek-vanilla-merge
    deepseek-random-stateless-merge
    deepseek-verbalized-stateless-merge
    deepseek-react-v2-merge
    qwen-vanilla-merge
    qwen-random-stateless-merge
    qwen-verbalized-stateless-merge
    qwen-react-v2-merge
)
models=(
    deepseek/deepseek-v4-flash
    deepseek/deepseek-v4-flash
    deepseek/deepseek-v4-flash
    deepseek/deepseek-v4-flash
    hosted_vllm/Qwen/Qwen3.8-27B
    hosted_vllm/Qwen/Qwen3.8-27B
    hosted_vllm/Qwen/Qwen3.8-27B
    hosted_vllm/Qwen/Qwen3.8-27B
    deepseek/deepseek-v4-flash
    deepseek/deepseek-v4-flash
    deepseek/deepseek-v4-flash
    deepseek/deepseek-v4-flash
    hosted_vllm/Qwen/Qwen3.8-27B
    hosted_vllm/Qwen/Qwen3.8-27B
    hosted_vllm/Qwen/Qwen3.8-27B
    hosted_vllm/Qwen/Qwen3.8-27B
)
conditions=(vanilla random action react_v2 vanilla random action react_v2 vanilla random action react_v2 vanilla random action react_v2)
merge_flags=(0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1)
budgets=(16 16 16 16 16 16 16 16 32 32 32 32 32 32 32 32)
arm_count="${#arm_names[@]}"

if ! [[ "$smoke_start_arm" =~ ^[0-9]+$ ]] || (( smoke_start_arm < 1 || smoke_start_arm > arm_count )); then
    echo "OPENROUTER_SMOKE_START_ARM must be an integer from 1 through ${arm_count}." >&2
    exit 2
fi
if [[ "$smoke_start_arm" != "1" && -z "$configured_smoke_tag" ]]; then
    echo "SMOKE_TAG is required when OPENROUTER_SMOKE_START_ARM is greater than 1." >&2
    exit 2
fi

selected_indexes=()
for ((index = smoke_start_arm - 1; index < ${#arm_names[@]}; index++)); do
    selected_indexes+=("$index")
done

echo "HotPotQA OpenRouter technical-smoke matrix"
echo "  fullwiki split: 6 train / 5 validation / 2 test"
echo "  retrieval: NON-SCIENTIFIC selected-context technical-mini BM25 index"
echo "  arms: ${smoke_start_arm}-${arm_count} selected; budgets are scheduling thresholds, not hard spend caps"
if [[ "$smoke_start_arm" != "1" ]]; then
    echo "  resume: arms 1-$((smoke_start_arm - 1)) are skipped and not revalidated"
fi
echo "  tag: $smoke_tag"

for index in "${selected_indexes[@]}"; do
    command=(
        uv run python -m examples.hotpotqa.main
        --api-profile openrouter
        --runtime-profile technical-smoke
        --solver-model "${models[$index]}"
        --reflection-model "${models[$index]}"
        --condition "${conditions[$index]}"
        --max-metric-calls "${budgets[$index]}"
        --train-limit 6
        --val-limit 5
        --test-limit 2
        --max-workers 1
        --program 2stage
        --seed-style structured
        --reflection-level 2
        --edit-tool-set broad
        --template-family auto
        --retrieval-k 7
        --seed 0
        --technical-mini-index
        --technical-mini-index-dir "$technical_mini_index_dir"
        --tag "${smoke_tag}-${arm_names[$index]}"
    )
    if [[ "${merge_flags[$index]}" == "1" ]]; then
        command+=(--merge)
    fi
    printf 'PLAN %d/%d %-42s' "$((index + 1))" "$arm_count" "${arm_names[$index]}"
    printf ' %q' "${command[@]}"
    printf '\n'
done

if [[ "$mode" == "--dry-run" ]]; then
    exit 0
fi

for dependency in uv curl jq git; do
    if ! command -v "$dependency" >/dev/null 2>&1; then
        echo "Missing required command: $dependency" >&2
        exit 1
    fi
done
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "OPENROUTER_API_KEY is required for --execute." >&2
    exit 1
fi
if ! [[ "$smoke_key_limit_usd" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "OPENROUTER_SMOKE_KEY_LIMIT_USD must be a positive dollar amount." >&2
    exit 1
fi
if ! git diff --quiet --ignore-submodules -- || ! git diff --cached --quiet --ignore-submodules --; then
    echo "Tracked Git changes are present. Commit them before starting a reproducible smoke run." >&2
    exit 1
fi

preflight_cache_dir="outputs/openrouter_tiny_cache/${smoke_tag}/preflight"
mkdir -p "$preflight_cache_dir"
DSPY_CACHEDIR="$preflight_cache_dir" uv run python - "$technical_mini_index_dir" <<'PY'
import sys

from examples.common.wiki17_bm25 import HotPotQATechnicalMiniBM25Retriever
from examples.hotpotqa.utils import load_hotpotqa_dataset, validate_hotpotqa_dspy_runtime

version, commit = validate_hotpotqa_dspy_runtime()
trainset, valset, testset = load_hotpotqa_dataset(train_limit=6, val_limit=5, test_limit=2, seed=0)
if (len(trainset), len(valset), len(testset)) != (6, 5, 2):
    raise RuntimeError("HotPotQA fullwiki preflight did not produce the required 6/5/2 split.")
retriever = HotPotQATechnicalMiniBM25Retriever([*trainset, *valset, *testset], sys.argv[1])
manifest = retriever.prepare()
if not retriever.search(trainset[0]["question"], 7):
    raise RuntimeError("The HotPotQA technical-mini retriever returned no passages.")
print(f"DSPy preflight: {version} at {commit}")
print("HotPotQA fullwiki preflight: 6/5/2 split ready")
print(
    "Technical-mini retrieval preflight: "
    f"{manifest['document_count']} selected-context documents indexed and searchable"
)
PY

qwen_endpoints="$(curl -fsS https://openrouter.ai/api/v1/models/qwen/qwen3.8-27b/endpoints)"
if ! jq -e '
    any(
        .data.endpoints[];
        .tag == "akashml/bf16"
        and .status == 0
        and .quantization == "bf16"
        and (["tools", "tool_choice", "top_k", "reasoning", "reasoning_effort"] - .supported_parameters | length == 0)
        and (.pricing.prompt | tonumber) <= 0.00000040
        and (.pricing.completion | tonumber) <= 0.00000255
    )
' >/dev/null <<<"$qwen_endpoints"; then
    echo "AkashML's Qwen3.8-27B BF16 endpoint is missing, unhealthy, incompatible, or above the pinned price." >&2
    exit 1
fi

model_catalog="$(curl -fsS https://openrouter.ai/api/v1/models)"
if ! jq -e '
    any(
        .data[];
        .id == "qwen/qwen3.8-27b"
        and ((.reasoning.supported_efforts // []) | index("low")) != null
    )
' >/dev/null <<<"$model_catalog"; then
    echo "Qwen3.8-27B no longer exposes low reasoning for the technical smoke run." >&2
    exit 1
fi

deepseek_endpoints="$(curl -fsS https://openrouter.ai/api/v1/models/deepseek/deepseek-v4-flash-0731/endpoints)"
if ! jq -e '
    any(
        .data.endpoints[];
        .tag == "deepseek"
        and .status == 0
        and (["tools", "tool_choice", "reasoning", "reasoning_effort"] - .supported_parameters | length == 0)
        and ([.pricing.prompt, .pricing.overrides[]?.prompt] | map(tonumber) | max) <= 0.00000044
        and ([.pricing.completion, .pricing.overrides[]?.completion] | map(tonumber) | max) <= 0.00000132
    )
' >/dev/null <<<"$deepseek_endpoints"; then
    echo "DeepSeek's V4 Flash 0731 endpoint is missing, unhealthy, incompatible, or above the pinned price." >&2
    exit 1
fi
echo "OpenRouter endpoint preflight: AkashML BF16, Qwen low reasoning, and official DeepSeek verified"

if ! key_json="$(
    curl -fsS --config - <<CURL_CONFIG
url = "https://openrouter.ai/api/v1/key"
header = "Authorization: Bearer ${OPENROUTER_API_KEY}"
CURL_CONFIG
)"; then
    echo "OpenRouter rejected the configured API key." >&2
    exit 1
fi
if ! jq -e --argjson ceiling "$smoke_key_limit_usd" '
    .data.limit != null
    and .data.limit > 0
    and .data.limit <= $ceiling
    and .data.limit_reset == null
    and .data.limit_remaining > 0
' >/dev/null <<<"$key_json"; then
    echo "Use a dedicated OpenRouter key with a non-resetting finite limit of at most \$$smoke_key_limit_usd." >&2
    exit 1
fi
matrix_usage_start="$(jq -r '.data.usage' <<<"$key_json")"
jq -r '"OpenRouter key preflight: limit=$\(.data.limit), remaining=$\(.data.limit_remaining), usage=$\(.data.usage)"' <<<"$key_json"

for index in "${selected_indexes[@]}"; do
    command=(
        uv run python -m examples.hotpotqa.main
        --api-profile openrouter
        --runtime-profile technical-smoke
        --solver-model "${models[$index]}"
        --reflection-model "${models[$index]}"
        --condition "${conditions[$index]}"
        --max-metric-calls "${budgets[$index]}"
        --train-limit 6
        --val-limit 5
        --test-limit 2
        --max-workers 1
        --program 2stage
        --seed-style structured
        --reflection-level 2
        --edit-tool-set broad
        --template-family auto
        --retrieval-k 7
        --seed 0
        --technical-mini-index
        --technical-mini-index-dir "$technical_mini_index_dir"
        --tag "${smoke_tag}-${arm_names[$index]}"
    )
    if [[ "${merge_flags[$index]}" == "1" ]]; then
        command+=(--merge)
    fi

    if ! arm_key_before="$(
        curl -fsS --config - <<CURL_CONFIG
url = "https://openrouter.ai/api/v1/key"
header = "Authorization: Bearer ${OPENROUTER_API_KEY}"
CURL_CONFIG
    )"; then
        echo "Could not read OpenRouter usage before ${arm_names[$index]}." >&2
        exit 1
    fi
    arm_usage_start="$(jq -r '.data.usage' <<<"$arm_key_before")"
    arm_cache_dir="outputs/openrouter_tiny_cache/${smoke_tag}/${arm_names[$index]}"
    mkdir -p "$arm_cache_dir"
    echo "RUN $((index + 1))/${arm_count} ${arm_names[$index]}"
    if ! DSPY_CACHEDIR="$arm_cache_dir" "${command[@]}"; then
        echo "FAILED ${arm_names[$index]}; stopping before later arms." >&2
        exit 1
    fi
    if ! arm_key_after="$(
        curl -fsS --config - <<CURL_CONFIG
url = "https://openrouter.ai/api/v1/key"
header = "Authorization: Bearer ${OPENROUTER_API_KEY}"
CURL_CONFIG
    )"; then
        echo "Could not read OpenRouter usage after ${arm_names[$index]}." >&2
        exit 1
    fi
    arm_usage_end="$(jq -r '.data.usage' <<<"$arm_key_after")"
    arm_usage_delta="$(awk -v start="$arm_usage_start" -v finish="$arm_usage_end" 'BEGIN {printf "%.6f", finish - start}')"
    echo "PASS ${arm_names[$index]} (OpenRouter usage delta: \$$arm_usage_delta)"
done

if ! final_key_json="$(
    curl -fsS --config - <<CURL_CONFIG
url = "https://openrouter.ai/api/v1/key"
header = "Authorization: Bearer ${OPENROUTER_API_KEY}"
CURL_CONFIG
)"; then
    echo "Selected arms passed, but final OpenRouter usage could not be read." >&2
    exit 1
fi
matrix_usage_end="$(jq -r '.data.usage' <<<"$final_key_json")"
matrix_usage_delta="$(awk -v start="$matrix_usage_start" -v finish="$matrix_usage_end" 'BEGIN {printf "%.6f", finish - start}')"
if [[ "$smoke_start_arm" == "1" ]]; then
    echo "All ${arm_count} HotPotQA smoke arms passed. OpenRouter usage delta: \$$matrix_usage_delta"
else
    echo "HotPotQA smoke arms ${smoke_start_arm}-${arm_count} passed. Invocation usage delta: \$$matrix_usage_delta"
fi
