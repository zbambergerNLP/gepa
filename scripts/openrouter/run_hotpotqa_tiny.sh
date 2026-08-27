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
wiki17_dir="${WIKI17_DIR:-${cache_base}/gepa/wiki17}"
smoke_tag="${SMOKE_TAG:-openrouter-tiny-$(date -u +%Y%m%dT%H%M%SZ)}"
smoke_key_limit_usd="${OPENROUTER_SMOKE_KEY_LIMIT_USD:-25}"

arm_names=(
    deepseek-vanilla-no-merge
    deepseek-react-v2-no-merge
    qwen-vanilla-no-merge
    qwen-react-v2-no-merge
    deepseek-vanilla-merge
    deepseek-react-v2-merge
    qwen-vanilla-merge
    qwen-react-v2-merge
)
models=(
    deepseek/deepseek-v4-flash
    deepseek/deepseek-v4-flash
    hosted_vllm/Qwen/Qwen3.8-27B
    hosted_vllm/Qwen/Qwen3.8-27B
    deepseek/deepseek-v4-flash
    deepseek/deepseek-v4-flash
    hosted_vllm/Qwen/Qwen3.8-27B
    hosted_vllm/Qwen/Qwen3.8-27B
)
conditions=(vanilla react_v2 vanilla react_v2 vanilla react_v2 vanilla react_v2)
merge_flags=(0 0 0 0 1 1 1 1)
budgets=(6 6 6 6 28 28 28 28)

echo "HotPotQA OpenRouter technical-smoke matrix"
echo "  fullwiki split: 6 train / 5 validation / 2 test"
echo "  arms: 8 isolated processes; budgets are scheduling thresholds, not hard spend caps"
echo "  tag: $smoke_tag"

for index in "${!arm_names[@]}"; do
    command=(
        uv run python -m examples.hotpotqa.main
        --api-profile openrouter
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
        --wiki17-dir "$wiki17_dir"
        --tag "${smoke_tag}-${arm_names[$index]}"
    )
    if [[ "${merge_flags[$index]}" == "1" ]]; then
        command+=(--merge)
    fi
    printf 'PLAN %d/8 %-32s' "$((index + 1))" "${arm_names[$index]}"
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

memory_bytes=""
if [[ "$(uname -s)" == "Darwin" ]]; then
    memory_bytes="$(sysctl -n hw.memsize)"
elif [[ -r /proc/meminfo ]]; then
    memory_kib="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
    memory_bytes="$((memory_kib * 1024))"
fi
minimum_memory_bytes="$((48 * 1024 * 1024 * 1024))"
if [[ -z "$memory_bytes" || "$memory_bytes" -lt "$minimum_memory_bytes" ]]; then
    echo "The frozen Wiki-2017 retriever requires a host with at least 48 GiB RAM for this launcher." >&2
    echo "API inference removes the GPU requirement, but the 5.23M-document corpus and BM25 index remain local." >&2
    exit 1
fi

preflight_cache_dir="outputs/openrouter_tiny_cache/${smoke_tag}/preflight"
mkdir -p "$preflight_cache_dir"
DSPY_CACHEDIR="$preflight_cache_dir" uv run python - <<'PY'
from examples.hotpotqa.utils import load_hotpotqa_dataset, validate_hotpotqa_dspy_runtime

version, commit = validate_hotpotqa_dspy_runtime()
trainset, valset, testset = load_hotpotqa_dataset(train_limit=6, val_limit=5, test_limit=2, seed=0)
if (len(trainset), len(valset), len(testset)) != (6, 5, 2):
    raise RuntimeError("HotPotQA fullwiki preflight did not produce the required 6/5/2 split.")
print(f"DSPy preflight: {version} at {commit}")
print("HotPotQA fullwiki preflight: 6/5/2 split ready")
PY
uv run python -m examples.common.wiki17_bm25 verify --root "$wiki17_dir" >/dev/null
echo "Wiki-2017 preflight: verified $wiki17_dir"

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
echo "OpenRouter endpoint preflight: AkashML BF16 and official DeepSeek routes verified"

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

for index in "${!arm_names[@]}"; do
    command=(
        uv run python -m examples.hotpotqa.main
        --api-profile openrouter
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
        --wiki17-dir "$wiki17_dir"
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
    echo "RUN $((index + 1))/8 ${arm_names[$index]}"
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
    echo "All arms passed, but final OpenRouter usage could not be read." >&2
    exit 1
fi
matrix_usage_end="$(jq -r '.data.usage' <<<"$final_key_json")"
matrix_usage_delta="$(awk -v start="$matrix_usage_start" -v finish="$matrix_usage_end" 'BEGIN {printf "%.6f", finish - start}')"
echo "All eight HotPotQA smoke arms passed. OpenRouter usage delta: \$$matrix_usage_delta"
