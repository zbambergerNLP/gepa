"""Run vanilla and action-conditioned GEPA variants on HotPotQA.

The default benchmark path follows the GEPA paper artifact's data split,
frozen Wiki-2017 BM25 retrieval, two-hop four-component program, component
feedback, exact-match objective, and 6,871-call budget. The locked production
campaign runs four methods at that standard budget and only the headline
vanilla/ReAct V2 pair at the 13,742-call two-times budget. Model identities
remain configurable, and all conditions receive the same model assignments and
task evidence.

Conditions:
    vanilla         - stock GEPA reflective mutation
    react_v2        - verbalized section/action Controller, Manifestor, ReAct V2 proposer
    react_v2_random - uniform-random section/action selection, Manifestor, ReAct V2 proposer
    random          - stateless action-conditioned reflection with uniform-random actions
    action          - stateless action-conditioned reflection with verbalized sampling

Usage:
    uv run python -m examples.hotpotqa.main [--condition vanilla|random|action|react_v2|react_v2_random|all]
        [--max-metric-calls N] [--train-limit N] [--val-limit N] [--test-limit N]
    # Smoke (20 ex, 14/3/3):
    uv run python -m examples.hotpotqa.main --data-path examples/hotpotqa/data/hotpotqa_distractor_sample.jsonl --max-metric-calls 200 --condition both
"""

import argparse
import hashlib
import itertools
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlsplit

from examples.common.experiment_models import (
    DEEPSEEK_API_BASE,
    DEEPSEEK_V4_FLASH_0731_OPENROUTER_MODEL,
    DEEPSEEK_V4_FLASH_MODEL,
    EXPERIMENT_NUM_RETRIES,
    QWEN3_8_27B_MODEL,
    experiment_decoding,
    experiment_model_version,
    experiment_request_overrides,
    resolve_experiment_model,
    validate_experiment_model_pair,
)
from examples.common.react_v2 import (
    benchmark_data_identity,
    build_react_v2_strategy,
    ensure_wikipedia_run_contract,
    experiment_run_key,
    file_sha256,
    resolve_template_family,
    structured_prompt,
)
from examples.common.wiki17_bm25 import (
    DEFAULT_HOTPOTQA_TECHNICAL_MINI_ROOT,
    DEFAULT_WIKI17_ROOT,
    GEPA_ARTIFACT_COMMIT,
    HotPotQATechnicalMiniBM25Retriever,
    Wiki17BM25Retriever,
)
from examples.common.wikipedia import WikipediaRetriever
from examples.hotpotqa.utils import (
    HOTPOTQA_DEEPSEEK_RESPONSE_MODEL_ENV,
    HOTPOTQA_DEEPSEEK_SYSTEM_FINGERPRINT_ENV,
    HOTPOTQA_DSPY_COMMIT,
    HOTPOTQA_DSPY_VERSION,
    HOTPOTQA_HF_REVISION,
    HOTPOTQA_RUNTIME_PROFILES,
    HOTPOTQA_SCIENTIFIC_SPLIT_SHA256,
    artifact_component_records,
    build_hotpotqa_task_lm,
    f1_score,
    hotpotqa_metric,
    load_hotpotqa_dataset,
    normalize_answer,
    resolve_hotpotqa_lm_kwargs,
    run_single_stage,
    run_two_stage,
)
from gepa.core.action_tracking import ActionDiversityCallback
from gepa.lm import LM
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    MergeConfig,
    ReflectionConfig,
    SideInfo,
    optimize_anything,
)
from gepa.response_journal import RESPONSE_JOURNAL_SCHEMA_VERSION, RESPONSE_JOURNAL_SCOPE_POLICY
from gepa.strategies.action_space import (
    RandomActionSelector,
    VerbalizedActionSelector,
    stateless_selector_policy_contract,
)
from gepa.strategies.document_template import TEMPLATE_FAMILIES
from gepa.strategies.instruction_proposal import InstructionProposalSignature
from gepa.strategies.intervention import (
    CONTROLLER_POLICY_CONTRACT,
    SEMANTIC_ACTION_CATALOGS,
    SEMANTIC_ACTIONS,
    STATELESS_ACTION_MENU_VERSION,
    UNIFORM_RANDOM_CONTROLLER_POLICY_CONTRACT,
    StatelessActionConstraint,
)
from gepa.strategies.proposal_sampling import SingleMutationSampling
from gepa.strategies.proposal_selection import AllImprovements

# GEPA artifact components: summarize1 -> create_query_hop2 -> summarize2 -> final_answer.
SEED_CANDIDATE = {
    "summarize1": "Given the fields `question`, `passages`, produce the fields `summary`.",
    "create_query_hop2": "Given the fields `question`, `summary_1`, produce the fields `query`.",
    "summarize2": "Given the fields `question`, `context`, `passages`, produce the fields `summary`.",
    "final_answer": "Given the fields `question`, `summary_1`, `summary_2`, produce the fields `answer`.",
}

# 1-stage ablation (single prompt)
SEED_CANDIDATE_1STAGE = {
    "answer_question": (
        "Answer the question from the retrieved passages. Ignore irrelevant passages and "
        "combine evidence across pages. Return a concise answer."
    ),
}

_INITIAL_PROMPT = SEED_CANDIDATE_1STAGE["answer_question"]

_CONDITION_DIR_NAMES = {
    "vanilla": "hotpotqa_vanilla",
    "react_v2": "hotpotqa_react_v2",
    "react_v2_random": "hotpotqa_react_v2_random",
    "random": "hotpotqa_random_action",
    "action": "hotpotqa_verbalized_action",
}

_CONDITION_LABELS = {
    "vanilla": "GEPA",
    "react_v2": "ReAct V2",
    "react_v2_random": "Random-Controller ReAct V2",
    "random": "Random-action GEPA",
    "action": "Verbalized-action GEPA",
}

_PAPER_MAX_MERGE_INVOCATIONS = 5
_PAPER_MERGE_VAL_OVERLAP_FLOOR = 5
_SCIENTIFIC_CONDITIONS_BY_BUDGET = {
    6_871: ("vanilla", "react_v2", "react_v2_random", "action"),
    13_742: ("vanilla", "react_v2"),
}
_SCIENTIFIC_METRIC_CALL_BUDGETS = set(_SCIENTIFIC_CONDITIONS_BY_BUDGET)
_SCIENTIFIC_PYTHON_VERSION = "3.11.13"
_SCIENTIFIC_UV_VERSION = "0.9.13"
_SCIENTIFIC_SPLIT_COUNTS = {"train": 150, "val": 300, "test": 300}
_REACT_V2_CONDITIONS = {"react_v2", "react_v2_random"}
_SEMANTIC_CONDITIONS = {"react_v2", "react_v2_random", "random", "action"}


def _validated_runtime_profile(args) -> str:
    """Validate and return the selected HotPotQA runtime profile.

    Args:
        args: Parsed arguments or an equivalent configuration namespace.

    Returns:
        Validated runtime-profile name.

    Raises:
        ValueError: The profile is unknown, technical smoke is not paired with
            its explicit non-scientific inputs, or an enforced scientific run
            changes a methodology axis.
    """
    runtime_profile = getattr(args, "runtime_profile", "scientific")
    if runtime_profile not in HOTPOTQA_RUNTIME_PROFILES:
        supported = ", ".join(HOTPOTQA_RUNTIME_PROFILES)
        raise ValueError(f"Unsupported HotPotQA runtime profile {runtime_profile!r}; expected one of: {supported}")
    if runtime_profile == "technical-smoke" and (
        not getattr(args, "technical_mini_index", False) or args.api_profile != "openrouter"
    ):
        raise ValueError(
            "--runtime-profile technical-smoke requires --technical-mini-index and --api-profile openrouter"
        )
    if runtime_profile == "scientific" and getattr(args, "enforce_scientific_contract", False):
        changed_axes = []
        required_values = (
            ("api_profile", "direct"),
            ("program", "2stage"),
            ("seed_style", "structured"),
            ("seed", 0),
            ("retrieval_k", 7),
            ("reflection_level", 2),
            ("edit_tool_set", "broad"),
            ("template_family", "auto"),
        )
        for name, expected in required_values:
            if getattr(args, name, expected) != expected:
                changed_axes.append(f"--{name.replace('_', '-')} must be {expected!r}")
        if getattr(args, "data_path", None) is not None:
            changed_axes.append("--data-path must be omitted")
        if getattr(args, "technical_mini_index", False):
            changed_axes.append("--technical-mini-index must be omitted")
        for name in ("train_limit", "val_limit", "test_limit"):
            if getattr(args, name, None) is not None:
                changed_axes.append(f"--{name.replace('_', '-')} must be omitted")
        if getattr(args, "merge", False):
            changed_axes.append("--merge must be omitted")
        max_metric_calls = getattr(args, "max_metric_calls", 6_871)
        if max_metric_calls not in _SCIENTIFIC_METRIC_CALL_BUDGETS:
            changed_axes.append("--max-metric-calls must be 6871 or 13742")
        else:
            requested_condition = getattr(args, "condition", "both")
            approved_conditions = _SCIENTIFIC_CONDITIONS_BY_BUDGET[max_metric_calls]
            if requested_condition not in {*approved_conditions, "all", "both"}:
                approved_text = ", ".join(approved_conditions)
                changed_axes.append(
                    f"--condition {requested_condition!r} is not in the approved {max_metric_calls}-call "
                    f"campaign cells ({approved_text})"
                )
        if getattr(args, "max_workers", 1) < 1:
            changed_axes.append("--max-workers must be positive")
        source_commit = os.environ.get("HOTPOTQA_SOURCE_COMMIT", "")
        if len(source_commit) != 40 or any(character not in "0123456789abcdef" for character in source_commit):
            changed_axes.append("HOTPOTQA_SOURCE_COMMIT must identify the exact experiment source")
        source_manifest = os.environ.get("HOTPOTQA_SOURCE_MANIFEST_SHA256", "")
        if len(source_manifest) != 64 or any(
            character not in "0123456789abcdef" for character in source_manifest
        ):
            changed_axes.append("HOTPOTQA_SOURCE_MANIFEST_SHA256 must identify the exact source bytes")
        if os.environ.get("HOTPOTQA_PYTHON_VERSION") != _SCIENTIFIC_PYTHON_VERSION:
            changed_axes.append(f"HOTPOTQA_PYTHON_VERSION must be {_SCIENTIFIC_PYTHON_VERSION!r}")
        if os.environ.get("HOTPOTQA_UV_VERSION") != _SCIENTIFIC_UV_VERSION:
            changed_axes.append(f"HOTPOTQA_UV_VERSION must be {_SCIENTIFIC_UV_VERSION!r}")
        uv_sha256 = os.environ.get("HOTPOTQA_UV_SHA256", "")
        if len(uv_sha256) != 64 or any(character not in "0123456789abcdef" for character in uv_sha256):
            changed_axes.append("HOTPOTQA_UV_SHA256 must identify the exact uv binary")
        if not os.environ.get("HOTPOTQA_LITELLM_VERSION"):
            changed_axes.append("HOTPOTQA_LITELLM_VERSION must identify the client runtime")
        if not os.environ.get("HOTPOTQA_CAMPAIGN_ID"):
            changed_axes.append("HOTPOTQA_CAMPAIGN_ID must identify the experiment campaign")
        env_spec = os.environ.get("HOTPOTQA_ENV_SPEC_SHA256", "")
        if len(env_spec) != 64 or any(character not in "0123456789abcdef" for character in env_spec):
            changed_axes.append("HOTPOTQA_ENV_SPEC_SHA256 must identify the exact dependency lock")
        realized_environment = os.environ.get("HOTPOTQA_GEPA_ENV_SHA256", "")
        if len(realized_environment) != 64 or any(
            character not in "0123456789abcdef" for character in realized_environment
        ):
            changed_axes.append("HOTPOTQA_GEPA_ENV_SHA256 must identify the frozen task environment")
        expected_model_version = experiment_model_version(args.solver_model)
        if os.environ.get("HOTPOTQA_MODEL_REVISION") != expected_model_version:
            changed_axes.append(f"HOTPOTQA_MODEL_REVISION must be {expected_model_version!r}")
        solver_api_base = args.solver_api_base if args.solver_api_base is not None else args.api_base
        reflection_api_base = (
            args.reflection_api_base if args.reflection_api_base is not None else args.api_base
        )
        if args.solver_model == QWEN3_8_27B_MODEL:
            for role, api_base in (("solver", solver_api_base), ("reflection", reflection_api_base)):
                parsed_api_base = urlsplit(api_base or "")
                try:
                    valid_loopback = (
                        parsed_api_base.scheme == "http"
                        and parsed_api_base.hostname in {"localhost", "127.0.0.1", "::1"}
                        and parsed_api_base.port is not None
                        and parsed_api_base.path.rstrip("/") == "/v1"
                        and not parsed_api_base.query
                        and not parsed_api_base.fragment
                    )
                except ValueError:
                    valid_loopback = False
                if not valid_loopback:
                    changed_axes.append(f"--{role}-api-base must identify the local vLLM /v1 endpoint")
            model_integrity = os.environ.get("HOTPOTQA_MODEL_INTEGRITY_SHA256", "")
            if len(model_integrity) != 64 or any(
                character not in "0123456789abcdef" for character in model_integrity
            ):
                changed_axes.append("HOTPOTQA_MODEL_INTEGRITY_SHA256 must identify the verified checkpoint bytes")
            if os.environ.get("HOTPOTQA_WEIGHT_DTYPE") != "bfloat16":
                changed_axes.append("HOTPOTQA_WEIGHT_DTYPE must be 'bfloat16'")
            if os.environ.get("HOTPOTQA_KV_CACHE_DTYPE") != "auto":
                changed_axes.append("HOTPOTQA_KV_CACHE_DTYPE must be 'auto'")
            if os.environ.get("HOTPOTQA_VLLM_BATCH_INVARIANT") != "false":
                changed_axes.append("HOTPOTQA_VLLM_BATCH_INVARIANT must be 'false'")
            if os.environ.get("HOTPOTQA_VLLM_SINGLE_SEQUENCE_REPLICAS") != "true":
                changed_axes.append("HOTPOTQA_VLLM_SINGLE_SEQUENCE_REPLICAS must be 'true'")
            if not os.environ.get("HOTPOTQA_VLLM_VERSION"):
                changed_axes.append("HOTPOTQA_VLLM_VERSION must identify the serving runtime")
            if not os.environ.get("HOTPOTQA_TRANSFORMERS_VERSION"):
                changed_axes.append("HOTPOTQA_TRANSFORMERS_VERSION must identify the serving runtime")
            posit_commit = os.environ.get("HOTPOTQA_POSIT_COMMIT", "")
            if len(posit_commit) != 40 or any(character not in "0123456789abcdef" for character in posit_commit):
                changed_axes.append("HOTPOTQA_POSIT_COMMIT must identify the exact serving source")
            posit_environment = os.environ.get("HOTPOTQA_POSIT_ENV_SHA256", "")
            if len(posit_environment) != 64 or any(
                character not in "0123456789abcdef" for character in posit_environment
            ):
                changed_axes.append(
                    "HOTPOTQA_POSIT_ENV_SHA256 must identify the frozen serving environment"
                )
            gpu_runtime_text = os.environ.get("HOTPOTQA_GPU_RUNTIME", "")
            try:
                gpu_runtime = json.loads(gpu_runtime_text)
            except json.JSONDecodeError:
                gpu_runtime = None
            if not isinstance(gpu_runtime, dict):
                changed_axes.append("HOTPOTQA_GPU_RUNTIME must identify the allocated H200 runtime")
            else:
                gpu_count = gpu_runtime.get("count")
                gpu_names = gpu_runtime.get("names")
                gpu_capabilities = gpu_runtime.get("compute_capabilities")
                driver_version = gpu_runtime.get("driver_version")
                canonical_runtime = json.dumps(gpu_runtime, sort_keys=True, separators=(",", ":"))
                if (
                    gpu_count != 8
                    or not isinstance(gpu_names, list)
                    or len(gpu_names) != gpu_count
                    or not all(isinstance(name, str) and "H200" in name.upper() for name in gpu_names)
                    or not isinstance(gpu_capabilities, list)
                    or gpu_capabilities != ["9.0"] * gpu_count
                    or not isinstance(driver_version, str)
                    or not driver_version
                    or canonical_runtime != gpu_runtime_text
                ):
                    changed_axes.append(
                        "HOTPOTQA_GPU_RUNTIME must record only H200 devices with compute capability 9.0 "
                        "and one NVIDIA driver version"
                    )
            serve_arguments = os.environ.get("HOTPOTQA_VLLM_SERVE_ARGUMENTS", "")
            required_serve_settings = (
                "tp=1",
                "gpu_memory_utilization=0.92",
                "max_model_len=262144",
                "rope_scaling=none",
                "max_num_seqs=1",
                "dtype=bfloat16",
                "kv_cache_dtype=auto",
                "prefix_caching=false",
                "reasoning_parser=qwen3",
                "auto_tool_choice=true",
                "tool_parser=qwen3_coder",
                "seed=0",
                "batch_invariant=false",
                "single_sequence_replicas=true",
            )
            for setting in required_serve_settings:
                if setting not in serve_arguments.split(";"):
                    changed_axes.append(f"HOTPOTQA_VLLM_SERVE_ARGUMENTS must include {setting!r}")
        elif args.solver_model == DEEPSEEK_V4_FLASH_MODEL:
            for role, api_base in (("solver", solver_api_base), ("reflection", reflection_api_base)):
                if api_base != DEEPSEEK_API_BASE:
                    changed_axes.append(f"--{role}-api-base must be {DEEPSEEK_API_BASE!r}")
            if not os.environ.get(HOTPOTQA_DEEPSEEK_RESPONSE_MODEL_ENV):
                changed_axes.append(
                    f"{HOTPOTQA_DEEPSEEK_RESPONSE_MODEL_ENV} must identify the provider response model"
                )
            if not os.environ.get(HOTPOTQA_DEEPSEEK_SYSTEM_FINGERPRINT_ENV):
                changed_axes.append(
                    f"{HOTPOTQA_DEEPSEEK_SYSTEM_FINGERPRINT_ENV} must identify the provider runtime"
                )
        if changed_axes:
            details = "; ".join(changed_axes)
            raise ValueError(f"The enforced HotPotQA scientific contract rejected changed methodology: {details}.")
    return runtime_profile


def _validate_scientific_data_identity(args) -> None:
    """Require the exact ordered HotPotQA records used by production runs.

    Args:
        args: Parsed arguments carrying the computed ``data_identity`` and the
            scientific-contract enforcement flag.

    Raises:
        ValueError: An enforced scientific run has a different source
            revision, split size, or ordered-record content digest.
    """
    if not getattr(args, "enforce_scientific_contract", False):
        return
    source = args.data_identity.get("source", {})
    if source.get("revision") != HOTPOTQA_HF_REVISION:
        raise ValueError(
            "The enforced HotPotQA scientific contract requires Hugging Face revision "
            f"{HOTPOTQA_HF_REVISION}."
        )
    for split_name, expected_count in _SCIENTIFIC_SPLIT_COUNTS.items():
        split = args.data_identity.get("splits", {}).get(split_name, {})
        expected_digest = HOTPOTQA_SCIENTIFIC_SPLIT_SHA256[split_name]
        if split.get("count") != expected_count or split.get("sha256") != expected_digest:
            raise ValueError(
                "The enforced HotPotQA scientific contract rejected the selected "
                f"{split_name} split: expected count={expected_count}, sha256={expected_digest}."
            )


def _component_kinds(program: str) -> dict[str, str]:
    """Identify every optimized HotPotQA instruction as a system message.

    Args:
        program: ``"1stage"`` or the two-stage program variant.

    Returns:
        Mapping from each seed component to ``system_prompt``.
    """
    seed = SEED_CANDIDATE_1STAGE if program == "1stage" else SEED_CANDIDATE
    return dict.fromkeys(seed, "system_prompt")


def condition_run_dir(condition: str, program: str, tag: str = "", run_key: str = "") -> str:
    """Build the output directory for one HotPotQA condition.

    Args:
        condition: Experiment condition naming the directory family.
        program: Program variant; ``"1stage"`` adds its identifying suffix.
        tag: Optional human-readable run suffix.
        run_key: Optional compatibility key for resumable state.

    Returns:
        Relative output directory for the condition.
    """
    suffix = "_1stage" if program == "1stage" else ""
    tag_suffix = f"_{tag}" if tag else ""
    key_suffix = f"_{run_key}" if run_key else ""
    return f"outputs/{_CONDITION_DIR_NAMES[condition]}{suffix}{key_suffix}{tag_suffix}"


def _contract_api_base(api_base: str | None, *, scientific_contract: bool) -> str | None:
    """Remove an ephemeral loopback port from scientific run identity.

    Della assigns the local vLLM process a port derived from the Slurm job ID.
    That transport detail must not prevent an interrupted production run from
    finding its existing output and held-out checkpoints. External endpoints
    remain material because they may identify a different serving deployment.

    Args:
        api_base: Runtime completion endpoint, when one is configured.
        scientific_contract: Whether the fail-closed production contract is
            active.

    Returns:
        Stable local endpoint identity for a scientific loopback URL, otherwise
        the original API base.
    """
    if not scientific_contract or api_base is None:
        return api_base
    parsed = urlsplit(api_base)
    if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        return api_base
    endpoint_path = parsed.path.rstrip("/")
    return f"local-loopback{endpoint_path}"


def build_run_contract(condition: str, args) -> dict:
    """Build the complete persisted configuration for one condition.

    Args:
        condition: Optimization condition being recorded.
        args: Parsed experiment arguments and computed data identity.

    Returns:
        JSON-serializable model, optimizer, retrieval, and data contract.
    """
    validate_experiment_model_pair(args.solver_model, args.reflection_model)
    family = resolve_template_family(args.template_family, args.solver_model)
    solver_api_base = args.solver_api_base if args.solver_api_base is not None else args.api_base
    reflection_api_base = args.reflection_api_base if args.reflection_api_base is not None else args.api_base
    scientific_contract = bool(getattr(args, "enforce_scientific_contract", False))
    solver_api_identity = _contract_api_base(solver_api_base, scientific_contract=scientific_contract)
    reflection_api_identity = _contract_api_base(reflection_api_base, scientific_contract=scientific_contract)
    solver_runtime_model = resolve_experiment_model(args.solver_model, args.api_profile)
    reflection_runtime_model = resolve_experiment_model(args.reflection_model, args.api_profile)
    runtime_profile = _validated_runtime_profile(args)
    solver_lm_kwargs = resolve_hotpotqa_lm_kwargs(solver_runtime_model, None, runtime_profile)
    reflection_lm_kwargs = resolve_hotpotqa_lm_kwargs(reflection_runtime_model, None, runtime_profile)
    solver_decoding_fields = list(experiment_decoding(solver_runtime_model))
    if "seed" in solver_lm_kwargs:
        solver_decoding_fields.append("seed")
    solver_request_fields = experiment_request_overrides(solver_runtime_model)
    reflection_decoding_fields = list(experiment_decoding(reflection_runtime_model))
    if "seed" in reflection_lm_kwargs:
        reflection_decoding_fields.append("seed")
    reflection_request_fields = experiment_request_overrides(reflection_runtime_model)
    reflection_decoding = {
        field: deepcopy(reflection_lm_kwargs[field]) for field in reflection_decoding_fields
    }
    reflection_level = args.reflection_level if condition in _REACT_V2_CONDITIONS else 0
    reflection_role_decoding = None
    if condition in _REACT_V2_CONDITIONS:
        manifestor_decoding = deepcopy(reflection_decoding)
        manifestor_decoding["temperature"] = 0
        ignored_manifestor_fields = []
        if reflection_runtime_model in {
            DEEPSEEK_V4_FLASH_MODEL,
            DEEPSEEK_V4_FLASH_0731_OPENROUTER_MODEL,
        }:
            ignored_manifestor_fields = ["temperature"]
        reflection_role_decoding = {
            "controller": (
                {
                    "requested": deepcopy(reflection_decoding),
                    "provider_ignored_fields": [],
                }
                if condition == "react_v2"
                else None
            ),
            "manifestor": (
                {
                    "requested": manifestor_decoding,
                    "provider_ignored_fields": ignored_manifestor_fields,
                }
                if reflection_level >= 2
                else None
            ),
            "react_v2_proposer": {
                "requested": deepcopy(reflection_decoding),
                "provider_ignored_fields": [],
            },
        }
    provider_response_identity = None
    if args.api_profile == "direct" and args.solver_model == DEEPSEEK_V4_FLASH_MODEL:
        response_model = os.environ.get(HOTPOTQA_DEEPSEEK_RESPONSE_MODEL_ENV)
        system_fingerprint = os.environ.get(HOTPOTQA_DEEPSEEK_SYSTEM_FINGERPRINT_ENV)
        if response_model and system_fingerprint:
            provider_response_identity = {
                "model": response_model,
                "system_fingerprint": system_fingerprint,
            }
    edit_tool_set = args.edit_tool_set if condition in _REACT_V2_CONDITIONS else None
    stateless_semantic = condition in ("random", "action")
    retrieval_provenance = getattr(args, "retrieval_provenance", None)
    if retrieval_provenance is None:
        retrieval_provenance = Wiki17BM25Retriever(args.wiki17_dir).provenance()
    technical_mini_index = retrieval_provenance["backend"] == "hotpotqa-technical-mini-bm25s"
    merge = None
    if args.merge:
        merge = {
            "max_merge_invocations": _PAPER_MAX_MERGE_INVOCATIONS,
            "merge_val_overlap_floor": _PAPER_MERGE_VAL_OVERLAP_FLOOR,
        }
    template = TEMPLATE_FAMILIES[family]["system_prompt"]
    rendered_seed = seed_candidate(args.program, args.seed_style, family)
    stateless_action_menu = None
    if stateless_semantic:
        stateless_action_menu = {
            "version": STATELESS_ACTION_MENU_VERSION,
            "semantic_action_catalog_version": SEMANTIC_ACTION_CATALOGS[template.kind]["version"],
            "kind": template.kind,
            "sections": list(template.sections),
            "choices": [
                {
                    "id": f"{spec.name}@{section}/{spec.edit_tool.value}",
                    "semantic_action": spec.name,
                    "operator": spec.edit_tool.value,
                    "target_section": section,
                }
                for section in template.sections
                for spec in SEMANTIC_ACTIONS
            ],
        }
    semantic_controller_policy = None
    if reflection_level == 2:
        if condition == "react_v2_random":
            semantic_controller_policy = deepcopy(UNIFORM_RANDOM_CONTROLLER_POLICY_CONTRACT)
        else:
            semantic_controller_policy = deepcopy(CONTROLLER_POLICY_CONTRACT)
    return {
        "schema_version": 14,
        "benchmark": "hotpotqa-technical-mini" if technical_mini_index else "hotpotqa-fullwiki-wiki17",
        "reference_artifact_commit": GEPA_ARTIFACT_COMMIT,
        "scientific_contract_enforced": scientific_contract,
        "condition": condition,
        "models": {
            "api_profile": args.api_profile,
            "runtime_profile": runtime_profile,
            "solver": args.solver_model,
            "solver_runtime": solver_runtime_model,
            "solver_version": experiment_model_version(solver_runtime_model),
            "solver_api_base": solver_api_identity,
            "solver_provider_response_identity": deepcopy(provider_response_identity),
            "solver_decoding": {field: deepcopy(solver_lm_kwargs[field]) for field in solver_decoding_fields},
            "solver_request_overrides": {field: deepcopy(solver_lm_kwargs[field]) for field in solver_request_fields},
            "solver_num_retries": EXPERIMENT_NUM_RETRIES,
            "reflection": args.reflection_model,
            "reflection_runtime": reflection_runtime_model,
            "reflection_version": experiment_model_version(reflection_runtime_model),
            "reflection_api_base": reflection_api_identity,
            "reflection_provider_response_identity": deepcopy(provider_response_identity),
            "reflection_decoding": reflection_decoding,
            "reflection_role_decoding": reflection_role_decoding,
            "reflection_request_overrides": {
                field: deepcopy(reflection_lm_kwargs[field]) for field in reflection_request_fields
            },
            "reflection_num_retries": EXPERIMENT_NUM_RETRIES,
        },
        "optimizer": {
            "max_metric_calls": args.max_metric_calls,
            "seed": args.seed,
            "candidate_selection_strategy": "pareto",
            "proposal_sampling_strategy": {
                "name": "single_mutation",
                "parents_per_iteration": 1,
                "mutations_per_parent": 1,
            },
            "proposal_selection_strategy": "all_improvements",
            "frontier_type": "instance",
            "validation_evaluation": "full_eval",
            "acceptance_criterion": "strict_improvement",
            "raise_on_exception": True,
            "batch_sampler": "epoch_shuffled",
            "reflection_minibatch_size": 3,
            "component_selector": "round_robin",
            "skip_perfect_score": runtime_profile != "technical-smoke",
            "perfect_score": 1.0,
            "merge": merge,
            "vanilla_reflection_prompt": "canonical_gepa" if condition == "vanilla" else None,
            "seed_style": args.seed_style,
            "artifact_seed_instructions": dict(SEED_CANDIDATE_1STAGE if args.program == "1stage" else SEED_CANDIDATE),
            "rendered_seed": rendered_seed,
            "template_family": family,
            "component_kinds": _component_kinds(args.program),
            "reflection_level": reflection_level,
            "edit_tool_set": edit_tool_set,
            "semantic_action_space": (
                deepcopy(SEMANTIC_ACTION_CATALOGS["prompt"]) if reflection_level == 2 or stateless_semantic else None
            ),
            "semantic_controller_policy": semantic_controller_policy,
            "stateless_action_menu": stateless_action_menu,
            "stateless_selector_policy": (
                stateless_selector_policy_contract("random" if condition == "random" else "verbalized")
                if stateless_semantic
                else None
            ),
            "response_journal": {
                "schema_version": RESPONSE_JOURNAL_SCHEMA_VERSION,
                "scope_policy": RESPONSE_JOURNAL_SCOPE_POLICY,
                "addressing": "lm-namespace-call-ordinal",
                "request_storage": "sha256-only",
            },
            "branch_history": (
                {
                    "storage": "target_scoped_user_assistant_messages",
                    "direct_deepseek_native_delivery": "quoted_user_context",
                    "other_delivery": "provider_chat_messages",
                }
                if condition in _REACT_V2_CONDITIONS
                else None
            ),
        },
        "program": {
            "name": args.program,
            "predictor_type": "dspy_chain_of_thought" if args.program == "2stage" else "direct",
            "predictor_adapter": "dspy_chat_adapter" if args.program == "2stage" else None,
            "dspy_runtime_version": HOTPOTQA_DSPY_VERSION if args.program == "2stage" else None,
            "dspy_runtime_commit": HOTPOTQA_DSPY_COMMIT if args.program == "2stage" else None,
            "retrieval_k": args.retrieval_k,
            "parallel_workers": args.max_workers,
            "cache_evaluation": True,
            "dspy_disk_cache": True,
            "dspy_memory_cache": False,
            "dspy_history": False,
            "primary_metric": "normalized_exact_match",
            "reported_supplemental_metric": "token_f1",
            "task_inputs": ["question"],
            "components": list(rendered_seed),
            "component_output_fields": (
                {
                    "summarize1": ["reasoning", "summary"],
                    "create_query_hop2": ["reasoning", "query"],
                    "summarize2": ["reasoning", "summary"],
                    "final_answer": ["reasoning", "answer"],
                }
                if args.program == "2stage"
                else {"answer_question": ["answer"]}
            ),
        },
        "retrieval": deepcopy(retrieval_provenance),
        "data": args.data_identity,
        "execution_runtime": {
            "campaign_id": os.environ.get("HOTPOTQA_CAMPAIGN_ID"),
            "source_commit": os.environ.get("HOTPOTQA_SOURCE_COMMIT"),
            "source_manifest_sha256": os.environ.get("HOTPOTQA_SOURCE_MANIFEST_SHA256"),
            "python_version": os.environ.get("HOTPOTQA_PYTHON_VERSION"),
            "uv_version": os.environ.get("HOTPOTQA_UV_VERSION"),
            "uv_sha256": os.environ.get("HOTPOTQA_UV_SHA256"),
            "env_spec_sha256": os.environ.get("HOTPOTQA_ENV_SPEC_SHA256"),
            "gepa_env_sha256": os.environ.get("HOTPOTQA_GEPA_ENV_SHA256"),
            "posit_commit": os.environ.get("HOTPOTQA_POSIT_COMMIT"),
            "posit_env_sha256": os.environ.get("HOTPOTQA_POSIT_ENV_SHA256"),
            "gpu_runtime": (
                json.loads(os.environ["HOTPOTQA_GPU_RUNTIME"])
                if os.environ.get("HOTPOTQA_GPU_RUNTIME")
                else None
            ),
            "vllm_version": os.environ.get("HOTPOTQA_VLLM_VERSION"),
            "torch_version": os.environ.get("HOTPOTQA_TORCH_VERSION"),
            "cuda_version": os.environ.get("HOTPOTQA_CUDA_VERSION"),
            "cuda_module": os.environ.get("HOTPOTQA_CUDA_MODULE"),
            "transformers_version": os.environ.get("HOTPOTQA_TRANSFORMERS_VERSION"),
            "litellm_version": os.environ.get("HOTPOTQA_LITELLM_VERSION"),
            "model_revision": os.environ.get("HOTPOTQA_MODEL_REVISION"),
            "model_integrity_sha256": os.environ.get("HOTPOTQA_MODEL_INTEGRITY_SHA256"),
            "deepseek_response_model": os.environ.get(HOTPOTQA_DEEPSEEK_RESPONSE_MODEL_ENV),
            "deepseek_system_fingerprint": os.environ.get(
                HOTPOTQA_DEEPSEEK_SYSTEM_FINGERPRINT_ENV
            ),
            "weight_dtype": os.environ.get("HOTPOTQA_WEIGHT_DTYPE"),
            "kv_cache_dtype": os.environ.get("HOTPOTQA_KV_CACHE_DTYPE"),
            "vllm_serve_arguments": os.environ.get("HOTPOTQA_VLLM_SERVE_ARGUMENTS"),
            "vllm_batch_invariant": os.environ.get("HOTPOTQA_VLLM_BATCH_INVARIANT"),
            "vllm_single_sequence_replicas": os.environ.get(
                "HOTPOTQA_VLLM_SINGLE_SEQUENCE_REPLICAS"
            ),
        },
        "tag": args.tag,
    }


def _run_key(condition: str, args) -> str:
    """Fingerprint the material settings of one HotPotQA run.

    Args:
        condition: Optimization condition represented by the key.
        args: Parsed experiment arguments used to build the run contract.

    Returns:
        Stable compatibility key for the output directory.
    """
    family = resolve_template_family(args.template_family, args.solver_model)
    contract = build_run_contract(condition, args)
    run_key = experiment_run_key(
        condition=condition,
        template_family=family,
        reflection_level=contract["optimizer"]["reflection_level"],
        edit_tool_set=contract["optimizer"]["edit_tool_set"] or "none",
        settings=contract,
    )
    if args.merge:
        return f"merge-{run_key}"
    return run_key


def seed_candidate(
    program: str = "2stage",
    seed_style: str = "plain",
    template_family: str = "generic",
) -> dict:
    """Build the initial prompt components for a program variant.

    Args:
        program: Single-stage or two-stage HotPotQA program.
        seed_style: ``"plain"`` or provider-structured seed rendering.
        template_family: Provider family used for structured seeds.

    Returns:
        Independent seed text for every optimized component.
    """
    seed = dict(SEED_CANDIDATE_1STAGE if program == "1stage" else SEED_CANDIDATE)
    if seed_style == "structured":
        seed = {
            component: structured_prompt(text, template_family, "system_prompt") for component, text in seed.items()
        }
    return seed


def run_program(
    candidate: dict,
    question: str,
    program: str,
    model: str,
    api_base: str | None,
    retriever: WikipediaRetriever,
    retrieval_k: int,
    task_lm: object | None = None,
    lm_kwargs: dict[str, object] | None = None,
) -> tuple[str | None, str, dict[str, object]]:
    """Run a candidate and retain its complete component trace.

    Args:
        candidate: Current prompt component mapping, or a legacy prompt string.
        question: HotPotQA question to answer.
        program: Single-stage or two-stage execution path.
        model: Solver model identifier.
        api_base: Optional solver API endpoint.
        retriever: Wikipedia passage retriever.
        retrieval_k: Passages requested for each retrieval hop.
        task_lm: Shared DSPy task-model client for the two-stage program.
        lm_kwargs: Fully resolved solver request settings.

    Returns:
        Generated second-hop query when present, final answer, and execution
        trace used to build component-specific feedback.
    """
    if isinstance(candidate, str):
        answer = run_single_stage(
            candidate,
            question,
            retriever,
            model=model,
            api_base=api_base,
            retrieval_k=retrieval_k,
            lm_kwargs=lm_kwargs,
        )
        return None, answer, {"answer": answer}
    if program == "1stage":
        prompt = candidate.get("answer_question") or next(iter(candidate.values()))
        answer = run_single_stage(
            prompt,
            question,
            retriever,
            model=model,
            api_base=api_base,
            retrieval_k=retrieval_k,
            lm_kwargs=lm_kwargs,
        )
        return None, answer, {"answer": answer}
    query, answer, trace = run_two_stage(
        candidate.get("summarize1", ""),
        candidate.get("create_query_hop2", ""),
        candidate.get("summarize2", ""),
        candidate.get("final_answer", ""),
        question,
        retriever,
        model=model,
        api_base=api_base,
        retrieval_k=retrieval_k,
        task_lm=task_lm,
        lm_kwargs=lm_kwargs,
    )
    return query, answer, trace


def _is_task_output_parse_error(error: ValueError) -> bool:
    """Recognize the pinned DSPy adapter's structured-output failure.

    Args:
        error: Candidate execution error raised by the task program.

    Returns:
        Whether the error is the narrow malformed-task-output case that the
        artifact scores as zero rather than a systemic execution failure.
    """
    if not error.args:
        return False
    message = str(error.args[0])
    return message.startswith("Failed to parse response as per signature")


def make_evaluator(
    solver_model: str,
    retriever: WikipediaRetriever,
    api_base: str | None = None,
    program: str = "2stage",
    retrieval_k: int = 7,
    solver_lm_kwargs: dict[str, object] | None = None,
):
    """Create a HotPotQA evaluator closed over solver and retrieval settings.

    Args:
        solver_model: Model used to execute candidate prompts.
        retriever: Wikipedia passage retriever.
        api_base: Optional solver API endpoint.
        program: Single-stage or two-stage execution path.
        retrieval_k: Passages requested for each retrieval hop.
        solver_lm_kwargs: Fully resolved solver request settings.

    Returns:
        Evaluator accepted by ``optimize_anything``.
    """
    task_lm = build_hotpotqa_task_lm(solver_model, api_base, solver_lm_kwargs) if program == "2stage" else None

    def evaluate(candidate, example: dict) -> tuple[float, SideInfo]:
        """Score one candidate on one question and retain reflection evidence.

        Args:
            candidate: Prompt components being evaluated.
            example: Question and reference answer record.

        Returns:
            Primary score and execution details used for reflection.

        Raises:
            ValueError: Candidate execution fails for a reason other than
                DSPy's pinned output-field parser rejecting the task response.
            RuntimeError: Candidate execution encounters a systemic task,
                retrieval, or model-provider failure.
        """
        try:
            _query, prediction, trace = run_program(
                candidate,
                example["question"],
                program,
                solver_model,
                api_base,
                retriever,
                retrieval_k,
                task_lm,
                solver_lm_kwargs,
            )
        except ValueError as exc:
            if not _is_task_output_parse_error(exc):
                raise
            return 0.0, {
                "evaluation_error": {
                    "type": "task_output_parse_error",
                    "message": "Task-model output omitted DSPy's required structured fields; this example scored 0.",
                }
            }

        score, feedback = hotpotqa_metric(prediction, example["answer"])
        if program == "1stage":
            side_info: SideInfo = {
                "answer_question_specific_info": {
                    "Inputs": {"question": example["question"]},
                    "Generated Outputs": {"answer": prediction},
                    "Feedback": feedback,
                }
            }
        else:
            records = artifact_component_records(example, trace, score)
            side_info = {f"{component}_specific_info": record for component, record in records.items()}
        return score, side_info

    return evaluate


def evaluate_on_set(
    candidate,
    dataset: list[dict],
    solver_model: str,
    retriever: WikipediaRetriever,
    api_base: str | None = None,
    max_workers: int = 32,
    program: str = "2stage",
    retrieval_k: int = 7,
    solver_lm_kwargs: dict[str, object] | None = None,
    checkpoint_dir: str | Path | None = None,
) -> tuple[float, float]:
    """Evaluate a candidate on a dataset, returning mean exact match and F1.

    Args:
        candidate: Prompt components being evaluated.
        dataset: Question and answer records.
        solver_model: Model used to execute candidate prompts.
        retriever: Wikipedia passage retriever.
        api_base: Optional solver API endpoint.
        max_workers: Maximum concurrent examples.
        program: Single-stage or two-stage execution path.
        retrieval_k: Passages requested for each retrieval hop.
        solver_lm_kwargs: Fully resolved solver request settings.
        checkpoint_dir: Optional directory for atomic per-example predictions
            and an aggregate summary. Existing records for the exact candidate
            are reused after an interrupted held-out evaluation.

    Returns:
        Mean exact-match and token-F1 scores, or zeros for an empty dataset.
    """
    task_lm = build_hotpotqa_task_lm(solver_model, api_base, solver_lm_kwargs) if program == "2stage" else None

    candidate_payload = json.dumps(candidate, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    candidate_sha256 = hashlib.sha256(candidate_payload.encode()).hexdigest()
    checkpoint_root = Path(checkpoint_dir) / candidate_sha256 if checkpoint_dir is not None else None
    if checkpoint_root is not None:
        checkpoint_root.mkdir(parents=True, exist_ok=True)

    def score_one(index_and_example: tuple[int, dict]) -> tuple[float, float]:
        """Run and score one HotPotQA example.

        Args:
            index_and_example: Stable dataset position and question record.

        Returns:
            Exact-match and token-F1 scores.

        Raises:
            ValueError: A persisted record is malformed or candidate execution
                fails outside the expected DSPy output-parser case.
        """
        index, example = index_and_example
        record_path = None
        if checkpoint_root is not None:
            id_digest = hashlib.sha256(str(example.get("id", "")).encode()).hexdigest()[:12]
            record_path = checkpoint_root / f"{index:04d}-{id_digest}.json"
            if record_path.is_file():
                record = json.loads(record_path.read_text(encoding="utf-8"))
                if record.get("candidate_sha256") != candidate_sha256 or record.get("id") != str(example.get("id", "")):
                    raise ValueError(f"Held-out checkpoint identity mismatch at {record_path}.")
                return float(record["exact_match"]), float(record["f1"])

        prediction = None
        parse_error = False
        try:
            _, prediction, _trace = run_program(
                candidate,
                example["question"],
                program,
                solver_model,
                api_base,
                retriever,
                retrieval_k,
                task_lm,
                solver_lm_kwargs,
            )
        except ValueError as exc:
            if not _is_task_output_parse_error(exc):
                raise
            parse_error = True
        if parse_error:
            exact_match = 0.0
            f1 = 0.0
        else:
            assert prediction is not None
            exact_match = float(normalize_answer(prediction) == normalize_answer(example["answer"]))
            f1 = f1_score(prediction, example["answer"])

        if record_path is not None:
            record = {
                "schema_version": 1,
                "candidate_sha256": candidate_sha256,
                "id": str(example.get("id", "")),
                "prediction": prediction,
                "exact_match": exact_match,
                "f1": f1,
                "task_output_parse_error": parse_error,
            }
            temporary_path = record_path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.part")
            temporary_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary_path.replace(record_path)
        return exact_match, f1

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        scores = list(pool.map(score_one, enumerate(dataset)))
    if not scores:
        return 0.0, 0.0
    mean_em = sum(s[0] for s in scores) / len(scores)
    mean_f1 = sum(s[1] for s in scores) / len(scores)
    if checkpoint_root is not None:
        summary = {
            "schema_version": 1,
            "candidate_sha256": candidate_sha256,
            "example_count": len(scores),
            "exact_match": mean_em,
            "f1": mean_f1,
        }
        summary_path = checkpoint_root / "summary.json"
        temporary_path = summary_path.with_suffix(f".{os.getpid()}.part")
        temporary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary_path.replace(summary_path)
    return mean_em, mean_f1


def prompt_diversity(candidates: list[dict]) -> dict[str, dict[str, float]]:
    """Measure textual diversity of explored candidates per component.

    Args:
        candidates: Explored prompt mappings. Legacy string values are
            normalized when encountered.

    Returns:
        Mean pairwise token-set distance and unique-text count per component.
    """
    if not candidates:
        return {}
    # Normalize string candidates to dict for diversity
    norm = []
    for c in candidates:
        if isinstance(c, str):
            norm.append({"prompt": c})
        else:
            norm.append(c)
    if not norm:
        return {}
    diversity: dict[str, dict[str, float]] = {}
    for component in norm[0]:
        texts = [c[component] for c in norm]
        token_sets = [set(t.lower().split()) for t in texts]
        distances = []
        for a, b in itertools.combinations(token_sets, 2):
            union = a | b
            distances.append(1.0 - (len(a & b) / len(union)) if union else 0.0)
        diversity[component] = {
            "mean_pairwise_jaccard_distance": sum(distances) / len(distances) if distances else 0.0,
            "num_unique_texts": float(len(set(texts))),
        }
    return diversity


def dump_candidates(result, run_dir: str, run_contract: dict) -> str:
    """Write explored candidates, lineage, scores, and contract to JSON.

    Args:
        result: Completed GEPA result containing candidate history.
        run_dir: Directory that receives ``candidates.json``.
        run_contract: Material configuration attached to the artifact.

    Returns:
        Path to the written candidate artifact.
    """
    payload = {
        "run_contract": run_contract,
        "best_idx": result.best_idx,
        "total_metric_calls": result.total_metric_calls,
        "num_full_val_evals": result.num_full_val_evals,
        "candidates": result.candidates,
        "parents": result.parents,
        "val_aggregate_scores": result.val_aggregate_scores,
        "discovery_eval_counts": result.discovery_eval_counts,
    }
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "candidates.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def dump_action_summary(tracker: ActionDiversityCallback, run_dir: str, selector=None) -> str:
    """Persist aggregate and per-action diversity evidence.

    Args:
        tracker: Callback containing action outcomes and generated text.
        run_dir: Directory that receives ``action_summary.json``.
        selector: Optional verbalized selector whose history is included.

    Returns:
        Path to the written action summary.
    """
    payload = {
        "summary": tracker.summary(),
        "action_score_deltas": dict(tracker.action_score_deltas),
        "action_texts": dict(tracker.action_texts),
    }
    if selector is not None and getattr(selector, "history", None):
        payload["verbalized_history"] = selector.history
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "action_summary.json")
    temporary_path = f"{path}.{os.getpid()}.part"
    with open(temporary_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(temporary_path, path)
    return path


def build_config(condition: str, args, reflection_lm_kwargs: dict, run_dir: str | None = None):
    """Build optimizer configuration and any stateless action selector.

    Args:
        condition: Optimization condition to configure.
        args: Parsed experiment arguments.
        reflection_lm_kwargs: Reflection-model client settings.
        run_dir: Optional explicit output directory.

    Returns:
        GEPA configuration and the condition's optional action selector.
    """
    resolved_family = resolve_template_family(args.template_family, args.solver_model)
    reflection_runtime_model = resolve_experiment_model(args.reflection_model, args.api_profile)
    runtime_profile = _validated_runtime_profile(args)
    resolved_run_dir = run_dir or condition_run_dir(condition, args.program, args.tag, _run_key(condition, args))
    response_journal_path = os.path.join(resolved_run_dir, ".lm-response-journal", "responses.sqlite3")
    reflection_proposer_kwargs = deepcopy(reflection_lm_kwargs or {})
    reflection_proposer_kwargs["response_journal_path"] = response_journal_path
    reflection_proposer_kwargs["response_journal_namespace"] = "reflection-proposer"
    template = TEMPLATE_FAMILIES[resolved_family]["system_prompt"]
    action_space = [
        StatelessActionConstraint(spec, section, template) for section in template.sections for spec in SEMANTIC_ACTIONS
    ]
    action_selector = None
    if condition == "random":
        action_selector = RandomActionSelector(action_space)
    elif condition == "action":
        action_selector_kwargs = deepcopy(reflection_lm_kwargs or {})
        action_selector_kwargs["response_journal_path"] = response_journal_path
        action_selector_kwargs["response_journal_namespace"] = "stateless-controller"
        action_selector = VerbalizedActionSelector(
            action_space,
            lm=LM(reflection_runtime_model, **action_selector_kwargs),
        )

    reflection_strategy = None
    if condition in _REACT_V2_CONDITIONS:
        react_v2_kwargs = deepcopy(reflection_lm_kwargs or {})
        react_v2_kwargs["response_journal_path"] = response_journal_path
        reflection_strategy, _ = build_react_v2_strategy(
            reflection_model=reflection_runtime_model,
            task_model=args.solver_model,
            proposer_model=args.reflection_model,
            lm_kwargs=react_v2_kwargs,
            level=args.reflection_level,
            edit_tool_set=args.edit_tool_set,
            template_family=args.template_family,
            component_kinds=_component_kinds(args.program),
            controller_selection="uniform_random" if condition == "react_v2_random" else "verbalized",
            rng=random.Random(args.seed),
        )

    merge_config = None
    if args.merge:
        merge_config = MergeConfig(
            max_merge_invocations=_PAPER_MAX_MERGE_INVOCATIONS,
            merge_val_overlap_floor=_PAPER_MERGE_VAL_OVERLAP_FLOOR,
        )

    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=resolved_run_dir,
            seed=args.seed,
            max_metric_calls=args.max_metric_calls,
            val_evaluation_policy="full_eval",
            candidate_selection_strategy="pareto",
            sampling_strategy=SingleMutationSampling(),
            selection_strategy=AllImprovements(),
            frontier_type="instance",
            acceptance_criterion="strict_improvement",
            raise_on_exception=True,
            parallel=True,
            max_workers=args.max_workers,
            cache_evaluation=True,
        ),
        reflection=ReflectionConfig(
            skip_perfect_score=runtime_profile != "technical-smoke",
            perfect_score=1.0,
            batch_sampler="epoch_shuffled",
            reflection_minibatch_size=3,
            module_selector="round_robin",
            reflection_lm=reflection_runtime_model,
            reflection_lm_kwargs=reflection_proposer_kwargs,
            reflection_strategy=reflection_strategy,
            reflection_prompt_template=InstructionProposalSignature.default_prompt_template,
            action_selector=action_selector,
        ),
        merge=merge_config,
    )
    return config, action_selector


def run_condition(
    name: str,
    seed: dict | str,
    trainset: list[dict],
    valset: list[dict],
    config: GEPAConfig,
    evaluator,
    callbacks: list | None = None,
):
    """Run one optimization condition and return its GEPA result.

    Args:
        name: Human-readable condition label printed before execution.
        seed: Initial prompt or prompt-component mapping.
        trainset: Records used for candidate discovery.
        valset: Records used for validation.
        config: GEPA engine and reflection configuration.
        evaluator: Candidate evaluation callable.
        callbacks: Optional callbacks attached before optimization.

    Returns:
        Completed result from ``optimize_anything``.
    """
    print(f"\n{'=' * 60}")
    print(f"  Running: {name}")
    print(f"{'=' * 60}\n")

    if callbacks:
        config.callbacks = callbacks

    result = optimize_anything(
        seed_candidate=seed,
        evaluator=evaluator,
        dataset=trainset,
        valset=valset,
        config=config,
    )

    return result


def _verify_scientific_retriever_integrity(retriever: Wiki17BM25Retriever) -> None:
    """Verify Wiki-2017 once while preserving a fail-closed production handoff.

    The Slurm entry point deep-hashes the corpus and every index file while it
    holds the shared artifact lock. It then passes the integrity-manifest
    digest into this process. Rechecking that small manifest avoids hashing the
    complete 1.78 GB artifact again after the H200 server has started. Direct
    invocations retain the original deep verification.

    Args:
        retriever: Frozen Wiki-2017 retriever selected for the scientific run.

    Raises:
        ValueError: A production attestation is malformed, missing its locked
            manifest, or does not match the manifest bytes.
    """
    verified_integrity_sha256 = None
    if os.environ.get("HOTPOTQA_PRODUCTION_LAUNCH") == "1":
        verified_integrity_sha256 = os.environ.get(
            "HOTPOTQA_VERIFIED_WIKI17_INTEGRITY_SHA256"
        )
    if verified_integrity_sha256 is None:
        retriever.verify_integrity()
        return
    if len(verified_integrity_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in verified_integrity_sha256
    ):
        raise ValueError("The production Wiki-2017 integrity attestation is malformed.")
    try:
        manifest_sha256 = file_sha256(retriever.integrity_path)
    except OSError as exc:
        raise ValueError("The locked Wiki-2017 integrity manifest is unavailable.") from exc
    if manifest_sha256 != verified_integrity_sha256:
        raise ValueError(
            "The production Wiki-2017 integrity attestation does not match the locked manifest."
        )


def main():
    """Parse CLI arguments and run the requested HotPotQA conditions.

    The default scientific profile preserves the benchmark configuration. The
    explicit technical-smoke profile is restricted to OpenRouter and the
    non-scientific selected-context mini index.
    """
    parser = argparse.ArgumentParser(description="HotpotQA evaluation for action-conditioned reflection")
    parser.add_argument("--data-path", type=str, default=None, help="Path to HotpotQA JSONL sample (smoke, 14/3/3)")
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=6871,
        help="Budget per condition (paper: 6871, smoke: 200, two-times compute: 13742)",
    )
    parser.add_argument(
        "--solver-model",
        type=str,
        default=QWEN3_8_27B_MODEL,
        help="Student model; use the same supported model as --reflection-model",
    )
    parser.add_argument(
        "--reflection-model",
        type=str,
        default=QWEN3_8_27B_MODEL,
        help="Proposer model; use the same supported model as --solver-model",
    )
    parser.add_argument(
        "--api-base", type=str, default=None, help="Base URL for vLLM server (e.g. http://localhost:8000/v1)"
    )
    parser.add_argument("--solver-api-base", type=str, default=None, help="Base URL used only by the student/solver LM")
    parser.add_argument(
        "--reflection-api-base", type=str, default=None, help="Base URL used only by the reflection/proposer LM"
    )
    parser.add_argument(
        "--api-profile",
        choices=["direct", "openrouter"],
        default="direct",
        help="API route for both model roles; OpenRouter uses fixed provider endpoints",
    )
    parser.add_argument(
        "--runtime-profile",
        choices=HOTPOTQA_RUNTIME_PROFILES,
        default="scientific",
        help="Request budget profile; technical-smoke is only for the OpenRouter mini-index integration run",
    )
    parser.add_argument(
        "--enforce-scientific-contract",
        action="store_true",
        help=(
            "Fail unless the paper-aligned data, retrieval, task program, seed, templates, merge setting, "
            "and approved standard- or expanded-budget campaign cell are selected"
        ),
    )
    parser.add_argument(
        "--wiki17-dir",
        type=Path,
        default=DEFAULT_WIKI17_ROOT,
        help="Prepared frozen Wiki-2017 corpus and BM25S index directory",
    )
    parser.add_argument(
        "--technical-mini-index",
        action="store_true",
        help=(
            "Use an explicitly non-scientific BM25 index built from the selected records' contexts; "
            "intended only to verify the complete code path on a small host"
        ),
    )
    parser.add_argument(
        "--technical-mini-index-dir",
        type=Path,
        default=DEFAULT_HOTPOTQA_TECHNICAL_MINI_ROOT,
        help="Cache directory for the selected-context technical-mini BM25 index",
    )
    parser.add_argument(
        "--retrieval-k", type=int, default=7, help="Wiki-2017 abstracts retrieved per hop (artifact: 7)"
    )
    parser.add_argument("--max-workers", type=int, default=32, help="Parallel evaluator workers (artifact: 32)")
    parser.add_argument("--train-limit", type=int, default=None, help="Limit train-set size (paper: 150)")
    parser.add_argument("--val-limit", type=int, default=None, help="Limit val-set size (paper: 300)")
    parser.add_argument("--test-limit", type=int, default=None, help="Limit test-set size (paper: 300)")
    parser.add_argument("--seed", type=int, default=0, help="Experiment seed (artifact: 0)")
    parser.add_argument(
        "--program",
        type=str,
        default="2stage",
        choices=["2stage", "1stage"],
        help="Program structure: 2stage (two retrieval hops, four artifact components) or 1stage ablation",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="both",
        choices=["vanilla", "react_v2", "react_v2_random", "random", "action", "all", "both"],
        help="Which condition(s) to run",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Enable the paper's five-invocation GEPA merge configuration for every selected condition",
    )
    parser.add_argument(
        "--seed-style",
        type=str,
        default="structured",
        choices=["plain", "structured"],
        help="Seed prompts: plain paper sentences or the template selected for the solver provider",
    )
    parser.add_argument(
        "--reflection-level",
        type=int,
        default=2,
        choices=[1, 2],
        help="Reflection level: 1 selects a section; 2 also selects and applies a semantic action",
    )
    parser.add_argument(
        "--edit-tool-set",
        choices=["minimal", "broad"],
        default="broad",
        help="Edit tools: insert/delete only, or insert/delete/replace/move",
    )
    parser.add_argument(
        "--template-family",
        choices=["auto", "generic", "openai", "anthropic", "google", "alibaba"],
        default="auto",
        help="Prompt template family; auto selects one from the student/solver model",
    )
    parser.add_argument("--tag", type=str, default="", help="Suffix appended to run dirs (e.g. rev2, 6871)")
    args = parser.parse_args()
    try:
        validate_experiment_model_pair(args.solver_model, args.reflection_model)
        _validated_runtime_profile(args)
    except ValueError as exc:
        parser.error(str(exc))

    trainset, valset, testset = load_hotpotqa_dataset(
        data_path=args.data_path,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        test_limit=args.test_limit,
        seed=args.seed,
    )
    if args.data_path is None:
        data_source = {
            "type": "huggingface",
            "dataset": "hotpot_qa",
            "config": "fullwiki",
            "revision": HOTPOTQA_HF_REVISION,
            "source_split": "train",
            "split_policy": "ordered-40-40-20-then-independent-seed1-sampling",
            "experiment_seed": args.seed,
        }
    else:
        data_path = Path(args.data_path).expanduser().resolve()
        data_source = {"type": "jsonl", "path": str(data_path), "sha256": file_sha256(data_path)}
    args.data_identity = benchmark_data_identity(
        source=data_source,
        trainset=trainset,
        valset=valset,
        testset=testset,
    )
    try:
        _validate_scientific_data_identity(args)
    except ValueError as exc:
        parser.error(str(exc))

    print(
        f"Loaded {len(trainset)} train / {len(valset)} val / {len(testset)} test examples ({args.program}, {args.seed_style})"
    )
    if args.data_path:
        print(f"  (explicit smoke data from {args.data_path}; gold passages are feedback-only)")
    else:
        print("  (GEPA artifact split from hotpot_qa/fullwiki train; paper Table 1: 150/300/300)")

    if args.technical_mini_index:
        retriever = HotPotQATechnicalMiniBM25Retriever(
            [*trainset, *valset, *testset],
            args.technical_mini_index_dir,
        )
        retriever.prepare()
        args.retrieval_provenance = retriever.provenance()
        print(
            "  (retrieval: NON-SCIENTIFIC technical-mini BM25S index built from selected benchmark contexts; "
            f"{args.retrieval_provenance['document_count']} documents, "
            f"k1={args.retrieval_provenance['k1']}, b={args.retrieval_provenance['b']})"
        )
    else:
        retriever = Wiki17BM25Retriever(args.wiki17_dir)
        if args.enforce_scientific_contract:
            try:
                _verify_scientific_retriever_integrity(retriever)
            except ValueError as exc:
                parser.error(str(exc))
        args.retrieval_provenance = retriever.provenance()
        if args.enforce_scientific_contract and not args.retrieval_provenance["integrity_manifest_sha256"]:
            parser.error("The enforced HotPotQA scientific contract requires a verified Wiki-2017 integrity manifest.")
        print(
            f"  (retrieval: frozen Wiki-2017 BM25S {args.retrieval_provenance['bm25s_version']}, "
            f"k1={args.retrieval_provenance['k1']}, b={args.retrieval_provenance['b']})"
        )
    if trainset:
        warm_passages = retriever.search(trainset[0]["question"], args.retrieval_k)
        if len(warm_passages) != args.retrieval_k:
            parser.error(
                f"Retriever preflight returned {len(warm_passages)} passages; expected {args.retrieval_k}."
            )
        print(f"  (retrieval preflight: {len(warm_passages)} passages for the first training question)")
    solver_api_base = args.solver_api_base if args.solver_api_base is not None else args.api_base
    reflection_api_base = args.reflection_api_base if args.reflection_api_base is not None else args.api_base
    solver_runtime_model = resolve_experiment_model(args.solver_model, args.api_profile)
    reflection_runtime_model = resolve_experiment_model(args.reflection_model, args.api_profile)
    solver_lm_kwargs = resolve_hotpotqa_lm_kwargs(
        solver_runtime_model,
        solver_api_base,
        args.runtime_profile,
    )
    reflection_lm_kwargs = resolve_hotpotqa_lm_kwargs(
        reflection_runtime_model,
        reflection_api_base,
        args.runtime_profile,
    )
    evaluator = make_evaluator(
        solver_runtime_model,
        retriever,
        api_base=solver_api_base,
        program=args.program,
        retrieval_k=args.retrieval_k,
        solver_lm_kwargs=solver_lm_kwargs,
    )

    if args.condition == "all" and args.enforce_scientific_contract:
        conditions = list(_SCIENTIFIC_CONDITIONS_BY_BUDGET[args.max_metric_calls])
    elif args.condition == "all":
        conditions = ["vanilla", "random", "action", "react_v2_random", "react_v2"]
    elif args.condition == "both":
        conditions = ["vanilla", "react_v2"]
    else:
        conditions = [args.condition]

    resolved_family = resolve_template_family(args.template_family, args.solver_model)
    semantic_conditions = _SEMANTIC_CONDITIONS.intersection(conditions)
    if semantic_conditions and args.seed_style != "structured":
        parser.error(f"--condition {', '.join(sorted(semantic_conditions))} requires --seed-style structured")

    results = {}
    run_dirs: dict[str, str] = {}
    trackers: dict[str, ActionDiversityCallback] = {}
    for condition in conditions:
        run_contract = build_run_contract(condition, args)
        run_dir = condition_run_dir(condition, args.program, args.tag, _run_key(condition, args))
        run_dirs[condition] = run_dir
        ensure_wikipedia_run_contract(run_dir, run_contract)
        config, selector = build_config(condition, args, reflection_lm_kwargs, run_dir=run_dir)
        callbacks = None
        if condition in _SEMANTIC_CONDITIONS:
            trackers[condition] = ActionDiversityCallback(selector=selector)
            callbacks = [trackers[condition]]
        seed = seed_candidate(args.program, args.seed_style, resolved_family)
        condition_label = _CONDITION_LABELS[condition]
        if args.merge:
            condition_label = f"{condition_label}+Merge"
        condition_result = run_condition(
            f"{condition_label} ({args.program}, {args.seed_style} seeds)",
            seed,
            trainset,
            valset,
            config,
            evaluator,
            callbacks=callbacks,
        )
        if args.enforce_scientific_contract and (
            not isinstance(condition_result.total_metric_calls, int)
            or condition_result.total_metric_calls < args.max_metric_calls
        ):
            raise RuntimeError(
                f"Scientific HotPotQA condition {condition!r} stopped after "
                f"{condition_result.total_metric_calls!r} metric calls; "
                f"the locked budget is {args.max_metric_calls}. Resubmit the same commit and campaign to resume."
            )
        results[condition] = condition_result
        path = dump_candidates(results[condition], run_dir, run_contract)
        print(f"[{condition}] wrote {path}")
        if condition in trackers:
            path = dump_action_summary(trackers[condition], run_dir, selector=selector)
            print(f"[{condition}] wrote {path}")

    # Report: best prompts (full text)
    print(f"\n{'=' * 60}")
    print("  Best prompts")
    print(f"{'=' * 60}")
    for name, result in results.items():
        print(f"\n----- [{name}] best candidate (val score {result.val_aggregate_scores[result.best_idx]:.4f}) -----")
        cand = result.best_candidate
        if isinstance(cand, str):
            print(cand)
        else:
            for component, text in cand.items():
                print(f"\n[{name}] {component}:\n{text}")

    # Report: test EM/F1 + diversity
    print(f"\n{'=' * 60}")
    print("  Comparison")
    print(f"{'=' * 60}\n")

    for name, result in results.items():
        test_em, test_f1 = evaluate_on_set(
            result.best_candidate,
            testset,
            solver_runtime_model,
            retriever,
            api_base=solver_api_base,
            max_workers=args.max_workers,
            program=args.program,
            retrieval_k=args.retrieval_k,
            solver_lm_kwargs=solver_lm_kwargs,
            checkpoint_dir=Path(run_dirs[name]) / "heldout",
        )
        diversity = prompt_diversity(result.candidates)
        final_metrics = {
            "schema_version": 1,
            "condition": name,
            "candidate_sha256": hashlib.sha256(
                json.dumps(
                    result.best_candidate,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "candidates_explored": len(result.candidates),
            "best_validation_exact_match": float(result.val_aggregate_scores[result.best_idx]),
            "test_exact_match": float(test_em),
            "test_f1": float(test_f1),
            "test_example_count": len(testset),
            "diversity": diversity,
        }
        final_metrics_path = Path(run_dirs[name]) / "final_metrics.json"
        temporary_metrics_path = final_metrics_path.with_suffix(f".{os.getpid()}.part")
        temporary_metrics_path.write_text(
            json.dumps(final_metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_metrics_path.replace(final_metrics_path)
        print(f"[{name}]")
        print(f"  candidates explored:      {len(result.candidates)}")
        print(f"  best val score (EM):      {result.val_aggregate_scores[result.best_idx]:.4f}")
        print(f"  test EM:                  {test_em:.2%}")
        print(f"  test F1:                  {test_f1:.2%}")
        print(f"  final metrics:            {final_metrics_path}")
        for component, stats in diversity.items():
            print(
                f"  diversity[{component}]: jaccard_dist={stats['mean_pairwise_jaccard_distance']:.3f} "
                f"unique={int(stats['num_unique_texts'])}/{len(result.candidates)}"
            )
        print()

    # Action diversity metrics (random / action conditions)
    for name, tracker in trackers.items():
        print(f"{'=' * 60}")
        print(f"  Action Diversity Metrics [{name}]")
        print(f"{'=' * 60}\n")
        summary = tracker.summary()
        print(f"Total proposals: {summary['total_proposals']}")
        print(f"Total accepted:  {summary['total_accepted']}")
        print(f"\nPer-action proposal counts: {summary['action_proposal_counts']}")
        print(f"Per-action acceptance rates: {summary['action_acceptance_rates']}")
        print(f"Textual diversity per iteration: {summary['textual_diversity_per_iteration']}\n")


if __name__ == "__main__":
    main()
