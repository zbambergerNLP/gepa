"""Tests for ReAct V2 wiring shared by the Wikipedia benchmarks."""

import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gepa.core.data_loader import ListDataLoader
from gepa.strategies.batch_sampler import IndependentEpochShuffledBatchSampler
from gepa.strategies.text_limits import TextLimits

sys.path.insert(0, str(Path(__file__).parents[1]))

from examples.common.experiment_models import (
    DEEPSEEK_V4_1_FLASH_MODEL,
    DEEPSEEK_V4_1_FLASH_REVISION,
    EXPERIMENT_NUM_RETRIES,
    QWEN3_8_27B_MODEL,
    QWEN3_8_27B_REVISION,
    experiment_decoding,
    experiment_model_version,
    experiment_request_overrides,
    validate_experiment_model_pair,
)
from examples.common.react_v2 import (
    build_react_v2_strategy,
    ensure_wikipedia_run_contract,
    file_sha256,
    resolve_template_family,
    structured_prompt,
)
from examples.hotpotqa.main import (
    _SCIENTIFIC_CONDITIONS_BY_BUDGET,
    _validate_scientific_contract,
    _validate_scientific_data_identity,
    _verify_scientific_retriever_integrity,
)
from examples.hotpotqa.main import (
    _run_key as hotpotqa_run_key,
)
from examples.hotpotqa.main import (
    build_config as build_hotpotqa_config,
)
from examples.hotpotqa.main import (
    build_run_contract as build_hotpotqa_run_contract,
)
from examples.hotpotqa.main import (
    dump_candidates as dump_hotpotqa_candidates,
)
from examples.hotpotqa.main import (
    run_condition as run_hotpotqa_condition,
)
from examples.hotpotqa.main import (
    seed_candidate as hotpotqa_seed_candidate,
)
from examples.hotpotqa.utils import HOTPOTQA_HF_REVISION, HOTPOTQA_SCIENTIFIC_SPLIT_SHA256, resolve_hotpotqa_lm_kwargs
from examples.hover.main import _run_key as hover_run_key
from examples.hover.main import build_config as build_hover_config
from examples.hover.main import build_run_contract as build_hover_run_contract
from examples.hover.main import dump_candidates as dump_hover_candidates
from examples.hover.main import seed_candidate as hover_seed_candidate
from gepa.strategies.action_space import (
    RandomActionSelector,
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

LOCAL_API_BASE = "http://127.0.0.1:8000/v1"


def h200_gpu_runtime(count: int) -> str:
    """Build the canonical hardware record for a model's allocated GPU count."""
    return json.dumps(
        {
            "compute_capabilities": ["9.0"] * count,
            "count": count,
            "driver_version": "580.82",
            "names": ["NVIDIA H200"] * count,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


QWEN_SERVE_ARGUMENTS = (
    "tp=1;gpu_memory_utilization=0.92;max_model_len=262144;rope_scaling=none;max_num_seqs=1;"
    "dtype=bfloat16;kv_cache_dtype=auto;prefix_caching=false;reasoning_parser=qwen3;"
    "auto_tool_choice=true;tool_parser=qwen3_coder;seed=0;batch_invariant=false;"
    "single_sequence_replicas=true"
)
DEEPSEEK_SERVE_ARGUMENTS = (
    "tp=4;ep=4;dp=1;api_servers=1;gpu_memory_utilization=0.92;max_model_len=262144;max_num_seqs=1;"
    "dtype=bfloat16;weight_dtype=fp8;expert_dtype=fp4;engram_cpu_offload=true;kv_cache_dtype=fp8;block_size=auto;prefix_caching=false;"
    "tokenizer_mode=deepseek_v41;reasoning_parser=deepseek_v41;auto_tool_choice=true;tool_parser=deepseek_v41;"
    "speculative_decoding=false;seed=0;batch_invariant=false;single_sequence_replicas=true"
)
COMMON_SCIENTIFIC_RUNTIME = {
    "HOTPOTQA_MODEL_INTEGRITY_SHA256": "c" * 64,
    "HOTPOTQA_TRANSFORMERS_VERSION": "5.8.0",
    "HOTPOTQA_SOURCE_COMMIT": "a" * 40,
    "HOTPOTQA_SOURCE_MANIFEST_SHA256": "e" * 64,
    "HOTPOTQA_PYTHON_VERSION": "3.11.13",
    "HOTPOTQA_UV_VERSION": "0.9.13",
    "HOTPOTQA_UV_SHA256": "f" * 64,
    "HOTPOTQA_LITELLM_VERSION": "1.80.0",
    "HOTPOTQA_CAMPAIGN_ID": "hotpotqa-final-v1",
    "HOTPOTQA_ENV_SPEC_SHA256": "d" * 64,
    "HOTPOTQA_GEPA_ENV_SHA256": "2" * 64,
}
QWEN_SCIENTIFIC_RUNTIME = {
    **COMMON_SCIENTIFIC_RUNTIME,
    "HOTPOTQA_GPU_RUNTIME": h200_gpu_runtime(1),
    "HOTPOTQA_MODEL_REVISION": QWEN3_8_27B_REVISION,
    "HOTPOTQA_WEIGHT_DTYPE": "bfloat16",
    "HOTPOTQA_KV_CACHE_DTYPE": "auto",
    "HOTPOTQA_SERVING_ENGINE": "vllm",
    "HOTPOTQA_VLLM_BATCH_INVARIANT": "false",
    "HOTPOTQA_VLLM_SINGLE_SEQUENCE_REPLICAS": "true",
    "HOTPOTQA_VLLM_VERSION": "0.17.0",
    "HOTPOTQA_SERVING_LOCK_SHA256": "b" * 64,
    "HOTPOTQA_SERVING_ENV_SHA256": "1" * 64,
    "HOTPOTQA_SERVE_ARGUMENTS": QWEN_SERVE_ARGUMENTS,
}
DEEPSEEK_SCIENTIFIC_RUNTIME = {
    **COMMON_SCIENTIFIC_RUNTIME,
    "HOTPOTQA_GPU_RUNTIME": h200_gpu_runtime(4),
    "HOTPOTQA_MODEL_REVISION": DEEPSEEK_V4_1_FLASH_REVISION,
    "HOTPOTQA_WEIGHT_DTYPE": "fp8",
    "HOTPOTQA_KV_CACHE_DTYPE": "fp8",
    "HOTPOTQA_SERVING_ENGINE": "vllm",
    "HOTPOTQA_VLLM_VERSION": "0.1.1.dev5+ge77daef89",
    "HOTPOTQA_SERVING_LOCK_SHA256": "b" * 64,
    "HOTPOTQA_SERVING_ENV_SHA256": "1" * 64,
    "HOTPOTQA_VLLM_BATCH_INVARIANT": "false",
    "HOTPOTQA_VLLM_SINGLE_SEQUENCE_REPLICAS": "true",
    "HOTPOTQA_SERVE_ARGUMENTS": DEEPSEEK_SERVE_ARGUMENTS,
}


def test_student_model_selects_provider_specific_template() -> None:
    """Infer Alibaba prompt structure from the Qwen student identifier."""
    assert resolve_template_family("auto", QWEN3_8_27B_MODEL) == "alibaba"


def test_structured_benchmark_seeds_follow_selected_template() -> None:
    """Render structured benchmark seeds only into Alibaba's objective section."""
    template = TEMPLATE_FAMILIES["alibaba"]["system_prompt"]
    prompt = structured_prompt("Retrieve two evidence hops.", "alibaba")

    assert template.parse(prompt)["Objective"] == "Retrieve two evidence hops."
    assert [line for line in prompt.splitlines() if line.startswith("## ")] == ["## Objective"]
    assert all(
        template.parse(text)["Objective"]
        for text in hotpotqa_seed_candidate("2stage", "structured", "alibaba").values()
    )
    assert all(
        template.parse(text)["Objective"] for text in hover_seed_candidate("2stage", "structured", "alibaba").values()
    )
    for text in hotpotqa_seed_candidate("2stage", "structured", "alibaba").values():
        bodies = template.parse(text)
        assert bodies["Objective"]
        assert all(not body for section, body in bodies.items() if section != "Objective")
        assert "not specified" not in text.lower()


def test_hotpot_seed_instructions_match_the_artifact_signatures() -> None:
    """Lock the four initial instructions generated by DSPy signatures."""
    assert hotpotqa_seed_candidate("2stage", "plain", "generic") == {
        "summarize1": "Given the fields `question`, `passages`, produce the fields `summary`.",
        "create_query_hop2": "Given the fields `question`, `summary_1`, produce the fields `query`.",
        "summarize2": "Given the fields `question`, `context`, `passages`, produce the fields `summary`.",
        "final_answer": "Given the fields `question`, `summary_1`, `summary_2`, produce the fields `answer`.",
    }


def test_hover_seed_instructions_match_the_artifact_signatures() -> None:
    """Lock the four initial HoVer instructions generated by DSPy signatures."""
    assert hover_seed_candidate("2stage", "plain", "generic") == {
        "summarize1": "Given the fields `claim`, `passages`, produce the fields `summary`.",
        "create_query_hop2": "Given the fields `claim`, `summary_1`, produce the fields `query`.",
        "summarize2": "Given the fields `claim`, `context`, `passages`, produce the fields `summary`.",
        "create_query_hop3": "Given the fields `claim`, `summary_1`, `summary_2`, produce the fields `query`.",
    }


@pytest.mark.parametrize(
    ("family", "component_kind", "task_section"),
    [
        ("generic", "system_prompt", "Task"),
        ("generic", "user_prompt", "Task"),
        ("openai", "system_prompt", "Instructions"),
        ("openai", "user_prompt", "Input"),
        ("anthropic", "system_prompt", "Instructions"),
        ("anthropic", "user_prompt", "Instructions"),
        ("google", "system_prompt", "Instructions"),
        ("google", "user_prompt", "Task"),
        ("alibaba", "system_prompt", "Objective"),
        ("alibaba", "user_prompt", "Objective"),
    ],
)
def test_structured_prompt_populates_only_the_role_specific_task_section(
    family: str,
    component_kind: str,
    task_section: str,
) -> None:
    """Keep seed content in the correct message-role schema without placeholders.

    Args:
        family: Provider template family under test.
        component_kind: System or user message role.
        task_section: Section expected to receive seed text.
    """
    template = TEMPLATE_FAMILIES[family][component_kind]
    prompt = structured_prompt("Do the task.", family, component_kind)
    bodies = template.parse(prompt)
    assert bodies[task_section] == "Do the task."
    assert all(not body for section, body in bodies.items() if section != task_section)


@pytest.mark.parametrize(
    ("model", "expected_family"),
    [(QWEN3_8_27B_MODEL, "alibaba"), (DEEPSEEK_V4_1_FLASH_MODEL, "generic")],
)
def test_experiment_model_pairs_build_without_running_an_experiment(model: str, expected_family: str) -> None:
    """Build each homogeneous model condition without calling its model.

    Args:
        model: Exact model identifier assigned to both roles.
        expected_family: Provider template inferred for the student model.
    """
    strategy_rng = random.Random(7)
    strategy, family = build_react_v2_strategy(
        reflection_model=model,
        task_model=model,
        lm_kwargs={"api_base": "http://localhost:8000/v1"},
        level=2,
        edit_tool_set="broad",
        template_family="auto",
        component_kinds={"summarize1": "system_prompt"},
        rng=strategy_rng,
    )

    assert family == expected_family
    assert strategy.proposer_model == model
    assert strategy.component_kinds == {"summarize1": "system_prompt"}
    assert strategy.rng is strategy_rng
    assert strategy.max_chars is None
    assert {tool.value for tool in strategy.edit_tools} == {
        "INSERT_TEXT",
        "DELETE_TEXT",
        "REPLACE_TEXT",
        "MOVE_TEXT",
    }


@pytest.mark.parametrize("condition", ["vanilla", "react_v2", "react_v2_random", "action"])
@pytest.mark.parametrize("model", [QWEN3_8_27B_MODEL, DEEPSEEK_V4_1_FLASH_MODEL])
def test_hotpotqa_text_limits_reach_roles_and_contracts(condition: str, model: str, tmp_path: Path) -> None:
    """Keep optional limits identical in actual role wiring and saved run identity."""
    limits = TextLimits(
        max_component_chars=25000,
        max_candidate_chars=100000,
        selector_target_chars=20000,
        max_prompt_chars=200000,
        controller_feedback_chars=17000,
        stateless_feedback_chars=15000,
        manifestor_trace_chars=80000,
        manifestor_steering_chars=3000,
        history_text_chars=7000,
    )
    args = _hotpot_args(solver_model=model, reflection_model=model, text_limits=limits.to_dict())
    config, selector = build_hotpotqa_config(condition, args, resolve_hotpotqa_lm_kwargs(model, None))
    contract = build_hotpotqa_run_contract(condition, args)
    assert config.reflection.text_limits == limits
    assert contract["optimizer"]["text_limits"] == limits.to_dict()
    assert contract["optimizer"]["document_length"] == limits.document_contract()
    assert contract["optimizer"]["manifestor_traces_chars"] == 80000
    if condition in ("react_v2", "react_v2_random"):
        assert config.reflection.reflection_strategy.text_limits == limits
        assert config.reflection.reflection_strategy.max_chars == 25000
    if condition == "action":
        assert selector.text_limits == limits
    ensure_wikipedia_run_contract(tmp_path, contract)
    args.text_limits["manifestor_steering_chars"] = 3001
    with pytest.raises(ValueError, match="different Wikipedia benchmark configuration"):
        ensure_wikipedia_run_contract(tmp_path, build_hotpotqa_run_contract(condition, args))


@pytest.mark.parametrize("model", [QWEN3_8_27B_MODEL, DEEPSEEK_V4_1_FLASH_MODEL])
@pytest.mark.parametrize(
    ("budget", "condition"),
    [
        (budget, condition)
        for budget, conditions in _SCIENTIFIC_CONDITIONS_BY_BUDGET.items()
        for condition in conditions
    ],
)
def test_hotpot_provider_sampling_reaches_every_model_role(model: str, budget: int, condition: str) -> None:
    """Apply role-specific provider sampling through all seven cells and persist actual values."""
    args = _hotpot_args(solver_model=model, reflection_model=model, max_metric_calls=budget)
    general = experiment_decoding(model, agentic=False)
    agentic = experiment_decoding(model, agentic=True)
    reflection_kwargs = resolve_hotpotqa_lm_kwargs(model, None, role="optimizer")
    request_overrides = experiment_request_overrides(model, explicit_reasoning=True)
    config, selector = build_hotpotqa_config(condition, args, reflection_kwargs)
    contract = build_hotpotqa_run_contract(condition, args)
    assert config.engine.cache_evaluation is False
    sampler = config.reflection.batch_sampler
    assert isinstance(sampler, IndependentEpochShuffledBatchSampler)
    assert contract["optimizer"]["training_batch_order"] == sampler.contract()
    assert sampler.seed == args.seed
    assert sampler.minibatch_size == config.reflection.reflection_minibatch_size == 3
    expected_rng = random.Random(args.seed)
    loader = ListDataLoader(list(range(150)))
    for epoch in range(2):
        expected_ids = list(range(150))
        expected_rng.shuffle(expected_ids)
        actual_ids = [
            index
            for iteration in range(epoch * 50, (epoch + 1) * 50)
            for index in sampler.next_minibatch_ids(loader, SimpleNamespace(i=iteration))
        ]
        assert actual_ids == expected_ids
    assert contract["program"]["cache_evaluation"] is False
    assert contract["program"]["dspy_disk_cache"] is False
    assert contract["program"]["dspy_memory_cache"] is False
    assert contract["optimizer"]["document_length"] == {
        "version": 2,
        "max_component_chars": None,
        "max_candidate_chars": None,
        "selector_target_chars": None,
    }
    optimizer_general = {**general, "max_tokens": reflection_kwargs["max_tokens"]}
    optimizer_agentic = {**agentic, "max_tokens": reflection_kwargs["max_tokens"]}
    expected = general["temperature"]

    assert expected == 1.0
    assert contract["models"]["solver_decoding"]["temperature"] == expected
    assert config.reflection.reflection_lm_kwargs["temperature"] == expected
    expected_output_cap = 131_072 if model == DEEPSEEK_V4_1_FLASH_MODEL else 32_768
    assert contract["models"]["solver_decoding"]["max_tokens"] == 65_536
    assert contract["models"]["reflection_decoding"]["max_tokens"] == expected_output_cap
    assert config.reflection.reflection_lm_kwargs["max_tokens"] == expected_output_cap
    assert contract["models"]["solver_decoding"]["top_p"] == general["top_p"]
    assert config.reflection.reflection_lm_kwargs["top_p"] == general["top_p"]
    assert config.reflection.reflection_lm_kwargs["extra_body"] == request_overrides["extra_body"]
    assert contract["models"]["solver_request_overrides"] == {
        **request_overrides,
        "extra_body": {**request_overrides["extra_body"], "thinking_token_budget": 32_768},
    }
    assert contract["models"]["reflection_request_overrides"] == request_overrides
    if condition == "random":
        assert isinstance(selector, RandomActionSelector)
        assert not hasattr(selector, "lm")
    elif selector is not None:
        assert selector.lm.completion_kwargs["max_tokens"] == expected_output_cap
        assert selector.lm.completion_kwargs["temperature"] == expected
        assert selector.lm.completion_kwargs["top_p"] == general["top_p"]
        assert selector.lm.completion_kwargs["extra_body"] == request_overrides["extra_body"]
    strategy = config.reflection.reflection_strategy
    if strategy is not None:
        assert strategy.base_lm.completion_kwargs["max_tokens"] == expected_output_cap
        assert strategy.manifestor_lm.completion_kwargs["max_tokens"] == expected_output_cap
        assert strategy.base_lm.completion_kwargs["temperature"] == expected
        assert strategy.manifestor_lm.completion_kwargs["temperature"] == expected
        assert strategy.base_lm.completion_kwargs["top_p"] == agentic["top_p"]
        assert strategy.manifestor_lm.completion_kwargs["top_p"] == general["top_p"]
        assert strategy.base_lm.completion_kwargs["extra_body"] == request_overrides["extra_body"]
        assert strategy.manifestor_lm.completion_kwargs["extra_body"] == request_overrides["extra_body"]
        roles = contract["models"]["reflection_role_decoding"]
        assert roles["manifestor"]["requested"] == {**optimizer_general, "seed": 0}
        assert roles["react_v2_proposer"]["requested"] == {**optimizer_agentic, "seed": 0}
        if condition == "react_v2":
            assert strategy.controller_lm.completion_kwargs["max_tokens"] == expected_output_cap
            assert strategy.controller_lm.completion_kwargs["top_p"] == general["top_p"]
            assert strategy.controller_lm.completion_kwargs["extra_body"] == request_overrides["extra_body"]
            assert roles["controller"]["requested"] == {**optimizer_general, "seed": 0}
        else:
            assert roles["controller"] is None


@pytest.mark.parametrize("role", ["solver", "reflection"])
def test_hotpot_rejects_resume_with_changed_request_timeout(tmp_path: Path, role: str) -> None:
    """Keep a request-deadline change from silently altering a resumed run."""
    contract = build_hotpotqa_run_contract("react_v2", _hotpot_args())
    assert contract["models"][f"{role}_request_timeout_seconds"] == 3600
    old = deepcopy(contract)
    old["models"][f"{role}_request_timeout_seconds"] = 600
    ensure_wikipedia_run_contract(tmp_path, old)
    with pytest.raises(ValueError, match="different Wikipedia benchmark configuration"):
        ensure_wikipedia_run_contract(tmp_path, contract)


@pytest.mark.parametrize("damage", ["missing", "shared", "seed"])
def test_hotpot_rejects_changed_training_batch_order(tmp_path: Path, damage: str) -> None:
    """Require a fresh run after changing the training shuffle policy."""
    contract = build_hotpotqa_run_contract("react_v2", _hotpot_args())
    old = deepcopy(contract)
    if damage == "missing":
        del old["optimizer"]["training_batch_order"]
    elif damage == "shared":
        old["optimizer"]["training_batch_order"]["rng_stream"] = "shared"
    else:
        old["optimizer"]["training_batch_order"]["seed"] += 1
    ensure_wikipedia_run_contract(tmp_path, old)
    with pytest.raises(ValueError, match="different Wikipedia benchmark configuration"):
        ensure_wikipedia_run_contract(tmp_path, contract)


def test_editor_mode_changes_only_forest_contract_and_wandb_changes_neither(tmp_path):
    """Keep ordinary GEPA comparable while preventing editor checkpoint migration."""
    args = _hotpot_args()
    vanilla = build_hotpotqa_run_contract("vanilla", args)
    forest = build_hotpotqa_run_contract("react_v2", args)
    args.editor_mode = "single_call"
    args.wandb_project = "reporting"
    args.wandb_entity = "team"
    assert build_hotpotqa_run_contract("vanilla", args) == vanilla
    updated = build_hotpotqa_run_contract("react_v2", args)
    assert updated["optimizer"]["react_execution"]["max_iterations"] == 1
    normalized = deepcopy(updated)
    normalized["optimizer"]["react_execution"] = forest["optimizer"]["react_execution"]
    normalized["optimizer"]["branch_history"] = forest["optimizer"]["branch_history"]
    assert normalized == forest
    ensure_wikipedia_run_contract(tmp_path, forest)
    with pytest.raises(ValueError):
        ensure_wikipedia_run_contract(tmp_path, updated)


def test_hotpot_training_order_survives_real_engine_resume(tmp_path: Path) -> None:
    """Exercise HotPotQA's launcher and checkpoints without making model requests."""
    args = _hotpot_args(seed=19, max_metric_calls=1000)
    run_dir = tmp_path / "run"
    stop_file = run_dir / "gepa.stop"
    batches = []

    class Recorder:
        """Capture training batches and request one durable pause."""

        def on_minibatch_sampled(self, event):
            """Stop after the fifth minibatch has finished its work."""
            batches.append(event["minibatch_ids"])
            if len(batches) == 5:
                stop_file.touch()

    def evaluate(candidate, example):
        """Skip reflection so the test never contacts a model provider."""
        return 1.0, {"feedback": "Perfect; no reflection needed"}

    def run():
        """Rebuild the entry-point configuration for each process restart."""
        config, _ = build_hotpotqa_config("vanilla", args, {}, run_dir=str(run_dir))
        config.engine.max_candidate_proposals = 110
        config.engine.parallel = False
        run_hotpotqa_condition(
            "offline sampling",
            {"system_prompt": "Answer the question"},
            [{"id": index} for index in range(150)],
            [{"id": "validation"}],
            config,
            evaluate,
            callbacks=[Recorder()],
        )

    run()
    assert len(batches) == 5
    stop_file.unlink()
    run()
    assert len(batches) == 110
    run()
    assert len(batches) == 110
    rng = random.Random(args.seed)
    expected = []
    for _ in range(3):
        ids = list(range(150))
        rng.shuffle(ids)
        expected.extend(ids[start : start + 3] for start in range(0, 150, 3))
    assert batches == expected[:110]


def test_hotpot_rejects_resume_with_the_previous_manifestor_temperature(tmp_path: Path) -> None:
    """Prevent continuing a Manifestor-0.0 checkpoint under provider-temperature sampling."""
    contract = build_hotpotqa_run_contract("react_v2", _hotpot_args())
    old = deepcopy(contract)
    old["models"]["reflection_role_decoding"]["manifestor"]["requested"]["temperature"] = 0.0
    ensure_wikipedia_run_contract(tmp_path, old)

    with pytest.raises(ValueError, match="different Wikipedia benchmark configuration"):
        ensure_wikipedia_run_contract(tmp_path, contract)


@pytest.mark.parametrize("role", ["controller", "manifestor", "react_v2_proposer"])
def test_hotpot_rejects_resume_with_changed_role_top_p(tmp_path: Path, role: str) -> None:
    """Reject sampling drift in any FOREST role before continuing optimization."""
    args = _hotpot_args(solver_model=DEEPSEEK_V4_1_FLASH_MODEL, reflection_model=DEEPSEEK_V4_1_FLASH_MODEL)
    contract = build_hotpotqa_run_contract("react_v2", args)
    old = deepcopy(contract)
    old["models"]["reflection_role_decoding"][role]["requested"]["top_p"] = 0.5
    ensure_wikipedia_run_contract(tmp_path, old)
    with pytest.raises(ValueError, match="different Wikipedia benchmark configuration"):
        ensure_wikipedia_run_contract(tmp_path, contract)


@pytest.mark.parametrize("role", ["solver", "reflection"])
@pytest.mark.parametrize("damage", ["implicit", "lower_effort", "thinking_disabled"])
def test_hotpot_rejects_missing_or_changed_reasoning_settings(tmp_path: Path, role: str, damage: str) -> None:
    """Reject earlier implicit Qwen settings and changed reasoning behavior on resume."""
    contract = build_hotpotqa_run_contract("react_v2", _hotpot_args())
    old = deepcopy(contract)
    if damage == "implicit":
        old["models"][f"{role}_request_overrides"] = {}
    else:
        template_kwargs = old["models"][f"{role}_request_overrides"]["extra_body"]["chat_template_kwargs"]
        if damage == "lower_effort":
            template_kwargs["reasoning_effort"] = "low"
        else:
            template_kwargs["enable_thinking"] = False
    ensure_wikipedia_run_contract(tmp_path, old)
    with pytest.raises(ValueError, match="different Wikipedia benchmark configuration"):
        ensure_wikipedia_run_contract(tmp_path, contract)


@pytest.mark.parametrize("model", [QWEN3_8_27B_MODEL, DEEPSEEK_V4_1_FLASH_MODEL])
def test_role_specific_sampling_keeps_distinct_journal_namespaces(tmp_path: Path, model: str) -> None:
    """Share identical Controller/editor clients and isolate different sampling profiles."""
    strategy, family = build_react_v2_strategy(
        reflection_model=model,
        task_model=model,
        lm_kwargs={
            **experiment_decoding(model, agentic=False),
            "response_journal_path": str(tmp_path / "responses.sqlite3"),
        },
        level=2,
        edit_tool_set="broad",
        template_family="auto",
        react_top_p=float(experiment_decoding(model, agentic=True)["top_p"]),
        manifestor_temperature=1.0,
    )
    assert strategy.controller_lm is strategy.base_lm
    expected_namespaces = ("controller-proposer", "controller-proposer", "manifestor")
    assert (
        tuple(
            client._response_journal.namespace
            for client in (strategy.base_lm, strategy.controller_lm, strategy.manifestor_lm)
        )
        == expected_namespaces
    )
    contract = strategy.run_contract({"sys": structured_prompt("Answer accurately.", family, "system_prompt")})
    assert contract["controller_lm"]["completion_kwargs"]["top_p"] == experiment_decoding(model, agentic=False)["top_p"]
    assert contract["proposer_lm"]["completion_kwargs"]["top_p"] == 0.95


def _hotpot_args(**overrides):
    """Build a complete HotPotQA argument namespace with targeted overrides.

    Args:
        **overrides: Values replacing the shared test defaults.

    Returns:
        Namespace accepted by both Wikipedia benchmark configuration builders.
    """
    values = {
        "api_base": None,
        "condition": "both",
        "data_identity": {
            "source": {"type": "huggingface", "dataset": "hotpot_qa", "config": "fullwiki"},
            "splits": {"train": {"count": 150, "ids": ["train"], "sha256": "train-v1"}},
        },
        "data_path": None,
        "edit_tool_set": "broad",
        "enforce_scientific_contract": False,
        "max_workers": 32,
        "max_metric_calls": 100,
        "merge": False,
        "program": "2stage",
        "reflection_api_base": "http://localhost:8000/v1",
        "reflection_level": 2,
        "reflection_model": QWEN3_8_27B_MODEL,
        "retrieval_k": 7,
        "seed": 0,
        "seed_style": "structured",
        "solver_api_base": "http://localhost:8000/v1",
        "solver_model": QWEN3_8_27B_MODEL,
        "tag": "",
        "template_family": "auto",
        "test_limit": 300,
        "train_limit": 150,
        "val_limit": 300,
        "wiki17_dir": "/tmp/gepa-wiki17",
        "wikipedia_cache": "/tmp/gepa-wikipedia.sqlite3",
        "wikipedia_endpoint": "https://en.wikipedia.org/w/api.php",
        "wikipedia_timeout": 20.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _scientific_data_identity() -> dict[str, object]:
    """Build the exact pinned identity accepted by scientific HotPotQA runs.

    Returns:
        Source revision, selected-record counts, and ordered content digests.
    """
    identity = {
        "source": {
            "type": "huggingface",
            "dataset": "hotpot_qa",
            "config": "fullwiki",
            "revision": HOTPOTQA_HF_REVISION,
        },
        "splits": {
            split_name: {
                "count": count,
                "sha256": HOTPOTQA_SCIENTIFIC_SPLIT_SHA256[split_name],
            }
            for split_name, count in (("train", 150), ("val", 300), ("test", 300))
        },
    }
    return identity


@pytest.mark.parametrize(
    ("model", "expected_version"),
    [
        (QWEN3_8_27B_MODEL, QWEN3_8_27B_REVISION),
        (DEEPSEEK_V4_1_FLASH_MODEL, DEEPSEEK_V4_1_FLASH_REVISION),
    ],
)
def test_experiment_models_resolve_to_declared_runtime_identities(model: str, expected_version: str) -> None:
    """Map each transport to its checkpoint, version, or rolling alias.

    Args:
        model: Canonical local model identifier.
        expected_version: Checkpoint revision, dated version, or direct alias.
    """
    actual_version = experiment_model_version(model)
    assert actual_version == expected_version


def test_experiment_model_version_rejects_unknown_models() -> None:
    """Refuse to give unapproved model names a scientific version identity."""
    unknown_model = "provider/unknown"
    with pytest.raises(ValueError, match="Unsupported experiment model"):
        experiment_model_version(unknown_model)


@pytest.mark.parametrize(
    ("max_metric_calls", "condition"),
    [
        (6_871, "vanilla"),
        (6_871, "react_v2"),
        (6_871, "react_v2_random"),
        (6_871, "action"),
        (6_871, "random"),
        (13_742, "vanilla"),
        (13_742, "react_v2"),
    ],
)
def test_hotpot_scientific_contract_accepts_only_the_pinned_qwen_runtime(
    monkeypatch,
    max_metric_calls: int,
    condition: str,
) -> None:
    """Accept the full paper-aligned Qwen method with immutable runtime pins.

    Args:
        monkeypatch: Pytest fixture used to install recorded runtime metadata.
        max_metric_calls: Approved standard or expanded campaign budget.
        condition: Approved method at the selected budget.
    """
    for name, value in QWEN_SCIENTIFIC_RUNTIME.items():
        monkeypatch.setenv(name, value)
    args = _hotpot_args(
        condition=condition,
        enforce_scientific_contract=True,
        max_metric_calls=max_metric_calls,
        train_limit=None,
        val_limit=None,
        test_limit=None,
        data_identity=_scientific_data_identity(),
    )

    _validate_scientific_contract(args)
    _validate_scientific_data_identity(args)


def test_hotpot_scientific_campaign_contains_only_the_seven_approved_cells() -> None:
    """Lock the seven comparisons and their shared classic GEPA topology."""
    assert _SCIENTIFIC_CONDITIONS_BY_BUDGET == {
        6_871: ("vanilla", "react_v2", "react_v2_random", "action", "random"),
        13_742: ("vanilla", "react_v2"),
    }
    assert 13_742 == 2 * 6_871
    assert sum(len(conditions) for conditions in _SCIENTIFIC_CONDITIONS_BY_BUDGET.values()) == 7
    for max_metric_calls, conditions in _SCIENTIFIC_CONDITIONS_BY_BUDGET.items():
        for condition in conditions:
            config, _ = build_hotpotqa_config(
                condition,
                _hotpot_args(max_metric_calls=max_metric_calls),
                {},
            )
            assert isinstance(config.engine.sampling_strategy, SingleMutationSampling)
            assert isinstance(config.engine.selection_strategy, AllImprovements)


def test_hotpot_scientific_contract_accepts_the_pinned_deepseek_runtime(monkeypatch) -> None:
    """Accept DeepSeek only under the exact local vLLM serving contract.

    Args:
        monkeypatch: Pytest fixture used to install recorded runtime metadata.
    """
    for name, value in DEEPSEEK_SCIENTIFIC_RUNTIME.items():
        monkeypatch.setenv(name, value)
    args = _hotpot_args(
        condition="react_v2",
        enforce_scientific_contract=True,
        max_metric_calls=13_742,
        solver_model=DEEPSEEK_V4_1_FLASH_MODEL,
        reflection_model=DEEPSEEK_V4_1_FLASH_MODEL,
        solver_api_base=LOCAL_API_BASE,
        reflection_api_base=LOCAL_API_BASE,
        train_limit=None,
        val_limit=None,
        test_limit=None,
        data_identity=_scientific_data_identity(),
    )

    _validate_scientific_contract(args)
    _validate_scientific_data_identity(args)


@pytest.mark.parametrize(
    "model,runtime,expected_count",
    [(QWEN3_8_27B_MODEL, QWEN_SCIENTIFIC_RUNTIME, 1), (DEEPSEEK_V4_1_FLASH_MODEL, DEEPSEEK_SCIENTIFIC_RUNTIME, 4)],
)
@pytest.mark.parametrize("gpu_count", [1, 4, 8])
def test_hotpot_scientific_contract_enforces_model_gpu_count(monkeypatch, model, runtime, expected_count, gpu_count):
    """Reject the former eight-GPU default and allocations meant for the other model."""
    for name, value in {**runtime, "HOTPOTQA_GPU_RUNTIME": h200_gpu_runtime(gpu_count)}.items():
        monkeypatch.setenv(name, value)
    args = _hotpot_args(
        enforce_scientific_contract=True,
        max_metric_calls=6_871,
        solver_model=model,
        reflection_model=model,
        solver_api_base=LOCAL_API_BASE,
        reflection_api_base=LOCAL_API_BASE,
        train_limit=None,
        val_limit=None,
        test_limit=None,
    )
    if gpu_count == expected_count:
        _validate_scientific_contract(args)
    else:
        with pytest.raises(ValueError, match=f"exactly {expected_count} H200"):
            _validate_scientific_contract(args)


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"HOTPOTQA_MODEL_REVISION": "moving-main"}, "HOTPOTQA_MODEL_REVISION"),
        ({"HOTPOTQA_MODEL_INTEGRITY_SHA256": "moving-manifest"}, "HOTPOTQA_MODEL_INTEGRITY_SHA256"),
        ({"HOTPOTQA_WEIGHT_DTYPE": "bfloat16"}, "HOTPOTQA_WEIGHT_DTYPE"),
        ({"HOTPOTQA_KV_CACHE_DTYPE": "bfloat16"}, "HOTPOTQA_KV_CACHE_DTYPE"),
        ({"HOTPOTQA_SERVING_ENGINE": "sglang"}, "HOTPOTQA_SERVING_ENGINE"),
        ({"HOTPOTQA_VLLM_VERSION": "0.24.0"}, "HOTPOTQA_VLLM_VERSION"),
        ({"HOTPOTQA_SERVING_LOCK_SHA256": "moving-main"}, "HOTPOTQA_SERVING_LOCK_SHA256"),
        ({"HOTPOTQA_SERVING_ENV_SHA256": "moving-env"}, "HOTPOTQA_SERVING_ENV_SHA256"),
        ({"HOTPOTQA_TRANSFORMERS_VERSION": ""}, "HOTPOTQA_TRANSFORMERS_VERSION"),
        ({"HOTPOTQA_GPU_RUNTIME": "{}"}, "HOTPOTQA_GPU_RUNTIME"),
        ({"HOTPOTQA_SERVE_ARGUMENTS": "tp=8"}, "HOTPOTQA_SERVE_ARGUMENTS"),
        (
            {
                "HOTPOTQA_SERVE_ARGUMENTS": DEEPSEEK_SERVE_ARGUMENTS.replace(
                    "engram_cpu_offload=true", "engram_cpu_offload=false"
                )
            },
            "engram_cpu_offload=true",
        ),
    ],
)
def test_hotpot_scientific_contract_rejects_deepseek_runtime_drift(
    monkeypatch,
    environment: dict[str, str],
    message: str,
) -> None:
    """Reject DeepSeek checkpoint, image, engine, or topology drift.

    Args:
        monkeypatch: Pytest fixture used to install runtime metadata.
        environment: One altered DeepSeek runtime field.
        message: Runtime field expected in the rejection.
    """
    for name, value in {**DEEPSEEK_SCIENTIFIC_RUNTIME, **environment}.items():
        monkeypatch.setenv(name, value)
    args = _hotpot_args(
        condition="react_v2",
        enforce_scientific_contract=True,
        max_metric_calls=13_742,
        solver_model=DEEPSEEK_V4_1_FLASH_MODEL,
        reflection_model=DEEPSEEK_V4_1_FLASH_MODEL,
        solver_api_base=LOCAL_API_BASE,
        reflection_api_base=LOCAL_API_BASE,
        train_limit=None,
        val_limit=None,
        test_limit=None,
        data_identity=_scientific_data_identity(),
    )

    with pytest.raises(ValueError, match=message):
        _validate_scientific_contract(args)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("solver_api_base", None),
        ("solver_api_base", "https://api.deepseek.com/v1"),
        ("reflection_api_base", None),
        ("reflection_api_base", "http://127.0.0.1/v1"),
    ],
)
def test_hotpot_scientific_contract_rejects_nonlocal_deepseek_endpoints(
    monkeypatch,
    field: str,
    value: str | None,
) -> None:
    """Require both DeepSeek roles to use a local loopback endpoint with a port.

    Args:
        monkeypatch: Pytest fixture used to install recorded runtime metadata.
        field: Student or proposer endpoint changed by the test.
        value: Missing, external, or incomplete loopback endpoint.
    """
    for name, runtime_value in DEEPSEEK_SCIENTIFIC_RUNTIME.items():
        monkeypatch.setenv(name, runtime_value)
    values = {
        "enforce_scientific_contract": True,
        "max_metric_calls": 13_742,
        "solver_model": DEEPSEEK_V4_1_FLASH_MODEL,
        "reflection_model": DEEPSEEK_V4_1_FLASH_MODEL,
        "solver_api_base": LOCAL_API_BASE,
        "reflection_api_base": LOCAL_API_BASE,
        "train_limit": None,
        "val_limit": None,
        "test_limit": None,
        "data_identity": _scientific_data_identity(),
    }
    values[field] = value
    args = _hotpot_args(**values)

    with pytest.raises(ValueError, match=field.replace("_", "-")):
        _validate_scientific_contract(args)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"program": "1stage"}, "--program"),
        ({"seed_style": "plain"}, "--seed-style"),
        ({"seed": 1}, "--seed"),
        ({"retrieval_k": 8}, "--retrieval-k"),
        ({"reflection_level": 1}, "--reflection-level"),
        ({"edit_tool_set": "minimal"}, "--edit-tool-set"),
        ({"template_family": "generic"}, "--template-family"),
        ({"data_path": "/tmp/custom.jsonl"}, "--data-path"),
        ({"train_limit": 149}, "--train-limit"),
        ({"val_limit": 299}, "--val-limit"),
        ({"test_limit": 299}, "--test-limit"),
        ({"merge": True}, "--merge"),
        ({"max_metric_calls": 6_870}, "--max-metric-calls"),
        ({"condition": "random", "max_metric_calls": 13_742}, "--condition"),
        ({"condition": "action", "max_metric_calls": 13_742}, "--condition"),
        ({"condition": "react_v2_random", "max_metric_calls": 13_742}, "--condition"),
        ({"max_workers": 0}, "--max-workers"),
    ],
)
def test_hotpot_scientific_contract_rejects_methodology_drift(
    monkeypatch,
    overrides: dict[str, object],
    message: str,
) -> None:
    """Fail closed when any quality-relevant production axis changes.

    Args:
        monkeypatch: Pytest fixture used to install recorded runtime metadata.
        overrides: One disallowed methodology change.
        message: Rejected command-line axis expected in the error.
    """
    for name, value in QWEN_SCIENTIFIC_RUNTIME.items():
        monkeypatch.setenv(name, value)
    scientific_values = {
        "enforce_scientific_contract": True,
        "max_metric_calls": 6_871,
        "train_limit": None,
        "val_limit": None,
        "test_limit": None,
        "data_identity": _scientific_data_identity(),
    }
    scientific_values.update(overrides)
    args = _hotpot_args(**scientific_values)

    with pytest.raises(ValueError, match=message):
        _validate_scientific_contract(args)


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"HOTPOTQA_MODEL_REVISION": "moving-main"}, "HOTPOTQA_MODEL_REVISION"),
        ({"HOTPOTQA_MODEL_INTEGRITY_SHA256": "moving-manifest"}, "HOTPOTQA_MODEL_INTEGRITY_SHA256"),
        ({"HOTPOTQA_WEIGHT_DTYPE": "float16"}, "HOTPOTQA_WEIGHT_DTYPE"),
        ({"HOTPOTQA_KV_CACHE_DTYPE": "fp8"}, "HOTPOTQA_KV_CACHE_DTYPE"),
        ({"HOTPOTQA_SERVING_ENGINE": "sglang"}, "HOTPOTQA_SERVING_ENGINE"),
        ({"HOTPOTQA_VLLM_BATCH_INVARIANT": "true"}, "HOTPOTQA_VLLM_BATCH_INVARIANT"),
        (
            {"HOTPOTQA_VLLM_SINGLE_SEQUENCE_REPLICAS": "false"},
            "HOTPOTQA_VLLM_SINGLE_SEQUENCE_REPLICAS",
        ),
        ({"HOTPOTQA_VLLM_VERSION": ""}, "HOTPOTQA_VLLM_VERSION"),
        ({"HOTPOTQA_TRANSFORMERS_VERSION": ""}, "HOTPOTQA_TRANSFORMERS_VERSION"),
        ({"HOTPOTQA_SERVING_LOCK_SHA256": "moving-main"}, "HOTPOTQA_SERVING_LOCK_SHA256"),
        ({"HOTPOTQA_SERVING_ENV_SHA256": "moving-environment"}, "HOTPOTQA_SERVING_ENV_SHA256"),
        ({"HOTPOTQA_GPU_RUNTIME": "{}"}, "HOTPOTQA_GPU_RUNTIME"),
        ({"HOTPOTQA_SOURCE_COMMIT": "moving-main"}, "HOTPOTQA_SOURCE_COMMIT"),
        ({"HOTPOTQA_SOURCE_MANIFEST_SHA256": "moving-manifest"}, "HOTPOTQA_SOURCE_MANIFEST_SHA256"),
        ({"HOTPOTQA_PYTHON_VERSION": "3.11.14"}, "HOTPOTQA_PYTHON_VERSION"),
        ({"HOTPOTQA_UV_VERSION": "0.9.14"}, "HOTPOTQA_UV_VERSION"),
        ({"HOTPOTQA_UV_SHA256": "moving-uv"}, "HOTPOTQA_UV_SHA256"),
        ({"HOTPOTQA_LITELLM_VERSION": ""}, "HOTPOTQA_LITELLM_VERSION"),
        ({"HOTPOTQA_CAMPAIGN_ID": ""}, "HOTPOTQA_CAMPAIGN_ID"),
        ({"HOTPOTQA_ENV_SPEC_SHA256": "moving-lock"}, "HOTPOTQA_ENV_SPEC_SHA256"),
        ({"HOTPOTQA_GEPA_ENV_SHA256": "moving-environment"}, "HOTPOTQA_GEPA_ENV_SHA256"),
        ({"HOTPOTQA_SERVE_ARGUMENTS": "tp=1"}, "HOTPOTQA_SERVE_ARGUMENTS"),
    ],
)
def test_hotpot_scientific_contract_rejects_qwen_runtime_drift(
    monkeypatch,
    environment: dict[str, str],
    message: str,
) -> None:
    """Fail closed when checkpoint or exact-quality serving metadata drifts.

    Args:
        monkeypatch: Pytest fixture used to install runtime metadata.
        environment: One altered Qwen runtime field.
        message: Runtime field expected in the rejection.
    """
    for name, value in {**QWEN_SCIENTIFIC_RUNTIME, **environment}.items():
        monkeypatch.setenv(name, value)
    args = _hotpot_args(
        enforce_scientific_contract=True,
        max_metric_calls=6_871,
        train_limit=None,
        val_limit=None,
        test_limit=None,
        data_identity=_scientific_data_identity(),
    )

    with pytest.raises(ValueError, match=message):
        _validate_scientific_contract(args)


@pytest.mark.parametrize(
    "model,runtime",
    [(QWEN3_8_27B_MODEL, QWEN_SCIENTIFIC_RUNTIME), (DEEPSEEK_V4_1_FLASH_MODEL, DEEPSEEK_SCIENTIFIC_RUNTIME)],
)
@pytest.mark.parametrize("sequences", [1, 2, 4, 3])
def test_batching_profile_is_validated_and_frozen(monkeypatch, tmp_path, model, runtime, sequences):
    """Allow the measured batch sizes and reject resume into a different profile."""
    for name, value in runtime.items():
        monkeypatch.setenv(name, value)
    args = _hotpot_args(
        solver_model=model,
        reflection_model=model,
        enforce_scientific_contract=True,
        max_metric_calls=6871,
        train_limit=None,
        val_limit=None,
        test_limit=None,
        data_identity=_scientific_data_identity(),
    )
    initial = build_hotpotqa_run_contract("react_v2", args)
    ensure_wikipedia_run_contract(tmp_path, initial)
    single = "true" if sequences == 1 else "false"
    serve = runtime["HOTPOTQA_SERVE_ARGUMENTS"].replace("max_num_seqs=1", f"max_num_seqs={sequences}")
    serve = serve.replace("single_sequence_replicas=true", f"single_sequence_replicas={single}")
    monkeypatch.setenv("HOTPOTQA_SERVE_ARGUMENTS", serve)
    monkeypatch.setenv("HOTPOTQA_VLLM_SINGLE_SEQUENCE_REPLICAS", single)
    if sequences == 3:
        with pytest.raises(ValueError, match="max_num_seqs"):
            build_hotpotqa_run_contract("react_v2", args)
        return
    contract = build_hotpotqa_run_contract("react_v2", args)
    assert f"max_num_seqs={sequences}" in contract["execution_runtime"]["serve_arguments"]
    if sequences == 1:
        ensure_wikipedia_run_contract(tmp_path, contract)
    else:
        with pytest.raises(ValueError, match="different Wikipedia benchmark configuration"):
            ensure_wikipedia_run_contract(tmp_path, contract)


@pytest.mark.parametrize(
    ("changed_field", "message"),
    [
        ("revision", "revision"),
        ("train_count", "train split"),
        ("train_digest", "train split"),
        ("val_count", "val split"),
        ("val_digest", "val split"),
        ("test_count", "test split"),
        ("test_digest", "test split"),
    ],
)
def test_hotpot_scientific_data_identity_rejects_split_drift(changed_field: str, message: str) -> None:
    """Reject source, count, and ordered-record changes before optimization.

    Args:
        changed_field: Source, count, or digest field changed by the test.
        message: Rejected source or split expected in the error.
    """
    identity = _scientific_data_identity()
    if changed_field == "revision":
        identity["source"]["revision"] = "moving-main"
    else:
        split, field = changed_field.split("_")
        if field == "count":
            identity["splits"][split]["count"] -= 1
        else:
            identity["splits"][split]["sha256"] = "corrupt"
    args = _hotpot_args(enforce_scientific_contract=True, data_identity=identity)

    with pytest.raises(ValueError, match=message):
        _validate_scientific_data_identity(args)


def test_production_wiki17_attestation_avoids_a_second_deep_hash(tmp_path, monkeypatch) -> None:
    """Reuse the launcher's locked deep verification after checking its manifest digest.

    Args:
        tmp_path: Pytest directory containing the integrity manifest.
        monkeypatch: Pytest fixture used to install the production attestation.
    """
    integrity_path = tmp_path / "integrity.json"
    integrity_path.write_text('{"verified":true}', encoding="utf-8")
    expected_sha256 = file_sha256(integrity_path)
    verify_integrity = Mock()
    retriever = SimpleNamespace(
        integrity_path=integrity_path,
        verify_integrity=verify_integrity,
    )
    monkeypatch.setenv("HOTPOTQA_PRODUCTION_LAUNCH", "1")
    monkeypatch.setenv("HOTPOTQA_VERIFIED_WIKI17_INTEGRITY_SHA256", expected_sha256)

    _verify_scientific_retriever_integrity(retriever)

    verify_integrity.assert_not_called()


def test_direct_wiki17_run_retains_deep_verification(tmp_path, monkeypatch) -> None:
    """Deep-hash Wiki-2017 when no locked Slurm attestation is available.

    Args:
        tmp_path: Pytest directory containing an unused integrity manifest.
        monkeypatch: Pytest fixture used to clear production launch state.
    """
    verify_integrity = Mock()
    retriever = SimpleNamespace(
        integrity_path=tmp_path / "integrity.json",
        verify_integrity=verify_integrity,
    )
    monkeypatch.delenv("HOTPOTQA_PRODUCTION_LAUNCH", raising=False)
    monkeypatch.delenv("HOTPOTQA_VERIFIED_WIKI17_INTEGRITY_SHA256", raising=False)

    _verify_scientific_retriever_integrity(retriever)

    verify_integrity.assert_called_once_with()


@pytest.mark.parametrize("attestation", ["not-a-digest", "0" * 64])
def test_production_wiki17_attestation_rejects_invalid_identity(
    attestation: str,
    tmp_path,
    monkeypatch,
) -> None:
    """Reject malformed and byte-mismatched production attestations.

    Args:
        attestation: Malformed or incorrect manifest digest under test.
        tmp_path: Pytest directory containing the integrity manifest.
        monkeypatch: Pytest fixture used to install production launch state.
    """
    integrity_path = tmp_path / "integrity.json"
    integrity_path.write_text('{"verified":true}', encoding="utf-8")
    retriever = SimpleNamespace(
        integrity_path=integrity_path,
        verify_integrity=Mock(),
    )
    monkeypatch.setenv("HOTPOTQA_PRODUCTION_LAUNCH", "1")
    monkeypatch.setenv("HOTPOTQA_VERIFIED_WIKI17_INTEGRITY_SHA256", attestation)

    with pytest.raises(ValueError, match="attestation"):
        _verify_scientific_retriever_integrity(retriever)


def test_material_ablation_axes_get_distinct_resumable_run_keys() -> None:
    """Give edit-tool and reflection-level ablations distinct resume keys."""
    broad = hotpotqa_run_key("react_v2", _hotpot_args(edit_tool_set="broad"))
    minimal = hotpotqa_run_key("react_v2", _hotpot_args(edit_tool_set="minimal"))
    level_one = hotpotqa_run_key("react_v2", _hotpot_args(reflection_level=1))

    assert len({broad, minimal, level_one}) == 3
    assert "l2-broad" in broad


def test_complete_run_keys_cover_budget_seed_retrieval_and_data_identity() -> None:
    """Fingerprint every material model, budget, retrieval, and data axis."""
    base = hotpotqa_run_key("vanilla", _hotpot_args())
    changed = {
        hotpotqa_run_key("vanilla", _hotpot_args(max_metric_calls=101)),
        hotpotqa_run_key(
            "vanilla",
            _hotpot_args(solver_model=DEEPSEEK_V4_1_FLASH_MODEL, reflection_model=DEEPSEEK_V4_1_FLASH_MODEL),
        ),
        hotpotqa_run_key("vanilla", _hotpot_args(solver_api_base="https://solver.example/v1")),
        hotpotqa_run_key("vanilla", _hotpot_args(reflection_api_base="https://other-reflection.example/v1")),
        hotpotqa_run_key("vanilla", _hotpot_args(seed_style="plain")),
        hotpotqa_run_key("vanilla", _hotpot_args(retrieval_k=8)),
        hotpotqa_run_key(
            "vanilla",
            _hotpot_args(
                data_identity={
                    "source": {"type": "huggingface", "dataset": "hotpot_qa", "config": "fullwiki"},
                    "splits": {"train": {"count": 150, "ids": ["train"], "sha256": "train-v2"}},
                }
            ),
        ),
    }

    assert base not in changed
    assert len(changed) == 7
    vanilla_contract = build_hotpotqa_run_contract("vanilla", _hotpot_args())
    assert vanilla_contract["optimizer"]["reflection_level"] == 0
    assert vanilla_contract["optimizer"]["semantic_action_space"] is None
    assert vanilla_contract["optimizer"]["semantic_controller_policy"] is None
    level_one_contract = build_hotpotqa_run_contract("react_v2", _hotpot_args(reflection_level=1))
    assert level_one_contract["optimizer"]["semantic_action_space"] is None
    assert level_one_contract["optimizer"]["semantic_controller_policy"] is None


def test_scientific_run_key_normalizes_ports_and_rejects_external_endpoints(monkeypatch) -> None:
    """Resume Qwen across Slurm ports while rejecting external endpoints.

    Args:
        monkeypatch: Pytest fixture used to install pinned runtime metadata.
    """
    for name, value in QWEN_SCIENTIFIC_RUNTIME.items():
        monkeypatch.setenv(name, value)
    shared = {
        "enforce_scientific_contract": True,
        "max_metric_calls": 6_871,
        "train_limit": None,
        "val_limit": None,
        "test_limit": None,
        "data_identity": _scientific_data_identity(),
    }
    first = _hotpot_args(
        **shared,
        solver_api_base="http://localhost:10001/v1",
        reflection_api_base="http://localhost:10001/v1",
    )
    resubmitted = _hotpot_args(
        **shared,
        solver_api_base="http://localhost:14999/v1",
        reflection_api_base="http://localhost:14999/v1",
    )
    external = _hotpot_args(
        **shared,
        solver_api_base="https://solver.example/v1",
        reflection_api_base="https://solver.example/v1",
    )

    assert hotpotqa_run_key("react_v2", first) == hotpotqa_run_key("react_v2", resubmitted)
    with pytest.raises(ValueError, match="solver-api-base"):
        hotpotqa_run_key("react_v2", external)
    contract = build_hotpotqa_run_contract("react_v2", first)
    assert contract["models"]["solver_api_base"] == "local-loopback/v1"
    assert contract["models"]["reflection_api_base"] == "local-loopback/v1"


def test_hotpotqa_merge_is_an_opt_in_axis_shared_by_every_condition() -> None:
    """Keep the paper merge policy orthogonal to every reflection strategy."""
    no_merge_args = _hotpot_args()
    merge_args = _hotpot_args(merge=True)

    assert build_hotpotqa_run_contract("vanilla", no_merge_args)["optimizer"]["merge"] is None
    assert build_hotpotqa_config("vanilla", no_merge_args, {})[0].merge is None

    expected = {
        "max_merge_invocations": 5,
        "merge_val_overlap_floor": 5,
    }
    for condition in ("vanilla", "random", "action", "react_v2_random", "react_v2"):
        contract = build_hotpotqa_run_contract(condition, merge_args)
        config, _ = build_hotpotqa_config(condition, merge_args, {})

        assert contract["optimizer"]["merge"] == expected
        assert config.merge is not None
        assert config.merge.max_merge_invocations == 5
        assert config.merge.merge_val_overlap_floor == 5

    assert hotpotqa_run_key("vanilla", no_merge_args) != hotpotqa_run_key("vanilla", merge_args)
    assert hotpotqa_run_key("vanilla", merge_args).startswith("merge-")


@pytest.mark.parametrize(
    ("config_builder", "contract_builder", "extra_args"),
    [
        (build_hotpotqa_config, build_hotpotqa_run_contract, {}),
        (build_hover_config, build_hover_run_contract, {"final_retrieval_k": 10}),
    ],
)
def test_every_condition_uses_the_same_within_run_example_concurrency(
    config_builder,
    contract_builder,
    extra_args: dict,
) -> None:
    """Propagate one evaluator width through every comparable method.

    Args:
        config_builder: Benchmark-specific GEPA configuration builder.
        contract_builder: Benchmark-specific persisted-contract builder.
        extra_args: Benchmark-specific argument overrides.
    """
    args = _hotpot_args(max_workers=47, **extra_args)
    reflection_kwargs = {
        "num_retries": EXPERIMENT_NUM_RETRIES,
        **experiment_decoding(args.reflection_model),
    }

    conditions = (
        ("vanilla", "random", "action", "react_v2_random", "react_v2")
        if config_builder is build_hotpotqa_config
        else ("vanilla", "random", "action", "react_v2")
    )
    for condition in conditions:
        config, _ = config_builder(condition, args, reflection_kwargs)
        contract = contract_builder(condition, args)

        assert config.engine.parallel is True
        assert config.engine.max_workers == 47
        assert config.engine.raise_on_exception is True
        assert contract["program"]["parallel_workers"] == 47


def test_hotpot_and_hover_contracts_record_exact_model_pair() -> None:
    """Record the same experiment model and endpoint for both roles."""
    hotpot = build_hotpotqa_run_contract("react_v2", _hotpot_args())
    hover_args = _hotpot_args(final_retrieval_k=10)
    hover = build_hover_run_contract("react_v2", hover_args)

    for contract in (hotpot, hover):
        assert contract["models"]["solver"] == QWEN3_8_27B_MODEL
        assert contract["models"]["solver_api_base"] == "http://localhost:8000/v1"
        assert contract["models"]["reflection"] == QWEN3_8_27B_MODEL
        assert contract["models"]["reflection_api_base"] == "http://localhost:8000/v1"
        assert contract["models"]["solver_num_retries"] == EXPERIMENT_NUM_RETRIES
        assert contract["models"]["reflection_num_retries"] == EXPERIMENT_NUM_RETRIES
        assert contract["optimizer"]["max_metric_calls"] == 100
        assert contract["optimizer"]["seed_style"] == "structured"
        assert set(contract["optimizer"]["component_kinds"].values()) == {"system_prompt"}
        assert contract["optimizer"]["semantic_action_space"] == SEMANTIC_ACTION_CATALOGS["prompt"]
        assert contract["optimizer"]["semantic_controller_policy"] == CONTROLLER_POLICY_CONTRACT

    expected_hotpot_decoding = {**experiment_decoding(QWEN3_8_27B_MODEL), "max_tokens": 65_536, "seed": 0}
    assert hotpot["models"]["solver_decoding"] == expected_hotpot_decoding
    assert hotpot["models"]["reflection_decoding"] == {**expected_hotpot_decoding, "max_tokens": 32_768}
    assert hover["models"]["solver_decoding"] == experiment_decoding(QWEN3_8_27B_MODEL)
    assert hover["models"]["reflection_decoding"] == experiment_decoding(QWEN3_8_27B_MODEL)

    assert hotpot["schema_version"] == 28
    assert hotpot["baseline_protocol"]["test_repetitions"] == 1
    assert hotpot["optimizer"]["react_execution"]["completion"] == "explicit_finish"
    assert hotpot["optimizer"]["react_execution"]["max_iterations"] is None
    assert hotpot["optimizer"]["react_execution"]["max_tool_calls"] is None
    assert hotpot["optimizer"]["manifestor_traces_chars"] is None
    assert hotpot["optimizer"]["reflection_context"]["version"] == 2
    assert hotpot["optimizer"]["reflection_context"]["duplicates"] == "preserved"
    assert hotpot["scientific_contract_enforced"] is False
    assert hotpot["models"]["solver_version"] == QWEN3_8_27B_REVISION
    assert hotpot["models"]["reflection_version"] == QWEN3_8_27B_REVISION
    assert hotpot["retrieval"]["backend"] == "wiki17-bm25s"
    assert hotpot["retrieval"]["k1"] == 0.9
    assert hotpot["retrieval"]["b"] == 0.4
    assert hotpot["optimizer"]["reflection_minibatch_size"] == 3
    assert hotpot["optimizer"]["proposal_sampling_strategy"] == {
        "name": "single_mutation",
        "parents_per_iteration": 1,
        "mutations_per_parent": 1,
    }
    assert hotpot["optimizer"]["proposal_selection_strategy"] == "all_improvements"
    assert hotpot["optimizer"]["frontier_type"] == "instance"
    assert hotpot["optimizer"]["skip_perfect_score"] is True
    assert hotpot["optimizer"]["raise_on_exception"] is True
    assert hotpot["optimizer"]["branch_history"] == {
        "storage": "target_scoped_user_assistant_messages",
        "delivery": "provider_chat_messages",
    }
    assert hotpot["program"]["parallel_workers"] == 32
    assert hotpot["program"]["predictor_type"] == "dspy_chain_of_thought"
    assert hotpot["program"]["predictor_adapter"] == "dspy_chat_adapter"
    assert hotpot["program"]["dspy_runtime_version"] == "2.6.23"
    assert hotpot["program"]["dspy_runtime_commit"] == "62dc3b634d7dc0c4889abcf905cb4c391ea6b396"
    assert hotpot["program"]["cache_evaluation"] is False
    assert hotpot["program"]["dspy_disk_cache"] is False
    assert hotpot["program"]["dspy_memory_cache"] is False
    assert hotpot["program"]["dspy_history"] is False
    assert hotpot["execution_runtime"] == {
        "campaign_id": None,
        "source_commit": None,
        "source_manifest_sha256": None,
        "python_version": None,
        "uv_version": None,
        "uv_sha256": None,
        "env_spec_sha256": None,
        "gepa_env_sha256": None,
        "serving_engine": None,
        "serving_lock_sha256": None,
        "serving_env_sha256": None,
        "gpu_runtime": None,
        "vllm_version": None,
        "torch_version": None,
        "cuda_version": None,
        "cuda_module": None,
        "transformers_version": None,
        "litellm_version": None,
        "model_revision": None,
        "model_integrity_sha256": None,
        "weight_dtype": None,
        "kv_cache_dtype": None,
        "serve_arguments": None,
        "vllm_batch_invariant": None,
        "vllm_single_sequence_replicas": None,
    }
    assert hotpot["program"]["component_output_fields"] == {
        "summarize1": ["reasoning", "summary"],
        "create_query_hop2": ["reasoning", "query"],
        "summarize2": ["reasoning", "summary"],
        "final_answer": ["reasoning", "answer"],
    }
    assert hover["schema_version"] == 4
    assert hover["benchmark"] == "hover-train-wiki17"
    assert hover["retrieval"]["backend"] == "wiki17-bm25s"
    assert hover["retrieval"]["k1"] == 0.9
    assert hover["retrieval"]["b"] == 0.4
    assert hover["optimizer"]["reflection_minibatch_size"] == 3
    assert hover["optimizer"]["frontier_type"] == "instance"
    assert hover["optimizer"]["skip_perfect_score"] is True
    assert hover["program"]["parallel_workers"] == 32
    assert hover["program"]["retrieval_k"] == 7
    assert hover["program"]["final_retrieval_k"] == 10
    assert hover["program"]["predictor_type"] == "dspy_chain_of_thought"
    assert hover["program"]["predictor_adapter"] == "dspy_chat_adapter"
    assert hover["program"]["component_output_fields"] == {
        "summarize1": ["reasoning", "summary"],
        "create_query_hop2": ["reasoning", "query"],
        "summarize2": ["reasoning", "summary"],
        "create_query_hop3": ["reasoning", "query"],
    }
    assert hover["models"]["solver_decoding"] == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": 16_384,
    }

    assert hover_run_key("react_v2", hover_args) != hover_run_key("react_v2", _hotpot_args(final_retrieval_k=11))


def test_deepseek_contract_uses_the_deepseek_pair_and_local_request_settings() -> None:
    """Record DeepSeek decoding and maximum-reasoning template arguments."""
    args = _hotpot_args(
        solver_model=DEEPSEEK_V4_1_FLASH_MODEL,
        reflection_model=DEEPSEEK_V4_1_FLASH_MODEL,
        solver_api_base=LOCAL_API_BASE,
        reflection_api_base=LOCAL_API_BASE,
    )

    contract = build_hotpotqa_run_contract("react_v2", args)
    deepseek_decoding = {
        **experiment_decoding(DEEPSEEK_V4_1_FLASH_MODEL, agentic=False),
        "max_tokens": 65_536,
        "seed": 0,
    }
    deepseek_request_overrides = experiment_request_overrides(DEEPSEEK_V4_1_FLASH_MODEL)

    assert contract["models"] == {
        "solver": DEEPSEEK_V4_1_FLASH_MODEL,
        "solver_version": DEEPSEEK_V4_1_FLASH_REVISION,
        "solver_api_base": LOCAL_API_BASE,
        "solver_decoding": deepseek_decoding,
        "solver_request_overrides": {
            **deepseek_request_overrides,
            "extra_body": {**deepseek_request_overrides["extra_body"], "thinking_token_budget": 32_768},
        },
        "solver_num_retries": 0,
        "solver_request_timeout_seconds": 3600,
        "reflection": DEEPSEEK_V4_1_FLASH_MODEL,
        "reflection_version": DEEPSEEK_V4_1_FLASH_REVISION,
        "reflection_api_base": LOCAL_API_BASE,
        "reflection_decoding": {**deepseek_decoding, "max_tokens": 131_072},
        "reflection_role_decoding": {
            "controller": {
                "requested": {**deepseek_decoding, "max_tokens": 131_072},
                "provider_ignored_fields": [],
            },
            "manifestor": {
                "requested": {**deepseek_decoding, "max_tokens": 131_072},
                "provider_ignored_fields": [],
            },
            "react_v2_proposer": {
                "requested": {**deepseek_decoding, "max_tokens": 131_072},
                "provider_ignored_fields": [],
            },
        },
        "reflection_request_overrides": deepseek_request_overrides,
        "reflection_num_retries": 0,
        "reflection_request_timeout_seconds": 3600,
    }


def test_deepseek_serving_environment_is_material_to_contract_and_run_key(monkeypatch) -> None:
    """Persist local vLLM environment identity and isolate runtime changes.

    Args:
        monkeypatch: Pytest fixture used to change the serving environment digest.
    """
    for name, value in DEEPSEEK_SCIENTIFIC_RUNTIME.items():
        monkeypatch.setenv(name, value)
    args = _hotpot_args(
        solver_model=DEEPSEEK_V4_1_FLASH_MODEL,
        reflection_model=DEEPSEEK_V4_1_FLASH_MODEL,
        solver_api_base=LOCAL_API_BASE,
        reflection_api_base=LOCAL_API_BASE,
    )
    first_contract = build_hotpotqa_run_contract("react_v2", args)
    first_key = hotpotqa_run_key("react_v2", args)

    assert first_contract["execution_runtime"]["serving_engine"] == "vllm"
    assert first_contract["execution_runtime"]["serving_env_sha256"] == "1" * 64

    monkeypatch.setenv("HOTPOTQA_SERVING_ENV_SHA256", "4" * 64)

    assert hotpotqa_run_key("react_v2", args) != first_key


@pytest.mark.parametrize(
    "condition,budget", [(c, b) for b, cells in _SCIENTIFIC_CONDITIONS_BY_BUDGET.items() for c in cells]
)
def test_hotpot_teacher_student_all_roles_and_separate_runtime(monkeypatch, tmp_path, condition, budget):
    """Route all optimizer roles to DeepSeek and keep the original Qwen task prompts."""
    from examples.hotpotqa.main import TEACHER_RUNTIME_KEYS

    for name, value in QWEN_SCIENTIFIC_RUNTIME.items():
        monkeypatch.setenv(name, value)
    teacher = {
        "HOTPOTQA_" + key: DEEPSEEK_SCIENTIFIC_RUNTIME.get("HOTPOTQA_" + key, "fixture") for key in TEACHER_RUNTIME_KEYS
    }
    monkeypatch.setenv("HOTPOTQA_TEACHER_RUNTIME", json.dumps(teacher))
    args = _hotpot_args(
        condition=condition,
        max_metric_calls=budget,
        enforce_scientific_contract=True,
        reflection_model=DEEPSEEK_V4_1_FLASH_MODEL,
        solver_api_base=LOCAL_API_BASE,
        reflection_api_base="http://127.0.0.1:8201/v1",
        train_limit=None,
        val_limit=None,
        test_limit=None,
        data_identity=_scientific_data_identity(),
    )
    contract = build_hotpotqa_run_contract(condition, args)
    assert contract["models"]["solver"] == QWEN3_8_27B_MODEL
    assert contract["models"]["reflection"] == DEEPSEEK_V4_1_FLASH_MODEL
    assert contract["models"]["reflection_decoding"]["max_tokens"] == 131072
    assert contract["execution_runtime"]["teacher_runtime"] == teacher
    assert contract["optimizer"]["rendered_seed"] == hotpotqa_seed_candidate("2stage", "structured", "alibaba")
    config, selector = build_hotpotqa_config(condition, args, {}, str(tmp_path / condition))
    assert config.reflection.reflection_lm == DEEPSEEK_V4_1_FLASH_MODEL
    strategy = config.reflection.reflection_strategy
    if strategy is not None:
        assert strategy.base_lm.model == DEEPSEEK_V4_1_FLASH_MODEL
        assert strategy.controller_lm.model == DEEPSEEK_V4_1_FLASH_MODEL
        assert strategy.manifestor_lm.model == DEEPSEEK_V4_1_FLASH_MODEL
    if condition == "action":
        assert selector.lm.model == DEEPSEEK_V4_1_FLASH_MODEL
    first_key = hotpotqa_run_key(condition, args)
    teacher["HOTPOTQA_MODEL_INTEGRITY_SHA256"] = "e" * 64
    monkeypatch.setenv("HOTPOTQA_TEACHER_RUNTIME", json.dumps(teacher))
    assert hotpotqa_run_key(condition, args) != first_key
    teacher["HOTPOTQA_GPU_RUNTIME"] = h200_gpu_runtime(1)
    monkeypatch.setenv("HOTPOTQA_TEACHER_RUNTIME", json.dumps(teacher))
    with pytest.raises(ValueError, match="exactly 4 H200"):
        build_hotpotqa_run_contract(condition, args)


def test_hotpot_cross_model_requires_complete_teacher_identity(monkeypatch):
    """Reject a mixed run whose second serving environment has not been verified."""
    for name, value in QWEN_SCIENTIFIC_RUNTIME.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("HOTPOTQA_TEACHER_RUNTIME", raising=False)
    args = _hotpot_args(
        reflection_model=DEEPSEEK_V4_1_FLASH_MODEL,
        enforce_scientific_contract=True,
        train_limit=None,
        val_limit=None,
        test_limit=None,
        data_identity=_scientific_data_identity(),
    )
    with pytest.raises(ValueError, match="complete separate serving identity"):
        _validate_scientific_contract(args)


def test_experiment_model_pair_rejects_cross_model_runs() -> None:
    """Reject a Qwen student paired with the DeepSeek proposer."""
    with pytest.raises(ValueError, match="same model"):
        validate_experiment_model_pair(QWEN3_8_27B_MODEL, DEEPSEEK_V4_1_FLASH_MODEL)


def test_experiment_model_pair_rejects_unknown_models() -> None:
    """Reject homogeneous pairs outside the two configured experiment arms."""
    with pytest.raises(ValueError, match="Unsupported experiment model"):
        validate_experiment_model_pair("provider/unknown", "provider/unknown")


def test_hover_vanilla_and_react_v2_isolate_only_the_proposal_strategy() -> None:
    """Keep every non-treatment HoVer contract axis identical across the pair."""
    args = _hotpot_args(final_retrieval_k=10)
    vanilla = deepcopy(build_hover_run_contract("vanilla", args))
    react = deepcopy(build_hover_run_contract("react_v2", args))

    assert vanilla["models"] == react["models"]
    assert vanilla["program"] == react["program"]
    assert vanilla["retrieval"] == react["retrieval"]
    assert vanilla["data"] == react["data"]

    vanilla.pop("condition")
    react.pop("condition")
    treatment_fields = {
        "reflection_level",
        "edit_tool_set",
        "semantic_action_space",
        "semantic_controller_policy",
        "vanilla_reflection_prompt",
    }
    for field in treatment_fields:
        vanilla["optimizer"].pop(field)
        react["optimizer"].pop(field)

    assert vanilla == react


@pytest.mark.parametrize(
    ("contract_builder", "config_builder", "args"),
    [
        pytest.param(
            build_hotpotqa_run_contract,
            build_hotpotqa_config,
            _hotpot_args(),
            id="hotpotqa",
        ),
    ],
)
def test_random_controller_react_v2_changes_only_controller_selection(
    contract_builder,
    config_builder,
    args: SimpleNamespace,
) -> None:
    """Keep the three-role treatment fixed while replacing Controller ranking.

    Args:
        contract_builder: Benchmark run-contract factory under test.
        config_builder: Benchmark optimizer-configuration factory under test.
        args: Complete benchmark argument namespace.
    """
    verbalized_contract = deepcopy(contract_builder("react_v2", args))
    random_contract = deepcopy(contract_builder("react_v2_random", args))
    verbalized_config, verbalized_selector = config_builder("react_v2", args, {})
    random_config, random_selector = config_builder("react_v2_random", args, {})
    verbalized_strategy = verbalized_config.reflection.reflection_strategy
    random_strategy = random_config.reflection.reflection_strategy

    assert verbalized_selector is None
    assert random_selector is None
    assert verbalized_config.reflection.action_selector is None
    assert random_config.reflection.action_selector is None
    assert verbalized_strategy is not None
    assert random_strategy is not None
    assert verbalized_strategy.controller_selection == "verbalized"
    assert random_strategy.controller_selection == "uniform_random"
    assert verbalized_strategy.level == random_strategy.level == 2
    assert verbalized_strategy.edit_tool_set == random_strategy.edit_tool_set == "broad"
    assert verbalized_strategy.component_kinds == random_strategy.component_kinds
    assert verbalized_strategy.template_family == random_strategy.template_family
    assert verbalized_strategy.rng.getstate() == random_strategy.rng.getstate() == random.Random(args.seed).getstate()
    assert verbalized_contract["optimizer"]["semantic_controller_policy"] == CONTROLLER_POLICY_CONTRACT
    assert random_contract["optimizer"]["semantic_controller_policy"] == UNIFORM_RANDOM_CONTROLLER_POLICY_CONTRACT
    assert random_contract["optimizer"]["stateless_action_menu"] is None
    assert random_contract["optimizer"]["stateless_selector_policy"] is None

    verbalized_contract.pop("condition")
    random_contract.pop("condition")
    verbalized_contract["optimizer"].pop("semantic_controller_policy")
    random_contract["optimizer"].pop("semantic_controller_policy")
    if contract_builder is build_hotpotqa_run_contract:
        verbalized_roles = verbalized_contract["models"]["reflection_role_decoding"]
        random_roles = random_contract["models"]["reflection_role_decoding"]
        assert verbalized_roles["controller"] is not None
        assert random_roles["controller"] is None
        verbalized_roles["controller"] = None
    assert verbalized_contract == random_contract

    if contract_builder is build_hotpotqa_run_contract:
        assert hotpotqa_run_key("react_v2", args) != hotpotqa_run_key("react_v2_random", args)
        assert "l2-broad" in hotpotqa_run_key("react_v2_random", args)
    else:
        assert hover_run_key("react_v2", args) != hover_run_key("react_v2_random", args)
        assert "l2-broad" in hover_run_key("react_v2_random", args)


def test_hover_react_v2_uses_an_experiment_seeded_controller_rng() -> None:
    """Isolate Controller sampling from process-global random state."""
    args = _hotpot_args(final_retrieval_k=10, seed=19)
    reflection_kwargs = {"num_retries": EXPERIMENT_NUM_RETRIES, **experiment_decoding(args.reflection_model)}
    config, selector = build_hover_config("react_v2", args, reflection_kwargs)
    vanilla_config, vanilla_selector = build_hover_config("vanilla", args, reflection_kwargs)

    assert selector is None
    assert vanilla_selector is None
    assert config.engine.seed == 19
    assert config.engine.max_workers == 32
    assert config.engine.val_evaluation_policy == "full_eval"
    assert config.engine.candidate_selection_strategy == "pareto"
    assert config.engine.frontier_type == "instance"
    assert config.engine.acceptance_criterion == "strict_improvement"
    assert config.reflection.reflection_lm_kwargs == {
        "num_retries": EXPERIMENT_NUM_RETRIES,
        **experiment_decoding(QWEN3_8_27B_MODEL),
    }
    assert config.reflection.skip_perfect_score is True
    assert config.reflection.perfect_score == 1.0
    assert config.reflection.batch_sampler == "epoch_shuffled"
    assert config.reflection.reflection_minibatch_size == 3
    assert config.reflection.module_selector == "round_robin"
    assert config.reflection.reflection_prompt_template == InstructionProposalSignature.default_prompt_template
    assert config.reflection.reflection_strategy is not None
    assert config.reflection.reflection_strategy.rng.getstate() == random.Random(19).getstate()
    assert config.reflection.reflection_strategy.manifestor_lm.completion_kwargs["temperature"] == 0.0
    assert vanilla_config.engine.seed == config.engine.seed
    assert vanilla_config.engine.max_workers == config.engine.max_workers
    assert vanilla_config.engine.val_evaluation_policy == config.engine.val_evaluation_policy
    assert vanilla_config.engine.candidate_selection_strategy == config.engine.candidate_selection_strategy
    assert vanilla_config.engine.frontier_type == config.engine.frontier_type
    assert vanilla_config.engine.acceptance_criterion == config.engine.acceptance_criterion
    assert vanilla_config.reflection.skip_perfect_score == config.reflection.skip_perfect_score
    assert vanilla_config.reflection.perfect_score == config.reflection.perfect_score
    assert vanilla_config.reflection.batch_sampler == config.reflection.batch_sampler
    assert vanilla_config.reflection.reflection_minibatch_size == config.reflection.reflection_minibatch_size
    assert vanilla_config.reflection.module_selector == config.reflection.module_selector
    assert vanilla_config.reflection.reflection_prompt_template == config.reflection.reflection_prompt_template


@pytest.mark.parametrize("field", ["cache_evaluation", "dspy_disk_cache", "dspy_memory_cache"])
@pytest.mark.parametrize("missing", [False, True])
def test_hotpot_rejects_changed_or_unrecorded_caching(tmp_path: Path, field: str, missing: bool) -> None:
    """Reject checkpoints that could reuse earlier evaluations or model responses."""
    contract = build_hotpotqa_run_contract("react_v2", _hotpot_args())
    changed = deepcopy(contract)
    if missing:
        del changed["program"][field]
    else:
        changed["program"][field] = True
    ensure_wikipedia_run_contract(tmp_path, changed)
    with pytest.raises(ValueError, match="different Wikipedia benchmark configuration"):
        ensure_wikipedia_run_contract(tmp_path, contract)


def test_run_contract_rejects_drift_and_legacy_state(tmp_path: Path) -> None:
    """Accept exact resume state while rejecting drift and unversioned legacy state.

    Args:
        tmp_path: Pytest directory used for isolated run contracts.
    """
    contract = build_hotpotqa_run_contract("react_v2", _hotpot_args())
    run_dir = tmp_path / "run"

    path = ensure_wikipedia_run_contract(run_dir, contract)

    assert ensure_wikipedia_run_contract(run_dir, contract) == path
    with pytest.raises(ValueError, match="different Wikipedia benchmark configuration"):
        ensure_wikipedia_run_contract(run_dir, {**contract, "tag": "drift"})
    assert contract["provider_retry_policy"]["max_attempts"] == 3
    for missing in (True, False):
        changed = deepcopy(contract)
        if missing:
            del changed["provider_retry_policy"]
        else:
            changed["provider_retry_policy"]["max_attempts"] = 9
        with pytest.raises(ValueError, match="different Wikipedia benchmark configuration"):
            ensure_wikipedia_run_contract(run_dir, changed)

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    (legacy_dir / "gepa_state.bin").write_bytes(b"legacy")
    with pytest.raises(ValueError, match="GEPA state but no wikipedia-run-contract.json"):
        ensure_wikipedia_run_contract(legacy_dir, contract)


@pytest.mark.parametrize("dump_candidates", [dump_hotpotqa_candidates, dump_hover_candidates])
def test_candidate_artifacts_embed_run_contract(tmp_path: Path, dump_candidates) -> None:
    """Embed the exact run contract in both benchmark candidate artifacts.

    Args:
        tmp_path: Pytest directory receiving the artifact.
        dump_candidates: Parameterized benchmark artifact writer.
    """
    contract = build_hotpotqa_run_contract("react_v2", _hotpot_args())
    result = SimpleNamespace(
        best_idx=0,
        total_metric_calls=100,
        num_full_val_evals=1,
        candidates=[{"prompt": "candidate"}],
        parents=[None],
        val_aggregate_scores=[1.0],
        discovery_eval_counts=[1],
    )

    path = dump_candidates(result, str(tmp_path / dump_candidates.__module__.replace(".", "-")), contract)

    assert json.loads(Path(path).read_text())["run_contract"] == contract


@pytest.mark.parametrize("condition", ["random", "action"])
def test_stateless_conditions_use_the_canonical_provider_action_menu(condition: str) -> None:
    """Share one provider-aware semantic menu across stateless conditions.

    Args:
        condition: Random or verbalized stateless selector condition.
    """
    args = _hotpot_args()
    template = TEMPLATE_FAMILIES["alibaba"]["system_prompt"]
    contract = build_hotpotqa_run_contract(condition, args)
    config, selector = build_hotpotqa_config(condition, args, {})
    hover_config, hover_selector = build_hover_config(condition, _hotpot_args(final_retrieval_k=10), {})
    expected_menu = [
        StatelessActionConstraint(spec, section, template) for section in template.sections for spec in SEMANTIC_ACTIONS
    ]
    expected_contract = {
        "version": STATELESS_ACTION_MENU_VERSION,
        "semantic_action_catalog_version": SEMANTIC_ACTION_CATALOGS[template.kind]["version"],
        "kind": template.kind,
        "sections": list(template.sections),
        "choices": [
            {
                "id": choice.menu_id,
                "semantic_action": choice.semantic_action.name,
                "operator": choice.edit_tool.value,
                "target_section": choice.target_section,
            }
            for choice in expected_menu
        ],
    }

    assert contract["optimizer"]["semantic_action_space"] == SEMANTIC_ACTION_CATALOGS["prompt"]
    assert contract["optimizer"]["semantic_controller_policy"] is None
    assert contract["optimizer"]["stateless_action_menu"] == expected_contract
    assert contract["optimizer"]["stateless_selector_policy"] == stateless_selector_policy_contract(
        "random" if condition == "random" else "verbalized"
    )
    assert config.reflection.action_selector is selector
    assert hover_config.reflection.action_selector is hover_selector
    assert [choice.menu_id for choice in hover_selector.actions] == [choice.menu_id for choice in expected_menu]
    assert [choice.menu_id for choice in selector.actions] == [choice.menu_id for choice in expected_menu]
    assert {choice.target_section for choice in selector.actions} == set(template.sections)
    assert all(choice.edit_tool is choice.semantic_action.edit_tool for choice in selector.actions)


def test_stateless_action_menu_contract_matches_between_wikipedia_benchmarks() -> None:
    """Persist identical stateless action-menu contracts for HotPotQA and HOVER."""
    args = _hotpot_args(final_retrieval_k=10)
    expected = build_hotpotqa_run_contract("random", args)["optimizer"]["stateless_action_menu"]

    for build_contract, schema_version in (
        (build_hotpotqa_run_contract, 28),
        (build_hover_run_contract, 4),
    ):
        contract = build_contract("random", args)
        assert contract["schema_version"] == schema_version
        assert contract["optimizer"]["stateless_action_menu"] == expected
        assert contract["optimizer"]["stateless_selector_policy"] == stateless_selector_policy_contract("random")
        assert "legacy_actions" not in contract["optimizer"]
