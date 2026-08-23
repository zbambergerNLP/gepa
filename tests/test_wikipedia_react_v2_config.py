"""Tests for ReAct V2 wiring shared by the Wikipedia benchmarks."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from examples.common.react_v2 import (
    build_react_v2_strategy,
    ensure_wikipedia_run_contract,
    resolve_template_family,
    structured_prompt,
)
from examples.hotpotqa.main import _run_key as hotpotqa_run_key
from examples.hotpotqa.main import build_config as build_hotpotqa_config
from examples.hotpotqa.main import build_run_contract as build_hotpotqa_run_contract
from examples.hotpotqa.main import dump_candidates as dump_hotpotqa_candidates
from examples.hotpotqa.main import seed_candidate as hotpotqa_seed_candidate
from examples.hover.main import _run_key as hover_run_key
from examples.hover.main import build_config as build_hover_config
from examples.hover.main import build_run_contract as build_hover_run_contract
from examples.hover.main import dump_candidates as dump_hover_candidates
from examples.hover.main import seed_candidate as hover_seed_candidate
from gepa.strategies.document_template import TEMPLATE_FAMILIES
from gepa.strategies.intervention import controller_policy_contract, semantic_action_catalog


def test_student_model_selects_provider_specific_template() -> None:
    assert resolve_template_family("auto", "hosted_vllm/Qwen3.8") == "alibaba"


def test_structured_benchmark_seeds_follow_selected_template() -> None:
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
    """Keep seed content in the correct message-role schema without placeholders."""
    template = TEMPLATE_FAMILIES[family][component_kind]
    prompt = structured_prompt("Do the task.", family, component_kind)
    bodies = template.parse(prompt)
    assert bodies[task_section] == "Do the task."
    assert all(not body for section, body in bodies.items() if section != task_section)


def test_planned_model_roles_build_without_running_an_experiment() -> None:
    strategy, family = build_react_v2_strategy(
        reflection_model="deepseek/deepseek-v4-flash",
        task_model="hosted_vllm/Qwen3.8",
        lm_kwargs={"api_base": "http://localhost:8000/v1"},
        level=2,
        edit_tool_set="broad",
        template_family="auto",
        component_kinds={"summarize1": "system_prompt"},
    )

    assert family == "alibaba"
    assert strategy.proposer_model == "deepseek/deepseek-v4-flash"
    assert strategy.manifestor_injection_site == "user"
    assert strategy.component_kinds == {"summarize1": "system_prompt"}
    assert {tool.value for tool in strategy.edit_tools} == {
        "INSERT_TEXT",
        "DELETE_TEXT",
        "REPLACE_TEXT",
        "MOVE_TEXT",
    }


def test_openai_proposer_routes_manifestor_to_portable_user_message() -> None:
    strategy, _ = build_react_v2_strategy(
        reflection_model="openai/gpt-5",
        task_model="hosted_vllm/Qwen3.8",
        lm_kwargs={},
        level=2,
        edit_tool_set="minimal",
        template_family="auto",
    )

    assert strategy.manifestor_injection_site == "user"


def _hotpot_args(**overrides):
    values = {
        "actions": "structured",
        "api_base": None,
        "data_identity": {
            "source": {"type": "huggingface", "dataset": "hotpot_qa", "config": "fullwiki"},
            "splits": {"train": {"count": 150, "ids": ["train"], "sha256": "train-v1"}},
        },
        "data_path": None,
        "edit_tool_set": "broad",
        "max_metric_calls": 100,
        "program": "2stage",
        "reflection_api_base": "https://reflection.example/v1",
        "reflection_level": 2,
        "reflection_model": "deepseek/deepseek-v4-flash",
        "retrieval_k": 7,
        "seed": 0,
        "seed_style": "structured",
        "solver_api_base": "http://localhost:8000/v1",
        "solver_model": "hosted_vllm/Qwen3.8",
        "tag": "",
        "template_family": "auto",
        "test_limit": 300,
        "train_limit": 150,
        "val_limit": 300,
        "wikipedia_cache": "/tmp/gepa-wikipedia.sqlite3",
        "wikipedia_endpoint": "https://en.wikipedia.org/w/api.php",
        "wikipedia_timeout": 20.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_material_ablation_axes_get_distinct_resumable_run_keys() -> None:
    broad = hotpotqa_run_key("react_v2", _hotpot_args(edit_tool_set="broad"))
    minimal = hotpotqa_run_key("react_v2", _hotpot_args(edit_tool_set="minimal"))
    level_one = hotpotqa_run_key("react_v2", _hotpot_args(reflection_level=1))

    assert len({broad, minimal, level_one}) == 3
    assert "l2-broad" in broad


def test_complete_run_keys_cover_budget_seed_retrieval_and_data_identity() -> None:
    base = hotpotqa_run_key("vanilla", _hotpot_args())
    changed = {
        hotpotqa_run_key("vanilla", _hotpot_args(max_metric_calls=101)),
        hotpotqa_run_key("vanilla", _hotpot_args(solver_model="hosted_vllm/another-student")),
        hotpotqa_run_key("vanilla", _hotpot_args(reflection_model="deepseek/another-proposer")),
        hotpotqa_run_key("vanilla", _hotpot_args(solver_api_base="https://solver.example/v1")),
        hotpotqa_run_key("vanilla", _hotpot_args(reflection_api_base="https://other-reflection.example/v1")),
        hotpotqa_run_key("vanilla", _hotpot_args(seed_style="plain")),
        hotpotqa_run_key("vanilla", _hotpot_args(wikipedia_endpoint="https://example.test/w/api.php")),
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
    assert len(changed) == 8
    vanilla_contract = build_hotpotqa_run_contract("vanilla", _hotpot_args())
    assert vanilla_contract["optimizer"]["reflection_level"] == 0
    assert vanilla_contract["optimizer"]["semantic_action_space"] is None
    assert vanilla_contract["optimizer"]["semantic_controller_policy"] is None
    level_one_contract = build_hotpotqa_run_contract("react_v2", _hotpot_args(reflection_level=1))
    assert level_one_contract["optimizer"]["semantic_action_space"] is None
    assert level_one_contract["optimizer"]["semantic_controller_policy"] is None


def test_hotpot_and_hover_contracts_record_exact_model_roles() -> None:
    hotpot = build_hotpotqa_run_contract("react_v2", _hotpot_args())
    hover_args = _hotpot_args(final_retrieval_k=10)
    hover = build_hover_run_contract("react_v2", hover_args)

    for contract in (hotpot, hover):
        assert contract["schema_version"] == 2
        assert contract["models"] == {
            "solver": "hosted_vllm/Qwen3.8",
            "solver_api_base": "http://localhost:8000/v1",
            "reflection": "deepseek/deepseek-v4-flash",
            "reflection_api_base": "https://reflection.example/v1",
        }
        assert contract["optimizer"]["max_metric_calls"] == 100
        assert contract["optimizer"]["seed_style"] == "structured"
        assert set(contract["optimizer"]["component_kinds"].values()) == {"system_prompt"}
        assert contract["optimizer"]["semantic_action_space"] == semantic_action_catalog("prompt")
        assert contract["optimizer"]["semantic_controller_policy"] == controller_policy_contract()
        assert contract["retrieval"]["endpoint"] == "https://en.wikipedia.org/w/api.php"

    assert hover_run_key("react_v2", hover_args) != hover_run_key("react_v2", _hotpot_args(final_retrieval_k=11))


def test_hover_rlm_condition_builds_an_explicit_strategy_with_matched_budget() -> None:
    """Make RLM selectable without running it or changing the ReAct V2 primary path."""
    args = _hotpot_args(final_retrieval_k=10)

    contract = build_hover_run_contract("rlm", args)
    react_contract = build_hover_run_contract("react_v2", args)
    config, selector = build_hover_config("rlm", args, {}, run_dir="/tmp/gepa-hover-rlm-test")
    strategy = config.reflection.reflection_strategy

    assert selector is None
    assert strategy.proposer_backend == "rlm"
    assert strategy.level == 2
    assert strategy.edit_tool_set == "broad"
    assert set(strategy.component_kinds.values()) == {"system_prompt"}
    assert strategy.rlm_budget.max_model_calls == 8
    assert contract["optimizer"]["proposer_backend"] == "rlm"
    assert contract["optimizer"]["max_proposer_model_calls"] == 8
    assert contract["optimizer"]["rlm_budget"] == {
        "max_root_iterations": 4,
        "max_child_iterations": 2,
        "max_repl_calls": 6,
        "max_llm_queries": 2,
        "max_rlm_queries": 1,
        "max_recursion_depth": 1,
        "max_exec_seconds": 5,
        "max_output_chars": 4000,
    }
    assert react_contract["optimizer"]["proposer_backend"] == "react_v2"
    assert react_contract["optimizer"]["max_proposer_model_calls"] == 8
    assert react_contract["optimizer"]["rlm_budget"] is None
    assert hover_run_key("rlm", args) != hover_run_key("react_v2", args)


def test_run_contract_rejects_drift_and_legacy_state(tmp_path: Path) -> None:
    contract = build_hotpotqa_run_contract("react_v2", _hotpot_args())
    run_dir = tmp_path / "run"

    path = ensure_wikipedia_run_contract(run_dir, contract)

    assert ensure_wikipedia_run_contract(run_dir, contract) == path
    with pytest.raises(ValueError, match="different Wikipedia benchmark configuration"):
        ensure_wikipedia_run_contract(run_dir, {**contract, "tag": "drift"})

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    (legacy_dir / "gepa_state.bin").write_bytes(b"legacy")
    with pytest.raises(ValueError, match="GEPA state but no wikipedia-run-contract.json"):
        ensure_wikipedia_run_contract(legacy_dir, contract)


@pytest.mark.parametrize("dump_candidates", [dump_hotpotqa_candidates, dump_hover_candidates])
def test_candidate_artifacts_embed_run_contract(tmp_path: Path, dump_candidates) -> None:
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


def test_legacy_structured_condition_uses_student_provider_sections() -> None:
    _, selector = build_hotpotqa_config("random", _hotpot_args(), {})

    targeted_sections = {action.target_section for action in selector.actions if action.target_section is not None}
    assert targeted_sections == set(TEMPLATE_FAMILIES["alibaba"]["system_prompt"].sections)
