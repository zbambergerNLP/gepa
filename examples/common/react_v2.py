"""Shared ReAct V2 experiment wiring for Wikipedia benchmarks."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gepa.lm import LM
from gepa.proposer.reflective_mutation.three_role import ThreeRoleReflectionLM
from gepa.strategies.document_template import TEMPLATE_FAMILIES, infer_template_family

_TASK_SECTIONS = {
    "system_prompt": {
        "generic": "Task",
        "openai": "Instructions",
        "anthropic": "Instructions",
        "google": "Instructions",
        "alibaba": "Objective",
    },
    "user_prompt": {
        "generic": "Task",
        "openai": "Input",
        "anthropic": "Instructions",
        "google": "Task",
        "alibaba": "Objective",
    },
}

WIKIPEDIA_RUN_CONTRACT_FILENAME = "wikipedia-run-contract.json"


def resolve_template_family(requested: str, task_model: str) -> str:
    """Resolve an explicit or model-inferred prompt-template family.

    Args:
        requested: Requested family name, or ``"auto"`` to infer it from the
            task model.
        task_model: Provider/model identifier used for automatic inference.

    Returns:
        The validated template-family name.

    Raises:
        ValueError: The requested or inferred family is not registered.
    """
    family = infer_template_family(task_model) if requested == "auto" else requested
    if family not in TEMPLATE_FAMILIES:
        raise ValueError(f"Unknown template family {family!r}; choose from {sorted(TEMPLATE_FAMILIES)} or 'auto'.")
    return family


def structured_prompt(task_sentence: str, template_family: str, component_kind: str = "system_prompt") -> str:
    """Render one role-specific seed without placeholder or empty sections.

    Args:
        task_sentence: Seed instruction to place in the template's task section.
        template_family: Registered provider template family.
        component_kind: Message role to render, either ``system_prompt`` or
            ``user_prompt``.

    Returns:
        Rendered prompt containing only populated sections.

    Raises:
        KeyError: ``template_family`` is not registered.
        ValueError: ``component_kind`` is not a supported prompt role.
    """
    if component_kind not in _TASK_SECTIONS:
        raise ValueError(f"structured_prompt requires a system_prompt or user_prompt; got {component_kind!r}.")
    template = TEMPLATE_FAMILIES[template_family][component_kind]
    return template.render({_TASK_SECTIONS[component_kind][template_family]: task_sentence})


def experiment_run_key(
    *,
    condition: str,
    template_family: str,
    reflection_level: int,
    edit_tool_set: str,
    settings: Mapping[str, Any],
) -> str:
    """Return a readable, stable key that prevents incompatible run resumption.

    Args:
        condition: Experiment condition represented by the key.
        template_family: Provider template family used for the run.
        reflection_level: Controller reflection level.
        edit_tool_set: Configured edit-operator set.
        settings: Remaining material run settings to fingerprint.

    Returns:
        Human-readable axes followed by a stable settings digest.
    """
    payload = {
        "condition": condition,
        "template_family": template_family,
        "reflection_level": reflection_level,
        "edit_tool_set": edit_tool_set,
        **settings,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:10]
    axes = template_family
    if condition == "react_v2":
        axes = f"{axes}-l{reflection_level}-{edit_tool_set}"
    return f"{axes}-{digest}"


def benchmark_data_identity(
    *,
    source: Mapping[str, Any],
    trainset: list[dict],
    valset: list[dict],
    testset: list[dict],
) -> dict[str, Any]:
    """Fingerprint the exact ordered records selected for a benchmark run.

    Args:
        source: Description of the benchmark data source.
        trainset: Ordered training records.
        valset: Ordered validation records.
        testset: Ordered test records.

    Returns:
        Source metadata plus count, ordered IDs, and content digest per split.
    """
    identity: dict[str, Any] = {"source": dict(source), "splits": {}}
    for name, records in (("train", trainset), ("val", valset), ("test", testset)):
        serialized = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        identity["splits"][name] = {
            "count": len(records),
            "ids": [str(record.get("id", "")) for record in records],
            "sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        }
    return identity


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a local benchmark source file.

    Args:
        path: File to hash in bounded chunks.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_wikipedia_run_contract(run_dir: str | Path, contract: Mapping[str, Any]) -> Path:
    """Persist an exact run contract and reject incompatible resume state.

    Args:
        run_dir: Experiment directory that owns the resumable state.
        contract: Complete material configuration for the requested run.

    Returns:
        Path to the existing or newly written contract file.

    Raises:
        ValueError: Existing state has a different contract, or legacy GEPA
            state has no contract to validate.
    """
    directory = Path(run_dir)
    path = directory / WIKIPEDIA_RUN_CONTRACT_FILENAME
    normalized = json.loads(json.dumps(dict(contract), sort_keys=True, default=str))
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != normalized:
            raise ValueError(f"Run directory {directory} contains a different Wikipedia benchmark configuration.")
        return path
    if (directory / "gepa_state.bin").exists():
        raise ValueError(
            f"Run directory {directory} has GEPA state but no {WIKIPEDIA_RUN_CONTRACT_FILENAME}; choose a clean directory."
        )
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n")
    return path


def build_react_v2_strategy(
    *,
    reflection_model: str,
    task_model: str,
    proposer_model: str | None = None,
    lm_kwargs: dict[str, Any],
    level: int,
    edit_tool_set: str,
    template_family: str,
    component_kinds: dict[str, str] | None = None,
    rng: random.Random | None = None,
) -> tuple[ThreeRoleReflectionLM, str]:
    """Build Controller -> Manifestor -> ReAct V2 with deterministic guidance.

    Args:
        reflection_model: Runtime model used by the Controller, Manifestor, and proposer.
        task_model: Student model whose provider determines automatic templates.
        proposer_model: Optional canonical proposer identity recorded separately
            from its API runtime model.
        lm_kwargs: Shared reflection-model client settings.
        level: Reflection level used to build the Controller menu.
        edit_tool_set: Named edit-operator basis exposed to the proposer.
        template_family: Explicit provider family or ``"auto"``.
        component_kinds: Optional message role for each optimized component.
        rng: Optional Controller RNG kept separate from GEPA's engine RNG.

    Returns:
        Configured three-role strategy and its resolved template family.
    """
    resolved_family = resolve_template_family(template_family, task_model)
    manifestor_kwargs = dict(lm_kwargs)
    manifestor_kwargs["temperature"] = 0
    strategy = ThreeRoleReflectionLM(
        base_lm=LM(reflection_model, **lm_kwargs),
        level=level,
        edit_tool_set=edit_tool_set,
        component_kinds=component_kinds,
        template_family=resolved_family,
        manifestor_lm=LM(reflection_model, **manifestor_kwargs),
        proposer_model=proposer_model or reflection_model,
        rng=rng,
    )
    return strategy, resolved_family
