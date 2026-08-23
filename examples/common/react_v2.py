"""Shared ReAct V2 experiment wiring for Wikipedia benchmarks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gepa.lm import LM
from gepa.proposer.reflective_mutation.three_role import ThreeRoleReflectionLM
from gepa.strategies.document_template import TEMPLATE_FAMILIES, infer_template_family

_TASK_SECTIONS = {
    "generic": "Task",
    "openai": "Instructions",
    "openai-gpt-5.6": "Goal",
    "anthropic": "Instructions",
    "google": "Task",
    "alibaba": "Objective",
}

WIKIPEDIA_RUN_CONTRACT_FILENAME = "wikipedia-run-contract.json"


def resolve_template_family(requested: str, task_model: str) -> str:
    """Resolve an explicit or model-inferred prompt-template family."""
    family = infer_template_family(task_model) if requested == "auto" else requested
    if family not in TEMPLATE_FAMILIES:
        raise ValueError(f"Unknown template family {family!r}; choose from {sorted(TEMPLATE_FAMILIES)} or 'auto'.")
    return family


def structured_prompt(task_sentence: str, template_family: str) -> str:
    """Render one seed instruction in its provider-specific canonical template."""
    template = TEMPLATE_FAMILIES[template_family]["prompt"]
    task_section = _TASK_SECTIONS[template_family]
    bodies = dict.fromkeys(template.sections, "(not specified)")
    bodies[task_section] = task_sentence
    return template.render(bodies)


def experiment_run_key(
    *,
    condition: str,
    template_family: str,
    reflection_level: int,
    edit_tool_set: str,
    settings: Mapping[str, Any],
) -> str:
    """Return a readable, stable key that prevents incompatible run resumption."""
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
    """Fingerprint the exact ordered records selected for a benchmark run."""
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
    """Return the SHA-256 digest of a local benchmark source file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_wikipedia_run_contract(run_dir: str | Path, contract: Mapping[str, Any]) -> Path:
    """Persist an exact run contract and reject incompatible resume state."""
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
    lm_kwargs: dict[str, Any],
    level: int,
    edit_tool_set: str,
    template_family: str,
) -> tuple[ThreeRoleReflectionLM, str]:
    """Build Controller -> Manifestor -> ReAct V2 with model-aware routing."""
    resolved_family = resolve_template_family(template_family, task_model)
    manifestor_kwargs = dict(lm_kwargs)
    manifestor_kwargs["temperature"] = 0
    strategy = ThreeRoleReflectionLM(
        base_lm=LM(reflection_model, **lm_kwargs),
        level=level,
        edit_tool_set=edit_tool_set,
        template_family=resolved_family,
        manifestor_lm=LM(reflection_model, **manifestor_kwargs),
        proposer_model=reflection_model,
    )
    return strategy, resolved_family
