"""Restrict editable text while retaining the same Terminal-Bench runtime harness."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from gepa.adapters.terminal_bench_adapter.documents import (
    COMPONENT_KINDS,
    INITIAL_COMPONENTS,
    seed_documents,
    validate_documents,
)
from gepa.strategies.document_template import TEMPLATE_FAMILIES

DEFAULT_OPTIMIZATION_SCOPE = "system_prompt"
OPTIMIZATION_SCOPES = (DEFAULT_OPTIMIZATION_SCOPE,)


@dataclass(frozen=True)
class TerminalBenchTextScope:
    """Map editable candidates to a complete harness with fixed auxiliary text."""

    name: str = DEFAULT_OPTIMIZATION_SCOPE
    template_family: str = "generic"

    def __post_init__(self) -> None:
        """Reject unknown scopes and template families before any model work."""
        if self.name not in OPTIMIZATION_SCOPES:
            raise ValueError(f"optimization scope must be one of {OPTIMIZATION_SCOPES}")
        if self.template_family not in TEMPLATE_FAMILIES:
            raise ValueError(f"Unknown template family: {self.template_family}")

    @property
    def component_kinds(self) -> dict[str, str]:
        """Expose only the components the selected scope allows the optimizer to edit."""
        return {"system_prompt": "system_prompt"}

    def seed_candidate(self) -> dict[str, str]:
        """Represent the common initial harness in the selected editable shape."""
        documents = seed_documents(self.template_family)
        return {"system_prompt": documents["instruction_prompt"]}

    def materialize(self, candidate: Mapping[str, str]) -> dict[str, str]:
        """Validate editable keys and fill fixed text before invoking Harbor.

        Raises:
            ValueError: A candidate adds, removes, or uses non-text components.
        """
        if set(candidate) != set(self.component_kinds) or any(
            not isinstance(text, str) or not text.strip() for text in candidate.values()
        ):
            raise ValueError(
                f"Candidate must contain exactly the editable {self.name} components: {list(self.component_kinds)}"
            )
        documents = seed_documents(self.template_family)
        documents["instruction_prompt"] = candidate["system_prompt"]
        validate_documents(documents)
        return documents

    def contract(self) -> dict[str, Any]:
        """Record the edit boundary and common initial-prompt rendering policy."""
        return {
            "version": 2,
            "name": self.name,
            "editable_components": list(self.component_kinds),
            "initial_prompt_components": list(INITIAL_COMPONENTS),
            "candidate_message_role": "system",
            "initial_prompt_rendering": "fixed_user_guidance_and_task_v2",
            "fixed_components": [name for name in COMPONENT_KINDS if name != "instruction_prompt"],
            "fixed_text_source": "provider_seed_documents",
        }
