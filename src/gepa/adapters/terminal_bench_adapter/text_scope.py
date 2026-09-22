"""Restrict editable text while retaining the same Terminal-Bench runtime harness."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from gepa.adapters.terminal_bench_adapter.documents import (
    COMPONENT_KINDS,
    INITIAL_COMPONENTS,
    render_initial_instructions,
    seed_documents,
    validate_documents,
)
from gepa.strategies.document_template import TEMPLATE_FAMILIES

DEFAULT_OPTIMIZATION_SCOPE = "system_prompt"
OPTIMIZATION_SCOPES = (DEFAULT_OPTIMIZATION_SCOPE, "all_text")


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
        return dict(COMPONENT_KINDS) if self.name == "all_text" else {"instruction_prompt": "user_prompt"}

    def seed_candidate(self) -> dict[str, str]:
        """Represent the common initial harness in the selected editable shape."""
        documents = seed_documents(self.template_family)
        if self.name == "all_text":
            return documents
        return {"instruction_prompt": render_initial_instructions(documents)}

    def materialize(self, candidate: Mapping[str, str]) -> dict[str, str]:
        """Validate editable keys and fill fixed text before invoking Harbor.

        Raises:
            ValueError: A candidate adds, removes, or uses non-text components.
        """
        if self.name == "all_text":
            validate_documents(candidate)
            return dict(candidate)
        if set(candidate) != set(self.component_kinds) or any(not isinstance(text, str) for text in candidate.values()):
            raise ValueError(
                f"Candidate must contain exactly the editable {self.name} components: {list(self.component_kinds)}"
            )
        documents = seed_documents(self.template_family)
        # These instructions are already inside the one editable prompt.
        documents.update(dict.fromkeys(INITIAL_COMPONENTS[1:], ""))
        documents.update(candidate)
        validate_documents(documents)
        return documents

    def contract(self) -> dict[str, Any]:
        """Record the edit boundary and common initial-prompt rendering policy."""
        return {
            "version": 1,
            "name": self.name,
            "editable_components": list(self.component_kinds),
            "initial_prompt_components": list(INITIAL_COMPONENTS),
            "initial_prompt_rendering": "nested_guidance_headings_v1",
            "fixed_components": []
            if self.name == "all_text"
            else [name for name in COMPONENT_KINDS if name not in INITIAL_COMPONENTS],
            "fixed_text_source": "provider_seed_documents",
        }
