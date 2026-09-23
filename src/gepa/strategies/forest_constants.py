"""Shared FOREST role names and strategy configuration choices."""

from typing import Final

VERBALIZED_SELECTION: Final = "verbalized"
UNIFORM_RANDOM_SELECTION: Final = "uniform_random"
REACT_EDITOR_MODE: Final = "react"
SINGLE_CALL_EDITOR_MODE: Final = "single_call"
MINIMAL_EDIT_TOOL_SET: Final = "minimal"
BROAD_EDIT_TOOL_SET: Final = "broad"

CONTROLLER_ROLE: Final = "controller"
MANIFESTOR_ROLE: Final = "manifestor"
EDITOR_ROLE: Final = "editor"
PROPOSER_ROLE: Final = "proposer"
SOLVER_ROLE: Final = "solver"
OPTIMIZER_ROLE: Final = "optimizer"

SEMANTIC_REFLECTION_LEVEL: Final = 2
DEFAULT_REFLECTION_LEVEL: Final = SEMANTIC_REFLECTION_LEVEL
DEFAULT_REFLECTION_MINIBATCH_SIZE: Final = 3
