"""Harbor-side Terminus wrapper for candidate-specific prompt templates.

This module is imported by Harbor's Python 3.12 environment, not GEPA's
environment. Keep the boundary free of imports from ``gepa``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harbor.agents.terminus_2 import Terminus2


class PromptedTerminus(Terminus2):
    """Run Terminus 2 with GEPA's immutable per-evaluation prompt file.

    Args:
        logs_dir: Harbor agent log directory.
        prompt_template_path: Complete Terminus template rendered for this
            candidate evaluation.
        disable_skills: Ignore task-provided and agent-provided skills. This is
            fixed to ``True`` by the initial harness so the only agent tool is
            the standard Terminus tmux terminal interface.
        kwargs: Standard Harbor ``Terminus2`` options. The harness intentionally
            omits ``max_turns``, leaving the agent effectively unbounded within
            each task's pinned Harbor timeout.

    Raises:
        ValueError: The candidate prompt template is missing.
    """

    def __init__(
        self,
        logs_dir: Path,
        prompt_template_path: str,
        disable_skills: bool = True,
        **kwargs: Any,
    ) -> None:
        self._candidate_prompt_template_path = Path(prompt_template_path).expanduser().resolve()
        if not self._candidate_prompt_template_path.is_file():
            raise ValueError(f"candidate prompt template does not exist: {self._candidate_prompt_template_path}")

        injected_skills_dir = kwargs.pop("skills_dir", None)
        injected_mcp_servers = kwargs.pop("mcp_servers", None)
        super().__init__(
            logs_dir=logs_dir,
            skills_dir=None if disable_skills else injected_skills_dir,
            mcp_servers=[] if disable_skills else injected_mcp_servers,
            **kwargs,
        )

    def _get_prompt_template_path(self) -> Path:
        """Return the candidate-specific prompt path used during base initialization.

        Returns:
            Absolute path written by the GEPA-side Harbor runner.
        """
        return self._candidate_prompt_template_path
