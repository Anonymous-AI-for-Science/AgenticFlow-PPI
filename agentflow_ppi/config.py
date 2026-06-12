"""Configuration helpers for the AgentFlow-PPI artifact."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ArtifactPaths:
    """Centralized path container used by scripts and tests."""

    project_root: Path

    @property
    def code_root(self) -> Path:
        return self.project_root / "code"

    @property
    def paper_root(self) -> Path:
        return self.project_root / "paper"


