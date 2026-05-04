"""Shared curriculum-stage metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageSpec:
    """Runtime-visible curriculum stage contract."""

    name: str
    summary: str
    executable: bool
    implemented_components: tuple[str, ...]
    blockers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly stage metadata."""
        return {
            "name": self.name,
            "summary": self.summary,
            "executable": self.executable,
            "implemented_components": self.implemented_components,
            "blockers": self.blockers,
        }


__all__ = ["StageSpec"]
