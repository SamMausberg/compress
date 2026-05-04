"""Archival memory store for replay and sleep-phase candidates."""

from __future__ import annotations

from dataclasses import dataclass, field

from vpm.memory.active import MemoryCapsule, dict_capsules


@dataclass
class ArchivalMemory:
    """Replay material and non-active capsules."""

    capsules: dict[str, MemoryCapsule] = field(default_factory=dict_capsules)

    def add(self, capsule: MemoryCapsule) -> None:
        """Store a capsule in archival memory."""
        self.capsules[capsule.key] = capsule

    def __len__(self) -> int:
        """Return archived capsule count."""
        return len(self.capsules)


__all__ = ["ArchivalMemory"]
