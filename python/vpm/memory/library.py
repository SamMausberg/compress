"""Hierarchical memory library facade."""

from __future__ import annotations

from dataclasses import dataclass, field

from vpm._reports import float_field, object_list, object_map
from vpm.memory.active import MemoryCapsule, dict_capsules


@dataclass
class MemoryLibrary:
    """Two-tier active/archive memory used by the MVP workflow."""

    active: dict[str, MemoryCapsule] = field(default_factory=dict_capsules)
    archive: dict[str, MemoryCapsule] = field(default_factory=dict_capsules)

    def admit(self, key: str, value: object, report: dict[str, object]) -> MemoryCapsule | None:
        """Admit only gate-passed reports with equivalence witness accounting."""
        gate = object_map(report.get("gate"))
        if gate is None or gate.get("passed") is not True:
            return None
        score = float_field(gate, "certificate_score")
        canonical = report.get("canonical")
        witnesses = witness_names(canonical)
        capsule = MemoryCapsule(key, value, score, witnesses)
        if witnesses or score > 0.0:
            self.active[key] = capsule
        else:
            self.archive[key] = capsule
        return capsule


def witness_names(canonical: object) -> tuple[str, ...]:
    """Extract canonicalization witness names from a native report."""
    canonical_map = object_map(canonical)
    if canonical_map is None:
        return ()
    witnesses = object_list(canonical_map.get("witnesses"))
    if witnesses is None:
        return ()
    names: list[str] = []
    for witness in witnesses:
        witness_map = object_map(witness)
        if witness_map is None:
            continue
        rule = witness_map.get("rule")
        if isinstance(rule, str):
            names.append(rule)
    return tuple(names)


__all__ = ["MemoryLibrary", "witness_names"]
