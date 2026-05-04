"""Named §8 loss registry and weighted aggregation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

LOSS_NAMES = (
    "free_energy",
    "base",
    "cmp",
    "trace",
    "value",
    "repair",
    "halt",
    "ver",
    "cal",
    "safe",
    "mem",
    "supp",
    "render",
    "ctx",
    "sem",
    "src",
    "rebut",
    "ent",
    "real",
    "tb",
    "mf",
    "split",
    "sub",
    "dom",
    "dep",
    "front",
    "probe",
)


@dataclass(frozen=True)
class LossComponent:
    """One weighted loss component."""

    name: str
    value: float
    weight: float = 1.0
    blocked: bool = False
    reason: str = ""

    @property
    def contribution(self) -> float:
        """Weighted contribution, zeroed when a hard gate blocks the term."""
        return 0.0 if self.blocked else self.value * self.weight

    def to_dict(self) -> dict[str, float | bool | str]:
        """JSON-friendly loss component."""
        return {
            "name": self.name,
            "value": self.value,
            "weight": self.weight,
            "blocked": self.blocked,
            "reason": self.reason,
            "contribution": self.contribution,
        }


@dataclass(frozen=True)
class LossReport:
    """Weighted objective report over every registered loss."""

    components: tuple[LossComponent, ...]
    missing: tuple[str, ...]

    @property
    def total(self) -> float:
        """Total weighted objective."""
        return sum(component.contribution for component in self.components)

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly loss report."""
        return {
            "total": self.total,
            "missing": self.missing,
            "components": [component.to_dict() for component in self.components],
        }


def loss_report(
    values: Mapping[str, float],
    *,
    weights: Mapping[str, float] | None = None,
    blocked: Mapping[str, str] | None = None,
) -> LossReport:
    """Aggregate every registered loss name into one objective report."""
    active_weights = weights or {}
    active_blocked = blocked or {}
    components = tuple(
        LossComponent(
            name=name,
            value=values.get(name, 0.0),
            weight=active_weights.get(name, 1.0),
            blocked=name in active_blocked,
            reason=active_blocked.get(name, ""),
        )
        for name in LOSS_NAMES
    )
    missing = tuple(name for name in LOSS_NAMES if name not in values)
    return LossReport(components, missing)


__all__ = ["LOSS_NAMES", "LossComponent", "LossReport", "loss_report"]
