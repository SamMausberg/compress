"""Constrained active-set selection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActiveSetBudget:
    """Hard active-set resource budgets."""

    latency: float
    memory: float
    capability_risk: float
    information: float


@dataclass(frozen=True)
class ActiveSetItem:
    """Candidate item for active-memory selection."""

    key: str
    utility_lcb: float
    latency_cost: float = 1.0
    memory_cost: float = 1.0
    capability_risk: float = 0.0
    information_delta: float = 0.0
    overlap_penalty: float = 0.0
    retrieval_entropy: float = 0.0
    mode_conflict: float = 0.0

    @property
    def score(self) -> float:
        """Greedy objective from eq. 131 without pairwise recomputation."""
        return self.utility_lcb - self.overlap_penalty - self.retrieval_entropy - self.mode_conflict

    def to_dict(self) -> dict[str, float | str]:
        """JSON-friendly active-set item."""
        return {
            "key": self.key,
            "utility_lcb": self.utility_lcb,
            "latency_cost": self.latency_cost,
            "memory_cost": self.memory_cost,
            "capability_risk": self.capability_risk,
            "information_delta": self.information_delta,
            "overlap_penalty": self.overlap_penalty,
            "retrieval_entropy": self.retrieval_entropy,
            "mode_conflict": self.mode_conflict,
            "score": self.score,
        }


@dataclass(frozen=True)
class ActiveSetSelection:
    """Selected active set and budget use."""

    selected: tuple[ActiveSetItem, ...]
    latency_used: float
    memory_used: float
    capability_risk_used: float
    information_used: float

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly active-set selection."""
        return {
            "selected": [item.to_dict() for item in self.selected],
            "latency_used": self.latency_used,
            "memory_used": self.memory_used,
            "capability_risk_used": self.capability_risk_used,
            "information_used": self.information_used,
        }


def select_active_set(
    items: tuple[ActiveSetItem, ...],
    budget: ActiveSetBudget,
) -> ActiveSetSelection:
    """Greedily solve the constrained active-set problem from eq. 131."""
    selected: list[ActiveSetItem] = []
    latency = 0.0
    memory = 0.0
    risk = 0.0
    information = 0.0
    for item in sorted(items, key=lambda candidate: (candidate.score, candidate.key), reverse=True):
        if item.score <= 0.0:
            continue
        if latency + item.latency_cost > budget.latency:
            continue
        if memory + item.memory_cost > budget.memory:
            continue
        if risk + item.capability_risk > budget.capability_risk:
            continue
        if information + item.information_delta > budget.information:
            continue
        selected.append(item)
        latency += item.latency_cost
        memory += item.memory_cost
        risk += item.capability_risk
        information += item.information_delta
    return ActiveSetSelection(tuple(selected), latency, memory, risk, information)


__all__ = [
    "ActiveSetBudget",
    "ActiveSetItem",
    "ActiveSetSelection",
    "select_active_set",
]
