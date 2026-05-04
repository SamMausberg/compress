"""Active teacher-query scoring."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActiveQuery:
    """One teacher-query candidate from eq. 178."""

    name: str
    entropy_reduction: float
    expected_certified_utility: float
    regression_probability: float
    cost: float
    risk_cost: float = 0.0

    @property
    def score(self) -> float:
        """Value per total query cost."""
        total_cost = self.cost + self.risk_cost
        if total_cost <= 0.0:
            return float("-inf")
        benefit = (
            self.entropy_reduction + self.expected_certified_utility + self.regression_probability
        )
        return benefit / total_cost


def select_active_query(queries: tuple[ActiveQuery, ...]) -> ActiveQuery:
    """Select the highest-scoring active teacher query."""
    if not queries:
        raise ValueError("at least one active query is required")
    return max(queries, key=lambda query: (query.score, query.name))


__all__ = ["ActiveQuery", "select_active_query"]
