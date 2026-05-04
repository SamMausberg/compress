"""Dual-price budget allocation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetChannel:
    """One budget channel with estimated marginal terms."""

    name: str
    marginal_utility: float
    loss_penalty: float = 0.0
    risk_penalty: float = 0.0
    frontier_penalty: float = 0.0

    @property
    def net_return(self) -> float:
        """KKT left-derivative estimate from eq. 177."""
        return self.marginal_utility - self.loss_penalty - self.risk_penalty - self.frontier_penalty

    def to_dict(self) -> dict[str, float | str]:
        """JSON-friendly budget channel."""
        return {
            "name": self.name,
            "marginal_utility": self.marginal_utility,
            "loss_penalty": self.loss_penalty,
            "risk_penalty": self.risk_penalty,
            "frontier_penalty": self.frontier_penalty,
            "net_return": self.net_return,
        }


@dataclass(frozen=True)
class BudgetPlan:
    """Discrete budget-quanta allocation."""

    selected: tuple[BudgetChannel, ...]
    idle: tuple[BudgetChannel, ...]
    lambda0: float

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly budget plan."""
        return {
            "selected": [channel.to_dict() for channel in self.selected],
            "idle": [channel.to_dict() for channel in self.idle],
            "lambda0": self.lambda0,
        }


def allocate_budget_channels(
    channels: tuple[BudgetChannel, ...],
    *,
    quanta: int,
) -> BudgetPlan:
    """Allocate discrete budget only to positive-return channels."""
    ranked = tuple(
        sorted(channels, key=lambda channel: (channel.net_return, channel.name), reverse=True)
    )
    selected = tuple(channel for channel in ranked if channel.net_return > 0.0)[:quanta]
    idle = tuple(channel for channel in ranked if channel not in selected)
    lambda0 = min((channel.net_return for channel in selected), default=0.0)
    return BudgetPlan(selected, idle, lambda0)


__all__ = ["BudgetChannel", "BudgetPlan", "allocate_budget_channels"]
