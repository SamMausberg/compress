"""Gradient-scale-normalized loss weights."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True)
class LossWeightState:
    """One adaptive loss-weight state."""

    name: str
    weight: float
    grad_ema: float
    lower: float = 0.0
    upper: float = 10.0

    def to_dict(self) -> dict[str, float | str]:
        """JSON-friendly weight state."""
        return {
            "name": self.name,
            "weight": self.weight,
            "grad_ema": self.grad_ema,
            "lower": self.lower,
            "upper": self.upper,
        }


def balance_loss_weights(
    states: tuple[LossWeightState, ...],
    gradient_norms: Mapping[str, float],
    *,
    alpha: float = 0.5,
    beta: float = 0.1,
    tau: float = 1.0e-8,
) -> tuple[LossWeightState, ...]:
    """Apply the gradient-scale normalization update from eq. 169."""
    updated_emas = {
        state.name: ((1.0 - beta) * state.grad_ema) + (beta * gradient_norms.get(state.name, 0.0))
        for state in states
    }
    target = median(updated_emas.values()) if updated_emas else 0.0
    return tuple(
        LossWeightState(
            name=state.name,
            weight=clamp(
                state.weight * ((target / (updated_emas[state.name] + tau)) ** alpha),
                state.lower,
                state.upper,
            ),
            grad_ema=updated_emas[state.name],
            lower=state.lower,
            upper=state.upper,
        )
        for state in states
    )


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a scalar to a closed interval."""
    return min(max(value, lower), upper)


__all__ = ["LossWeightState", "balance_loss_weights"]
