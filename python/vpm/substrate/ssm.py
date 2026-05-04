"""Selective SSM-style recurrent state update."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, tanh


@dataclass(frozen=True)
class SSMState:
    """Small recurrent state vector."""

    hidden: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly state."""
        return {"hidden": self.hidden}


@dataclass(frozen=True)
class SSMParameters:
    """Scalar selective-state parameters."""

    decay: float = 0.5
    input_gain: float = 1.0
    gate_gain: float = 1.0


DEFAULT_SSM_PARAMETERS = SSMParameters()


def run_selective_ssm(
    feature_rows: tuple[tuple[float, ...], ...],
    params: SSMParameters = DEFAULT_SSM_PARAMETERS,
) -> SSMState:
    """Run a selective recurrent update over encoded feature rows."""
    width = len(feature_rows[0]) if feature_rows else 0
    state = SSMState((0.0,) * width)
    for row in feature_rows:
        state = selective_step(state, row, params)
    return state


def selective_step(
    state: SSMState,
    features: tuple[float, ...],
    params: SSMParameters,
) -> SSMState:
    """Apply one gated state update."""
    gate = sigmoid(params.gate_gain * sum(abs(value) for value in features))
    hidden = tuple(
        tanh((params.decay * previous) + (gate * params.input_gain * current))
        for previous, current in zip(state.hidden, features, strict=True)
    )
    return SSMState(hidden)


def sigmoid(value: float) -> float:
    """Numerically stable logistic gate."""
    if value >= 0.0:
        scale = exp(-value)
        return 1.0 / (1.0 + scale)
    scale = exp(value)
    return scale / (1.0 + scale)


__all__ = [
    "DEFAULT_SSM_PARAMETERS",
    "SSMParameters",
    "SSMState",
    "run_selective_ssm",
    "selective_step",
]
