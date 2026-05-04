"""Numerically stable scalar loss primitives."""

from __future__ import annotations

from math import isnan, log


def binary_cross_entropy(probability: float, target: bool) -> float:
    """Binary cross-entropy for a probability and boolean target."""
    p = clamp_probability(probability)
    return -log(p) if target else -log(1.0 - p)


def squared_error(prediction: float, target: float) -> float:
    """Squared L2 error."""
    error = prediction - target
    return error * error


def hinge(margin: float) -> float:
    """Positive-part hinge."""
    return max(0.0, margin)


def clamp_probability(probability: float) -> float:
    """Clamp probabilities away from singular log endpoints."""
    if isnan(probability):
        return 0.5
    return min(max(probability, 1.0e-12), 1.0 - 1.0e-12)


__all__ = ["binary_cross_entropy", "clamp_probability", "hinge", "squared_error"]
