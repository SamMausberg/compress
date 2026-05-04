"""Executable scalar components for hard-gated losses."""

from __future__ import annotations

from vpm.training.losses.primitives import binary_cross_entropy, hinge, squared_error


def split_loss(
    leak_probability: float,
    split_clean: bool,
    *,
    audit_selected: bool = False,
    audit_weight: float = 1.0,
) -> float:
    """Eq. 161 split predictor loss with audit-selection penalty."""
    leak_target = not split_clean
    audit_penalty = audit_weight if audit_selected else 0.0
    return binary_cross_entropy(leak_probability, leak_target) + audit_penalty


def support_loss(
    epsilon_prune: float,
    epsilon_max: float,
    keep_probability: float,
    oracle_keep: bool,
) -> float:
    """Eq. 164 support recall loss."""
    return hinge(epsilon_prune - epsilon_max) + binary_cross_entropy(
        keep_probability,
        oracle_keep,
    )


def frontier_loss(
    predicted_delta: float,
    actual_delta: float,
    admit_probability: float,
    *,
    duplicate_penalty: float = 0.0,
    distortion_excess: float = 0.0,
) -> float:
    """Eq. 146 frontier/admission prediction loss."""
    return (
        squared_error(predicted_delta, actual_delta)
        + binary_cross_entropy(admit_probability, actual_delta > 0.0)
        + duplicate_penalty
        + hinge(distortion_excess)
    )


def trace_loss(
    action_probabilities: tuple[float, ...],
    importance_weights: tuple[float, ...],
    *,
    effective_sample_size: float,
    minimum_ess: float,
) -> float:
    """Eqs. 155-156 trace imitation loss with ESS gating."""
    if effective_sample_size < minimum_ess:
        return 0.0
    terms = (
        weight * binary_cross_entropy(probability, True)
        for probability, weight in zip(action_probabilities, importance_weights, strict=True)
    )
    return sum(terms)


__all__ = ["frontier_loss", "split_loss", "support_loss", "trace_loss"]
