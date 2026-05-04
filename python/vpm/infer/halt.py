"""§5 expected-value-of-computation halt rule."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HaltDecision:
    """Audit record for an eq. 101 halt decision."""

    should_halt: bool
    reason: str
    evc: float
    contract_met: bool
    clarification_dominates: bool
    expected_utility_gain: float
    total_penalty: float

    def to_dict(self) -> dict[str, float | bool | str]:
        """JSON-friendly halt decision."""
        return {
            "should_halt": self.should_halt,
            "reason": self.reason,
            "evc": self.evc,
            "contract_met": self.contract_met,
            "clarification_dominates": self.clarification_dominates,
            "expected_utility_gain": self.expected_utility_gain,
            "total_penalty": self.total_penalty,
        }


@dataclass(frozen=True)
class HaltWeights:
    """Penalty weights for eq. 101."""

    compute: float = 1.0
    loss: float = 1.0
    risk: float = 1.0
    support: float = 1.0
    context: float = 1.0
    semantic: float = 1.0
    source: float = 1.0
    rebuttal: float = 1.0
    realization: float = 1.0


DEFAULT_HALT_WEIGHTS = HaltWeights()


def halt_decision(
    *,
    certificate: float,
    threshold: float,
    expected_utility_gain: float,
    compute_delta: float = 0.0,
    loss_delta: float = 0.0,
    risk_delta: float = 0.0,
    support_delta: float = 0.0,
    context_delta: float = 0.0,
    semantic_delta: float = 0.0,
    source_delta: float = 0.0,
    rebuttal_delta: float = 0.0,
    realization_delta: float = 0.0,
    clarification_score: float = 0.0,
    best_action_score: float = 0.0,
    weights: HaltWeights = DEFAULT_HALT_WEIGHTS,
) -> HaltDecision:
    """Apply the contract/clarification/EVC halt rule from eq. 101."""
    total_penalty = (
        (weights.compute * compute_delta)
        + (weights.loss * loss_delta)
        + (weights.risk * risk_delta)
        + (weights.support * support_delta)
        + (weights.context * context_delta)
        + (weights.semantic * semantic_delta)
        + (weights.source * source_delta)
        + (weights.rebuttal * rebuttal_delta)
        + (weights.realization * realization_delta)
    )
    evc = expected_utility_gain - total_penalty
    contract_met = certificate >= threshold
    clarification_dominates = clarification_score > max(best_action_score, evc)
    if contract_met:
        return HaltDecision(
            True,
            "contract_met",
            evc,
            True,
            clarification_dominates,
            expected_utility_gain,
            total_penalty,
        )
    if clarification_dominates:
        return HaltDecision(
            True,
            "clarification_dominates",
            evc,
            contract_met,
            True,
            expected_utility_gain,
            total_penalty,
        )
    if evc <= 0.0:
        return HaltDecision(
            True,
            "evc_non_positive",
            evc,
            contract_met,
            clarification_dominates,
            expected_utility_gain,
            total_penalty,
        )
    return HaltDecision(
        False,
        "continue",
        evc,
        contract_met,
        clarification_dominates,
        expected_utility_gain,
        total_penalty,
    )


__all__ = ["DEFAULT_HALT_WEIGHTS", "HaltDecision", "HaltWeights", "halt_decision"]
