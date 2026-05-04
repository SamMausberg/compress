"""§5 uncertainty-action selection for active tests."""

from __future__ import annotations

from dataclasses import dataclass
from math import log2


@dataclass(frozen=True)
class TestSelectionWeights:
    """Weights for the eq. 100 active-test objective."""

    information: float = 0.4
    uncertainty: float = 0.4
    risk: float = 1.0
    hidden_bits: float = 0.1
    annoyance: float = 0.1
    support: float = 1.0
    context: float = 1.0
    semantic: float = 1.0
    source: float = 1.0
    rebuttal: float = 1.0
    realization: float = 1.0


DEFAULT_TEST_SELECTION_WEIGHTS = TestSelectionWeights()


@dataclass(frozen=True)
class TestAction:
    """One allowed active test over uncertainty."""

    name: str
    cost: float
    expected_cert_gain: float
    information_gain: float
    uncertainty_reduction: float
    risk: float = 0.0
    hidden_bits: float = 0.0
    annoyance: float = 0.0
    support_loss: float = 0.0
    context_loss: float = 0.0
    semantic_loss: float = 0.0
    source_loss: float = 0.0
    rebuttal_loss: float = 0.0
    realization_loss: float = 0.0

    def score(self, weights: TestSelectionWeights = DEFAULT_TEST_SELECTION_WEIGHTS) -> float:
        """Eq. 100-style value per unit cost."""
        if self.cost <= 0.0:
            return float("-inf")
        benefit = (
            self.expected_cert_gain
            + (weights.information * self.information_gain)
            + (weights.uncertainty * self.uncertainty_reduction)
        )
        penalty = (
            (weights.risk * self.risk)
            + (weights.hidden_bits * self.hidden_bits)
            + (weights.annoyance * self.annoyance)
            + (weights.support * self.support_loss)
            + (weights.context * self.context_loss)
            + (weights.semantic * self.semantic_loss)
            + (weights.source * self.source_loss)
            + (weights.rebuttal * self.rebuttal_loss)
            + (weights.realization * self.realization_loss)
        )
        return (benefit - penalty) / self.cost

    def to_dict(self) -> dict[str, float | str]:
        """JSON-friendly active-test action."""
        return {
            "name": self.name,
            "cost": self.cost,
            "expected_cert_gain": self.expected_cert_gain,
            "information_gain": self.information_gain,
            "uncertainty_reduction": self.uncertainty_reduction,
            "risk": self.risk,
            "hidden_bits": self.hidden_bits,
            "annoyance": self.annoyance,
            "support_loss": self.support_loss,
            "context_loss": self.context_loss,
            "semantic_loss": self.semantic_loss,
            "source_loss": self.source_loss,
            "rebuttal_loss": self.rebuttal_loss,
            "realization_loss": self.realization_loss,
        }


@dataclass(frozen=True)
class TestSelectionTrace:
    """Audit record for active-test selection."""

    selected: TestAction
    alternatives: tuple[TestAction, ...]
    selected_score: float
    alternative_scores: tuple[tuple[str, float], ...]

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly selection trace."""
        return {
            "selected": self.selected.to_dict(),
            "selected_score": self.selected_score,
            "alternative_scores": self.alternative_scores,
            "alternatives": [action.to_dict() for action in self.alternatives],
        }


def select_test(
    actions: tuple[TestAction, ...],
    weights: TestSelectionWeights = DEFAULT_TEST_SELECTION_WEIGHTS,
) -> TestSelectionTrace:
    """Select the allowed active test with highest eq. 100-style score."""
    if not actions:
        raise ValueError("at least one active-test action is required")
    scored = tuple((action, action.score(weights)) for action in actions)
    selected, selected_score = max(scored, key=lambda item: (item[1], item[0].name))
    return TestSelectionTrace(
        selected=selected,
        alternatives=actions,
        selected_score=selected_score,
        alternative_scores=tuple((action.name, score) for action, score in scored),
    )


def support_reduction_action(
    candidates_before: tuple[str, ...],
    candidates_after: tuple[str, ...],
    support_loss: float,
    *,
    name: str = "active-reveal-expected",
) -> TestAction:
    """Build the active-test action used by the executable C2 subset."""
    before = len(candidates_before)
    after = len(candidates_after)
    if before <= 0:
        information_gain = 0.0
        uncertainty_reduction = 0.0
    else:
        ratio = before / max(after, 1)
        information_gain = log2(ratio)
        uncertainty_reduction = max(0.0, (before - after) / before)
    return TestAction(
        name=name,
        cost=1.0,
        expected_cert_gain=1.0 if after == 1 else 0.0,
        information_gain=information_gain,
        uncertainty_reduction=uncertainty_reduction,
        support_loss=support_loss,
    )


__all__ = [
    "DEFAULT_TEST_SELECTION_WEIGHTS",
    "TestAction",
    "TestSelectionTrace",
    "TestSelectionWeights",
    "select_test",
    "support_reduction_action",
]
