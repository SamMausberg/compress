"""Bounded compiler proposal scoring."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreWeights:
    """Scalar weights for executable proposal scoring."""

    type_valid: float = 1.0
    value_match: float = 2.0
    parser_loss: float = 1.0
    context_loss: float = 1.0
    semantic_loss: float = 1.0
    realization_loss: float = 1.0


DEFAULT_SCORE_WEIGHTS = ScoreWeights()


def proposal_score(
    *,
    type_valid: bool,
    value_matches: bool,
    parser_loss: float = 0.0,
    context_loss: float = 0.0,
    semantic_loss: float = 0.0,
    realization_loss: float = 0.0,
    weights: ScoreWeights = DEFAULT_SCORE_WEIGHTS,
) -> float:
    """Eq. 88-style bounded proposal score."""
    benefit = (weights.type_valid if type_valid else 0.0) + (
        weights.value_match if value_matches else 0.0
    )
    penalty = (
        (weights.parser_loss * parser_loss)
        + (weights.context_loss * context_loss)
        + (weights.semantic_loss * semantic_loss)
        + (weights.realization_loss * realization_loss)
    )
    return benefit - penalty


__all__ = ["DEFAULT_SCORE_WEIGHTS", "ScoreWeights", "proposal_score"]
