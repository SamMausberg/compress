"""Executable projection from substrate state to operation proposals."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

from vpm.substrate.prototype import OPERATIONS, OperationProposal
from vpm.substrate.slots import SlotBinding


@dataclass(frozen=True)
class ProjectionTrace:
    """Audit trace for a substrate proposal projection."""

    proposal: OperationProposal
    scores: tuple[tuple[str, float], ...]
    support_loss: float

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly projection trace."""
        return {
            "proposal": self.proposal.to_dict(),
            "scores": self.scores,
            "support_loss": self.support_loss,
        }


def project_operation(
    binding: SlotBinding,
    candidate_scores: dict[str, float],
) -> ProjectionTrace:
    """Project slots to the highest-scoring executable operation proposal."""
    scores = tuple((operation, candidate_scores.get(operation, 0.0)) for operation in OPERATIONS)
    best_operation, best_score = max(scores, key=lambda item: (item[1], item[0]))
    confidence = softmax_confidence(best_score, tuple(score for _, score in scores))
    unsupported = sum(1 for operation in candidate_scores if operation not in OPERATIONS)
    support_loss = binding.omission_loss + (unsupported / max(len(candidate_scores), 1))
    return ProjectionTrace(OperationProposal(best_operation, confidence), scores, support_loss)


def softmax_confidence(score: float, scores: tuple[float, ...]) -> float:
    """Return softmax probability for one score."""
    if not scores:
        return 0.0
    offset = max(scores)
    weights = tuple(exp(value - offset) for value in scores)
    normalizer = sum(weights)
    return exp(score - offset) / normalizer if normalizer else 0.0


__all__ = ["ProjectionTrace", "project_operation"]
