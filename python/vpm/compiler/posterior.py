"""Compiler posterior over executable alternatives."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

from vpm.compiler.score_head import proposal_score
from vpm.tasks.c0 import C0Task, C0Value

SUPPORTED_OPERATIONS = ("add", "mul", "concat", "eq")


@dataclass(frozen=True)
class CompilerAlternative:
    """One compiler posterior alternative."""

    operation: str
    type_valid: bool
    value_matches: bool
    score: float
    probability: float
    support_loss: float

    def to_dict(self) -> dict[str, float | bool | str]:
        """JSON-friendly alternative."""
        return {
            "operation": self.operation,
            "type_valid": self.type_valid,
            "value_matches": self.value_matches,
            "score": self.score,
            "probability": self.probability,
            "support_loss": self.support_loss,
        }


@dataclass(frozen=True)
class CompilerPosterior:
    """Normalized posterior over compiler alternatives."""

    task_id: str
    alternatives: tuple[CompilerAlternative, ...]

    @property
    def selected(self) -> CompilerAlternative | None:
        """Highest-probability alternative."""
        if not self.alternatives:
            return None
        return max(
            self.alternatives,
            key=lambda alternative: (alternative.probability, alternative.operation),
        )

    @property
    def support_loss(self) -> float:
        """Posterior mass outside the selected candidate."""
        selected = self.selected
        return 1.0 - selected.probability if selected else 1.0

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly posterior."""
        return {
            "task_id": self.task_id,
            "selected": self.selected.to_dict() if self.selected else None,
            "support_loss": self.support_loss,
            "alternatives": [alternative.to_dict() for alternative in self.alternatives],
        }


def compiler_posterior(
    task: C0Task,
    operations: tuple[str, ...] = SUPPORTED_OPERATIONS,
) -> CompilerPosterior:
    """Build a normalized posterior over type-valid executable alternatives."""
    raw = tuple(raw_alternative(task, operation) for operation in operations)
    valid = tuple(alternative for alternative in raw if alternative.type_valid)
    scores = tuple(alternative.score for alternative in valid)
    probabilities = softmax(scores)
    alternatives = tuple(
        CompilerAlternative(
            operation=alternative.operation,
            type_valid=alternative.type_valid,
            value_matches=alternative.value_matches,
            score=alternative.score,
            probability=probability,
            support_loss=1.0 - probability,
        )
        for alternative, probability in zip(valid, probabilities, strict=True)
    )
    return CompilerPosterior(task.task_id, alternatives)


def raw_alternative(task: C0Task, operation: str) -> CompilerAlternative:
    """Score one unnormalized compiler alternative."""
    type_valid = operation_applicable(operation, task.left, task.right)
    value_matches = False
    if type_valid:
        value_matches = same_typed_value(
            candidate_value(operation, task.left, task.right), task.expected
        )
    score = proposal_score(type_valid=type_valid, value_matches=value_matches)
    return CompilerAlternative(operation, type_valid, value_matches, score, 0.0, 1.0)


def candidate_value(operation: str, left: C0Value, right: C0Value) -> C0Value:
    """Evaluate one type-valid C0 candidate operation."""
    if operation == "add":
        return int(left) + int(right)
    if operation == "mul":
        return int(left) * int(right)
    if operation == "concat":
        return str(left) + str(right)
    if operation == "eq":
        return left == right
    raise ValueError(f"unsupported operation: {operation}")


def operation_applicable(operation: str, left: C0Value, right: C0Value) -> bool:
    """Return true when an operation is type-valid for the operands."""
    if operation in {"add", "mul"}:
        return (
            isinstance(left, int)
            and isinstance(right, int)
            and not isinstance(left, bool)
            and not isinstance(right, bool)
        )
    if operation == "concat":
        return isinstance(left, str) and isinstance(right, str)
    return operation == "eq"


def same_typed_value(left: C0Value, right: C0Value) -> bool:
    """Compare without bool/int coercion."""
    return type(left) is type(right) and left == right


def softmax(scores: tuple[float, ...]) -> tuple[float, ...]:
    """Normalize scores into probabilities."""
    if not scores:
        return ()
    offset = max(scores)
    weights = tuple(exp(score - offset) for score in scores)
    normalizer = sum(weights)
    return tuple(weight / normalizer for weight in weights)


__all__ = [
    "SUPPORTED_OPERATIONS",
    "CompilerAlternative",
    "CompilerPosterior",
    "candidate_value",
    "compiler_posterior",
    "operation_applicable",
    "same_typed_value",
]
