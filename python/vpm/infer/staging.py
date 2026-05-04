"""§5 staged resolution scheduler for inference."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class InferenceStage(StrEnum):
    """Resolution stages from eq. 97."""

    INVARIANT = "iota"
    SKETCH = "sigma"
    PROGRAM = "pi"
    RESIDUAL = "eta"


STAGE_ORDER = (
    InferenceStage.INVARIANT,
    InferenceStage.SKETCH,
    InferenceStage.PROGRAM,
    InferenceStage.RESIDUAL,
)


@dataclass(frozen=True)
class StageWeights:
    """Penalty weights for the eq. 98 stage-entry objective."""

    cost: float = 1.0
    loss: float = 1.0
    risk: float = 1.0
    support: float = 1.0
    context: float = 1.0
    semantic: float = 1.0
    source: float = 1.0
    rebuttal: float = 1.0
    realization: float = 1.0


DEFAULT_STAGE_WEIGHTS = StageWeights()


@dataclass(frozen=True)
class StageTransition:
    """One proposed adjacent stage transition."""

    name: str
    source: InferenceStage
    target: InferenceStage
    expected_cert_gain: float
    expected_utility_gain: float
    cost_delta: float = 0.0
    loss_delta: float = 0.0
    risk_delta: float = 0.0
    support_delta: float = 0.0
    context_delta: float = 0.0
    semantic_delta: float = 0.0
    source_delta: float = 0.0
    rebuttal_delta: float = 0.0
    realization_delta: float = 0.0
    independent_verifier_certs: tuple[float, ...] = ()
    authority_ok: bool = True

    def score(self, weights: StageWeights = DEFAULT_STAGE_WEIGHTS) -> float:
        """Eq. 98-style net value for entering the next stage."""
        benefit = self.expected_cert_gain + self.expected_utility_gain
        penalty = (
            (weights.cost * self.cost_delta)
            + (weights.loss * self.loss_delta)
            + (weights.risk * self.risk_delta)
            + (weights.support * self.support_delta)
            + (weights.context * self.context_delta)
            + (weights.semantic * self.semantic_delta)
            + (weights.source * self.source_delta)
            + (weights.rebuttal * self.rebuttal_delta)
            + (weights.realization * self.realization_delta)
        )
        return benefit - penalty

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly transition."""
        return {
            "name": self.name,
            "source": self.source.value,
            "target": self.target.value,
            "expected_cert_gain": self.expected_cert_gain,
            "expected_utility_gain": self.expected_utility_gain,
            "cost_delta": self.cost_delta,
            "loss_delta": self.loss_delta,
            "risk_delta": self.risk_delta,
            "support_delta": self.support_delta,
            "context_delta": self.context_delta,
            "semantic_delta": self.semantic_delta,
            "source_delta": self.source_delta,
            "rebuttal_delta": self.rebuttal_delta,
            "realization_delta": self.realization_delta,
            "independent_verifier_certs": self.independent_verifier_certs,
            "authority_ok": self.authority_ok,
        }


@dataclass(frozen=True)
class StageTransitionDecision:
    """Audit record for one stage-entry decision."""

    transition: StageTransition
    score: float
    entered: bool
    reason: str
    certificate_cap: float | None
    residual_escrowed: bool

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly stage transition decision."""
        return {
            "transition": self.transition.to_dict(),
            "score": self.score,
            "entered": self.entered,
            "reason": self.reason,
            "certificate_cap": self.certificate_cap,
            "residual_escrowed": self.residual_escrowed,
        }


@dataclass(frozen=True)
class StageScheduleTrace:
    """Audit trace for a sequence of stage decisions."""

    start_stage: InferenceStage
    final_stage: InferenceStage
    decisions: tuple[StageTransitionDecision, ...]

    def reached(self, stage: InferenceStage) -> bool:
        """Return true when the final stage is at or beyond ``stage``."""
        return stage_rank(self.final_stage) >= stage_rank(stage)

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly stage schedule."""
        return {
            "start_stage": self.start_stage.value,
            "final_stage": self.final_stage.value,
            "decisions": [decision.to_dict() for decision in self.decisions],
        }


def schedule_stages(
    transitions: tuple[StageTransition, ...],
    *,
    start: InferenceStage = InferenceStage.INVARIANT,
    weights: StageWeights = DEFAULT_STAGE_WEIGHTS,
) -> StageScheduleTrace:
    """Enter adjacent stages while the eq. 98 value is positive."""
    current = start
    decisions: list[StageTransitionDecision] = []
    for transition in transitions:
        decision = decide_stage_transition(transition, current, weights)
        decisions.append(decision)
        if not decision.entered:
            break
        current = transition.target
    return StageScheduleTrace(start, current, tuple(decisions))


def decide_stage_transition(
    transition: StageTransition,
    current: InferenceStage,
    weights: StageWeights = DEFAULT_STAGE_WEIGHTS,
) -> StageTransitionDecision:
    """Evaluate one stage transition against order and value constraints."""
    score = transition.score(weights)
    cap = residual_certificate_cap(transition)
    escrowed = transition.target is InferenceStage.RESIDUAL and cap == 0.0
    if transition.source is not current:
        return StageTransitionDecision(
            transition,
            score,
            False,
            "source_not_current",
            cap,
            escrowed,
        )
    if not adjacent_stage(transition.source, transition.target):
        return StageTransitionDecision(
            transition,
            score,
            False,
            "non_adjacent_transition",
            cap,
            escrowed,
        )
    if score <= 0.0:
        return StageTransitionDecision(
            transition,
            score,
            False,
            "non_positive_value",
            cap,
            escrowed,
        )
    reason = "entered_escrowed" if escrowed else "entered"
    return StageTransitionDecision(transition, score, True, reason, cap, escrowed)


def residual_certificate_cap(transition: StageTransition) -> float | None:
    """Return the eq. 99 certificate cap for residual-stage transitions."""
    if transition.target is not InferenceStage.RESIDUAL:
        return None
    if not transition.authority_ok or not transition.independent_verifier_certs:
        return 0.0
    return min(transition.independent_verifier_certs)


def adjacent_stage(source: InferenceStage, target: InferenceStage) -> bool:
    """Return true for the only allowed forward transition."""
    return stage_rank(target) == stage_rank(source) + 1


def stage_rank(stage: InferenceStage) -> int:
    """Return the ordinal rank of a stage in eq. 97."""
    return STAGE_ORDER.index(stage)


__all__ = [
    "DEFAULT_STAGE_WEIGHTS",
    "InferenceStage",
    "StageScheduleTrace",
    "StageTransition",
    "StageTransitionDecision",
    "StageWeights",
    "schedule_stages",
]
