"""Replay-safe active-memory admission and demotion rules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdmissionThresholds:
    """Contract thresholds used by ``Admit_act``."""

    cert_eq: float = 1.0
    args_entropy: float = 1.0
    decode_cost: float = 4.0
    context_loss: float = 0.0
    semantic_loss: float = 0.0
    source_loss: float = 0.0
    rebuttal_loss: float = 0.0
    realization_loss: float = 0.0
    dependence_loss: float = 0.0


DEFAULT_ADMISSION_THRESHOLDS = AdmissionThresholds()


@dataclass(frozen=True)
class AdmissionEvidence:
    """Audited evidence for one active-memory admission decision."""

    frontier_lcb: float
    cert_act: bool
    cert_eq: float
    no_capability_escalation: bool
    replay_pass: bool
    args_entropy: float = 0.0
    decode_cost: float = 1.0
    context_loss: float = 0.0
    semantic_loss: float = 0.0
    source_loss: float = 0.0
    rebuttal_loss: float = 0.0
    realization_loss: float = 0.0
    dependence_loss: float = 0.0
    risk_ok: bool = True

    def to_dict(self) -> dict[str, float | bool]:
        """JSON-friendly admission evidence."""
        return {
            "frontier_lcb": self.frontier_lcb,
            "cert_act": self.cert_act,
            "cert_eq": self.cert_eq,
            "no_capability_escalation": self.no_capability_escalation,
            "replay_pass": self.replay_pass,
            "args_entropy": self.args_entropy,
            "decode_cost": self.decode_cost,
            "context_loss": self.context_loss,
            "semantic_loss": self.semantic_loss,
            "source_loss": self.source_loss,
            "rebuttal_loss": self.rebuttal_loss,
            "realization_loss": self.realization_loss,
            "dependence_loss": self.dependence_loss,
            "risk_ok": self.risk_ok,
        }


@dataclass(frozen=True)
class AdmissionDecision:
    """Admission rule result."""

    admitted: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly admission decision."""
        return {"admitted": self.admitted, "reasons": self.reasons}


@dataclass(frozen=True)
class DemotionEvidence:
    """Audited evidence for one active-memory demotion decision."""

    utility_ucb: float
    regression: bool = False
    stale: bool = False
    branch_explosion: bool = False
    scope_mismatch: bool = False
    capability_escalation: bool = False
    context_shift: bool = False
    semantic_shift: bool = False
    source_stale: bool = False
    rebuttal_stale: bool = False
    realization_drift: bool = False
    dependence_shift: bool = False
    frontier_delta: float = 0.0


def admit_active(
    evidence: AdmissionEvidence,
    thresholds: AdmissionThresholds = DEFAULT_ADMISSION_THRESHOLDS,
) -> AdmissionDecision:
    """Apply the active-memory admission rule from eq. 129."""
    reasons: list[str] = []
    if evidence.frontier_lcb <= 0.0:
        reasons.append("frontier lcb not positive")
    if not evidence.cert_act:
        reasons.append("active certificate missing")
    if evidence.cert_eq < thresholds.cert_eq:
        reasons.append("equivalence certificate below threshold")
    if not evidence.no_capability_escalation:
        reasons.append("capability escalation")
    if not evidence.replay_pass:
        reasons.append("replay gate failed")
    if evidence.args_entropy > thresholds.args_entropy:
        reasons.append("argument entropy too high")
    if evidence.decode_cost > thresholds.decode_cost:
        reasons.append("decode cost too high")
    if not evidence.risk_ok:
        reasons.append("risk budget exceeded")
    reasons.extend(loss_reasons(evidence, thresholds))
    return AdmissionDecision(not reasons, tuple(reasons))


def demote_active(evidence: DemotionEvidence) -> AdmissionDecision:
    """Apply the symmetric active-memory demotion rule from eq. 130."""
    reasons: list[str] = []
    if evidence.utility_ucb < 0.0:
        reasons.append("utility ucb negative")
    checks = (
        (evidence.regression, "regression"),
        (evidence.stale, "stale"),
        (evidence.branch_explosion, "branch explosion"),
        (evidence.scope_mismatch, "scope mismatch"),
        (evidence.capability_escalation, "capability escalation"),
        (evidence.context_shift, "context shift"),
        (evidence.semantic_shift, "semantic shift"),
        (evidence.source_stale, "source stale"),
        (evidence.rebuttal_stale, "rebuttal stale"),
        (evidence.realization_drift, "realization drift"),
        (evidence.dependence_shift, "dependence shift"),
    )
    reasons.extend(reason for triggered, reason in checks if triggered)
    if evidence.frontier_delta < 0.0:
        reasons.append("frontier regressed")
    return AdmissionDecision(bool(reasons), tuple(reasons))


def loss_reasons(
    evidence: AdmissionEvidence,
    thresholds: AdmissionThresholds,
) -> tuple[str, ...]:
    """Collect loss-budget violations."""
    checks = (
        (evidence.context_loss, thresholds.context_loss, "context loss too high"),
        (evidence.semantic_loss, thresholds.semantic_loss, "semantic loss too high"),
        (evidence.source_loss, thresholds.source_loss, "source loss too high"),
        (evidence.rebuttal_loss, thresholds.rebuttal_loss, "rebuttal loss too high"),
        (evidence.realization_loss, thresholds.realization_loss, "realization loss too high"),
        (evidence.dependence_loss, thresholds.dependence_loss, "dependence loss too high"),
    )
    return tuple(reason for observed, maximum, reason in checks if observed > maximum)


__all__ = [
    "DEFAULT_ADMISSION_THRESHOLDS",
    "AdmissionDecision",
    "AdmissionEvidence",
    "AdmissionThresholds",
    "DemotionEvidence",
    "admit_active",
    "demote_active",
]
