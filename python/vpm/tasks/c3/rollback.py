"""C3 cumulative-risk ledger with certified rollback credits."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

RiskVector = dict[str, float]

ABSORBING_CHANNELS = frozenset(("privacy", "capability", "exfiltration", "influence"))
ROLLBACK_CERT_THRESHOLD = 1.0


class RiskChannel(StrEnum):
    """Componentwise risk channels from §6 eq. 110."""

    IMPACT = "impact"
    PRIVACY = "privacy"
    EXFILTRATION = "exfiltration"
    CAPABILITY = "capability"
    INFLUENCE = "influence"
    CONFLICT = "conflict"
    MODEL = "model"
    DEPENDENCE = "dependence"


def risk_vector() -> RiskVector:
    """Typed default factory for risk-vector fields."""
    return {}


@dataclass(frozen=True)
class RollbackPlan:
    """Certified rollback credits for one action."""

    credits: RiskVector = field(default_factory=risk_vector)
    monitor_certificates: RiskVector = field(default_factory=risk_vector)
    reversal_witnesses: tuple[str, ...] = ()


@dataclass(frozen=True)
class RollbackAction:
    """One action in the cumulative-risk ledger."""

    action_id: str
    risk_delta: RiskVector
    credit_cap: RiskVector = field(default_factory=risk_vector)
    rollback: RollbackPlan = field(default_factory=RollbackPlan)
    expected_pass: bool = True


@dataclass(frozen=True)
class RollbackLedgerEntry:
    """Audited ledger row after one rollback-credit update."""

    action_id: str
    risk_delta: RiskVector
    requested_credit: RiskVector
    applied_credit: RiskVector
    cumulative_risk: RiskVector
    passed: bool
    errors: tuple[str, ...]
    expected_pass: bool

    @property
    def violation(self) -> bool:
        """True when the ledger row disagrees with the expected policy."""
        return self.passed != self.expected_pass

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly rollback ledger row."""
        return {
            "action_id": self.action_id,
            "risk_delta": self.risk_delta,
            "requested_credit": self.requested_credit,
            "applied_credit": self.applied_credit,
            "cumulative_risk": self.cumulative_risk,
            "passed": self.passed,
            "errors": self.errors,
            "expected_pass": self.expected_pass,
            "violation": self.violation,
        }


@dataclass(frozen=True)
class RollbackLedgerReport:
    """C3 rollback-credit ledger report."""

    entries: tuple[RollbackLedgerEntry, ...]

    @property
    def actions(self) -> int:
        """Number of actions evaluated."""
        return len(self.entries)

    @property
    def accepted(self) -> int:
        """Number of ledger updates accepted."""
        return sum(entry.passed for entry in self.entries)

    @property
    def rejected(self) -> int:
        """Number of ledger updates rejected."""
        return sum(not entry.passed for entry in self.entries)

    @property
    def violations(self) -> int:
        """Number of expectation mismatches."""
        return sum(entry.violation for entry in self.entries)

    @property
    def passed(self) -> bool:
        """True when every rollback-credit expectation is met."""
        return self.violations == 0

    @property
    def cumulative_risk(self) -> RiskVector:
        """Final cumulative-risk vector."""
        return self.entries[-1].cumulative_risk if self.entries else {}

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly rollback ledger report."""
        return {
            "passed": self.passed,
            "actions": self.actions,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "violations": self.violations,
            "cumulative_risk": self.cumulative_risk,
            "entries": [entry.to_dict() for entry in self.entries],
        }


def rollback_curriculum() -> tuple[RollbackAction, ...]:
    """Return clean and adversarial rollback-credit probes."""
    return (
        RollbackAction(
            action_id="c3-rollback-impact-control",
            risk_delta={RiskChannel.IMPACT.value: 0.4},
            credit_cap={RiskChannel.IMPACT.value: 0.25},
            rollback=RollbackPlan(
                credits={RiskChannel.IMPACT.value: 0.25},
                monitor_certificates={RiskChannel.IMPACT.value: 1.0},
            ),
        ),
        RollbackAction(
            action_id="c3-rollback-conflict-control",
            risk_delta={RiskChannel.CONFLICT.value: 0.3},
            credit_cap={RiskChannel.CONFLICT.value: 0.1},
            rollback=RollbackPlan(
                credits={RiskChannel.CONFLICT.value: 0.1},
                monitor_certificates={RiskChannel.CONFLICT.value: 1.0},
            ),
        ),
        RollbackAction(
            action_id="c3-rollback-reject-over-credit",
            risk_delta={RiskChannel.IMPACT.value: 0.1},
            credit_cap={RiskChannel.IMPACT.value: 0.05},
            rollback=RollbackPlan(
                credits={RiskChannel.IMPACT.value: 0.1},
                monitor_certificates={RiskChannel.IMPACT.value: 1.0},
            ),
            expected_pass=False,
        ),
        RollbackAction(
            action_id="c3-rollback-reject-uncertified",
            risk_delta={RiskChannel.CONFLICT.value: 0.1},
            credit_cap={RiskChannel.CONFLICT.value: 0.1},
            rollback=RollbackPlan(
                credits={RiskChannel.CONFLICT.value: 0.1},
                monitor_certificates={RiskChannel.CONFLICT.value: 0.5},
            ),
            expected_pass=False,
        ),
        RollbackAction(
            action_id="c3-rollback-reject-absorbing-privacy",
            risk_delta={RiskChannel.PRIVACY.value: 0.1},
            credit_cap={RiskChannel.PRIVACY.value: 0.1},
            rollback=RollbackPlan(
                credits={RiskChannel.PRIVACY.value: 0.1},
                monitor_certificates={RiskChannel.PRIVACY.value: 1.0},
            ),
            expected_pass=False,
        ),
    )


def run_rollback_ledger(
    actions: tuple[RollbackAction, ...] | None = None,
) -> RollbackLedgerReport:
    """Run a cumulative-risk ledger over rollback-credit probes."""
    cumulative: RiskVector = {}
    entries: list[RollbackLedgerEntry] = []
    for action in rollback_curriculum() if actions is None else actions:
        entry = rollback_entry(action, cumulative)
        entries.append(entry)
        if entry.passed:
            cumulative = dict(entry.cumulative_risk)
    return RollbackLedgerReport(tuple(entries))


def rollback_entry(action: RollbackAction, current: RiskVector) -> RollbackLedgerEntry:
    """Evaluate one action against eqs. 111-112 rollback-credit constraints."""
    errors = rollback_errors(action)
    applied = {} if errors else positive_vector(action.rollback.credits)
    cumulative = (
        dict(current) if errors else update_cumulative_risk(current, action.risk_delta, applied)
    )
    return RollbackLedgerEntry(
        action_id=action.action_id,
        risk_delta=positive_vector(action.risk_delta),
        requested_credit=positive_vector(action.rollback.credits),
        applied_credit=applied,
        cumulative_risk=cumulative,
        passed=not errors,
        errors=errors,
        expected_pass=action.expected_pass,
    )


def rollback_errors(action: RollbackAction) -> tuple[str, ...]:
    """Return rollback-credit validation errors for one action."""
    errors: list[str] = []
    for channel, credit in positive_vector(action.rollback.credits).items():
        cap = action.credit_cap.get(channel, 0.0)
        certificate = action.rollback.monitor_certificates.get(channel, 0.0)
        if credit > cap:
            errors.append(f"{channel}: rollback credit exceeds cap")
        if certificate < ROLLBACK_CERT_THRESHOLD:
            errors.append(f"{channel}: rollback monitor certificate below threshold")
        if channel in ABSORBING_CHANNELS and channel not in action.rollback.reversal_witnesses:
            errors.append(f"{channel}: absorbing channel lacks reversal witness")
    for channel, value in action.risk_delta.items():
        if value < 0.0:
            errors.append(f"{channel}: risk delta must be non-negative")
    return tuple(errors)


def update_cumulative_risk(
    current: RiskVector,
    delta: RiskVector,
    credit: RiskVector,
) -> RiskVector:
    """Apply ``r_{t+1}=r_t+delta-credit`` componentwise with zero floor."""
    channels = sorted(set(current) | set(delta) | set(credit))
    updated: RiskVector = {}
    for channel in channels:
        value = current.get(channel, 0.0) + delta.get(channel, 0.0) - credit.get(channel, 0.0)
        updated[channel] = max(0.0, value)
    return updated


def positive_vector(vector: RiskVector) -> RiskVector:
    """Keep positive risk components in deterministic key order."""
    return {key: vector[key] for key in sorted(vector) if vector[key] > 0.0}


__all__ = [
    "ABSORBING_CHANNELS",
    "ROLLBACK_CERT_THRESHOLD",
    "RiskChannel",
    "RollbackAction",
    "RollbackLedgerEntry",
    "RollbackLedgerReport",
    "RollbackPlan",
    "rollback_curriculum",
    "rollback_entry",
    "rollback_errors",
    "run_rollback_ledger",
    "update_cumulative_risk",
]
