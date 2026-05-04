"""Substrate recall and critical-edge loss helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SubstrateRecallReport:
    """Substrate recall calibration report."""

    omitted_edges: int
    total_edges: int
    epsilon_sub: float
    epsilon_crit: float
    threshold: float

    @property
    def passed(self) -> bool:
        """True when both omission bounds clear the threshold."""
        return self.epsilon_sub <= self.threshold and self.epsilon_crit <= self.threshold

    def to_dict(self) -> dict[str, float | bool | int]:
        """JSON-friendly report."""
        return {
            "omitted_edges": self.omitted_edges,
            "total_edges": self.total_edges,
            "epsilon_sub": self.epsilon_sub,
            "epsilon_crit": self.epsilon_crit,
            "threshold": self.threshold,
            "passed": self.passed,
        }


def substrate_recall_report(
    omitted_edges: int,
    total_edges: int,
    *,
    critical_omissions: int = 0,
    threshold: float = 0.0,
) -> SubstrateRecallReport:
    """Build epsilon_sub and epsilon_crit from edge omission counts."""
    safe_total = max(total_edges, 0)
    epsilon_sub = omitted_edges / safe_total if safe_total else 0.0
    epsilon_crit = critical_omissions / safe_total if safe_total else 0.0
    return SubstrateRecallReport(
        omitted_edges=max(omitted_edges, 0),
        total_edges=safe_total,
        epsilon_sub=epsilon_sub,
        epsilon_crit=epsilon_crit,
        threshold=threshold,
    )


def substrate_loss(report: SubstrateRecallReport) -> float:
    """Positive-part substrate recall loss."""
    return max(0.0, report.epsilon_sub - report.threshold) + max(
        0.0,
        report.epsilon_crit - report.threshold,
    )


__all__ = ["SubstrateRecallReport", "substrate_loss", "substrate_recall_report"]
