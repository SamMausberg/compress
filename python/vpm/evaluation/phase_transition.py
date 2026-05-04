"""Compression phase-transition diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

from vpm.evaluation.compression import CompressionReplayReport, evaluate_c5


@dataclass(frozen=True)
class PhaseTransitionReport:
    """C5 compression phase-transition evidence."""

    compression: CompressionReplayReport

    @property
    def observed(self) -> bool:
        """True when replay shows useful compression without gate violations."""
        return (
            self.compression.admitted > 0
            and self.compression.demoted > 0
            and self.compression.frontier_movement_rate > 0.0
            and self.compression.sublinear_active == self.compression.admitted
            and self.compression.violations == 0
        )

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly phase-transition report."""
        return {
            "observed": self.observed,
            "admission_rate": self.compression.admission_rate,
            "demotion_rate": self.compression.demotion_rate,
            "frontier_movement_rate": self.compression.frontier_movement_rate,
            "sublinear_active": self.compression.sublinear_active,
            "violations": self.compression.violations,
            "compression": self.compression.to_dict(),
        }


def evaluate_phase_transition() -> PhaseTransitionReport:
    """Run the C5 compression phase-transition diagnostic."""
    return PhaseTransitionReport(evaluate_c5())


__all__ = ["PhaseTransitionReport", "evaluate_phase_transition"]
