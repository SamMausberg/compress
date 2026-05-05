"""M6 red-team replay harness."""

from __future__ import annotations

from dataclasses import dataclass

from vpm.audit.failure_modes import FailureModeReport, evaluate_failure_modes
from vpm.evaluation.ablations import AblationReport, evaluate_ablations
from vpm.evaluation.hard_domains import HardDomainReport, evaluate_hard_domains


@dataclass(frozen=True)
class RedTeamReport:
    """Combined M6 failure, ablation, and hard-domain replay report."""

    failures: FailureModeReport
    ablations: AblationReport
    hard_domains: HardDomainReport

    @property
    def passed(self) -> bool:
        """True when every executable M6 replay axis passes."""
        return (
            self.failures.passed and self.ablations.passed and self.hard_domains.solve_rate == 1.0
        )

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly red-team report."""
        return {
            "passed": self.passed,
            "failures": self.failures.to_dict(),
            "ablations": self.ablations.to_dict(),
            "hard_domains": self.hard_domains.to_dict(),
        }


def red_team_replay() -> RedTeamReport:
    """Run M6 executable red-team replay."""
    return RedTeamReport(
        failures=evaluate_failure_modes(),
        ablations=evaluate_ablations(),
        hard_domains=evaluate_hard_domains(),
    )


__all__ = ["RedTeamReport", "red_team_replay"]
