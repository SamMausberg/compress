"""Compute-budget accounting for matched evaluations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComputeLedgerEntry:
    """One test-time compute event."""

    task_id: str
    component: str
    units: float
    declared: bool = True

    def to_dict(self) -> dict[str, float | bool | str]:
        """JSON-friendly ledger entry."""
        return {
            "task_id": self.task_id,
            "component": self.component,
            "units": self.units,
            "declared": self.declared,
        }


@dataclass(frozen=True)
class ComputeAccountingReport:
    """Budget accounting report."""

    entries: tuple[ComputeLedgerEntry, ...]
    budget: float

    @property
    def total_units(self) -> float:
        """Total declared and undeclared compute units."""
        return sum(entry.units for entry in self.entries)

    @property
    def hidden_units(self) -> float:
        """Compute units that were not declared."""
        return sum(entry.units for entry in self.entries if not entry.declared)

    @property
    def budget_exceeded(self) -> bool:
        """True when total units exceed the matched budget."""
        return self.total_units > self.budget

    @property
    def passed(self) -> bool:
        """True when all compute is declared and within budget."""
        return self.hidden_units == 0.0 and not self.budget_exceeded

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly compute report."""
        return {
            "passed": self.passed,
            "budget": self.budget,
            "total_units": self.total_units,
            "hidden_units": self.hidden_units,
            "budget_exceeded": self.budget_exceeded,
            "entries": [entry.to_dict() for entry in self.entries],
        }


def evaluate_compute_accounting() -> ComputeAccountingReport:
    """Run the shipped matched-compute ledger smoke."""
    return ComputeAccountingReport(
        entries=(
            ComputeLedgerEntry("hard-math-gauss-20", "exact-verifier", 1.0),
            ComputeLedgerEntry("hard-formal-modus-ponens", "proof-checker", 1.0),
            ComputeLedgerEntry("hard-tool-square-12", "tool-replay", 1.0),
            ComputeLedgerEntry("hard-source-water", "source-gate", 1.0),
        ),
        budget=4.0,
    )


def hidden_compute_probe() -> ComputeAccountingReport:
    """Return a deliberately dirty ledger used by failure-mode tests."""
    return ComputeAccountingReport(
        entries=(
            ComputeLedgerEntry("probe", "declared", 1.0),
            ComputeLedgerEntry("probe", "undeclared-extra-search", 1.0, declared=False),
        ),
        budget=2.0,
    )


__all__ = [
    "ComputeAccountingReport",
    "ComputeLedgerEntry",
    "evaluate_compute_accounting",
    "hidden_compute_probe",
]
