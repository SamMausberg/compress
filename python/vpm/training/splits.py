"""Split-topology policy for certificate-carrying events."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SplitAssignment:
    """Role-separated split sets from §8 eqs. 137-139."""

    generator: frozenset[str] = frozenset()
    context: frozenset[str] = frozenset()
    semantic: frozenset[str] = frozenset()
    source: frozenset[str] = frozenset()
    rebuttal: frozenset[str] = frozenset()
    realization: frozenset[str] = frozenset()
    dependence: frozenset[str] = frozenset()
    frontier: frozenset[str] = frozenset()
    entailment: frozenset[str] = frozenset()
    verifier_train: frozenset[str] = frozenset()
    verifier_calibration: frozenset[str] = frozenset()
    audit: frozenset[str] = frozenset()

    @property
    def generator_evidence(self) -> frozenset[str]:
        """Evidence inspected by candidate generation."""
        return (
            self.generator
            | self.context
            | self.semantic
            | self.source
            | self.rebuttal
            | self.realization
            | self.dependence
            | self.frontier
        )

    @property
    def witness_evidence(self) -> frozenset[str]:
        """Evidence used by verifier training, calibration, or audit."""
        return self.verifier_train | self.verifier_calibration | self.audit

    @property
    def audit_forbidden(self) -> frozenset[str]:
        """Evidence that must be disjoint from the audit split."""
        return (
            self.generator_evidence
            | self.entailment
            | self.verifier_train
            | self.verifier_calibration
        )

    def violations(self) -> tuple[str, ...]:
        """Return split-policy violations."""
        violations: list[str] = []
        if overlap := self.generator_evidence & self.witness_evidence:
            violations.append(f"generator/witness overlap: {sorted(overlap)}")
        if overlap := self.verifier_train & self.verifier_calibration:
            violations.append(f"train/calibration overlap: {sorted(overlap)}")
        if overlap := self.audit & self.audit_forbidden:
            violations.append(f"audit leakage: {sorted(overlap)}")
        return tuple(violations)

    @property
    def clean(self) -> bool:
        """True when all eq. 139 disjointness constraints hold."""
        return not self.violations()

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly split assignment."""
        return {
            "generator": sorted(self.generator),
            "context": sorted(self.context),
            "semantic": sorted(self.semantic),
            "source": sorted(self.source),
            "rebuttal": sorted(self.rebuttal),
            "realization": sorted(self.realization),
            "dependence": sorted(self.dependence),
            "frontier": sorted(self.frontier),
            "entailment": sorted(self.entailment),
            "verifier_train": sorted(self.verifier_train),
            "verifier_calibration": sorted(self.verifier_calibration),
            "audit": sorted(self.audit),
            "clean": self.clean,
            "violations": self.violations(),
        }


def split_report(assignments: tuple[SplitAssignment, ...]) -> dict[str, object]:
    """Summarize split hygiene over a batch of certificate events."""
    dirty = tuple(assignment for assignment in assignments if not assignment.clean)
    return {
        "events": len(assignments),
        "clean": len(dirty) == 0,
        "violations": [assignment.violations() for assignment in dirty],
    }


__all__ = ["SplitAssignment", "split_report"]
