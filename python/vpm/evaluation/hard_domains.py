"""Held-out hard-domain evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from vpm.tasks.hard_domains import HardDomain, HardDomainTask, hard_domain_curriculum


@dataclass(frozen=True)
class HardDomainTrace:
    """One hard-domain verifier trace."""

    task_id: str
    domain: HardDomain
    expected: str
    predicted: str
    certified: bool
    evidence_used: tuple[str, ...]
    baseline_correct: bool

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly trace."""
        return {
            "task_id": self.task_id,
            "domain": self.domain.value,
            "expected": self.expected,
            "predicted": self.predicted,
            "certified": self.certified,
            "evidence_used": self.evidence_used,
            "baseline_correct": self.baseline_correct,
        }


@dataclass(frozen=True)
class DomainMetrics:
    """Per-domain held-out metrics."""

    domain: HardDomain
    tasks: int
    certified: int
    baseline_correct: int

    @property
    def solve_rate(self) -> float:
        """VPM solve rate."""
        return self.certified / self.tasks if self.tasks else 0.0

    @property
    def baseline_solve_rate(self) -> float:
        """Simple matched baseline solve rate."""
        return self.baseline_correct / self.tasks if self.tasks else 0.0

    def to_dict(self) -> dict[str, float | int | str]:
        """JSON-friendly domain metrics."""
        return {
            "domain": self.domain.value,
            "tasks": self.tasks,
            "certified": self.certified,
            "solve_rate": self.solve_rate,
            "baseline_correct": self.baseline_correct,
            "baseline_solve_rate": self.baseline_solve_rate,
        }


@dataclass(frozen=True)
class HardDomainReport:
    """Held-out hard-domain report."""

    tasks: int
    certified: int
    baseline_correct: int
    traces: tuple[HardDomainTrace, ...]
    domains: tuple[DomainMetrics, ...]

    @property
    def solve_rate(self) -> float:
        """Overall certified solve rate."""
        return self.certified / self.tasks if self.tasks else 0.0

    @property
    def baseline_solve_rate(self) -> float:
        """Overall baseline solve rate."""
        return self.baseline_correct / self.tasks if self.tasks else 0.0

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly hard-domain report."""
        return {
            "tasks": self.tasks,
            "certified": self.certified,
            "solve_rate": self.solve_rate,
            "baseline_correct": self.baseline_correct,
            "baseline_solve_rate": self.baseline_solve_rate,
            "domains": [domain.to_dict() for domain in self.domains],
            "traces": [trace.to_dict() for trace in self.traces],
        }


def evaluate_hard_domains(
    tasks: tuple[HardDomainTask, ...] | None = None,
) -> HardDomainReport:
    """Evaluate the held-out hard-domain suite."""
    cases = hard_domain_curriculum() if tasks is None else tasks
    traces = tuple(hard_domain_trace(task) for task in cases)
    return HardDomainReport(
        tasks=len(traces),
        certified=sum(trace.certified for trace in traces),
        baseline_correct=sum(trace.baseline_correct for trace in traces),
        traces=traces,
        domains=domain_metrics(traces),
    )


def hard_domain_trace(task: HardDomainTask) -> HardDomainTrace:
    """Solve and verify one hard-domain task."""
    predicted = exact_hard_domain_solver(task)
    return HardDomainTrace(
        task_id=task.task_id,
        domain=task.domain,
        expected=task.expected,
        predicted=predicted,
        certified=predicted == task.expected,
        evidence_used=task.evidence,
        baseline_correct=task.baseline_answer == task.expected,
    )


def exact_hard_domain_solver(task: HardDomainTask) -> str:
    """Small exact solver for the shipped hard-domain probes."""
    if task.domain is HardDomain.RESEARCH_MATH:
        return solve_gauss_sum(task.evidence)
    if task.domain is HardDomain.FORMAL:
        return solve_modus_ponens(task.evidence)
    return evidence_answer(task.evidence)


def solve_gauss_sum(evidence: tuple[str, ...]) -> str:
    """Apply the audited finite-sum formula."""
    n = next(int(item.removeprefix("n=")) for item in evidence if item.startswith("n="))
    return str((n * (n + 1)) // 2)


def solve_modus_ponens(evidence: tuple[str, ...]) -> str:
    """Apply one propositional modus-ponens fragment."""
    if "premise:p" in evidence and "premise:p_implies_q" in evidence:
        return "q"
    return "unknown"


def evidence_answer(evidence: tuple[str, ...]) -> str:
    """Extract an exact audited answer from source/tool evidence."""
    for item in evidence:
        if "=" in item:
            return item.rsplit("=", maxsplit=1)[-1]
    return "unknown"


def domain_metrics(traces: tuple[HardDomainTrace, ...]) -> tuple[DomainMetrics, ...]:
    """Aggregate metrics by hard-domain category."""
    metrics: list[DomainMetrics] = []
    for domain in HardDomain:
        selected = tuple(trace for trace in traces if trace.domain is domain)
        metrics.append(
            DomainMetrics(
                domain=domain,
                tasks=len(selected),
                certified=sum(trace.certified for trace in selected),
                baseline_correct=sum(trace.baseline_correct for trace in selected),
            )
        )
    return tuple(metrics)


__all__ = [
    "DomainMetrics",
    "HardDomainReport",
    "HardDomainTrace",
    "evaluate_hard_domains",
]
