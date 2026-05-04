"""Continual-compression evaluation for C5."""

from __future__ import annotations

from dataclasses import dataclass

from vpm.infer import InferenceResult, run_task_candidate
from vpm.memory import MemoryLibrary
from vpm.substrate.prototype import OPERATIONS
from vpm.tasks.c5 import C5MacroCandidate, macro_replay_curriculum
from vpm.verifiers import gate_passed


@dataclass(frozen=True)
class MacroReplayTrace:
    """One replay-safe macro admission trace."""

    candidate_id: str
    macro_key: str
    operation: str
    replay_tasks: int
    certified: int
    active_memory: int
    expected_admitted: bool
    admitted: bool
    frontier_delta: float
    sublinear_active_memory: bool
    gate_violations: int
    reasons: tuple[str, ...]

    @property
    def demoted(self) -> bool:
        """True when the macro was not admitted to active memory."""
        return not self.admitted

    @property
    def admission_violation(self) -> bool:
        """True when observed admission disagrees with the expected outcome."""
        return self.admitted != self.expected_admitted

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly macro replay trace."""
        return {
            "candidate_id": self.candidate_id,
            "macro_key": self.macro_key,
            "operation": self.operation,
            "replay_tasks": self.replay_tasks,
            "certified": self.certified,
            "active_memory": self.active_memory,
            "expected_admitted": self.expected_admitted,
            "admitted": self.admitted,
            "demoted": self.demoted,
            "frontier_delta": self.frontier_delta,
            "sublinear_active_memory": self.sublinear_active_memory,
            "gate_violations": self.gate_violations,
            "reasons": self.reasons,
            "admission_violation": self.admission_violation,
        }


@dataclass(frozen=True)
class CompressionReplayReport:
    """C5 continual-compression replay report."""

    macros: int
    admitted: int
    demoted: int
    positive_frontier: int
    sublinear_active: int
    violations: int
    traces: tuple[MacroReplayTrace, ...]

    @property
    def admission_rate(self) -> float:
        """Fraction of macro candidates admitted."""
        return self.admitted / self.macros if self.macros else 0.0

    @property
    def demotion_rate(self) -> float:
        """Fraction of macro candidates demoted."""
        return self.demoted / self.macros if self.macros else 0.0

    @property
    def frontier_movement_rate(self) -> float:
        """Fraction of macro candidates with positive frontier movement."""
        return self.positive_frontier / self.macros if self.macros else 0.0

    @property
    def violation_rate(self) -> float:
        """Fraction of macro candidates with admission or gate violations."""
        return self.violations / self.macros if self.macros else 0.0

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly compression replay evaluation."""
        return {
            "macros": self.macros,
            "admitted": self.admitted,
            "demoted": self.demoted,
            "admission_rate": self.admission_rate,
            "demotion_rate": self.demotion_rate,
            "frontier_movement_rate": self.frontier_movement_rate,
            "sublinear_active": self.sublinear_active,
            "violations": self.violations,
            "violation_rate": self.violation_rate,
            "traces": [trace.to_dict() for trace in self.traces],
        }


def evaluate_c5(
    candidates: list[C5MacroCandidate] | None = None,
) -> CompressionReplayReport:
    """Run C5 replay-safe macro admission and demotion probes."""
    cases = macro_replay_curriculum() if candidates is None else candidates
    traces = tuple(macro_replay_trace(candidate) for candidate in cases)
    return CompressionReplayReport(
        macros=len(traces),
        admitted=sum(trace.admitted for trace in traces),
        demoted=sum(trace.demoted for trace in traces),
        positive_frontier=sum(trace.frontier_delta > 0.0 for trace in traces),
        sublinear_active=sum(trace.sublinear_active_memory for trace in traces),
        violations=sum(trace.admission_violation or trace.gate_violations > 0 for trace in traces),
        traces=traces,
    )


def macro_replay_trace(candidate: C5MacroCandidate) -> MacroReplayTrace:
    """Evaluate one macro candidate against independent replay tasks."""
    results = tuple(
        run_task_candidate(task, candidate.operation) for task in candidate.replay_tasks
    )
    certified = sum(gate_passed(result.native_report) for result in results)
    frontier_delta = replay_frontier_delta(certified, len(results))
    gate_violations = sum(gate_violation(result) for result in results)
    replay_safe = certified == len(results) and gate_violations == 0
    would_be_active = int(replay_safe and frontier_delta > 0.0)
    sublinear = 0 < would_be_active < certified
    admitted = replay_safe and frontier_delta > 0.0 and sublinear
    memory = admit_macro(candidate, results) if admitted else MemoryLibrary()
    return MacroReplayTrace(
        candidate_id=candidate.task_id,
        macro_key=candidate.macro_key,
        operation=candidate.operation,
        replay_tasks=len(results),
        certified=certified,
        active_memory=len(memory.active),
        expected_admitted=candidate.expected_admitted,
        admitted=admitted,
        frontier_delta=frontier_delta,
        sublinear_active_memory=sublinear,
        gate_violations=gate_violations,
        reasons=macro_replay_reasons(replay_safe, frontier_delta, sublinear),
    )


def admit_macro(
    candidate: C5MacroCandidate,
    results: tuple[InferenceResult, ...],
) -> MemoryLibrary:
    """Admit a replay-safe macro under one active key."""
    memory = MemoryLibrary()
    for result in results:
        memory.admit(candidate.macro_key, candidate.operation, result.native_report)
    return memory


def replay_frontier_delta(certified: int, replay_tasks: int) -> float:
    """Compare one-candidate macro utility to full enumerative utility."""
    if replay_tasks == 0:
        return 0.0
    macro_utility = certified / replay_tasks
    enumerative_utility = 1.0 / len(OPERATIONS)
    return macro_utility - enumerative_utility


def gate_violation(result: InferenceResult) -> bool:
    """Detect a critical gate violation in a replay result."""
    passed = gate_passed(result.native_report)
    return not passed and result.rendered != "refusal"


def macro_replay_reasons(
    replay_safe: bool,
    frontier_delta: float,
    sublinear: bool,
) -> tuple[str, ...]:
    """Collect failed macro admission reasons."""
    reasons: list[str] = []
    if not replay_safe:
        reasons.append("replay gate failed")
    if frontier_delta <= 0.0:
        reasons.append("frontier did not improve")
    if not sublinear:
        reasons.append("active memory growth not sublinear")
    return tuple(reasons)


__all__ = [
    "CompressionReplayReport",
    "MacroReplayTrace",
    "evaluate_c5",
    "macro_replay_trace",
]
