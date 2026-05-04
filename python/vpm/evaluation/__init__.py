"""§9 — Evaluation, failure modes, and minimal implementation.

See ``docs/architecture/09-evaluation-failure.md``.

This package will hold:

- ``metrics.py``         — every metric named in the §9 reporting
  paragraph (certified utility, solve rate, ECE, support loss, …).
- ``strata.py``          — stratification by evidence level, domain
  route, mode, and curriculum stage.
- ``ablations.py``       — toggle each architectural component listed
  in §9; the runner is the only canonical way to produce the
  ablation table.
- ``failure_modes.py``   — Criterion 1 clauses encoded as a
  ``FailureMode`` enum and a per-clause detector.
- ``phase_transition.py`` — the compression phase-transition
  diagnostic that defines VPM-0 success (last paragraph of §9).
- ``saturation.py``      — saturation diagnostic
  ``sup_m LCB^seq A(m) ≤ 0`` (eq. 136).
- ``report.py``          — pretty-prints the metric table and writes
  JSON for CI consumption.
"""

from __future__ import annotations

from dataclasses import dataclass

from vpm._reports import float_field, object_map
from vpm.infer import InferenceResult, run_task, run_task_candidate
from vpm.tasks.c0 import C0Task, curriculum
from vpm.tasks.c1 import C1Task, hidden_schema_curriculum
from vpm.verifiers import gate_passed


@dataclass(frozen=True)
class EvaluationReport:
    """Minimal Criterion-1-facing metrics for the executable kernel."""

    tasks: int
    solved: int
    gate_violations: int
    mean_certificate: float
    active_memory_growth: int

    @property
    def solve_rate(self) -> float:
        """Solved fraction."""
        return self.solved / self.tasks if self.tasks else 0.0

    def to_dict(self) -> dict[str, float | int]:
        """JSON-friendly metrics."""
        return {
            "tasks": self.tasks,
            "solved": self.solved,
            "solve_rate": self.solve_rate,
            "gate_violations": self.gate_violations,
            "mean_certificate": self.mean_certificate,
            "active_memory_growth": self.active_memory_growth,
        }


def evaluate_c0(tasks: list[C0Task] | None = None) -> EvaluationReport:
    """Run the C0 curriculum through the MVP inference loop."""
    cases = curriculum() if tasks is None else tasks
    results = [run_task(task) for task in cases]
    return summarize(results)


def evaluate_c1(tasks: list[C1Task] | None = None, limit: int = 3) -> EvaluationReport:
    """Run the executable C1 hidden-schema subset through the verifier gate."""
    cases = hidden_schema_curriculum(limit) if tasks is None else tasks
    results = [run_task_candidate(task.to_c0_task(), task.operation) for task in cases]
    return summarize(results)


def summarize(results: list[InferenceResult]) -> EvaluationReport:
    """Summarize inference results using the MVP metric subset."""
    solved = sum(1 for result in results if gate_passed(result.native_report))
    scores = [certificate(result.native_report) for result in results]
    return EvaluationReport(
        tasks=len(results),
        solved=solved,
        gate_violations=sum(
            1 for result in results if result.route == "solve" and not result.rendered
        ),
        mean_certificate=sum(scores) / len(scores) if scores else 0.0,
        active_memory_growth=sum(result.memory_active for result in results),
    )


def certificate(report: dict[str, object]) -> float:
    """Read certificate score from a report."""
    gate = object_map(report.get("gate"))
    if gate is None:
        return 0.0
    return float_field(gate, "certificate_score")


__all__ = ["EvaluationReport", "certificate", "evaluate_c0", "evaluate_c1", "summarize"]
