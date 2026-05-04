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
from vpm.evaluation.compression import CompressionReplayReport, evaluate_c5
from vpm.evaluation.dialogue import DialogueEvaluationReport, evaluate_c4
from vpm.infer import InferenceResult, run_task, run_task_candidate
from vpm.tasks.c0 import C0Task, curriculum
from vpm.tasks.c1 import C1Task, hidden_schema_curriculum
from vpm.tasks.c2 import ActiveTestTrace, C2Task, active_curriculum, active_test
from vpm.tasks.c3 import C3PolicyProbe, policy_probe_curriculum
from vpm.verifiers import gate_passed


@dataclass(frozen=True)
class EvidenceMetrics:
    """Source/rebuttal/realization evidence coverage."""

    tasks: int
    source_covered: int
    rebuttal_clear: int
    realization_ok: int
    mean_source_loss: float
    mean_rebuttal_loss: float
    mean_realization_loss: float

    @property
    def source_coverage_rate(self) -> float:
        """Fraction of tasks with exact or retrieved source support."""
        return self.source_covered / self.tasks if self.tasks else 0.0

    @property
    def rebuttal_clear_rate(self) -> float:
        """Fraction of tasks with no material rebuttal loss."""
        return self.rebuttal_clear / self.tasks if self.tasks else 0.0

    @property
    def realization_ok_rate(self) -> float:
        """Fraction of tasks whose rendered atom round-trips exactly."""
        return self.realization_ok / self.tasks if self.tasks else 0.0

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly evidence metrics."""
        return {
            "tasks": self.tasks,
            "source_covered": self.source_covered,
            "source_coverage_rate": self.source_coverage_rate,
            "rebuttal_clear": self.rebuttal_clear,
            "rebuttal_clear_rate": self.rebuttal_clear_rate,
            "realization_ok": self.realization_ok,
            "realization_ok_rate": self.realization_ok_rate,
            "mean_source_loss": self.mean_source_loss,
            "mean_rebuttal_loss": self.mean_rebuttal_loss,
            "mean_realization_loss": self.mean_realization_loss,
        }


@dataclass(frozen=True)
class EvaluationReport:
    """Minimal Criterion-1-facing metrics for the executable kernel."""

    tasks: int
    solved: int
    gate_violations: int
    mean_certificate: float
    active_memory_growth: int
    evidence: EvidenceMetrics

    @property
    def solve_rate(self) -> float:
        """Solved fraction."""
        return self.solved / self.tasks if self.tasks else 0.0

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly metrics."""
        return {
            "tasks": self.tasks,
            "solved": self.solved,
            "solve_rate": self.solve_rate,
            "gate_violations": self.gate_violations,
            "mean_certificate": self.mean_certificate,
            "active_memory_growth": self.active_memory_growth,
            "evidence": self.evidence.to_dict(),
        }


@dataclass(frozen=True)
class ActiveEvaluationReport:
    """C2 active-test evaluation plus verifier-gated outcomes."""

    tasks: int
    verifier: EvaluationReport
    traces: tuple[ActiveTestTrace, ...]

    @property
    def solve_rate(self) -> float:
        """Verifier-certified solve rate after active testing."""
        return self.verifier.solved / self.tasks if self.tasks else 0.0

    @property
    def support_reduction_rate(self) -> float:
        """Fraction of tasks whose active test reduced candidate support."""
        reduced = sum(trace.support_reduced for trace in self.traces)
        return reduced / self.tasks if self.tasks else 0.0

    @property
    def mean_candidates_before(self) -> float:
        """Mean executable candidate count before active testing."""
        return mean([float(len(trace.candidates_before)) for trace in self.traces])

    @property
    def mean_candidates_after(self) -> float:
        """Mean executable candidate count after active testing."""
        return mean([float(len(trace.candidates_after)) for trace in self.traces])

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly active-test report."""
        return {
            "tasks": self.tasks,
            "solve_rate": self.solve_rate,
            "support_reduction_rate": self.support_reduction_rate,
            "mean_candidates_before": self.mean_candidates_before,
            "mean_candidates_after": self.mean_candidates_after,
            "verifier": self.verifier.to_dict(),
            "traces": [trace.to_dict() for trace in self.traces],
        }


@dataclass(frozen=True)
class PolicyGateTrace:
    """One C3 policy probe outcome."""

    probe_id: str
    expected_pass: bool
    gate_passed: bool
    verification_passed: bool
    memory_active: int
    auth_ok: bool
    risk_ok: bool
    reasons: tuple[str, ...]

    @property
    def violation(self) -> bool:
        """True when the gate result disagrees with the expected policy outcome."""
        return self.gate_passed != self.expected_pass

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly policy trace."""
        return {
            "probe_id": self.probe_id,
            "expected_pass": self.expected_pass,
            "gate_passed": self.gate_passed,
            "verification_passed": self.verification_passed,
            "memory_active": self.memory_active,
            "auth_ok": self.auth_ok,
            "risk_ok": self.risk_ok,
            "reasons": self.reasons,
            "violation": self.violation,
        }


@dataclass(frozen=True)
class PolicyEvaluationReport:
    """C3 authority/risk policy evaluation."""

    probes: int
    rejected: int
    controls_passed: int
    violations: int
    traces: tuple[PolicyGateTrace, ...]

    @property
    def violation_rate(self) -> float:
        """Fraction of probes whose gate outcome violated the expected policy."""
        return self.violations / self.probes if self.probes else 0.0

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly policy evaluation."""
        return {
            "probes": self.probes,
            "rejected": self.rejected,
            "controls_passed": self.controls_passed,
            "violations": self.violations,
            "violation_rate": self.violation_rate,
            "traces": [trace.to_dict() for trace in self.traces],
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


def evaluate_c2(
    tasks: list[C2Task] | None = None,
    limit: int = 3,
) -> ActiveEvaluationReport:
    """Run C2 active-test selection and verifier-gated execution."""
    cases = active_curriculum(limit) if tasks is None else tasks
    traces = tuple(active_test(task) for task in cases)
    results = [
        run_task_candidate(task.to_c0_task(trace.selected_operation), trace.selected_operation)
        for task, trace in zip(cases, traces, strict=True)
        if trace.selected_operation is not None
    ]
    return ActiveEvaluationReport(
        tasks=len(cases),
        verifier=summarize(results),
        traces=traces,
    )


def evaluate_c3(probes: list[C3PolicyProbe] | None = None) -> PolicyEvaluationReport:
    """Run adversarial authority/risk probes through the verifier gate."""
    cases = policy_probe_curriculum() if probes is None else probes
    traces = tuple(policy_gate_trace(probe) for probe in cases)
    return PolicyEvaluationReport(
        probes=len(traces),
        rejected=sum(not trace.gate_passed for trace in traces),
        controls_passed=sum(trace.gate_passed and trace.expected_pass for trace in traces),
        violations=sum(trace.violation for trace in traces),
        traces=traces,
    )


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
        evidence=evidence_metrics(results),
    )


def evidence_metrics(results: list[InferenceResult]) -> EvidenceMetrics:
    """Summarize source/rebuttal/realization losses from inference results."""
    source_losses: list[float] = []
    rebuttal_losses: list[float] = []
    realization_losses: list[float] = []
    source_covered = 0
    rebuttal_clear = 0
    realization_ok = 0
    for result in results:
        retrieval = result.retrieval
        source_loss = retrieval.source_loss if retrieval else 1.0
        rebuttal_loss = retrieval.rebuttal_loss if retrieval else 1.0
        source_losses.append(source_loss)
        rebuttal_losses.append(rebuttal_loss)
        if retrieval and retrieval.sources and source_loss == 0.0:
            source_covered += 1
        if retrieval and not retrieval.rebuttals and rebuttal_loss == 0.0:
            rebuttal_clear += 1

        normal_form = result.compiled.normal_form if result.compiled else None
        realization_loss = normal_form.realization_loss if normal_form else 1.0
        realization_losses.append(realization_loss)
        if realization_loss == 0.0:
            realization_ok += 1

    return EvidenceMetrics(
        tasks=len(results),
        source_covered=source_covered,
        rebuttal_clear=rebuttal_clear,
        realization_ok=realization_ok,
        mean_source_loss=mean(source_losses),
        mean_rebuttal_loss=mean(rebuttal_losses),
        mean_realization_loss=mean(realization_losses),
    )


def policy_gate_trace(probe: C3PolicyProbe) -> PolicyGateTrace:
    """Run one C3 policy probe and summarize gate fields."""
    result = run_task(probe.task, labels=probe.labels, risk=probe.risk)
    gate = object_map(result.native_report.get("gate")) or {}
    authority = object_map(gate.get("authority")) or {}
    verification = object_map(result.native_report.get("verification")) or {}
    reasons = gate.get("reasons")
    return PolicyGateTrace(
        probe_id=probe.task_id,
        expected_pass=probe.expected_pass,
        gate_passed=gate.get("passed") is True,
        verification_passed=verification.get("passed") is True,
        memory_active=result.memory_active,
        auth_ok=authority.get("auth_ok") is True,
        risk_ok=authority.get("risk_ok") is True,
        reasons=tuple(reason for reason in reasons if isinstance(reason, str))
        if isinstance(reasons, list)
        else (),
    )


def certificate(report: dict[str, object]) -> float:
    """Read certificate score from a report."""
    gate = object_map(report.get("gate"))
    if gate is None:
        return 0.0
    return float_field(gate, "certificate_score")


def mean(values: list[float]) -> float:
    """Return a zero-safe arithmetic mean."""
    return sum(values) / len(values) if values else 0.0


__all__ = [
    "ActiveEvaluationReport",
    "CompressionReplayReport",
    "DialogueEvaluationReport",
    "EvaluationReport",
    "EvidenceMetrics",
    "PolicyEvaluationReport",
    "PolicyGateTrace",
    "certificate",
    "evaluate_c0",
    "evaluate_c1",
    "evaluate_c2",
    "evaluate_c3",
    "evaluate_c4",
    "evaluate_c5",
    "evidence_metrics",
    "policy_gate_trace",
    "summarize",
]
