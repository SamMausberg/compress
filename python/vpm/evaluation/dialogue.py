"""Controlled-dialogue evaluation for C4."""

from __future__ import annotations

from dataclasses import dataclass

from vpm.tasks.c4 import C4DialogueTask, DialogueGateTrace, dialogue_curriculum, gate_dialogue


@dataclass(frozen=True)
class DialogueEvaluationReport:
    """C4 source/rebuttal/realization evaluation."""

    tasks: int
    rendered: int
    refused: int
    source_covered: int
    rebuttal_clear: int
    realization_ok: int
    uncertainty_ok: int
    mean_uncertainty: float
    violations: int
    traces: tuple[DialogueGateTrace, ...]

    @property
    def render_rate(self) -> float:
        """Fraction of dialogue tasks rendered after all gates passed."""
        return self.rendered / self.tasks if self.tasks else 0.0

    @property
    def refusal_rate(self) -> float:
        """Fraction of dialogue tasks refused by the gate."""
        return self.refused / self.tasks if self.tasks else 0.0

    @property
    def source_coverage_rate(self) -> float:
        """Fraction of tasks with source support."""
        return self.source_covered / self.tasks if self.tasks else 0.0

    @property
    def rebuttal_clear_rate(self) -> float:
        """Fraction of tasks with no material rebuttal."""
        return self.rebuttal_clear / self.tasks if self.tasks else 0.0

    @property
    def realization_ok_rate(self) -> float:
        """Fraction of tasks whose renderer round-tripped the answer."""
        return self.realization_ok / self.tasks if self.tasks else 0.0

    @property
    def uncertainty_ok_rate(self) -> float:
        """Fraction of tasks within the calibrated uncertainty threshold."""
        return self.uncertainty_ok / self.tasks if self.tasks else 0.0

    @property
    def violation_rate(self) -> float:
        """Fraction of tasks with a rendered output despite a failed witness."""
        return self.violations / self.tasks if self.tasks else 0.0

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly dialogue evaluation."""
        return {
            "tasks": self.tasks,
            "rendered": self.rendered,
            "refused": self.refused,
            "render_rate": self.render_rate,
            "refusal_rate": self.refusal_rate,
            "source_coverage_rate": self.source_coverage_rate,
            "rebuttal_clear_rate": self.rebuttal_clear_rate,
            "realization_ok_rate": self.realization_ok_rate,
            "uncertainty_ok_rate": self.uncertainty_ok_rate,
            "mean_uncertainty": self.mean_uncertainty,
            "violations": self.violations,
            "violation_rate": self.violation_rate,
            "traces": [trace.to_dict() for trace in self.traces],
        }


def evaluate_c4(tasks: list[C4DialogueTask] | None = None) -> DialogueEvaluationReport:
    """Run C4 controlled dialogue through source/rebuttal/realization gates."""
    cases = dialogue_curriculum() if tasks is None else tasks
    traces = tuple(gate_dialogue(task) for task in cases)
    return DialogueEvaluationReport(
        tasks=len(traces),
        rendered=sum(trace.passed for trace in traces),
        refused=sum(not trace.passed for trace in traces),
        source_covered=sum(trace.source_ok for trace in traces),
        rebuttal_clear=sum(trace.rebuttal_ok for trace in traces),
        realization_ok=sum(trace.realization_ok for trace in traces),
        uncertainty_ok=sum(trace.uncertainty_ok for trace in traces),
        mean_uncertainty=sum(trace.uncertainty for trace in traces) / len(traces)
        if traces
        else 0.0,
        violations=sum(trace.rendered != "refusal" and not trace.passed for trace in traces),
        traces=traces,
    )


__all__ = ["DialogueEvaluationReport", "evaluate_c4"]
