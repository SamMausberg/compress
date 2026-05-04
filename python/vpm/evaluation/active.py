"""C2 active-test evaluation and halt traces."""

from __future__ import annotations

from dataclasses import dataclass

from vpm.evaluation import EvaluationReport, summarize
from vpm.infer import (
    HaltDecision,
    SupportGuardReport,
    TestSelectionTrace,
    guard_support,
    halt_decision,
    run_task_candidate,
    select_test,
    support_reduction_action,
)
from vpm.tasks.c2 import ActiveTestTrace, C2Task, active_curriculum, active_test


@dataclass(frozen=True)
class ActiveEvaluationReport:
    """C2 active-test evaluation plus verifier-gated outcomes."""

    tasks: int
    verifier: EvaluationReport
    traces: tuple[ActiveTestTrace, ...]
    support_guards: tuple[SupportGuardReport, ...]
    test_selections: tuple[TestSelectionTrace, ...]
    halt_decisions: tuple[HaltDecision, ...]

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
        return active_mean([float(len(trace.candidates_before)) for trace in self.traces])

    @property
    def mean_candidates_after(self) -> float:
        """Mean executable candidate count after active testing."""
        return active_mean([float(len(trace.candidates_after)) for trace in self.traces])

    @property
    def support_guard_pass_rate(self) -> float:
        """Fraction of pruning decisions accepted by the support guard."""
        passed = sum(guard.passed for guard in self.support_guards)
        return passed / len(self.support_guards) if self.support_guards else 0.0

    @property
    def rehydrated(self) -> int:
        """Number of pruning decisions that required candidate rehydration."""
        return sum(bool(guard.rehydrated) for guard in self.support_guards)

    @property
    def mean_test_score(self) -> float:
        """Mean selected active-test score."""
        return active_mean([selection.selected_score for selection in self.test_selections])

    @property
    def halt_rate(self) -> float:
        """Fraction of C2 traces whose inference loop should halt after testing."""
        halted = sum(decision.should_halt for decision in self.halt_decisions)
        return halted / len(self.halt_decisions) if self.halt_decisions else 0.0

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly active-test report."""
        return {
            "tasks": self.tasks,
            "solve_rate": self.solve_rate,
            "support_reduction_rate": self.support_reduction_rate,
            "support_guard_pass_rate": self.support_guard_pass_rate,
            "rehydrated": self.rehydrated,
            "mean_test_score": self.mean_test_score,
            "halt_rate": self.halt_rate,
            "mean_candidates_before": self.mean_candidates_before,
            "mean_candidates_after": self.mean_candidates_after,
            "verifier": self.verifier.to_dict(),
            "traces": [trace.to_dict() for trace in self.traces],
            "support_guards": [guard.to_dict() for guard in self.support_guards],
            "test_selections": [selection.to_dict() for selection in self.test_selections],
            "halt_decisions": [decision.to_dict() for decision in self.halt_decisions],
        }


def evaluate_c2(
    tasks: list[C2Task] | None = None,
    limit: int = 3,
) -> ActiveEvaluationReport:
    """Run C2 active-test selection and verifier-gated execution."""
    cases = active_curriculum(limit) if tasks is None else tasks
    traces = tuple(active_test(task) for task in cases)
    support_guards = support_guard_traces(traces)
    test_selections = test_selection_traces(traces, support_guards)
    halt_decisions = halt_traces(support_guards, test_selections)
    results = [
        run_task_candidate(task.to_c0_task(trace.selected_operation), trace.selected_operation)
        for task, trace, guard, decision in zip(
            cases,
            traces,
            support_guards,
            halt_decisions,
            strict=True,
        )
        if trace.selected_operation is not None and guard.passed and decision.should_halt
    ]
    return ActiveEvaluationReport(
        tasks=len(cases),
        verifier=summarize(results),
        traces=traces,
        support_guards=support_guards,
        test_selections=test_selections,
        halt_decisions=halt_decisions,
    )


def support_guard_traces(
    traces: tuple[ActiveTestTrace, ...],
) -> tuple[SupportGuardReport, ...]:
    """Build support-guard reports for active-test traces."""
    return tuple(
        guard_support(
            trace.candidates_before,
            trace.candidates_after,
            exact_rejection_witness=trace.selected_operation is not None,
        )
        for trace in traces
    )


def test_selection_traces(
    traces: tuple[ActiveTestTrace, ...],
    guards: tuple[SupportGuardReport, ...],
) -> tuple[TestSelectionTrace, ...]:
    """Build active-test selection traces."""
    return tuple(
        select_test(
            (
                support_reduction_action(
                    trace.candidates_before,
                    trace.candidates_after,
                    guard.epsilon_prune,
                ),
            )
        )
        for trace, guard in zip(traces, guards, strict=True)
    )


def halt_traces(
    guards: tuple[SupportGuardReport, ...],
    selections: tuple[TestSelectionTrace, ...],
) -> tuple[HaltDecision, ...]:
    """Build halt decisions after active testing."""
    return tuple(
        halt_decision(
            certificate=guard.certificate,
            threshold=1.0,
            expected_utility_gain=selection.selected_score,
            compute_delta=selection.selected.cost,
            support_delta=guard.epsilon_prune,
            best_action_score=selection.selected_score,
        )
        for guard, selection in zip(guards, selections, strict=True)
    )


def active_mean(values: list[float]) -> float:
    """Return a zero-safe arithmetic mean for active-test metrics."""
    return sum(values) / len(values) if values else 0.0


__all__ = ["ActiveEvaluationReport", "evaluate_c2"]
