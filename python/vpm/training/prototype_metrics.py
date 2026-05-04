"""Matched baselines and compression metrics for the C0 prototype."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from vpm.infer import run_task_candidate
from vpm.substrate.prototype import OPERATIONS
from vpm.tasks.c0 import C0Task
from vpm.verifiers import gate_passed


@dataclass(frozen=True)
class BaselineMetrics:
    """Matched baseline result."""

    name: str
    solve_rate: float
    operation_accuracy: float
    mean_candidates: float

    def to_dict(self) -> dict[str, float | str]:
        """JSON-friendly baseline metrics."""
        return {
            "name": self.name,
            "solve_rate": self.solve_rate,
            "operation_accuracy": self.operation_accuracy,
            "mean_candidates": self.mean_candidates,
        }


@dataclass(frozen=True)
class CompressionMetrics:
    """Compression/frontier diagnostics for held-out certified utility."""

    learned_mean_candidates: float
    certified_utility: float
    enumerative_utility: float
    frontier_delta_vs_enumerative: float
    active_memory_growth: float
    sublinear_active_memory: bool

    def to_dict(self) -> dict[str, float | bool]:
        """JSON-friendly compression diagnostics."""
        return {
            "learned_mean_candidates": self.learned_mean_candidates,
            "certified_utility": self.certified_utility,
            "enumerative_utility": self.enumerative_utility,
            "frontier_delta_vs_enumerative": self.frontier_delta_vs_enumerative,
            "active_memory_growth": self.active_memory_growth,
            "sublinear_active_memory": self.sublinear_active_memory,
        }


def matched_baselines(
    train_tasks: list[C0Task], heldout_tasks: list[C0Task]
) -> tuple[BaselineMetrics, ...]:
    """Evaluate fixed-budget and exact-search baselines."""
    return (
        fixed_budget_baseline("majority", majority_operation(train_tasks), heldout_tasks),
        *(
            fixed_budget_baseline(f"{operation}-only", operation, heldout_tasks)
            for operation in OPERATIONS
        ),
        enumerative_baseline(heldout_tasks),
    )


def compression_frontier_metrics(
    tasks: int,
    certified: int,
    macro_active: int,
    baselines: tuple[BaselineMetrics, ...],
) -> CompressionMetrics:
    """Compute utility/frontier movement against the exact enumerative baseline."""
    learned_mean_candidates = 1.0 if tasks else 0.0
    solve_rate = certified / tasks if tasks else 0.0
    certified_utility = solve_rate / learned_mean_candidates if learned_mean_candidates else 0.0
    enumerative = next(baseline for baseline in baselines if baseline.name == "enumerative-full")
    enumerative_utility = (
        enumerative.solve_rate / enumerative.mean_candidates if enumerative.mean_candidates else 0.0
    )
    active_memory_growth = macro_active / certified if certified else 0.0
    return CompressionMetrics(
        learned_mean_candidates=learned_mean_candidates,
        certified_utility=certified_utility,
        enumerative_utility=enumerative_utility,
        frontier_delta_vs_enumerative=certified_utility - enumerative_utility,
        active_memory_growth=active_memory_growth,
        sublinear_active_memory=0 < macro_active < certified,
    )


def majority_operation(tasks: list[C0Task]) -> str:
    """Most frequent operation in a training split."""
    counts = Counter(task.operation for task in tasks)
    return counts.most_common(1)[0][0] if counts else "add"


def fixed_budget_baseline(name: str, operation: str, tasks: list[C0Task]) -> BaselineMetrics:
    """Evaluate a one-candidate operation baseline."""
    certified = 0
    correct = 0
    for task in tasks:
        result = run_task_candidate(task, operation)
        certified += int(gate_passed(result.native_report))
        correct += int(operation == task.operation)
    total = len(tasks)
    return BaselineMetrics(name, certified / total, correct / total, 1.0)


def enumerative_baseline(tasks: list[C0Task]) -> BaselineMetrics:
    """Evaluate exact search over the supported operation set."""
    certified = 0
    candidates = 0
    for task in tasks:
        for operation in OPERATIONS:
            candidates += 1
            result = run_task_candidate(task, operation)
            if gate_passed(result.native_report):
                certified += 1
                break
    total = len(tasks)
    return BaselineMetrics("enumerative-full", certified / total, 1.0, candidates / total)


__all__ = [
    "BaselineMetrics",
    "CompressionMetrics",
    "compression_frontier_metrics",
    "enumerative_baseline",
    "fixed_budget_baseline",
    "matched_baselines",
]
