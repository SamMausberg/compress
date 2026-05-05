"""Curriculum stage ``C_2`` — partially observed and active worlds.

From §8 of ``docs/architecture/08-training-system.md``:

> ``C_2``: noisy/partial observations, active tests, small causal worlds,
> and verifiable planning.

Exit gate: support guard (§5 eqs. 92–95) fires at the calibrated rate;
test-selection ``e^*`` (§5 eq. 100) reduces certified posterior cost on
held-out trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass

from vpm.tasks.c0 import C0Task, C0Value, value_token
from vpm.tasks.c2.causal import (
    CausalObservation,
    CausalWorldTrace,
    NoisyCausalWorld,
    causal_world_curriculum,
    identify_causal_world,
    mean_outcome,
)
from vpm.tasks.c2.planning import (
    GridPlanningTask,
    GridPlanStep,
    GridPoint,
    PlanningTrace,
    grid_neighbors,
    plan_grid_task,
    planning_curriculum,
)
from vpm.tasks.spec import StageSpec

C2_OPERATIONS = ("add", "mul", "concat", "eq")


@dataclass(frozen=True)
class C2Task:
    """Partially observed active-test task backed by the C0 verifier."""

    task_id: str
    left: C0Value
    right: C0Value
    expected: C0Value
    operation: str
    candidates: tuple[str, ...]

    @property
    def observation(self) -> str:
        """Partial observation before the active test reveals expected output."""
        return (
            f"{value_token(self.left)} {value_token(self.right)}"
            f" candidates={','.join(self.candidates)}"
        )

    def to_c0_task(self, operation: str | None = None) -> C0Task:
        """Expose a verifier-backed C0 task after active-test selection."""
        return C0Task(
            task_id=self.task_id,
            left=self.left,
            right=self.right,
            expected=self.expected,
            operation=operation or self.operation,
        )


@dataclass(frozen=True)
class ActiveTestTrace:
    """Support-set reduction produced by an active test."""

    task_id: str
    selected_operation: str | None
    candidates_before: tuple[str, ...]
    candidates_after: tuple[str, ...]
    expected: C0Value
    support_reduced: bool

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly active-test trace."""
        return {
            "task_id": self.task_id,
            "selected_operation": self.selected_operation,
            "candidates_before": self.candidates_before,
            "candidates_after": self.candidates_after,
            "expected": self.expected,
            "support_reduced": self.support_reduced,
        }


def active_test(task: C2Task) -> ActiveTestTrace:
    """Reveal expected output and narrow executable candidates by typed equality."""
    outputs = candidate_outputs(task.left, task.right, task.candidates)
    narrowed = tuple(
        operation for operation, value in outputs if same_typed_value(value, task.expected)
    )
    return ActiveTestTrace(
        task_id=task.task_id,
        selected_operation=narrowed[0] if len(narrowed) == 1 else None,
        candidates_before=tuple(operation for operation, _ in outputs),
        candidates_after=narrowed,
        expected=task.expected,
        support_reduced=len(narrowed) < len(outputs),
    )


def active_curriculum(limit: int = 3) -> list[C2Task]:
    """Build deterministic C2 partial-observation tasks."""
    numbers = list(range(-limit, limit + 1))
    text_values = [f"p{index}" for index in range(0, (limit * 2) + 1)]
    tasks: list[C2Task] = []
    for left in numbers:
        for right in numbers:
            if left + right != left * right:
                tasks.append(active_task(left, right, left + right, "add"))
                tasks.append(active_task(left, right, left * right, "mul"))
            tasks.append(active_task(left, right, left == right, "eq"))
    for left in text_values:
        for right in text_values:
            tasks.append(active_task(left, right, left + right, "concat"))
            tasks.append(active_task(left, right, left == right, "eq"))
    return tasks


def active_task(
    left: C0Value,
    right: C0Value,
    expected: C0Value,
    operation: str,
) -> C2Task:
    """Construct one partial-observation active-test task."""
    candidates = tuple(
        operation for operation in C2_OPERATIONS if operation_applicable(operation, left, right)
    )
    return C2Task(
        task_id=f"c2-active-{value_token(left)}-{value_token(right)}-{value_token(expected)}",
        left=left,
        right=right,
        expected=expected,
        operation=operation,
        candidates=candidates,
    )


def candidate_outputs(
    left: C0Value,
    right: C0Value,
    candidates: tuple[str, ...],
) -> tuple[tuple[str, C0Value], ...]:
    """Evaluate supported candidate operations without assigning authority."""
    outputs: list[tuple[str, C0Value]] = []
    for operation in candidates:
        if not operation_applicable(operation, left, right):
            continue
        if operation == "add":
            outputs.append((operation, int(left) + int(right)))
        elif operation == "mul":
            outputs.append((operation, int(left) * int(right)))
        elif operation == "concat":
            outputs.append((operation, str(left) + str(right)))
        elif operation == "eq":
            outputs.append((operation, left == right))
    return tuple(outputs)


def operation_applicable(operation: str, left: C0Value, right: C0Value) -> bool:
    """Return true when an operation is type-valid for the observed operands."""
    if operation in {"add", "mul"}:
        return (
            isinstance(left, int)
            and isinstance(right, int)
            and not isinstance(left, bool)
            and not isinstance(right, bool)
        )
    if operation == "concat":
        return isinstance(left, str) and isinstance(right, str)
    return operation == "eq"


def same_typed_value(left: C0Value, right: C0Value) -> bool:
    """Compare values without allowing bool/int equality coercion."""
    return type(left) is type(right) and left == right


def stage_spec() -> StageSpec:
    """Runtime metadata for the C2 curriculum stage."""
    return StageSpec(
        name="C2",
        summary="partial observations, active tests, causal worlds",
        executable=True,
        implemented_components=(
            "partial-observation-generators",
            "active-test-selector",
            "uncertainty-action-scoring",
            "evc-halt-rule",
            "support-set-reduction",
            "c0-verifier-bridge",
            "noisy-causal-worlds",
            "multi-step-planning",
        ),
        blockers=(),
    )


__all__ = [
    "ActiveTestTrace",
    "C2Task",
    "CausalObservation",
    "CausalWorldTrace",
    "GridPlanStep",
    "GridPlanningTask",
    "GridPoint",
    "NoisyCausalWorld",
    "PlanningTrace",
    "active_curriculum",
    "active_task",
    "active_test",
    "candidate_outputs",
    "causal_world_curriculum",
    "grid_neighbors",
    "identify_causal_world",
    "mean_outcome",
    "plan_grid_task",
    "planning_curriculum",
    "same_typed_value",
    "stage_spec",
]
