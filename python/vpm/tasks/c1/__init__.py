"""Curriculum stage ``C_1`` — hidden-schema mechanism tasks.

From §8 of ``docs/architecture/08-training-system.md``:

> ``C_1``: hidden-schema splits, equality saturation, typed program
> synthesis, theorem-proving fragments.

Exit gate: held-out certified utility on hidden-schema splits exceeds
the same-budget transformer / state-space / neural-program-synthesis
baselines (Criterion 1, §9, first clause).
"""

from __future__ import annotations

from dataclasses import dataclass

from vpm.tasks.c0 import C0Task, C0Value, value_token
from vpm.tasks.spec import StageSpec


@dataclass(frozen=True)
class C1Task:
    """Operation-hidden relation task backed by an exact C0 verifier."""

    task_id: str
    schema_id: str
    left: C0Value
    right: C0Value
    expected: C0Value
    operation: str

    @property
    def observation(self) -> str:
        """Schema-hidden observation used by learned proposal code."""
        return f"{value_token(self.left)} {value_token(self.right)} -> {value_token(self.expected)}"

    def to_c0_task(self) -> C0Task:
        """Expose the verifier-backed task with its training/evaluation label."""
        return C0Task(
            task_id=self.task_id,
            left=self.left,
            right=self.right,
            expected=self.expected,
            operation=self.operation,
        )

    def to_hidden_c0_task(self) -> C0Task:
        """Expose only operands and expected value for learned inference."""
        return C0Task(
            task_id=f"{self.task_id}-hidden",
            left=self.left,
            right=self.right,
            expected=self.expected,
            operation="unknown",
        )


def hidden_schema_curriculum(limit: int = 3) -> list[C1Task]:
    """Build a deterministic hidden-schema curriculum over typed C0 relations."""
    numbers = list(range(-limit, limit + 1))
    text_values = [f"s{index}" for index in range(0, (limit * 2) + 1)]
    tasks: list[C1Task] = []
    for left in numbers:
        for right in numbers:
            if left + right != left * right:
                tasks.append(schema_task("lambda-0", left, right, left + right, "add"))
                tasks.append(schema_task("lambda-1", left, right, left * right, "mul"))
            tasks.append(schema_task("lambda-3", left, right, left == right, "eq"))
    for left in text_values:
        for right in text_values:
            tasks.append(schema_task("lambda-2", left, right, left + right, "concat"))
            tasks.append(schema_task("lambda-3", left, right, left == right, "eq"))
    tasks.extend(
        schema_task("lambda-3", left, right, left == right, "eq")
        for left in (False, True)
        for right in (False, True)
    )
    return tasks


def schema_split(limit: int = 3) -> tuple[list[C1Task], list[C1Task]]:
    """Split hidden-schema tasks without relying on randomized Python hashes."""
    train: list[C1Task] = []
    heldout: list[C1Task] = []
    for task in hidden_schema_curriculum(limit):
        (heldout if stable_task_key(task) % 5 == 0 else train).append(task)
    return train, heldout


def as_c0_tasks(tasks: list[C1Task]) -> list[C0Task]:
    """Convert C1 tasks to native-verifier C0 tasks."""
    return [task.to_c0_task() for task in tasks]


def schema_task(
    schema_id: str,
    left: C0Value,
    right: C0Value,
    expected: C0Value,
    operation: str,
) -> C1Task:
    """Construct an operation-hidden task with an opaque schema id."""
    return C1Task(
        task_id=(
            f"c1-{schema_id}-{value_token(left)}-{value_token(right)}-{value_token(expected)}"
        ),
        schema_id=schema_id,
        left=left,
        right=right,
        expected=expected,
        operation=operation,
    )


def stable_task_key(task: C1Task) -> int:
    """Stable split key independent of Python hash randomization."""
    raw = f"{task.schema_id}:{task.left}:{task.right}:{task.expected}"
    return sum((index + 1) * ord(char) for index, char in enumerate(raw))


def stage_spec() -> StageSpec:
    """Runtime metadata for the C1 curriculum stage."""
    return StageSpec(
        name="C1",
        summary="hidden-schema/equality-saturation tasks",
        executable=True,
        implemented_components=(
            "hidden-schema-generators",
            "same-budget-baselines",
            "c0-verifier-bridge",
        ),
        blockers=("multi-step synthesis", "theorem-proving fragments"),
    )


__all__ = [
    "C1Task",
    "as_c0_tasks",
    "hidden_schema_curriculum",
    "schema_split",
    "schema_task",
    "stage_spec",
]
