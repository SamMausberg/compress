"""Curriculum stage ``C_0`` — cheap-verifier reusable tasks.

From §8 of ``docs/architecture/08-training-system.md``:

> ``C_0``: deterministic grids, strings, FSMs, small graphs, arithmetic,
> finite-state tasks, and data transformations with exact verifiers.

Exit gate (§8 curriculum paragraph + Criterion 1, §9): domain routing
solves cheap-verifier reusable tasks; ``DomOK`` holds for every solved
trace; no ``Ask``/``Abstain`` route is taken on a tractable instance.
"""

from __future__ import annotations

from dataclasses import dataclass

from vpm.tasks.spec import StageSpec


@dataclass(frozen=True)
class C0Task:
    """Cheap-verifier arithmetic task for the MVP vertical slice."""

    task_id: str
    left: int
    right: int
    expected: int
    operation: str = "add"

    @property
    def observation(self) -> str:
        """Compact observation consumed by the compiler."""
        return f"{self.operation} {self.left} {self.right}"


def addition_task(left: int, right: int, expected: int | None = None) -> C0Task:
    """Build a deterministic addition task with an exact expected answer."""
    answer = left + right if expected is None else expected
    return C0Task(task_id=f"c0-add-{left}-{right}", left=left, right=right, expected=answer)


def curriculum() -> list[C0Task]:
    """Small local C0 curriculum used by examples and tests."""
    return [
        addition_task(2, 3),
        addition_task(8, 13),
        addition_task(-4, 9),
    ]


def stage_spec() -> StageSpec:
    """Runtime metadata for the executable C0 stage."""
    return StageSpec(
        name="C0",
        summary="cheap-verifier executable arithmetic/data tasks",
        executable=True,
        implemented_components=("compiler", "dsl", "ledger", "verifier", "gate", "memory"),
    )


__all__ = ["C0Task", "addition_task", "curriculum", "stage_spec"]
