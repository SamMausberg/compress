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

C0Value = int | str | bool


@dataclass(frozen=True)
class C0Task:
    """Cheap-verifier typed task for the MVP vertical slice."""

    task_id: str
    left: C0Value
    right: C0Value
    expected: C0Value
    operation: str = "add"

    @property
    def observation(self) -> str:
        """Compact observation consumed by the compiler."""
        return f"{self.operation} {self.left} {self.right}"


def addition_task(left: int, right: int, expected: int | None = None) -> C0Task:
    """Build a deterministic addition task with an exact expected answer."""
    answer = left + right if expected is None else expected
    return C0Task(task_id=f"c0-add-{left}-{right}", left=left, right=right, expected=answer)


def multiplication_task(left: int, right: int, expected: int | None = None) -> C0Task:
    """Build a deterministic multiplication task with an exact expected answer."""
    answer = left * right if expected is None else expected
    return C0Task(
        task_id=f"c0-mul-{left}-{right}",
        left=left,
        right=right,
        expected=answer,
        operation="mul",
    )


def concat_task(left: str, right: str, expected: str | None = None) -> C0Task:
    """Build a deterministic string-concatenation task."""
    ensure_token(left)
    ensure_token(right)
    answer = left + right if expected is None else expected
    ensure_token(answer)
    return C0Task(
        task_id=f"c0-concat-{left}-{right}",
        left=left,
        right=right,
        expected=answer,
        operation="concat",
    )


def equality_task(left: C0Value, right: C0Value, expected: bool | None = None) -> C0Task:
    """Build a deterministic typed-equality task."""
    answer = left == right if expected is None else expected
    return C0Task(
        task_id=f"c0-eq-{value_token(left)}-{value_token(right)}",
        left=left,
        right=right,
        expected=answer,
        operation="eq",
    )


def hidden_task(left: C0Value, right: C0Value, expected: C0Value) -> C0Task:
    """Build an operation-hidden task for learned proposal inference."""
    return C0Task(
        task_id=f"c0-hidden-{value_token(left)}-{value_token(right)}-{value_token(expected)}",
        left=left,
        right=right,
        expected=expected,
        operation="unknown",
    )


def arithmetic_task(operation: str, left: int, right: int, expected: int | None = None) -> C0Task:
    """Build a supported C0 arithmetic task."""
    if operation == "add":
        return addition_task(left, right, expected)
    if operation == "mul":
        return multiplication_task(left, right, expected)
    raise ValueError(f"unsupported C0 arithmetic operation: {operation}")


def typed_task(
    operation: str,
    left: str,
    right: str,
    expected: str | None = None,
) -> C0Task:
    """Build a C0 task from CLI-friendly typed tokens."""
    if operation in {"add", "mul"}:
        parsed_expected = None if expected is None else int(expected)
        return arithmetic_task(operation, int(left), int(right), parsed_expected)
    if operation == "concat":
        return concat_task(left, right, expected)
    if operation == "eq":
        parsed_left = parse_token(left)
        parsed_right = parse_token(right)
        parsed_expected = None if expected is None else parse_bool(expected)
        return equality_task(parsed_left, parsed_right, parsed_expected)
    raise ValueError(f"unsupported C0 typed operation: {operation}")


def typed_hidden_task(left: str, right: str, expected: str) -> C0Task:
    """Build an operation-hidden task from CLI-friendly typed tokens."""
    return hidden_task(parse_token(left), parse_token(right), parse_token(expected))


def curriculum() -> list[C0Task]:
    """Small local C0 curriculum used by examples and tests."""
    return [
        addition_task(2, 3),
        addition_task(8, 13),
        addition_task(-4, 9),
        multiplication_task(6, 7),
        concat_task("ab", "cd"),
        equality_task(5, 5),
        equality_task("left", "right"),
    ]


def parse_token(raw: str) -> C0Value:
    """Parse a compact typed token."""
    if raw == "true":
        return True
    if raw == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        ensure_token(raw)
        return raw


def parse_bool(raw: str) -> bool:
    """Parse a lowercase boolean token."""
    if raw == "true":
        return True
    if raw == "false":
        return False
    raise ValueError("expected must be true or false for eq")


def value_token(value: C0Value) -> str:
    """Render a compact typed token."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def ensure_token(value: str) -> None:
    """Reject whitespace-bearing strings until the C0 parser supports quoting."""
    if not value or any(char.isspace() for char in value):
        raise ValueError("C0 string operands must be non-empty tokens without whitespace")


def stage_spec() -> StageSpec:
    """Runtime metadata for the executable C0 stage."""
    return StageSpec(
        name="C0",
        summary="cheap-verifier executable arithmetic/data tasks",
        executable=True,
        implemented_components=("compiler", "dsl", "ledger", "verifier", "gate", "memory"),
    )


__all__ = [
    "C0Task",
    "C0Value",
    "addition_task",
    "arithmetic_task",
    "concat_task",
    "curriculum",
    "equality_task",
    "hidden_task",
    "multiplication_task",
    "stage_spec",
    "typed_hidden_task",
    "typed_task",
]
