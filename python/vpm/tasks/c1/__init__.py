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


@dataclass(frozen=True)
class SynthesisStep:
    """One typed operation in a candidate synthesized program."""

    operation: str
    left: str
    right: str
    output: str

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly synthesis step."""
        return {
            "operation": self.operation,
            "left": self.left,
            "right": self.right,
            "output": self.output,
        }


@dataclass(frozen=True)
class MultiStepSynthesisTask:
    """Hidden C1 program-synthesis task with candidate multi-step programs."""

    task_id: str
    bindings: tuple[tuple[str, C0Value], ...]
    expected: C0Value
    candidates: tuple[tuple[SynthesisStep, ...], ...]

    @property
    def observation(self) -> str:
        """Observation hides the selected program while exposing typed examples."""
        bindings = ",".join(f"{name}={value_token(value)}" for name, value in self.bindings)
        return f"synthesize {bindings} -> {value_token(self.expected)}"

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly synthesis task."""
        return {
            "task_id": self.task_id,
            "bindings": self.bindings,
            "expected": self.expected,
            "candidates": [[step.to_dict() for step in candidate] for candidate in self.candidates],
        }


@dataclass(frozen=True)
class MultiStepSynthesisTrace:
    """Selected multi-step synthesis candidate and verifier outcome."""

    task_id: str
    expected: C0Value
    output: C0Value | None
    selected_index: int | None
    selected_steps: tuple[SynthesisStep, ...]
    candidates_tested: int
    errors: tuple[str, ...]

    @property
    def multi_step(self) -> bool:
        """True when the selected program has more than one step."""
        return len(self.selected_steps) > 1

    @property
    def passed(self) -> bool:
        """True when a typed multi-step program matched the expected value."""
        return self.multi_step and same_synthesis_value(self.output, self.expected)

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly synthesis trace."""
        return {
            "task_id": self.task_id,
            "expected": self.expected,
            "output": self.output,
            "selected_index": self.selected_index,
            "selected_steps": [step.to_dict() for step in self.selected_steps],
            "candidates_tested": self.candidates_tested,
            "multi_step": self.multi_step,
            "passed": self.passed,
            "errors": self.errors,
        }


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


def multi_step_synthesis_curriculum() -> tuple[MultiStepSynthesisTask, ...]:
    """Build typed multi-step synthesis probes."""
    return (
        MultiStepSynthesisTask(
            task_id="c1-synth-add-then-mul",
            bindings=(("x", 2), ("y", 3), ("z", 4)),
            expected=20,
            candidates=(
                (SynthesisStep("mul", "x", "y", "out"),),
                (
                    SynthesisStep("add", "x", "y", "sum"),
                    SynthesisStep("mul", "sum", "z", "out"),
                ),
            ),
        ),
        MultiStepSynthesisTask(
            task_id="c1-synth-concat-then-eq",
            bindings=(("left", "ab"), ("right", "cd"), ("target", "abcd")),
            expected=True,
            candidates=(
                (SynthesisStep("concat", "left", "right", "out"),),
                (
                    SynthesisStep("concat", "left", "right", "joined"),
                    SynthesisStep("eq", "joined", "target", "out"),
                ),
            ),
        ),
    )


def run_multistep_synthesis(task: MultiStepSynthesisTask) -> MultiStepSynthesisTrace:
    """Select the first candidate program that exactly verifies."""
    errors: list[str] = []
    for index, candidate in enumerate(task.candidates):
        try:
            output = execute_synthesis_program(task.bindings, candidate)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if same_synthesis_value(output, task.expected):
            return MultiStepSynthesisTrace(
                task_id=task.task_id,
                expected=task.expected,
                output=output,
                selected_index=index,
                selected_steps=candidate,
                candidates_tested=index + 1,
                errors=tuple(errors),
            )
    return MultiStepSynthesisTrace(
        task_id=task.task_id,
        expected=task.expected,
        output=None,
        selected_index=None,
        selected_steps=(),
        candidates_tested=len(task.candidates),
        errors=tuple(errors),
    )


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


def execute_synthesis_program(
    bindings: tuple[tuple[str, C0Value], ...],
    program: tuple[SynthesisStep, ...],
) -> C0Value | None:
    """Execute a typed candidate synthesis program."""
    env = dict(bindings)
    for step in program:
        if step.left not in env or step.right not in env:
            raise ValueError(f"unbound synthesis input in {step.output}")
        env[step.output] = apply_synthesis_operation(
            step.operation,
            env[step.left],
            env[step.right],
        )
    return env[program[-1].output] if program else None


def apply_synthesis_operation(operation: str, left: C0Value, right: C0Value) -> C0Value:
    """Apply one supported typed synthesis operation."""
    if operation in {"add", "mul"}:
        if (
            not isinstance(left, int)
            or not isinstance(right, int)
            or isinstance(left, bool)
            or isinstance(right, bool)
        ):
            raise ValueError(f"{operation} expects integer operands")
        return left + right if operation == "add" else left * right
    if operation == "concat":
        if not isinstance(left, str) or not isinstance(right, str):
            raise ValueError("concat expects text operands")
        return left + right
    if operation == "eq":
        return type(left) is type(right) and left == right
    raise ValueError(f"unsupported synthesis operation: {operation}")


def same_synthesis_value(left: object, right: object) -> bool:
    """Compare synthesized values without bool/int coercion."""
    return type(left) is type(right) and left == right


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
            "multi-step-synthesis",
        ),
        blockers=("theorem-proving fragments",),
    )


__all__ = [
    "C1Task",
    "MultiStepSynthesisTask",
    "MultiStepSynthesisTrace",
    "SynthesisStep",
    "apply_synthesis_operation",
    "as_c0_tasks",
    "execute_synthesis_program",
    "hidden_schema_curriculum",
    "multi_step_synthesis_curriculum",
    "run_multistep_synthesis",
    "schema_split",
    "schema_task",
    "stage_spec",
]
