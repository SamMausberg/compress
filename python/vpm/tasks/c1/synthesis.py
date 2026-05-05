"""Typed program-synthesis controls for C1 tasks."""

from __future__ import annotations

from dataclasses import dataclass

from vpm.tasks.c0 import C0Value, value_token


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


__all__ = [
    "MultiStepSynthesisTask",
    "MultiStepSynthesisTrace",
    "SynthesisStep",
    "apply_synthesis_operation",
    "execute_synthesis_program",
    "multi_step_synthesis_curriculum",
    "run_multistep_synthesis",
    "same_synthesis_value",
]
