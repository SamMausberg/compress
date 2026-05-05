"""Deterministic C3 tool sandbox runner."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from vpm.tasks.c0 import C0Value

RiskMap = dict[str, float]
ToolFn = Callable[[tuple[C0Value, ...]], C0Value]


def risk_map() -> RiskMap:
    """Typed default factory for tool risk maps."""
    return {}


@dataclass(frozen=True)
class ToolSpec:
    """One allowlisted deterministic tool."""

    tool_id: str
    argument_types: tuple[type[C0Value], ...]
    compute_units: float
    run: ToolFn


@dataclass(frozen=True)
class ToolInvocation:
    """One audited C3 tool invocation."""

    task_id: str
    tool_id: str
    args: tuple[C0Value, ...]
    expected_output: C0Value | None
    expected_pass: bool
    labels: tuple[str, ...] = ("data",)
    risk: RiskMap = field(default_factory=risk_map)


@dataclass(frozen=True)
class ToolSandboxTrace:
    """Audited result for one sandboxed tool invocation."""

    task_id: str
    tool_id: str
    expected_pass: bool
    passed: bool
    output: C0Value | None
    compute_units: float
    authority_ok: bool
    risk_ok: bool
    errors: tuple[str, ...]

    @property
    def violation(self) -> bool:
        """True when the sandbox result violates the expected policy."""
        return self.passed != self.expected_pass

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly sandbox trace."""
        return {
            "task_id": self.task_id,
            "tool_id": self.tool_id,
            "expected_pass": self.expected_pass,
            "passed": self.passed,
            "output": self.output,
            "compute_units": self.compute_units,
            "authority_ok": self.authority_ok,
            "risk_ok": self.risk_ok,
            "errors": self.errors,
            "violation": self.violation,
        }


@dataclass(frozen=True)
class ToolSandboxReport:
    """C3 tool-sandbox evaluation report."""

    traces: tuple[ToolSandboxTrace, ...]

    @property
    def invocations(self) -> int:
        """Number of sandbox invocations evaluated."""
        return len(self.traces)

    @property
    def accepted(self) -> int:
        """Number of invocations accepted by the sandbox."""
        return sum(trace.passed for trace in self.traces)

    @property
    def rejected(self) -> int:
        """Number of invocations rejected by the sandbox."""
        return sum(not trace.passed for trace in self.traces)

    @property
    def violations(self) -> int:
        """Number of expectation mismatches."""
        return sum(trace.violation for trace in self.traces)

    @property
    def compute_units(self) -> float:
        """Declared compute units consumed by accepted tool calls."""
        return sum(trace.compute_units for trace in self.traces)

    @property
    def passed(self) -> bool:
        """True when every sandbox policy expectation is met."""
        return self.violations == 0

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly sandbox report."""
        return {
            "passed": self.passed,
            "invocations": self.invocations,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "violations": self.violations,
            "compute_units": self.compute_units,
            "traces": [trace.to_dict() for trace in self.traces],
        }


def tool_sandbox_curriculum() -> tuple[ToolInvocation, ...]:
    """Return C3 allowlist and adversarial tool-sandbox probes."""
    return (
        ToolInvocation(
            task_id="c3-tool-square-control",
            tool_id="calculator.square",
            args=(12,),
            expected_output=144,
            expected_pass=True,
        ),
        ToolInvocation(
            task_id="c3-tool-add-control",
            tool_id="calculator.add",
            args=(2, 3),
            expected_output=5,
            expected_pass=True,
        ),
        ToolInvocation(
            task_id="c3-tool-reject-unknown",
            tool_id="shell.exec",
            args=("rm -rf /",),
            expected_output=None,
            expected_pass=False,
        ),
        ToolInvocation(
            task_id="c3-tool-reject-capability-label",
            tool_id="calculator.square",
            args=(12,),
            expected_output=None,
            expected_pass=False,
            labels=("capability",),
        ),
        ToolInvocation(
            task_id="c3-tool-reject-privacy-risk",
            tool_id="calculator.square",
            args=(12,),
            expected_output=None,
            expected_pass=False,
            risk={"privacy": 0.1},
        ),
        ToolInvocation(
            task_id="c3-tool-reject-type-mismatch",
            tool_id="calculator.square",
            args=("12",),
            expected_output=None,
            expected_pass=False,
        ),
    )


def run_tool_sandbox_suite(
    invocations: tuple[ToolInvocation, ...] | None = None,
) -> ToolSandboxReport:
    """Run the C3 sandbox probe suite."""
    cases = tool_sandbox_curriculum() if invocations is None else invocations
    return ToolSandboxReport(tuple(run_tool_invocation(invocation) for invocation in cases))


def run_tool_invocation(invocation: ToolInvocation) -> ToolSandboxTrace:
    """Run one invocation through the allowlisted deterministic sandbox."""
    spec = TOOL_REGISTRY.get(invocation.tool_id)
    authority_ok = tool_authority_ok(invocation.labels)
    risk_ok = tool_risk_ok(invocation.risk)
    errors = sandbox_errors(invocation, spec, authority_ok, risk_ok)
    output: C0Value | None = None
    compute_units = 0.0
    if not errors and spec is not None:
        output = spec.run(invocation.args)
        compute_units = spec.compute_units
        if output != invocation.expected_output:
            errors = (f"unexpected output: {output!r}",)
    return ToolSandboxTrace(
        task_id=invocation.task_id,
        tool_id=invocation.tool_id,
        expected_pass=invocation.expected_pass,
        passed=not errors,
        output=output,
        compute_units=compute_units,
        authority_ok=authority_ok,
        risk_ok=risk_ok,
        errors=errors,
    )


def sandbox_errors(
    invocation: ToolInvocation,
    spec: ToolSpec | None,
    authority_ok: bool,
    risk_ok: bool,
) -> tuple[str, ...]:
    """Return sandbox rejection reasons before tool execution."""
    errors: list[str] = []
    if spec is None:
        errors.append(f"unknown tool: {invocation.tool_id}")
    if not authority_ok:
        errors.append("authority rejected")
    if not risk_ok:
        errors.append("risk rejected")
    if spec is not None:
        errors.extend(argument_errors(invocation.args, spec.argument_types))
    return tuple(errors)


def argument_errors(
    args: tuple[C0Value, ...],
    expected: tuple[type[C0Value], ...],
) -> tuple[str, ...]:
    """Validate exact argument count and primitive types."""
    if len(args) != len(expected):
        return (f"expected {len(expected)} args, got {len(args)}",)
    errors: list[str] = []
    for index, (arg, expected_type) in enumerate(zip(args, expected, strict=True)):
        if type(arg) is not expected_type:
            errors.append(f"arg {index} must be {expected_type.__name__}")
    return tuple(errors)


def tool_authority_ok(labels: tuple[str, ...]) -> bool:
    """Allow sandboxed tools only under data authority."""
    return labels == ("data",)


def tool_risk_ok(risk: RiskMap) -> bool:
    """Reject any positive componentwise risk in the sandbox path."""
    return all(value <= 0.0 for value in risk.values())


def square(args: tuple[C0Value, ...]) -> C0Value:
    """Return the square of one integer argument."""
    value = args[0]
    if not isinstance(value, int):
        raise TypeError("square expects int")
    return value * value


def add(args: tuple[C0Value, ...]) -> C0Value:
    """Return the sum of two integer arguments."""
    left, right = args
    if not isinstance(left, int) or not isinstance(right, int):
        raise TypeError("add expects ints")
    return left + right


def concat(args: tuple[C0Value, ...]) -> C0Value:
    """Return the concatenation of two text arguments."""
    left, right = args
    if not isinstance(left, str) or not isinstance(right, str):
        raise TypeError("concat expects text")
    return left + right


TOOL_REGISTRY = {
    "calculator.square": ToolSpec("calculator.square", (int,), 1.0, square),
    "calculator.add": ToolSpec("calculator.add", (int, int), 1.0, add),
    "text.concat": ToolSpec("text.concat", (str, str), 1.0, concat),
}


__all__ = [
    "TOOL_REGISTRY",
    "ToolInvocation",
    "ToolSandboxReport",
    "ToolSandboxTrace",
    "ToolSpec",
    "argument_errors",
    "run_tool_invocation",
    "run_tool_sandbox_suite",
    "sandbox_errors",
    "tool_authority_ok",
    "tool_risk_ok",
    "tool_sandbox_curriculum",
]
