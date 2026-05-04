"""Appendix A — Compact inference procedure.

See ``docs/architecture/A-inference-procedure.md`` for the canonical
``VPM-Infer`` pseudocode.

This package will hold:

- ``loop.py``               — typed coroutine implementing the outer
  ``for r in 0..Rmax`` loop; each iteration is a single Rust-side
  transaction over the ledger.
- ``routing.py``            — ``CtxOK``, ``SemOK``, ``RefineContract``,
  ``EstimateCertifiability``, ``BottleneckVector``, ``DomainRoute``.
- ``cell.py``               — ``MechanismStateCell_θ`` (the typed-message
  recurrent update of §5 eq. 96).
- ``calibrated_losses.py``  — produces ``(ε, ε_sub, ε_ctx, ε_sem,
  ε_src, ε_rebut, ε_real, ε_dep, rvec, dfront)`` on each step.
- ``support_guard.py``      — wraps ``crates/vpm-verify::support_guard``
  (eqs. 92–95).
- ``staging.py``            — stage scheduler ``ι → σ → π → η`` (eqs.
  97–98).
- ``test_select.py``        — uncertainty-action selection ``e^*``
  (eq. 100).
- ``halt.py``               — ``EVC_r`` halting rule (eq. 101).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from vpm.compiler import CompiledProgram, compile_task
from vpm.language import NormalForm, render_certified, render_question
from vpm.memory import MemoryLibrary
from vpm.retrieval import RetrievalBundle, retrieve
from vpm.substrate import SubstrateState, encode_update
from vpm.tasks.c0 import C0Task, addition_task
from vpm.verifiers import certificate_score, gate_passed, native_c0_report


@dataclass
class InferenceResult:
    """End-to-end MVP inference result."""

    task_id: str
    route: str
    rendered: str
    compiled: CompiledProgram | None
    retrieval: RetrievalBundle | None
    substrate: SubstrateState | None
    native_report: dict[str, object]
    memory_active: int
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly result shape for CLI/API/tests."""
        return {
            "task_id": self.task_id,
            "route": self.route,
            "rendered": self.rendered,
            "compiled": asdict(self.compiled) if self.compiled else None,
            "retrieval": asdict(self.retrieval) if self.retrieval else None,
            "substrate": asdict(self.substrate) if self.substrate else None,
            "native_report": self.native_report,
            "memory_active": self.memory_active,
            "errors": self.errors,
        }


def run_task(task: C0Task, memory: MemoryLibrary | None = None) -> InferenceResult:
    """Run Appendix-A-shaped inference for a C0 executable task."""
    library = memory if memory is not None else MemoryLibrary()
    try:
        compiled = compile_task(task)
    except ValueError as exc:
        rendered = render_question(NormalForm("unknown", (), 1.0, 1.0, 1.0, str(exc)))
        return InferenceResult(task.task_id, "ask", rendered, None, None, None, {}, 0, [str(exc)])

    retrieval = retrieve(compiled)
    substrate = encode_update(compiled)
    report = native_c0_report(compiled)
    route = native_route(report)
    value = native_value(report)
    rendered = render_certified(value, certificate_score(report)) if gate_passed(report) else "refusal"
    library.admit(task.task_id, value, report)
    return InferenceResult(
        task_id=task.task_id,
        route=route,
        rendered=rendered,
        compiled=compiled,
        retrieval=retrieval,
        substrate=substrate,
        native_report=report,
        memory_active=len(library.active),
    )


def run_c0_add(left: int, right: int, expected: int | None = None) -> InferenceResult:
    """Convenience entry point for the public MVP workflow."""
    return run_task(addition_task(left, right, expected))


def native_route(report: dict[str, object]) -> str:
    """Extract the native domain route."""
    gate = report.get("gate")
    if isinstance(gate, dict):
        route = gate.get("route")
        if isinstance(route, str):
            return route
    return "abstain"


def native_value(report: dict[str, object]) -> object:
    """Extract the native value enum as a Python value."""
    value = report.get("value")
    if isinstance(value, dict):
        value_type = value.get("type")
        payload = value.get("value")
        if value_type in {"Int", "int"}:
            return int(payload)
        if value_type in {"Text", "text"}:
            return str(payload)
        if value_type in {"Bool", "bool"}:
            return bool(payload)
    return value


__all__ = ["InferenceResult", "native_route", "native_value", "run_c0_add", "run_task"]
