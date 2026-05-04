"""§5 — Compiler ``q_φ``, posterior, and support-preserving inference.

See ``docs/architecture/05-compiler-posterior.md``.

This package will hold:

- ``posterior.py`` — ``q_φ(c, n, Γ̂, A, P | O, H)`` over alternatives,
  with parser / context / semantic / realization pruning bounds
  (eq. 83).
- ``cnf_posterior.py`` — wraps :mod:`vpm.language.context` and
  :mod:`vpm.language.semantic` to produce ``Ψ`` for the inference loop.
- ``energy.py``   — posterior energy ``E(μ, c, n, V; T, Γ)`` (eqs. 86–87).
- ``score_head.py`` — bounded trainable proposal prior ``s_θ`` (eq. 88).
- ``compile.py``  — orchestrates the above.

The compiler is **posterior-valued**: canonicalization may merge states
only with a reversible witness or a support-loss bound (eqs. 84–85),
both delegated to ``crates/vpm-egraph``.
"""

from __future__ import annotations

from dataclasses import dataclass

from vpm.language import NormalForm, normalize
from vpm.tasks.c0 import C0Task


@dataclass(frozen=True)
class CompiledProgram:
    """Posterior-selected executable candidate for the MVP."""

    task_id: str
    operation: str
    left: int
    right: int
    expected: int
    normal_form: NormalForm
    support_loss: float = 0.0

    @property
    def args(self) -> tuple[int, int]:
        """Typed program arguments."""
        return (self.left, self.right)


def cnf_posterior(task: C0Task | str) -> NormalForm:
    """Produce the C0 conversation normal form posterior."""
    observation = task.observation if isinstance(task, C0Task) else task
    return normalize(observation)


def compile_task(task: C0Task) -> CompiledProgram:
    """Compile a C0 task into the executable native program boundary."""
    normal_form = cnf_posterior(task)
    if not normal_form.ok:
        msg = normal_form.ask or "task is not in the C0 executable language"
        raise ValueError(msg)
    return CompiledProgram(
        task_id=task.task_id,
        operation=task.operation,
        left=task.left,
        right=task.right,
        expected=task.expected,
        normal_form=normal_form,
    )


__all__ = ["CompiledProgram", "cnf_posterior", "compile_task"]
