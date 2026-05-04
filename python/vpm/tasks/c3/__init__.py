"""Curriculum stage ``C_3`` — tools, authority, and adversarial channels.

From §8 of ``docs/architecture/08-training-system.md``:

> ``C_3``: formal tools, agent workflows, tool use, authority,
> declassification, prompt-injection, rollback, influence-risk,
> and conflict.

Exit gate: zero data-noninterference violations under prompt-injection
red-team replay (Proposition 3, §6); rollback credits restore every
gated coordinate they're applicable to (eqs. 111–112); ``Gate(a, Z, Γ)``
(eq. 113) rejects every adversarial action that violates a budget.
"""

from __future__ import annotations

__all__: list[str] = []
