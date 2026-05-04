"""§4 — Natural-language contractization, rebuttal, and semantic realization.

See ``docs/architecture/04-natural-language.md``.

This package will hold:

- ``normal_form.py``  — conversation normal form ``n = (ι, D, C^ctx,
  A^sem, Δ, B, Q^ask, R^real)`` (eqs. 50–52).
- ``context.py``      — context normalizer ``H_ctx``, ``ε_ctx``,
  ``CtxOK`` (eqs. 55–58).
- ``semantic.py``     — semantic contractizer ``H_sem``, ``ε_sem``,
  ``SemOK`` (eqs. 60–63).
- ``vague.py``        — undefined-vague-predicate handling (eq. 59).
- ``clarify.py``      — clarification policy ``q^*`` (eq. 64).
- ``realization.py``  — independent round-trip realization checker
  ``ε_real``, ``RealOK`` (eqs. 75–77). The checker model is **disjoint**
  from the renderer.
- ``render.py``       — atom-conditioned renderer; the final
  ``Gate_NL`` (eqs. 80–81) is delegated to
  :mod:`vpm._native.verify.gate_nl`.

Mode partition (eqs. 53–54): ``M_cert`` requires entailment, source,
rebuttal, authority, privacy, context, support, substrate, family-spending,
and realization certificates; ``M_soft`` requires mode separation,
influence safety, privacy, and no hidden certified-mode atom.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NormalForm:
    """Conversation normal form subset used by the MVP compiler."""

    intent: str
    atoms: tuple[str, ...]
    context_loss: float
    semantic_loss: float
    realization_loss: float
    ask: str | None = None

    @property
    def ok(self) -> bool:
        """True when the parser did not collapse unresolved meaning."""
        return self.context_loss < 1.0 and self.semantic_loss < 1.0


def normalize(observation: str) -> NormalForm:
    """Normalize a compact C0 observation such as ``add 2 3``."""
    parts = observation.strip().split()
    if len(parts) == 3 and parts[0] == "add":
        try:
            int(parts[1])
            int(parts[2])
        except ValueError:
            return NormalForm("unknown", (), 1.0, 1.0, 1.0, "Please provide integers.")
        return NormalForm("compute", (f"add({parts[1]},{parts[2]})",), 0.0, 0.0, 0.0)
    return NormalForm("unknown", (), 1.0, 1.0, 1.0, "Please provide an add task.")


def render_certified(value: object, certificate_score: float) -> str:
    """Render a certified value without introducing extra factual atoms."""
    return f"{value} [cert={certificate_score:.3f}]"


def render_question(normal_form: NormalForm) -> str:
    """Render a clarification question from a failed normal form."""
    return normal_form.ask or "Please clarify the task."


__all__ = ["NormalForm", "normalize", "render_certified", "render_question"]
