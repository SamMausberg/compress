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

__all__: list[str] = []
