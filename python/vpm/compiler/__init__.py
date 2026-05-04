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

__all__: list[str] = []
