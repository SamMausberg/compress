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

__all__: list[str] = []
