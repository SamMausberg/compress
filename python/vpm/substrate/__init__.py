"""§3 — Verifier-native neural substrate ``VNS_θ``.

See ``docs/architecture/03-neural-substrate.md``.

This package will hold:

- ``encoder.py``   — typed event hypergraph ``x_t = Enc_θ(o_t, Γ)`` (eq. 34).
- ``ssm.py``       — selective SSM block ``h_t^{ssm}`` (eq. 36; ref [4]).
- ``slots.py``     — slot binding update (eqs. 37–38; ref [9]).
- ``projection.py``— executable projection ``Proj_θ`` (eq. 39).
- ``losses.py``    — substrate losses ``L_base`` family (eqs. 42–46) and
  the substrate-recall calibration ``ε_sub`` / ``ε_crit`` (eqs. 47–49).
- ``graph_shadow.py`` — differentiable shadow ``G_t^θ`` of the e-graph
  in ``crates/vpm-egraph``.

The substrate is an **amortizer, not an authority** (Abstract): every
output of this module is a *proposal* and never a certificate.
"""

from __future__ import annotations

__all__: list[str] = []
