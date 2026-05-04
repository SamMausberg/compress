"""§2 / §6 — Verifier implementations (Python side).

See ``docs/architecture/02-typed-mechanisms-evidence.md`` and
``docs/architecture/06-verification-authority.md``.

The **runtime** (claim martingale, dependence-block factorization,
gate kernel) lives in Rust under ``crates/vpm-verify``. This Python
package holds:

- ``confidence.py``  — sequence-valid UCB ``p_{v,r}^+`` and
  empirical-Bernstein bounds (eqs. 22–23, 125–128).
- ``substrate.py``   — substrate-omission calibration ``ε_sub``
  (eq. 47) and critical-edge audit ``ε_crit`` (eqs. 48–49).
- ``entailment.py``  — entailment witness checker ``Ent(P_a, e_a, C^ctx)``
  (eq. 70).
- ``attacks.py``     — learned attacks for soft domains (eq. 102, soft
  branch).
- ``registry.py``    — Python view of the verifier registry; thin wrap
  around ``vpm._native.verify``.

A verifier cannot certify a candidate it generated unless an independent
replay set or independent verifier family agrees (the ``gen_v``
invariant of §2). This is enforced by typing ``EVal`` as opaque on the
Rust side.
"""

from __future__ import annotations

__all__: list[str] = []
