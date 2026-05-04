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

import json
from typing import cast

from vpm import _native
from vpm._reports import float_field, object_map
from vpm.compiler import CompiledProgram


def native_c0_report(compiled: CompiledProgram) -> dict[str, object]:
    """Run the native exact verifier/gate path for a compiled C0 task."""
    raw = _native.run_c0_arith_json(
        compiled.operation,
        compiled.left,
        compiled.right,
        compiled.expected,
    )
    return cast(dict[str, object], json.loads(raw))


def certificate_score(report: dict[str, object]) -> float:
    """Read the gate certificate score from a native report."""
    gate = object_map(report.get("gate"))
    if gate is None:
        return 0.0
    return float_field(gate, "certificate_score")


def gate_passed(report: dict[str, object]) -> bool:
    """True when the native conjunctive gate passed."""
    gate = object_map(report.get("gate"))
    return gate is not None and gate.get("passed") is True


__all__ = ["certificate_score", "gate_passed", "native_c0_report"]
