"""§9 — Evaluation, failure modes, and minimal implementation.

See ``docs/architecture/09-evaluation-failure.md``.

This package will hold:

- ``metrics.py``         — every metric named in the §9 reporting
  paragraph (certified utility, solve rate, ECE, support loss, …).
- ``strata.py``          — stratification by evidence level, domain
  route, mode, and curriculum stage.
- ``ablations.py``       — toggle each architectural component listed
  in §9; the runner is the only canonical way to produce the
  ablation table.
- ``failure_modes.py``   — Criterion 1 clauses encoded as a
  ``FailureMode`` enum and a per-clause detector.
- ``phase_transition.py`` — the compression phase-transition
  diagnostic that defines VPM-0 success (last paragraph of §9).
- ``saturation.py``      — saturation diagnostic
  ``sup_m LCB^seq A(m) ≤ 0`` (eq. 136).
- ``report.py``          — pretty-prints the metric table and writes
  JSON for CI consumption.
"""

from __future__ import annotations

__all__: list[str] = []
