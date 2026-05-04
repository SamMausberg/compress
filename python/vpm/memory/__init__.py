"""§7 — Two-tier compressed memory and capacity limits.

See ``docs/architecture/07-compression-memory.md``.

This package will hold:

- ``active.py``    — ``A_act`` low-latency active set.
- ``archival.py``  — ``A_arc`` archival store (proposal data,
  replay material, sleep-phase candidates).
- ``library.py``   — hierarchical library
  ``Z_t = (D^prim, D^local, D^global, D^ctx, D^sem, D^src, D^safe,
  R^resid)`` (eq. 120).
- ``frontier.py``  — held-out rate-distortion frontier movement
  ``ΔFront_t(m)`` and the empirical-Bernstein bound stack (eqs. 121,
  125–128).
- ``admit.py``     — admission rule ``Admit_act`` (eq. 129) and demotion
  rule ``Demote`` (eq. 130).
- ``active_set.py`` — constrained selection problem (eq. 131; LP /
  greedy with budget constraints).

A macro is not a compressed capability unless it has an expansion
witness ``W_m`` and a scoped equivalence certificate ``Cert_eq(m)``
(eq. 124). The witness check is delegated to ``crates/vpm-egraph``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MemoryCapsule:
    """Replay-safe memory capsule admitted after a passed gate."""

    key: str
    value: object
    certificate_score: float
    witnesses: tuple[str, ...]


@dataclass
class MemoryLibrary:
    """Two-tier active/archive memory used by the MVP workflow."""

    active: dict[str, MemoryCapsule] = field(default_factory=dict)
    archive: dict[str, MemoryCapsule] = field(default_factory=dict)

    def admit(self, key: str, value: object, report: dict[str, object]) -> MemoryCapsule | None:
        """Admit only gate-passed reports with equivalence witness accounting."""
        gate = report.get("gate")
        if not isinstance(gate, dict) or gate.get("passed") is not True:
            return None
        score = float(gate.get("certificate_score", 0.0))
        canonical = report.get("canonical")
        witnesses = witness_names(canonical)
        capsule = MemoryCapsule(key, value, score, witnesses)
        if witnesses or score > 0.0:
            self.active[key] = capsule
        else:
            self.archive[key] = capsule
        return capsule


def witness_names(canonical: object) -> tuple[str, ...]:
    """Extract canonicalization witness names from a native report."""
    if not isinstance(canonical, dict):
        return ()
    witnesses = canonical.get("witnesses")
    if not isinstance(witnesses, list):
        return ()
    names: list[str] = []
    for witness in witnesses:
        if isinstance(witness, dict):
            rule = witness.get("rule")
            if isinstance(rule, str):
                names.append(rule)
    return tuple(names)


__all__ = ["MemoryCapsule", "MemoryLibrary", "witness_names"]
