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

from vpm._reports import float_field, object_list, object_map


@dataclass(frozen=True)
class MemoryCapsule:
    """Replay-safe memory capsule admitted after a passed gate."""

    key: str
    value: object
    certificate_score: float
    witnesses: tuple[str, ...]


def dict_capsules() -> dict[str, MemoryCapsule]:
    """Typed default factory for capsule dictionaries."""
    return {}


@dataclass
class MemoryLibrary:
    """Two-tier active/archive memory used by the MVP workflow."""

    active: dict[str, MemoryCapsule] = field(default_factory=dict_capsules)
    archive: dict[str, MemoryCapsule] = field(default_factory=dict_capsules)

    def admit(self, key: str, value: object, report: dict[str, object]) -> MemoryCapsule | None:
        """Admit only gate-passed reports with equivalence witness accounting."""
        gate = object_map(report.get("gate"))
        if gate is None or gate.get("passed") is not True:
            return None
        score = float_field(gate, "certificate_score")
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
    canonical_map = object_map(canonical)
    if canonical_map is None:
        return ()
    witnesses = object_list(canonical_map.get("witnesses"))
    if witnesses is None:
        return ()
    names: list[str] = []
    for witness in witnesses:
        witness_map = object_map(witness)
        if witness_map is None:
            continue
        rule = witness_map.get("rule")
        if isinstance(rule, str):
            names.append(rule)
    return tuple(names)


__all__ = ["MemoryCapsule", "MemoryLibrary", "witness_names"]
