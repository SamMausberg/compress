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

from vpm.memory.active import MemoryCapsule
from vpm.memory.archival import ArchivalMemory
from vpm.memory.library import MemoryLibrary, witness_names

__all__ = ["ArchivalMemory", "MemoryCapsule", "MemoryLibrary", "witness_names"]
