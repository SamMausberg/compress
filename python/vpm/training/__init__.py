"""§8 — Training system and inefficiency controls.

See ``docs/architecture/08-training-system.md``.

This package will hold:

- ``splits.py``           — data-split topology (eqs. 137–139); a single
  trace cannot leak across the certificate-carrying event.
- ``teacher.py``          — truncated certified-trace posterior ``p_*``
  (eq. 141).
- ``losses/``             — one module per loss in eqs. 142–168
  (``base``, ``cmp``, ``trace``, ``value``, ``halt``, ``ver``,
  ``cal``, ``safe``, ``mem``, ``supp``, ``render``, ``ctx``,
  ``sem``, ``src``, ``rebut``, ``ent``, ``real``, ``tb``, ``mf``,
  ``split``, ``sub``, ``dom``, ``dep``, ``front``, ``probe``,
  ``repair``).
- ``weight_balancer.py``  — gradient-scale-normalized loss weights
  (eq. 169).
- ``coordinator.py``      — block-coordinate optimizer with frozen
  audit labels and hard gates (eqs. 170–174).
- ``budget.py``            — dual-price budget allocator ``B^*`` (eqs.
  175–176) and the KKT balancing diagnostic (eq. 177).
- ``active_query.py``     — teacher search on the posterior boundary
  (eq. 178).
- ``gflow.py``            — GFlowNet trajectory-balance loss for
  diverse certified mechanism proposals (eqs. 166–167; refs [7][8]).
- ``probes.py``           — counterfactual edge-deletion / parent-swap
  probes (§3 eqs. 48–49 audit).
- ``repair.py``           — outer repair-operator search ``ρ_t^*``
  (eqs. 17–18).

Training is **execution-first** and **block-coordinate**. Cheap
imitation losses cannot consume budget reserved for verification,
calibration, or recall heads while the active certificate bottleneck
remains elsewhere.
"""

from __future__ import annotations

__all__: list[str] = []
