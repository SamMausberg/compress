"""Curriculum stage ``C_2`` — partially observed and active worlds.

From §8 of ``docs/architecture/08-training-system.md``:

> ``C_2``: noisy/partial observations, active tests, small causal worlds,
> and verifiable planning.

Exit gate: support guard (§5 eqs. 92–95) fires at the calibrated rate;
test-selection ``e^*`` (§5 eq. 100) reduces certified posterior cost on
held-out trajectories.
"""

from __future__ import annotations

__all__: list[str] = []
