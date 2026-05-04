"""Curriculum stage ``C_0`` — cheap-verifier reusable tasks.

From §8 of ``docs/architecture/08-training-system.md``:

> ``C_0``: deterministic grids, strings, FSMs, small graphs, arithmetic,
> finite-state tasks, and data transformations with exact verifiers.

Exit gate (§8 curriculum paragraph + Criterion 1, §9): domain routing
solves cheap-verifier reusable tasks; ``DomOK`` holds for every solved
trace; no ``Ask``/``Abstain`` route is taken on a tractable instance.
"""

from __future__ import annotations

__all__: list[str] = []
