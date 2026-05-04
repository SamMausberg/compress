"""Curriculum stage ``C_1`` — hidden-schema mechanism tasks.

From §8 of ``docs/architecture/08-training-system.md``:

> ``C_1``: hidden-schema splits, equality saturation, typed program
> synthesis, theorem-proving fragments.

Exit gate: held-out certified utility on hidden-schema splits exceeds
the same-budget transformer / state-space / neural-program-synthesis
baselines (Criterion 1, §9, first clause).
"""

from __future__ import annotations

from vpm.tasks.spec import StageSpec


def stage_spec() -> StageSpec:
    """Runtime metadata for the C1 curriculum stage."""
    return StageSpec(
        name="C1",
        summary="hidden-schema/equality-saturation tasks",
        executable=False,
        implemented_components=("stage-metadata",),
        blockers=("hidden-schema generators", "same-budget baselines"),
    )


__all__ = ["stage_spec"]
