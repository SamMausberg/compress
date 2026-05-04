"""Curriculum stage ``C_5`` — continual compression and replay.

From §8 of ``docs/architecture/08-training-system.md``:

> ``C_5``: continual compression with replay-safe macro admission.

Exit gate: a compression phase transition is observed (§9, last
paragraph): solved-task cost falls, active memory grows sublinearly,
held-out certified utility rises, rate-distortion frontier decreases,
and support / context / semantic / source / rebuttal / realization /
dependency / substrate losses remain bounded. Vector-risk budgets stay
valid and false-pass bounds do not rise.
"""

from __future__ import annotations

from vpm.tasks.spec import StageSpec


def stage_spec() -> StageSpec:
    """Runtime metadata for the C5 curriculum stage."""
    return StageSpec(
        name="C5",
        summary="continual compression and replay-safe macro admission",
        executable=False,
        implemented_components=("memory-library", "canonicalization-witnesses", "stage-metadata"),
        blockers=("frontier movement estimator", "macro demotion replay"),
    )


__all__ = ["stage_spec"]
