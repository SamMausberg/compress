"""Curriculum stage ``C_2`` — partially observed and active worlds.

From §8 of ``docs/architecture/08-training-system.md``:

> ``C_2``: noisy/partial observations, active tests, small causal worlds,
> and verifiable planning.

Exit gate: support guard (§5 eqs. 92–95) fires at the calibrated rate;
test-selection ``e^*`` (§5 eq. 100) reduces certified posterior cost on
held-out trajectories.
"""

from __future__ import annotations

from vpm.tasks.spec import StageSpec


def stage_spec() -> StageSpec:
    """Runtime metadata for the C2 curriculum stage."""
    return StageSpec(
        name="C2",
        summary="partial observations, active tests, causal worlds",
        executable=False,
        implemented_components=("stage-metadata",),
        blockers=("active-test selector", "support guard calibration"),
    )


__all__ = ["stage_spec"]
