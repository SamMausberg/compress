"""Curriculum stage ``C_4`` — controlled dialogue and rendered text.

From §8 of ``docs/architecture/08-training-system.md``:

> ``C_4``: controlled dialogue/artifacts with context normal forms,
> semantic atom extraction, source-grounded QA, contradiction search,
> entailment checking, round-trip realization checking, calibrated
> uncertainty, and intent-entropy gates.

Exit gate: ``Gate_NL`` (§4 eqs. 80–81; Proposition 2) holds on every
rendered transcript; no certified-mode atom is rendered without
entailment + source + rebuttal + round-trip witnesses.
"""

from __future__ import annotations

from vpm.tasks.spec import StageSpec


def stage_spec() -> StageSpec:
    """Runtime metadata for the C4 curriculum stage."""
    return StageSpec(
        name="C4",
        summary="controlled dialogue, grounded QA, realization gates",
        executable=False,
        implemented_components=("normal-form-parser", "renderer", "stage-metadata"),
        blockers=("source corpus", "entailment checker", "round-trip checker"),
    )


__all__ = ["stage_spec"]
