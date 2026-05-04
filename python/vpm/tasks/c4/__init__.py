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

__all__: list[str] = []
