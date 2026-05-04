"""§4 — Source and rebuttal retrievers.

See ``docs/architecture/04-natural-language.md`` (eqs. 65–69) and the
retrieval-cache discipline at the end of §8.

This package will hold:

- ``source.py``  — minimal-sufficient-evidence retriever; produces
  ``R_t^+(a)`` and ``ε_src(a)`` against ``D_src^audit``.
- ``rebuttal.py`` — material-contradiction / scope-defeater retriever;
  produces ``R_t^-(a)`` and ``ε_rebut(a)``.
- ``cache.py``   — content-addressed cache keyed by atom, scope,
  freshness, and split (§8 efficiency paragraph).

For exact internal objects (arithmetic traces, proof terms), the policy
``R_a`` may set ``ε_src = ε_rebut = 0`` and shift the burden to
execution and proof verifiers in :mod:`vpm.verifiers` /
``crates/vpm-verify``.
"""

from __future__ import annotations

from dataclasses import dataclass

from vpm.compiler import CompiledProgram


@dataclass(frozen=True)
class RetrievalBundle:
    """Source/rebuttal retrieval result with calibrated loss bounds."""

    sources: tuple[str, ...]
    rebuttals: tuple[str, ...]
    source_loss: float
    rebuttal_loss: float


def retrieve(compiled: CompiledProgram) -> RetrievalBundle:
    """Retrieve exact internal support for C0 executable objects."""
    atom = compiled.normal_form.atoms[0]
    source = f"exact-arithmetic:{atom}"
    return RetrievalBundle((source,), (), 0.0, 0.0)


__all__ = ["RetrievalBundle", "retrieve"]
