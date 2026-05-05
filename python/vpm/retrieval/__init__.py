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
- ``calibration.py`` — controlled source/rebuttal recall probes under
  normal-form corpus shifts.

For exact internal objects (arithmetic traces, proof terms), the policy
``R_a`` may set ``ε_src = ε_rebut = 0`` and shift the burden to
execution and proof verifiers in :mod:`vpm.verifiers` /
``crates/vpm-verify``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vpm.language.ambiguity import (
    AmbiguityAction,
    ambiguity_reasons,
    audited_atom,
    normalized_text,
)
from vpm.retrieval.calibration import (
    RecallChannel,
    RecallFn,
    RecallShiftProbe,
    RecallShiftReport,
    RecallShiftTrace,
    calibrated_recall,
    dirty_recall_shift_probe,
    evaluate_recall_shift,
    exact_suffix_recall,
    recall_shift_curriculum,
)

if TYPE_CHECKING:
    from vpm.compiler import CompiledProgram


@dataclass(frozen=True)
class RetrievalBundle:
    """Source/rebuttal retrieval result with calibrated loss bounds."""

    sources: tuple[str, ...]
    rebuttals: tuple[str, ...]
    source_loss: float
    rebuttal_loss: float


@dataclass(frozen=True)
class OpenDomainDocument:
    """Audited open-domain document with normalized retrieval aliases."""

    doc_id: str
    atom: str
    answer: str
    source: str
    aliases: tuple[str, ...]
    rebuttals: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly audited document."""
        return {
            "doc_id": self.doc_id,
            "atom": self.atom,
            "answer": self.answer,
            "source": self.source,
            "aliases": self.aliases,
            "rebuttals": self.rebuttals,
        }


@dataclass(frozen=True)
class OpenDomainRetrievalResult:
    """Open-domain retrieval result routed through ambiguity guards."""

    prompt: str
    action: AmbiguityAction
    document: OpenDomainDocument | None
    source_loss: float
    rebuttal_loss: float
    reasons: tuple[str, ...]

    @property
    def retrieved(self) -> bool:
        """True when a unique audited source was retrieved for certification."""
        return self.action is AmbiguityAction.CERTIFY and self.document is not None

    @property
    def atom(self) -> str | None:
        """Retrieved atom, when available."""
        return self.document.atom if self.document else None

    @property
    def answer(self) -> str | None:
        """Retrieved answer, when available."""
        return self.document.answer if self.document else None

    @property
    def sources(self) -> tuple[str, ...]:
        """Retrieved source witnesses."""
        return (self.document.source,) if self.document else ()

    @property
    def rebuttals(self) -> tuple[str, ...]:
        """Retrieved material rebuttal witnesses."""
        return self.document.rebuttals if self.document else ()

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly open-domain retrieval result."""
        return {
            "prompt": self.prompt,
            "action": self.action.value,
            "retrieved": self.retrieved,
            "atom": self.atom,
            "answer": self.answer,
            "sources": self.sources,
            "rebuttals": self.rebuttals,
            "source_loss": self.source_loss,
            "rebuttal_loss": self.rebuttal_loss,
            "reasons": self.reasons,
            "document": self.document.to_dict() if self.document else None,
        }


def retrieve(compiled: CompiledProgram) -> RetrievalBundle:
    """Retrieve exact internal support for C0 executable objects."""
    atom = compiled.normal_form.atoms[0]
    source = f"exact-arithmetic:{atom}"
    return RetrievalBundle((source,), (), 0.0, 0.0)


def retrieve_open_domain_prompt(prompt: str) -> OpenDomainRetrievalResult:
    """Retrieve a unique audited source for an open-domain prompt."""
    reasons = ambiguity_reasons(prompt)
    if reasons:
        return OpenDomainRetrievalResult(
            prompt=prompt,
            action=AmbiguityAction.ASK
            if any("unresolved" in reason for reason in reasons)
            else AmbiguityAction.ABSTAIN,
            document=None,
            source_loss=1.0,
            rebuttal_loss=1.0,
            reasons=reasons,
        )

    matches = matched_open_domain_documents(prompt)
    if len(matches) == 1:
        return OpenDomainRetrievalResult(
            prompt=prompt,
            action=AmbiguityAction.CERTIFY,
            document=matches[0],
            source_loss=0.0,
            rebuttal_loss=0.0,
            reasons=(),
        )
    if len(matches) > 1:
        return OpenDomainRetrievalResult(
            prompt=prompt,
            action=AmbiguityAction.ASK,
            document=None,
            source_loss=1.0,
            rebuttal_loss=1.0,
            reasons=("semantic ambiguity unresolved",),
        )
    return OpenDomainRetrievalResult(
        prompt=prompt,
        action=AmbiguityAction.ABSTAIN,
        document=None,
        source_loss=1.0,
        rebuttal_loss=1.0,
        reasons=("open-domain source unavailable",),
    )


def matched_open_domain_documents(prompt: str) -> tuple[OpenDomainDocument, ...]:
    """Return audited documents matching an unambiguous open-domain prompt."""
    atom = audited_atom(prompt)
    if atom is not None:
        value = prompt.rsplit("=", maxsplit=1)[-1].strip()
        return tuple(
            document
            for document in open_domain_corpus()
            if document.atom == atom and document.answer.casefold() == value.casefold()
        )

    text = normalized_text(prompt)
    return tuple(
        document
        for document in open_domain_corpus()
        if any(alias in text for alias in document.aliases)
    )


def open_domain_corpus() -> tuple[OpenDomainDocument, ...]:
    """Small audited corpus used by C4 open-domain retrieval probes."""
    return (
        OpenDomainDocument(
            doc_id="c4-capital-france",
            atom="capital(france)",
            answer="Paris",
            source="audit:geo:france:capital=Paris",
            aliases=("capital of france", "france capital", "capital(france)"),
        ),
        OpenDomainDocument(
            doc_id="c4-symbol-water",
            atom="formula(water)",
            answer="H2O",
            source="audit:chem:water:formula=H2O",
            aliases=(
                "formula for water",
                "chemical formula for water",
                "formula(water)",
            ),
        ),
        OpenDomainDocument(
            doc_id="c4-contradicted-sky",
            atom="color(clear_day_sky)",
            answer="blue",
            source="audit:toy:sky:color=blue",
            aliases=(
                "color is the daytime clear sky",
                "daytime clear sky",
                "color(clear_day_sky)",
            ),
            rebuttals=("audit:toy:sky:defeater=color=green",),
        ),
    )


__all__ = [
    "OpenDomainDocument",
    "OpenDomainRetrievalResult",
    "RecallChannel",
    "RecallFn",
    "RecallShiftProbe",
    "RecallShiftReport",
    "RecallShiftTrace",
    "RetrievalBundle",
    "calibrated_recall",
    "dirty_recall_shift_probe",
    "evaluate_recall_shift",
    "exact_suffix_recall",
    "matched_open_domain_documents",
    "open_domain_corpus",
    "recall_shift_curriculum",
    "retrieve",
    "retrieve_open_domain_prompt",
]
