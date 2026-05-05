"""Open-domain context and semantic ambiguity regressions."""

from __future__ import annotations

import pytest

from vpm.evaluation.open_domain import (
    dirty_open_domain_collapse_probe,
    evaluate_open_domain_ambiguity,
)
from vpm.language.ambiguity import AmbiguityAction, guard_open_domain_prompt
from vpm.retrieval import retrieve_open_domain_prompt

pytestmark = pytest.mark.sanity


def test_open_domain_ambiguity_narrows_or_abstains() -> None:
    report = evaluate_open_domain_ambiguity()
    assert report.passed is True
    assert report.collapses == ()
    assert report.failures == ()

    ambiguous = guard_open_domain_prompt("Tell me about Jordan.")
    assert ambiguous.action is AmbiguityAction.ASK
    assert ambiguous.certified is False
    assert "semantic ambiguity unresolved" in ambiguous.reasons

    audited = guard_open_domain_prompt("audited:capital(france)=Paris")
    assert audited.action is AmbiguityAction.CERTIFY
    assert audited.certified_atoms == ("capital(france)",)

    retrieved = retrieve_open_domain_prompt("audited:capital(france)=Paris")
    assert retrieved.action is AmbiguityAction.CERTIFY
    assert retrieved.retrieved is True
    assert retrieved.sources == ("audit:geo:france:capital=Paris",)


def test_dirty_open_domain_collapse_probe_is_rejected() -> None:
    report = dirty_open_domain_collapse_probe()
    assert report.passed is False
    assert len(report.collapses) == 4
    assert all(trace.decision.certified for trace in report.collapses)
