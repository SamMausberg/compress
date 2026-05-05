"""Objective-facing release-readiness audit regressions."""

from __future__ import annotations

import pytest

from vpm.evaluation.release_audit import evaluate_release_readiness

pytestmark = pytest.mark.sanity


def test_release_readiness_reports_external_llm_baseline_blocker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VPM_LLM_BASELINE_JSON", raising=False)
    monkeypatch.delenv("VPM_HARD_LLM_BASELINE_JSON", raising=False)
    report = evaluate_release_readiness(limit=0)
    payload = report.to_dict()

    assert report.passed is False
    assert not any("stage blocker" in blocker for blocker in report.blockers)
    assert "same-budget external LLM baseline" in report.blockers
    assert "missing executed baseline family: llm" in report.blockers
    assert any(
        "missing executed hard-domain LLM baseline" in blocker for blocker in report.blockers
    )
    assert payload["passed"] is False
    assert any(
        criterion["criterion_id"] == "stages_m0_m6" and criterion["passed"]
        for criterion in payload["criteria"]
    )
    assert any(
        criterion["criterion_id"] == "matched_baselines" and not criterion["passed"]
        for criterion in payload["criteria"]
    )
    assert any(
        criterion["criterion_id"] == "hard_domain_llm_baseline" and not criterion["passed"]
        for criterion in payload["criteria"]
    )
