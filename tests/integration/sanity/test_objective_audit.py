"""Prompt-to-artifact objective audit regressions."""

from __future__ import annotations

import pytest

from vpm.evaluation.objective_audit import evaluate_objective_completion

pytestmark = pytest.mark.sanity


def test_objective_audit_exposes_remaining_external_llm_blockers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VPM_LLM_BASELINE_JSON", raising=False)
    monkeypatch.delenv("VPM_HARD_LLM_BASELINE_JSON", raising=False)

    report = evaluate_objective_completion(limit=0)
    payload = report.to_dict()
    items = {item.requirement_id: item for item in report.checklist}

    assert report.passed is False
    assert payload["passed"] is False
    assert payload["objective"].startswith("finish compress")
    assert set(items) == {
        "m0_m6_architecture",
        "substrate_compiler_support_guard",
        "memory_compression_training",
        "calibrated_gates_adversarial_suites",
        "heldout_baselines_matched_budgets",
        "hard_domains",
        "clean_ci_release_quality",
    }
    assert items["m0_m6_architecture"].passed is True
    assert items["heldout_baselines_matched_budgets"].passed is False
    assert any("same-budget external LLM baseline" in blocker for blocker in report.blockers)
    assert any("missing executed baseline family: llm" in blocker for blocker in report.blockers)
    assert any(
        "missing executed hard-domain LLM baseline" in blocker for blocker in report.blockers
    )
    assert all(item.artifacts for item in report.checklist)
    assert all(item.verification_commands for item in report.checklist)
