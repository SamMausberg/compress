"""Objective-facing release-readiness audit regressions."""

from __future__ import annotations

import json

import pytest

from vpm.evaluation.llm_baseline_c1 import score_llm_baseline_predictions
from vpm.evaluation.llm_baseline_hard import score_hard_llm_baseline_predictions
from vpm.evaluation.release_audit import evaluate_release_readiness
from vpm.tasks.c1 import schema_split
from vpm.tasks.hard_domains import hard_domain_curriculum

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


def test_release_readiness_accepts_configured_llm_baseline_artifacts(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c1 = tmp_path / "llm.json"
    hard = tmp_path / "hard-llm.json"
    c1_predictions = tmp_path / "llm-predictions.jsonl"
    hard_predictions = tmp_path / "hard-llm-predictions.jsonl"
    _train, heldout = schema_split(limit=0)
    c1_predictions.write_text(
        "".join(
            json.dumps(
                {
                    "task_id": task.task_id,
                    "operation": task.operation,
                    "compute_units": 1.0,
                },
                sort_keys=True,
            )
            + "\n"
            for task in heldout
        )
    )
    hard_predictions.write_text(
        "".join(
            json.dumps(
                {
                    "task_id": task.task_id,
                    "answer": task.expected,
                    "compute_units": 1.0,
                },
                sort_keys=True,
            )
            + "\n"
            for task in hard_domain_curriculum()
        )
    )
    c1.write_text(
        json.dumps(score_llm_baseline_predictions(c1_predictions, limit=0).to_external_json())
    )
    hard.write_text(
        json.dumps(score_hard_llm_baseline_predictions(hard_predictions).to_external_json())
    )
    monkeypatch.setenv("VPM_LLM_BASELINE_JSON", str(c1))
    monkeypatch.setenv("VPM_HARD_LLM_BASELINE_JSON", str(hard))

    report = evaluate_release_readiness(limit=0)

    assert report.passed is True
    assert report.blockers == ()
