"""Criterion-1 executable failure-mode regressions."""

from __future__ import annotations

import pytest

from vpm.evaluation.failure_modes import FailureMode, evaluate_failure_modes

pytestmark = pytest.mark.failure_modes


def test_executable_criterion1_failure_modes_are_unfired() -> None:
    report = evaluate_failure_modes()
    assert report.passed is True
    assert report.failures == ()
    assert {check.mode for check in report.checks} == set(FailureMode)
    assert report.uncovered_clauses


def test_failure_report_serializes_scope_and_uncovered_clauses() -> None:
    payload = evaluate_failure_modes().to_dict()
    assert payload["passed"] is True
    assert payload["failures"] == []
    assert "same-budget external LLM baseline" in payload["uncovered_clauses"]
    assert "hidden test-time compute accounting" not in payload["uncovered_clauses"]
    assert "source/rebuttal recall miss calibration under shift" not in payload["uncovered_clauses"]
    assert "dependence residualization calibration under shift" not in payload["uncovered_clauses"]
