"""Matched baseline audit checks."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from vpm.evaluation.baselines import (
    BaselineFamily,
    BaselineStatus,
    evaluate_baseline_suite,
    external_baseline,
)

pytestmark = pytest.mark.sanity


def test_baseline_suite_runs_local_neural_baselines_and_marks_llm_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VPM_LLM_BASELINE_JSON", raising=False)
    suite = evaluate_baseline_suite(limit=1)
    families = {(baseline.family, baseline.status) for baseline in suite.baselines}
    assert (BaselineFamily.VPM, BaselineStatus.EXECUTED) in families
    assert (BaselineFamily.PROGRAM_SYNTHESIS, BaselineStatus.EXECUTED) in families
    assert (BaselineFamily.TRANSFORMER, BaselineStatus.EXECUTED) in families
    assert (BaselineFamily.SSM, BaselineStatus.EXECUTED) in families
    assert (BaselineFamily.LLM, BaselineStatus.NOT_CONFIGURED) in families
    assert suite.ready_for_claims is False
    assert suite.missing_families == ("llm",)


def test_external_llm_baseline_must_fit_matched_budget(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "llm.json"
    report.write_text(
        json.dumps(
            {
                "name": "external-llm-c1",
                "solve_rate": 0.5,
                "compute_units": 1.0,
            }
        )
    )
    monkeypatch.setenv("VPM_LLM_BASELINE_JSON", str(report))

    baseline = external_baseline(
        BaselineFamily.LLM,
        "VPM_LLM_BASELINE_JSON",
        max_compute_units=1.0,
    )
    assert baseline.status is BaselineStatus.EXECUTED
    assert baseline.max_compute_units == 1.0
    assert baseline.reason == ""


def test_over_budget_external_llm_baseline_is_invalid(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "llm.json"
    report.write_text(json.dumps({"solve_rate": 0.5, "compute_units": 2.0}))
    monkeypatch.setenv("VPM_LLM_BASELINE_JSON", str(report))

    baseline = external_baseline(
        BaselineFamily.LLM,
        "VPM_LLM_BASELINE_JSON",
        max_compute_units=1.0,
    )
    assert baseline.status is BaselineStatus.INVALID
    assert "exceeds matched budget" in baseline.reason


def test_cli_runs_baseline_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VPM_LLM_BASELINE_JSON", raising=False)
    completed = subprocess.run(
        [sys.executable, "-m", "vpm", "eval-baselines", "--limit", "0", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["ready_for_claims"] is False
    assert payload["missing_families"] == ["llm"]
