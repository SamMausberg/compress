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


def external_llm_payload(
    *,
    task_kind: str = "c1",
    tasks: int = 1,
    solve_rate: float = 1.0,
    compute_units: float = 1.0,
) -> dict[str, object]:
    """Return a scorer-shaped external LLM artifact for baseline-audit tests."""
    solved = round(tasks * solve_rate)
    if task_kind == "c1":
        traces = [
            {
                "task_id": f"c1-{index}",
                "expected_operation": "add",
                "predicted_operation": "add" if index < solved else "mul",
                "certified": index < solved,
                "operation_correct": index < solved,
                "compute_units": compute_units / tasks,
                "model": "external-test-model",
                "errors": [],
            }
            for index in range(tasks)
        ]
        extra: dict[str, object] = {"operation_accuracy": solve_rate}
    else:
        traces = [
            {
                "task_id": f"hard-{index}",
                "domain": "formal",
                "expected_answer": "42",
                "predicted_answer": "42" if index < solved else "wrong",
                "correct": index < solved,
                "compute_units": compute_units / tasks,
                "model": "external-test-model",
                "errors": [],
            }
            for index in range(tasks)
        ]
        extra = {}
    return {
        "artifact_kind": "vpm-external-llm-baseline-v1",
        "name": f"external-llm-{task_kind}",
        "task_kind": task_kind,
        "status": "executed",
        "tasks": tasks,
        "solve_rate": solve_rate,
        "mean_candidates": 1.0,
        "compute_units": compute_units,
        "max_compute_units": float(tasks),
        "traces": traces,
        **extra,
    }


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
    report.write_text(json.dumps(external_llm_payload()))
    monkeypatch.setenv("VPM_LLM_BASELINE_JSON", str(report))

    baseline = external_baseline(
        BaselineFamily.LLM,
        "VPM_LLM_BASELINE_JSON",
        max_compute_units=1.0,
        task_kind="c1",
    )
    assert baseline.status is BaselineStatus.EXECUTED
    assert baseline.max_compute_units == 1.0
    assert baseline.reason == ""


def test_compact_external_llm_summary_is_invalid(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "llm.json"
    report.write_text(
        json.dumps({"name": "external-llm-c1", "solve_rate": 1.0, "compute_units": 1.0})
    )
    monkeypatch.setenv("VPM_LLM_BASELINE_JSON", str(report))

    baseline = external_baseline(
        BaselineFamily.LLM,
        "VPM_LLM_BASELINE_JSON",
        max_compute_units=1.0,
        task_kind="c1",
    )
    assert baseline.status is BaselineStatus.INVALID
    assert "artifact_kind must be vpm-external-llm-baseline-v1" in baseline.reason


def test_traced_external_llm_artifact_requires_model(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = external_llm_payload()
    traces = payload["traces"]
    assert isinstance(traces, list)
    trace = traces[0]
    assert isinstance(trace, dict)
    del trace["model"]
    report = tmp_path / "llm.json"
    report.write_text(json.dumps(payload))
    monkeypatch.setenv("VPM_LLM_BASELINE_JSON", str(report))

    baseline = external_baseline(
        BaselineFamily.LLM,
        "VPM_LLM_BASELINE_JSON",
        max_compute_units=1.0,
        task_kind="c1",
    )
    assert baseline.status is BaselineStatus.INVALID
    assert "model must be a string" in baseline.reason


def test_over_budget_external_llm_baseline_is_invalid(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "llm.json"
    report.write_text(json.dumps(external_llm_payload(compute_units=2.0)))
    monkeypatch.setenv("VPM_LLM_BASELINE_JSON", str(report))

    baseline = external_baseline(
        BaselineFamily.LLM,
        "VPM_LLM_BASELINE_JSON",
        max_compute_units=1.0,
        task_kind="c1",
    )
    assert baseline.status is BaselineStatus.INVALID
    assert "exceeds matched budget" in baseline.reason


def test_zero_compute_external_llm_baseline_is_invalid(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "llm.json"
    report.write_text(json.dumps(external_llm_payload(solve_rate=0.0, compute_units=0.0)))
    monkeypatch.setenv("VPM_LLM_BASELINE_JSON", str(report))

    baseline = external_baseline(
        BaselineFamily.LLM,
        "VPM_LLM_BASELINE_JSON",
        max_compute_units=1.0,
        task_kind="c1",
    )
    assert baseline.status is BaselineStatus.INVALID
    assert "compute_units must be positive" in baseline.reason


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
