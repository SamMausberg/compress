"""External LLM baseline export and scoring regressions."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from vpm.evaluation.baselines import BaselineStatus
from vpm.evaluation.llm_baseline import (
    score_llm_baseline_predictions,
    write_llm_baseline_tasks,
)
from vpm.tasks.c1 import schema_split

pytestmark = pytest.mark.sanity


def test_llm_baseline_export_omits_gold_operations(tmp_path) -> None:
    output = tmp_path / "tasks.jsonl"
    written = write_llm_baseline_tasks(output, limit=0)
    lines = [json.loads(line) for line in output.read_text().splitlines()]

    assert written == len(lines)
    assert written > 0
    assert all("expected_operation" not in line for line in lines)
    assert all("allowed_operations" in line for line in lines)
    assert all("prompt" in line for line in lines)


def test_llm_baseline_scoring_produces_valid_external_json(tmp_path) -> None:
    _train, heldout = schema_split(limit=0)
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
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

    report = score_llm_baseline_predictions(predictions, limit=0)
    assert report.status is BaselineStatus.EXECUTED
    assert report.solve_rate == 1.0
    assert report.operation_accuracy == 1.0
    assert report.compute_units == len(heldout)
    assert report.to_external_json()["compute_units"] == len(heldout)


def test_llm_baseline_scoring_rejects_missing_compute_units(tmp_path) -> None:
    _train, heldout = schema_split(limit=0)
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps({"task_id": heldout[0].task_id, "operation": heldout[0].operation}) + "\n"
    )

    report = score_llm_baseline_predictions(predictions, limit=0)
    assert report.status is BaselineStatus.INVALID
    assert any("compute_units is required" in error for error in report.errors)
    assert any("missing prediction" in error for error in report.errors)
    assert report.to_dict()["external_json"] is None
    with pytest.raises(ValueError, match="cannot export invalid"):
        report.to_external_json()


def test_cli_exports_and_scores_llm_baseline(tmp_path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    subprocess.run(  # noqa: S603
        [sys.executable, "-m", "vpm", "export-llm-baseline", str(tasks), "--limit", "0"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert tasks.exists()

    _train, heldout = schema_split(limit=0)
    predictions = tmp_path / "predictions.jsonl"
    scored = tmp_path / "llm-baseline.json"
    predictions.write_text(
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
    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "vpm",
            "score-llm-baseline",
            str(predictions),
            "--limit",
            "0",
            "--output",
            str(scored),
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "executed"
    assert json.loads(scored.read_text())["solve_rate"] == 1.0
