"""MVP vertical-slice tests."""

from __future__ import annotations

import json
import subprocess
import sys

from vpm import _native
from vpm.evaluation import evaluate_c0
from vpm.infer import run_c0_add
from vpm.tasks import stages
from vpm.training import allocate_budget


def test_native_report_contains_ledger_trace_and_gate() -> None:
    report = json.loads(_native.run_c0_add_json(2, 3, 5))
    assert report["value"] == {"type": "Int", "value": 5}
    assert report["verification"]["passed"] is True
    assert report["gate"]["passed"] is True
    assert report["ledger"]["entries"] >= 6
    assert report["trace_nodes"] >= 3


def test_wrong_expected_value_fails_gate() -> None:
    result = run_c0_add(2, 3, expected=6)
    assert result.route == "solve"
    assert result.rendered == "refusal"
    assert result.native_report["gate"]["passed"] is False


def test_evaluation_and_budget_are_connected() -> None:
    metrics = evaluate_c0()
    budget = allocate_budget(metrics)
    assert metrics.solve_rate == 1.0
    assert budget.verification == 0.4


def test_all_curriculum_modules_have_runtime_metadata() -> None:
    specs = stages()
    assert [spec.name for spec in specs] == ["C0", "C1", "C2", "C3", "C4", "C5"]
    assert specs[0].executable is True
    assert all(spec.implemented_components for spec in specs)


def test_cli_runs_vertical_slice() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "vpm", "run-c0-add", "2", "3"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.startswith("5 ")
