"""Matched compute-accounting checks."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from vpm.evaluation.compute_accounting import evaluate_compute_accounting, hidden_compute_probe

pytestmark = pytest.mark.sanity


def test_compute_accounting_declares_all_shipped_units() -> None:
    report = evaluate_compute_accounting()
    assert report.passed is True
    assert report.total_units == report.budget
    assert report.hidden_units == 0.0


def test_hidden_compute_probe_is_rejected() -> None:
    report = hidden_compute_probe()
    assert report.passed is False
    assert report.hidden_units > 0.0


def test_cli_runs_compute_accounting() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "vpm", "eval-compute", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["passed"] is True
    assert payload["hidden_units"] == 0.0
