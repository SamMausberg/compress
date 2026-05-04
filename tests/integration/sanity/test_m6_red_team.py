"""M6 executable red-team replay checks."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from vpm.evaluation.ablations import AblationName, evaluate_ablations
from vpm.evaluation.red_team import red_team_replay

pytestmark = pytest.mark.sanity


def test_ablation_table_records_expected_regressions() -> None:
    report = evaluate_ablations()
    assert report.passed is True
    assert {result.name for result in report.results} == set(AblationName)
    assert all(result.expected_regression for result in report.results)


def test_red_team_replay_combines_failures_and_ablations() -> None:
    report = red_team_replay()
    assert report.passed is True
    payload = report.to_dict()
    assert payload["failures"]["passed"] is True
    assert payload["ablations"]["passed"] is True
    assert payload["hard_domains"]["solve_rate"] == 1.0


def test_cli_runs_red_team_replay() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "vpm", "eval-red-team", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["passed"] is True
    assert payload["hard_domains"]["tasks"] == 4
