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
)

pytestmark = pytest.mark.sanity


def test_baseline_suite_runs_executable_and_marks_external_missing() -> None:
    suite = evaluate_baseline_suite(limit=1)
    families = {(baseline.family, baseline.status) for baseline in suite.baselines}
    assert (BaselineFamily.VPM, BaselineStatus.EXECUTED) in families
    assert (BaselineFamily.PROGRAM_SYNTHESIS, BaselineStatus.EXECUTED) in families
    assert (BaselineFamily.TRANSFORMER, BaselineStatus.NOT_CONFIGURED) in families
    assert (BaselineFamily.SSM, BaselineStatus.NOT_CONFIGURED) in families
    assert (BaselineFamily.LLM, BaselineStatus.NOT_CONFIGURED) in families
    assert suite.ready_for_claims is False
    assert suite.missing_families == ("transformer", "ssm", "llm")


def test_cli_runs_baseline_audit() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "vpm", "eval-baselines", "--limit", "1", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["ready_for_claims"] is False
    assert "llm" in payload["missing_families"]
