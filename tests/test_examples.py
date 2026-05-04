"""Runnable example coverage."""

from __future__ import annotations

import json
import subprocess
import sys


def test_vpm0_run_example_emits_diagnostics_and_metrics() -> None:
    completed = subprocess.run(
        [sys.executable, "examples/vpm0/run.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["doctor"]["native_extension_ok"] is True
    assert payload["single"]["rendered"].startswith("5 ")
    assert payload["metrics"]["solve_rate"] == 1.0
    assert payload["metrics"]["evidence"]["source_coverage_rate"] == 1.0
