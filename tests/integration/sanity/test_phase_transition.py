"""Compression phase-transition diagnostics."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from vpm.evaluation.phase_transition import evaluate_phase_transition
from vpm.evaluation.saturation import evaluate_saturation

pytestmark = pytest.mark.sanity


def test_c5_phase_transition_and_saturation_diagnostics() -> None:
    phase = evaluate_phase_transition()
    saturation = evaluate_saturation(phase.compression)
    assert phase.observed is True
    assert saturation.saturated is False
    assert saturation.positive_macros == phase.compression.admitted


def test_cli_runs_phase_transition_diagnostics() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "vpm", "eval-phase", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["phase_transition"]["observed"] is True
    assert payload["saturation"]["saturated"] is False
