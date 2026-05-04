"""Controlled source/rebuttal recall-shift calibration regressions."""

from __future__ import annotations

import pytest

from vpm.retrieval.calibration import dirty_recall_shift_probe, evaluate_recall_shift

pytestmark = pytest.mark.sanity


def test_recall_shift_calibration_covers_source_and_rebuttal_variants() -> None:
    report = evaluate_recall_shift()
    assert report.passed is True
    assert report.source_epsilon == 0.0
    assert report.rebuttal_epsilon == 0.0
    assert report.shifted_epsilon == 0.0
    assert report.false_recall_rate == 0.0
    assert report.misses == ()
    assert report.false_recalls == ()
    assert any(
        trace.shifted and trace.channel == "source" and trace.recalled for trace in report.traces
    )
    assert any(
        trace.shifted and trace.channel == "rebuttal" and trace.recalled for trace in report.traces
    )


def test_dirty_recall_shift_probe_is_rejected() -> None:
    report = dirty_recall_shift_probe()
    assert report.passed is False
    assert report.shifted_epsilon == 1.0
    assert report.false_recall_rate == 0.0
    assert all(trace.shifted for trace in report.traces)
    assert all(trace.miss for trace in report.traces)
