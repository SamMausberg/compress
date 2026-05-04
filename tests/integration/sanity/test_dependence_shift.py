"""Dependence residualization under shift regressions."""

from __future__ import annotations

import pytest

from vpm.verifiers.dependence import dirty_dependence_shift_probe, evaluate_dependence_shift

pytestmark = pytest.mark.sanity


def test_dependence_shift_blocks_uncalibrated_correlated_passes() -> None:
    report = evaluate_dependence_shift()
    assert report.passed is True
    assert report.epsilon_dep == 0.0
    assert report.shifted_epsilon == 0.0
    assert report.leaks == ()

    unresidualized = next(trace for trace in report.traces if trace.block_key[0] == "claim-dup")
    assert unresidualized.blocked_signal_ids == ("dup-unresid-a",)

    clean_residual = next(
        trace for trace in report.traces if trace.block_key[0] == "claim-clean-resid"
    )
    assert clean_residual.residualized is True
    assert clean_residual.blocked_signal_ids == ()

    shifted_residual = next(
        trace for trace in report.traces if trace.block_key[0] == "claim-shifted-resid"
    )
    assert shifted_residual.residualized is False
    assert shifted_residual.shifted_opportunities == 1
    assert shifted_residual.blocked_signal_ids == ("resid-shifted-a",)


def test_dirty_dependence_shift_probe_is_rejected() -> None:
    report = dirty_dependence_shift_probe()
    assert report.passed is False
    assert report.epsilon_dep == 1.0
    assert report.shifted_epsilon == 1.0
    assert len(report.leaks) == 1
    assert report.leaks[0].leaked_independence is True
