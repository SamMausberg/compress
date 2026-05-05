"""C3 rollback-credit ledger regressions."""

from __future__ import annotations

import pytest

from vpm.evaluation import evaluate_c3
from vpm.tasks import run_rollback_ledger, stages
from vpm.tasks.c3.rollback import (
    RiskChannel,
    RollbackAction,
    RollbackPlan,
    rollback_entry,
)

pytestmark = pytest.mark.sanity


def test_rollback_ledger_applies_only_certified_in_cap_credits() -> None:
    report = run_rollback_ledger()

    assert report.passed is True
    assert report.actions == 5
    assert report.accepted == 2
    assert report.rejected == 3
    assert report.cumulative_risk == {
        "conflict": pytest.approx(0.2),
        "impact": pytest.approx(0.15),
    }
    assert all(not entry.violation for entry in report.entries)


def test_rollback_rejects_absorbing_privacy_without_reversal_witness() -> None:
    entry = rollback_entry(
        RollbackAction(
            action_id="privacy-without-witness",
            risk_delta={RiskChannel.PRIVACY.value: 0.1},
            credit_cap={RiskChannel.PRIVACY.value: 0.1},
            rollback=RollbackPlan(
                credits={RiskChannel.PRIVACY.value: 0.1},
                monitor_certificates={RiskChannel.PRIVACY.value: 1.0},
            ),
            expected_pass=False,
        ),
        {},
    )

    assert entry.passed is False
    assert entry.applied_credit == {}
    assert entry.errors == ("privacy: absorbing channel lacks reversal witness",)


def test_evaluate_c3_reports_rollback_ledger_results() -> None:
    report = evaluate_c3()
    payload = report.to_dict()

    assert report.rollback_ledger.passed is True
    assert payload["rollback_ledger"]["accepted"] == 2
    assert payload["rollback_ledger"]["rejected"] == 3


def test_c3_stage_metadata_marks_rollback_ledger_implemented() -> None:
    c3 = next(spec for spec in stages() if spec.name == "C3")

    assert "rollback-credit-ledger" in c3.implemented_components
    assert c3.blockers == ()
