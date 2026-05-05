"""Executable memory frontier and admission controls."""

from __future__ import annotations

import pytest

from vpm.memory import (
    ActiveSetBudget,
    ActiveSetItem,
    AdmissionEvidence,
    OnlineFrontierEstimator,
    admit_active,
    online_replay_frontier_report,
    replay_frontier_report,
    select_active_set,
)
from vpm.tasks import stages

pytestmark = pytest.mark.sanity


def test_exact_replay_frontier_uses_tight_sequence_bound() -> None:
    report = replay_frontier_report(3, 3, enumerative_utility=0.25)
    assert report.frontier_delta == 0.75
    assert report.bound.lcb == 0.75
    assert report.positive_lcb is True
    assert report.to_dict()["bound"]["radius"] == 0.0


def test_online_frontier_estimator_matches_exact_replay_report() -> None:
    estimator = OnlineFrontierEstimator("macro:add", enumerative_utility=0.25)
    for outcome in (True, True, True):
        estimator = estimator.observe(outcome)
    online = estimator.report()
    direct = online_replay_frontier_report(
        (True, True, True),
        macro_key="macro:add",
        enumerative_utility=0.25,
    )

    assert estimator.observations == 3
    assert estimator.certified == 3
    assert online.frontier_delta == 0.75
    assert online.bound.lcb == 0.75
    assert direct.to_dict() == online.to_dict()


def test_c5_stage_metadata_marks_online_frontier_implemented() -> None:
    c5 = next(spec for spec in stages() if spec.name == "C5")

    assert "online-frontier-estimator" in c5.implemented_components
    assert "online frontier estimator" not in c5.blockers
    assert c5.blockers == ("cross-stage replay scheduler",)


def test_active_admission_requires_frontier_replay_and_equivalence() -> None:
    accepted = admit_active(
        AdmissionEvidence(
            frontier_lcb=0.5,
            cert_act=True,
            cert_eq=1.0,
            no_capability_escalation=True,
            replay_pass=True,
        )
    )
    assert accepted.admitted is True
    assert accepted.reasons == ()

    rejected = admit_active(
        AdmissionEvidence(
            frontier_lcb=-0.1,
            cert_act=False,
            cert_eq=0.0,
            no_capability_escalation=False,
            replay_pass=False,
        )
    )
    assert rejected.admitted is False
    assert "frontier lcb not positive" in rejected.reasons
    assert "replay gate failed" in rejected.reasons


def test_active_set_selection_respects_hard_budgets() -> None:
    selection = select_active_set(
        (
            ActiveSetItem("low", utility_lcb=0.1, memory_cost=1.0),
            ActiveSetItem("high", utility_lcb=1.0, memory_cost=1.0),
            ActiveSetItem("too-large", utility_lcb=2.0, memory_cost=3.0),
            ActiveSetItem("negative", utility_lcb=-1.0, memory_cost=1.0),
        ),
        ActiveSetBudget(latency=3.0, memory=2.0, capability_risk=0.0, information=0.0),
    )
    assert [item.key for item in selection.selected] == ["high", "low"]
    assert selection.memory_used == 2.0
