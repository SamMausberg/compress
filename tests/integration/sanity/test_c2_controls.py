"""Executable C2 active-test controls."""

from __future__ import annotations

import pytest

from vpm.evaluation import evaluate_c2
from vpm.infer import halt_decision, run_task_candidate, select_test
from vpm.infer.support_guard import guard_support
from vpm.infer.test_select import TestAction as ActiveTestAction
from vpm.tasks import active_curriculum, active_test

pytestmark = pytest.mark.sanity


def test_c2_active_tests_reduce_support_and_certify() -> None:
    task = active_curriculum(1)[0]
    assert task.operation not in task.observation.split()
    trace = active_test(task)
    assert trace.support_reduced is True
    assert len(trace.candidates_before) > len(trace.candidates_after) == 1

    result = run_task_candidate(task.to_c0_task(trace.selected_operation), trace.selected_operation)
    assert result.native_report["gate"]["passed"] is True

    metrics = evaluate_c2(active_curriculum(1))
    assert metrics.solve_rate == 1.0
    assert metrics.support_reduction_rate == 1.0
    assert metrics.support_guard_pass_rate == 1.0
    assert metrics.rehydrated == 0
    assert metrics.mean_test_score > 0.0
    assert metrics.halt_rate == 1.0
    assert metrics.mean_candidates_after == 1.0
    assert metrics.to_dict()["support_guards"][0]["passed"] is True
    assert metrics.to_dict()["test_selections"][0]["selected"]["name"] == "active-reveal-expected"
    assert metrics.to_dict()["halt_decisions"][0]["reason"] == "contract_met"
    assert metrics.to_dict()["verifier"]["evidence"]["source_coverage_rate"] == 1.0


def test_test_selection_and_halt_rule_choose_high_value_actions() -> None:
    low_value = ActiveTestAction("low", 1.0, 0.0, 0.0, 0.0)
    high_value = ActiveTestAction("high", 1.0, 1.0, 1.0, 1.0)
    selection = select_test((low_value, high_value))
    assert selection.selected.name == "high"
    assert selection.selected_score > 1.0

    keep_running = halt_decision(
        certificate=0.0,
        threshold=1.0,
        expected_utility_gain=selection.selected_score,
        compute_delta=0.1,
    )
    assert keep_running.should_halt is False
    assert keep_running.reason == "continue"

    stop = halt_decision(
        certificate=1.0,
        threshold=1.0,
        expected_utility_gain=0.0,
    )
    assert stop.should_halt is True
    assert stop.reason == "contract_met"


def test_support_guard_rehydrates_unsafe_pruning() -> None:
    report = guard_support(
        ("add", "mul", "eq"),
        ("add",),
        exact_rejection_witness=False,
        epsilon_max=0.2,
    )
    assert report.passed is False
    assert report.action == "rehydrate"
    assert report.rehydrated == ("mul", "eq")
    assert report.candidates_final == ("add", "mul", "eq")
    assert report.epsilon_prune > report.epsilon_max
