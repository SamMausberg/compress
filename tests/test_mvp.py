"""MVP vertical-slice tests."""

from __future__ import annotations

import json
import subprocess
import sys

from vpm import _native
from vpm.evaluation import (
    evaluate_c0,
    evaluate_c1,
    evaluate_c2,
    evaluate_c3,
    evaluate_c4,
    evaluate_c5,
)
from vpm.infer import run_c0_add, run_task, run_task_candidate
from vpm.memory import MemoryLibrary
from vpm.tasks import (
    active_curriculum,
    active_test,
    arithmetic_task,
    as_c0_tasks,
    concat_task,
    dialogue_curriculum,
    equality_task,
    gate_dialogue,
    hidden_schema_curriculum,
    macro_replay_curriculum,
    multiplication_task,
    policy_probe_curriculum,
    schema_split,
    stages,
)
from vpm.training import allocate_budget, matched_baselines
from vpm.verifiers import native_value_json


def test_native_report_contains_ledger_trace_and_gate() -> None:
    report = json.loads(_native.run_c0_add_json(2, 3, 5))
    assert report["value"] == {"type": "Int", "value": 5}
    assert report["verification"]["passed"] is True
    assert report["gate"]["passed"] is True
    assert report["ledger"]["entries"] >= 6
    assert report["trace_nodes"] >= 3


def test_native_typed_report_contains_string_and_bool_values() -> None:
    concat = json.loads(
        _native.run_c0_typed_json(
            "concat",
            native_value_json("ab"),
            native_value_json("cd"),
            native_value_json("abcd"),
        )
    )
    equality = json.loads(
        _native.run_c0_typed_json(
            "eq",
            native_value_json(5),
            native_value_json(5),
            native_value_json(True),
        )
    )
    assert concat["value"] == {"type": "Text", "value": "abcd"}
    assert concat["gate"]["passed"] is True
    assert equality["value"] == {"type": "Bool", "value": True}
    assert equality["gate"]["passed"] is True


def test_wrong_expected_value_fails_gate() -> None:
    result = run_c0_add(2, 3, expected=6)
    assert result.route == "solve"
    assert result.rendered == "refusal"
    assert result.native_report["gate"]["passed"] is False


def test_multiplication_flows_through_native_vertical_slice() -> None:
    result = run_task(multiplication_task(6, 7))
    assert result.route == "solve"
    assert result.rendered.startswith("42 ")
    assert result.native_report["verification"]["passed"] is True


def test_string_and_equality_tasks_flow_through_vertical_slice() -> None:
    concat = run_task(concat_task("ab", "cd"))
    equality = run_task(equality_task("left", "right"))
    assert concat.route == "solve"
    assert concat.rendered.startswith("abcd ")
    assert concat.native_report["verification"]["passed"] is True
    assert equality.route == "solve"
    assert equality.rendered.startswith("False ")
    assert equality.native_report["verification"]["passed"] is True


def test_wrong_candidate_is_refused_and_not_admitted_to_memory() -> None:
    memory = MemoryLibrary()
    result = run_task_candidate(arithmetic_task("mul", 3, 4), "add", memory)
    assert result.rendered == "refusal"
    assert result.native_report["verification"]["passed"] is False
    assert result.native_report["gate"]["passed"] is False
    assert result.memory_active == 0
    assert memory.active == {}


def test_authority_policy_rejects_certified_value_without_memory_write() -> None:
    memory = MemoryLibrary()
    result = run_task(arithmetic_task("add", 2, 3), memory, labels=("capability",))
    assert result.rendered == "refusal"
    assert result.native_report["verification"]["passed"] is True
    assert result.native_report["gate"]["passed"] is False
    assert result.native_report["gate"]["authority"]["auth_ok"] is False
    assert result.memory_active == 0
    assert memory.active == {}


def test_componentwise_risk_policy_rejects_certified_value() -> None:
    result = run_task(arithmetic_task("add", 2, 3), risk={"privacy": 0.1})
    assert result.rendered == "refusal"
    assert result.native_report["verification"]["passed"] is True
    assert result.native_report["gate"]["passed"] is False
    assert result.native_report["gate"]["authority"]["risk_ok"] is False
    assert result.native_report["ledger"]["total_risk"]["privacy"] == 0.1


def test_evaluation_and_budget_are_connected() -> None:
    metrics = evaluate_c0()
    budget = allocate_budget(metrics)
    assert metrics.solve_rate == 1.0
    assert metrics.evidence.source_coverage_rate == 1.0
    assert metrics.evidence.rebuttal_clear_rate == 1.0
    assert metrics.evidence.realization_ok_rate == 1.0
    assert budget.verification == 0.4


def test_c1_hidden_schema_tasks_bridge_to_verifier_and_baselines() -> None:
    train, heldout = schema_split(2)
    assert train
    assert heldout
    for task in heldout:
        assert task.schema_id not in task.observation
        assert task.operation not in task.observation.split()

    exact = run_task_candidate(heldout[0].to_c0_task(), heldout[0].operation)
    assert exact.rendered != "refusal"
    assert exact.native_report["gate"]["passed"] is True
    hidden = heldout[0].to_hidden_c0_task()
    assert hidden.operation == "unknown"
    assert hidden.left == heldout[0].left

    baselines = {
        baseline.name: baseline
        for baseline in matched_baselines(as_c0_tasks(train), as_c0_tasks(heldout))
    }
    assert baselines["enumerative-full"].solve_rate == 1.0
    assert baselines["enumerative-full"].mean_candidates > 1.0


def test_c1_evaluation_runs_executable_hidden_schema_subset() -> None:
    metrics = evaluate_c1(hidden_schema_curriculum(1))
    assert metrics.solve_rate == 1.0
    assert metrics.gate_violations == 0
    assert metrics.evidence.mean_source_loss == 0.0
    assert metrics.to_dict()["evidence"]["realization_ok_rate"] == 1.0


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
    assert metrics.mean_candidates_after == 1.0
    assert metrics.to_dict()["verifier"]["evidence"]["source_coverage_rate"] == 1.0


def test_c3_policy_probes_reject_adversarial_gate_requests() -> None:
    metrics = evaluate_c3(policy_probe_curriculum())
    assert metrics.violations == 0
    assert metrics.violation_rate == 0.0
    assert metrics.rejected >= 3
    assert metrics.controls_passed == 1
    rejected = [trace for trace in metrics.traces if not trace.expected_pass]
    assert all(trace.verification_passed for trace in rejected)
    assert all(trace.memory_active == 0 for trace in rejected)
    assert any(not trace.auth_ok for trace in rejected)
    assert any(not trace.risk_ok for trace in rejected)


def test_c4_dialogue_gates_require_source_rebuttal_and_realization() -> None:
    tasks = dialogue_curriculum()
    passing = gate_dialogue(tasks[0])
    rejected = gate_dialogue(tasks[-1])
    assert passing.rendered.startswith("Paris ")
    assert passing.passed is True
    assert rejected.rendered == "refusal"
    assert "material rebuttal present" in rejected.reasons

    metrics = evaluate_c4(tasks)
    assert metrics.rendered == 2
    assert metrics.refused == 1
    assert metrics.violations == 0
    assert metrics.source_coverage_rate == 1.0
    assert metrics.realization_ok_rate == 1.0


def test_c5_macro_replay_admits_only_safe_compressive_macros() -> None:
    metrics = evaluate_c5(macro_replay_curriculum())
    assert metrics.admitted == 3
    assert metrics.demoted == 1
    assert metrics.violations == 0
    assert metrics.frontier_movement_rate == 0.75
    assert all(trace.sublinear_active_memory for trace in metrics.traces if trace.admitted)

    rejected = [trace for trace in metrics.traces if not trace.expected_admitted]
    assert rejected[0].admitted is False
    assert rejected[0].certified == 0
    assert "replay gate failed" in rejected[0].reasons


def test_all_curriculum_modules_have_runtime_metadata() -> None:
    specs = stages()
    assert [spec.name for spec in specs] == ["C0", "C1", "C2", "C3", "C4", "C5"]
    assert specs[0].executable is True
    assert specs[1].executable is True
    assert specs[2].executable is True
    assert specs[3].executable is True
    assert specs[4].executable is True
    assert specs[5].executable is True
    assert all(spec.implemented_components for spec in specs)


def test_cli_runs_vertical_slice() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "vpm", "run-c0-add", "2", "3"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.startswith("5 ")


def test_cli_doctor_reports_runtime() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "vpm", "doctor", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["native_extension_ok"] is True
    assert payload["torch"].startswith("2.")


def test_cli_runs_generic_c0_task() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "vpm", "run-c0", "mul", "6", "7"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.startswith("42 ")


def test_cli_runs_generic_text_task() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "vpm", "run-c0", "concat", "ab", "cd"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.startswith("abcd ")


def test_cli_runs_c1_evaluation() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "vpm", "eval-c1", "--limit", "1", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["solve_rate"] == 1.0
    assert payload["tasks"] > 0
    assert payload["evidence"]["source_coverage_rate"] == 1.0


def test_cli_runs_c2_evaluation() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "vpm", "eval-c2", "--limit", "1", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["solve_rate"] == 1.0
    assert payload["support_reduction_rate"] == 1.0
    assert payload["mean_candidates_after"] == 1.0


def test_cli_runs_c3_policy_evaluation() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "vpm", "eval-c3", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["violation_rate"] == 0.0
    assert payload["rejected"] >= 3


def test_cli_runs_c4_dialogue_evaluation() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "vpm", "eval-c4", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["rendered"] == 2
    assert payload["refused"] == 1
    assert payload["violation_rate"] == 0.0


def test_cli_runs_c5_macro_replay_evaluation() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "vpm", "eval-c5", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["admitted"] == 3
    assert payload["demoted"] == 1
    assert payload["violation_rate"] == 0.0


def test_cli_exposes_authority_rejection() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vpm",
            "run-c0",
            "add",
            "2",
            "3",
            "--authority",
            "capability",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["rendered"] == "refusal"
    assert payload["native_report"]["gate"]["authority"]["auth_ok"] is False
