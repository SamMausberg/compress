"""MVP vertical-slice tests."""

from __future__ import annotations

import json
import subprocess
import sys

from vpm import _native
from vpm.evaluation import evaluate_c0
from vpm.infer import run_c0_add, run_task, run_task_candidate
from vpm.memory import MemoryLibrary
from vpm.tasks import arithmetic_task, concat_task, equality_task, multiplication_task, stages
from vpm.training import allocate_budget
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


def test_evaluation_and_budget_are_connected() -> None:
    metrics = evaluate_c0()
    budget = allocate_budget(metrics)
    assert metrics.solve_rate == 1.0
    assert budget.verification == 0.4


def test_all_curriculum_modules_have_runtime_metadata() -> None:
    specs = stages()
    assert [spec.name for spec in specs] == ["C0", "C1", "C2", "C3", "C4", "C5"]
    assert specs[0].executable is True
    assert all(spec.implemented_components for spec in specs)


def test_cli_runs_vertical_slice() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "vpm", "run-c0-add", "2", "3"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.startswith("5 ")


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
