"""C3 deterministic tool-sandbox regressions."""

from __future__ import annotations

import pytest

from vpm.evaluation import evaluate_c3
from vpm.tasks import run_tool_sandbox_suite, stages
from vpm.tasks.c3.tools import ToolInvocation, run_tool_invocation

pytestmark = pytest.mark.sanity


def test_tool_sandbox_accepts_only_allowlisted_data_authority_calls() -> None:
    report = run_tool_sandbox_suite()

    assert report.passed is True
    assert report.invocations == 6
    assert report.accepted == 2
    assert report.rejected == 4
    assert report.compute_units == 2.0
    assert all(not trace.violation for trace in report.traces)


def test_tool_sandbox_rejects_unknown_tools_without_compute() -> None:
    trace = run_tool_invocation(
        ToolInvocation(
            task_id="dirty-shell",
            tool_id="shell.exec",
            args=("echo unsafe",),
            expected_output=None,
            expected_pass=False,
        )
    )

    assert trace.passed is False
    assert trace.compute_units == 0.0
    assert trace.output is None
    assert trace.errors == ("unknown tool: shell.exec",)


def test_evaluate_c3_reports_tool_sandbox_results() -> None:
    report = evaluate_c3()
    payload = report.to_dict()

    assert report.tool_sandbox.passed is True
    assert payload["tool_sandbox"]["accepted"] == 2
    assert payload["tool_sandbox"]["rejected"] == 4


def test_c3_stage_metadata_marks_tool_sandbox_implemented() -> None:
    c3 = next(spec for spec in stages() if spec.name == "C3")

    assert "tool-sandbox-runner" in c3.implemented_components
    assert "tool sandbox runner" not in c3.blockers
    assert c3.blockers == ()
