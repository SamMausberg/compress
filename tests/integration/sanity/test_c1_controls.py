"""Executable C1 synthesis controls."""

from __future__ import annotations

import pytest

from vpm.tasks import (
    multi_step_synthesis_curriculum,
    run_multistep_synthesis,
    stages,
)

pytestmark = pytest.mark.sanity


def test_c1_multistep_synthesis_selects_verified_programs() -> None:
    traces = tuple(run_multistep_synthesis(task) for task in multi_step_synthesis_curriculum())

    assert len(traces) == 2
    assert all(trace.passed for trace in traces)
    assert all(trace.multi_step for trace in traces)
    assert traces[0].output == 20
    assert traces[0].candidates_tested == 2
    assert [step.operation for step in traces[1].selected_steps] == ["concat", "eq"]


def test_c1_stage_metadata_marks_multistep_synthesis_implemented() -> None:
    c1 = next(spec for spec in stages() if spec.name == "C1")

    assert "multi-step-synthesis" in c1.implemented_components
    assert "multi-step synthesis" not in c1.blockers
    assert c1.blockers == ("theorem-proving fragments",)
