"""Executable C1 synthesis controls."""

from __future__ import annotations

import pytest

from vpm.tasks import (
    multi_step_synthesis_curriculum,
    prove_theorem_fragment,
    run_multistep_synthesis,
    stages,
    theorem_fragment_curriculum,
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
    assert c1.blockers == ()


def test_c1_theorem_fragments_prove_modus_ponens_chains() -> None:
    traces = tuple(prove_theorem_fragment(task) for task in theorem_fragment_curriculum())

    assert len(traces) == 2
    assert all(trace.passed for trace in traces)
    assert traces[0].proof_steps[0].rule == "modus_ponens"
    assert traces[0].derived == ("p", "q")
    assert [step.conclusion for step in traces[1].proof_steps] == ["wet", "slippery"]


def test_c1_stage_metadata_marks_theorem_fragments_implemented() -> None:
    c1 = next(spec for spec in stages() if spec.name == "C1")

    assert "theorem-proving-fragments" in c1.implemented_components
    assert "theorem-proving fragments" not in c1.blockers
    assert c1.blockers == ()
