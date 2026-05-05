"""C4 calibrated uncertainty gate regressions."""

from __future__ import annotations

import pytest

from vpm.evaluation import evaluate_c4
from vpm.tasks import stages
from vpm.tasks.c4 import C4DialogueTask, gate_dialogue

pytestmark = pytest.mark.sanity


def test_c4_uncertainty_blocks_missing_witness_rendering() -> None:
    trace = gate_dialogue(
        C4DialogueTask(
            task_id="c4-missing-source",
            question="What is the capital of France?",
            atom="capital(france)",
            answer="Paris",
            sources=(),
        )
    )

    assert trace.rendered == "refusal"
    assert trace.uncertainty == 0.5
    assert trace.uncertainty_ok is False
    assert "uncertainty above calibrated threshold" in trace.reasons


def test_c4_evaluation_reports_uncertainty_calibration() -> None:
    report = evaluate_c4()
    payload = report.to_dict()

    assert report.uncertainty_ok == 2
    assert report.uncertainty_ok_rate == pytest.approx(2 / 3)
    assert report.mean_uncertainty == pytest.approx(1 / 6)
    assert payload["mean_uncertainty"] == pytest.approx(1 / 6)


def test_c4_stage_metadata_marks_uncertainty_gate_implemented() -> None:
    c4 = next(spec for spec in stages() if spec.name == "C4")

    assert "calibrated-uncertainty-gate" in c4.implemented_components
    assert "calibrated uncertainty" not in c4.blockers
    assert c4.blockers == ("open-domain retrieval",)
