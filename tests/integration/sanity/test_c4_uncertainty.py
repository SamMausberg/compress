"""C4 calibrated uncertainty gate regressions."""

from __future__ import annotations

import pytest

from vpm.evaluation import evaluate_c4
from vpm.language.ambiguity import AmbiguityAction
from vpm.retrieval import retrieve_open_domain_prompt
from vpm.tasks import stages
from vpm.tasks.c4 import C4DialogueTask, gate_dialogue, open_domain_dialogue_task

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

    assert report.tasks == 3
    assert report.uncertainty_ok == 2
    assert report.uncertainty_ok_rate == pytest.approx(2 / 3)
    assert report.mean_uncertainty == pytest.approx(1 / 6)
    assert payload["mean_uncertainty"] == pytest.approx(1 / 6)


def test_c4_open_domain_retrieval_feeds_dialogue_gate() -> None:
    retrieval = retrieve_open_domain_prompt("What is the chemical formula for water?")
    task = open_domain_dialogue_task("What is the chemical formula for water?")

    assert retrieval.action is AmbiguityAction.CERTIFY
    assert retrieval.retrieved is True
    assert retrieval.atom == "formula(water)"
    assert retrieval.source_loss == 0.0
    assert task is not None

    trace = gate_dialogue(task)
    assert trace.passed is True
    assert trace.rendered.startswith("H2O ")


def test_c4_open_domain_retrieval_keeps_ambiguous_prompts_uncertified() -> None:
    retrieval = retrieve_open_domain_prompt("Tell me about Jordan.")

    assert retrieval.action is AmbiguityAction.ASK
    assert retrieval.retrieved is False
    assert retrieval.sources == ()
    assert "semantic ambiguity unresolved" in retrieval.reasons


def test_c4_stage_metadata_marks_open_domain_retrieval_implemented() -> None:
    c4 = next(spec for spec in stages() if spec.name == "C4")

    assert "calibrated-uncertainty-gate" in c4.implemented_components
    assert "open-domain-retrieval" in c4.implemented_components
    assert "calibrated uncertainty" not in c4.blockers
    assert "open-domain retrieval" not in c4.blockers
    assert c4.blockers == ()
