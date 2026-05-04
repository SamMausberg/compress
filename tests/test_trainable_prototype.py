"""Trainable non-transformer prototype tests."""

from __future__ import annotations

from vpm.tasks import C0Task, typed_hidden_task
from vpm.training import (
    TrainingConfig,
    curriculum_split,
    evaluate_saved_c1_prototype,
    evaluate_saved_prototype,
    run_learned_task,
    train_c0_prototype,
    train_c1_prototype,
)
from vpm.verifiers import gate_passed


def test_trainable_prototype_learns_and_stays_verifier_gated(tmp_path) -> None:
    artifact = tmp_path / "prototype.npz"
    model, report = train_c0_prototype(
        TrainingConfig(limit=2, epochs=12, hidden_dim=12, device="cpu", artifact=artifact)
    )

    baselines = {baseline.name: baseline for baseline in report.heldout.baselines}
    assert report.final_accuracy > report.initial_accuracy
    assert report.heldout.solve_rate > baselines["majority"].solve_rate
    assert report.heldout.operation_accuracy > baselines["majority"].operation_accuracy
    assert report.heldout.macro_memory_active >= 2
    assert report.heldout.compression_ratio > 1.0
    enumerative = baselines["enumerative-full"]
    enumerative_utility = enumerative.solve_rate / enumerative.mean_candidates
    assert report.heldout.compression.learned_mean_candidates == 1.0
    assert report.heldout.compression.certified_utility > enumerative_utility
    assert report.heldout.compression.frontier_delta_vs_enumerative > 0.0
    assert report.heldout.compression.sublinear_active_memory is True
    assert (
        report.heldout.to_dict()["compression"]["frontier_delta_vs_enumerative"]
        == report.heldout.compression.frontier_delta_vs_enumerative
    )
    assert any(trace.passed and trace.memory_key for trace in report.heldout.traces)
    assert any(not trace.passed for trace in report.heldout.traces)

    loaded_report = evaluate_saved_prototype(artifact, limit=2, device="cpu")
    assert loaded_report.solve_rate == report.heldout.solve_rate
    assert len(loaded_report.traces) == loaded_report.tasks

    _, heldout = curriculum_split(2)
    task, inference = first_certified_exact(model, heldout)
    assert inference.proposal.operation == task.operation
    assert gate_passed(inference.result.native_report)
    assert inference.result.rendered.startswith(f"{task.expected} ")

    hidden = typed_hidden_task(str(task.left), str(task.right), str(task.expected))
    hidden_inference = run_learned_task(model, hidden)
    assert hidden_inference.proposal.operation == task.operation
    assert gate_passed(hidden_inference.result.native_report)

    authority_rejected = run_learned_task(model, task, labels=("capability",))
    assert authority_rejected.proposal.operation == task.operation
    assert authority_rejected.result.native_report["verification"]["passed"] is True
    assert authority_rejected.result.native_report["gate"]["passed"] is False
    assert authority_rejected.result.native_report["gate"]["authority"]["auth_ok"] is False
    assert authority_rejected.result.rendered == "refusal"
    assert authority_rejected.result.memory_active == 0

    risk_rejected = run_learned_task(model, hidden, risk={"privacy": 0.1})
    assert risk_rejected.proposal.operation == task.operation
    assert risk_rejected.result.native_report["verification"]["passed"] is True
    assert risk_rejected.result.native_report["gate"]["authority"]["risk_ok"] is False
    assert risk_rejected.result.rendered == "refusal"


def test_c1_hidden_schema_prototype_trains_against_matched_baselines(tmp_path) -> None:
    artifact = tmp_path / "c1-prototype.npz"
    _, report = train_c1_prototype(
        TrainingConfig(limit=2, epochs=12, hidden_dim=12, device="cpu", artifact=artifact)
    )

    baselines = {baseline.name: baseline for baseline in report.heldout.baselines}
    assert report.final_accuracy > report.initial_accuracy
    assert report.heldout.solve_rate > baselines["majority"].solve_rate
    assert report.heldout.operation_accuracy > baselines["majority"].operation_accuracy
    assert baselines["enumerative-full"].solve_rate == 1.0
    assert report.heldout.compression.frontier_delta_vs_enumerative > 0.0
    assert any(trace.passed and trace.memory_key for trace in report.heldout.traces)
    assert any(not trace.passed for trace in report.heldout.traces)

    loaded_report = evaluate_saved_c1_prototype(artifact, limit=2, device="cpu")
    assert loaded_report.solve_rate == report.heldout.solve_rate
    assert len(loaded_report.traces) == loaded_report.tasks


def first_certified_exact(model, tasks: list[C0Task]):
    """Return one held-out task solved by the learned proposal."""
    for task in tasks:
        inference = run_learned_task(model, task)
        if inference.proposal.operation == task.operation and gate_passed(
            inference.result.native_report
        ):
            return task, inference
    raise AssertionError("no exact learned proposal certified")
