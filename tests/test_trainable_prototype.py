"""Trainable non-transformer prototype tests."""

from __future__ import annotations

from vpm.tasks import C0Task
from vpm.training import (
    TrainingConfig,
    curriculum_split,
    evaluate_saved_prototype,
    run_learned_task,
    train_c0_prototype,
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


def first_certified_exact(model, tasks: list[C0Task]):
    """Return one held-out task solved by the learned proposal."""
    for task in tasks:
        inference = run_learned_task(model, task)
        if inference.proposal.operation == task.operation and gate_passed(
            inference.result.native_report
        ):
            return task, inference
    raise AssertionError("no exact learned proposal certified")
