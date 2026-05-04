"""Trainable non-transformer prototype tests."""

from __future__ import annotations

from vpm.tasks import arithmetic_task
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
        TrainingConfig(limit=2, epochs=8, hidden_dim=8, device="cpu", artifact=artifact)
    )

    baselines = {baseline.name: baseline for baseline in report.heldout.baselines}
    assert report.final_accuracy > report.initial_accuracy
    assert report.heldout.solve_rate == 1.0
    assert report.heldout.solve_rate > baselines["majority"].solve_rate
    assert report.heldout.macro_memory_active == 2
    assert report.heldout.compression_ratio > 1.0

    loaded_report = evaluate_saved_prototype(artifact, limit=2, device="cpu")
    assert loaded_report.solve_rate == report.heldout.solve_rate

    _, heldout = curriculum_split(2)
    mul_task = next(task for task in heldout if task.operation == "mul")
    inference = run_learned_task(model, arithmetic_task("mul", mul_task.left, mul_task.right))
    assert inference.proposal.operation == "mul"
    assert gate_passed(inference.result.native_report)
    assert inference.result.rendered.startswith(f"{mul_task.expected} ")
