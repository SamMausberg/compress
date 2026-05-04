"""Executable substrate component controls."""

from __future__ import annotations

import pytest

from vpm.substrate import (
    SSMState,
    bind_slots,
    encode_task_graph,
    project_operation,
    run_selective_ssm,
    substrate_loss,
    substrate_recall_report,
)
from vpm.tasks import arithmetic_task
from vpm.training import edge_deletion_probe

pytestmark = pytest.mark.sanity


def test_typed_event_encoder_does_not_leak_operation_label() -> None:
    task = arithmetic_task("mul", 3, 4)
    graph = encode_task_graph(task, scale=8)
    rendered = repr(graph.to_dict())
    assert task.operation not in rendered
    assert [event.role for event in graph.events] == ["left", "right", "expected"]
    assert graph.edges == (("left", "expected"), ("right", "expected"))


def test_selective_ssm_and_slots_track_critical_omissions() -> None:
    graph = encode_task_graph(arithmetic_task("add", 2, 5), scale=8)
    state = run_selective_ssm(graph.feature_rows)
    assert isinstance(state, SSMState)
    assert any(value != 0.0 for value in state.hidden)

    binding = bind_slots(graph, state, slot_count=1)
    assert binding.omission_loss == 0.5
    report = substrate_recall_report(
        omitted_edges=len(binding.omitted_edges),
        total_edges=len(graph.edges),
        critical_omissions=len(binding.omitted_edges),
        threshold=0.0,
    )
    assert report.passed is False
    assert substrate_loss(report) > 0.0


def test_projection_is_proposal_only_and_probe_detects_deleted_edge() -> None:
    graph = encode_task_graph(arithmetic_task("add", 2, 5), scale=8)
    state = run_selective_ssm(graph.feature_rows)
    binding = bind_slots(graph, state, slot_count=2)
    projection = project_operation(binding, {"add": 2.0, "mul": 0.5})
    assert projection.proposal.operation == "add"
    assert 0.0 < projection.proposal.confidence < 1.0

    probe = edge_deletion_probe(graph, ("left", "expected"))
    assert probe.failed is True
    assert probe.report.epsilon_crit > 0.0
