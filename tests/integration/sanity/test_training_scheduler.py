"""Executable training scheduler, budget, and query controls."""

from __future__ import annotations

import pytest

from vpm.training import (
    ActiveQuery,
    BudgetChannel,
    LossWeightState,
    allocate_budget_channels,
    balance_loss_weights,
    block_coordinate_plan,
    mechanism_reward,
    select_active_query,
    trajectory_balance_loss,
)

pytestmark = pytest.mark.sanity


def test_weight_balancer_downweights_large_gradient_losses() -> None:
    updated = balance_loss_weights(
        (
            LossWeightState("ver", weight=1.0, grad_ema=1.0, lower=0.1, upper=10.0),
            LossWeightState("render", weight=1.0, grad_ema=1.0, lower=0.1, upper=10.0),
        ),
        {"ver": 1.0, "render": 100.0},
        beta=1.0,
    )
    weights = {state.name: state.weight for state in updated}
    assert weights["render"] < weights["ver"]


def test_block_coordinate_plan_keeps_audit_and_gate_inputs_frozen() -> None:
    blocks = block_coordinate_plan()
    assert [block.name for block in blocks] == [
        "language",
        "evidence",
        "substrate_compiler",
        "render_memory",
    ]
    assert "audit_labels" in blocks[0].frozen_inputs
    assert "split" in blocks[1].losses


def test_budget_channels_allocate_only_positive_net_returns() -> None:
    plan = allocate_budget_channels(
        (
            BudgetChannel("exact", marginal_utility=2.0, loss_penalty=0.5),
            BudgetChannel("renderer", marginal_utility=0.1, loss_penalty=0.5),
            BudgetChannel("replay", marginal_utility=1.0, frontier_penalty=0.1),
        ),
        quanta=2,
    )
    assert [channel.name for channel in plan.selected] == ["exact", "replay"]
    assert plan.lambda0 == plan.selected[-1].net_return


def test_gflow_reward_and_active_query_are_hard_gated() -> None:
    reward = mechanism_reward(
        energy=0.0,
        certificate=2.0,
        certificate_threshold=1.0,
        support_loss=0.0,
        support_max=0.1,
        frontier_delta=1.0,
        risk_ok=True,
    )
    blocked = mechanism_reward(
        energy=0.0,
        certificate=0.0,
        certificate_threshold=1.0,
        support_loss=0.0,
        support_max=0.1,
        frontier_delta=1.0,
        risk_ok=True,
    )
    assert reward > blocked
    assert trajectory_balance_loss(0.0, (), (), reward) >= 0.0

    selected = select_active_query(
        (
            ActiveQuery(
                "cheap",
                entropy_reduction=0.2,
                expected_certified_utility=0.2,
                regression_probability=0.0,
                cost=0.1,
            ),
            ActiveQuery(
                "expensive",
                entropy_reduction=1.0,
                expected_certified_utility=1.0,
                regression_probability=0.0,
                cost=10.0,
            ),
        )
    )
    assert selected.name == "cheap"
