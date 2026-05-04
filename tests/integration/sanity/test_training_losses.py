"""Executable scalar training-loss controls."""

from __future__ import annotations

import pytest

from vpm.training.losses import (
    LOSS_NAMES,
    frontier_loss,
    loss_report,
    split_loss,
    support_loss,
    trace_loss,
)

pytestmark = pytest.mark.sanity


def test_loss_registry_covers_every_section8_component() -> None:
    expected = {
        "free_energy",
        "base",
        "cmp",
        "trace",
        "value",
        "repair",
        "halt",
        "ver",
        "cal",
        "safe",
        "mem",
        "supp",
        "render",
        "ctx",
        "sem",
        "src",
        "rebut",
        "ent",
        "real",
        "tb",
        "mf",
        "split",
        "sub",
        "dom",
        "dep",
        "front",
        "probe",
    }
    assert set(LOSS_NAMES) == expected

    report = loss_report(
        {"base": 1.0, "ver": 2.0, "render": 10.0},
        weights={"ver": 0.5},
        blocked={"render": "renderer gate detached"},
    )
    assert report.total == 2.0
    assert "cmp" in report.missing
    assert report.to_dict()["components"][0]["name"] == "free_energy"


def test_hard_gated_loss_components_have_expected_direction() -> None:
    assert split_loss(0.9, split_clean=False) < split_loss(0.1, split_clean=False)
    assert support_loss(0.3, 0.2, 0.9, True) > support_loss(0.1, 0.2, 0.9, True)
    assert frontier_loss(0.5, 0.5, 0.9) < frontier_loss(-0.5, 0.5, 0.1)


def test_trace_loss_is_zero_when_effective_sample_size_is_low() -> None:
    assert (
        trace_loss(
            (0.9,),
            (1.0,),
            effective_sample_size=1.0,
            minimum_ess=2.0,
        )
        == 0.0
    )
    assert (
        trace_loss(
            (0.9,),
            (1.0,),
            effective_sample_size=2.0,
            minimum_ess=2.0,
        )
        > 0.0
    )
