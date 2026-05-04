"""Scalar training-loss accounting for §8."""

from __future__ import annotations

from vpm.training.losses.components import frontier_loss, split_loss, support_loss, trace_loss
from vpm.training.losses.primitives import binary_cross_entropy, hinge, squared_error
from vpm.training.losses.registry import LOSS_NAMES, LossComponent, LossReport, loss_report

__all__ = [
    "LOSS_NAMES",
    "LossComponent",
    "LossReport",
    "binary_cross_entropy",
    "frontier_loss",
    "hinge",
    "loss_report",
    "split_loss",
    "squared_error",
    "support_loss",
    "trace_loss",
]
