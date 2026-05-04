"""§3 — Verifier-native neural substrate ``VNS_θ``.

See ``docs/architecture/03-neural-substrate.md``.

This package will hold:

- ``encoder.py``   — typed event hypergraph ``x_t = Enc_θ(o_t, Γ)`` (eq. 34).
- ``ssm.py``       — selective SSM block ``h_t^{ssm}`` (eq. 36; ref [4]).
- ``slots.py``     — slot binding update (eqs. 37–38; ref [9]).
- ``projection.py``— executable projection ``Proj_θ`` (eq. 39).
- ``losses.py``    — substrate losses ``L_base`` family (eqs. 42–46) and
  the substrate-recall calibration ``ε_sub`` / ``ε_crit`` (eqs. 47–49).
- ``graph_shadow.py`` — differentiable shadow ``G_t^θ`` of the e-graph
  in ``crates/vpm-egraph``.

The substrate is an **amortizer, not an authority** (Abstract): every
output of this module is a *proposal* and never a certificate.
"""

from __future__ import annotations

from dataclasses import dataclass

from vpm.compiler import CompiledProgram
from vpm.substrate.encoder import (
    TypedEvent,
    TypedEventGraph,
    encode_task_graph,
    value_features,
    value_kind,
)
from vpm.substrate.losses import SubstrateRecallReport, substrate_loss, substrate_recall_report
from vpm.substrate.projection import ProjectionTrace, project_operation
from vpm.substrate.prototype import (
    OPERATIONS,
    ArithmeticProposalNet,
    OperationProposal,
    load_prototype,
    predict_operation,
    save_prototype,
)
from vpm.substrate.slots import SlotBinding, bind_slots
from vpm.substrate.ssm import SSMParameters, SSMState, run_selective_ssm


@dataclass(frozen=True)
class SubstrateState:
    """Substrate proposal state for a substrate-free MVP run."""

    events: tuple[str, ...]
    slots: tuple[str, ...]
    omission_loss: float


def encode_update(compiled: CompiledProgram) -> SubstrateState:
    """Encode a typed event without giving it certificate authority."""
    event = f"{compiled.operation}:{compiled.left}:{compiled.right}"
    slot = f"result:{compiled.expected}"
    return SubstrateState((event,), (slot,), 0.0)


__all__ = [
    "OPERATIONS",
    "ArithmeticProposalNet",
    "OperationProposal",
    "ProjectionTrace",
    "SSMParameters",
    "SSMState",
    "SlotBinding",
    "SubstrateRecallReport",
    "SubstrateState",
    "TypedEvent",
    "TypedEventGraph",
    "bind_slots",
    "encode_task_graph",
    "encode_update",
    "load_prototype",
    "predict_operation",
    "project_operation",
    "run_selective_ssm",
    "save_prototype",
    "substrate_loss",
    "substrate_recall_report",
    "value_features",
    "value_kind",
]
