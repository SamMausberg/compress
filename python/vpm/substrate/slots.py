"""Slot binding for substrate proposal state."""

from __future__ import annotations

from dataclasses import dataclass

from vpm.substrate.encoder import TypedEventGraph
from vpm.substrate.ssm import SSMState


@dataclass(frozen=True)
class SlotBinding:
    """Bound substrate slots and recall accounting."""

    slots: tuple[tuple[float, ...], ...]
    bound_edges: tuple[tuple[str, str], ...]
    omitted_edges: tuple[tuple[str, str], ...]

    @property
    def omission_loss(self) -> float:
        """Fraction of graph edges omitted by the slot binding."""
        total = len(self.bound_edges) + len(self.omitted_edges)
        return len(self.omitted_edges) / total if total else 0.0

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly slot binding."""
        return {
            "slots": self.slots,
            "bound_edges": self.bound_edges,
            "omitted_edges": self.omitted_edges,
            "omission_loss": self.omission_loss,
        }


def bind_slots(
    graph: TypedEventGraph,
    state: SSMState,
    *,
    slot_count: int = 2,
    critical_edges: tuple[tuple[str, str], ...] | None = None,
) -> SlotBinding:
    """Bind a recurrent state into slots while tracking critical omissions."""
    active_edges = critical_edges if critical_edges is not None else graph.edges
    slots = tuple(state.hidden for _ in range(max(slot_count, 0)))
    capacity = len(slots)
    bound_edges = tuple(active_edges[:capacity])
    omitted_edges = tuple(active_edges[capacity:])
    return SlotBinding(slots, bound_edges, omitted_edges)


__all__ = ["SlotBinding", "bind_slots"]
