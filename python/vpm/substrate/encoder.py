"""Typed event encoder for substrate proposals."""

from __future__ import annotations

from dataclasses import dataclass

from vpm.tasks.c0 import C0Task, C0Value, value_token

VALUE_KINDS = ("int", "text", "bool")
FEATURES = len(VALUE_KINDS) + 3


@dataclass(frozen=True)
class TypedEvent:
    """One operation-hidden typed event."""

    role: str
    kind: str
    features: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly event."""
        return {"role": self.role, "kind": self.kind, "features": self.features}


@dataclass(frozen=True)
class TypedEventGraph:
    """Small typed-event hypergraph used by the prototype substrate."""

    task_id: str
    events: tuple[TypedEvent, ...]
    edges: tuple[tuple[str, str], ...]

    @property
    def feature_rows(self) -> tuple[tuple[float, ...], ...]:
        """Dense feature rows for recurrent/SSM modules."""
        return tuple(event.features for event in self.events)

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly graph."""
        return {
            "task_id": self.task_id,
            "events": [event.to_dict() for event in self.events],
            "edges": self.edges,
        }


def encode_task_graph(task: C0Task, scale: float = 1.0) -> TypedEventGraph:
    """Encode operands and expected value without leaking the operation label."""
    safe_scale = max(float(scale), 1.0)
    roles = ("left", "right", "expected")
    values = (task.left, task.right, task.expected)
    events = tuple(
        TypedEvent(role, value_kind(value), value_features(value, safe_scale))
        for role, value in zip(roles, values, strict=True)
    )
    return TypedEventGraph(
        task_id=f"substrate-{value_token(task.left)}-{value_token(task.right)}-{value_token(task.expected)}",
        events=events,
        edges=(("left", "expected"), ("right", "expected")),
    )


def value_features(value: C0Value, scale: float) -> tuple[float, ...]:
    """Return dense type/value features for one C0 value."""
    row = [0.0] * FEATURES
    scalar_offset = len(VALUE_KINDS)
    if isinstance(value, bool):
        row[VALUE_KINDS.index("bool")] = 1.0
        row[scalar_offset] = 1.0 if value else 0.0
    elif isinstance(value, int):
        row[VALUE_KINDS.index("int")] = 1.0
        row[scalar_offset] = abs(float(value)) / scale
        row[scalar_offset + 1] = -1.0 if value < 0 else 1.0
    else:
        row[VALUE_KINDS.index("text")] = 1.0
        row[scalar_offset] = len(value) / scale
        row[scalar_offset + 2] = stable_text_feature(value)
    return tuple(row)


def value_kind(value: C0Value) -> str:
    """Return the non-coercive value kind."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    return "text"


def stable_text_feature(value: str) -> float:
    """Return a deterministic bounded text feature."""
    if not value:
        return 0.0
    total = sum((index + 1) * ord(char) for index, char in enumerate(value))
    return (total % 997) / 997.0


__all__ = [
    "FEATURES",
    "VALUE_KINDS",
    "TypedEvent",
    "TypedEventGraph",
    "encode_task_graph",
    "stable_text_feature",
    "value_features",
    "value_kind",
]
