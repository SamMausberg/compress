"""Low-latency active memory capsules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryCapsule:
    """Replay-safe memory capsule admitted after a passed gate."""

    key: str
    value: object
    certificate_score: float
    witnesses: tuple[str, ...]


def dict_capsules() -> dict[str, MemoryCapsule]:
    """Typed default factory for capsule dictionaries."""
    return {}


__all__ = ["MemoryCapsule", "dict_capsules"]
