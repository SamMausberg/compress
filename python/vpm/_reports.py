"""Typed helpers for JSON-like native reports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

JsonMap = Mapping[str, object]


def object_map(value: object) -> JsonMap | None:
    """Narrow an object to a string-keyed mapping."""
    if isinstance(value, Mapping):
        return cast(JsonMap, value)
    return None


def object_list(value: object) -> list[object] | None:
    """Narrow an object to a JSON list."""
    if isinstance(value, list):
        return cast(list[object], value)
    return None


def float_field(mapping: JsonMap, key: str, default: float = 0.0) -> float:
    """Read a numeric field as float."""
    value = mapping.get(key, default)
    if isinstance(value, int | float):
        return float(value)
    return default
