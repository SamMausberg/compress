"""§5 support guard and candidate rehydration helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass

from vpm import _native
from vpm._reports import float_field, object_map


@dataclass(frozen=True)
class SupportGuardReport:
    """Candidate-level support-pruning report."""

    candidates_before: tuple[str, ...]
    candidates_after: tuple[str, ...]
    candidates_final: tuple[str, ...]
    rehydrated: tuple[str, ...]
    retained_mass: float
    omitted_mass: float
    recall_upper: float
    shift_loss: float
    selection_loss: float
    epsilon_prune: float
    certificate: float
    epsilon_max: float
    passed: bool
    action: str

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly support guard report."""
        return {
            "candidates_before": self.candidates_before,
            "candidates_after": self.candidates_after,
            "candidates_final": self.candidates_final,
            "rehydrated": self.rehydrated,
            "retained_mass": self.retained_mass,
            "omitted_mass": self.omitted_mass,
            "recall_upper": self.recall_upper,
            "shift_loss": self.shift_loss,
            "selection_loss": self.selection_loss,
            "epsilon_prune": self.epsilon_prune,
            "certificate": self.certificate,
            "epsilon_max": self.epsilon_max,
            "passed": self.passed,
            "action": self.action,
        }


def guard_support(
    candidates_before: tuple[str, ...],
    candidates_after: tuple[str, ...],
    *,
    exact_rejection_witness: bool,
    recall_upper: float = 0.0,
    shift_loss: float = 0.0,
    selection_loss: float = 0.0,
    epsilon_max: float = 0.2,
) -> SupportGuardReport:
    """Guard a pruning step and rehydrate omitted candidates when unsafe."""
    unknown = tuple(
        candidate for candidate in candidates_after if candidate not in candidates_before
    )
    if unknown:
        raise ValueError(f"retained candidates were not in the original support: {unknown}")

    retained_mass = retained_support_mass(
        len(candidates_before),
        len(candidates_after),
        exact_rejection_witness,
    )
    native = native_support_guard(
        len(candidates_before),
        len(candidates_after),
        retained_mass,
        recall_upper,
        shift_loss,
        selection_loss,
        epsilon_max,
    )
    omitted = tuple(
        candidate for candidate in candidates_before if candidate not in candidates_after
    )
    rehydrated = omitted if native["action"] == "rehydrate" else ()
    candidates_final = candidates_before if rehydrated else candidates_after
    return SupportGuardReport(
        candidates_before=candidates_before,
        candidates_after=candidates_after,
        candidates_final=candidates_final,
        rehydrated=rehydrated,
        retained_mass=float_field(native, "retained_mass"),
        omitted_mass=float_field(native, "omitted_mass"),
        recall_upper=float_field(native, "recall_upper"),
        shift_loss=float_field(native, "shift_loss"),
        selection_loss=float_field(native, "selection_loss"),
        epsilon_prune=float_field(native, "epsilon_prune"),
        certificate=float_field(native, "certificate"),
        epsilon_max=float_field(native, "epsilon_max"),
        passed=native.get("passed") is True,
        action=str(native.get("action", "rehydrate")),
    )


def retained_support_mass(
    before_count: int,
    after_count: int,
    exact_rejection_witness: bool,
) -> float:
    """Return retained support mass for exact or heuristic pruning."""
    if before_count <= 0:
        return 0.0
    if exact_rejection_witness:
        return 1.0
    return after_count / before_count


def native_support_guard(
    candidates_before: int,
    candidates_after: int,
    retained_mass: float,
    recall_upper: float,
    shift_loss: float,
    selection_loss: float,
    epsilon_max: float,
) -> dict[str, object]:
    """Call the Rust support guard and return a typed JSON map."""
    raw = _native.support_guard_json(
        candidates_before,
        candidates_after,
        retained_mass,
        recall_upper,
        shift_loss,
        selection_loss,
        epsilon_max,
    )
    payload = object_map(json.loads(raw))
    if payload is None:
        raise ValueError("native support guard returned non-object JSON")
    return dict(payload)


__all__ = [
    "SupportGuardReport",
    "guard_support",
    "native_support_guard",
    "retained_support_mass",
]
