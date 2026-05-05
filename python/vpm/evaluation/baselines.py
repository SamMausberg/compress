"""Matched-baseline audit reports."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from vpm._reports import object_map
from vpm.evaluation import evaluate_c1
from vpm.evaluation.neural_baselines import train_local_neural_baselines
from vpm.tasks.c1 import as_c0_tasks, schema_split
from vpm.training.prototype_metrics import matched_baselines


class BaselineFamily(StrEnum):
    """Baseline families required for matched-compute claims."""

    VPM = "vpm"
    PROGRAM_SYNTHESIS = "program_synthesis"
    TRANSFORMER = "transformer"
    SSM = "ssm"
    LLM = "llm"


class BaselineStatus(StrEnum):
    """Execution status for one baseline."""

    EXECUTED = "executed"
    NOT_CONFIGURED = "not_configured"
    INVALID = "invalid"


@dataclass(frozen=True)
class BaselineAudit:
    """One matched-baseline result or missing-baseline record."""

    name: str
    family: BaselineFamily
    status: BaselineStatus
    solve_rate: float | None
    operation_accuracy: float | None
    mean_candidates: float | None
    compute_units: float | None
    reason: str = ""
    max_compute_units: float | None = None

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly baseline audit."""
        return {
            "name": self.name,
            "family": self.family.value,
            "status": self.status.value,
            "solve_rate": self.solve_rate,
            "operation_accuracy": self.operation_accuracy,
            "mean_candidates": self.mean_candidates,
            "compute_units": self.compute_units,
            "reason": self.reason,
            "max_compute_units": self.max_compute_units,
        }


@dataclass(frozen=True)
class BaselineSuite:
    """Matched-baseline audit suite."""

    limit: int
    baselines: tuple[BaselineAudit, ...]

    @property
    def missing_families(self) -> tuple[str, ...]:
        """Families that lack an executed baseline."""
        return tuple(
            family.value
            for family in BaselineFamily
            if not any(
                baseline.family is family and baseline.status is BaselineStatus.EXECUTED
                for baseline in self.baselines
            )
        )

    @property
    def ready_for_claims(self) -> bool:
        """True only when every required family has an executed baseline."""
        return not self.missing_families

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly baseline suite."""
        return {
            "limit": self.limit,
            "ready_for_claims": self.ready_for_claims,
            "missing_families": self.missing_families,
            "baselines": [baseline.to_dict() for baseline in self.baselines],
        }


def evaluate_baseline_suite(limit: int = 2) -> BaselineSuite:
    """Run executable baselines and audit configured external baselines."""
    train, heldout = schema_split(limit)
    vpm = evaluate_c1(heldout, limit=limit)
    executable = [
        BaselineAudit(
            "vpm-c1-executable",
            BaselineFamily.VPM,
            BaselineStatus.EXECUTED,
            vpm.solve_rate,
            None,
            1.0,
            float(vpm.tasks),
        )
    ]
    executable.extend(
        BaselineAudit(
            baseline.name,
            BaselineFamily.PROGRAM_SYNTHESIS,
            BaselineStatus.EXECUTED,
            baseline.solve_rate,
            baseline.operation_accuracy,
            baseline.mean_candidates,
            baseline.mean_candidates * len(heldout),
        )
        for baseline in matched_baselines(as_c0_tasks(train), as_c0_tasks(heldout))
    )
    executable.extend(
        BaselineAudit(
            baseline.name,
            family,
            BaselineStatus.EXECUTED,
            baseline.solve_rate,
            baseline.operation_accuracy,
            baseline.mean_candidates,
            baseline.compute_units,
            "local matched C1 baseline",
        )
        for baseline, family in zip(
            train_local_neural_baselines(as_c0_tasks(train), as_c0_tasks(heldout), scale=limit),
            (BaselineFamily.TRANSFORMER, BaselineFamily.SSM),
            strict=True,
        )
    )
    external = tuple(
        external_baseline(family, env_var, max_compute_units=float(vpm.tasks))
        for family, env_var in ((BaselineFamily.LLM, "VPM_LLM_BASELINE_JSON"),)
    )
    return BaselineSuite(limit, tuple(executable) + external)


def external_baseline(
    family: BaselineFamily,
    env_var: str,
    *,
    max_compute_units: float,
) -> BaselineAudit:
    """Load an externally produced baseline JSON report if configured."""
    configured = os.environ.get(env_var)
    if configured is None:
        return BaselineAudit(
            family.value,
            family,
            BaselineStatus.NOT_CONFIGURED,
            None,
            None,
            None,
            None,
            f"{env_var} is not set",
            max_compute_units,
        )
    payload = object_map(json.loads(Path(configured).read_text()))
    if payload is None:
        raise ValueError(f"{env_var} did not contain a JSON object")
    solve_rate = required_float(payload, "solve_rate")
    compute_units = required_float(payload, "compute_units")
    errors = external_baseline_errors(solve_rate, compute_units, max_compute_units)
    status = BaselineStatus.INVALID if errors else BaselineStatus.EXECUTED
    return BaselineAudit(
        str(payload.get("name", family.value)),
        family,
        status,
        solve_rate,
        optional_float(payload.get("operation_accuracy")),
        optional_float(payload.get("mean_candidates")),
        compute_units,
        "; ".join(errors),
        max_compute_units,
    )


def external_baseline_errors(
    solve_rate: float,
    compute_units: float,
    max_compute_units: float,
) -> tuple[str, ...]:
    """Return validation errors for an external matched baseline."""
    errors: list[str] = []
    if solve_rate < 0.0 or solve_rate > 1.0:
        errors.append("solve_rate must be in [0, 1]")
    if compute_units < 0.0:
        errors.append("compute_units must be non-negative")
    if compute_units > max_compute_units:
        errors.append(
            f"compute_units {compute_units:.3f} exceeds matched budget {max_compute_units:.3f}"
        )
    return tuple(errors)


def required_float(payload: Mapping[str, object], field: str) -> float:
    """Parse a required numeric JSON field."""
    if field not in payload:
        raise ValueError(f"missing required numeric JSON field: {field}")
    value = payload[field]
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    raise ValueError(f"expected numeric JSON field: {field}")


def optional_float(value: object) -> float | None:
    """Parse an optional numeric JSON field."""
    if value is None:
        return None
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    raise ValueError("expected optional numeric JSON field")


__all__ = [
    "BaselineAudit",
    "BaselineFamily",
    "BaselineStatus",
    "BaselineSuite",
    "evaluate_baseline_suite",
    "external_baseline",
    "external_baseline_errors",
]
