"""Matched-baseline audit reports."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from vpm._reports import float_field, object_map
from vpm.evaluation import evaluate_c1
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
    external = tuple(
        external_baseline(family, env_var)
        for family, env_var in (
            (BaselineFamily.TRANSFORMER, "VPM_TRANSFORMER_BASELINE_JSON"),
            (BaselineFamily.SSM, "VPM_SSM_BASELINE_JSON"),
            (BaselineFamily.LLM, "VPM_LLM_BASELINE_JSON"),
        )
    )
    return BaselineSuite(limit, tuple(executable) + external)


def external_baseline(family: BaselineFamily, env_var: str) -> BaselineAudit:
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
        )
    payload = object_map(json.loads(Path(configured).read_text()))
    if payload is None:
        raise ValueError(f"{env_var} did not contain a JSON object")
    return BaselineAudit(
        str(payload.get("name", family.value)),
        family,
        BaselineStatus.EXECUTED,
        float_field(payload, "solve_rate"),
        optional_float(payload.get("operation_accuracy")),
        optional_float(payload.get("mean_candidates")),
        optional_float(payload.get("compute_units")),
    )


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
]
