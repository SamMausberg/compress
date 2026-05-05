"""Matched-baseline audit reports."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import cast

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
        external_baseline(
            family,
            env_var,
            max_compute_units=float(vpm.tasks),
            task_kind="c1",
            expected_task_ids=tuple(task.task_id for task in heldout),
        )
        for family, env_var in ((BaselineFamily.LLM, "VPM_LLM_BASELINE_JSON"),)
    )
    return BaselineSuite(limit, tuple(executable) + external)


def external_baseline(
    family: BaselineFamily,
    env_var: str,
    *,
    max_compute_units: float,
    task_kind: str | None = None,
    expected_task_ids: tuple[str, ...] | None = None,
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
    errors = external_baseline_errors(
        solve_rate,
        compute_units,
        max_compute_units,
    ) + external_baseline_artifact_errors(
        payload,
        solve_rate=solve_rate,
        compute_units=compute_units,
        max_compute_units=max_compute_units,
        task_kind=task_kind,
        expected_task_ids=expected_task_ids,
    )
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
    if compute_units <= 0.0 and max_compute_units > 0.0:
        errors.append("compute_units must be positive for non-empty matched baselines")
    if compute_units > max_compute_units:
        errors.append(
            f"compute_units {compute_units:.3f} exceeds matched budget {max_compute_units:.3f}"
        )
    return tuple(errors)


def external_baseline_artifact_errors(
    payload: Mapping[str, object],
    *,
    solve_rate: float,
    compute_units: float,
    max_compute_units: float,
    task_kind: str | None,
    expected_task_ids: tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Validate scorer-produced artifact provenance for external baselines."""
    if task_kind is None:
        return ()
    errors: list[str] = []
    if payload.get("artifact_kind") != "vpm-external-llm-baseline-v1":
        errors.append("artifact_kind must be vpm-external-llm-baseline-v1")
    if payload.get("task_kind") != task_kind:
        errors.append(f"task_kind must be {task_kind}")
    if payload.get("status") != BaselineStatus.EXECUTED.value:
        errors.append("status must be executed")
    task_count = validate_task_count(payload.get("tasks"), max_compute_units, errors)
    traces = payload.get("traces")
    if not isinstance(traces, list) or not traces:
        errors.append("traces must be a non-empty list")
        return tuple(errors)
    trace_items = cast(list[object], traces)
    if task_count and len(trace_items) != task_count:
        errors.append("traces length must equal tasks")
    validate_trace_totals(
        trace_items,
        task_kind,
        compute_units=compute_units,
        solve_rate=solve_rate,
        task_count=task_count,
        errors=errors,
    )
    if expected_task_ids is not None:
        validate_task_manifest(payload, trace_items, expected_task_ids, errors)
    return tuple(errors)


def validate_task_count(
    tasks: object,
    max_compute_units: float,
    errors: list[str],
) -> int:
    """Validate artifact task count and return it when usable."""
    expected_tasks = int(max_compute_units) if max_compute_units.is_integer() else None
    if not isinstance(tasks, int) or isinstance(tasks, bool) or tasks <= 0:
        errors.append("tasks must be a positive integer")
        return 0
    if expected_tasks is not None and tasks != expected_tasks:
        errors.append(f"tasks must cover the full held-out split ({expected_tasks})")
    return tasks


def validate_trace_totals(
    traces: list[object],
    task_kind: str,
    *,
    compute_units: float,
    solve_rate: float,
    task_count: int,
    errors: list[str],
) -> None:
    """Validate aggregate trace score fields against artifact totals."""
    trace_compute = 0.0
    trace_solved = 0
    for index, trace in enumerate(traces, start=1):
        trace_compute += trace_compute_units(trace, index, task_kind, errors)
        trace_solved += int(trace_passed(trace, task_kind))
    if abs(trace_compute - compute_units) > 1e-9:
        errors.append("trace compute_units must sum to compute_units")
    if task_count and abs((trace_solved / task_count) - solve_rate) > 1e-9:
        errors.append("trace solve rate must equal solve_rate")


def task_manifest(task_ids: tuple[str, ...]) -> dict[str, object]:
    """Return the canonical external-baseline held-out task manifest."""
    return {
        "task_ids": task_ids,
        "task_digest": task_manifest_digest(task_ids),
    }


def task_manifest_digest(task_ids: tuple[str, ...]) -> str:
    """Return a stable digest for an ordered held-out task-id set."""
    encoded = json.dumps(list(task_ids), separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def validate_task_manifest(
    payload: Mapping[str, object],
    traces: list[object],
    expected_task_ids: tuple[str, ...],
    errors: list[str],
) -> None:
    """Validate external artifact binding to the exact held-out task set."""
    trace_ids = tuple(trace_task_id(trace) for trace in traces)
    if trace_ids != expected_task_ids:
        errors.append("trace task_ids must match the expected held-out task set")
    manifest = payload.get("task_manifest")
    if not isinstance(manifest, Mapping):
        errors.append("task_manifest must be an object")
        return
    manifest_map = cast(Mapping[str, object], manifest)
    manifest_task_ids = string_sequence(manifest_map.get("task_ids"))
    if manifest_task_ids is None:
        errors.append("task_manifest.task_ids must be a string list")
    elif manifest_task_ids != expected_task_ids:
        errors.append("task_manifest.task_ids must match the expected held-out task set")
    digest = manifest_map.get("task_digest")
    if digest != task_manifest_digest(expected_task_ids):
        errors.append("task_manifest.task_digest must match the expected held-out task set")


def string_sequence(value: object) -> tuple[str, ...] | None:
    """Return a tuple when value is a JSON string list."""
    if not isinstance(value, list):
        return None
    items = cast(list[object], value)
    if not all(isinstance(item, str) for item in items):
        return None
    return tuple(cast(list[str], items))


def trace_task_id(trace: object) -> str | None:
    """Return a trace task id when the trace shape permits it."""
    if not isinstance(trace, Mapping):
        return None
    trace_map = cast(Mapping[str, object], trace)
    task_id = trace_map.get("task_id")
    return task_id if isinstance(task_id, str) else None


def trace_passed(trace: object, task_kind: str) -> bool:
    """Return whether one trace is marked solved for its task kind."""
    if not isinstance(trace, Mapping):
        return False
    trace_map = cast(Mapping[str, object], trace)
    solved_field = "certified" if task_kind == "c1" else "correct"
    return trace_map.get(solved_field) is True


def trace_compute_units(
    trace: object,
    index: int,
    task_kind: str,
    errors: list[str],
) -> float:
    """Validate one scorer trace and return its logged compute units."""
    if not isinstance(trace, Mapping):
        errors.append(f"trace {index}: must be an object")
        return 0.0
    trace_map = cast(Mapping[str, object], trace)
    task_id = trace_map.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        errors.append(f"trace {index}: task_id must be a non-empty string")
    compute = trace_map.get("compute_units")
    if not isinstance(compute, int | float) or isinstance(compute, bool):
        errors.append(f"trace {index}: compute_units must be numeric")
        compute_units = 0.0
    else:
        compute_units = float(compute)
        if compute_units < 0.0:
            errors.append(f"trace {index}: compute_units must be non-negative")
    trace_errors = trace_map.get("errors")
    if not isinstance(trace_errors, list | tuple):
        errors.append(f"trace {index}: errors must be a sequence")
    elif trace_errors:
        errors.append(f"trace {index}: errors must be empty")
    if task_kind == "c1":
        require_string_trace_field(trace_map, index, "model", errors)
        require_bool_trace_field(trace_map, index, "certified", errors)
        require_bool_trace_field(trace_map, index, "operation_correct", errors)
        require_string_trace_field(trace_map, index, "expected_operation", errors)
    elif task_kind == "hard":
        require_string_trace_field(trace_map, index, "model", errors)
        require_bool_trace_field(trace_map, index, "correct", errors)
        require_string_trace_field(trace_map, index, "expected_answer", errors)
        require_string_trace_field(trace_map, index, "domain", errors)
    return compute_units


def require_bool_trace_field(
    trace: Mapping[str, object],
    index: int,
    field: str,
    errors: list[str],
) -> None:
    """Record an error when a trace bool field is missing or malformed."""
    if not isinstance(trace.get(field), bool):
        errors.append(f"trace {index}: {field} must be boolean")


def require_string_trace_field(
    trace: Mapping[str, object],
    index: int,
    field: str,
    errors: list[str],
) -> None:
    """Record an error when a trace string field is missing or malformed."""
    if not isinstance(trace.get(field), str):
        errors.append(f"trace {index}: {field} must be a string")


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
    "task_manifest",
    "task_manifest_digest",
]
