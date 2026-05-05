"""External LLM baseline export and scoring for held-out C1 tasks."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from vpm.evaluation.baselines import (
    BaselineAudit,
    BaselineFamily,
    BaselineStatus,
    external_baseline_errors,
    optional_float,
    task_manifest,
)
from vpm.infer import run_task_candidate
from vpm.substrate.prototype import OPERATIONS
from vpm.tasks.c0 import value_token
from vpm.tasks.c1 import C1Task, schema_split
from vpm.verifiers import gate_passed


@dataclass(frozen=True)
class LlmBaselineTaskSpec:
    """One public task exported for an external LLM baseline run."""

    task_id: str
    prompt: str
    left: object
    right: object
    observed_output: object
    allowed_operations: tuple[str, ...] = OPERATIONS

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly public task spec without gold labels."""
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "left": self.left,
            "right": self.right,
            "observed_output": self.observed_output,
            "allowed_operations": self.allowed_operations,
        }


@dataclass(frozen=True)
class LlmBaselinePrediction:
    """One external LLM operation prediction."""

    task_id: str
    operation: str | None
    compute_units: float | None
    model: str | None
    raw_output: str = ""

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly prediction."""
        return {
            "task_id": self.task_id,
            "operation": self.operation,
            "compute_units": self.compute_units,
            "model": self.model,
            "raw_output": self.raw_output,
        }


@dataclass(frozen=True)
class LlmBaselineTrace:
    """Scored result for one external LLM prediction."""

    task_id: str
    expected_operation: str
    predicted_operation: str | None
    certified: bool
    operation_correct: bool
    compute_units: float
    model: str | None
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly scored trace."""
        return {
            "task_id": self.task_id,
            "expected_operation": self.expected_operation,
            "predicted_operation": self.predicted_operation,
            "certified": self.certified,
            "operation_correct": self.operation_correct,
            "compute_units": self.compute_units,
            "model": self.model,
            "errors": self.errors,
        }


@dataclass(frozen=True)
class LlmBaselineScore:
    """Validated external LLM baseline score."""

    name: str
    limit: int
    traces: tuple[LlmBaselineTrace, ...]
    max_compute_units: float
    file_errors: tuple[str, ...] = ()

    @property
    def tasks(self) -> int:
        """Number of held-out tasks scored."""
        return len(self.traces)

    @property
    def task_ids(self) -> tuple[str, ...]:
        """Ordered held-out task ids covered by this score."""
        return tuple(trace.task_id for trace in self.traces)

    @property
    def solved(self) -> int:
        """Number of verifier-certified predictions."""
        return sum(trace.certified for trace in self.traces)

    @property
    def operation_correct(self) -> int:
        """Number of correct operation predictions."""
        return sum(trace.operation_correct for trace in self.traces)

    @property
    def solve_rate(self) -> float:
        """Verifier-certified solve rate."""
        return self.solved / self.tasks if self.tasks else 0.0

    @property
    def operation_accuracy(self) -> float:
        """Operation-label accuracy."""
        return self.operation_correct / self.tasks if self.tasks else 0.0

    @property
    def compute_units(self) -> float:
        """Logged external baseline compute units."""
        return sum(trace.compute_units for trace in self.traces)

    @property
    def errors(self) -> tuple[str, ...]:
        """All validation errors for the score."""
        trace_errors = tuple(
            f"{trace.task_id}: {error}" for trace in self.traces for error in trace.errors
        )
        budget_errors = external_baseline_errors(
            self.solve_rate,
            self.compute_units,
            self.max_compute_units,
        )
        return self.file_errors + trace_errors + budget_errors

    @property
    def status(self) -> BaselineStatus:
        """Execution status after schema and budget validation."""
        return BaselineStatus.INVALID if self.errors else BaselineStatus.EXECUTED

    def to_baseline_audit(self) -> BaselineAudit:
        """Convert to the shared baseline-audit shape."""
        return BaselineAudit(
            self.name,
            BaselineFamily.LLM,
            self.status,
            self.solve_rate,
            self.operation_accuracy,
            1.0 if self.tasks else 0.0,
            self.compute_units,
            "; ".join(self.errors),
            self.max_compute_units,
        )

    def to_external_json(self) -> dict[str, object]:
        """Return the JSON accepted by ``VPM_LLM_BASELINE_JSON`` when valid."""
        if self.status is not BaselineStatus.EXECUTED:
            raise ValueError("cannot export invalid external LLM baseline score")
        return {
            "artifact_kind": "vpm-external-llm-baseline-v1",
            "name": self.name,
            "task_kind": "c1",
            "status": self.status.value,
            "tasks": self.tasks,
            "solve_rate": self.solve_rate,
            "operation_accuracy": self.operation_accuracy,
            "mean_candidates": 1.0 if self.tasks else 0.0,
            "compute_units": self.compute_units,
            "max_compute_units": self.max_compute_units,
            "task_manifest": task_manifest(self.task_ids),
            "traces": [trace.to_dict() for trace in self.traces],
        }

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly score report."""
        return {
            "name": self.name,
            "limit": self.limit,
            "tasks": self.tasks,
            "solved": self.solved,
            "solve_rate": self.solve_rate,
            "operation_accuracy": self.operation_accuracy,
            "compute_units": self.compute_units,
            "max_compute_units": self.max_compute_units,
            "status": self.status.value,
            "errors": self.errors,
            "baseline": self.to_baseline_audit().to_dict(),
            "external_json": (
                self.to_external_json() if self.status is BaselineStatus.EXECUTED else None
            ),
            "traces": [trace.to_dict() for trace in self.traces],
        }


def llm_baseline_tasks(limit: int = 2) -> tuple[LlmBaselineTaskSpec, ...]:
    """Return public held-out C1 prompts for external LLM operation prediction."""
    _train, heldout = schema_split(limit)
    return tuple(llm_task_spec(task) for task in heldout)


def llm_task_spec(task: C1Task) -> LlmBaselineTaskSpec:
    """Build one public external-baseline prompt without leaking the gold operation."""
    allowed = ", ".join(OPERATIONS)
    prompt = (
        "Given a hidden typed relation in the form `left right -> observed_output`, "
        f"choose exactly one operation from [{allowed}] that certifies the relation. "
        'Return JSON only, for example {"operation":"add"}. '
        f"Task: {value_token(task.left)} {value_token(task.right)} -> "
        f"{value_token(task.expected)}"
    )
    return LlmBaselineTaskSpec(
        task_id=task.task_id,
        prompt=prompt,
        left=task.left,
        right=task.right,
        observed_output=task.expected,
    )


def write_llm_baseline_tasks(path: Path, *, limit: int = 2) -> int:
    """Write held-out external LLM baseline tasks as JSONL."""
    tasks = llm_baseline_tasks(limit)
    path.write_text("".join(f"{json.dumps(task.to_dict(), sort_keys=True)}\n" for task in tasks))
    return len(tasks)


def score_llm_baseline_predictions(
    predictions_path: Path,
    *,
    limit: int = 2,
    name: str = "external-llm-c1",
) -> LlmBaselineScore:
    """Score external LLM operation predictions against the held-out C1 split."""
    _train, heldout = schema_split(limit)
    predictions, file_errors = read_llm_predictions(predictions_path)
    traces = tuple(score_prediction(task, predictions.get(task.task_id)) for task in heldout)
    return LlmBaselineScore(
        name=name,
        limit=limit,
        traces=traces,
        max_compute_units=float(len(heldout)),
        file_errors=file_errors,
    )


def read_llm_predictions(
    path: Path,
) -> tuple[dict[str, LlmBaselinePrediction], tuple[str, ...]]:
    """Read external LLM baseline predictions from JSONL."""
    predictions: dict[str, LlmBaselinePrediction] = {}
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: invalid JSON: {exc.msg}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"line {line_number}: prediction must be a JSON object")
            continue
        prediction, prediction_errors = parse_prediction(
            cast(Mapping[str, object], payload),
            line_number,
        )
        errors.extend(prediction_errors)
        if prediction is None:
            continue
        if prediction.task_id in predictions:
            errors.append(f"line {line_number}: duplicate prediction for {prediction.task_id}")
        predictions[prediction.task_id] = prediction
    return predictions, tuple(errors)


def parse_prediction(
    payload: Mapping[str, object],
    line_number: int,
) -> tuple[LlmBaselinePrediction | None, tuple[str, ...]]:
    """Parse one external LLM prediction object."""
    errors: list[str] = []
    task_id = payload.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        return None, (f"line {line_number}: task_id must be a non-empty string",)
    operation = payload.get("operation")
    if operation is not None and not isinstance(operation, str):
        errors.append(f"line {line_number}: operation must be a string")
        operation = None
    try:
        compute_units = optional_float(payload.get("compute_units"))
    except ValueError:
        errors.append(f"line {line_number}: compute_units must be numeric")
        compute_units = None
    if compute_units is None:
        errors.append(f"line {line_number}: compute_units is required")
    raw_output = payload.get("raw_output", "")
    if not isinstance(raw_output, str):
        errors.append(f"line {line_number}: raw_output must be a string")
        raw_output = ""
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        errors.append(f"line {line_number}: model must be a non-empty string")
        model = None
    else:
        model = model.strip()
    return (
        LlmBaselinePrediction(
            task_id=task_id,
            operation=operation,
            compute_units=compute_units,
            model=model,
            raw_output=raw_output,
        ),
        tuple(errors),
    )


def score_prediction(
    task: C1Task,
    prediction: LlmBaselinePrediction | None,
) -> LlmBaselineTrace:
    """Score one prediction with the exact verifier-backed C1 bridge."""
    if prediction is None:
        return LlmBaselineTrace(
            task.task_id,
            task.operation,
            None,
            False,
            False,
            0.0,
            None,
            ("missing prediction",),
        )
    errors: list[str] = []
    operation = normalized_operation(prediction.operation)
    if operation is None:
        errors.append("operation is required")
    elif operation not in OPERATIONS:
        errors.append(f"unsupported operation: {operation}")
    compute_units = prediction.compute_units or 0.0
    if prediction.compute_units is None:
        errors.append("compute_units is required")
    elif prediction.compute_units < 0.0:
        errors.append("compute_units must be non-negative")
    if prediction.model is None:
        errors.append("model is required")
    certified = False
    if operation in OPERATIONS:
        result = run_task_candidate(task.to_c0_task(), operation)
        certified = gate_passed(result.native_report)
    return LlmBaselineTrace(
        task_id=task.task_id,
        expected_operation=task.operation,
        predicted_operation=operation,
        certified=certified,
        operation_correct=operation == task.operation,
        compute_units=compute_units,
        model=prediction.model,
        errors=tuple(errors),
    )


def normalized_operation(operation: str | None) -> str | None:
    """Normalize a predicted operation token."""
    if operation is None:
        return None
    return operation.strip().casefold()


__all__ = [
    "LlmBaselinePrediction",
    "LlmBaselineScore",
    "LlmBaselineTaskSpec",
    "LlmBaselineTrace",
    "llm_baseline_tasks",
    "llm_task_spec",
    "normalized_operation",
    "parse_prediction",
    "read_llm_predictions",
    "score_llm_baseline_predictions",
    "score_prediction",
    "write_llm_baseline_tasks",
]
