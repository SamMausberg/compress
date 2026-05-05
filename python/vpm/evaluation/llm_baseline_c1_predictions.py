"""Prediction parsing for held-out C1 external LLM baselines."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from vpm.evaluation.baselines import optional_float


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
    raw_output = payload.get("raw_output")
    if not isinstance(raw_output, str) or not raw_output.strip():
        errors.append(f"line {line_number}: raw_output must be a non-empty string")
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


def normalized_operation(operation: str | None) -> str | None:
    """Normalize a predicted operation token."""
    if operation is None:
        return None
    return operation.strip().casefold()


__all__ = [
    "LlmBaselinePrediction",
    "normalized_operation",
    "parse_prediction",
    "read_llm_predictions",
]
