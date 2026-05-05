"""External LLM baseline export and scoring facade."""

from __future__ import annotations

from vpm.evaluation.llm_baseline_c1 import (
    LlmBaselinePrediction,
    LlmBaselineScore,
    LlmBaselineTaskSpec,
    LlmBaselineTrace,
    llm_baseline_tasks,
    llm_task_spec,
    normalized_operation,
    parse_prediction,
    read_llm_predictions,
    score_llm_baseline_predictions,
    score_prediction,
    write_llm_baseline_tasks,
)
from vpm.evaluation.llm_baseline_hard import (
    HardLlmBaselinePrediction,
    HardLlmBaselineScore,
    HardLlmBaselineTaskSpec,
    HardLlmBaselineTrace,
    hard_llm_baseline_tasks,
    hard_llm_task_spec,
    normalize_hard_answer,
    parse_hard_prediction,
    read_hard_llm_predictions,
    score_hard_llm_baseline_predictions,
    score_hard_prediction,
    write_hard_llm_baseline_tasks,
)

__all__ = [
    "HardLlmBaselinePrediction",
    "HardLlmBaselineScore",
    "HardLlmBaselineTaskSpec",
    "HardLlmBaselineTrace",
    "LlmBaselinePrediction",
    "LlmBaselineScore",
    "LlmBaselineTaskSpec",
    "LlmBaselineTrace",
    "hard_llm_baseline_tasks",
    "hard_llm_task_spec",
    "llm_baseline_tasks",
    "llm_task_spec",
    "normalize_hard_answer",
    "normalized_operation",
    "parse_hard_prediction",
    "parse_prediction",
    "read_hard_llm_predictions",
    "read_llm_predictions",
    "score_hard_llm_baseline_predictions",
    "score_hard_prediction",
    "score_llm_baseline_predictions",
    "score_prediction",
    "write_hard_llm_baseline_tasks",
    "write_llm_baseline_tasks",
]
