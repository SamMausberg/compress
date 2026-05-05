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
from vpm.evaluation.openai_release_baseline import (
    OpenAIReleaseBaselineArtifacts,
    run_openai_release_baselines,
)
from vpm.providers.openai_baseline import (
    LlmTaskKind,
    OpenAILlmConfig,
    OpenAIResponder,
    openai_config_from_env,
    run_openai_llm_predictions,
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
    "LlmTaskKind",
    "OpenAILlmConfig",
    "OpenAIReleaseBaselineArtifacts",
    "OpenAIResponder",
    "hard_llm_baseline_tasks",
    "hard_llm_task_spec",
    "llm_baseline_tasks",
    "llm_task_spec",
    "normalize_hard_answer",
    "normalized_operation",
    "openai_config_from_env",
    "parse_hard_prediction",
    "parse_prediction",
    "read_hard_llm_predictions",
    "read_llm_predictions",
    "run_openai_llm_predictions",
    "run_openai_release_baselines",
    "score_hard_llm_baseline_predictions",
    "score_hard_prediction",
    "score_llm_baseline_predictions",
    "score_prediction",
    "write_hard_llm_baseline_tasks",
    "write_llm_baseline_tasks",
]
