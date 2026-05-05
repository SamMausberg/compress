"""One-shot OpenAI external-baseline release runner."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from vpm.evaluation.baselines import BaselineStatus
from vpm.evaluation.llm_baseline_c1 import (
    LlmBaselineScore,
    score_llm_baseline_predictions,
    write_llm_baseline_tasks,
)
from vpm.evaluation.llm_baseline_hard import (
    HardLlmBaselineScore,
    score_hard_llm_baseline_predictions,
    write_hard_llm_baseline_tasks,
)
from vpm.providers.openai_baseline import (
    LlmTaskKind,
    OpenAILlmConfig,
    OpenAIResponder,
    run_openai_llm_predictions,
)


@dataclass(frozen=True)
class OpenAIReleaseBaselineArtifacts:
    """Paths and scores produced by a release-baseline run."""

    output_dir: Path
    c1_tasks: Path
    c1_predictions: Path
    c1_baseline_json: Path
    hard_tasks: Path
    hard_predictions: Path
    hard_baseline_json: Path
    c1_score: LlmBaselineScore
    hard_score: HardLlmBaselineScore

    @property
    def env_exports(self) -> tuple[str, str]:
        """Environment assignments required by the release audit."""
        return (
            f"VPM_LLM_BASELINE_JSON={self.c1_baseline_json}",
            f"VPM_HARD_LLM_BASELINE_JSON={self.hard_baseline_json}",
        )

    @property
    def passed(self) -> bool:
        """True when both scored baselines are valid external artifacts."""
        return (
            self.c1_score.status is BaselineStatus.EXECUTED
            and self.hard_score.status is BaselineStatus.EXECUTED
        )

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly artifact report."""
        return {
            "output_dir": str(self.output_dir),
            "passed": self.passed,
            "env_exports": self.env_exports,
            "c1": {
                "tasks": str(self.c1_tasks),
                "predictions": str(self.c1_predictions),
                "baseline_json": str(self.c1_baseline_json),
                "score": self.c1_score.to_dict(),
            },
            "hard": {
                "tasks": str(self.hard_tasks),
                "predictions": str(self.hard_predictions),
                "baseline_json": str(self.hard_baseline_json),
                "score": self.hard_score.to_dict(),
            },
        }


def run_openai_release_baselines(
    output_dir: Path,
    *,
    config: OpenAILlmConfig,
    c1_limit: int = 2,
    responder: OpenAIResponder | None = None,
) -> OpenAIReleaseBaselineArtifacts:
    """Export, run, score, and write both release LLM baseline JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    c1_tasks = output_dir / "c1-llm-tasks.jsonl"
    c1_predictions = output_dir / "c1-llm-predictions.jsonl"
    c1_baseline_json = output_dir / "llm-baseline.json"
    hard_tasks = output_dir / "hard-llm-tasks.jsonl"
    hard_predictions = output_dir / "hard-llm-predictions.jsonl"
    hard_baseline_json = output_dir / "hard-llm-baseline.json"

    write_llm_baseline_tasks(c1_tasks, limit=c1_limit)
    write_hard_llm_baseline_tasks(hard_tasks)
    run_openai_llm_predictions(
        c1_tasks,
        c1_predictions,
        kind=LlmTaskKind.C1,
        config=config,
        responder=responder,
    )
    run_openai_llm_predictions(
        hard_tasks,
        hard_predictions,
        kind=LlmTaskKind.HARD,
        config=config,
        responder=responder,
    )

    c1_score = score_llm_baseline_predictions(c1_predictions, limit=c1_limit)
    hard_score = score_hard_llm_baseline_predictions(hard_predictions)
    write_valid_baseline_json(c1_baseline_json, c1_score.to_external_json())
    write_valid_baseline_json(hard_baseline_json, hard_score.to_external_json())
    return OpenAIReleaseBaselineArtifacts(
        output_dir=output_dir,
        c1_tasks=c1_tasks,
        c1_predictions=c1_predictions,
        c1_baseline_json=c1_baseline_json,
        hard_tasks=hard_tasks,
        hard_predictions=hard_predictions,
        hard_baseline_json=hard_baseline_json,
        c1_score=c1_score,
        hard_score=hard_score,
    )


def write_valid_baseline_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write a release-audit baseline JSON with stable formatting."""
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True))


__all__ = [
    "OpenAIReleaseBaselineArtifacts",
    "run_openai_release_baselines",
    "write_valid_baseline_json",
]
