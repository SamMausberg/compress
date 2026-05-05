"""Typer command registration for external LLM baseline harnesses."""

# pyright: reportUnusedFunction=false

from __future__ import annotations

import json
from pathlib import Path

import typer

from vpm.evaluation.baselines import BaselineStatus
from vpm.evaluation.llm_baseline import (
    LlmTaskKind,
    openai_config_from_env,
    run_openai_llm_predictions,
    score_hard_llm_baseline_predictions,
    score_llm_baseline_predictions,
    write_hard_llm_baseline_tasks,
    write_llm_baseline_tasks,
)

LLM_BASELINE_TASK_OUTPUT = typer.Argument(..., help="JSONL task file to write.")
LLM_BASELINE_PREDICTIONS = typer.Argument(..., help="JSONL prediction file to score.")
LLM_BASELINE_SCORE_OUTPUT = typer.Option(
    None,
    "--output",
    help="Optional JSON file to write for VPM_LLM_BASELINE_JSON.",
)
HARD_LLM_BASELINE_TASK_OUTPUT = typer.Argument(..., help="JSONL hard-domain task file to write.")
HARD_LLM_BASELINE_PREDICTIONS = typer.Argument(
    ...,
    help="JSONL hard-domain prediction file to score.",
)
HARD_LLM_BASELINE_SCORE_OUTPUT = typer.Option(
    None,
    "--output",
    help="Optional hard-domain LLM baseline JSON file to write.",
)
OPENAI_BASELINE_TASKS = typer.Argument(..., help="Exported VPM baseline task JSONL.")
OPENAI_BASELINE_PREDICTIONS = typer.Argument(..., help="Prediction JSONL to write.")
OPENAI_BASELINE_KIND = typer.Option(..., "--kind", help="Task file kind to run.")
OPENAI_BASELINE_MODEL = typer.Option(
    None,
    "--model",
    envvar="VPM_OPENAI_BASELINE_MODEL",
    help="OpenAI model to use; may also be set with VPM_OPENAI_BASELINE_MODEL.",
)
OPENAI_BASELINE_API_KEY_ENV = typer.Option(
    "OPENAI_API_KEY",
    help="Environment variable containing the OpenAI API key.",
)
OPENAI_BASELINE_BASE_URL = typer.Option(
    "https://api.openai.com/v1",
    help="OpenAI-compatible API base URL.",
)
OPENAI_BASELINE_TIMEOUT = typer.Option(60.0, help="Per-request timeout.")
OPENAI_BASELINE_REASONING_EFFORT = typer.Option(
    None,
    help="Optional Responses API reasoning.effort value.",
)


def register_llm_baseline_eval_commands(app: typer.Typer) -> None:
    """Register external LLM baseline export and scoring commands."""

    @app.command("export-llm-baseline")
    def export_llm_baseline_command(
        output: Path = LLM_BASELINE_TASK_OUTPUT,
        limit: int = typer.Option(2, help="Absolute integer limit used for C1 splits."),
    ) -> None:
        """Export held-out C1 prompts for a same-budget external LLM baseline."""
        tasks = write_llm_baseline_tasks(output, limit=limit)
        typer.echo(f"wrote={tasks} path={output}")

    @app.command("score-llm-baseline")
    def score_llm_baseline_command(
        predictions: Path = LLM_BASELINE_PREDICTIONS,
        limit: int = typer.Option(2, help="Absolute integer limit used for C1 splits."),
        output: Path | None = LLM_BASELINE_SCORE_OUTPUT,
        as_json: bool = typer.Option(False, "--json", help="Print metrics as JSON."),
    ) -> None:
        """Score external LLM operation predictions against held-out C1 tasks."""
        report = score_llm_baseline_predictions(predictions, limit=limit)
        if output is not None:
            if report.status is not BaselineStatus.EXECUTED:
                raise typer.BadParameter(
                    "refusing to write VPM_LLM_BASELINE_JSON for invalid score"
                )
            output.write_text(json.dumps(report.to_external_json(), indent=2, sort_keys=True))
        if as_json:
            typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            typer.echo(
                f"status={report.status.value} "
                f"solve_rate={report.solve_rate:.3f} "
                f"compute_units={report.compute_units:.3f} "
                f"max_compute_units={report.max_compute_units:.3f}"
            )

    @app.command("run-openai-llm-baseline")
    def run_openai_llm_baseline_command(
        tasks: Path = OPENAI_BASELINE_TASKS,
        predictions: Path = OPENAI_BASELINE_PREDICTIONS,
        kind: LlmTaskKind = OPENAI_BASELINE_KIND,
        model: str | None = OPENAI_BASELINE_MODEL,
        api_key_env: str = OPENAI_BASELINE_API_KEY_ENV,
        base_url: str = OPENAI_BASELINE_BASE_URL,
        timeout_seconds: float = OPENAI_BASELINE_TIMEOUT,
        reasoning_effort: str | None = OPENAI_BASELINE_REASONING_EFFORT,
    ) -> None:
        """Run exported tasks through OpenAI and write scorer-ready predictions."""
        if model is None:
            raise typer.BadParameter("--model or VPM_OPENAI_BASELINE_MODEL is required")
        try:
            config = openai_config_from_env(
                model=model,
                api_key_env=api_key_env,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
                reasoning_effort=reasoning_effort,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        written = run_openai_llm_predictions(tasks, predictions, kind=kind, config=config)
        typer.echo(f"wrote={written} path={predictions} kind={kind.value} model={model}")


def register_hard_llm_baseline_eval_commands(app: typer.Typer) -> None:
    """Register hard-domain external LLM baseline commands."""

    @app.command("export-hard-llm-baseline")
    def export_hard_llm_baseline_command(
        output: Path = HARD_LLM_BASELINE_TASK_OUTPUT,
    ) -> None:
        """Export held-out hard-domain prompts for an external LLM baseline."""
        tasks = write_hard_llm_baseline_tasks(output)
        typer.echo(f"wrote={tasks} path={output}")

    @app.command("score-hard-llm-baseline")
    def score_hard_llm_baseline_command(
        predictions: Path = HARD_LLM_BASELINE_PREDICTIONS,
        output: Path | None = HARD_LLM_BASELINE_SCORE_OUTPUT,
        as_json: bool = typer.Option(False, "--json", help="Print metrics as JSON."),
    ) -> None:
        """Score hard-domain external LLM answer predictions."""
        report = score_hard_llm_baseline_predictions(predictions)
        if output is not None:
            if report.status is not BaselineStatus.EXECUTED:
                raise typer.BadParameter("refusing to write invalid hard-domain LLM score")
            output.write_text(json.dumps(report.to_external_json(), indent=2, sort_keys=True))
        if as_json:
            typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            typer.echo(
                f"status={report.status.value} "
                f"solve_rate={report.solve_rate:.3f} "
                f"compute_units={report.compute_units:.3f} "
                f"max_compute_units={report.max_compute_units:.3f}"
            )


__all__ = ["register_hard_llm_baseline_eval_commands", "register_llm_baseline_eval_commands"]
