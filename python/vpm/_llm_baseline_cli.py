"""Typer command registration for external LLM baseline harnesses."""

# pyright: reportUnusedFunction=false

from __future__ import annotations

import json
from pathlib import Path

import typer

from vpm.evaluation.baselines import BaselineStatus
from vpm.evaluation.llm_baseline import (
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
