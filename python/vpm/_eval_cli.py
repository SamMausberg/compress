"""Evaluation CLI command registration."""

# pyright: reportUnusedFunction=false

from __future__ import annotations

import json

import typer

from vpm.evaluation import (
    evaluate_c0,
    evaluate_c1,
    evaluate_c2,
    evaluate_c3,
    evaluate_c4,
    evaluate_c5,
)
from vpm.evaluation.failure_modes import evaluate_failure_modes


def register_eval_commands(app: typer.Typer) -> None:
    """Register evaluation commands on the top-level Typer app."""

    @app.command("eval-c0")
    def eval_c0_command(
        as_json: bool = typer.Option(False, "--json", help="Print metrics as JSON."),
    ) -> None:
        """Run the bundled C0 curriculum and summarize MVP metrics."""
        report = evaluate_c0()
        if as_json:
            typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            typer.echo(f"solve_rate={report.solve_rate:.3f} tasks={report.tasks}")

    @app.command("eval-c1")
    def eval_c1_command(
        limit: int = typer.Option(3, help="Absolute integer limit used for C1 generation."),
        as_json: bool = typer.Option(False, "--json", help="Print metrics as JSON."),
    ) -> None:
        """Run the executable C1 hidden-schema subset and summarize metrics."""
        report = evaluate_c1(limit=limit)
        if as_json:
            typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            typer.echo(f"solve_rate={report.solve_rate:.3f} tasks={report.tasks}")

    @app.command("eval-c2")
    def eval_c2_command(
        limit: int = typer.Option(3, help="Absolute integer limit used for C2 generation."),
        as_json: bool = typer.Option(False, "--json", help="Print metrics as JSON."),
    ) -> None:
        """Run C2 active-test selection and verifier-gated execution."""
        report = evaluate_c2(limit=limit)
        if as_json:
            typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            typer.echo(
                f"solve_rate={report.solve_rate:.3f} "
                f"support_reduction={report.support_reduction_rate:.3f} "
                f"tasks={report.tasks}"
            )

    @app.command("eval-c3")
    def eval_c3_command(
        as_json: bool = typer.Option(False, "--json", help="Print metrics as JSON."),
    ) -> None:
        """Run C3 adversarial authority/risk policy probes."""
        report = evaluate_c3()
        if as_json:
            typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            typer.echo(
                f"violation_rate={report.violation_rate:.3f} "
                f"rejected={report.rejected} "
                f"probes={report.probes}"
            )

    @app.command("eval-c4")
    def eval_c4_command(
        as_json: bool = typer.Option(False, "--json", help="Print metrics as JSON."),
    ) -> None:
        """Run C4 controlled dialogue source/rebuttal/realization gates."""
        report = evaluate_c4()
        if as_json:
            typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            typer.echo(
                f"render_rate={report.render_rate:.3f} "
                f"refusal_rate={report.refusal_rate:.3f} "
                f"violations={report.violations}"
            )

    @app.command("eval-c5")
    def eval_c5_command(
        as_json: bool = typer.Option(False, "--json", help="Print metrics as JSON."),
    ) -> None:
        """Run C5 replay-safe macro admission and demotion probes."""
        report = evaluate_c5()
        if as_json:
            typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            typer.echo(
                f"admission_rate={report.admission_rate:.3f} "
                f"demoted={report.demoted} "
                f"violations={report.violations}"
            )

    @app.command("eval-failures")
    def eval_failures_command(
        as_json: bool = typer.Option(False, "--json", help="Print metrics as JSON."),
    ) -> None:
        """Run executable Criterion-1 failure-mode checks."""
        report = evaluate_failure_modes()
        if as_json:
            typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            typer.echo(
                f"passed={report.passed} "
                f"failures={len(report.failures)} "
                f"uncovered={len(report.uncovered_clauses)}"
            )


__all__ = ["register_eval_commands"]
