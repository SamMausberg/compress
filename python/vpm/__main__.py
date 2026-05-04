"""``python -m vpm`` — top-level CLI entry point.

Wired to ``[project.scripts] vpm = "vpm.__main__:app"`` in
``pyproject.toml``. The CLI surface is intentionally minimal at this
stage; subcommands will be added per milestone in
``docs/implementation/README.md``.
"""

from __future__ import annotations

import json

import typer

from . import __version__
from .evaluation import evaluate_c0
from .infer import run_c0_add
from .tasks import stages

app = typer.Typer(
    name="vpm",
    help="VPM-5.3 reference implementation. See docs/architecture/.",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command("run-c0-add")
def run_c0_add_command(
    left: int,
    right: int,
    expected: int | None = None,
    as_json: bool = typer.Option(False, "--json", help="Print the full JSON report."),
) -> None:
    """Run the C0 add vertical slice."""
    result = run_c0_add(left, right, expected)
    if as_json:
        typer.echo(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        typer.echo(result.rendered)


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


@app.command("stages")
def stages_command() -> None:
    """Print curriculum-stage implementation metadata."""
    typer.echo(json.dumps([stage.to_dict() for stage in stages()], indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
