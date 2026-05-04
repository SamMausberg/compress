"""``python -m vpm`` — top-level CLI entry point.

Wired to ``[project.scripts] vpm = "vpm.__main__:app"`` in
``pyproject.toml``. The CLI surface is intentionally minimal at this
stage; subcommands will be added per milestone in
``docs/implementation/README.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from . import __version__
from ._cli import prototype_summary, training_summary
from ._eval_cli import register_eval_commands
from .diagnostics import collect_diagnostics
from .infer import run_c0_add, run_task
from .substrate import load_prototype
from .tasks import stages, typed_hidden_task, typed_task
from .training import (
    TrainingConfig,
    evaluate_saved_c1_prototype,
    evaluate_saved_prototype,
    run_learned_task,
    train_c0_prototype,
    train_c1_prototype,
)

DEFAULT_PROTOTYPE_ARTIFACT = Path("artifacts/vpm_c0_prototype.npz")
DEFAULT_C1_PROTOTYPE_ARTIFACT = Path("artifacts/vpm_c1_prototype.npz")

app = typer.Typer(
    name="vpm",
    help="VPM-5.3 reference implementation. See docs/architecture/.",
    no_args_is_help=True,
)
register_eval_commands(app)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command("doctor")
def doctor_command(
    as_json: bool = typer.Option(False, "--json", help="Print diagnostics as JSON."),
) -> None:
    """Check the local native extension and PyTorch/CUDA runtime."""
    report = collect_diagnostics()
    if as_json:
        typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        typer.echo(
            " ".join(
                [
                    f"python={report.python}",
                    f"torch={report.torch}",
                    f"torch_cuda={report.torch_cuda}",
                    f"cuda={report.cuda_available}",
                    f"device={report.cuda_device}",
                    f"cuda_probe={report.cuda_probe_ok}",
                    f"native={report.native_extension_ok}",
                ]
            )
        )


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


@app.command("run-c0")
def run_c0_command(
    operation: str,
    left: str,
    right: str,
    expected: str | None = None,
    authority: Annotated[
        list[str] | None,
        typer.Option(
            "--authority",
            help="Authority label to request; repeat for multiple labels.",
        ),
    ] = None,
    risk_privacy: float = typer.Option(0.0, help="Privacy risk charge."),
    risk_capability: float = typer.Option(0.0, help="Capability risk charge."),
    risk_impact: float = typer.Option(0.0, help="Impact risk charge."),
    as_json: bool = typer.Option(False, "--json", help="Print the full JSON report."),
) -> None:
    """Run a supported C0 typed task through the verifier gate."""
    result = run_task(
        typed_task(operation, left, right, expected),
        labels=tuple(authority or ["data"]),
        risk={
            "privacy": risk_privacy,
            "capability": risk_capability,
            "impact": risk_impact,
        },
    )
    if as_json:
        typer.echo(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        typer.echo(result.rendered)


@app.command("train-c0")
def train_c0_command(
    limit: int = typer.Option(8, help="Absolute integer limit for the generated curriculum."),
    epochs: int = typer.Option(80, help="Training epochs."),
    hidden_dim: int = typer.Option(32, help="Recurrent substrate width."),
    device: str = typer.Option("auto", help="Device: auto, cpu, cuda, or cuda:N."),
    artifact: Annotated[
        Path,
        typer.Option(help="NPZ artifact path for learned weights."),
    ] = DEFAULT_PROTOTYPE_ARTIFACT,
    as_json: bool = typer.Option(False, "--json", help="Print the full JSON report."),
) -> None:
    """Train the non-transformer C0 substrate proposal model."""
    _, report = train_c0_prototype(
        TrainingConfig(
            limit=limit,
            epochs=epochs,
            hidden_dim=hidden_dim,
            device=device,
            artifact=artifact,
        )
    )
    if as_json:
        typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        typer.echo(training_summary(report))


@app.command("train-c1")
def train_c1_command(
    limit: int = typer.Option(8, help="Absolute integer limit for the generated curriculum."),
    epochs: int = typer.Option(80, help="Training epochs."),
    hidden_dim: int = typer.Option(32, help="Recurrent substrate width."),
    device: str = typer.Option("auto", help="Device: auto, cpu, cuda, or cuda:N."),
    artifact: Annotated[
        Path,
        typer.Option(help="NPZ artifact path for learned C1 weights."),
    ] = DEFAULT_C1_PROTOTYPE_ARTIFACT,
    as_json: bool = typer.Option(False, "--json", help="Print the full JSON report."),
) -> None:
    """Train the non-transformer substrate on C1 hidden-schema tasks."""
    _, report = train_c1_prototype(
        TrainingConfig(
            limit=limit,
            epochs=epochs,
            hidden_dim=hidden_dim,
            device=device,
            artifact=artifact,
        )
    )
    if as_json:
        typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        typer.echo(training_summary(report))


@app.command("infer-c0")
def infer_c0_command(
    operation: str,
    left: str,
    right: str,
    expected: str | None = None,
    authority: Annotated[
        list[str] | None,
        typer.Option(
            "--authority",
            help="Authority label to request; repeat for multiple labels.",
        ),
    ] = None,
    risk_privacy: float = typer.Option(0.0, help="Privacy risk charge."),
    risk_capability: float = typer.Option(0.0, help="Capability risk charge."),
    risk_impact: float = typer.Option(0.0, help="Impact risk charge."),
    artifact: Annotated[
        Path,
        typer.Option(help="NPZ artifact path from train-c0."),
    ] = DEFAULT_PROTOTYPE_ARTIFACT,
    device: str = typer.Option("auto", help="Device: auto, cpu, cuda, or cuda:N."),
    as_json: bool = typer.Option(False, "--json", help="Print the full JSON report."),
) -> None:
    """Run one C0 task through a learned proposal and verifier gate."""
    model = load_prototype(artifact, device)
    payload = run_learned_task(
        model,
        typed_task(operation, left, right, expected),
        labels=tuple(authority or ["data"]),
        risk={
            "privacy": risk_privacy,
            "capability": risk_capability,
            "impact": risk_impact,
        },
    )
    if as_json:
        typer.echo(json.dumps(payload.to_dict(), indent=2, sort_keys=True))
    else:
        typer.echo(payload.result.rendered)


@app.command("infer-c0-auto")
def infer_c0_auto_command(
    left: str,
    right: str,
    expected: str,
    authority: Annotated[
        list[str] | None,
        typer.Option(
            "--authority",
            help="Authority label to request; repeat for multiple labels.",
        ),
    ] = None,
    risk_privacy: float = typer.Option(0.0, help="Privacy risk charge."),
    risk_capability: float = typer.Option(0.0, help="Capability risk charge."),
    risk_impact: float = typer.Option(0.0, help="Impact risk charge."),
    artifact: Annotated[
        Path,
        typer.Option(help="NPZ artifact path from train-c0."),
    ] = DEFAULT_PROTOTYPE_ARTIFACT,
    device: str = typer.Option("auto", help="Device: auto, cpu, cuda, or cuda:N."),
    as_json: bool = typer.Option(False, "--json", help="Print the full JSON report."),
) -> None:
    """Infer the C0 operation from operands plus expected value, then verify it."""
    model = load_prototype(artifact, device)
    payload = run_learned_task(
        model,
        typed_hidden_task(left, right, expected),
        labels=tuple(authority or ["data"]),
        risk={
            "privacy": risk_privacy,
            "capability": risk_capability,
            "impact": risk_impact,
        },
    )
    if as_json:
        typer.echo(json.dumps(payload.to_dict(), indent=2, sort_keys=True))
    else:
        typer.echo(payload.result.rendered)


@app.command("eval-prototype")
def eval_prototype_command(
    artifact: Annotated[
        Path,
        typer.Option(help="NPZ artifact path from train-c0."),
    ] = DEFAULT_PROTOTYPE_ARTIFACT,
    limit: int = typer.Option(8, help="Absolute integer limit used for held-out generation."),
    device: str = typer.Option("auto", help="Device: auto, cpu, cuda, or cuda:N."),
    as_json: bool = typer.Option(False, "--json", help="Print the full JSON report."),
) -> None:
    """Evaluate a saved prototype against matched baselines."""
    report = evaluate_saved_prototype(artifact, limit, device)
    if as_json:
        typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        typer.echo(prototype_summary(report))


@app.command("eval-c1-prototype")
def eval_c1_prototype_command(
    artifact: Annotated[
        Path,
        typer.Option(help="NPZ artifact path from train-c1."),
    ] = DEFAULT_C1_PROTOTYPE_ARTIFACT,
    limit: int = typer.Option(8, help="Absolute integer limit used for held-out generation."),
    device: str = typer.Option("auto", help="Device: auto, cpu, cuda, or cuda:N."),
    as_json: bool = typer.Option(False, "--json", help="Print the full JSON report."),
) -> None:
    """Evaluate a saved prototype against C1 hidden-schema baselines."""
    report = evaluate_saved_c1_prototype(artifact, limit, device)
    if as_json:
        typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        typer.echo(prototype_summary(report))


@app.command("stages")
def stages_command() -> None:
    """Print curriculum-stage implementation metadata."""
    typer.echo(json.dumps([stage.to_dict() for stage in stages()], indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
