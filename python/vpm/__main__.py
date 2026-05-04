"""``python -m vpm`` — top-level CLI entry point.

Wired to ``[project.scripts] vpm = "vpm.__main__:app"`` in
``pyproject.toml``. The CLI surface is intentionally minimal at this
stage; subcommands will be added per milestone in
``docs/implementation/README.md``.
"""

from __future__ import annotations

import typer

from vpm import __version__

app = typer.Typer(
    name="vpm",
    help="VPM-5.3 reference implementation. See docs/architecture/.",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
