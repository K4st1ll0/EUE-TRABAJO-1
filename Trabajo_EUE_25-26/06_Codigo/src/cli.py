from __future__ import annotations

from pathlib import Path

import typer

from .main import rebuild_docs, run_baseline, run_inspect

app = typer.Typer(help="CLI de la fase 1 para la correlacion FEM E-Box.")


@app.command()
def inspect(config: Path | None = typer.Option(None, help="Ruta al YAML de proyecto.")) -> None:
    result = run_inspect(config)
    for label, value in result.items():
        typer.echo(f"{label}: {value}")


@app.command()
def baseline(config: Path | None = typer.Option(None, help="Ruta al YAML de proyecto.")) -> None:
    result = run_baseline(config)
    for label, value in result.items():
        typer.echo(f"{label}: {value}")


@app.command(name="docs")
def docs_command(config: Path | None = typer.Option(None, help="Ruta al YAML de proyecto.")) -> None:
    result = rebuild_docs(config)
    for label, value in result.items():
        typer.echo(f"{label}: {value}")


if __name__ == "__main__":
    app()
