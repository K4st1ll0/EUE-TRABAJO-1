from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json
import subprocess
import time


@dataclass(frozen=True)
class NastranExecutionResult:
    command: tuple[str, ...]
    run_dir: Path
    stdout_path: Path
    stderr_path: Path
    outputs: dict[str, Path]
    returncode: int | None
    success: bool
    dry_run: bool
    duration_seconds: float
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["run_dir"] = str(self.run_dir)
        payload["stdout_path"] = str(self.stdout_path)
        payload["stderr_path"] = str(self.stderr_path)
        payload["outputs"] = {key: str(value) for key, value in self.outputs.items()}
        return payload


def _locate_output_files(run_dir: Path, stem: str) -> dict[str, Path]:
    output_extensions = ("f06", "f04", "log", "h5", "op2", "pch")
    located: dict[str, Path] = {}
    for candidate in run_dir.iterdir():
        if not candidate.is_file():
            continue
        suffix = candidate.suffix.lower().lstrip(".")
        if suffix in output_extensions and candidate.stem.lower() == stem.lower():
            located[suffix] = candidate
    return located


def write_execution_summary(result: NastranExecutionResult, path: Path) -> None:
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")


def run_nastran(
    executable: Path,
    bdf_path: Path,
    run_dir: Path,
    arguments: tuple[str, ...],
    dry_run: bool,
) -> NastranExecutionResult:
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / "nastran_stdout.txt"
    stderr_path = logs_dir / "nastran_stderr.txt"
    command = (str(executable), bdf_path.name, *arguments)

    if dry_run:
        stdout_path.write_text(
            "Dry-run enabled. Nastran execution skipped on purpose.\n",
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        return NastranExecutionResult(
            command=command,
            run_dir=run_dir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            outputs={},
            returncode=None,
            success=True,
            dry_run=True,
            duration_seconds=0.0,
        )

    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=run_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    duration_seconds = time.perf_counter() - start

    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")

    outputs = _locate_output_files(run_dir, bdf_path.stem)
    error_message = None
    success = completed.returncode == 0 and "f06" in outputs
    if not success:
        if completed.returncode != 0:
            error_message = f"Nastran exited with code {completed.returncode}."
        elif "f06" not in outputs:
            error_message = "Nastran finished without producing an F06 file."

    return NastranExecutionResult(
        command=command,
        run_dir=run_dir,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        outputs=outputs,
        returncode=completed.returncode,
        success=success,
        dry_run=False,
        duration_seconds=duration_seconds,
        error_message=error_message,
    )

