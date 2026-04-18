from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import Any

from .bdf_editor import parse_nastran_float


def _parse_mass_from_text(text: str) -> float | None:
    patterns = [
        r"TOTAL MASS\s*=\s*([+\-0-9.DEde]+)",
        r"TOTAL\s+MASS\s+([+\-0-9.DEde]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            mass_value = parse_nastran_float(match.group(1))
            if mass_value is not None:
                return mass_value * 1000.0
    return None


def _file_has_fatal(text: str) -> bool:
    fatal_patterns = (
        r"^\*\*\*\s+USER FATAL MESSAGE",
        r"^\*\*\*\s+SYSTEM FATAL MESSAGE",
        r"JOB TERMINATED DUE TO",
    )
    return any(re.search(pattern, text, re.IGNORECASE | re.MULTILINE) for pattern in fatal_patterns)


def _file_has_success_marker(text: str) -> bool:
    success_markers = ("MSC NASTRAN FINISHED", "* * * END OF JOB * * *", "NSEXIT: EXIT(0)")
    upper_text = text.upper()
    return any(marker in upper_text for marker in success_markers)


def _collect_output_files(project_root: Path, run_dir: Path) -> list[str]:
    files = [path for path in run_dir.iterdir() if path.is_file()]
    return sorted(path.relative_to(project_root).as_posix() for path in files)


def _inspect_generated_outputs(project_root: Path, output_files: list[str]) -> dict[str, Any]:
    parsed_total_mass_g = None
    fatal_files: list[str] = []
    success_marker_files: list[str] = []
    inspected_extensions = {".f06", ".log", ".f04"}

    for relative_path in output_files:
        candidate = project_root / relative_path
        if candidate.suffix.lower() not in inspected_extensions:
            continue
        text = candidate.read_text(encoding="utf-8", errors="ignore")
        if parsed_total_mass_g is None:
            parsed_total_mass_g = _parse_mass_from_text(text)
        if _file_has_fatal(text):
            fatal_files.append(relative_path)
        if _file_has_success_marker(text):
            success_marker_files.append(relative_path)

    return {
        "parsed_total_mass_g": parsed_total_mass_g,
        "fatal_files": fatal_files,
        "success_marker_files": success_marker_files,
    }


def run_nastran(
    config: dict[str, Any],
    run_dir: Path,
    bdf_path: Path,
    run_name: str,
) -> dict[str, Any]:
    runner = config["runner"]
    project_root = config["project_root"]
    command = str(runner["command"]).format(
        bdf=str(bdf_path),
        run_dir=str(run_dir),
        run_name=run_name,
        job_name=str(config["paths"]["job_name"]),
    )
    log_path = run_dir / "runner.log"
    started_at = time.time()

    if runner.get("dry_run", True):
        message = "Dry run enabled in config.runner.dry_run. Nastran was not launched."
        log_path.write_text(message + "\n", encoding="utf-8")
        return {
            "mode": "dry-run",
            "command": command,
            "success": True,
            "exit_code": None,
            "timed_out": False,
            "duration_seconds": round(time.time() - started_at, 3),
            "log_path": log_path.relative_to(project_root).as_posix(),
            "output_files": _collect_output_files(project_root, run_dir),
            "parsed_total_mass_g": None,
            "notes": [message],
        }

    try:
        completed = subprocess.run(
            command,
            cwd=run_dir,
            shell=True,
            capture_output=True,
            text=True,
            timeout=int(runner.get("timeout_seconds", 600)),
        )
        stdout_text = completed.stdout or ""
        stderr_text = completed.stderr or ""
        log_path.write_text(
            f"$ command\n{command}\n\n$ stdout\n{stdout_text}\n\n$ stderr\n{stderr_text}\n",
            encoding="utf-8",
        )
        output_files = _collect_output_files(project_root, run_dir)
        output_inspection = _inspect_generated_outputs(project_root, output_files)
        success = completed.returncode == 0 and not output_inspection["fatal_files"]
        notes: list[str] = []
        if output_inspection["fatal_files"]:
            notes.append("Se detectaron mensajes FATAL en: " + ", ".join(output_inspection["fatal_files"]))
        if not output_inspection["success_marker_files"]:
            notes.append("No se encontro marcador explicito de fin correcto en .log/.f04/.f06.")
        return {
            "mode": "live",
            "command": command,
            "success": success,
            "exit_code": completed.returncode,
            "timed_out": False,
            "duration_seconds": round(time.time() - started_at, 3),
            "log_path": log_path.relative_to(project_root).as_posix(),
            "output_files": output_files,
            "parsed_total_mass_g": output_inspection["parsed_total_mass_g"],
            "notes": notes,
            "fatal_files": output_inspection["fatal_files"],
            "success_marker_files": output_inspection["success_marker_files"],
        }
    except subprocess.TimeoutExpired as exc:
        stdout_text = exc.stdout or ""
        stderr_text = exc.stderr or ""
        timeout_seconds = int(runner.get("timeout_seconds", 600))
        log_path.write_text(
            f"$ command\n{command}\n\n$ timeout\n{timeout_seconds} seconds\n\n"
            f"$ stdout\n{stdout_text}\n\n$ stderr\n{stderr_text}\n",
            encoding="utf-8",
        )
        return {
            "mode": "live",
            "command": command,
            "success": False,
            "exit_code": None,
            "timed_out": True,
            "duration_seconds": round(time.time() - started_at, 3),
            "log_path": log_path.relative_to(project_root).as_posix(),
            "output_files": _collect_output_files(project_root, run_dir),
            "parsed_total_mass_g": None,
            "notes": [f"Nastran timed out after {timeout_seconds} seconds."],
            "fatal_files": [],
            "success_marker_files": [],
        }
