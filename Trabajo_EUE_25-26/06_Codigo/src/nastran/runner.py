from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
from typing import Any

from src.fem.model_checks import DeckCheckResult
from src.utils.hashing import sha256_file
from src.utils.io import write_text
from src.utils.paths import ensure_directory


@dataclass(slots=True)
class BaselineRunLayout:
    run_id: str
    run_dir: Path
    input_dir: Path
    output_dir: Path
    logs_dir: Path
    summary_json: Path
    frequencies_csv: Path
    report_html: Path
    docs_report_html: Path


def next_run_id(runs_root: Path, family: str = "baseline") -> str:
    family_dir = runs_root / family
    ensure_directory(family_dir)
    max_index = 0
    for child in family_dir.iterdir():
        if child.is_dir() and child.name.startswith("run_"):
            try:
                max_index = max(max_index, int(child.name.split("_", 1)[1]))
            except ValueError:
                continue
    return f"run_{max_index + 1:06d}"


def create_run_layout(runs_root: Path, run_id: str) -> BaselineRunLayout:
    run_dir = runs_root / "baseline" / run_id
    input_dir = ensure_directory(run_dir / "input")
    output_dir = ensure_directory(run_dir / "output")
    logs_dir = ensure_directory(run_dir / "logs")
    return BaselineRunLayout(
        run_id=run_id,
        run_dir=run_dir,
        input_dir=input_dir,
        output_dir=output_dir,
        logs_dir=logs_dir,
        summary_json=run_dir / "summary.json",
        frequencies_csv=run_dir / "frequencies.csv",
        report_html=run_dir / "report.html",
        docs_report_html=Path(),
    )


def prepare_baseline_deck(
    base_bdf: Path,
    structure: DeckCheckResult,
    input_dir: Path,
    output_dir: Path,
    requested_modes: int,
) -> dict[str, Any]:
    source_copy = input_dir / "model_source.bdf"
    execution_source = output_dir / "model_source.bdf"
    shutil.copy2(base_bdf, source_copy)
    shutil.copy2(base_bdf, execution_source)
    input_deck_path = input_dir / "baseline_deck.bdf"
    execution_deck_path = output_dir / "baseline_deck.bdf"

    if structure.should_wrap:
        content = "\n".join(
            [
                "$ Auto-generated baseline modal wrapper",
                "SOL 103",
                "CEND",
                "ECHO = NONE",
                "SUBCASE 1",
                "   SUBTITLE = Phase1Baseline",
                "   METHOD = 1",
                "   SPC = 2",
                "BEGIN BULK",
                "PARAM,PRTMAXIM,YES",
                f"EIGRL,1,,,{requested_modes},0,,,MASS",
                "INCLUDE 'model_source.bdf'",
                "SPCADD,2,1",
                "ENDDATA",
                "",
            ]
        )
        write_text(input_deck_path, content)
        write_text(execution_deck_path, content)
    else:
        shutil.copy2(source_copy, input_deck_path)
        shutil.copy2(execution_source, execution_deck_path)

    return {
        "deck_path": execution_deck_path,
        "input_deck_path": input_deck_path,
        "execution_deck_path": execution_deck_path,
        "source_copy": source_copy,
        "execution_source": execution_source,
        "deck_hash": sha256_file(input_deck_path),
        "source_hash": sha256_file(source_copy),
    }


def build_nastran_command(executable: Path | None, deck_path: Path) -> list[str]:
    if executable is None:
        return ["nastran.exe", str(deck_path)]
    return [str(executable), str(deck_path)]


def execute_nastran(
    executable: Path | None,
    deck_path: Path,
    output_dir: Path,
    logs_dir: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    command = build_nastran_command(executable, deck_path)
    prepared_command = subprocess.list2cmdline(command)
    stdout_log = logs_dir / "nastran_stdout.log"
    stderr_log = logs_dir / "nastran_stderr.log"

    if executable is None:
        write_text(stdout_log, "")
        write_text(stderr_log, "No se lanzo Nastran porque no se detecto ejecutable.")
        return {
            "status": "prepared_only",
            "command": prepared_command,
            "return_code": None,
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "launched_at": None,
            "finished_at": None,
        }

    launched_at = datetime.now().isoformat(timespec="seconds")
    completed = subprocess.run(
        command,
        cwd=output_dir,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    finished_at = datetime.now().isoformat(timespec="seconds")
    write_text(stdout_log, completed.stdout)
    write_text(stderr_log, completed.stderr)
    fatal_detected = "USER FATAL" in completed.stdout.upper() or "USER FATAL" in completed.stderr.upper()
    return {
        "status": "completed" if completed.returncode == 0 and not fatal_detected else "failed",
        "command": prepared_command,
        "return_code": completed.returncode,
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "launched_at": launched_at,
        "finished_at": finished_at,
        "fatal_detected": fatal_detected,
    }


def find_run_output_f06(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    candidates = sorted(output_dir.glob("*.f06"))
    return candidates[0] if candidates else None
