from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from src.correlation.frequency_matcher import compare_modal_frequencies
from src.fem.bdf_parser import inspect_bdf
from src.nastran.f06_parser import parse_real_eigenvalues
from src.nastran.launcher_detection import detect_nastran_executable
from src.nastran.runner import (
    BaselineRunLayout,
    create_run_layout,
    execute_nastran,
    find_run_output_f06,
    next_run_id,
    prepare_baseline_deck,
)
from src.reporting.baseline_report import (
    write_baseline_report,
    write_bdf_summary_html,
    write_project_index,
)
from src.utils.config_loader import ProjectConfig, load_project_config
from src.utils.hashing import sha256_data, sha256_file
from src.utils.io import read_json, write_csv_rows, write_json, write_text
from src.utils.logging_utils import setup_logger
from src.utils.paths import ensure_directory


def _contains_user_fatal(path: Path | None) -> bool:
    if path is None or not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    return "USER FATAL MESSAGE" in text.upper()


def _load_config(config_path: Path | None = None) -> ProjectConfig:
    config = load_project_config(config_path)
    ensure_directory(config.paths.docs_root)
    ensure_directory(config.paths.docs_root / "reports")
    ensure_directory(config.paths.docs_root / "runs")
    ensure_directory(config.paths.runs_root)
    ensure_directory(config.code_root / "reports")
    return config


def _latest_baseline_summaries(config: ProjectConfig) -> list[dict[str, Any]]:
    baseline_root = config.paths.runs_root / "baseline"
    if not baseline_root.exists():
        return []
    results: list[dict[str, Any]] = []
    for summary_path in sorted(baseline_root.glob("run_*/summary.json")):
        try:
            results.append(read_json(summary_path))
        except Exception:
            continue
    return results


def rebuild_docs(config_path: Path | None = None) -> dict[str, str]:
    config = _load_config(config_path)
    inspect_json = config.paths.docs_root / "reports" / "bdf_summary.json"
    if inspect_json.exists():
        inspect_summary = read_json(inspect_json)
        write_bdf_summary_html(inspect_summary, config.paths.docs_root / "reports" / "bdf_summary.html")

    baseline_summaries = _latest_baseline_summaries(config)
    for summary in baseline_summaries:
        docs_run_path = config.paths.docs_root / "runs" / f"baseline_{summary['run_id']}.html"
        write_baseline_report(summary, docs_run_path, docs_copy=True)

    index_path = write_project_index(
        config.paths.docs_root,
        inspect_json if inspect_json.exists() else None,
        baseline_summaries,
        config.reporting.project_status_label,
    )
    return {"docs_index": str(index_path)}


def run_inspect(config_path: Path | None = None) -> dict[str, str]:
    config = _load_config(config_path)
    logger = setup_logger("inspect", config.code_root / "reports" / "inspect.log")
    logger.info("Cargando configuracion desde %s", config.config_path)
    for message in config.validation_messages:
        logger.warning(message)

    inspection = inspect_bdf(config.paths.base_bdf)
    generated_at = datetime.now().isoformat(timespec="seconds")
    summary = inspection.to_dict()
    summary["generated_at"] = generated_at
    summary["paths"] = {
        "base_bdf": str(config.paths.base_bdf),
        "docs_root": str(config.paths.docs_root),
        "config_path": str(config.config_path),
    }
    summary["element_breakdown_rows"] = [
        {"element_type": key, "count": value}
        for key, value in inspection.element_breakdown.items()
    ]
    summary["config_hash"] = sha256_data(config.as_dict())
    if config.validation_messages:
        summary["warnings"].extend(config.validation_messages)

    json_path = config.paths.docs_root / "reports" / "bdf_summary.json"
    csv_path = config.paths.docs_root / "reports" / "bdf_summary.csv"
    html_path = config.paths.docs_root / "reports" / "bdf_summary.html"
    write_json(json_path, summary)
    write_csv_rows(csv_path, inspection.summary_rows())
    write_bdf_summary_html(summary, html_path)
    rebuild_docs(config.config_path)

    logger.info("Resumen BDF generado en %s", html_path)
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "html": str(html_path),
    }


def _build_layout(config: ProjectConfig, run_id: str) -> BaselineRunLayout:
    layout = create_run_layout(config.paths.runs_root, run_id)
    layout.docs_report_html = config.paths.docs_root / "runs" / f"baseline_{run_id}.html"
    return layout


def run_baseline(config_path: Path | None = None) -> dict[str, str]:
    config = _load_config(config_path)
    run_id = next_run_id(config.paths.runs_root)
    layout = _build_layout(config, run_id)
    logger = setup_logger("baseline", layout.logs_dir / "baseline.log")
    logger.info("Preparando baseline %s", run_id)

    errors: list[str] = []
    warnings: list[str] = []
    pending_items: list[str] = []

    try:
        inspection = inspect_bdf(config.paths.base_bdf)
        warnings.extend(inspection.warnings)
        deck_info = prepare_baseline_deck(
            config.paths.base_bdf,
            inspection.structure,
            layout.input_dir,
            layout.output_dir,
            config.solver.requested_modes,
        )
        search = detect_nastran_executable(config.solver.nastran_executable)
        for message in search.messages:
            logger.info(message)
        execution = execute_nastran(
            search.executable,
            deck_info["deck_path"],
            layout.output_dir,
            layout.logs_dir,
            config.solver.timeout_seconds,
        )

        run_f06 = find_run_output_f06(layout.output_dir)
        if run_f06 is not None:
            f06_path = run_f06
            f06_source = "run_output"
        elif config.paths.baseline_f06_fallback.exists():
            f06_path = config.paths.baseline_f06_fallback
            f06_source = "configured_fallback"
            pending_items.append(
                "El run no produjo F06 propio; se uso el fallback definido en YAML."
            )
        else:
            f06_path = None
            f06_source = "missing"
            errors.append("No se encontro F06 del run ni fallback configurado.")

        output_log = layout.output_dir / "baseline_deck.log"
        if _contains_user_fatal(run_f06):
            errors.append("Se detectaron USER FATAL MESSAGE en el F06 del run.")
        if _contains_user_fatal(output_log):
            errors.append("Se detectaron USER FATAL MESSAGE en el log del run.")

        baseline_parse = parse_real_eigenvalues(f06_path) if f06_path else None
        reference_parse = parse_real_eigenvalues(config.paths.reference_modes_f06)
        if baseline_parse and baseline_parse.raw_block:
            write_text(layout.output_dir / "baseline_eigenvalues_raw.txt", baseline_parse.raw_block)
        if reference_parse.raw_block:
            write_text(layout.output_dir / "reference_eigenvalues_raw.txt", reference_parse.raw_block)

        comparison_rows: list[dict[str, Any]] = []
        modal_summary: dict[str, Any] = {"compared_modes": 0}
        if baseline_parse is not None:
            comparison_rows, modal_summary = compare_modal_frequencies(
                baseline_parse, reference_parse, config.comparison.compare_first_modes
            )
            if not baseline_parse.records:
                pending_items.append("El F06 baseline no produjo filas parseables en REAL EIGENVALUES.")
        else:
            pending_items.append("No se pudo comparar frecuencias por falta de baseline F06.")

        write_csv_rows(layout.frequencies_csv, comparison_rows)

        status = "completed"
        if errors:
            status = "failed"
        elif pending_items or warnings or execution["status"] != "completed":
            status = "completed_with_warnings"

        summary = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "run_id": run_id,
            "status": status,
            "deck_strategy": inspection.structure.decision,
            "deck_decision": inspection.structure.messages[0],
            "deck_messages": inspection.structure.messages,
            "paths": {
                "base_bdf": str(config.paths.base_bdf),
                "deck_path": str(deck_info["deck_path"]),
                "input_deck_path": str(deck_info["input_deck_path"]),
                "execution_deck_path": str(deck_info["execution_deck_path"]),
                "input_dir": str(layout.input_dir),
                "output_dir": str(layout.output_dir),
                "reference_modes_f06": str(config.paths.reference_modes_f06),
                "baseline_f06_fallback": str(config.paths.baseline_f06_fallback),
            },
            "hashes": {
                "config_hash": sha256_data(config.as_dict()),
                "base_bdf_hash": sha256_file(config.paths.base_bdf),
                "source_copy_hash": deck_info["source_hash"],
                "deck_hash": deck_info["deck_hash"],
            },
            "nastran_search": search.as_dict(),
            "execution": execution,
            "f06_source": f06_source,
            "baseline_f06_path": str(f06_path) if f06_path else None,
            "reference_f06_path": str(config.paths.reference_modes_f06),
            "baseline_records": [record.as_dict() for record in (baseline_parse.records if baseline_parse else [])],
            "reference_records": [record.as_dict() for record in reference_parse.records],
            "comparison_rows": comparison_rows,
            "modal_summary": modal_summary,
            "warnings": warnings
            + (baseline_parse.warnings if baseline_parse else [])
            + reference_parse.warnings,
            "errors": errors,
            "pending_items": pending_items,
        }

        write_json(layout.summary_json, summary)
        write_baseline_report(summary, layout.report_html, docs_copy=False)
        write_baseline_report(summary, layout.docs_report_html, docs_copy=True)
        rebuild_docs(config.config_path)
        logger.info("Baseline %s listo en %s", run_id, layout.run_dir)
        return {
            "summary": str(layout.summary_json),
            "csv": str(layout.frequencies_csv),
            "report": str(layout.report_html),
            "docs_report": str(layout.docs_report_html),
        }
    except Exception as exc:
        error_summary = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "run_id": run_id,
            "status": "failed",
            "errors": [str(exc)],
            "warnings": warnings,
            "pending_items": pending_items,
            "paths": {"run_dir": str(layout.run_dir)},
            "comparison_rows": [],
            "modal_summary": {"compared_modes": 0},
            "nastran_search": {
                "executable": None,
                "checked_locations": [],
                "candidates": [],
                "messages": [],
                "selected_reason": None,
            },
            "execution": {
                "status": "failed_before_execution",
                "command": None,
                "return_code": None,
            },
            "deck_strategy": "unknown",
            "deck_decision": "Error antes de preparar el deck.",
            "deck_messages": [],
            "f06_source": "missing",
        }
        write_json(layout.summary_json, error_summary)
        write_baseline_report(error_summary, layout.report_html, docs_copy=False)
        write_baseline_report(error_summary, layout.docs_report_html, docs_copy=True)
        rebuild_docs(config.config_path)
        logger.exception("Baseline %s ha fallado", run_id)
        raise
