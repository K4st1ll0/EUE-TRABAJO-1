from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
import csv
import json
import re

from .bdf_editor import (
    MaterialFieldUpdate,
    copy_and_apply_mat1_updates,
    write_modifications_json,
)
from .config import LoadedConfigBundle, write_config_snapshot
from .f06_parser import (
    ParsedModalData,
    parse_modal_data,
    parse_reference_modal_data,
    write_parsed_results,
)
from .mass_calculator import calculate_total_mass, write_mass_summary
from .modal_metrics import compare_modal_results
from .nastran_runner import run_nastran, write_execution_summary
from .report_generator import generate_run_report, rebuild_reports_index


class Phase2PendingError(RuntimeError):
    """Raised when a phase-2 workflow is requested in the phase-1 implementation."""


RUN_ID_COMPONENTS = (
    ("Al", "aluminum_density_factor"),
    ("PCB", "pcb_density_factor"),
    ("ALE", "aluminum_E_factor"),
    ("PCBE", "pcb_E_factor"),
)
RUN_SEQUENCE_PATTERN = re.compile(r"^(\d{3})_")


def _format_run_value(value: float) -> str:
    return f"{value:.3f}"


def _build_run_id(parameter_values: dict[str, float]) -> str:
    components = []
    for label, parameter_name in RUN_ID_COMPONENTS:
        if parameter_name not in parameter_values:
            raise KeyError(f"Missing required parameter for run naming: {parameter_name}")
        components.append(f"{label}{_format_run_value(parameter_values[parameter_name])}")
    return "_".join(components)


def _next_run_sequence(runs_dir: Path) -> int:
    highest = 0
    for child in runs_dir.iterdir():
        if not child.is_dir():
            continue
        match = RUN_SEQUENCE_PATTERN.match(child.name)
        if match:
            highest = max(highest, int(match.group(1)))

    next_value = highest + 1
    if next_value > 999:
        raise RuntimeError("Run folder limit reached. This naming scheme only supports prefixes from 001 to 999.")
    return next_value


def _allocate_run_id(runs_dir: Path, parameter_values: dict[str, float]) -> str:
    base_run_id = _build_run_id(parameter_values)
    sequence = _next_run_sequence(runs_dir)
    run_id = f"{sequence:03d}_{base_run_id}"
    if (runs_dir / run_id).exists():
        raise RuntimeError(f"Run folder collision detected for '{run_id}'.")
    return run_id


def _collect_parameter_values(bundle: LoadedConfigBundle) -> tuple[dict[str, float], list[dict[str, Any]]]:
    parameter_values: dict[str, float] = {}
    parameter_rows: list[dict[str, Any]] = []
    for name, definition in bundle.parameters.parameters.items():
        value = definition.require_single_value()
        parameter_values[name] = value
        parameter_rows.append(
            {
                "name": name,
                "group": definition.group,
                "target": definition.target,
                "mode": definition.mode,
                "value": value,
            }
        )
    return parameter_values, parameter_rows


def _build_material_updates(bundle: LoadedConfigBundle, parameter_values: dict[str, float]) -> list[MaterialFieldUpdate]:
    updates: list[MaterialFieldUpdate] = []
    for parameter_name, definition in bundle.parameters.parameters.items():
        factor = parameter_values[parameter_name]
        for mat1_id in bundle.parameters.material_groups[definition.group]:
            updates.append(
                MaterialFieldUpdate(
                    mat1_id=mat1_id,
                    field_name=definition.target,
                    factor=factor,
                    parameter_name=parameter_name,
                    material_group=definition.group,
                )
            )
    return updates


def _modal_data_to_serializable(data: ParsedModalData | None) -> dict[str, Any] | None:
    if data is None:
        return None
    return data.to_dict()


def _write_metrics_csv(run_payload: dict[str, Any], path: Path) -> None:
    modal_metrics = run_payload.get("modal_metrics") or {}
    budget_comparison = (run_payload.get("mass") or {}).get("budget_comparison") or {}
    row = {
        "run_id": run_payload["run_id"],
        "command": run_payload["command"],
        "status": run_payload["status"],
        "score": modal_metrics.get("score"),
        "score_mode": modal_metrics.get("score_mode"),
        "mean_mac": modal_metrics.get("mean_mac"),
        "worst_mac": modal_metrics.get("worst_mac"),
        "mean_relative_frequency_error": modal_metrics.get("mean_relative_frequency_error"),
        "stiffness_fit_indicator": modal_metrics.get("stiffness_fit_indicator"),
        "relative_mass_error": run_payload["mass"]["relative_error"],
        "mass_total_kg": run_payload["mass"]["total_mass"],
        "mass_target_kg": run_payload["mass"]["target_mass"],
        "mass_budget_basis": budget_comparison.get("basis"),
        "mass_budget_total_kg": budget_comparison.get("budget_total_mass"),
        "mass_budget_delta_kg": budget_comparison.get("delta_mass"),
        "mass_budget_relative_delta": budget_comparison.get("relative_delta"),
    }
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _load_history(runs_dir: Path) -> list[dict[str, Any]]:
    history = []
    for manifest_path in sorted(runs_dir.glob("*/run_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        history.append(_summarize_for_history(manifest))
    history.sort(key=lambda item: item.get("created_at", ""))
    return history


def _summarize_for_history(run_payload: dict[str, Any]) -> dict[str, Any]:
    modal_metrics = run_payload.get("modal_metrics") or {}
    return {
        "run_id": run_payload["run_id"],
        "command": run_payload["command"],
        "status": run_payload["status"],
        "created_at": run_payload["created_at"],
        "score": modal_metrics.get("score", 0.0),
        "mean_mac": modal_metrics.get("mean_mac"),
    }


def _finalize_run_payload(
    run_payload: dict[str, Any],
    run_dir: Path,
    report_path: Path,
) -> None:
    run_payload["report_path"] = str(report_path)
    modal_metrics = run_payload.get("modal_metrics") or {}
    score_value = modal_metrics.get("score")
    mean_mac = modal_metrics.get("mean_mac")
    run_payload["score_display"] = "n/a" if score_value is None else f"{score_value:.6f}"
    run_payload["mean_mac_display"] = "n/a" if mean_mac is None else f"{mean_mac:.4f}"

    generated_files: list[str] = []
    for file_path in sorted(run_dir.rglob("*")):
        if file_path.is_file():
            generated_files.append(str(file_path.relative_to(run_dir)))
    run_payload["generated_files"] = generated_files


def run_single_case(bundle: LoadedConfigBundle, command_name: str) -> dict[str, Any]:
    created_at = datetime.now().astimezone().isoformat()
    parameter_values, parameter_rows = _collect_parameter_values(bundle)
    run_id = _allocate_run_id(bundle.project.paths.runs_dir, parameter_values)
    run_dir = bundle.project.paths.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    material_updates = _build_material_updates(bundle, parameter_values)
    generated_bdf = run_dir / bundle.project.paths.bdf_input.name

    snapshot_dir = write_config_snapshot(bundle, run_dir)
    bdf_result = copy_and_apply_mat1_updates(
        source_bdf=bundle.project.paths.bdf_input,
        destination_bdf=generated_bdf,
        updates=material_updates,
    )
    write_modifications_json(bdf_result, run_dir / "modifications.json")

    mass_summary = calculate_total_mass(
        bdf_path=generated_bdf,
        target_mass=bundle.project.mass.target_kg,
        budget_workbook=bundle.project.mass.budget_workbook,
        budget_sheet=bundle.project.mass.budget_sheet,
        budget_basis=bundle.project.mass.budget_basis,
    )
    write_mass_summary(mass_summary, run_dir / "mass_summary.json")

    reference_data = parse_reference_modal_data(
        frequencies_f06=bundle.project.paths.reference_frequencies_f06,
        vectors_f06=bundle.project.paths.reference_vectors_f06,
        reference_point_ids=list(bundle.sensors.reference_point_order),
        num_modes=bundle.project.analysis.num_modes,
    )

    execution_result = run_nastran(
        executable=bundle.project.nastran.executable,
        bdf_path=generated_bdf,
        run_dir=run_dir,
        arguments=bundle.project.nastran.arguments,
        dry_run=bundle.project.analysis.dry_run,
    )
    write_execution_summary(execution_result, run_dir / "nastran_execution.json")

    errors: list[str] = []
    model_data: ParsedModalData | None = None
    modal_metrics = None
    status = "success"

    if execution_result.dry_run:
        status = "dry_run"
    elif not execution_result.success:
        status = "failed"
        if execution_result.error_message:
            errors.append(execution_result.error_message)
    else:
        model_f06 = execution_result.outputs["f06"]
        model_data = parse_modal_data(
            f06_path=model_f06,
            point_ids=list(bundle.sensors.model_node_order),
            num_modes=bundle.project.analysis.num_modes,
        )
        modal_metrics = compare_modal_results(
            reference_frequencies=[
                reference_data.frequencies_hz[index]
                for index in range(1, bundle.project.analysis.num_modes + 1)
            ],
            model_frequencies=[
                model_data.frequencies_hz[index]
                for index in range(1, bundle.project.analysis.num_modes + 1)
            ],
            reference_vectors=reference_data.vectors,
            model_vectors=model_data.vectors,
            reference_point_order=list(bundle.sensors.reference_point_order),
            model_point_order=list(bundle.sensors.model_node_order),
            num_modes=bundle.project.analysis.num_modes,
            frequency_penalty_weight=bundle.project.pairing.frequency_penalty_weight,
            scoring_mode=bundle.project.scoring.mode,
            scoring_weights=asdict(bundle.project.scoring.weights),
            relative_mass_error=mass_summary.relative_error,
        ).to_dict()

    parsed_results_payload = {
        "reference": _modal_data_to_serializable(reference_data),
        "model": _modal_data_to_serializable(model_data),
    }
    write_parsed_results(run_dir / "parsed_results.json", parsed_results_payload)

    run_payload: dict[str, Any] = {
        "run_id": run_id,
        "command": command_name,
        "created_at": created_at,
        "status": status,
        "project_name": bundle.project.name,
        "dry_run": bundle.project.analysis.dry_run,
        "source_bdf": str(bundle.project.paths.bdf_input),
        "generated_bdf": str(generated_bdf),
        "reference_frequencies_f06": str(bundle.project.paths.reference_frequencies_f06),
        "reference_vectors_f06": str(bundle.project.paths.reference_vectors_f06),
        "config_snapshot_dir": str(snapshot_dir),
        "parameter_values": parameter_values,
        "parameter_rows": parameter_rows,
        "material_groups": bundle.parameters.material_groups,
        "modifications": [asdict(item) for item in bdf_result.modifications],
        "mass": mass_summary.to_dict(),
        "nastran": execution_result.to_dict(),
        "reference": _modal_data_to_serializable(reference_data),
        "model": _modal_data_to_serializable(model_data),
        "modal_metrics": modal_metrics,
        "errors": errors,
    }

    metrics_json_path = run_dir / "metrics.json"
    metrics_json_path.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
    _write_metrics_csv(run_payload, run_dir / "metrics.csv")

    report_path = run_dir / "report.html"
    run_payload["report_path"] = str(report_path)
    run_payload["generated_files"] = []

    history = _load_history(bundle.project.paths.runs_dir)
    history.append(_summarize_for_history(run_payload))
    generate_run_report(
        run_payload=run_payload,
        history=history,
        template_dir=bundle.project_root / "src" / "templates",
        output_path=report_path,
    )

    _finalize_run_payload(run_payload, run_dir, report_path)
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
    generate_run_report(
        run_payload=run_payload,
        history=history,
        template_dir=bundle.project_root / "src" / "templates",
        output_path=report_path,
    )
    rebuild_reports_index(bundle.project.paths.runs_dir, bundle.project.paths.reports_dir)
    metrics_json_path.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
    return run_payload


def run_mass_fit(*_: Any, **__: Any) -> None:
    raise Phase2PendingError("mass-fit belongs to phase 2 and is not implemented in this iteration.")


def run_sweep(*_: Any, **__: Any) -> None:
    raise Phase2PendingError("sweep belongs to phase 2 and is not implemented in this iteration.")
