from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from itertools import product
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
from .config import LoadedConfigBundle, SweepPresetConfig, write_config_snapshot
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
    """Raised when a workflow is intentionally left pending."""


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


def _collect_parameter_values(
    bundle: LoadedConfigBundle,
    overrides: dict[str, float] | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    overrides = overrides or {}
    parameter_values: dict[str, float] = {}
    parameter_rows: list[dict[str, Any]] = []
    for name, definition in bundle.parameters.parameters.items():
        value = overrides.get(name, definition.require_single_value())
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
    mass_payload = run_payload.get("mass") or {}
    row = {
        "run_id": run_payload["run_id"],
        "command": run_payload["command"],
        "sweep_preset": run_payload.get("sweep_preset"),
        "status": run_payload["status"],
        "accepted": run_payload.get("accepted"),
        "selected_ranking_mode": modal_metrics.get("selected_ranking_mode"),
        "score_legacy": modal_metrics.get("score_legacy"),
        "score_robust": modal_metrics.get("score_robust"),
        "mean_mac": modal_metrics.get("mean_mac"),
        "worst_mac": modal_metrics.get("worst_mac"),
        "mean_relative_frequency_error": modal_metrics.get("mean_relative_frequency_error"),
        "max_relative_frequency_error": modal_metrics.get("max_relative_frequency_error"),
        "bad_pair_count": modal_metrics.get("bad_pair_count"),
        "warning_pair_count": modal_metrics.get("warning_pair_count"),
        "valid_reference_mode_count": modal_metrics.get("valid_reference_mode_count"),
        "valid_model_mode_count": modal_metrics.get("valid_model_mode_count"),
        "paired_valid_mode_count": modal_metrics.get("paired_valid_mode_count"),
        "mass_total_kg": mass_payload.get("total_mass"),
        "mass_target_kg": mass_payload.get("target_mass"),
        "mass_target_basis": mass_payload.get("target_basis"),
        "mass_baseline_status": mass_payload.get("baseline_status"),
        "relative_mass_error": mass_payload.get("relative_error"),
    }
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _build_runs_index_entry(run_payload: dict[str, Any]) -> dict[str, Any]:
    modal_metrics = run_payload.get("modal_metrics") or {}
    mass_payload = run_payload.get("mass") or {}
    ranking_key = modal_metrics.get("ranking_key")
    if not ranking_key:
        mass_relative_error = mass_payload.get("relative_error")
        ranking_key = [] if mass_relative_error is None else [mass_relative_error]
    return {
        "run_id": run_payload["run_id"],
        "created_at": run_payload["created_at"],
        "command": run_payload["command"],
        "status": run_payload["status"],
        "sweep_preset": run_payload.get("sweep_preset"),
        "selected_ranking_mode": modal_metrics.get("selected_ranking_mode"),
        "accepted": run_payload.get("accepted", False),
        "ranking_key": ranking_key,
        "score_robust": modal_metrics.get("score_robust"),
        "score_legacy": modal_metrics.get("score_legacy"),
        "mean_mac": modal_metrics.get("mean_mac"),
        "mean_relative_frequency_error": modal_metrics.get("mean_relative_frequency_error"),
        "bad_pair_count": modal_metrics.get("bad_pair_count"),
        "mass_target_basis": mass_payload.get("target_basis"),
        "mass_baseline_status": mass_payload.get("baseline_status"),
        "report_path": run_payload.get("report_path"),
        "parameter_values": run_payload.get("parameter_values", {}),
    }


def _load_history(runs_dir: Path) -> list[dict[str, Any]]:
    history = []
    for manifest_path in sorted(runs_dir.glob("*/run_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        history.append(_build_runs_index_entry(manifest))
    history.sort(key=lambda item: item.get("created_at", ""))
    return history


def _finalize_run_payload(
    run_payload: dict[str, Any],
    run_dir: Path,
    report_path: Path,
) -> None:
    run_payload["report_path"] = str(report_path)
    modal_metrics = run_payload.get("modal_metrics") or {}
    score_legacy = modal_metrics.get("score_legacy")
    score_robust = modal_metrics.get("score_robust")
    mean_mac = modal_metrics.get("mean_mac")
    run_payload["score_legacy_display"] = "n/a" if score_legacy is None else f"{score_legacy:.6f}"
    run_payload["score_robust_display"] = "n/a" if score_robust is None else f"{score_robust:.6f}"
    run_payload["mean_mac_display"] = "n/a" if mean_mac is None else f"{mean_mac:.4f}"

    generated_files: list[str] = []
    for file_path in sorted(run_dir.rglob("*")):
        if file_path.is_file():
            generated_files.append(str(file_path.relative_to(run_dir)))
    run_payload["generated_files"] = generated_files


def _load_runs_index(index_path: Path) -> list[dict[str, Any]]:
    if not index_path.exists():
        return []
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return []
    entries = payload.get("runs", [])
    return entries if isinstance(entries, list) else []


def _resolve_best_run_for_preset(bundle: LoadedConfigBundle, preset: SweepPresetConfig) -> dict[str, Any]:
    index_path = bundle.project.paths.reports_dir / "runs_index.json"
    entries = _load_runs_index(index_path)
    filtered: list[dict[str, Any]] = []
    for source_preset in preset.select_from_presets:
        candidate_entries = [
            entry
            for entry in entries
            if entry.get("sweep_preset") == source_preset
            and entry.get("selected_ranking_mode") == bundle.project.ranking.mode
        ]
        if candidate_entries:
            filtered = candidate_entries
            break

    if not filtered:
        raise Phase2PendingError(
            "No deterministic sweep source was found in reports/runs_index.json for "
            f"preset '{preset.name}'. Run the source preset first."
        )

    def sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
        ranking_key = item.get("ranking_key") or []
        return tuple(ranking_key) + (item.get("run_id", ""),)

    accepted_entries = [entry for entry in filtered if entry.get("accepted") is True]
    if accepted_entries:
        return sorted(accepted_entries, key=sort_key)[0]
    return sorted(filtered, key=sort_key)[0]


def _load_manifest_for_run(bundle: LoadedConfigBundle, run_id: str) -> dict[str, Any]:
    manifest_path = bundle.project.paths.runs_dir / run_id / "run_manifest.json"
    if not manifest_path.exists():
        raise Phase2PendingError(f"Run manifest not found for deterministic sweep source '{run_id}'.")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _expand_sweep_cases(bundle: LoadedConfigBundle, preset_name: str) -> list[dict[str, float]]:
    if preset_name not in bundle.sweep_presets.presets:
        raise Phase2PendingError(f"Unknown sweep preset '{preset_name}'.")

    preset = bundle.sweep_presets.presets[preset_name]
    base_values, _ = _collect_parameter_values(bundle)
    source_manifest: dict[str, Any] | None = None
    if preset.select_from_presets:
        source_entry = _resolve_best_run_for_preset(bundle, preset)
        source_manifest = _load_manifest_for_run(bundle, source_entry["run_id"])

    seed_values = dict(base_values)
    seed_values.update(preset.fixed_parameters)
    if source_manifest is not None:
        source_parameter_values = source_manifest.get("parameter_values", {})
        for parameter_name in preset.fixed_parameters_from_best_run:
            if parameter_name not in source_parameter_values:
                raise Phase2PendingError(
                    f"Run '{source_manifest.get('run_id')}' does not expose parameter '{parameter_name}'."
                )
            seed_values[parameter_name] = float(source_parameter_values[parameter_name])

    varying_names = list(preset.varying_parameters)
    expanded_values: list[list[float]] = []
    for parameter_name in varying_names:
        spec = preset.varying_parameters[parameter_name]
        center_value = None
        if spec.mode == "centered_range":
            if source_manifest is None:
                raise Phase2PendingError(
                    f"Sweep preset '{preset.name}' requires a deterministic source run to derive the center."
                )
            center_value = float(source_manifest["parameter_values"][parameter_name])
        expanded_values.append(spec.expand(center_value=center_value))

    cases: list[dict[str, float]] = []
    for combination in product(*expanded_values):
        case_values = dict(seed_values)
        for parameter_name, value in zip(varying_names, combination, strict=True):
            case_values[parameter_name] = value
        cases.append(case_values)
    return cases


def _expand_mass_fit_cases(bundle: LoadedConfigBundle) -> list[dict[str, float]]:
    seed_values, _ = _collect_parameter_values(bundle)
    varying_names: list[str] = []
    expanded_values: list[list[float]] = []

    for parameter_name, definition in bundle.parameters.parameters.items():
        if definition.target != "rho":
            seed_values[parameter_name] = definition.require_single_value()
            continue

        if definition.range is not None:
            values = definition.range.expand()
        else:
            values = definition.expand_values()
        varying_names.append(parameter_name)
        expanded_values.append(values)

    if not varying_names:
        return [seed_values]

    cases: list[dict[str, float]] = []
    for combination in product(*expanded_values):
        case_values = dict(seed_values)
        for parameter_name, value in zip(varying_names, combination, strict=True):
            case_values[parameter_name] = value
        cases.append(case_values)
    return cases


def _mass_fit_sort_key(run_payload: dict[str, Any]) -> tuple[Any, ...]:
    mass_payload = run_payload.get("mass") or {}
    subset_rows = mass_payload.get("subset_rows") or []
    subset_abs_delta = 0.0
    for row in subset_rows:
        delta_mass = row.get("delta_mass")
        if delta_mass is None:
            continue
        subset_abs_delta += abs(float(delta_mass))
    relative_error = mass_payload.get("relative_error")
    if relative_error is None:
        relative_error = float("inf")
    baseline_status = mass_payload.get("baseline_status")
    return (
        0 if baseline_status == "active" else 1,
        float(relative_error),
        subset_abs_delta,
        run_payload.get("run_id", ""),
    )


def run_single_case(
    bundle: LoadedConfigBundle,
    command_name: str,
    *,
    parameter_overrides: dict[str, float] | None = None,
    sweep_preset: str | None = None,
) -> dict[str, Any]:
    created_at = datetime.now().astimezone().isoformat()
    parameter_values, parameter_rows = _collect_parameter_values(bundle, parameter_overrides)
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
        mass_config=bundle.project.mass,
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
    warnings: list[str] = list(reference_data.warnings)
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
        warnings.extend(model_data.warnings)
        comparison_result = compare_modal_results(
            reference_data=reference_data,
            model_data=model_data,
            reference_point_order=list(bundle.sensors.reference_point_order),
            model_point_order=list(bundle.sensors.model_node_order),
            num_modes=bundle.project.analysis.num_modes,
            pairing=bundle.project.pairing,
            acceptance=bundle.project.acceptance,
            scoring=bundle.project.scoring,
            ranking=bundle.project.ranking,
            relative_mass_error=mass_summary.relative_error,
            mode_family_labels=bundle.project.mode_family_labels,
        )
        modal_metrics = comparison_result.to_dict()

    parsed_results_payload = {
        "reference": _modal_data_to_serializable(reference_data),
        "model": _modal_data_to_serializable(model_data),
    }
    write_parsed_results(run_dir / "parsed_results.json", parsed_results_payload)

    accepted = bool(modal_metrics and modal_metrics.get("accepted"))
    reject_reasons = [] if modal_metrics is None else list(modal_metrics.get("reject_reasons", []))
    valid_reference_mode_count = reference_data.valid_mode_count
    valid_model_mode_count = 0 if model_data is None else model_data.valid_mode_count
    paired_valid_mode_count = 0 if modal_metrics is None else int(modal_metrics.get("paired_valid_mode_count", 0))

    run_payload: dict[str, Any] = {
        "run_id": run_id,
        "command": command_name,
        "created_at": created_at,
        "status": status,
        "project_name": bundle.project.name,
        "dry_run": bundle.project.analysis.dry_run,
        "sweep_preset": sweep_preset,
        "source_bdf": str(bundle.project.paths.bdf_input),
        "generated_bdf": str(generated_bdf),
        "reference_frequencies_f06": str(bundle.project.paths.reference_frequencies_f06),
        "reference_vectors_f06": str(bundle.project.paths.reference_vectors_f06),
        "config_snapshot_dir": str(snapshot_dir),
        "parameter_values": parameter_values,
        "parameter_rows": parameter_rows,
        "pairing_config": asdict(bundle.project.pairing),
        "acceptance_config": asdict(bundle.project.acceptance),
        "ranking_config": asdict(bundle.project.ranking),
        "material_groups": bundle.parameters.material_groups,
        "modifications": [asdict(item) for item in bdf_result.modifications],
        "mass": mass_summary.to_dict(),
        "mass_target_basis": mass_summary.target_basis,
        "mass_baseline_status": mass_summary.baseline_status,
        "nastran": execution_result.to_dict(),
        "reference": _modal_data_to_serializable(reference_data),
        "model": _modal_data_to_serializable(model_data),
        "modal_metrics": modal_metrics,
        "accepted": accepted,
        "reject_reasons": reject_reasons,
        "valid_reference_mode_count": valid_reference_mode_count,
        "valid_model_mode_count": valid_model_mode_count,
        "paired_valid_mode_count": paired_valid_mode_count,
        "warnings": warnings,
        "errors": errors,
    }

    metrics_json_path = run_dir / "metrics.json"
    metrics_json_path.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
    _write_metrics_csv(run_payload, run_dir / "metrics.csv")

    report_path = run_dir / "report.html"
    run_payload["report_path"] = str(report_path)
    run_payload["generated_files"] = []

    history = _load_history(bundle.project.paths.runs_dir)
    history.append(_build_runs_index_entry(run_payload))
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


def run_mass_fit(bundle: LoadedConfigBundle) -> list[dict[str, Any]]:
    cases = _expand_mass_fit_cases(bundle)
    results = []
    for parameter_values in cases:
        results.append(
            run_single_case(
                bundle,
                "mass-fit",
                parameter_overrides=parameter_values,
            )
        )
    return sorted(results, key=_mass_fit_sort_key)


def run_sweep(bundle: LoadedConfigBundle, preset_name: str) -> list[dict[str, Any]]:
    cases = _expand_sweep_cases(bundle, preset_name)
    results = []
    for parameter_values in cases:
        results.append(
            run_single_case(
                bundle,
                "sweep",
                parameter_overrides=parameter_values,
                sweep_preset=preset_name,
            )
        )
    return results
