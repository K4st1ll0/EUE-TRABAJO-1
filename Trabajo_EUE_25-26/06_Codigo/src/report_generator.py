from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import json

from .config import ACTIVE_MASS_BUDGET_BASIS, mass_budget_baseline_status


def _build_mac_cell(value: float | None) -> dict[str, str]:
    if value is None:
        return {
            "value": "n/a",
            "style": "background: #f1f5f9; color: #64748b;",
        }
    value = max(0.0, min(1.0, value))
    low = (247, 239, 228)
    high = (34, 113, 91)
    rgb = tuple(int(low[index] + (high[index] - low[index]) * value) for index in range(3))
    foreground = "#f8fafc" if value > 0.55 else "#13232f"
    return {
        "value": f"{value:.4f}",
        "style": f"background: rgb{rgb}; color: {foreground};",
    }


def _build_frequency_error_cell(value: float | None, gate: float) -> dict[str, str]:
    if value is None:
        return {
            "value": "n/a",
            "style": "background: #f1f5f9; color: #64748b;",
        }
    if gate <= 0:
        normalized = 1.0
    else:
        normalized = min(value / gate, 2.0) / 2.0
    low = (231, 245, 242)
    high = (180, 83, 9)
    rgb = tuple(int(low[index] + (high[index] - low[index]) * normalized) for index in range(3))
    foreground = "#ffffff" if normalized > 0.6 else "#13232f"
    return {
        "value": f"{value:.4f}",
        "style": f"background: rgb{rgb}; color: {foreground};",
    }


def _build_allowed_cell(value: bool) -> dict[str, str]:
    if value:
        return {"value": "yes", "style": "background: #d1fae5; color: #065f46;"}
    return {"value": "no", "style": "background: #fee2e2; color: #991b1b;"}


def _prepare_heatmap_rows(matrix: list[list[Any]], cell_builder: Any) -> list[dict[str, Any]]:
    rows = []
    for index, row in enumerate(matrix, start=1):
        rows.append(
            {
                "reference_mode": index,
                "cells": [cell_builder(value) for value in row],
            }
        )
    return rows


def _prepare_mode_rows(run_payload: dict[str, Any]) -> list[dict[str, Any]]:
    reference = run_payload.get("reference") or {}
    model = run_payload.get("model") or {}
    modal_metrics = run_payload.get("modal_metrics") or {}
    pair_map = {
        int(entry["reference_mode"]): entry for entry in modal_metrics.get("pairing", [])
    }
    reference_frequencies = reference.get("frequencies_hz", {})
    model_frequencies = model.get("frequencies_hz", {})
    reference_valid_modes = set(reference.get("valid_modes", []))
    model_valid_modes = set(model.get("valid_modes", []))
    num_modes = len(reference.get("requested_modes", [])) or len(modal_metrics.get("mac_matrix", [])) or 10

    rows = []
    for mode in range(1, num_modes + 1):
        pairing_entry = pair_map.get(mode)
        model_mode = None if pairing_entry is None else pairing_entry["model_mode"]
        rows.append(
            {
                "reference_mode": mode,
                "reference_valid": mode in reference_valid_modes,
                "model_valid": model_mode in model_valid_modes if model_mode is not None else False,
                "reference_frequency_hz": reference_frequencies.get(mode),
                "model_mode": model_mode,
                "model_frequency_hz": None
                if model_mode is None
                else model_frequencies.get(model_mode),
                "relative_frequency_error": None
                if pairing_entry is None
                else pairing_entry["relative_frequency_error"],
                "mac": None if pairing_entry is None else pairing_entry["mac"],
                "bad_pair": False if pairing_entry is None else pairing_entry["bad_pair"],
                "warning_pair": False if pairing_entry is None else pairing_entry["warning_pair"],
                "outside_gate": False if pairing_entry is None else pairing_entry["outside_gate"],
                "suspicious": False if pairing_entry is None else pairing_entry["suspicious"],
                "reference_family_label": None
                if pairing_entry is None
                else pairing_entry.get("reference_family_label"),
                "model_family_label": None
                if pairing_entry is None
                else pairing_entry.get("model_family_label"),
            }
        )
    return rows


def _prepare_mass_budget_rows(run_payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    mass_payload = run_payload.get("mass") or {}
    budget_comparison = mass_payload.get("budget_comparison") or {}
    legacy_rows = budget_comparison.get("rows")
    if isinstance(legacy_rows, list) and legacy_rows:
        rows = []
        for row in legacy_rows:
            rows.append(
                {
                    "label": row.get("category"),
                    "target_mass": row.get("budget_mass"),
                    "computed_mass": row.get("model_mass"),
                    "delta_mass": row.get("delta_mass"),
                    "relative_delta": row.get("relative_delta"),
                }
            )
        notes = budget_comparison.get("notes")
        return rows, notes if isinstance(notes, list) else []

    rows = []
    for row in mass_payload.get("subset_rows", []):
        rows.append(
            {
                "label": row.get("label"),
                "target_mass": row.get("target_mass"),
                "computed_mass": row.get("computed_mass"),
                "delta_mass": row.get("delta_mass"),
                "relative_delta": row.get("relative_delta"),
            }
        )

    total_mass = mass_payload.get("total_mass")
    target_mass = mass_payload.get("target_mass")
    if total_mass is not None and target_mass is not None:
        rows.append(
            {
                "label": "Total mass",
                "target_mass": target_mass,
                "computed_mass": total_mass,
                "delta_mass": float(total_mass) - float(target_mass),
                "relative_delta": None
                if abs(float(target_mass)) < 1e-12
                else (float(total_mass) - float(target_mass)) / float(target_mass),
            }
        )
    return rows, []


def _load_template(template_dir: Path, template_name: str) -> Any:
    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except ImportError as exc:
        raise RuntimeError(
            "Jinja2 is required to render the HTML report. Install the dependencies from requirements.txt."
        ) from exc

    environment = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return environment.get_template(template_name)


def _to_number_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return [1.0e30]
    numbers: list[float] = []
    for item in value:
        try:
            numbers.append(float(item))
        except (TypeError, ValueError):
            numbers.append(1.0e30)
    return numbers or [1.0e30]


def _extract_mass_target_metadata(payload: dict[str, Any]) -> tuple[str | None, str]:
    mass_payload = payload.get("mass") or {}
    target_basis = mass_payload.get("target_basis")
    if target_basis is None:
        budget_comparison = mass_payload.get("budget_comparison") or {}
        target_basis = budget_comparison.get("basis")
    if target_basis is None:
        target_basis = payload.get("mass_target_basis")
    if target_basis is None:
        snapshot_dir = payload.get("config_snapshot_dir")
        if snapshot_dir:
            snapshot_path = Path(snapshot_dir) / "resolved_snapshot.json"
            if snapshot_path.exists():
                try:
                    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
                    target_basis = (
                        ((snapshot.get("project") or {}).get("mass") or {}).get("budget_basis")
                    )
                except Exception:
                    target_basis = None
    normalized_basis = None if target_basis is None else str(target_basis).lower()
    if normalized_basis is None:
        return None, "unknown"
    baseline_status = mass_payload.get("baseline_status")
    if baseline_status is None:
        baseline_status = mass_budget_baseline_status(normalized_basis)
    return normalized_basis, str(baseline_status)


def _manifest_to_index_entry(manifest: dict[str, Any]) -> dict[str, Any]:
    modal_metrics = manifest.get("modal_metrics") or {}
    mass_target_basis, mass_baseline_status = _extract_mass_target_metadata(manifest)
    mass_payload = manifest.get("mass") or {}
    ranking_key = modal_metrics.get("ranking_key")
    if not ranking_key:
        mass_relative_error = mass_payload.get("relative_error")
        ranking_key = [] if mass_relative_error is None else [mass_relative_error]
    return {
        "run_id": manifest.get("run_id"),
        "created_at": manifest.get("created_at"),
        "command": manifest.get("command"),
        "status": manifest.get("status"),
        "sweep_preset": manifest.get("sweep_preset"),
        "selected_ranking_mode": modal_metrics.get(
            "selected_ranking_mode",
            modal_metrics.get("score_mode"),
        ),
        "accepted": manifest.get("accepted", False),
        "ranking_key": ranking_key,
        "score_robust": modal_metrics.get("score_robust"),
        "score_legacy": modal_metrics.get("score_legacy", modal_metrics.get("score")),
        "mean_mac": modal_metrics.get("mean_mac"),
        "mean_relative_frequency_error": modal_metrics.get("mean_relative_frequency_error"),
        "bad_pair_count": modal_metrics.get("bad_pair_count"),
        "mass_target_basis": mass_target_basis,
        "mass_baseline_status": mass_baseline_status,
        "report_path": manifest.get("report_path"),
        "parameter_values": manifest.get("parameter_values", {}),
    }


def _entry_sort_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(_to_number_list(entry.get("ranking_key"))) + (entry.get("run_id", ""),)


def generate_run_report(
    run_payload: dict[str, Any],
    history: list[dict[str, Any]],
    template_dir: Path,
    output_path: Path,
) -> None:
    template = _load_template(template_dir, "report.html.j2")
    modal_metrics = run_payload.get("modal_metrics") or {}
    freq_gate = ((run_payload.get("pairing_config") or {}).get("freq_gate")) or 0.20
    mass_budget_rows, mass_budget_notes = _prepare_mass_budget_rows(run_payload)
    mac_rows = _prepare_heatmap_rows(modal_metrics.get("mac_matrix", []), _build_mac_cell)
    frequency_rows = _prepare_heatmap_rows(
        modal_metrics.get("frequency_error_matrix", []),
        lambda value: _build_frequency_error_cell(value, freq_gate),
    )
    allowed_rows = _prepare_heatmap_rows(
        modal_metrics.get("allowed_pairs_matrix", []),
        _build_allowed_cell,
    )
    context = {
        "run": run_payload,
        "mode_rows": _prepare_mode_rows(run_payload),
        "mass_budget_rows": mass_budget_rows,
        "mass_budget_notes": mass_budget_notes,
        "mac_heatmap_rows": mac_rows,
        "frequency_error_heatmap_rows": frequency_rows,
        "allowed_pairs_heatmap_rows": allowed_rows,
        "history": history,
        "report_generated_at": run_payload["created_at"],
    }
    output_path.write_text(template.render(**context), encoding="utf-8")


def rebuild_reports_index(runs_dir: Path, reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    template_dir = runs_dir.parent / "src" / "templates"
    manifests = []
    for manifest_path in sorted(runs_dir.glob("*/run_manifest.json")):
        manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))

    entries = [_manifest_to_index_entry(manifest) for manifest in manifests]
    for entry in entries:
        report_path = entry.get("report_path")
        relative_report_path = None
        if report_path:
            try:
                relative_report_path = Path(report_path).relative_to(runs_dir.parent).as_posix()
            except Exception:
                relative_report_path = None
        entry["relative_report_path"] = relative_report_path
    entries.sort(key=lambda item: item.get("created_at", ""), reverse=True)

    ranked_entries = sorted(entries, key=_entry_sort_key)
    best_global = ranked_entries[0] if ranked_entries else None
    accepted_ranked = [entry for entry in ranked_entries if entry.get("accepted") is True]
    best_accepted = accepted_ranked[0] if accepted_ranked else None

    index_payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "active_mass_target_basis": ACTIVE_MASS_BUDGET_BASIS,
        "best_accepted_run_id": None if best_accepted is None else best_accepted.get("run_id"),
        "best_global_run_id": None if best_global is None else best_global.get("run_id"),
        "runs": entries,
    }
    json_path = reports_dir / "runs_index.json"
    json_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")

    template = _load_template(template_dir, "index.html.j2")
    html = template.render(
        runs=entries,
        best_global=best_global,
        best_accepted=best_accepted,
        generated_at=index_payload["generated_at"],
        active_mass_target_basis=index_payload["active_mass_target_basis"],
    )
    index_path = reports_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
    return index_path
