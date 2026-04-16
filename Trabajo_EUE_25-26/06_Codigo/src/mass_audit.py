from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from .config import LoadedConfigBundle
from .mass_calculator import calculate_total_mass, write_mass_summary
from .report_generator import generate_run_report, rebuild_reports_index


class MassAuditError(RuntimeError):
    """Raised when a functional mass audit cannot be rebuilt safely."""


def _resolve_run_dir(runs_dir: Path, run_id_token: str) -> Path:
    token = run_id_token.strip()
    if not token:
        raise MassAuditError("mass-audit requires a non-empty --run-id value.")

    exact_match = runs_dir / token
    if exact_match.is_dir():
        return exact_match

    prefix_matches = sorted(
        child
        for child in runs_dir.iterdir()
        if child.is_dir() and child.name.startswith(f"{token}_")
    )
    if not prefix_matches:
        raise MassAuditError(f"No run folder matches '{token}' in {runs_dir}.")
    if len(prefix_matches) > 1:
        names = ", ".join(path.name for path in prefix_matches)
        raise MassAuditError(f"Run id '{token}' is ambiguous. Matches: {names}")
    return prefix_matches[0]


def _audit_stem(run_dir: Path) -> str:
    return run_dir.name.split("_", 1)[0]


def _load_existing_audit(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _load_json_mapping(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _normalize_baseline(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {
            "budget_imputation_rows": [],
            "imputation_residual_mass": None,
            "budget_imputation_warnings": [],
            "subset_rows": [],
        }

    warnings = payload.get("budget_imputation_warnings")
    if not isinstance(warnings, list):
        warnings = payload.get("warnings")

    return {
        "budget_imputation_rows": payload.get("budget_imputation_rows", [])
        if isinstance(payload.get("budget_imputation_rows"), list)
        else [],
        "imputation_residual_mass": payload.get("imputation_residual_mass"),
        "budget_imputation_warnings": warnings if isinstance(warnings, list) else [],
        "subset_rows": payload.get("subset_rows", []) if isinstance(payload.get("subset_rows"), list) else [],
    }


def _index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        label = row.get("label")
        if isinstance(label, str):
            indexed[label] = row
    return indexed


def _difference(after: Any, before: Any) -> float | None:
    if before is None or after is None:
        return None
    return float(after) - float(before)


def _build_budget_row_deltas(
    before_rows: list[dict[str, Any]],
    after_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    before_by_label = _index_rows(before_rows)
    after_by_label = _index_rows(after_rows)
    ordered_labels = [row["label"] for row in after_rows if isinstance(row.get("label"), str)]
    ordered_labels.extend(label for label in before_by_label if label not in after_by_label)

    rows: list[dict[str, Any]] = []
    for label in ordered_labels:
        before = before_by_label.get(label, {})
        after = after_by_label.get(label, {})
        rows.append(
            {
                "label": label,
                "target_mass": after.get("target_mass", before.get("target_mass")),
                "before_imputed_mass": before.get("imputed_mass"),
                "after_imputed_mass": after.get("imputed_mass"),
                "imputed_mass_delta": _difference(after.get("imputed_mass"), before.get("imputed_mass")),
                "before_delta_mass": before.get("delta_mass"),
                "after_delta_mass": after.get("delta_mass"),
                "delta_mass_delta": _difference(after.get("delta_mass"), before.get("delta_mass")),
                "before_relative_delta": before.get("relative_delta"),
                "after_relative_delta": after.get("relative_delta"),
            }
        )
    return rows


def _rows_equal(before_rows: list[dict[str, Any]], after_rows: list[dict[str, Any]]) -> bool | None:
    if not before_rows:
        return None
    if len(before_rows) != len(after_rows):
        return False
    numeric_keys = {"target_mass", "computed_mass", "delta_mass", "relative_delta"}
    for before_row, after_row in zip(before_rows, after_rows, strict=True):
        before_keys = set(before_row)
        after_keys = set(after_row)
        if before_keys != after_keys:
            return False
        for key in before_keys:
            before_value = before_row.get(key)
            after_value = after_row.get(key)
            if key in numeric_keys:
                if before_value is None or after_value is None:
                    if before_value != after_value:
                        return False
                    continue
                if abs(float(after_value) - float(before_value)) > 1.0e-12:
                    return False
                continue
            if before_value != after_value:
                return False
    return True


def _build_comparison(
    before_payload: dict[str, Any],
    after_payload: dict[str, Any],
) -> dict[str, Any]:
    before_rows = before_payload["budget_imputation_rows"]
    after_rows = after_payload["budget_imputation_rows"]
    before_warnings = before_payload["budget_imputation_warnings"]
    after_warnings = after_payload["budget_imputation_warnings"]

    return {
        "before": {
            "budget_imputation_rows": before_rows,
            "imputation_residual_mass": before_payload["imputation_residual_mass"],
            "budget_imputation_warnings": before_warnings,
        },
        "after": {
            "budget_imputation_rows": after_rows,
            "imputation_residual_mass": after_payload["imputation_residual_mass"],
            "budget_imputation_warnings": after_warnings,
        },
        "delta": {
            "budget_imputation_rows": _build_budget_row_deltas(before_rows, after_rows),
            "imputation_residual_mass": _difference(
                after_payload["imputation_residual_mass"],
                before_payload["imputation_residual_mass"],
            ),
            "budget_imputation_warnings": {
                "added": sorted(set(after_warnings) - set(before_warnings)),
                "removed": sorted(set(before_warnings) - set(after_warnings)),
            },
        },
    }


def _format_mass(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def _format_delta(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.6f}"


def _format_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:+.3f}%"


def _template_dir_for_bundle(bundle: LoadedConfigBundle) -> Path:
    project_template_dir = bundle.project_root / "src" / "templates"
    if project_template_dir.exists():
        return project_template_dir
    return Path(__file__).resolve().parent / "templates"


def _non_mass_warnings(
    existing_run_warnings: list[Any],
    old_mass_payload: dict[str, Any] | None,
) -> list[str]:
    old_mass_payload = old_mass_payload or {}
    mass_warning_strings = {
        str(item)
        for item in list(old_mass_payload.get("warnings", [])) + list(old_mass_payload.get("budget_imputation_warnings", []))
    }
    warnings: list[str] = []
    for warning in existing_run_warnings:
        if not isinstance(warning, str):
            continue
        if warning in mass_warning_strings:
            continue
        warnings.append(warning)
    return warnings


def _sync_run_payload_mass(
    existing_payload: dict[str, Any],
    summary_payload: dict[str, Any],
) -> dict[str, Any]:
    updated_payload = dict(existing_payload)
    old_mass_payload = existing_payload.get("mass")
    updated_payload["mass"] = summary_payload
    updated_payload["mass_target_basis"] = summary_payload.get("target_basis")
    updated_payload["mass_baseline_status"] = summary_payload.get("baseline_status")
    updated_payload["warnings"] = _non_mass_warnings(
        list(existing_payload.get("warnings", [])),
        old_mass_payload if isinstance(old_mass_payload, dict) else None,
    )
    return updated_payload


def _render_markdown(audit_payload: dict[str, Any]) -> str:
    comparison = audit_payload["comparison"]
    delta_rows = comparison["delta"]["budget_imputation_rows"]
    warnings_delta = comparison["delta"]["budget_imputation_warnings"]
    lines = [
        f"# Functional Budget Imputation Audit - Run {audit_payload['audit_run_label']}",
        "",
        f"Run: `{audit_payload['run_id']}`",
        f"BDF: `{audit_payload['generated_bdf']}`",
        f"Config source: `{audit_payload['config_source']}`",
        f"Config path: `{audit_payload['config_path']}`",
        f"Mass source of truth: `{audit_payload['mass_source_of_truth']}`",
        "",
        "## Methodological Notes",
        "",
    ]
    for note in audit_payload["budget_imputation_notes"]:
        lines.append(f"- {note}")

    lines.extend(
        [
            "",
            "## Legacy Mapping Snapshot",
            "",
            f"- subset_rows invariant vs previous functional audit: `{audit_payload['subset_rows_invariant_vs_before']}`",
            "",
            "## Budget Imputation: Before vs After",
            "",
            "| Category | Target [kg] | Before Imputed [kg] | After Imputed [kg] | Change [kg] | Before Delta [%] | After Delta [%] |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in delta_rows:
        lines.append(
            "| "
            + f"{row['label']} | {_format_mass(row['target_mass'])} | "
            + f"{_format_mass(row['before_imputed_mass'])} | {_format_mass(row['after_imputed_mass'])} | "
            + f"{_format_delta(row['imputed_mass_delta'])} | {_format_percent(row['before_relative_delta'])} | "
            + f"{_format_percent(row['after_relative_delta'])} |"
        )

    lines.extend(
        [
            "",
            "## Residual After Imputation",
            "",
            "| Before [kg] | After [kg] | Change [kg] |",
            "| ---: | ---: | ---: |",
            "| "
            + f"{_format_mass(comparison['before']['imputation_residual_mass'])} | "
            + f"{_format_mass(comparison['after']['imputation_residual_mass'])} | "
            + f"{_format_delta(comparison['delta']['imputation_residual_mass'])} |",
            "",
            "## Functional Warnings",
            "",
            f"- Before: `{comparison['before']['budget_imputation_warnings']}`",
            f"- After: `{comparison['after']['budget_imputation_warnings']}`",
            f"- Added: `{warnings_delta['added']}`",
            f"- Removed: `{warnings_delta['removed']}`",
            "",
            "## Families",
            "",
            "| Family | Contributions | Mass [kg] |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in audit_payload["family_rows"]:
        lines.append(
            f"| {row['label']} | {row['contribution_count']} | {_format_mass(row['computed_mass'])} |"
        )

    lines.extend(
        [
            "",
            "## Allocation Rules Applied",
            "",
            "| Family | Mode | Target | Mass [kg] |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for row in audit_payload["family_budget_allocations"]:
        lines.append(
            f"| {row['family_label']} | {row['mode']} | {row['target_label']} | {_format_mass(row['mass_contribution'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_mass_audit_payload(
    *,
    bundle: LoadedConfigBundle,
    run_dir: Path,
    before_payload: dict[str, Any] | None,
    summary_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    generated_bdf = run_dir / bundle.project.paths.bdf_input.name
    if not generated_bdf.exists():
        raise MassAuditError(f"Generated BDF not found for run '{run_dir.name}': {generated_bdf}")

    if summary_payload is None:
        summary = calculate_total_mass(
            bdf_path=generated_bdf,
            mass_config=bundle.project.mass,
        )
        summary_payload = summary.to_dict()
    normalized_before = _normalize_baseline(before_payload)
    after_payload = {
        "budget_imputation_rows": summary_payload["budget_imputation_rows"],
        "imputation_residual_mass": summary_payload["imputation_residual_mass"],
        "budget_imputation_warnings": list(summary_payload["budget_imputation_warnings"]),
        "subset_rows": summary_payload["subset_rows"],
    }
    comparison = _build_comparison(normalized_before, after_payload)

    return {
        "run_id": run_dir.name,
        "audit_run_label": _audit_stem(run_dir),
        "generated_bdf": str(generated_bdf),
        "config_source": "live project_config.yaml",
        "config_path": str(bundle.source_files["project_config"]),
        "mass_source_of_truth": str(run_dir / "mass_summary.json"),
        "target_basis": summary_payload["target_basis"],
        "target_mass": summary_payload["target_mass"],
        "subset_rows": summary_payload["subset_rows"],
        "subset_warnings": list(summary_payload["warnings"]),
        "subset_rows_invariant_vs_before": _rows_equal(
            normalized_before["subset_rows"],
            summary_payload["subset_rows"],
        ),
        "budget_imputation_rows": summary_payload["budget_imputation_rows"],
        "budget_imputation_warnings": list(summary_payload["budget_imputation_warnings"]),
        "budget_imputation_notes": list(summary_payload["budget_imputation_notes"]),
        "imputation_residual_mass": summary_payload["imputation_residual_mass"],
        "family_rows": summary_payload["family_rows"],
        "family_budget_allocations": summary_payload["family_budget_allocations"],
        "comparison": comparison,
    }


def run_mass_audit(bundle: LoadedConfigBundle, run_id_token: str) -> tuple[Path, Path, dict[str, Any]]:
    run_dir = _resolve_run_dir(bundle.project.paths.runs_dir, run_id_token)
    audit_stem = _audit_stem(run_dir)
    mass_summary_path = run_dir / "mass_summary.json"
    manifest_path = run_dir / "run_manifest.json"
    metrics_path = run_dir / "metrics.json"
    report_path = run_dir / "report.html"
    json_path = bundle.project.paths.reports_dir / f"mass_budget_imputation_audit_{audit_stem}.json"
    md_path = bundle.project.paths.reports_dir / f"mass_budget_imputation_audit_{audit_stem}.md"
    before_payload = _load_existing_audit(json_path)
    summary = calculate_total_mass(
        bdf_path=run_dir / bundle.project.paths.bdf_input.name,
        mass_config=bundle.project.mass,
    )
    write_mass_summary(summary, mass_summary_path)
    summary_payload = summary.to_dict()

    existing_manifest = _load_json_mapping(manifest_path)
    if existing_manifest is not None:
        synced_manifest = _sync_run_payload_mass(existing_manifest, summary_payload)
        manifest_path.write_text(json.dumps(synced_manifest, indent=2), encoding="utf-8")
        metrics_path.write_text(json.dumps(synced_manifest, indent=2), encoding="utf-8")
        generate_run_report(
            run_payload=synced_manifest,
            history=[],
            template_dir=_template_dir_for_bundle(bundle),
            output_path=report_path,
        )
        project_template_dir = bundle.project_root / "src" / "templates"
        if project_template_dir.exists():
            rebuild_reports_index(bundle.project.paths.runs_dir, bundle.project.paths.reports_dir)

    audit_payload = build_mass_audit_payload(
        bundle=bundle,
        run_dir=run_dir,
        before_payload=before_payload,
        summary_payload=summary_payload,
    )
    json_path.write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(audit_payload), encoding="utf-8")
    return json_path, md_path, audit_payload
