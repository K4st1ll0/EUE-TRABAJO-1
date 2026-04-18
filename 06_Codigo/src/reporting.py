from __future__ import annotations

import json
import os
from html import escape
from pathlib import Path
from typing import Any

from .config import baseline_factors, now_iso, project_relative
from .mass_model import (
    build_mass_fit_summary,
    calculate_absolute_error_percent,
    calculate_signed_error_percent,
    compute_local_sensitivity,
    estimate_masses,
    score_mass_fit,
)
from .modal_model import build_modal_correlation


STYLE = """
body { background: #f6f4ee; color: #1f2430; font-family: "Segoe UI", Arial, sans-serif; margin: 0; }
.page { max-width: 1200px; margin: 0 auto; padding: 32px 24px 48px; }
h1, h2, h3 { margin: 0 0 12px; color: #18202b; }
h1 { font-size: 30px; }
h2 { font-size: 20px; margin-top: 28px; }
h3 { font-size: 16px; margin-top: 20px; }
p { line-height: 1.5; }
.grid { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin: 18px 0 8px; }
.card { background: #ffffff; border: 1px solid #d7d1c4; border-radius: 12px; padding: 16px 18px; box-shadow: 0 8px 24px rgba(24, 32, 43, 0.05); }
.label { display: block; color: #6c756f; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
.value { font-size: 24px; font-weight: 600; }
.warning { background: #fff0c8; border: 1px solid #d7b964; border-radius: 12px; padding: 16px 18px; margin: 18px 0; }
.muted { color: #5c645f; }
table { width: 100%; border-collapse: collapse; background: #ffffff; border: 1px solid #d7d1c4; border-radius: 12px; overflow: hidden; margin: 12px 0 20px; }
th, td { padding: 11px 12px; border-bottom: 1px solid #e8e2d6; text-align: left; vertical-align: top; }
th { background: #ece4d6; font-size: 13px; color: #3d4651; }
tr.best { background: #eef6ea; }
tr.failed { background: #fff2ef; }
tr.feasible { background: #eef6ea; }
code { background: #efe8db; padding: 2px 6px; border-radius: 6px; }
pre { background: #ffffff; border: 1px solid #d7d1c4; border-radius: 12px; padding: 16px; overflow-x: auto; white-space: pre-wrap; }
a { color: #0a5c70; text-decoration: none; }
a:hover { text-decoration: underline; }
ul { padding-left: 18px; }
.pill { display: inline-block; padding: 4px 8px; border-radius: 999px; font-size: 12px; background: #ece4d6; color: #3c4957; }
.cell-good { background: #e8f3e6; }
.cell-warn { background: #fff5d7; }
.cell-bad { background: #fbe7e3; }
.summary-grid { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); margin: 18px 0 8px; }
.multiline { line-height: 1.45; }
@media (max-width: 720px) {
  .page { padding: 24px 16px 36px; }
  h1 { font-size: 24px; }
  th, td { font-size: 13px; padding: 9px 10px; }
}
"""


def _format_number(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _format_percent_value(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}%"


def _format_bool(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "yes" if value else "no"


def _mass_error_cell_class(abs_error_percent: float | None) -> str:
    if abs_error_percent is None:
        return ""
    if float(abs_error_percent) <= 5.0 + 1e-9:
        return "cell-good"
    if float(abs_error_percent) <= 10.0 + 1e-9:
        return "cell-warn"
    return "cell-bad"


def _relpath(from_dir: Path, target: Path) -> str:
    return Path(os.path.relpath(target, start=from_dir)).as_posix()


def _status_priority(status: str | None) -> int:
    order = {
        "completed": 0,
        "dry_run_completed": 1,
        "failed": 2,
    }
    return order.get(str(status), 9)


def _latest_baseline_run(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    baseline_runs = [run for run in runs if run.get("is_baseline_reference")]
    if not baseline_runs:
        return None
    return max(baseline_runs, key=lambda item: (int(item.get("run_index", 0)), -_status_priority(item.get("status"))))


def _load_generated_bdf_text(config: dict[str, Any], run: dict[str, Any]) -> str | None:
    generated_bdf = run.get("generated_bdf")
    if not generated_bdf:
        return None
    candidate = Path(config["project_root"]) / str(generated_bdf)
    if not candidate.exists():
        return None
    return candidate.read_text(encoding="utf-8", errors="ignore")


def _is_modal_run(run: dict[str, Any]) -> bool:
    strategy = str(run.get("strategy") or "")
    return strategy.startswith("modal_") or "modal_request" in run or "modal" in run


def _find_output_file(run: dict[str, Any], suffixes: list[str], config: dict[str, Any]) -> Path | None:
    for suffix in suffixes:
        for relative_path in run.get("output_files", []):
            candidate = Path(config["project_root"]) / str(relative_path)
            if candidate.suffix.lower() == str(suffix).lower() and candidate.exists():
                return candidate
    return None


def _refresh_run_modal_fields(config: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    if not _is_modal_run(payload):
        return payload

    model_result_path = _find_output_file(
        payload,
        [str(item) for item in config["modal"].get("result_source_preference", [".f06"])],
        config,
    )
    if model_result_path is None:
        payload["modal"] = {
            "status": "pending",
            "base_mass_run_name": (payload.get("modal_request") or {}).get("base_mass_run_name", config["modal"]["base_mass_run_name"]),
            "parser_notes": ["No se encontro aun un resultado modal parseable segun modal.result_source_preference."],
        }
        payload["modal_score"] = None
        return payload

    reference_path = Path(config["resolved_paths"]["modal_reference_file"])
    try:
        reference_text = reference_path.read_text(encoding="utf-8", errors="ignore")
        model_text = model_result_path.read_text(encoding="utf-8", errors="ignore")
        modal_payload = build_modal_correlation(
            reference_text=reference_text,
            model_text=model_text,
            config=config,
            reference_source_name=project_relative(config, reference_path),
            model_source_name=project_relative(config, model_result_path),
        )
        modal_payload["base_mass_run_name"] = (payload.get("modal_request") or {}).get(
            "base_mass_run_name",
            config["modal"]["base_mass_run_name"],
        )
        modal_payload["reference_source"] = {
            "path": project_relative(config, reference_path),
            "source_type": str(config["modal"]["reference"]["source_type"]),
            "rationale": str(config["modal"]["reference"].get("rationale", "")),
            "notes": list(config["modal"]["reference"].get("notes", [])),
        }
        modal_payload["model_output_file"] = project_relative(config, model_result_path)
        payload["modal"] = modal_payload
        payload["modal_score"] = modal_payload["summary"]["modal_score"]
        payload["modal_score_components"] = modal_payload["summary"]["modal_score_components"]
    except Exception as exc:
        payload["modal"] = {
            "status": "failed",
            "base_mass_run_name": (payload.get("modal_request") or {}).get("base_mass_run_name", config["modal"]["base_mass_run_name"]),
            "reference_source": {
                "path": project_relative(config, reference_path),
                "source_type": str(config["modal"]["reference"]["source_type"]),
            },
            "model_output_file": project_relative(config, model_result_path),
            "exception": f"{type(exc).__name__}: {exc}",
            "parser_notes": [
                "El parser modal no pudo completar la correlacion para esta run.",
            ],
        }
        payload["modal_score"] = None
    return payload


def _build_mass_error_payload(
    config: dict[str, Any],
    score_payload: dict[str, Any],
    mass_summary: dict[str, Any],
    parsed_nastran_mass_g: float | None,
) -> dict[str, Any]:
    internal_total = float(mass_summary["total_mass_g"])
    target_total_mass_g = float(config["targets"]["total_mass_g"])
    total_mass_signed_error_percent = calculate_signed_error_percent(internal_total, target_total_mass_g)
    total_mass_absolute_error_percent = calculate_absolute_error_percent(total_mass_signed_error_percent)
    return {
        "target_total_mass_g": target_total_mass_g,
        "target_mass_budget_sum_g": float(config["targets"]["mass_budget_target_sum_g"]),
        "target_mass_budget_minus_total_g": float(config["targets"]["mass_budget_target_total_delta_g"]),
        "total_mass_error_g": score_payload["total_mass_error_g"],
        "total_mass_rel_error": score_payload["components"]["total_mass_rel_error"],
        "total_mass_signed_error_percent": total_mass_signed_error_percent,
        "total_mass_absolute_error_percent": total_mass_absolute_error_percent,
        "mass_budget_error_g": score_payload["mass_budget_error_g"],
        "mass_budget_rel_errors": score_payload["mass_budget_rel_errors"],
        "parsed_nastran_total_mass_g": parsed_nastran_mass_g,
        "internal_vs_parsed_nastran_delta_g": None
        if parsed_nastran_mass_g is None
        else round(parsed_nastran_mass_g - internal_total, 6),
    }


def _refresh_run_mass_fields(config: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    text = _load_generated_bdf_text(config, payload)
    if text is None:
        return payload

    factors = payload.get("parameter_factors") or baseline_factors(config)
    mass_summary = estimate_masses(text, config)
    mass_fit_summary = build_mass_fit_summary(
        mass_summary,
        config,
        factors=factors,
        baseline_reference_factors=baseline_factors(config),
    )
    parsed_nastran_mass_g = payload.get("nastran", {}).get("parsed_total_mass_g")
    if parsed_nastran_mass_g is None:
        parsed_nastran_mass_g = payload.get("mass_errors", {}).get("parsed_nastran_total_mass_g")

    nastran_failed = payload.get("status") == "failed" and payload.get("nastran", {}).get("mode") == "live"
    score_payload = score_mass_fit(
        mass_summary,
        config,
        factors=factors,
        baseline_reference_factors=baseline_factors(config),
        mass_fit_summary=mass_fit_summary,
        nastran_failed=nastran_failed,
    )

    payload["internal_mass"] = mass_summary
    payload["mass_fit"] = mass_fit_summary
    payload["masses_by_material_g"] = mass_summary["masses_by_material_g"]
    payload["model_mass_contributions"] = mass_summary["model_mass_contributions"]
    payload["mass_budget_comparison"] = mass_summary["mass_budget_comparison"]
    payload["mass_budget_comparison_rows"] = mass_fit_summary["mass_budget_comparison_rows"]
    payload["adjusted_material_mass_rows"] = mass_fit_summary["adjusted_material_mass_rows"]
    payload["direct_reference_mass_g"] = mass_summary["direct_reference_mass_g"]
    payload["shared_mass_g"] = mass_summary["shared_mass_g"]
    payload["unassigned_mass_g"] = mass_summary["unassigned_mass_g"]
    payload["unassigned_materials_g"] = mass_summary["unassigned_materials_g"]
    payload["total_mass_g"] = mass_summary["total_mass_g"]
    payload["total_mass_signed_error_percent"] = mass_fit_summary["total_mass_signed_error_percent"]
    payload["total_mass_absolute_error_percent"] = mass_fit_summary["total_mass_absolute_error_percent"]
    payload["score"] = score_payload["score"]
    payload["score_components"] = score_payload["components"]
    payload["mass_errors"] = _build_mass_error_payload(config, score_payload, mass_summary, parsed_nastran_mass_g)
    payload["cross_checks"] = {
        "internal_total_mass_g": mass_summary["total_mass_g"],
        "parsed_nastran_total_mass_g": parsed_nastran_mass_g,
        "internal_vs_parsed_nastran_delta_g": payload["mass_errors"]["internal_vs_parsed_nastran_delta_g"],
    }
    payload["target_total_mass_g"] = float(config["targets"]["total_mass_g"])
    return payload


def _normalize_run_record(config: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    payload = dict(run)
    allowed_mass_keys = {
        "masses_by_material_g",
    }
    for legacy_key in list(payload):
        if legacy_key.startswith("masses_by_") and legacy_key.endswith("_g") and legacy_key not in allowed_mass_keys:
            payload.pop(legacy_key, None)
    payload.pop("free_total_closure_materials_g", None)
    nastran_mode = payload.get("nastran", {}).get("mode")
    if nastran_mode == "dry-run":
        payload["status"] = "dry_run_completed"
    elif payload.get("status") == "dry_run_completed" and nastran_mode != "dry-run":
        payload["status"] = "completed"

    payload = _refresh_run_mass_fields(config, payload)
    payload = _refresh_run_modal_fields(config, payload)
    modal = payload.get("modal") or {}
    payload["paired_mode_diagnostics"] = modal.get("paired_mode_diagnostics", [])
    payload["problematic_modes"] = modal.get("problematic_modes", {})

    total_mass_g = payload.get("total_mass_g")
    target_total_g = payload.get("target_total_mass_g") or config["targets"]["total_mass_g"]
    if total_mass_g is not None and target_total_g:
        total_rel_error = abs(float(total_mass_g) - float(target_total_g)) / float(target_total_g)
        total_mass_signed_error_percent = calculate_signed_error_percent(float(total_mass_g), float(target_total_g))
        total_mass_absolute_error_percent = calculate_absolute_error_percent(total_mass_signed_error_percent)
        payload.setdefault("mass_errors", {})
        payload["mass_errors"].setdefault("target_total_mass_g", float(target_total_g))
        payload["mass_errors"].setdefault("total_mass_error_g", round(float(total_mass_g) - float(target_total_g), 6))
        payload["mass_errors"].setdefault("total_mass_rel_error", round(total_rel_error, 8))
        payload["mass_errors"].setdefault("total_mass_signed_error_percent", total_mass_signed_error_percent)
        payload["mass_errors"].setdefault("total_mass_absolute_error_percent", total_mass_absolute_error_percent)
        payload.setdefault("total_mass_signed_error_percent", total_mass_signed_error_percent)
        payload.setdefault("total_mass_absolute_error_percent", total_mass_absolute_error_percent)

    if payload.get("unassigned_mass_g") is None:
        unassigned_materials = payload.get("unassigned_materials_g") or {}
        payload["unassigned_mass_g"] = round(sum(float(value) for value in unassigned_materials.values()), 6)

    payload.setdefault("mass_budget_comparison_rows", list(payload.get("mass_budget_comparison", [])))
    payload.setdefault("adjusted_material_mass_rows", [])

    return payload


def load_run_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    runs_dir = Path(config["resolved_paths"]["runs_dir"])
    records: list[dict[str, Any]] = []
    for run_json in sorted(runs_dir.glob("*/run.json")):
        try:
            with run_json.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError:
            continue
        payload["_source_path"] = str(run_json)
        records.append(_normalize_run_record(config, payload))
    records.sort(key=lambda item: int(item.get("run_index", 10**9)))
    return records


def _mass_fit_rank(run: dict[str, Any]) -> tuple[float, float, float, float]:
    ranking = (run.get("mass_fit") or {}).get("ranking_key")
    if ranking:
        return (float(ranking[0]), float(ranking[1]), float(ranking[2]), float(ranking[3]))
    return (1e9, 1e9, 1e9, 1e9)


def _modal_rank_strict(run: dict[str, Any]) -> tuple[float, float, float, float, float]:
    modal = run.get("modal") or {}
    summary = modal.get("summary") or {}
    status_priority = 0.0 if modal.get("status") == "completed" else 1.0
    return (
        status_priority,
        -float(summary.get("mean_mac", 0.0)),
        float(summary.get("mean_frequency_error", 1e9)),
        -float(summary.get("worst_mac", 0.0)),
        float(summary.get("worst_frequency_error", 1e9)),
    )


def _modal_rank_balanced(run: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    modal = run.get("modal") or {}
    summary = modal.get("summary") or {}
    status_priority = 0.0 if modal.get("status") == "completed" else 1.0
    return (
        status_priority,
        -float(summary.get("good_on_both_count", 0.0)),
        float(summary.get("mean_frequency_error", 1e9)),
        -float(summary.get("mean_mac", 0.0)),
        -float(summary.get("worst_mac", 0.0)),
        float(summary.get("worst_frequency_error", 1e9)),
    )


def _modal_rank(run: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return _modal_rank_strict(run)


def _changed_youngs_factors(config: dict[str, Any], run: dict[str, Any]) -> dict[str, float]:
    factors = run.get("youngs_modulus_factors") or {}
    changed: dict[str, float] = {}
    for material in config["materials"]:
        name = str(material["name"])
        factor = float(factors.get(name, 1.0))
        if abs(factor - 1.0) > 1e-9:
            changed[name] = round(factor, 6)
    return changed


def _changed_youngs_summary(config: dict[str, Any], run: dict[str, Any]) -> str:
    changed = _changed_youngs_factors(config, run)
    if not changed:
        return "baseline E"
    return ", ".join(f"{name}={value:.3f}" for name, value in changed.items())


def _summarize_run(config: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    mass_fit = run.get("mass_fit") or {}
    modal = run.get("modal") or {}
    modal_summary = modal.get("summary") or {}
    modal_fit = run.get("modal_fit") or {}
    modal_local = run.get("modal_local_refinement") or {}
    modal_balanced_local = run.get("modal_balanced_local_refinement") or {}
    modal_targeted_fit = run.get("modal_targeted_fit") or {}
    most_limiting_variable = mass_fit.get("most_limiting_variable") or {}
    return {
        "run_name": run.get("run_name"),
        "run_index": run.get("run_index"),
        "timestamp": run.get("timestamp"),
        "status": run.get("status"),
        "strategy": run.get("strategy"),
        "is_baseline_reference": run.get("is_baseline_reference", False),
        "score": run.get("score"),
        "total_mass_g": run.get("total_mass_g"),
        "total_mass_rel_error": run.get("mass_errors", {}).get("total_mass_rel_error"),
        "total_mass_signed_error_percent": run.get("mass_errors", {}).get("total_mass_signed_error_percent"),
        "total_mass_absolute_error_percent": run.get("mass_errors", {}).get("total_mass_absolute_error_percent"),
        "feasible": mass_fit.get("feasible"),
        "constraints_satisfied_count": mass_fit.get("constraints_satisfied_count"),
        "constraint_count": mass_fit.get("constraint_count"),
        "most_limiting_variable": most_limiting_variable.get("material_name"),
        "unassigned_mass_g": run.get("unassigned_mass_g"),
        "shared_mass_g": run.get("shared_mass_g"),
        "mass_budget_comparison_rows": run.get("mass_budget_comparison_rows", []),
        "adjusted_material_mass_rows": run.get("adjusted_material_mass_rows", []),
        "nastran_mode": run.get("nastran", {}).get("mode"),
        "report_path": run.get("report_paths", {}).get("master"),
        "run_json_path": run.get("run_json_path"),
        "modal_status": modal.get("status"),
        "modal_score": run.get("modal_score"),
        "modal_mean_mac": modal_summary.get("mean_mac"),
        "modal_worst_mac": modal_summary.get("worst_mac"),
        "modal_mean_frequency_error": modal_summary.get("mean_frequency_error"),
        "modal_worst_frequency_error": modal_summary.get("worst_frequency_error"),
        "modal_good_on_both_count": modal_summary.get("good_on_both_count"),
        "modal_base_mass_run_name": modal.get("base_mass_run_name"),
        "youngs_modulus_factors": run.get("youngs_modulus_factors", {}),
        "changed_youngs_modulus_factors": _changed_youngs_factors(config, run),
        "changed_youngs_summary": _changed_youngs_summary(config, run),
        "modal_fit_candidate_material": modal_fit.get("candidate_material_name"),
        "modal_fit_candidate_e_factor": modal_fit.get("candidate_e_factor"),
        "modal_fit_method": modal_fit.get("method"),
        "modal_local_stage": modal_local.get("stage"),
        "modal_local_rank_overall": modal_local.get("rank_overall"),
        "modal_local_rank_within_stage": modal_local.get("rank_within_stage"),
        "modal_local_candidate_values": modal_local.get("candidate_values", {}),
        "modal_local_method": modal_local.get("method"),
        "modal_balanced_local_rank_overall": modal_balanced_local.get("rank_overall"),
        "modal_balanced_local_candidate_values": modal_balanced_local.get("candidate_values", {}),
        "modal_balanced_local_method": modal_balanced_local.get("method"),
        "modal_targeted_rank_overall": modal_targeted_fit.get("rank_overall"),
        "modal_targeted_candidate_values": modal_targeted_fit.get("candidate_values", {}),
        "modal_targeted_method": modal_targeted_fit.get("method"),
        "modal_targeted_moves": modal_targeted_fit.get("moves", []),
        "paired_mode_diagnostics": run.get("paired_mode_diagnostics", []),
        "problematic_modes": run.get("problematic_modes", {}),
        "modal_local_sensitivity": run.get("modal_local_sensitivity"),
        "targeted_recommendation": run.get("targeted_recommendation"),
        "best_targeted_modal_run": run.get("best_targeted_modal_run"),
        "exception": run.get("exception"),
        "notes": run.get("notes", []),
    }


def select_best_run(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not runs:
        return None
    return min(
        runs,
        key=lambda item: (
            _status_priority(item.get("status")),
            0 if (item.get("mass_fit") or {}).get("feasible") else 1,
            *_mass_fit_rank(item),
            -int(item.get("run_index", 0)),
        ),
    )


def _select_best_feasible_run(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    feasible_runs = [run for run in runs if (run.get("mass_fit") or {}).get("feasible")]
    if not feasible_runs:
        return None
    non_modal_feasible_runs = [run for run in feasible_runs if not _is_modal_run(run)]
    if non_modal_feasible_runs:
        feasible_runs = non_modal_feasible_runs
    return min(
        feasible_runs,
        key=lambda item: (_status_priority(item.get("status")), *_mass_fit_rank(item), -int(item.get("run_index", 0))),
    )


def _select_best_modal_run(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    modal_runs = [run for run in runs if (run.get("modal") or {}).get("status") == "completed"]
    if not modal_runs:
        return None
    return min(modal_runs, key=_modal_rank_strict)


def _select_best_modal_run_balanced(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    modal_runs = [run for run in runs if (run.get("modal") or {}).get("status") == "completed"]
    if not modal_runs:
        return None
    return min(modal_runs, key=_modal_rank_balanced)


def _build_baseline_sensitivity(config: dict[str, Any], baseline_run: dict[str, Any] | None) -> dict[str, Any] | None:
    if baseline_run is None:
        return None
    text = _load_generated_bdf_text(config, baseline_run)
    if text is None:
        return None
    return compute_local_sensitivity(text, config)


def _frequency_error_direction(improvement_absolute: float | None) -> str:
    if improvement_absolute is None:
        return "frequency error change unavailable"
    if float(improvement_absolute) > 1e-9:
        return "frequency error decreased"
    if float(improvement_absolute) < -1e-9:
        return "frequency error increased"
    return "frequency error unchanged"


def _modal_selection_view(runs: list[dict[str, Any]]) -> dict[str, Any]:
    strict_run = _select_best_modal_run(runs)
    balanced_run = _select_best_modal_run_balanced(runs)
    recommended_run = balanced_run or strict_run
    reason = "No modal runs completed yet."
    tradeoff = "No modal trade-off available yet."
    next_step = "Execute modal-baseline first."

    if strict_run and balanced_run:
        strict_name = str(strict_run.get("run_name"))
        balanced_name = str(balanced_run.get("run_name"))
        if strict_name == balanced_name:
            reason = f"Strict and balanced rankings agree on {strict_name}."
            tradeoff = "No conflict between MAC maximization and frequency balance in the current modal set."
            next_step = "Use this run as the next modal reference for further local refinement."
        else:
            strict_summary = (strict_run.get("modal") or {}).get("summary", {})
            balanced_summary = (balanced_run.get("modal") or {}).get("summary", {})
            reason = (
                f"Recommended run is {balanced_name} because it keeps good_on_both="
                f"{int(balanced_summary.get('good_on_both_count', 0))} with mean frequency error "
                f"{float(balanced_summary.get('mean_frequency_error', 0.0)):.6f}, while strict ranking still prefers {strict_name} for higher mean MAC."
            )
            tradeoff = (
                f"{strict_name} pushes mean MAC to {float(strict_summary.get('mean_mac', 0.0)):.6f}, "
                f"but {balanced_name} preserves a more balanced frequency error of "
                f"{float(balanced_summary.get('mean_frequency_error', 0.0)):.6f}."
            )
            next_step = f"Refine locally around {balanced_name} rather than around the strict-only candidate."
    elif recommended_run:
        recommended_name = str(recommended_run.get("run_name"))
        reason = f"Only one modal ranking candidate is available, so {recommended_name} is recommended by default."
        tradeoff = "There are not enough modal runs yet to expose a ranking trade-off."
        next_step = f"Use {recommended_name} as the reference if you want a further local modal refinement."

    return {
        "best_modal_run_strict": strict_run,
        "best_modal_run_balanced": balanced_run,
        "recommended_modal_run": recommended_run,
        "reason": reason,
        "tradeoff": tradeoff,
        "next_step": next_step,
    }


def _diagnostic_modal_run(runs: list[dict[str, Any]], fallback_run: dict[str, Any] | None) -> dict[str, Any] | None:
    ordered_runs = sorted(runs, key=lambda item: int(item.get("run_index", 0)), reverse=True)
    for run in ordered_runs:
        if run.get("targeted_recommendation"):
            return run
    for run in ordered_runs:
        sensitivity = run.get("modal_local_sensitivity") or {}
        if sensitivity.get("rows"):
            return run
    return fallback_run


def _control_definitions_snapshot(config: dict[str, Any]) -> list[dict[str, Any]]:
    snapshot: list[dict[str, Any]] = []
    for control_key, control_cfg in config["mass_fit_controls"].items():
        target_cfg = config["mass_fit_targets"][control_key]
        snapshot.append(
            {
                "control_key": control_key,
                "label": target_cfg["label"],
                "direct_materials": list(control_cfg["direct_materials"]),
                "shared_materials": list(control_cfg["shared_materials"]),
            }
        )
    snapshot.append(
        {
            "control_key": "total",
            "label": config["mass_fit_targets"]["total"]["label"],
            "direct_materials": [],
            "shared_materials": [],
        }
    )
    return snapshot


def _targets_with_bands_snapshot(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for control_key, target_cfg in config["mass_fit_targets"].items():
        target_g = float(target_cfg["target_g"])
        tol = float(target_cfg["tolerance_fraction"])
        rows.append(
            {
                "control_key": control_key,
                "label": target_cfg["label"],
                "target_g": target_g,
                "lower_bound_g": round(target_g * (1.0 - tol), 6),
                "upper_bound_g": round(target_g * (1.0 + tol), 6),
                "tolerance_fraction": tol,
            }
        )
    return rows


def build_ia_payload(
    config: dict[str, Any],
    runs: list[dict[str, Any]],
    baseline_sensitivity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    best_run = select_best_run(runs)
    best_feasible_run = _select_best_feasible_run(runs)
    modal_selection = _modal_selection_view(runs)
    best_modal_run = modal_selection["best_modal_run_strict"]
    best_modal_run_balanced = modal_selection["best_modal_run_balanced"]
    recommended_modal_run = modal_selection["recommended_modal_run"]
    diagnostic_modal_run = _diagnostic_modal_run(runs, recommended_modal_run)
    diagnostic_modal_run = next(
        (
            run
            for run in sorted(runs, key=lambda item: int(item.get("run_index", 0)), reverse=True)
            if run.get("modal_local_sensitivity") or run.get("targeted_recommendation")
        ),
        recommended_modal_run,
    )
    baseline_run = _latest_baseline_run(runs)
    discrepancy = float(config["targets"]["mass_budget_target_total_delta_g"])
    failed_runs = [run for run in runs if run.get("status") == "failed"]
    best_partial_run = best_run if best_feasible_run is None else None
    modal_runs = [run for run in runs if _is_modal_run(run)]
    modal_completed_runs = [run for run in modal_runs if (run.get("modal") or {}).get("status") == "completed"]
    modal_fit_runs = sorted(
        [
            run
            for run in modal_runs
            if run.get("strategy") == "modal_fit" and (run.get("modal") or {}).get("status") == "completed"
        ],
        key=_modal_rank_strict,
    )
    modal_local_runs = sorted(
        [
            run
            for run in modal_runs
            if run.get("strategy") == "modal_fit_local" and (run.get("modal") or {}).get("status") == "completed"
        ],
        key=_modal_rank_strict,
    )
    balanced_local_runs = sorted(
        [
            run
            for run in modal_runs
            if (run.get("modal_balanced_local_refinement") or {}) and (run.get("modal") or {}).get("status") == "completed"
        ],
        key=_modal_rank_balanced,
    )
    targeted_runs = sorted(
        [
            run
            for run in modal_runs
            if (run.get("modal_targeted_fit") or {}) and (run.get("modal") or {}).get("status") == "completed"
        ],
        key=_modal_rank_balanced,
    )
    matching_modal_baselines = [
        run
        for run in modal_runs
        if run.get("strategy") == "modal_baseline"
        and str((run.get("modal") or {}).get("base_mass_run_name") or "") == str(config["modal"]["base_mass_run_name"])
        and (run.get("modal") or {}).get("status") == "completed"
    ]
    modal_baseline_run = (
        max(matching_modal_baselines, key=lambda item: int(item.get("run_index", 0)))
        if matching_modal_baselines
        else None
    )
    modal_reference_path = project_relative(config, config["resolved_paths"]["modal_reference_file"])

    known_issues = [
        "La suma de las partidas objetivo del presupuesto es 3811 g y el target total es 3810 g; se conserva la discrepancia sin corregirla."
        if discrepancy
        else "No hay discrepancia entre las partidas objetivo y la masa total objetivo.",
        "El estimador interno cubre GRID, MAT1, PSHELL, PBARL tipo BAR, CQUAD4, CTRIA3 y CBAR; CBUSH, PBUSH, RBE2 y SPC se tratan como masa nula en esta fase.",
    ]
    if config["runner"].get("dry_run", True):
        known_issues.append("La configuracion actual esta en dry_run=True.")
    if failed_runs:
        known_issues.append(f"Hay {len(failed_runs)} runs fallidas registradas y se conservan en historial con su run.json y report.html.")
    if baseline_run and baseline_run.get("mass_errors", {}).get("parsed_nastran_total_mass_g") is None:
        known_issues.append("El baseline actual no aporta por ahora una masa total parseable desde Nastran; el contraste cuantitativo se apoya en la masa interna.")
    known_issues.append("El parser modal actual consume .f06 con REAL EIGENVECTOR y todavia no soporta .pch u .op2.")

    reference_run = best_feasible_run or best_run
    modal_fit_ranking = []
    for rank, run in enumerate(modal_fit_runs, start=1):
        summary = _summarize_run(config, run)
        summary["rank"] = rank
        modal_fit_ranking.append(summary)

    best_modal_candidate = modal_fit_runs[0] if modal_fit_runs else None
    modal_fit_improvement = None
    if modal_baseline_run and best_modal_candidate:
        baseline_summary = (modal_baseline_run.get("modal") or {}).get("summary", {})
        candidate_summary = (best_modal_candidate.get("modal") or {}).get("summary", {})
        modal_fit_improvement = {
            "mean_mac_delta": round(
                float(candidate_summary.get("mean_mac", 0.0)) - float(baseline_summary.get("mean_mac", 0.0)),
                8,
            ),
            "mean_frequency_error_delta": round(
                float(candidate_summary.get("mean_frequency_error", 0.0))
                - float(baseline_summary.get("mean_frequency_error", 0.0)),
                8,
            ),
        }
    modal_local_ranking = []
    for rank, run in enumerate(modal_local_runs, start=1):
        summary = _summarize_run(config, run)
        summary["rank"] = rank
        modal_local_ranking.append(summary)

    balanced_local_ranking = []
    for rank, run in enumerate(balanced_local_runs, start=1):
        summary = _summarize_run(config, run)
        summary["rank"] = rank
        balanced_local_ranking.append(summary)

    best_modal_local_run = balanced_local_runs[0] if balanced_local_runs else (modal_local_runs[0] if modal_local_runs else None)
    best_targeted_modal_run = targeted_runs[0] if targeted_runs else None
    recommended_run_diagnostics = (
        (diagnostic_modal_run.get("paired_mode_diagnostics") if diagnostic_modal_run else None)
        or ((diagnostic_modal_run.get("modal") or {}).get("paired_mode_diagnostics") if diagnostic_modal_run else None)
        or []
    )
    recommended_run_problematic = (
        (diagnostic_modal_run.get("problematic_modes") if diagnostic_modal_run else None)
        or ((diagnostic_modal_run.get("modal") or {}).get("problematic_modes") if diagnostic_modal_run else None)
        or {}
    )
    modal_local_sensitivity = (diagnostic_modal_run.get("modal_local_sensitivity") if diagnostic_modal_run else None) or {}
    targeted_recommendation = (diagnostic_modal_run.get("targeted_recommendation") if diagnostic_modal_run else None) or {}
    modal_local_conclusions: list[str] = []
    if best_modal_local_run is not None:
        local_block = (
            best_modal_local_run.get("modal_balanced_local_refinement")
            or best_modal_local_run.get("modal_local_refinement")
            or {}
        )
        best_local_summary = (best_modal_local_run.get("modal") or {}).get("summary", {})
        modal_local_conclusions.append(
            f"La mejor run de refinamiento local es {best_modal_local_run['run_name']} con mean MAC {best_local_summary.get('mean_mac'):.6f}."
        )
        vs_modal_fit = (
            (local_block.get("comparison_vs_reference_balanced") or {}).get("mean_mac")
            or (local_block.get("comparison_vs_best_one_at_a_time") or {}).get("mean_mac")
        )
        vs_modal_baseline = (local_block.get("comparison_vs_modal_baseline") or {}).get("mean_mac")
        if vs_modal_fit:
            modal_local_conclusions.append(
                "Frente a su referencia de refinamiento, el mean MAC mejora en "
                f"{float(vs_modal_fit['improvement_absolute']):+.6f}."
            )
        if vs_modal_baseline:
            modal_local_conclusions.append(
                "Frente a la modal baseline, el mean MAC mejora en "
                f"{float(vs_modal_baseline['improvement_absolute']):+.6f}."
            )

    return {
        "project": {
            "name": config["project"].get("name", "EUE mass-fit"),
            "phase": config["project"].get("phase", "Fase 1 - ajuste de masas"),
            "project_root": str(config["project_root"]),
            "base_bdf": project_relative(config, config["resolved_paths"]["base_bdf"]),
        },
        "runner": {
            "command": str(config["runner"]["command"]),
            "dry_run": bool(config["runner"].get("dry_run", False)),
            "timeout_seconds": int(config["runner"].get("timeout_seconds", 600)),
        },
        "paths": {
            "config": project_relative(config, config["resolved_paths"]["config_file"]),
            "runs_dir": project_relative(config, config["resolved_paths"]["runs_dir"]),
            "reports_dir": project_relative(config, config["resolved_paths"]["reports_dir"]),
            "handoff_dir": project_relative(config, config["resolved_paths"]["handoff_dir"]),
            "handoff_index": project_relative(config, config["resolved_paths"]["handoff_dir"] / "index.html"),
            "ia_json": project_relative(config, config["resolved_paths"]["handoff_dir"] / "IA.json"),
        },
        "strategy_current": "feasibility_with_slack",
        "modal_strategy_current": str(config["modal"]["fit"].get("method", "youngs_modulus_one_at_a_time")),
        "strategy_current_modal_refinement": str(
            config["modal"]["balanced_local_fit"].get("method", "modal_balanced_local_grid_refinement")
        ),
        "strategy_current_modal_targeted": str(
            config["modal"]["targeted_fit"].get("method", "modal_targeted_micro_refinement")
        ),
        "phase_2_status": "modal_pipeline_ready" if modal_completed_runs else "modal_pipeline_pending",
        "implementation_summary": {
            "approach": "Heuristica de factibilidad con bandas +/-5% y slack, usando el estimador interno para explorar y la infraestructura de runs para persistir candidatos aceptados.",
            "commands": [
                "py -3 -m src.main baseline",
                "py -3 -m src.main feasibility-fit",
                "py -3 -m src.main modal-baseline",
                "py -3 -m src.main modal-fit",
                "py -3 -m src.main modal-fit-local",
                "py -3 -m src.main modal-fit-balanced-local",
                "py -3 -m src.main modal-fit-targeted",
                "py -3 -m src.main sweep",
                "py -3 -m src.main rebuild-report",
                "py -3 -m src.main show-config",
            ],
        },
        "controls": _control_definitions_snapshot(config),
        "targets_with_bands": _targets_with_bands_snapshot(config),
        "physical_interpretation": {
            "bandeja_fpga_observation": "Bandeja_FPGA participa a la vez en el control operativo de FPGA y en el de backplanes con peso 1.0 en ambos, segun la configuracion actual.",
            "total_closure_role": "Al_7075-T7351 actua como variable principal de cierre global de masa total.",
        },
        "modal_reference": {
            "path": modal_reference_path,
            "source_type": str(config["modal"]["reference"]["source_type"]),
            "rationale": str(config["modal"]["reference"].get("rationale", "")),
            "notes": list(config["modal"]["reference"].get("notes", [])),
            "measurement_points": list(config["modal"]["reference"].get("measurement_points", [])),
            "base_mass_run_name": str(config["modal"]["base_mass_run_name"]),
            "num_modes": int(config["modal"]["num_modes"]),
            "mac_good_threshold": float(config["modal"]["mac_good_threshold"]),
            "frequency_error_good_threshold": float(config["modal"]["frequency_error_good_threshold"]),
            "pairing_frequency_penalty_weight": float(config["modal"]["pairing_frequency_penalty_weight"]),
        },
        "baseline": {
            "configured_factors": baseline_factors(config),
            "reference_run": _summarize_run(config, baseline_run) if baseline_run else None,
            "control_rows": (baseline_run.get("mass_fit") or {}).get("control_rows", []) if baseline_run else [],
            "local_sensitivity": baseline_sensitivity,
        },
        "best_feasible_run": _summarize_run(config, best_feasible_run) if best_feasible_run else None,
        "best_partial_run": _summarize_run(config, best_partial_run) if best_partial_run else None,
        "best_modal_run": _summarize_run(config, best_modal_run) if best_modal_run else None,
        "best_modal_run_strict": _summarize_run(config, best_modal_run) if best_modal_run else None,
        "best_modal_run_balanced": _summarize_run(config, best_modal_run_balanced) if best_modal_run_balanced else None,
        "recommended_modal_run": _summarize_run(config, recommended_modal_run) if recommended_modal_run else None,
        "runs": [_summarize_run(config, run) for run in runs],
        "best_run": _summarize_run(config, best_run) if best_run else None,
        "best_solution_feasible": bool(best_feasible_run is not None),
        "modal_pipeline_ready": bool(best_modal_run is not None),
        "modal_selection_view": {
            "best_strict_modal_run": _summarize_run(config, best_modal_run) if best_modal_run else None,
            "best_balanced_modal_run": _summarize_run(config, best_modal_run_balanced) if best_modal_run_balanced else None,
            "recommended_modal_run": _summarize_run(config, recommended_modal_run) if recommended_modal_run else None,
            "reason": modal_selection["reason"],
            "principal_tradeoff": modal_selection["tradeoff"],
            "next_step_recommended": modal_selection["next_step"],
        },
        "modal_runs": [_summarize_run(config, run) for run in modal_runs],
        "modal_fit": {
            "strategy": str(config["modal"]["fit"].get("method", "youngs_modulus_one_at_a_time")),
            "densities_frozen_from_run": str(config["modal"]["base_mass_run_name"]),
            "candidate_factors": [float(value) for value in config["modal"]["fit"].get("candidate_factors", [0.95, 1.05])],
            "variable_order": [str(value) for value in config["modal"]["fit"].get("variable_order", [])],
            "launch_nastran": bool(config["modal"]["fit"].get("launch_nastran", True)),
            "ranking_basis": [
                "1. maximize mean MAC paired",
                "2. minimize mean frequency error paired",
                "3. maximize worst MAC paired",
                "4. minimize worst frequency error paired",
            ],
            "baseline_modal_run": _summarize_run(config, modal_baseline_run) if modal_baseline_run else None,
            "best_candidate": _summarize_run(config, best_modal_candidate) if best_modal_candidate else None,
            "candidate_ranking": modal_fit_ranking,
            "candidate_count": len(modal_fit_ranking),
            "improvement_vs_modal_baseline": modal_fit_improvement,
        },
        "best_modal_local_run": _summarize_run(config, best_modal_local_run) if best_modal_local_run else None,
        "best_targeted_modal_run": _summarize_run(config, best_targeted_modal_run) if best_targeted_modal_run else None,
        "modal_local_refinement": {
            "strategy": str(config["modal"]["local_fit"].get("method", "modal_local_grid_refinement")),
            "densities_frozen_from_run": str(config["modal"]["base_mass_run_name"]),
            "reference_modal_fit_run_name": (
                str(best_modal_candidate.get("run_name"))
                if best_modal_candidate is not None
                else str(config["modal"]["local_fit"].get("reference_modal_fit_run_name") or "")
            ),
            "candidate_grid": {
                "primary_grid": {
                    str(name): [float(value) for value in values]
                    for name, values in config["modal"]["local_fit"].get("primary_grid", {}).items()
                },
                "secondary_stage": {
                    "enabled": bool(config["modal"]["local_fit"]["secondary_stage"].get("enabled", True)),
                    "trigger": str(config["modal"]["local_fit"]["secondary_stage"].get("trigger", "if_primary_completed")),
                    "shared_material": str(config["modal"]["local_fit"]["secondary_stage"].get("shared_material", "Bandeja_FPGA")),
                    "factors": [
                        float(value)
                        for value in config["modal"]["local_fit"]["secondary_stage"].get("factors", [0.95, 1.0, 1.05])
                    ],
                },
            },
            "best_run": _summarize_run(config, best_modal_local_run) if best_modal_local_run else None,
            "ranking_local": modal_local_ranking,
            "candidate_count": len(modal_local_ranking),
            "conclusions": modal_local_conclusions,
        },
        "balanced_local_refinement": {
            "strategy": str(config["modal"]["balanced_local_fit"].get("method", "modal_balanced_local_grid_refinement")),
            "densities_frozen_from_run": str(config["modal"]["base_mass_run_name"]),
            "reference_balanced_run_name": str(
                config["modal"]["balanced_local_fit"].get("reference_balanced_run_name") or ""
            ),
            "candidate_grid": {
                "primary_grid": {
                    str(name): [float(value) for value in values]
                    for name, values in config["modal"]["balanced_local_fit"].get("primary_grid", {}).items()
                }
            },
            "best_run": _summarize_run(config, balanced_local_runs[0]) if balanced_local_runs else None,
            "ranking_local": balanced_local_ranking,
            "candidate_count": len(balanced_local_ranking),
            "conclusions": modal_local_conclusions,
        },
        "recommended_run_diagnostics": recommended_run_diagnostics,
        "problematic_modes": recommended_run_problematic,
        "modal_local_sensitivity": modal_local_sensitivity,
        "targeted_recommendation": targeted_recommendation,
        "targeted_candidate_results": [_summarize_run(config, run) for run in targeted_runs],
        "frozen_mass_base_run_name": str(config["modal"]["base_mass_run_name"]),
        "modal_measurement_points": list(config["modal"]["measurement_points"]),
        "modal_vector_order": list(config["modal"]["modal_vector_order"]),
        "modal_parser_limitations": [
            "Solo se parsean bloques REAL EIGENVECTOR en formato .f06.",
            "Solo se usan T1, T2 y T3 en los nodos 486, 452, 2040 y 402.",
            "La referencia usa IDs de punto distintos y depende del mapeo explicito definido en modal.reference.measurement_points.",
            "No hay soporte todavia para .pch ni .op2.",
        ],
        "modal_pairing": (best_modal_run.get("modal") or {}).get("pairing") if best_modal_run else None,
        "limiting_variable": (reference_run.get("mass_fit") or {}).get("most_limiting_variable") if reference_run else None,
        "iteration_history_summary": (reference_run.get("feasibility_fit") or {}).get("iteration_history", []) if reference_run else [],
        "known_issues": known_issues,
        "conclusion": {
            "best_run_strict": str(best_modal_run.get("run_name")) if best_modal_run else None,
            "best_run_balanced": str(best_modal_run_balanced.get("run_name")) if best_modal_run_balanced else None,
            "recommended_run": str(recommended_modal_run.get("run_name")) if recommended_modal_run else None,
            "best_targeted_run": str(best_targeted_modal_run.get("run_name")) if best_targeted_modal_run else None,
            "principal_tradeoff": modal_selection["tradeoff"],
            "next_step_recommended": targeted_recommendation.get("notes", [modal_selection["next_step"]])[0],
        },
        "last_updated": now_iso(),
    }


def write_ia_json(
    config: dict[str, Any],
    runs: list[dict[str, Any]],
    baseline_sensitivity: dict[str, Any] | None = None,
) -> Path:
    payload = build_ia_payload(config, runs, baseline_sensitivity=baseline_sensitivity)
    output_path = Path(config["resolved_paths"]["handoff_dir"]) / "IA.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def _build_table(headers: list[str], rows: list[list[str]]) -> str:
    head_html = "".join(f"<th>{escape(header)}</th>" for header in headers)
    row_html = []
    for row in rows:
        row_html.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    return "<table><thead><tr>" + head_html + "</tr></thead><tbody>" + "".join(row_html) + "</tbody></table>"


def _multiline_mapping(mapping: dict[str, float], digits: int = 3) -> str:
    if not mapping:
        return "n/a"
    lines = [f"{escape(str(key))}: {_format_number(value, digits)} g" for key, value in mapping.items()]
    return "<div class='multiline'>" + "<br>".join(lines) + "</div>"


def _physical_interpretation_notes(config: dict[str, Any]) -> list[str]:
    return [
        "Las seis densidades MAT1 son variables independientes y se ajustan con bandas de tolerancia del +/-5%.",
        "PSU, MOD y PROC se controlan directamente con PCB_PSU, PCB_MOD y PCB_PROC.",
        "FPGA combina la masa de PCB_FPGA con la contribucion compartida de Bandeja_FPGA.",
        "Backplane modules se controla principalmente con Bandeja_FPGA.",
        "Al_7075-T7351 se usa para cerrar la masa total cuando existe slack sin romper restricciones parciales.",
    ]


def _control_rows_html(control_rows: list[dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in control_rows:
        rows.append(
            [
                escape(str(row["label"])),
                _format_number(row["target_g"], 3),
                _format_number(row["lower_bound_g"], 3),
                _format_number(row["upper_bound_g"], 3),
                _format_number(row["actual_g"], 3),
                escape(_format_bool(bool(row["within_band"]))),
                _format_number(row["slack_down_g"], 3),
                _format_number(row["slack_up_g"], 3),
            ]
        )
    return rows


def _highlighted_cell(value: str, abs_error_percent: float | None) -> str:
    cell_class = _mass_error_cell_class(abs_error_percent)
    if not cell_class:
        return value
    return f"<span class='{cell_class}' style='display:inline-block; padding:3px 8px; border-radius:6px;'>{value}</span>"


def _budget_row_label(item: dict[str, Any]) -> str:
    label = escape(str(item.get("target_item") or "n/a"))
    control_key = str(item.get("control_key") or "")
    if control_key == "fpga":
        note = "PCB_FPGA + Bandeja_FPGA shared"
    elif control_key == "backplanes":
        note = "Bandeja_FPGA shared"
    elif control_key == "total":
        note = "total model mass"
    else:
        note = ", ".join(str(name) for name in item.get("direct_materials", [])) or "direct control"
    return f"{label}<br><span class='muted'>{escape(note)}</span>"


def _mass_budget_comparison_rows_html(rows: list[dict[str, Any]], include_bounds: bool = False) -> list[list[str]]:
    table_rows: list[list[str]] = []
    for item in rows:
        base_cells = [
            _budget_row_label(item),
            _format_number(item.get("target_g"), 3),
        ]
        if include_bounds:
            base_cells.extend(
                [
                    _format_number(item.get("lower_bound_g"), 3),
                    _format_number(item.get("upper_bound_g"), 3),
                ]
            )
        base_cells.extend(
            [
                _format_number(item.get("actual_g"), 3),
                _highlighted_cell(
                    _format_percent_value(item.get("signed_error_percent")),
                    item.get("absolute_error_percent"),
                ),
                _highlighted_cell(
                    _format_percent_value(item.get("absolute_error_percent")),
                    item.get("absolute_error_percent"),
                ),
                _highlighted_cell(
                    escape(_format_bool(item.get("within_tolerance_5pct"))),
                    item.get("absolute_error_percent"),
                ),
            ]
        )
        table_rows.append(base_cells)
    return table_rows


def _adjusted_material_mass_rows_html(rows: list[dict[str, Any]]) -> list[list[str]]:
    table_rows: list[list[str]] = []
    for item in rows:
        target_g = item.get("associated_target_g")
        target_display = _format_number(target_g, 3) if target_g is not None else "n/a"
        if item.get("association_type") == "shared_control":
            target_display = "n/a (shared contribution)"
        table_rows.append(
            [
                escape(str(item.get("material"))),
                _format_number(item.get("density_factor"), 3),
                _format_number(item.get("final_density"), 6),
                _format_number(item.get("final_mass_g"), 3),
                escape(str(item.get("associated_budget_item") or "n/a")),
                target_display,
                _highlighted_cell(
                    _format_percent_value(item.get("signed_error_percent")),
                    item.get("absolute_error_percent"),
                ),
                _highlighted_cell(
                    _format_percent_value(item.get("absolute_error_percent")),
                    item.get("absolute_error_percent"),
                ),
            ]
        )
    return table_rows


def _compact_mass_summary_html(title: str, run: dict[str, Any] | None) -> str:
    if run is None:
        return f"<div class='card'><span class='label'>{escape(title)}</span><div class='value' style='font-size:18px'>n/a</div></div>"
    rows = [
        [
            _budget_row_label(item),
            _format_number(item.get("target_g"), 3),
            _format_number(item.get("actual_g"), 3),
            _highlighted_cell(_format_percent_value(item.get("signed_error_percent")), item.get("absolute_error_percent")),
            _highlighted_cell(escape(_format_bool(item.get("within_tolerance_5pct"))), item.get("absolute_error_percent")),
        ]
        for item in (run.get("mass_budget_comparison_rows") or [])
        if str(item.get("control_key")) != "total"
    ]
    return (
        "<div class='card'>"
        f"<span class='label'>{escape(title)}</span>"
        f"<div class='value' style='font-size:18px'>{escape(str(run.get('run_name') or 'n/a'))}</div>"
        + _build_table(
            ["Field", "Value"],
            [
                ["Total mass [g]", _format_number(run.get("total_mass_g"), 3)],
                ["Total mass signed error [%]", _format_percent_value(run.get("total_mass_signed_error_percent"))],
                ["Total mass absolute error [%]", _format_percent_value(run.get("total_mass_absolute_error_percent"))],
            ],
        )
        + _build_table(
            ["Budget item", "Target [g]", "Actual [g]", "Signed error %", "Within +/-5%"],
            rows or [["n/a", "0.000", "0.000", "n/a", "n/a"]],
        )
        + "</div>"
    )


def _comparison_delta_rows(comparison: dict[str, Any] | None, label: str) -> list[list[str]]:
    if not comparison:
        return []
    return [
        [f"{label} - delta mean MAC", _format_number(comparison["mean_mac"]["improvement_absolute"], 6)],
        [f"{label} - delta worst MAC", _format_number(comparison["worst_mac"]["improvement_absolute"], 6)],
        [
            f"{label} - delta mean frequency error",
            f"{_format_number(comparison['mean_frequency_error']['improvement_absolute'], 6)} ({_frequency_error_direction(comparison['mean_frequency_error']['improvement_absolute'])})",
        ],
        [f"{label} - delta good_on_both", escape(str(comparison["good_on_both_count"]["improvement_absolute"]))],
    ]


def _iteration_rows(run: dict[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    for item in (run.get("feasibility_fit") or {}).get("iteration_history", []):
        rows.append(
            [
                escape(str(item.get("pass_index"))),
                escape(_format_bool(item.get("feasible"))),
                escape(str(item.get("constraints_satisfied_count")) + " / " + str(item.get("constraint_count"))),
                escape(str((item.get("most_limiting_variable") or {}).get("material_name") or "n/a")),
                escape(json.dumps(item.get("ranking_key"), ensure_ascii=False)),
                escape("; ".join(str(step.get("reason")) for step in item.get("steps", [])) or "n/a"),
            ]
        )
    return rows


def _last_proportional_step(run: dict[str, Any]) -> dict[str, Any] | None:
    history = (run.get("feasibility_fit") or {}).get("iteration_history", [])
    for iteration in reversed(history):
        for step in reversed(iteration.get("steps", [])):
            if step.get("phase") == "B":
                return step
    return None


def _multiline_control_values(control_values_g: dict[str, float], config: dict[str, Any]) -> str:
    if not control_values_g:
        return "n/a"
    lines = []
    for control_key, value in control_values_g.items():
        label = config["mass_fit_targets"].get(control_key, {}).get("label", control_key)
        lines.append(f"{escape(str(label))}: {_format_number(value, 3)} g")
    return "<div class='multiline'>" + "<br>".join(lines) + "</div>"


def _multiline_vector(values_by_label: dict[str, float]) -> str:
    if not values_by_label:
        return "n/a"
    lines = [f"{escape(str(label))}: {_format_number(value, 6)}" for label, value in values_by_label.items()]
    return "<div class='multiline'>" + "<br>".join(lines) + "</div>"


def _mac_cell_color(value: float) -> str:
    value = max(0.0, min(1.0, float(value)))
    red = int(round(246 - (value * 110)))
    green = int(round(239 - (value * 28)))
    blue = int(round(232 - (value * 120)))
    return f"rgb({red}, {green}, {blue})"


def _mac_heatmap_html(modal: dict[str, Any]) -> str:
    matrix = modal.get("mac_matrix", [])
    if not matrix:
        return "<p class='muted'>No MAC matrix available.</p>"

    header = "".join(f"<th>M{index + 1}</th>" for index in range(len(matrix[0])))
    rows = []
    for row_index, row in enumerate(matrix):
        cells = [f"<th>R{row_index + 1}</th>"]
        for value in row:
            cells.append(
                f"<td style=\"background:{_mac_cell_color(float(value))}; text-align:center; font-weight:600;\">{_format_number(value, 3)}</td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<table><thead><tr><th>ref/model</th>"
        + header
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _mode_diagnostic_rows_html(diagnostics: list[dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for item in diagnostics:
        rows.append(
            [
                escape(str(item.get("reference_mode_number"))),
                escape(str(item.get("model_mode_number"))),
                _format_number(item.get("paired_mac"), 3),
                _format_percent_value(item.get("signed_frequency_error_percent")),
                _format_percent_value(item.get("absolute_frequency_error_percent")),
                escape(_format_bool(item.get("mac_good"))),
                escape(_format_bool(item.get("frequency_good"))),
                escape(_format_bool(item.get("good_on_both"))),
                escape(str(item.get("classification") or "n/a")),
                escape(str(item.get("failure_driver") or "n/a")),
                escape(str(item.get("reference_dominant_summary") or "n/a")),
                escape(str(item.get("model_dominant_summary") or "n/a")),
                escape(str(item.get("diagnostic_note") or "n/a")),
            ]
        )
    return rows


def _problematic_modes_rows_html(problematic_modes: dict[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    for item in problematic_modes.get("top_problematic", []):
        rows.append(
            [
                escape(str(item.get("priority_rank") or "n/a")),
                escape(str(item.get("reference_mode_number") or "n/a")),
                escape(str(item.get("model_mode_number") or "n/a")),
                escape(str(item.get("failure_driver") or "n/a")),
                _format_number(item.get("paired_mac"), 3),
                _format_percent_value(item.get("absolute_frequency_error_percent")),
                escape(str(item.get("model_node_summary") or "n/a")),
                escape(str(item.get("diagnostic_note") or "n/a")),
            ]
        )
    return rows


def _modal_local_sensitivity_rows_html(sensitivity: dict[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    for item in sensitivity.get("rows", []):
        rows.append(
            [
                escape(str(item.get("material_name") or "n/a")),
                escape(str(item.get("direction") or "n/a")),
                _format_number(item.get("factor_before"), 3),
                _format_number(item.get("factor_after"), 3),
                _format_number(item.get("delta_mean_mac"), 6),
                _format_number(item.get("delta_worst_mac"), 6),
                _format_number(item.get("delta_mean_frequency_error"), 6),
                escape(str(item.get("frequency_error_direction") or "n/a")),
                escape(str(item.get("delta_good_on_both") or 0)),
                escape(str(item.get("effect_note") or "n/a")),
            ]
        )
    return rows


def _targeted_recommendation_rows_html(recommendation: dict[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    for item in recommendation.get("selected_moves", []):
        rows.append(
            [
                escape(str(item.get("material_name") or "n/a")),
                _format_number(item.get("current_factor"), 3),
                _format_number(item.get("recommended_factor"), 3),
                escape(str(item.get("direction") or "n/a")),
                escape(str(item.get("effect_note") or "n/a")),
                escape(str(item.get("source_run_name") or "n/a")),
            ]
        )
    return rows


def _targeted_candidate_rows_html(candidate_rows: list[dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for item in candidate_rows:
        comparison = item.get("comparison_vs_recommended") or {}
        freq_delta = (comparison.get("mean_frequency_error") or {}).get("improvement_absolute")
        rows.append(
            [
                escape(str(item.get("rank") or "n/a")),
                escape(str(item.get("run_name") or "n/a")),
                escape(json.dumps(item.get("candidate_values", {}), ensure_ascii=False)),
                _format_number(item.get("mean_mac"), 3),
                _format_number(item.get("worst_mac"), 3),
                _format_percent(item.get("mean_frequency_error")),
                escape(str(item.get("good_on_both_count") or 0)),
                _format_number((comparison.get("mean_mac") or {}).get("improvement_absolute"), 6),
                _format_number((comparison.get("worst_mac") or {}).get("improvement_absolute"), 6),
                _format_number(freq_delta, 6),
                escape(_frequency_error_direction(freq_delta)),
            ]
        )
    return rows


def _modal_sections_html(config: dict[str, Any], run: dict[str, Any]) -> str:
    modal = run.get("modal") or {}
    if not _is_modal_run(run):
        return ""
    if modal.get("status") != "completed":
        parser_notes = "".join(f"<li>{escape(str(note))}</li>" for note in modal.get("parser_notes", [])) or "<li>Sin notas.</li>"
        exception = escape(str(modal.get("exception") or "None"))
        return f"""
      <h2>Modal Correlation</h2>
      <p class="muted">La run se ha marcado como modal, pero la correlacion aun no esta completada.</p>
      {_build_table(
          ["Field", "Value"],
          [
              ["Status", escape(str(modal.get("status") or "pending"))],
              ["Base mass run", escape(str(modal.get("base_mass_run_name") or "n/a"))],
              ["Reference source", escape(str((modal.get("reference_source") or {}).get("path") or "n/a"))],
              ["Model output", escape(str(modal.get("model_output_file") or "n/a"))],
          ],
      )}
      <ul>{parser_notes}</ul>
      <pre>{exception}</pre>
"""

    summary = modal["summary"]
    pairing = modal["pairing"]
    modal_local = run.get("modal_local_refinement") or {}
    modal_balanced_local = run.get("modal_balanced_local_refinement") or {}
    modal_setup_rows = [
        ["Base mass run", escape(str(modal.get("base_mass_run_name") or "n/a"))],
        ["Reference source", escape(str((modal.get("reference_source") or {}).get("path") or "n/a"))],
        ["Reference type", escape(str((modal.get("reference_source") or {}).get("source_type") or "n/a"))],
        ["Model output", escape(str(modal.get("model_output_file") or "n/a"))],
        ["Modes compared", escape(str(summary.get("num_modes")))],
        ["MAC threshold", _format_number(summary.get("mac_good_threshold"), 3)],
        ["Frequency error threshold", _format_percent(summary.get("frequency_error_good_threshold"))],
    ]
    modal_fit = run.get("modal_fit") or {}
    modal_fit_html = ""
    if modal_fit:
        modal_fit_html = _build_table(
            ["Field", "Value"],
            [
                ["Method", escape(str(modal_fit.get("method") or "n/a"))],
                ["Frozen density run", escape(str(modal_fit.get("frozen_density_run_name") or "n/a"))],
                ["Candidate MAT1", escape(str(modal_fit.get("candidate_material_name") or "n/a"))],
                ["Candidate E factor", _format_number(modal_fit.get("candidate_e_factor"), 3)],
                ["Changed E factors", escape(_changed_youngs_summary(config, run))],
                ["Mass feasibility frozen", escape(_format_bool(modal_fit.get("mass_feasibility_frozen")))],
            ],
        )
    modal_local_html = ""
    if modal_local:
        local_rows = [
            ["Method", escape(str(modal_local.get("method") or "n/a"))],
            ["Stage", escape(str(modal_local.get("stage") or "n/a"))],
            ["Rank overall", escape(str(modal_local.get("rank_overall") or "n/a"))],
            ["Rank within stage", escape(str(modal_local.get("rank_within_stage") or "n/a"))],
            ["Candidate values", escape(json.dumps(modal_local.get("candidate_values", {}), ensure_ascii=False))],
            ["Frozen density run", escape(str(modal_local.get("frozen_density_run_name") or "n/a"))],
            ["Reference modal baseline", escape(str(modal_local.get("reference_modal_baseline_run_name") or "n/a"))],
            ["Reference best one-at-a-time", escape(str(modal_local.get("reference_modal_fit_run_name") or "n/a"))],
        ]
        comparison_rows = []
        for label, comparison in [
            ("Vs modal baseline", modal_local.get("comparison_vs_modal_baseline")),
            ("Vs best one-at-a-time", modal_local.get("comparison_vs_best_one_at_a_time")),
        ]:
            if not comparison:
                continue
            comparison_rows.extend(
                [
                    [
                        f"{label} - mean MAC",
                        _format_number(comparison["mean_mac"]["improvement_absolute"], 6),
                        _format_percent(comparison["mean_mac"]["improvement_relative"]),
                    ],
                    [
                        f"{label} - worst MAC",
                        _format_number(comparison["worst_mac"]["improvement_absolute"], 6),
                        _format_percent(comparison["worst_mac"]["improvement_relative"]),
                    ],
                    [
                        f"{label} - mean freq error improvement",
                        _format_number(comparison["mean_frequency_error"]["improvement_absolute"], 6),
                        _format_percent(comparison["mean_frequency_error"]["improvement_relative"]),
                    ],
                    [
                        f"{label} - worst freq error improvement",
                        _format_number(comparison["worst_frequency_error"]["improvement_absolute"], 6),
                        _format_percent(comparison["worst_frequency_error"]["improvement_relative"]),
                    ],
                    [
                        f"{label} - good_on_both delta",
                        escape(str(comparison["good_on_both_count"]["improvement_absolute"])),
                        _format_percent(comparison["good_on_both_count"]["improvement_relative"]),
                    ],
                ]
            )
        modal_local_html = (
            "<h2>Modal Local Refinement</h2>"
            + _build_table(["Field", "Value"], local_rows)
            + _build_table(["Comparison", "Improvement", "Relative"], comparison_rows or [["n/a", "n/a", "n/a"]])
        )

    modal_balanced_local_html = ""
    if modal_balanced_local:
        balanced_rows = [
            ["Method", escape(str(modal_balanced_local.get("method") or "n/a"))],
            ["Ranking mode", escape(str(modal_balanced_local.get("ranking_mode") or "balanced"))],
            ["Stage", escape(str(modal_balanced_local.get("stage") or "n/a"))],
            ["Rank overall", escape(str(modal_balanced_local.get("rank_overall") or "n/a"))],
            ["Strict rank within grid", escape(str(modal_balanced_local.get("strict_rank_within_grid") or "n/a"))],
            ["Candidate values", escape(json.dumps(modal_balanced_local.get("candidate_values", {}), ensure_ascii=False))],
            ["Frozen density run", escape(str(modal_balanced_local.get("frozen_density_run_name") or "n/a"))],
            ["Reference balanced run", escape(str(modal_balanced_local.get("reference_balanced_run_name") or "n/a"))],
            ["Reference strict run", escape(str(modal_balanced_local.get("reference_strict_run_name") or "n/a"))],
        ]
        balanced_comparison_rows = []
        for label, comparison in [
            ("Vs balanced reference", modal_balanced_local.get("comparison_vs_reference_balanced")),
            ("Vs strict reference", modal_balanced_local.get("comparison_vs_reference_strict")),
            ("Vs modal baseline", modal_balanced_local.get("comparison_vs_modal_baseline")),
            ("Vs best one-at-a-time", modal_balanced_local.get("comparison_vs_best_one_at_a_time")),
        ]:
            if not comparison:
                continue
            balanced_comparison_rows.extend(
                [
                    [f"{label} - delta mean MAC", _format_number(comparison["mean_mac"]["improvement_absolute"], 6), _format_percent(comparison["mean_mac"]["improvement_relative"])],
                    [f"{label} - delta worst MAC", _format_number(comparison["worst_mac"]["improvement_absolute"], 6), _format_percent(comparison["worst_mac"]["improvement_relative"])],
                    [
                        f"{label} - delta mean frequency error",
                        _format_number(comparison["mean_frequency_error"]["improvement_absolute"], 6),
                        _frequency_error_direction(comparison["mean_frequency_error"]["improvement_absolute"]),
                    ],
                    [
                        f"{label} - delta worst frequency error",
                        _format_number(comparison["worst_frequency_error"]["improvement_absolute"], 6),
                        _frequency_error_direction(comparison["worst_frequency_error"]["improvement_absolute"]),
                    ],
                    [f"{label} - delta good_on_both", escape(str(comparison["good_on_both_count"]["improvement_absolute"])), _format_percent(comparison["good_on_both_count"]["improvement_relative"])],
                ]
            )
        modal_balanced_local_html = (
            "<h2>Balanced Local Refinement</h2>"
            + _build_table(["Field", "Value"], balanced_rows)
            + _build_table(["Comparison", "Delta", "Interpretation"], balanced_comparison_rows or [["n/a", "n/a", "n/a"]])
        )

    modal_targeted = run.get("modal_targeted_fit") or {}
    modal_targeted_html = ""
    if modal_targeted:
        targeted_rows = [
            ["Method", escape(str(modal_targeted.get("method") or "n/a"))],
            ["Rank overall", escape(str(modal_targeted.get("rank_overall") or "n/a"))],
            ["Strict rank within targeted", escape(str(modal_targeted.get("strict_rank_within_targeted") or "n/a"))],
            ["Base recommended run", escape(str(modal_targeted.get("base_recommended_run_name") or "n/a"))],
            ["Candidate values", escape(json.dumps(modal_targeted.get("candidate_values", {}), ensure_ascii=False))],
            ["Moves", escape(json.dumps(modal_targeted.get("moves", []), ensure_ascii=False))],
        ]
        targeted_comparison = modal_targeted.get("comparison_vs_recommended")
        targeted_comparison_rows = _comparison_delta_rows(targeted_comparison, "Vs recommended modal run")
        modal_targeted_html = (
            "<h2>Targeted Modal Fit</h2>"
            + _build_table(["Field", "Value"], targeted_rows)
            + _build_table(["Comparison", "Delta"], targeted_comparison_rows or [["n/a", "n/a"]])
        )

    paired_mode_diagnostics = run.get("paired_mode_diagnostics") or modal.get("paired_mode_diagnostics") or []
    problematic_modes = run.get("problematic_modes") or modal.get("problematic_modes") or {}
    base_sensitivity = run.get("modal_local_sensitivity") or {}
    targeted_recommendation = run.get("targeted_recommendation") or {}

    reference_mode_rows = []
    for mode in modal.get("reference", {}).get("modes", []):
        reference_mode_rows.append(
            [
                escape(str(mode.get("mode_number"))),
                _format_number(mode.get("frequency_hz"), 6),
                _multiline_vector(mode.get("modal_vector_by_label", {})),
            ]
        )

    model_mode_rows = []
    for mode in modal.get("model", {}).get("modes", []):
        model_mode_rows.append(
            [
                escape(str(mode.get("mode_number"))),
                _format_number(mode.get("frequency_hz"), 6),
                _multiline_vector(mode.get("modal_vector_by_label", {})),
            ]
        )

    diagonal_rows = []
    diagonal_mac_by_index = {
        int(item["reference_index"]): item for item in pairing.get("diagonal_comparison", [])
    }
    for index, item in enumerate(modal.get("frequency_comparison_diagonal", []), start=1):
        diagonal_mac = diagonal_mac_by_index.get(index, {}).get("mac")
        diagonal_rows.append(
            [
                escape(str(item.get("reference_mode_number"))),
                escape(str(item.get("model_mode_number"))),
                _format_number(item.get("reference_frequency_hz"), 6),
                _format_number(item.get("model_frequency_hz"), 6),
                _format_percent(item.get("frequency_error_rel")),
                _format_number(diagonal_mac, 3),
            ]
        )

    pairing_rows = []
    for item in pairing.get("pairs", []):
        pairing_rows.append(
            [
                escape(str(item.get("reference_mode_number"))),
                escape(str(item.get("model_mode_number"))),
                _format_number(item.get("reference_frequency_hz"), 6),
                _format_number(item.get("model_frequency_hz"), 6),
                _format_percent(item.get("frequency_error_rel")),
                _format_number(item.get("mac"), 3),
                escape(_format_bool(item.get("mac_good"))),
                escape(_format_bool(item.get("frequency_good"))),
                escape(_format_bool(item.get("good_on_both"))),
            ]
        )

    parser_notes = "".join(f"<li>{escape(str(note))}</li>" for note in modal.get("parser_notes", [])) or "<li>Sin notas.</li>"
    reference_notes = "".join(
        f"<li>{escape(str(note))}</li>" for note in (modal.get("reference_source") or {}).get("notes", [])
    ) or "<li>Sin notas adicionales.</li>"

    return f"""
      <h2>Modal Correlation</h2>
      {_build_table(["Field", "Value"], modal_setup_rows)}
      {modal_fit_html}
      {modal_local_html}
      {modal_balanced_local_html}
      {modal_targeted_html}

      <div class="grid">
        <div class="card"><span class="label">Mean MAC</span><div class="value">{_format_number(summary.get('mean_mac'), 3)}</div></div>
        <div class="card"><span class="label">Worst MAC</span><div class="value">{_format_number(summary.get('worst_mac'), 3)}</div></div>
        <div class="card"><span class="label">Mean Freq. Error</span><div class="value">{_format_percent(summary.get('mean_frequency_error'))}</div></div>
        <div class="card"><span class="label">Worst Freq. Error</span><div class="value">{_format_percent(summary.get('worst_frequency_error'))}</div></div>
      </div>

      {_build_table(
          ["Metric", "Value"],
          [
              ["Modes with MAC >= threshold", escape(str(summary.get("mac_good_count")))],
              ["Modes with freq. error <= threshold", escape(str(summary.get("frequency_good_count")))],
              ["Modes good on both", escape(str(summary.get("good_on_both_count")))],
              ["Modal ranking score", _format_number(summary.get("modal_score"), 6)],
          ],
      )}

      <h2>Reference Modal Vectors</h2>
      {_build_table(["Mode", "Frequency [Hz]", "Vector"], reference_mode_rows or [["n/a", "n/a", "n/a"]])}

      <h2>Model Modal Vectors</h2>
      {_build_table(["Mode", "Frequency [Hz]", "Vector"], model_mode_rows or [["n/a", "n/a", "n/a"]])}

      <h2>Diagonal Transparency</h2>
      {_build_table(
          ["Ref mode", "Model mode", "Ref freq [Hz]", "Model freq [Hz]", "Freq. error", "Diagonal MAC"],
          diagonal_rows or [["n/a", "n/a", "n/a", "n/a", "n/a", "n/a"]],
      )}

      <h2>MAC Matrix 10x10</h2>
      {_mac_heatmap_html(modal)}

      <h2>Final Pairing</h2>
      {_build_table(
          ["Ref mode", "Model mode", "Ref freq [Hz]", "Model freq [Hz]", "Freq. error", "MAC", "MAC ok", "Freq ok", "Good on both"],
          pairing_rows or [["n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"]],
      )}

      <h2>Mode-by-Mode Diagnostic</h2>
      {_build_table(
          [
              "Ref mode",
              "Model mode",
              "Paired MAC",
              "Signed freq error [%]",
              "Absolute freq error [%]",
              "MAC ok",
              "Freq ok",
              "Good on both",
              "Class",
              "Driver",
              "Reference dominant DOFs",
              "Model dominant DOFs",
              "Diagnostic note",
          ],
          _mode_diagnostic_rows_html(paired_mode_diagnostics) or [["n/a"] * 13],
      )}

      <h2>Problematic Modes Priority</h2>
      {_build_table(
          ["Priority", "Ref mode", "Model mode", "Failure driver", "MAC", "Absolute freq error [%]", "Model dominant nodes", "Interpretation"],
          _problematic_modes_rows_html(problematic_modes) or [["n/a"] * 8],
      )}
      {_build_table(
          ["Class", "Count"],
          [
              ["Good", escape(str((problematic_modes.get("counts") or {}).get("good", 0)))],
              ["Acceptable", escape(str((problematic_modes.get("counts") or {}).get("acceptable", 0)))],
              ["Problematic", escape(str((problematic_modes.get("counts") or {}).get("problematic", 0)))],
          ],
      )}

      <h2>Interpretation Física del Ajuste</h2>
      <ul>
        <li>Las seis densidades MAT1 siguen congeladas desde la run 005 en esta fase modal.</li>
        <li>La run modal recomendada se usa como base para diagnosticar modos problematicos y decidir micro-ajustes de E.</li>
        <li>El diagnostico por modo se construye con energia relativa por DOF sobre el vector modal normalizado.</li>
        <li>Los modos se clasifican como Good, Acceptable o Problematic segun los umbrales configurados en <code>config/project.yaml</code>.</li>
      </ul>

      <h2>Local Modal Sensitivity Around Recommended Run</h2>
      {_build_table(
          [
              "Material",
              "Direction",
              "Factor before",
              "Factor after",
              "dMean MAC",
              "dWorst MAC",
              "dMean freq error",
              "Frequency trend",
              "dGood on both",
              "Main effect",
          ],
          _modal_local_sensitivity_rows_html(base_sensitivity) or [["n/a"] * 10],
      )}

      <h2>Targeted Next-Step Recommendation</h2>
      {_build_table(
          ["Material", "Current factor", "Recommended factor", "Direction", "Reason", "Source run"],
          _targeted_recommendation_rows_html(targeted_recommendation) or [["n/a"] * 6],
      )}

      <h2>Reference Notes</h2>
      <ul>{reference_notes}</ul>

      <h2>Parsing Notes</h2>
      <ul>{parser_notes}</ul>
"""


def _run_report_html(
    config: dict[str, Any],
    run: dict[str, Any],
    output_dir: Path,
    latest_baseline_run_name: str | None,
    baseline_sensitivity: dict[str, Any] | None,
) -> str:
    report_paths = run.get("report_paths", {})
    output_files = run.get("output_files", [])
    mass_fit = run.get("mass_fit") or {}
    modal = run.get("modal") or {}
    control_rows = mass_fit.get("control_rows", [])
    mass_budget_comparison_rows = run.get("mass_budget_comparison_rows") or run.get("mass_budget_comparison", [])
    adjusted_material_mass_rows = run.get("adjusted_material_mass_rows") or []
    material_rows = []
    updates_by_name = {item["material_name"]: item for item in run.get("material_updates", [])}
    for material in config["materials"]:
        name = str(material["name"])
        update = updates_by_name.get(name, {})
        material_rows.append(
            [
                escape(name),
                str(material["id"]),
                _format_number(run.get("parameter_factors", {}).get(name), 3),
                _format_number(run.get("youngs_modulus_factors", {}).get(name), 3),
                _format_number(update.get("rho_original"), 6),
                _format_number(update.get("rho_new"), 6),
                _format_number(update.get("e_original"), 6),
                _format_number(update.get("e_new"), 6),
                _format_number(run.get("masses_by_material_g", {}).get(name), 3),
            ]
        )

    property_type_rows = [
        [escape(str(property_type)), _format_number(mass_g, 3)]
        for property_type, mass_g in sorted(run.get("internal_mass", {}).get("masses_by_property_type_g", {}).items())
    ]
    property_rows = []
    for item in run.get("internal_mass", {}).get("masses_by_property_g", []):
        property_rows.append(
            [
                escape(str(item.get("property_type"))),
                escape(str(item.get("property_id"))),
                escape(str(item.get("label") or "n/a")),
                escape(str(item.get("material_name") or "n/a")),
                _format_number(item.get("mass_g"), 3),
            ]
        )

    contribution_rows = []
    for item in run.get("model_mass_contributions", []):
        contribution_rows.append(
            [
                escape(str(item.get("material_name"))),
                escape(str(item.get("assignment_kind"))),
                escape(", ".join(str(value) for value in item.get("controls", [])) or "n/a"),
                _format_number(item.get("mass_g"), 3),
                escape(str(item.get("interpretation") or "n/a")),
            ]
        )

    control_table_rows = _control_rows_html(control_rows)
    budget_rows = _mass_budget_comparison_rows_html(mass_budget_comparison_rows, include_bounds=False)
    adjusted_material_rows = _adjusted_material_mass_rows_html(adjusted_material_mass_rows)

    unassigned_rows = []
    for name, mass_g in (run.get("unassigned_materials_g") or {}).items():
        unassigned_rows.append([escape(str(name)), _format_number(mass_g, 3)])
    if not unassigned_rows:
        unassigned_rows.append(["Sin masa comun adicional", "0.000"])

    file_links = []
    for relative_path in output_files:
        target = config["project_root"] / relative_path
        file_links.append(f'<li><a href="{escape(_relpath(output_dir, target))}">{escape(relative_path)}</a></li>')
    if not file_links:
        file_links.append("<li>No se generaron outputs adicionales.</li>")

    log_path = run.get("logs", {}).get("runner_log")
    log_link = "n/a"
    if log_path:
        log_link = f'<a href="{escape(_relpath(output_dir, config["project_root"] / log_path))}">{escape(log_path)}</a>'

    notes = "".join(f"<li>{escape(str(note))}</li>" for note in run.get("notes", [])) or "<li>Sin notas.</li>"
    warnings = "".join(
        f"<li>{escape(str(note))}</li>" for note in run.get("internal_mass", {}).get("warnings", [])
    ) or "<li>Sin warnings del estimador interno.</li>"
    unsupported_cards = ", ".join(run.get("internal_mass", {}).get("unsupported_cards", [])) or "ninguna"
    coverage_limitations = "".join(
        f"<li>{escape(str(note))}</li>"
        for note in run.get("internal_mass", {}).get("coverage_review", {}).get("limitations", [])
    ) or "<li>Sin limitaciones adicionales declaradas.</li>"

    sensitivity_html = ""
    if baseline_sensitivity and run.get("run_name") == latest_baseline_run_name:
        sensitivity_rows = []
        for row in baseline_sensitivity.get("rows", []):
            sensitivity_rows.append(
                [
                    escape(str(row.get("material_name"))),
                    escape(str(row.get("variation_label"))),
                    _format_number(row.get("factor_multiplier"), 3),
                    _format_number(row.get("total_mass_g"), 3),
                    _format_number(row.get("delta_total_mass_g"), 3),
                    _multiline_mapping(row.get("masses_by_material_g", {}), digits=3),
                    _multiline_control_values(row.get("control_values_g", {}), config),
                ]
            )
        sensitivity_notes = "".join(
            f"<li>{escape(str(note))}</li>" for note in baseline_sensitivity.get("notes", [])
        )
        sensitivity_html = f"""
      <h2>Local Sensitivity Around Baseline</h2>
      <p class="muted">Perturbacion one-at-a-time de +/-{_format_number(float(baseline_sensitivity.get('variation_fraction', 0.05)) * 100, 1)}% sobre cada densidad. Se muestra el efecto sobre la masa total y sobre los controles operativos.</p>
      {_build_table(
          [
              "Material",
              "Variation",
              "Relative factor",
              "Total mass [g]",
              "Delta total [g]",
              "Mass by material",
              "Control values",
          ],
          sensitivity_rows,
      )}
      <ul>{sensitivity_notes}</ul>
"""

    iteration_rows = _iteration_rows(run)
    last_step = _last_proportional_step(run)
    last_step_table = [["No correction", "n/a"]]
    if last_step:
        last_step_table = [
            ["Direction", escape(str(last_step.get("direction")))],
            ["Applied", escape(_format_bool(last_step.get("applied")))],
            ["Progress", _format_number(last_step.get("progress"), 6)],
            ["Mass before [g]", _format_number(last_step.get("mass_before_g"), 3)],
            ["Mass after [g]", _format_number(last_step.get("mass_after_g"), 3)],
            ["Eligible variables", escape(", ".join(item["material_name"] for item in last_step.get("eligible_variables", [])) or "n/a")],
        ]

    modal_sections_html = _modal_sections_html(config, run)

    body = f"""
    <div class="page">
      <p><a href="{escape(_relpath(output_dir, config['resolved_paths']['handoff_dir'] / 'index.html'))}">Volver al reporte maestro</a></p>
      <h1>{escape(str(run.get('run_name')))}</h1>
      <p class="muted">Run index {escape(str(run.get('run_index')))} | {escape(str(run.get('timestamp')))} | <span class="pill">{escape(str(run.get('status')))}</span></p>
      <div class="grid">
        <div class="card"><span class="label">Strategy</span><div class="value">{escape(str(run.get('strategy')))}</div></div>
        <div class="card"><span class="label">Feasible</span><div class="value">{escape(_format_bool(mass_fit.get('feasible')))}</div></div>
        <div class="card"><span class="label">Constraints</span><div class="value">{escape(str(mass_fit.get('constraints_satisfied_count')) + " / " + str(mass_fit.get('constraint_count')))}</div></div>
        <div class="card"><span class="label">Total Mass [g]</span><div class="value">{_format_number(run.get('total_mass_g'), 3)}</div></div>
        <div class="card"><span class="label">Total Mass Signed Error</span><div class="value">{_format_percent_value(run.get('mass_errors', {}).get('total_mass_signed_error_percent'))}</div></div>
        <div class="card"><span class="label">Total Mass Absolute Error</span><div class="value">{_format_percent_value(run.get('mass_errors', {}).get('total_mass_absolute_error_percent'))}</div></div>
      </div>

      <h2>Physical Interpretation</h2>
      <ul>{''.join(f"<li>{escape(note)}</li>" for note in _physical_interpretation_notes(config))}</ul>

      <h2>Run Summary</h2>
      {_build_table(
          ["Field", "Value"],
          [
              ["Generated BDF", escape(str(run.get("generated_bdf")))],
              ["Input BDF", escape(str(run.get("input_bdf")))],
              ["Nastran command", f"<code>{escape(str(run.get('nastran_command')))}</code>"],
              ["Runner mode", escape(str(run.get("nastran", {}).get("mode")))],
              ["Runner log", log_link],
              ["Master report", escape(str(report_paths.get("master")))],
              ["Mirror report", escape(str(report_paths.get("run_local")))],
              ["Most limiting variable", escape(str((mass_fit.get("most_limiting_variable") or {}).get("material_name") or "n/a"))],
              ["Modal status", escape(str(modal.get("status") or "n/a"))],
              ["Changed E factors", escape(_changed_youngs_summary(config, run))],
          ],
      )}

      {modal_sections_html}

      <h2>Mass-Fit Constraints</h2>
      {_build_table(
          ["Control", "Target [g]", "Lower [g]", "Upper [g]", "Actual [g]", "Within band", "Slack down [g]", "Slack up [g]"],
          control_table_rows or [["n/a", "0.000", "0.000", "0.000", "0.000", "n/a", "0.000", "0.000"]],
      )}

      <h2>Accepted Iterations</h2>
      {_build_table(
          ["Pass", "Feasible", "Constraints", "Most limiting variable", "Ranking key", "Step reasons"],
          iteration_rows or [["n/a", "n/a", "n/a", "n/a", "n/a", "No internal history persisted for this run."]],
      )}

      <h2>Last Proportional Correction</h2>
      {_build_table(["Field", "Value"], last_step_table)}

      <h2>Mass By MAT1</h2>
      {_build_table(["Material", "MAT1", "Rho factor", "E factor", "Rho base", "Rho new", "E base", "E new", "Mass [g]"], material_rows)}

      <h2>Mass Budget Comparison</h2>
      {_build_table(
          ["Budget item", "Target mass [g]", "Actual mass [g]", "Signed error [%]", "Absolute error [%]", "Within +/-5%"],
          budget_rows or [["n/a", "0.000", "0.000", "n/a", "n/a", "n/a"]],
      )}

      <h2>Adjusted Material Mass Breakdown</h2>
      {_build_table(
          ["Material", "Density factor", "Final density", "Final mass [g]", "Associated budget item", "Associated target [g]", "Signed error [%]", "Absolute error [%]"],
          adjusted_material_rows or [["n/a", "0.000", "0.000000", "0.000", "n/a", "n/a", "n/a", "n/a"]],
      )}

      <h2>Mass By Property Type</h2>
      {_build_table(["Property type", "Mass [g]"], property_type_rows or [["n/a", "0.000"]])}

      <h2>Mass By Property Contribution</h2>
      {_build_table(["Type", "Property", "Label", "Material", "Mass [g]"], property_rows or [["n/a", "n/a", "n/a", "n/a", "0.000"]])}

      <h2>Model Mass Contributions</h2>
      {_build_table(
          ["Material", "Contribution class", "Controls", "Mass [g]", "Interpretation"],
          contribution_rows or [["n/a", "n/a", "n/a", "0.000", "n/a"]],
      )}

      <h2>Operational Control Breakdown</h2>
      {_build_table(
          ["Control", "Target [g]", "Lower [g]", "Upper [g]", "Actual [g]", "Within band", "Slack down [g]", "Slack up [g]"],
          control_table_rows or [["n/a", "0.000", "0.000", "0.000", "0.000", "n/a", "0.000", "0.000"]],
      )}

      <h2>Common Structural Mass</h2>
      {_build_table(
          ["Field", "Value"],
          [
              ["Direct control mass [g]", _format_number(run.get("direct_reference_mass_g"), 3)],
              ["Shared mass [g]", _format_number(run.get("shared_mass_g"), 3)],
              ["Common structural mass [g]", _format_number(run.get("unassigned_mass_g"), 3)],
              ["Total mass [g]", _format_number(run.get("total_mass_g"), 3)],
          ],
      )}
      {_build_table(["Material", "Mass [g]"], unassigned_rows)}

      <h2>Score Breakdown</h2>
      {_build_table(
          ["Component", "Value"],
          [
              ["Unsatisfied constraints", escape(str(run.get("score_components", {}).get("unsatisfied_constraints")))],
              ["Max normalized violation", _format_number(run.get("score_components", {}).get("max_normalized_violation"), 6)],
              ["Total mass rel. error", _format_percent(run.get("score_components", {}).get("total_mass_rel_error"))],
              ["Change from baseline L1", _format_number(run.get("score_components", {}).get("change_from_baseline_l1"), 6)],
              ["Failure penalty", _format_number(run.get("score_components", {}).get("failure_penalty_applied"), 3)],
          ],
      )}

      <h2>Internal Mass Estimator</h2>
      {_build_table(
          ["Field", "Value"],
          [
              ["Shell elements", escape(str(run.get("internal_mass", {}).get("element_counts", {}).get("shells")))],
              ["Bar elements", escape(str(run.get("internal_mass", {}).get("element_counts", {}).get("bars")))],
              ["Supported mass cards found", escape(json.dumps(run.get("internal_mass", {}).get("coverage_review", {}).get("supported_mass_cards_found", {}), ensure_ascii=False))],
              ["Zero-mass assumptions", escape(json.dumps(run.get("internal_mass", {}).get("coverage_review", {}).get("zero_mass_assumed_cards_found", {}), ensure_ascii=False))],
              ["Unsupported cards seen", escape(unsupported_cards)],
          ],
      )}
      <ul>{warnings}</ul>
      <ul>{coverage_limitations}</ul>

      {sensitivity_html}

      <h2>Logs And Files</h2>
      <ul>{''.join(file_links)}</ul>

      <h2>Notes</h2>
      <ul>{notes}</ul>

      <h2>Exception</h2>
      <pre>{escape(str(run.get("exception") or "None"))}</pre>
    </div>
    """
    return f"<!doctype html><html lang='es'><head><meta charset='utf-8'><title>{escape(str(run.get('run_name')))}</title><style>{STYLE}</style></head><body>{body}</body></html>"


def write_run_reports(
    config: dict[str, Any],
    run: dict[str, Any],
    latest_baseline_run_name: str | None = None,
    baseline_sensitivity: dict[str, Any] | None = None,
) -> None:
    reports_root = Path(config["resolved_paths"]["reports_dir"]) / str(run["run_name"])
    run_root = Path(config["resolved_paths"]["runs_dir"]) / str(run["run_name"])
    reports_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    master_html = _run_report_html(config, run, reports_root, latest_baseline_run_name, baseline_sensitivity)
    mirror_html = _run_report_html(config, run, run_root, latest_baseline_run_name, baseline_sensitivity)

    (reports_root / "report.html").write_text(master_html, encoding="utf-8")
    (run_root / "report.html").write_text(mirror_html, encoding="utf-8")


def write_master_report(
    config: dict[str, Any],
    runs: list[dict[str, Any]],
    baseline_sensitivity: dict[str, Any] | None = None,
) -> Path:
    best_run = select_best_run(runs)
    best_feasible_run = _select_best_feasible_run(runs)
    modal_selection = _modal_selection_view(runs)
    best_modal_run = modal_selection["best_modal_run_strict"]
    best_modal_run_balanced = modal_selection["best_modal_run_balanced"]
    recommended_modal_run = modal_selection["recommended_modal_run"]
    diagnostic_modal_run = _diagnostic_modal_run(runs, recommended_modal_run)
    modal_fit_runs = sorted(
        [
            run
            for run in runs
            if run.get("strategy") == "modal_fit" and (run.get("modal") or {}).get("status") == "completed"
        ],
        key=_modal_rank_strict,
    )
    legacy_modal_local_runs = sorted(
        [
            run
            for run in runs
            if run.get("strategy") == "modal_fit_local" and (run.get("modal") or {}).get("status") == "completed"
        ],
        key=_modal_rank_strict,
    )
    balanced_local_runs = sorted(
        [
            run
            for run in runs
            if (run.get("modal_balanced_local_refinement") or {}) and (run.get("modal") or {}).get("status") == "completed"
        ],
        key=_modal_rank_balanced,
    )
    targeted_runs = sorted(
        [
            run
            for run in runs
            if (run.get("modal_targeted_fit") or {}) and (run.get("modal") or {}).get("status") == "completed"
        ],
        key=_modal_rank_balanced,
    )
    best_modal_candidate = modal_fit_runs[0] if modal_fit_runs else None
    best_balanced_local_run = balanced_local_runs[0] if balanced_local_runs else None
    best_targeted_modal_run = targeted_runs[0] if targeted_runs else None
    baseline_run = _latest_baseline_run(runs)
    output_path = Path(config["resolved_paths"]["handoff_dir"]) / "index.html"
    discrepancy = float(config["targets"]["mass_budget_target_total_delta_g"])

    rows: list[str] = []
    for run in runs:
        report_target = config["project_root"] / run["report_paths"]["master"]
        row_class = "best" if best_run and run["run_name"] == best_run["run_name"] else ""
        if run.get("status") == "failed":
            row_class = f"{row_class} failed".strip()
        if (run.get("mass_fit") or {}).get("feasible"):
            row_class = f"{row_class} feasible".strip()
        rows.append(
            f"<tr class='{row_class}'>"
            f"<td>{escape(str(run.get('run_index')))}</td>"
            f"<td><a href='{escape(_relpath(output_path.parent, report_target))}'>{escape(str(run.get('run_name')))}</a></td>"
            f"<td>{escape(str(run.get('strategy')))}</td>"
            f"<td>{escape(str(run.get('status')))}</td>"
            f"<td>{escape(_format_bool((run.get('mass_fit') or {}).get('feasible')))}</td>"
            f"<td>{escape(str((run.get('mass_fit') or {}).get('constraints_satisfied_count')) + ' / ' + str((run.get('mass_fit') or {}).get('constraint_count')))}</td>"
            f"<td>{_format_number(run.get('total_mass_g'), 3)}</td>"
            f"<td>{_format_percent(run.get('mass_errors', {}).get('total_mass_rel_error'))}</td>"
            f"<td>{escape(str(((run.get('mass_fit') or {}).get('most_limiting_variable') or {}).get('material_name') or 'n/a'))}</td>"
            f"<td>{escape(str(run.get('nastran', {}).get('mode')))}</td>"
            f"</tr>"
        )

    modal_rows: list[str] = []
    for run in [item for item in runs if (item.get("modal") or {}).get("status") == "completed"]:
        report_target = config["project_root"] / run["report_paths"]["master"]
        modal_summary = (run.get("modal") or {}).get("summary", {})
        badge = []
        if best_modal_run and run["run_name"] == best_modal_run["run_name"]:
            badge.append("strict")
        if best_modal_run_balanced and run["run_name"] == best_modal_run_balanced["run_name"]:
            badge.append("balanced")
        if recommended_modal_run and run["run_name"] == recommended_modal_run["run_name"]:
            badge.append("recommended")
        if best_targeted_modal_run and run["run_name"] == best_targeted_modal_run["run_name"]:
            badge.append("targeted")
        row_class = "best" if badge else ""
        modal_rows.append(
            f"<tr class='{row_class}'>"
            f"<td><a href='{escape(_relpath(output_path.parent, report_target))}'>{escape(str(run.get('run_name')))}</a></td>"
            f"<td>{escape(str((run.get('modal') or {}).get('base_mass_run_name') or 'n/a'))}</td>"
            f"<td>{escape(_changed_youngs_summary(config, run))}</td>"
            f"<td>{_format_number(modal_summary.get('mean_mac'), 3)}</td>"
            f"<td>{_format_number(modal_summary.get('worst_mac'), 3)}</td>"
            f"<td>{_format_percent(modal_summary.get('mean_frequency_error'))}</td>"
            f"<td>{escape(str(modal_summary.get('good_on_both_count') or 0))}</td>"
            f"<td>{escape(', '.join(badge) or 'candidate')}</td>"
            f"<td>{escape(str(run.get('status')))}</td>"
            f"</tr>"
        )

    modal_fit_rows: list[str] = []
    for rank, run in enumerate(modal_fit_runs, start=1):
        report_target = config["project_root"] / run["report_paths"]["master"]
        modal_summary = (run.get("modal") or {}).get("summary", {})
        modal_fit = run.get("modal_fit") or {}
        row_class = "best" if best_modal_candidate and run["run_name"] == best_modal_candidate["run_name"] else ""
        modal_fit_rows.append(
            f"<tr class='{row_class}'>"
            f"<td>{rank}</td>"
            f"<td><a href='{escape(_relpath(output_path.parent, report_target))}'>{escape(str(run.get('run_name')))}</a></td>"
            f"<td>{escape(str(modal_fit.get('candidate_material_name') or 'n/a'))}</td>"
            f"<td>{_format_number(modal_fit.get('candidate_e_factor'), 3)}</td>"
            f"<td>{escape(_changed_youngs_summary(config, run))}</td>"
            f"<td>{_format_number(modal_summary.get('mean_mac'), 3)}</td>"
            f"<td>{_format_percent(modal_summary.get('mean_frequency_error'))}</td>"
            f"<td>{_format_number(modal_summary.get('worst_mac'), 3)}</td>"
            f"<td>{escape(str(modal_summary.get('good_on_both_count') or 0))}</td>"
            f"</tr>"
        )

    balanced_local_rows: list[str] = []
    for rank, run in enumerate(balanced_local_runs, start=1):
        report_target = config["project_root"] / run["report_paths"]["master"]
        modal_summary = (run.get("modal") or {}).get("summary", {})
        local_block = run.get("modal_balanced_local_refinement") or {}
        row_class = "best" if best_balanced_local_run and run["run_name"] == best_balanced_local_run["run_name"] else ""
        balanced_local_rows.append(
            f"<tr class='{row_class}'>"
            f"<td>{rank}</td>"
            f"<td><a href='{escape(_relpath(output_path.parent, report_target))}'>{escape(str(run.get('run_name')))}</a></td>"
            f"<td>{escape(json.dumps(local_block.get('candidate_values', {}), ensure_ascii=False))}</td>"
            f"<td>{_format_number(modal_summary.get('mean_mac'), 3)}</td>"
            f"<td>{_format_number(modal_summary.get('worst_mac'), 3)}</td>"
            f"<td>{_format_percent(modal_summary.get('mean_frequency_error'))}</td>"
            f"<td>{escape(str(modal_summary.get('good_on_both_count') or 0))}</td>"
            f"<td>{escape(str(local_block.get('rank_overall') or 'n/a'))}</td>"
            f"</tr>"
        )

    targeted_candidate_rows = _targeted_candidate_rows_html(
        [
            {
                "rank": rank,
                "run_name": run.get("run_name"),
                "candidate_values": (run.get("modal_targeted_fit") or {}).get("candidate_values", {}),
                "mean_mac": (run.get("modal") or {}).get("summary", {}).get("mean_mac"),
                "worst_mac": (run.get("modal") or {}).get("summary", {}).get("worst_mac"),
                "mean_frequency_error": (run.get("modal") or {}).get("summary", {}).get("mean_frequency_error"),
                "good_on_both_count": (run.get("modal") or {}).get("summary", {}).get("good_on_both_count"),
                "comparison_vs_recommended": (run.get("modal_targeted_fit") or {}).get("comparison_vs_recommended"),
            }
            for rank, run in enumerate(targeted_runs, start=1)
        ]
    )

    baseline_summary = "No baseline run has been generated yet."
    if baseline_run:
        baseline_summary = (
            f"{escape(str(baseline_run['run_name']))} | feasible {_format_bool((baseline_run.get('mass_fit') or {}).get('feasible'))} | "
            f"constraints {escape(str((baseline_run.get('mass_fit') or {}).get('constraints_satisfied_count')) + ' / ' + str((baseline_run.get('mass_fit') or {}).get('constraint_count')))}"
        )

    best_summary = "No feasible solution yet."
    if best_feasible_run:
        best_summary = (
            f"{escape(str(best_feasible_run['run_name']))} | total mass {_format_number(best_feasible_run.get('total_mass_g'), 3)} g | "
            f"constraints {escape(str((best_feasible_run.get('mass_fit') or {}).get('constraints_satisfied_count')) + ' / ' + str((best_feasible_run.get('mass_fit') or {}).get('constraint_count')))}"
        )
    elif best_run:
        best_summary = (
            f"{escape(str(best_run['run_name']))} | no factible aun | most limiting "
            f"{escape(str(((best_run.get('mass_fit') or {}).get('most_limiting_variable') or {}).get('material_name') or 'n/a'))}"
        )

    best_modal_summary = "No modal run completed yet."
    if best_modal_run:
        modal_summary = (best_modal_run.get("modal") or {}).get("summary", {})
        best_modal_summary = (
            f"{escape(str(best_modal_run['run_name']))} | mean MAC {_format_number(modal_summary.get('mean_mac'), 3)} | "
            f"worst MAC {_format_number(modal_summary.get('worst_mac'), 3)} | mean freq error {_format_percent(modal_summary.get('mean_frequency_error'))}"
        )

    best_modal_candidate_summary = "No modal-fit candidates completed yet."
    if best_modal_candidate:
        modal_summary = (best_modal_candidate.get("modal") or {}).get("summary", {})
        modal_fit = best_modal_candidate.get("modal_fit") or {}
        best_modal_candidate_summary = (
            f"{escape(str(best_modal_candidate['run_name']))} | {escape(str(modal_fit.get('candidate_material_name') or 'n/a'))} "
            f"at E factor {_format_number(modal_fit.get('candidate_e_factor'), 3)} | mean MAC {_format_number(modal_summary.get('mean_mac'), 3)} "
            f"| mean freq error {_format_percent(modal_summary.get('mean_frequency_error'))}"
        )

    best_modal_local_summary = "No balanced local refinement runs completed yet."
    if best_balanced_local_run:
        modal_summary = (best_balanced_local_run.get("modal") or {}).get("summary", {})
        local_block = best_balanced_local_run.get("modal_balanced_local_refinement") or {}
        best_modal_local_summary = (
            f"{escape(str(best_balanced_local_run['run_name']))} | rank {escape(str(local_block.get('rank_overall') or 'n/a'))} | "
            f"mean MAC {_format_number(modal_summary.get('mean_mac'), 3)} | mean freq error {_format_percent(modal_summary.get('mean_frequency_error'))}"
        )

    best_targeted_summary = "No targeted modal runs completed yet."
    if best_targeted_modal_run:
        modal_summary = (best_targeted_modal_run.get("modal") or {}).get("summary", {})
        best_targeted_summary = (
            f"{escape(str(best_targeted_modal_run['run_name']))} | mean MAC {_format_number(modal_summary.get('mean_mac'), 3)} | "
            f"mean freq error {_format_percent(modal_summary.get('mean_frequency_error'))} | good_on_both {escape(str(modal_summary.get('good_on_both_count') or 0))}"
        )

    balanced_local_comparison_rows: list[list[str]] = []
    if best_balanced_local_run:
        local_block = best_balanced_local_run.get("modal_balanced_local_refinement") or {}
        comparison_specs = [
            ("Vs balanced reference " + str((local_block.get("comparison_vs_reference_balanced") or {}).get("reference_run_name") or "n/a"), local_block.get("comparison_vs_reference_balanced")),
            ("Vs strict reference " + str((local_block.get("comparison_vs_reference_strict") or {}).get("reference_run_name") or "n/a"), local_block.get("comparison_vs_reference_strict")),
            ("Vs modal baseline " + str((local_block.get("comparison_vs_modal_baseline") or {}).get("reference_run_name") or "n/a"), local_block.get("comparison_vs_modal_baseline")),
            ("Vs best one-at-a-time " + str((local_block.get("comparison_vs_best_one_at_a_time") or {}).get("reference_run_name") or "n/a"), local_block.get("comparison_vs_best_one_at_a_time")),
        ]
        for label, comparison in comparison_specs:
            balanced_local_comparison_rows.extend(_comparison_delta_rows(comparison, label))

    targets_rows = []
    for control_key, target_cfg in config["mass_fit_targets"].items():
        target_g = float(target_cfg["target_g"])
        tol = float(target_cfg["tolerance_fraction"])
        targets_rows.append(
            [
                escape(str(target_cfg["label"])),
                _format_number(target_g, 3),
                _format_number(target_g * (1.0 - tol), 3),
                _format_number(target_g * (1.0 + tol), 3),
            ]
        )

    controls_rows = []
    for control_key, control_cfg in config["mass_fit_controls"].items():
        shared_names = [f"{item['name']} x {item['weight']:.3f}" for item in control_cfg["shared_materials"]]
        controls_rows.append(
            [
                escape(str(config["mass_fit_targets"][control_key]["label"])),
                escape(", ".join(control_cfg["direct_materials"]) or "n/a"),
                escape(", ".join(shared_names) or "n/a"),
            ]
        )

    sensitivity_section = ""
    if baseline_sensitivity:
        sensitivity_rows = []
        for item in baseline_sensitivity.get("summary_by_material", []):
            sensitivity_rows.append(
                [
                    escape(str(item.get("material_name"))),
                    _format_number(item.get("minus_total_mass_g"), 3),
                    _format_number(item.get("minus_delta_total_mass_g"), 3),
                    _format_number(item.get("plus_total_mass_g"), 3),
                    _format_number(item.get("plus_delta_total_mass_g"), 3),
                ]
            )
        sensitivity_section = f"""
      <h2>Baseline Local Sensitivity</h2>
      <p class="muted">Perturbacion interna one-at-a-time de +/-{_format_number(float(baseline_sensitivity.get('variation_fraction', 0.05)) * 100, 1)}% alrededor del baseline.</p>
      {_build_table(
          ["Material", "Total mass at -5% [g]", "Delta at -5% [g]", "Total mass at +5% [g]", "Delta at +5% [g]"],
          sensitivity_rows,
      )}
"""

    modal_selection_rows = [
        [
            "Best strict modal run",
            escape(str(best_modal_run.get("run_name") if best_modal_run else "n/a")),
            _format_number((best_modal_run.get("modal") or {}).get("summary", {}).get("mean_mac") if best_modal_run else None, 3),
            _format_percent((best_modal_run.get("modal") or {}).get("summary", {}).get("mean_frequency_error") if best_modal_run else None),
            escape(str((best_modal_run.get("modal") or {}).get("summary", {}).get("good_on_both_count") if best_modal_run else "n/a")),
        ],
        [
            "Best balanced modal run",
            escape(str(best_modal_run_balanced.get("run_name") if best_modal_run_balanced else "n/a")),
            _format_number((best_modal_run_balanced.get("modal") or {}).get("summary", {}).get("mean_mac") if best_modal_run_balanced else None, 3),
            _format_percent((best_modal_run_balanced.get("modal") or {}).get("summary", {}).get("mean_frequency_error") if best_modal_run_balanced else None),
            escape(str((best_modal_run_balanced.get("modal") or {}).get("summary", {}).get("good_on_both_count") if best_modal_run_balanced else "n/a")),
        ],
        [
            "Recommended modal run",
            escape(str(recommended_modal_run.get("run_name") if recommended_modal_run else "n/a")),
            _format_number((recommended_modal_run.get("modal") or {}).get("summary", {}).get("mean_mac") if recommended_modal_run else None, 3),
            _format_percent((recommended_modal_run.get("modal") or {}).get("summary", {}).get("mean_frequency_error") if recommended_modal_run else None),
            escape(str((recommended_modal_run.get("modal") or {}).get("summary", {}).get("good_on_both_count") if recommended_modal_run else "n/a")),
        ],
    ]

    recommended_mode_diagnostics = (
        diagnostic_modal_run.get("paired_mode_diagnostics", [])
        if diagnostic_modal_run
        else []
    ) or (((diagnostic_modal_run.get("modal") or {}).get("paired_mode_diagnostics", [])) if diagnostic_modal_run else [])
    recommended_problematic_modes = (
        diagnostic_modal_run.get("problematic_modes", {})
        if diagnostic_modal_run
        else {}
    ) or (((diagnostic_modal_run.get("modal") or {}).get("problematic_modes", {})) if diagnostic_modal_run else {})
    recommended_local_sensitivity = (diagnostic_modal_run.get("modal_local_sensitivity") if diagnostic_modal_run else None) or {}
    recommended_targeted_recommendation = (diagnostic_modal_run.get("targeted_recommendation") if diagnostic_modal_run else None) or {}

    mode_by_mode_section = ""
    if diagnostic_modal_run:
        mode_by_mode_section = f"""
      <h2>Mode-by-Mode Diagnostic for Recommended Run</h2>
      <p class="muted">Diagnostico detallado generado sobre la run de referencia para fase 2.3 <code>{escape(str(diagnostic_modal_run.get('run_name')))}</code>.</p>
      {_build_table(
          [
              "Ref mode",
              "Model mode",
              "Paired MAC",
              "Signed freq error [%]",
              "Absolute freq error [%]",
              "MAC ok",
              "Freq ok",
              "Good on both",
              "Class",
              "Driver",
              "Reference dominant DOFs",
              "Model dominant DOFs",
              "Diagnostic note",
          ],
          _mode_diagnostic_rows_html(recommended_mode_diagnostics) or [["n/a"] * 13],
      )}
"""

    problematic_modes_section = ""
    if diagnostic_modal_run:
        problematic_modes_section = f"""
      <h2>Problematic Modes Priority</h2>
      {_build_table(
          ["Priority", "Ref mode", "Model mode", "Failure driver", "MAC", "Absolute freq error [%]", "Model dominant nodes", "Interpretation"],
          _problematic_modes_rows_html(recommended_problematic_modes) or [["n/a"] * 8],
      )}
"""

    local_modal_sensitivity_section = ""
    if recommended_local_sensitivity:
        local_modal_sensitivity_section = f"""
      <h2>Local Modal Sensitivity Around Recommended Run</h2>
      <p class="muted">Perturbacion local de +/-{_format_number(float(recommended_local_sensitivity.get('perturbation_fraction', 0.02)) * 100.0, 1)}% en torno a la run recomendada.</p>
      {_build_table(
          [
              "Material",
              "Direction",
              "Factor before",
              "Factor after",
              "dMean MAC",
              "dWorst MAC",
              "dMean freq error",
              "Frequency trend",
              "dGood on both",
              "Main effect",
          ],
          _modal_local_sensitivity_rows_html(recommended_local_sensitivity) or [["n/a"] * 10],
      )}
"""

    targeted_recommendation_section = ""
    if recommended_targeted_recommendation:
        targeted_recommendation_section = f"""
      <h2>Targeted Next-Step Recommendation</h2>
      {_build_table(
          ["Material", "Current factor", "Recommended factor", "Direction", "Reason", "Source run"],
          _targeted_recommendation_rows_html(recommended_targeted_recommendation) or [["n/a"] * 6],
      )}
      <ul>{''.join(f"<li>{escape(str(note))}</li>" for note in recommended_targeted_recommendation.get("notes", []))}</ul>
"""

    targeted_results_section = ""
    if targeted_runs:
        targeted_results_section = f"""
      <h2>Targeted Candidate Results</h2>
      <p class="muted">Micro-refinamiento guiado por el diagnostico modal y evaluado con ranking balanceado.</p>
      {_build_table(
          [
              "Rank",
              "Run",
              "Candidate values",
              "Mean MAC",
              "Worst MAC",
              "Mean freq error",
              "Good on both",
              "dMean MAC vs recommended",
              "dWorst MAC vs recommended",
              "dMean freq error vs recommended",
              "Interpretation",
          ],
          targeted_candidate_rows or [["n/a"] * 11],
      )}
"""

    body = f"""
    <div class="page">
      <h1>FEM Correlation Handoff</h1>
      <p class="muted">Fase 1: ajuste de densidades y masa con factibilidad por bandas +/-5%. Fase 2: correlacion modal sobre la run de masa congelada. Este HTML se regenera desde el estado real de <code>runs/</code>.</p>

      <div class="warning">
        <strong>Discrepancia declarada:</strong> la suma de las partidas objetivo del presupuesto es {_format_number(config['targets']['mass_budget_target_sum_g'], 3)} g
        y el target total es {_format_number(config['targets']['total_mass_g'], 3)} g. Delta = {_format_number(discrepancy, 3)} g.
        El sistema no corrige esto a escondidas; solo lo reporta.
      </div>

      <div class="grid">
        <div class="card"><span class="label">Baseline</span><div class="value" style="font-size:18px">{baseline_summary}</div></div>
        <div class="card"><span class="label">Best Solution</span><div class="value" style="font-size:18px">{best_summary}</div></div>
        <div class="card"><span class="label">Best Strict Modal Run</span><div class="value" style="font-size:18px">{best_modal_summary}</div></div>
        <div class="card"><span class="label">Best Modal Candidate</span><div class="value" style="font-size:18px">{best_modal_candidate_summary}</div></div>
        <div class="card"><span class="label">Best Balanced Local Refinement</span><div class="value" style="font-size:18px">{best_modal_local_summary}</div></div>
        <div class="card"><span class="label">Best Targeted Modal Run</span><div class="value" style="font-size:18px">{best_targeted_summary}</div></div>
        <div class="card"><span class="label">Runs Recorded</span><div class="value">{len(runs)}</div></div>
        <div class="card"><span class="label">IA</span><div class="value" style="font-size:18px"><a href="IA.json">Open IA.json</a></div></div>
      </div>

      <div class="summary-grid">
        {_compact_mass_summary_html("Best Feasible Run Mass Summary", best_feasible_run)}
        {_compact_mass_summary_html("Best Modal Run Mass Summary", best_modal_run)}
      </div>

      <h2>Operational Interpretation</h2>
      <ul>{''.join(f"<li>{escape(note)}</li>" for note in _physical_interpretation_notes(config))}</ul>

      <h2>Modal Selection View</h2>
      {_build_table(
          ["View", "Run", "Mean MAC", "Mean Freq Error", "Modes Good On Both"],
          modal_selection_rows,
      )}
      {_build_table(
          ["Field", "Value"],
          [
              ["Reason", escape(str(modal_selection["reason"]))],
              ["Principal trade-off", escape(str(modal_selection["tradeoff"]))],
              ["Next step recommended", escape(str(modal_selection["next_step"]))],
          ],
      )}

      {mode_by_mode_section}
      {problematic_modes_section}
      {local_modal_sensitivity_section}
      {targeted_recommendation_section}
      {targeted_results_section}

      <h2>Phase 2 Modal Correlation</h2>
      <p class="muted">La fase modal parte de la run congelada <code>{escape(str(config['modal']['base_mass_run_name']))}</code> y compara las 10 primeras frecuencias y vectores modales contra <code>{escape(project_relative(config, config['resolved_paths']['modal_reference_file']))}</code>.</p>
      <p class="muted">La referencia usa cuatro IDs de punto propios del modelo detallado; su correspondencia con los cuatro puntos canonicos del SFEM queda fijada en <code>config/project.yaml</code> y reflejada en <code>handoff/IA.json</code>.</p>
      {_build_table(
          ["Field", "Value"],
          [
              ["Reference source", escape(project_relative(config, config["resolved_paths"]["modal_reference_file"]))],
              ["MAC threshold", _format_number(config["modal"]["mac_good_threshold"], 3)],
              ["Frequency error threshold", _format_percent(config["modal"]["frequency_error_good_threshold"])],
              ["Pairing penalty weight", _format_number(config["modal"]["pairing_frequency_penalty_weight"], 3)],
          ],
      )}
      <table>
        <thead>
          <tr>
            <th>Run</th><th>Base Mass Run</th><th>E changes</th><th>Mean MAC</th><th>Worst MAC</th><th>Mean Freq Error</th><th>Modes Good On Both</th><th>Selection</th><th>Status</th>
          </tr>
        </thead>
        <tbody>{''.join(modal_rows) or "<tr><td colspan='9'>No modal runs yet.</td></tr>"}</tbody>
      </table>

      <h2>Modal-Fit Candidate Ranking</h2>
      <p class="muted">Barrido modal contenido con densidades congeladas a la run <code>{escape(str(config['modal']['base_mass_run_name']))}</code> y variacion one-at-a-time de <code>E</code> sobre las 6 MAT1.</p>
      {_build_table(
          ["Field", "Value"],
          [
              ["Strategy", escape(str(config["modal"]["fit"].get("method", "youngs_modulus_one_at_a_time")))],
              ["Candidate E factors", escape(", ".join(_format_number(value, 3) for value in config["modal"]["fit"].get("candidate_factors", [0.95, 1.05])))],
              ["Ranking priority", "1) mean MAC, 2) mean frequency error, 3) worst MAC, 4) worst frequency error"],
          ],
      )}
      <table>
        <thead>
          <tr>
            <th>Rank</th><th>Run</th><th>Candidate MAT1</th><th>E factor</th><th>E changes</th><th>Mean MAC</th><th>Mean Freq Error</th><th>Worst MAC</th><th>Modes Good On Both</th>
          </tr>
        </thead>
        <tbody>{''.join(modal_fit_rows) or "<tr><td colspan='9'>No modal-fit candidates yet.</td></tr>"}</tbody>
      </table>

      <h2>Balanced Local Refinement</h2>
      <p class="muted">Refinamiento local pequeno alrededor del candidato equilibrado <code>{escape(str(config['modal']['balanced_local_fit'].get('reference_balanced_run_name') or 'n/a'))}</code>, usando ranking balanceado.</p>
      {_build_table(
          ["Field", "Value"],
          [
              ["Strategy", escape(str(config["modal"]["balanced_local_fit"].get("method", "modal_balanced_local_grid_refinement")))],
              ["Primary grid", escape(json.dumps(config["modal"]["balanced_local_fit"].get("primary_grid", {}), ensure_ascii=False))],
              ["Ranking priority", "1) good_on_both, 2) mean frequency error, 3) mean MAC, 4) worst MAC, 5) worst frequency error"],
          ],
      )}
      {_build_table(
          ["Comparison", "Delta"],
          balanced_local_comparison_rows or [["n/a", "n/a"]],
      )}
      <table>
        <thead>
          <tr>
            <th>Rank</th><th>Run</th><th>Candidate values</th><th>Mean MAC</th><th>Worst MAC</th><th>Mean Freq Error</th><th>Modes Good On Both</th><th>Balanced Rank</th>
          </tr>
        </thead>
        <tbody>{''.join(balanced_local_rows) or "<tr><td colspan='8'>No balanced local refinement runs yet.</td></tr>"}</tbody>
      </table>

      <h2>Legacy Strict Local Refinement</h2>
      {_build_table(
          ["Field", "Value"],
          [
              ["Strategy", escape(str(config["modal"]["local_fit"].get("method", "modal_local_grid_refinement")))],
              ["Primary grid", escape(json.dumps(config["modal"]["local_fit"].get("primary_grid", {}), ensure_ascii=False))],
          ],
      )}
      {_build_table(
          ["Run", "Mean MAC", "Mean Freq Error"],
          [
              [
                  f"<a href='{escape(_relpath(output_path.parent, config['project_root'] / run['report_paths']['master']))}'>{escape(str(run.get('run_name')))}</a>",
                  _format_number((run.get("modal") or {}).get("summary", {}).get("mean_mac"), 3),
                  _format_percent((run.get("modal") or {}).get("summary", {}).get("mean_frequency_error")),
              ]
              for run in legacy_modal_local_runs[:5]
          ] or [["n/a", "n/a", "n/a"]],
      )}

      <h2>Control Definitions</h2>
      {_build_table(["Control", "Direct materials", "Shared materials"], controls_rows)}

      <h2>Targets And Bands</h2>
      {_build_table(["Control", "Target [g]", "Lower [g]", "Upper [g]"], targets_rows)}

      {sensitivity_section}

      <h2>Runs</h2>
      <table>
        <thead>
          <tr>
            <th>#</th><th>Run</th><th>Strategy</th><th>Status</th><th>Feasible</th><th>Constraints OK</th><th>Total Mass [g]</th><th>Total Mass Error [%]</th><th>Most Limiting Variable</th><th>Runner</th>
          </tr>
        </thead>
        <tbody>{''.join(rows) or "<tr><td colspan='10'>No runs yet.</td></tr>"}</tbody>
      </table>

      <h2>Automatic Conclusion</h2>
      {_build_table(
          ["Field", "Value"],
          [
              ["Best run by strict ranking", escape(str(best_modal_run.get("run_name")) if best_modal_run else "n/a")],
              ["Best run by balanced ranking", escape(str(best_modal_run_balanced.get("run_name")) if best_modal_run_balanced else "n/a")],
              ["Recommended run", escape(str(recommended_modal_run.get("run_name")) if recommended_modal_run else "n/a")],
              ["Best targeted run", escape(str(best_targeted_modal_run.get("run_name")) if best_targeted_modal_run else "n/a")],
              ["Principal trade-off", escape(str(modal_selection["tradeoff"]))],
              ["Next step recommended", escape(str((recommended_targeted_recommendation.get("notes") or [modal_selection["next_step"]])[0]))],
          ],
      )}
    </div>
    """
    output_path.write_text(
        f"<!doctype html><html lang='es'><head><meta charset='utf-8'><title>Mass Fit Handoff</title><style>{STYLE}</style></head><body>{body}</body></html>",
        encoding="utf-8",
    )
    return output_path


def rebuild_project_artifacts(config: dict[str, Any]) -> dict[str, Any]:
    runs = load_run_records(config)
    baseline_run = _latest_baseline_run(runs)
    baseline_sensitivity = _build_baseline_sensitivity(config, baseline_run)

    for run in runs:
        source_path = run.get("_source_path")
        if source_path:
            Path(source_path).write_text(
                json.dumps({key: value for key, value in run.items() if key != "_source_path"}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
    for run in runs:
        sensitivity_for_run = baseline_sensitivity if baseline_run and run.get("run_name") == baseline_run.get("run_name") else None
        write_run_reports(
            config,
            run,
            latest_baseline_run_name=baseline_run.get("run_name") if baseline_run else None,
            baseline_sensitivity=sensitivity_for_run,
        )
    ia_path = write_ia_json(config, runs, baseline_sensitivity=baseline_sensitivity)
    index_path = write_master_report(config, runs, baseline_sensitivity=baseline_sensitivity)
    return {
        "run_count": len(runs),
        "ia_json": project_relative(config, ia_path),
        "index_html": project_relative(config, index_path),
    }
