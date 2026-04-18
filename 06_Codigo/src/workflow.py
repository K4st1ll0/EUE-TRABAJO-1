from __future__ import annotations

import itertools
import json
import math
import time
from pathlib import Path
from typing import Any

from .bdf_editor import apply_mat1_factors
from .config import baseline_factors, material_names, material_sequence, now_iso, project_relative
from .mass_model import (
    build_mass_fit_summary,
    calculate_absolute_error_percent,
    calculate_signed_error_percent,
    estimate_masses,
    is_better_feasibility,
    score_mass_fit,
)
from .nastran_runner import run_nastran
from .reporting import load_run_records, rebuild_project_artifacts


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _existing_runs_by_signature(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for run in load_run_records(config):
        signature = run.get("run_signature") or run.get("factor_signature")
        if signature:
            records[str(signature)] = run
    return records


def _next_run_index(config: dict[str, Any]) -> int:
    runs = load_run_records(config)
    if not runs:
        return 1
    return max(int(run.get("run_index", 0)) for run in runs) + 1


def factor_signature(config: dict[str, Any], factors: dict[str, float], precision: int = 6) -> str:
    parts = []
    for material in material_sequence(config):
        name = str(material["name"])
        parts.append(f"{name}={float(factors[name]):.{precision}f}")
    return "|".join(parts)


def youngs_modulus_baseline_factors(config: dict[str, Any]) -> dict[str, float]:
    return {str(item["name"]): 1.0 for item in material_sequence(config)}


def youngs_signature(config: dict[str, Any], factors: dict[str, float], precision: int = 6) -> str:
    parts = []
    for material in material_sequence(config):
        name = str(material["name"])
        parts.append(f"{name}={float(factors[name]):.{precision}f}")
    return "|".join(parts)


def run_signature(
    config: dict[str, Any],
    density_factors: dict[str, float],
    youngs_factors: dict[str, float] | None = None,
    namespace: str | None = None,
    precision: int = 6,
) -> str:
    density_signature = factor_signature(config, density_factors, precision=precision)
    youngs_values = youngs_factors or youngs_modulus_baseline_factors(config)
    youngs_part = youngs_signature(config, youngs_values, precision=precision)
    signature = f"rho::{density_signature}||E::{youngs_part}"
    if namespace:
        return f"ns::{namespace}||{signature}"
    return signature


def build_run_name(
    config: dict[str, Any],
    run_index: int,
    factors: dict[str, float],
    youngs_factors: dict[str, float] | None = None,
) -> str:
    decimals = int(config["naming"]["decimals"])
    parts = []
    for material in material_sequence(config):
        tag = str(material["tag"])
        name = str(material["name"])
        parts.append(f"{tag}{float(factors[name]):.{decimals}f}")
    name_parts = [f"{run_index:03d}_" + "_".join(parts)]
    active_youngs = youngs_factors or youngs_modulus_baseline_factors(config)
    changed_youngs = []
    for material in material_sequence(config):
        material_name = str(material["name"])
        factor = float(active_youngs[material_name])
        if abs(factor - 1.0) <= 1e-9:
            continue
        changed_youngs.append(f"E{str(material['tag'])}{factor:.{decimals}f}")
    if changed_youngs:
        name_parts.append("_".join(changed_youngs))
    return "_".join(name_parts)


def _linspace(start: float, stop: float, count: int) -> list[float]:
    if count <= 1:
        return [start]
    step = (stop - start) / (count - 1)
    return [round(start + (index * step), 6) for index in range(count)]


def _normalize_factor_overrides(config: dict[str, Any], overrides: dict[str, float] | None) -> dict[str, float]:
    factors = baseline_factors(config)
    for name, value in (overrides or {}).items():
        factors[str(name)] = float(value)
    return factors


def _report_paths(config: dict[str, Any], run_name: str) -> dict[str, str]:
    return {
        "master": project_relative(config, Path(config["resolved_paths"]["reports_dir"]) / run_name / "report.html"),
        "run_local": project_relative(config, Path(config["resolved_paths"]["runs_dir"]) / run_name / "report.html"),
    }


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


def _run_payload_skeleton(
    config: dict[str, Any],
    run_index: int,
    run_name: str,
    strategy: str,
    factors: dict[str, float],
    youngs_factors: dict[str, float],
    signature_namespace: str | None,
    run_dir: Path,
    generated_bdf: Path,
    notes: list[str],
    is_baseline_reference: bool,
) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "run_index": run_index,
        "timestamp": now_iso(),
        "status": "running",
        "strategy": strategy,
        "is_baseline_reference": is_baseline_reference,
        "input_bdf": project_relative(config, config["resolved_paths"]["base_bdf"]),
        "generated_bdf": project_relative(config, generated_bdf),
        "run_json_path": project_relative(config, run_dir / "run.json"),
        "report_paths": _report_paths(config, run_name),
        "parameter_factors": {name: round(float(value), 6) for name, value in factors.items()},
        "youngs_modulus_factors": {name: round(float(value), 6) for name, value in youngs_factors.items()},
        "factor_signature": factor_signature(config, factors),
        "youngs_signature": youngs_signature(config, youngs_factors),
        "run_signature_namespace": signature_namespace,
        "run_signature": run_signature(config, factors, youngs_factors, namespace=signature_namespace),
        "nastran_command": None,
        "material_updates": [],
        "masses_by_material_g": {},
        "model_mass_contributions": [],
        "mass_budget_comparison": [],
        "mass_budget_comparison_rows": [],
        "adjusted_material_mass_rows": [],
        "direct_reference_mass_g": None,
        "shared_mass_g": None,
        "unassigned_mass_g": None,
        "unassigned_materials_g": {},
        "total_mass_g": None,
        "target_total_mass_g": float(config["targets"]["total_mass_g"]),
        "total_mass_signed_error_percent": None,
        "total_mass_absolute_error_percent": None,
        "mass_errors": {},
        "score": None,
        "score_components": {},
        "internal_mass": {},
        "mass_fit": {},
        "nastran": {},
        "cross_checks": {},
        "output_files": [],
        "logs": {},
        "notes": list(notes),
        "duration_seconds": None,
        "exception": None,
    }


def _execute_run(
    config: dict[str, Any],
    strategy: str,
    factors: dict[str, float],
    youngs_factors: dict[str, float] | None = None,
    signature_namespace: str | None = None,
    notes: list[str] | None = None,
    is_baseline_reference: bool = False,
    launch_nastran: bool = True,
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started_at = time.time()
    notes = list(notes or [])
    youngs_factors = {str(name): float(value) for name, value in (youngs_factors or youngs_modulus_baseline_factors(config)).items()}
    run_index = _next_run_index(config)
    run_name = build_run_name(config, run_index, factors, youngs_factors)
    run_dir = Path(config["resolved_paths"]["runs_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    generated_bdf = run_dir / f"{config['paths']['job_name']}.bdf"
    run_json_path = run_dir / "run.json"
    payload = _run_payload_skeleton(
        config=config,
        run_index=run_index,
        run_name=run_name,
        strategy=strategy,
        factors=factors,
        youngs_factors=youngs_factors,
        signature_namespace=signature_namespace,
        run_dir=run_dir,
        generated_bdf=generated_bdf,
        notes=notes,
        is_baseline_reference=is_baseline_reference,
    )

    try:
        base_bdf_path = Path(config["resolved_paths"]["base_bdf"])
        base_text = base_bdf_path.read_text(encoding="utf-8", errors="ignore")
        edited_text, updates = apply_mat1_factors(
            original_text=base_text,
            target_materials=config["materials"],
            density_factors=factors,
            youngs_modulus_factors=youngs_factors,
        )
        generated_bdf.write_text(edited_text, encoding="utf-8")

        payload["material_updates"] = updates
        mass_summary = estimate_masses(edited_text, config)
        mass_fit_summary = build_mass_fit_summary(
            mass_summary,
            config,
            factors=factors,
            baseline_reference_factors=baseline_factors(config),
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

        if launch_nastran:
            runner_result = run_nastran(config, run_dir, generated_bdf, run_name)
        else:
            runner_result = {
                "mode": "skipped",
                "command": "skipped_for_internal_acceptance",
                "success": True,
                "exit_code": None,
                "timed_out": False,
                "duration_seconds": 0.0,
                "log_path": project_relative(config, run_dir / "runner.log"),
                "output_files": [project_relative(config, generated_bdf)],
                "parsed_total_mass_g": None,
                "notes": ["Nastran se ha omitido para esta run aceptada por exploracion interna."],
                "fatal_files": [],
                "success_marker_files": [],
            }
            (run_dir / "runner.log").write_text(
                "Nastran skipped for this accepted feasibility-fit run.\n",
                encoding="utf-8",
            )

        payload["nastran_command"] = runner_result["command"]
        payload["nastran"] = {
            "mode": runner_result["mode"],
            "success": runner_result["success"],
            "exit_code": runner_result["exit_code"],
            "timed_out": runner_result["timed_out"],
            "duration_seconds": runner_result["duration_seconds"],
            "parsed_total_mass_g": runner_result["parsed_total_mass_g"],
            "fatal_files": runner_result.get("fatal_files", []),
            "success_marker_files": runner_result.get("success_marker_files", []),
        }
        payload["logs"] = {"runner_log": runner_result["log_path"]}
        payload["output_files"] = runner_result["output_files"]
        payload["notes"].extend(runner_result.get("notes", []))
        payload["notes"].extend(mass_summary.get("warnings", []))

        nastran_failed = runner_result["mode"] == "live" and not runner_result["success"]
        score_payload = score_mass_fit(
            mass_summary,
            config,
            factors=factors,
            baseline_reference_factors=baseline_factors(config),
            mass_fit_summary=mass_fit_summary,
            nastran_failed=nastran_failed,
        )
        payload["score"] = score_payload["score"]
        payload["score_components"] = score_payload["components"]
        payload["mass_errors"] = _build_mass_error_payload(
            config=config,
            score_payload=score_payload,
            mass_summary=mass_summary,
            parsed_nastran_mass_g=runner_result["parsed_total_mass_g"],
        )
        payload["cross_checks"] = {
            "internal_total_mass_g": mass_summary["total_mass_g"],
            "parsed_nastran_total_mass_g": runner_result["parsed_total_mass_g"],
            "internal_vs_parsed_nastran_delta_g": payload["mass_errors"]["internal_vs_parsed_nastran_delta_g"],
        }

        if extra_payload:
            for key, value in extra_payload.items():
                payload[key] = value

        if runner_result["mode"] == "dry-run":
            payload["status"] = "dry_run_completed"
        elif nastran_failed:
            payload["status"] = "failed"
        else:
            payload["status"] = "completed"
    except Exception as exc:
        payload["status"] = "failed"
        payload["exception"] = f"{type(exc).__name__}: {exc}"
        payload["notes"].append("La run fallo antes de completar el flujo normal.")
    finally:
        payload["duration_seconds"] = round(time.time() - started_at, 3)
        _write_json(run_json_path, payload)
        rebuild_project_artifacts(config)

    return payload


def ensure_run_for_factors(
    config: dict[str, Any],
    strategy: str,
    factors: dict[str, float],
    youngs_factors: dict[str, float] | None = None,
    signature_namespace: str | None = None,
    notes: list[str] | None = None,
    is_baseline_reference: bool = False,
    allow_existing: bool = True,
    launch_nastran: bool = True,
    extra_payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], bool]:
    signature = run_signature(config, factors, youngs_factors, namespace=signature_namespace)
    if allow_existing:
        existing = _existing_runs_by_signature(config).get(signature)
        if existing is not None:
            return existing, False
    return (
        _execute_run(
            config=config,
            strategy=strategy,
            factors=factors,
            youngs_factors=youngs_factors,
            signature_namespace=signature_namespace,
            notes=notes,
            is_baseline_reference=is_baseline_reference,
            launch_nastran=launch_nastran,
            extra_payload=extra_payload,
        ),
        True,
    )


def run_baseline(config: dict[str, Any]) -> dict[str, Any]:
    run, _ = ensure_run_for_factors(
        config=config,
        strategy="baseline",
        factors=baseline_factors(config),
        youngs_factors=youngs_modulus_baseline_factors(config),
        notes=["Baseline run con factores 1.0 en las seis densidades."],
        is_baseline_reference=True,
        allow_existing=False,
    )
    return run


def _run_by_name(config: dict[str, Any], run_name: str) -> dict[str, Any] | None:
    for run in load_run_records(config):
        if str(run.get("run_name")) == str(run_name):
            return run
    return None


def run_modal_baseline(config: dict[str, Any]) -> dict[str, Any]:
    base_mass_run_name = str(config["modal"]["base_mass_run_name"])
    base_mass_run = _run_by_name(config, base_mass_run_name)
    if base_mass_run is None:
        raise ValueError(
            f"Frozen mass baseline run '{base_mass_run_name}' was not found in runs/."
        )
    if not base_mass_run.get("parameter_factors"):
        raise ValueError(f"Run '{base_mass_run_name}' does not expose parameter_factors and cannot seed modal-baseline.")

    run = _execute_run(
        config=config,
        strategy="modal_baseline",
        factors={str(name): float(value) for name, value in base_mass_run["parameter_factors"].items()},
        youngs_factors=youngs_modulus_baseline_factors(config),
        notes=[
            f"Modal baseline seeded from frozen mass-fit run {base_mass_run_name}.",
            "No se reoptimiza la masa en esta fase; se usa la solucion de masa congelada como punto de partida modal.",
        ],
        is_baseline_reference=False,
        launch_nastran=True,
        extra_payload={
            "is_modal_baseline_reference": True,
            "modal_request": {
                "base_mass_run_name": base_mass_run_name,
                "reference_source": project_relative(config, config["resolved_paths"]["modal_reference_file"]),
                "reference_source_type": str(config["modal"]["reference"]["source_type"]),
                "num_modes": int(config["modal"]["num_modes"]),
            },
        },
    )
    return run


def _modal_fit_candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    fit_cfg = config["modal"].get("fit", {})
    candidate_factors = [float(value) for value in fit_cfg.get("candidate_factors", [0.95, 1.05])]
    variable_order = fit_cfg.get("variable_order") or material_names(config)
    candidates: list[dict[str, Any]] = []
    for material_name in variable_order:
        for factor in candidate_factors:
            if abs(float(factor) - 1.0) <= 1e-9:
                continue
            youngs_factors = youngs_modulus_baseline_factors(config)
            youngs_factors[str(material_name)] = round(float(factor), 6)
            candidates.append(
                {
                    "material_name": str(material_name),
                    "factor": round(float(factor), 6),
                    "youngs_factors": youngs_factors,
                }
            )
    return candidates


def _modal_run_rank_strict(run: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    modal = run.get("modal") or {}
    summary = modal.get("summary") or {}
    status_priority = 0.0 if modal.get("status") == "completed" else 1.0
    return (
        status_priority,
        -float(summary.get("mean_mac", 0.0)),
        float(summary.get("mean_frequency_error", 1e9)),
        -float(summary.get("worst_mac", 0.0)),
        float(summary.get("worst_frequency_error", 1e9)),
        -float(summary.get("good_on_both_count", 0.0)),
    )


def _modal_run_rank_balanced(run: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
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


def _modal_run_rank(run: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    return _modal_run_rank_strict(run)


def run_modal_fit(config: dict[str, Any]) -> dict[str, Any]:
    base_mass_run_name = str(config["modal"]["base_mass_run_name"])
    base_mass_run = _run_by_name(config, base_mass_run_name)
    if base_mass_run is None:
        raise ValueError(f"Frozen mass baseline run '{base_mass_run_name}' was not found in runs/.")
    frozen_density_factors = {
        str(name): float(value) for name, value in (base_mass_run.get("parameter_factors") or {}).items()
    }
    if not frozen_density_factors:
        raise ValueError(f"Run '{base_mass_run_name}' does not expose parameter_factors and cannot seed modal-fit.")

    baseline_modal_run, baseline_created = ensure_run_for_factors(
        config=config,
        strategy="modal_baseline",
        factors=frozen_density_factors,
        youngs_factors=youngs_modulus_baseline_factors(config),
        notes=[
            f"Modal baseline seeded from frozen mass-fit run {base_mass_run_name}.",
            "No se reoptimiza la masa en esta fase; se usa la solucion de masa congelada como punto de partida modal.",
        ],
        is_baseline_reference=False,
        allow_existing=True,
        launch_nastran=True,
        extra_payload={
            "is_modal_baseline_reference": True,
            "modal_request": {
                "base_mass_run_name": base_mass_run_name,
                "reference_source": project_relative(config, config["resolved_paths"]["modal_reference_file"]),
                "reference_source_type": str(config["modal"]["reference"]["source_type"]),
                "num_modes": int(config["modal"]["num_modes"]),
            },
        },
    )

    fit_cfg = config["modal"].get("fit", {})
    created_runs: list[str] = []
    reused_runs: list[str] = []
    candidate_runs: list[dict[str, Any]] = [baseline_modal_run]
    for candidate in _modal_fit_candidates(config):
        run, was_created = ensure_run_for_factors(
            config=config,
            strategy="modal_fit",
            factors=frozen_density_factors,
            youngs_factors=candidate["youngs_factors"],
            notes=[
                f"Modal-fit one-at-a-time candidate varying E only for {candidate['material_name']}.",
                f"Frozen mass baseline run: {base_mass_run_name}.",
                "Las densidades rho se mantienen congeladas respecto a la run 005; solo cambia E de una MAT1.",
            ],
            is_baseline_reference=False,
            allow_existing=True,
            launch_nastran=bool(fit_cfg.get("launch_nastran", True)),
            extra_payload={
                "modal_request": {
                    "base_mass_run_name": base_mass_run_name,
                    "reference_source": project_relative(config, config["resolved_paths"]["modal_reference_file"]),
                    "reference_source_type": str(config["modal"]["reference"]["source_type"]),
                    "num_modes": int(config["modal"]["num_modes"]),
                },
                "modal_fit": {
                    "method": str(fit_cfg.get("method", "youngs_modulus_one_at_a_time")),
                    "frozen_density_run_name": base_mass_run_name,
                    "candidate_material_name": str(candidate["material_name"]),
                    "candidate_e_factor": float(candidate["factor"]),
                    "objective_primary": "maximize mean MAC paired",
                    "objective_secondary": "minimize mean frequency error paired",
                    "mass_feasibility_frozen": True,
                },
            },
        )
        candidate_runs.append(run)
        if was_created:
            created_runs.append(str(run["run_name"]))
        else:
            reused_runs.append(str(run["run_name"]))

    artifacts = rebuild_project_artifacts(config)
    refreshed_runs = {str(run.get("run_name")): run for run in load_run_records(config)}
    refreshed_candidate_runs = [
        refreshed_runs.get(str(run.get("run_name")), run)
        for run in candidate_runs
    ]
    ranked_candidates = sorted(
        [
            run
            for run in refreshed_candidate_runs
            if run.get("strategy") == "modal_fit" and (run.get("modal") or {}).get("status") == "completed"
        ],
        key=_modal_run_rank,
    )
    best_candidate = ranked_candidates[0] if ranked_candidates else None
    candidate_ranking = []
    for rank, run in enumerate(ranked_candidates, start=1):
        modal_summary = (run.get("modal") or {}).get("summary", {})
        modal_fit = run.get("modal_fit") or {}
        candidate_ranking.append(
            {
                "rank": rank,
                "run_name": run.get("run_name"),
                "candidate_material_name": modal_fit.get("candidate_material_name"),
                "candidate_e_factor": modal_fit.get("candidate_e_factor"),
                "mean_mac": modal_summary.get("mean_mac"),
                "mean_frequency_error": modal_summary.get("mean_frequency_error"),
                "worst_mac": modal_summary.get("worst_mac"),
                "worst_frequency_error": modal_summary.get("worst_frequency_error"),
                "good_on_both_count": modal_summary.get("good_on_both_count"),
                "youngs_modulus_factors": run.get("youngs_modulus_factors", {}),
            }
        )
    return {
        "strategy": str(fit_cfg.get("method", "youngs_modulus_one_at_a_time")),
        "base_mass_run_name": base_mass_run_name,
        "baseline_modal_run_name": baseline_modal_run.get("run_name"),
        "baseline_modal_run_created": baseline_created,
        "new_runs_created": len(created_runs),
        "existing_runs_reused": len(reused_runs),
        "created_runs": created_runs,
        "reused_runs": reused_runs,
        "candidate_count": len(ranked_candidates),
        "ranking_basis": [
            "1. maximize mean MAC paired",
            "2. minimize mean frequency error paired",
            "3. maximize worst MAC paired",
            "4. minimize worst frequency error paired",
        ],
        "candidate_ranking": candidate_ranking,
        "best_candidate": None
        if best_candidate is None
        else {
            "run_name": best_candidate.get("run_name"),
            "mean_mac": (best_candidate.get("modal") or {}).get("summary", {}).get("mean_mac"),
            "mean_frequency_error": (best_candidate.get("modal") or {}).get("summary", {}).get("mean_frequency_error"),
            "youngs_modulus_factors": best_candidate.get("youngs_modulus_factors", {}),
        },
        "artifacts": artifacts,
    }


def _completed_modal_runs(
    config: dict[str, Any],
    strategies: set[str] | None = None,
) -> list[dict[str, Any]]:
    runs = []
    for run in load_run_records(config):
        if (run.get("modal") or {}).get("status") != "completed":
            continue
        if strategies and str(run.get("strategy")) not in strategies:
            continue
        runs.append(run)
    return runs


def _best_completed_modal_run(
    config: dict[str, Any],
    strategies: set[str],
    ranking: str = "strict",
) -> dict[str, Any] | None:
    runs = _completed_modal_runs(config, strategies)
    if not runs:
        return None
    ranker = _modal_run_rank_balanced if ranking == "balanced" else _modal_run_rank_strict
    return min(runs, key=ranker)


def _same_factor_map(left: dict[str, float], right: dict[str, float], tolerance: float = 1e-9) -> bool:
    keys = set(left) | set(right)
    return all(abs(float(left.get(key, 1.0)) - float(right.get(key, 1.0))) <= tolerance for key in keys)


def _find_existing_run_by_factors(
    config: dict[str, Any],
    density_factors: dict[str, float],
    youngs_factors: dict[str, float],
    strategies: set[str] | None = None,
) -> dict[str, Any] | None:
    matches: list[dict[str, Any]] = []
    candidates = load_run_records(config)
    for run in candidates:
        if strategies and str(run.get("strategy")) not in strategies:
            continue
        run_density_factors = {str(name): float(value) for name, value in (run.get("parameter_factors") or {}).items()}
        run_youngs_factors = {
            str(name): float(value)
            for name, value in (run.get("youngs_modulus_factors") or {}).items()
        }
        if not run_density_factors or not run_youngs_factors:
            continue
        if _same_factor_map(run_density_factors, density_factors) and _same_factor_map(run_youngs_factors, youngs_factors):
            matches.append(run)
    if not matches:
        return None

    def _match_rank(run: dict[str, Any]) -> tuple[float, float, float]:
        modal_status = (run.get("modal") or {}).get("status")
        run_status = str(run.get("status") or "")
        modal_completed = 0.0 if modal_status == "completed" else 1.0
        run_completed = 0.0 if run_status in {"completed", "dry_run_completed"} else 1.0
        return (modal_completed, run_completed, -float(run.get("run_index", 0)))

    return min(matches, key=_match_rank)


def _modal_metric_summary(run: dict[str, Any]) -> dict[str, float]:
    summary = (run.get("modal") or {}).get("summary", {})
    return {
        "mean_mac": float(summary.get("mean_mac", 0.0)),
        "worst_mac": float(summary.get("worst_mac", 0.0)),
        "mean_frequency_error": float(summary.get("mean_frequency_error", 0.0)),
        "worst_frequency_error": float(summary.get("worst_frequency_error", 0.0)),
        "good_on_both_count": float(summary.get("good_on_both_count", 0.0)),
    }


def _safe_relative_improvement(delta: float, reference_value: float) -> float | None:
    if abs(reference_value) <= 1e-12:
        return None
    return round(delta / reference_value, 8)


def _modal_improvement_payload(candidate_run: dict[str, Any], reference_run: dict[str, Any] | None) -> dict[str, Any] | None:
    if reference_run is None:
        return None
    candidate = _modal_metric_summary(candidate_run)
    reference = _modal_metric_summary(reference_run)
    mean_mac_delta = round(candidate["mean_mac"] - reference["mean_mac"], 8)
    worst_mac_delta = round(candidate["worst_mac"] - reference["worst_mac"], 8)
    mean_frequency_error_improvement = round(reference["mean_frequency_error"] - candidate["mean_frequency_error"], 8)
    worst_frequency_error_improvement = round(reference["worst_frequency_error"] - candidate["worst_frequency_error"], 8)
    good_on_both_delta = round(candidate["good_on_both_count"] - reference["good_on_both_count"], 8)
    return {
        "reference_run_name": str(reference_run.get("run_name")),
        "mean_mac": {
            "candidate": candidate["mean_mac"],
            "reference": reference["mean_mac"],
            "improvement_absolute": mean_mac_delta,
            "improvement_relative": _safe_relative_improvement(mean_mac_delta, reference["mean_mac"]),
        },
        "worst_mac": {
            "candidate": candidate["worst_mac"],
            "reference": reference["worst_mac"],
            "improvement_absolute": worst_mac_delta,
            "improvement_relative": _safe_relative_improvement(worst_mac_delta, reference["worst_mac"]),
        },
        "mean_frequency_error": {
            "candidate": candidate["mean_frequency_error"],
            "reference": reference["mean_frequency_error"],
            "improvement_absolute": mean_frequency_error_improvement,
            "improvement_relative": _safe_relative_improvement(
                mean_frequency_error_improvement,
                reference["mean_frequency_error"],
            ),
        },
        "worst_frequency_error": {
            "candidate": candidate["worst_frequency_error"],
            "reference": reference["worst_frequency_error"],
            "improvement_absolute": worst_frequency_error_improvement,
            "improvement_relative": _safe_relative_improvement(
                worst_frequency_error_improvement,
                reference["worst_frequency_error"],
            ),
        },
        "good_on_both_count": {
            "candidate": int(candidate["good_on_both_count"]),
            "reference": int(reference["good_on_both_count"]),
            "improvement_absolute": int(good_on_both_delta),
            "improvement_relative": _safe_relative_improvement(good_on_both_delta, reference["good_on_both_count"]),
        },
    }


def _modal_local_primary_candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    primary_grid = config["modal"]["local_fit"].get("primary_grid", {})
    if not primary_grid:
        return []
    variable_names = list(primary_grid.keys())
    factor_lists = [[round(float(value), 6) for value in primary_grid[name]] for name in variable_names]
    candidates: list[dict[str, Any]] = []
    for combination in itertools.product(*factor_lists):
        youngs_factors = youngs_modulus_baseline_factors(config)
        candidate_values: dict[str, float] = {}
        for material_name, factor in zip(variable_names, combination):
            youngs_factors[str(material_name)] = round(float(factor), 6)
            candidate_values[str(material_name)] = round(float(factor), 6)
        candidates.append(
            {
                "stage": "primary_grid",
                "candidate_values": candidate_values,
                "youngs_factors": youngs_factors,
            }
        )
    return candidates


def _modal_local_secondary_candidates(
    config: dict[str, Any],
    seed_youngs_factors: dict[str, float],
) -> list[dict[str, Any]]:
    secondary_cfg = config["modal"]["local_fit"]["secondary_stage"]
    shared_material = str(secondary_cfg["shared_material"])
    candidates: list[dict[str, Any]] = []
    for factor in secondary_cfg.get("factors", []):
        youngs_factors = dict(seed_youngs_factors)
        youngs_factors[shared_material] = round(float(factor), 6)
        candidates.append(
            {
                "stage": "secondary_shared_material",
                "candidate_values": {
                    "PCB_FPGA": round(float(seed_youngs_factors.get("PCB_FPGA", 1.0)), 6),
                    "PCB_MOD": round(float(seed_youngs_factors.get("PCB_MOD", 1.0)), 6),
                    shared_material: round(float(factor), 6),
                },
                "youngs_factors": youngs_factors,
            }
        )
    return candidates


def _modal_local_candidate_grid_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    local_fit_cfg = config["modal"]["local_fit"]
    return {
        "primary_grid": {
            str(name): [round(float(value), 6) for value in values]
            for name, values in local_fit_cfg.get("primary_grid", {}).items()
        },
        "secondary_stage": {
            "enabled": bool(local_fit_cfg["secondary_stage"].get("enabled", True)),
            "trigger": str(local_fit_cfg["secondary_stage"].get("trigger", "if_primary_completed")),
            "shared_material": str(local_fit_cfg["secondary_stage"].get("shared_material", "Bandeja_FPGA")),
            "factors": [round(float(value), 6) for value in local_fit_cfg["secondary_stage"].get("factors", [0.95, 1.0, 1.05])],
        },
    }


def _modal_local_conclusions(
    best_local_run: dict[str, Any] | None,
    modal_baseline_run: dict[str, Any] | None,
    best_one_at_a_time_run: dict[str, Any] | None,
) -> list[str]:
    if best_local_run is None:
        return ["El refinamiento local no encontro runs modales completadas."]

    notes = [
        f"La mejor run local es {best_local_run['run_name']}.",
    ]
    if modal_baseline_run is not None:
        vs_baseline = _modal_improvement_payload(best_local_run, modal_baseline_run)
        if vs_baseline is not None:
            notes.append(
                "Frente a la modal baseline, el mean MAC cambia en "
                f"{vs_baseline['mean_mac']['improvement_absolute']:+.6f} y la mejora de mean frequency error es "
                f"{vs_baseline['mean_frequency_error']['improvement_absolute']:+.6f}."
            )
    if best_one_at_a_time_run is not None:
        vs_best = _modal_improvement_payload(best_local_run, best_one_at_a_time_run)
        if vs_best is not None:
            notes.append(
                "Frente al mejor one-at-a-time, el mean MAC cambia en "
                f"{vs_best['mean_mac']['improvement_absolute']:+.6f} y la mejora de mean frequency error es "
                f"{vs_best['mean_frequency_error']['improvement_absolute']:+.6f}."
            )
    return notes


def _update_modal_local_run_payloads(
    config: dict[str, Any],
    local_runs: list[dict[str, Any]],
    modal_baseline_run: dict[str, Any] | None,
    best_one_at_a_time_run: dict[str, Any] | None,
) -> None:
    if not local_runs:
        return
    overall_rank = {str(run.get("run_name")): rank for rank, run in enumerate(sorted(local_runs, key=_modal_run_rank), start=1)}
    by_stage: dict[str, list[dict[str, Any]]] = {}
    for run in local_runs:
        stage = str((run.get("modal_local_refinement") or {}).get("stage") or "unknown")
        by_stage.setdefault(stage, []).append(run)
    stage_rankings: dict[str, dict[str, int]] = {}
    for stage, stage_runs in by_stage.items():
        stage_rankings[stage] = {
            str(run.get("run_name")): rank
            for rank, run in enumerate(sorted(stage_runs, key=_modal_run_rank), start=1)
        }

    for run in local_runs:
        source_path = run.get("_source_path")
        if not source_path:
            continue
        payload = {key: value for key, value in run.items() if key != "_source_path"}
        local_block = dict(payload.get("modal_local_refinement") or {})
        stage = str(local_block.get("stage") or "unknown")
        local_block["rank_overall"] = overall_rank.get(str(run.get("run_name")))
        local_block["rank_within_stage"] = stage_rankings.get(stage, {}).get(str(run.get("run_name")))
        local_block["comparison_vs_modal_baseline"] = _modal_improvement_payload(run, modal_baseline_run)
        local_block["comparison_vs_best_one_at_a_time"] = _modal_improvement_payload(run, best_one_at_a_time_run)
        payload["modal_local_refinement"] = local_block
        _write_json(Path(source_path), payload)


def run_modal_fit_local(config: dict[str, Any]) -> dict[str, Any]:
    base_mass_run_name = str(config["modal"]["base_mass_run_name"])
    base_mass_run = _run_by_name(config, base_mass_run_name)
    if base_mass_run is None:
        raise ValueError(f"Frozen mass baseline run '{base_mass_run_name}' was not found in runs/.")
    frozen_density_factors = {
        str(name): float(value) for name, value in (base_mass_run.get("parameter_factors") or {}).items()
    }
    if not frozen_density_factors:
        raise ValueError(f"Run '{base_mass_run_name}' does not expose parameter_factors and cannot seed modal-fit-local.")

    baseline_modal_run, baseline_created = ensure_run_for_factors(
        config=config,
        strategy="modal_baseline",
        factors=frozen_density_factors,
        youngs_factors=youngs_modulus_baseline_factors(config),
        notes=[
            f"Modal baseline seeded from frozen mass-fit run {base_mass_run_name}.",
            "No se reoptimiza la masa en esta fase; se usa la solucion de masa congelada como punto de partida modal.",
        ],
        is_baseline_reference=False,
        allow_existing=True,
        launch_nastran=True,
        extra_payload={
            "is_modal_baseline_reference": True,
            "modal_request": {
                "base_mass_run_name": base_mass_run_name,
                "reference_source": project_relative(config, config["resolved_paths"]["modal_reference_file"]),
                "reference_source_type": str(config["modal"]["reference"]["source_type"]),
                "num_modes": int(config["modal"]["num_modes"]),
            },
        },
    )

    local_fit_cfg = config["modal"]["local_fit"]
    reference_modal_fit_run_name = str(local_fit_cfg.get("reference_modal_fit_run_name") or "").strip()
    best_one_at_a_time_run = _run_by_name(config, reference_modal_fit_run_name) if reference_modal_fit_run_name else None
    if best_one_at_a_time_run is None or (best_one_at_a_time_run.get("modal") or {}).get("status") != "completed":
        best_one_at_a_time_run = _best_completed_modal_run(config, {"modal_fit"})
    if best_one_at_a_time_run is None:
        run_modal_fit(config)
        best_one_at_a_time_run = _best_completed_modal_run(config, {"modal_fit"})
    if best_one_at_a_time_run is None:
        raise ValueError("No completed modal_fit run is available to seed the local refinement comparisons.")

    created_runs: list[str] = []
    reused_runs: list[str] = []
    primary_candidate_runs: list[str] = []
    secondary_candidate_runs: list[str] = []
    candidate_grid = _modal_local_candidate_grid_snapshot(config)
    signature_namespace = "modal_fit_local"

    for candidate in _modal_local_primary_candidates(config):
        run, was_created = ensure_run_for_factors(
            config=config,
            strategy="modal_fit_local",
            factors=frozen_density_factors,
            youngs_factors=candidate["youngs_factors"],
            signature_namespace=signature_namespace,
            notes=[
                "Modal local refinement primary grid candidate.",
                f"Frozen mass baseline run: {base_mass_run_name}.",
                "Se mantienen congeladas las densidades de la run 005; solo varian E de PCB_FPGA y PCB_MOD.",
            ],
            is_baseline_reference=False,
            allow_existing=True,
            launch_nastran=bool(local_fit_cfg.get("launch_nastran", True)),
            extra_payload={
                "modal_request": {
                    "base_mass_run_name": base_mass_run_name,
                    "reference_source": project_relative(config, config["resolved_paths"]["modal_reference_file"]),
                    "reference_source_type": str(config["modal"]["reference"]["source_type"]),
                    "num_modes": int(config["modal"]["num_modes"]),
                },
                "modal_local_refinement": {
                    "method": str(local_fit_cfg.get("method", "modal_local_grid_refinement")),
                    "stage": str(candidate["stage"]),
                    "frozen_density_run_name": base_mass_run_name,
                    "reference_modal_baseline_run_name": str(baseline_modal_run.get("run_name")),
                    "reference_modal_fit_run_name": str(best_one_at_a_time_run.get("run_name")),
                    "candidate_values": candidate["candidate_values"],
                    "candidate_grid": candidate_grid,
                    "mass_feasibility_frozen": True,
                    "signature_namespace": signature_namespace,
                },
            },
        )
        primary_candidate_runs.append(str(run["run_name"]))
        if was_created:
            created_runs.append(str(run["run_name"]))
        else:
            reused_runs.append(str(run["run_name"]))

    rebuild_project_artifacts(config)
    local_runs_after_primary = [
        run
        for run in load_run_records(config)
        if str(run.get("strategy")) == "modal_fit_local" and (run.get("modal") or {}).get("status") == "completed"
    ]
    primary_completed_runs = [
        run
        for run in local_runs_after_primary
        if str((run.get("modal_local_refinement") or {}).get("stage")) == "primary_grid"
    ]
    best_primary_run = min(primary_completed_runs, key=_modal_run_rank) if primary_completed_runs else None

    secondary_cfg = local_fit_cfg["secondary_stage"]
    secondary_executed = False
    if bool(secondary_cfg.get("enabled", True)) and best_primary_run is not None:
        secondary_executed = True
        for candidate in _modal_local_secondary_candidates(config, best_primary_run.get("youngs_modulus_factors", {})):
            run, was_created = ensure_run_for_factors(
                config=config,
                strategy="modal_fit_local",
                factors=frozen_density_factors,
                youngs_factors=candidate["youngs_factors"],
                signature_namespace=signature_namespace,
                notes=[
                    "Modal local refinement secondary stage candidate.",
                    f"Primary seed run: {best_primary_run['run_name']}.",
                    "Se mantiene el refinamiento local de PCB_FPGA y PCB_MOD, variando adicionalmente E de Bandeja_FPGA.",
                ],
                is_baseline_reference=False,
                allow_existing=True,
                launch_nastran=bool(local_fit_cfg.get("launch_nastran", True)),
                extra_payload={
                    "modal_request": {
                        "base_mass_run_name": base_mass_run_name,
                        "reference_source": project_relative(config, config["resolved_paths"]["modal_reference_file"]),
                        "reference_source_type": str(config["modal"]["reference"]["source_type"]),
                        "num_modes": int(config["modal"]["num_modes"]),
                    },
                    "modal_local_refinement": {
                        "method": str(local_fit_cfg.get("method", "modal_local_grid_refinement")),
                        "stage": str(candidate["stage"]),
                        "frozen_density_run_name": base_mass_run_name,
                        "reference_modal_baseline_run_name": str(baseline_modal_run.get("run_name")),
                        "reference_modal_fit_run_name": str(best_one_at_a_time_run.get("run_name")),
                        "seed_primary_run_name": str(best_primary_run["run_name"]),
                        "candidate_values": candidate["candidate_values"],
                        "candidate_grid": candidate_grid,
                        "mass_feasibility_frozen": True,
                        "signature_namespace": signature_namespace,
                    },
                },
            )
            secondary_candidate_runs.append(str(run["run_name"]))
            if was_created:
                created_runs.append(str(run["run_name"]))
            else:
                reused_runs.append(str(run["run_name"]))

    rebuild_project_artifacts(config)
    refreshed_local_runs = [
        run
        for run in load_run_records(config)
        if str(run.get("strategy")) == "modal_fit_local" and (run.get("modal") or {}).get("status") == "completed"
    ]
    _update_modal_local_run_payloads(config, refreshed_local_runs, baseline_modal_run, best_one_at_a_time_run)
    artifacts = rebuild_project_artifacts(config)

    refreshed_local_runs = [
        run
        for run in load_run_records(config)
        if str(run.get("strategy")) == "modal_fit_local" and (run.get("modal") or {}).get("status") == "completed"
    ]
    ranked_local_runs = sorted(refreshed_local_runs, key=_modal_run_rank)
    best_local_run = ranked_local_runs[0] if ranked_local_runs else None
    local_ranking = []
    for rank, run in enumerate(ranked_local_runs, start=1):
        modal_summary = (run.get("modal") or {}).get("summary", {})
        local_block = run.get("modal_local_refinement") or {}
        local_ranking.append(
            {
                "rank": rank,
                "run_name": run.get("run_name"),
                "stage": local_block.get("stage"),
                "candidate_values": local_block.get("candidate_values", {}),
                "mean_mac": modal_summary.get("mean_mac"),
                "worst_mac": modal_summary.get("worst_mac"),
                "mean_frequency_error": modal_summary.get("mean_frequency_error"),
                "worst_frequency_error": modal_summary.get("worst_frequency_error"),
                "good_on_both_count": modal_summary.get("good_on_both_count"),
                "comparison_vs_modal_baseline": local_block.get("comparison_vs_modal_baseline"),
                "comparison_vs_best_one_at_a_time": local_block.get("comparison_vs_best_one_at_a_time"),
            }
        )

    return {
        "strategy": str(local_fit_cfg.get("method", "modal_local_grid_refinement")),
        "base_mass_run_name": base_mass_run_name,
        "baseline_modal_run_name": str(baseline_modal_run.get("run_name")),
        "baseline_modal_run_created": baseline_created,
        "reference_modal_fit_run_name": str(best_one_at_a_time_run.get("run_name")),
        "new_runs_created": len(created_runs),
        "existing_runs_reused": len(reused_runs),
        "created_runs": created_runs,
        "reused_runs": reused_runs,
        "primary_candidate_runs": primary_candidate_runs,
        "secondary_candidate_runs": secondary_candidate_runs,
        "secondary_stage_executed": secondary_executed,
        "candidate_grid": candidate_grid,
        "candidate_count": len(ranked_local_runs),
        "ranking_basis": [
            "1. maximize mean MAC paired",
            "2. minimize mean frequency error paired",
            "3. maximize worst MAC paired",
            "4. minimize worst frequency error paired",
        ],
        "best_local_run": None
        if best_local_run is None
        else {
            "run_name": best_local_run.get("run_name"),
            "stage": (best_local_run.get("modal_local_refinement") or {}).get("stage"),
            "candidate_values": (best_local_run.get("modal_local_refinement") or {}).get("candidate_values", {}),
            "mean_mac": (best_local_run.get("modal") or {}).get("summary", {}).get("mean_mac"),
            "worst_mac": (best_local_run.get("modal") or {}).get("summary", {}).get("worst_mac"),
            "mean_frequency_error": (best_local_run.get("modal") or {}).get("summary", {}).get("mean_frequency_error"),
            "worst_frequency_error": (best_local_run.get("modal") or {}).get("summary", {}).get("worst_frequency_error"),
            "good_on_both_count": (best_local_run.get("modal") or {}).get("summary", {}).get("good_on_both_count"),
            "comparison_vs_modal_baseline": (best_local_run.get("modal_local_refinement") or {}).get("comparison_vs_modal_baseline"),
            "comparison_vs_best_one_at_a_time": (best_local_run.get("modal_local_refinement") or {}).get("comparison_vs_best_one_at_a_time"),
        },
        "candidate_ranking": local_ranking,
        "conclusions": _modal_local_conclusions(best_local_run, baseline_modal_run, best_one_at_a_time_run),
        "artifacts": artifacts,
    }


def _modal_balanced_local_candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    primary_grid = config["modal"]["balanced_local_fit"].get("primary_grid", {})
    if not primary_grid:
        return []
    variable_names = list(primary_grid.keys())
    factor_lists = [[round(float(value), 6) for value in primary_grid[name]] for name in variable_names]
    candidates: list[dict[str, Any]] = []
    for combination in itertools.product(*factor_lists):
        youngs_factors = youngs_modulus_baseline_factors(config)
        candidate_values: dict[str, float] = {}
        for material_name, factor in zip(variable_names, combination):
            youngs_factors[str(material_name)] = round(float(factor), 6)
            candidate_values[str(material_name)] = round(float(factor), 6)
        candidates.append(
            {
                "stage": "balanced_primary_grid",
                "candidate_values": candidate_values,
                "youngs_factors": youngs_factors,
            }
        )
    return candidates


def _modal_balanced_candidate_grid_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    balanced_local_fit_cfg = config["modal"]["balanced_local_fit"]
    return {
        "primary_grid": {
            str(name): [round(float(value), 6) for value in values]
            for name, values in balanced_local_fit_cfg.get("primary_grid", {}).items()
        }
    }


def _candidate_runs_from_grid(
    config: dict[str, Any],
    density_factors: dict[str, float],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    matched_runs: list[dict[str, Any]] = []
    for candidate in candidates:
        run = _find_existing_run_by_factors(config, density_factors, candidate["youngs_factors"])
        if run is not None and (run.get("modal") or {}).get("status") == "completed":
            matched_runs.append(run)
    return matched_runs


def _update_modal_balanced_local_run_payloads(
    candidate_runs: list[dict[str, Any]],
    reference_balanced_run: dict[str, Any] | None,
    reference_strict_run: dict[str, Any] | None,
    modal_baseline_run: dict[str, Any] | None,
    best_one_at_a_time_run: dict[str, Any] | None,
    candidate_grid: dict[str, Any],
    method: str,
) -> None:
    if not candidate_runs:
        return

    overall_rank = {
        str(run.get("run_name")): rank
        for rank, run in enumerate(sorted(candidate_runs, key=_modal_run_rank_balanced), start=1)
    }
    strict_rank = {
        str(run.get("run_name")): rank
        for rank, run in enumerate(sorted(candidate_runs, key=_modal_run_rank_strict), start=1)
    }

    for run in candidate_runs:
        source_path = run.get("_source_path")
        if not source_path:
            continue
        payload = {key: value for key, value in run.items() if key != "_source_path"}
        balanced_block = dict(payload.get("modal_balanced_local_refinement") or {})
        if not balanced_block.get("candidate_values"):
            balanced_block["candidate_values"] = {
                material_name: round(float((payload.get("youngs_modulus_factors") or {}).get(material_name, 1.0)), 6)
                for material_name in candidate_grid.get("primary_grid", {})
            }
        balanced_block["method"] = method
        balanced_block["candidate_grid"] = candidate_grid
        balanced_block["rank_overall"] = overall_rank.get(str(run.get("run_name")))
        balanced_block["strict_rank_within_grid"] = strict_rank.get(str(run.get("run_name")))
        balanced_block["comparison_vs_reference_balanced"] = _modal_improvement_payload(run, reference_balanced_run)
        balanced_block["comparison_vs_reference_strict"] = _modal_improvement_payload(run, reference_strict_run)
        balanced_block["comparison_vs_modal_baseline"] = _modal_improvement_payload(run, modal_baseline_run)
        balanced_block["comparison_vs_best_one_at_a_time"] = _modal_improvement_payload(run, best_one_at_a_time_run)
        payload["modal_balanced_local_refinement"] = balanced_block
        _write_json(Path(source_path), payload)


def run_modal_fit_balanced_local(config: dict[str, Any]) -> dict[str, Any]:
    base_mass_run_name = str(config["modal"]["base_mass_run_name"])
    base_mass_run = _run_by_name(config, base_mass_run_name)
    if base_mass_run is None:
        raise ValueError(f"Frozen mass baseline run '{base_mass_run_name}' was not found in runs/.")
    frozen_density_factors = {
        str(name): float(value) for name, value in (base_mass_run.get("parameter_factors") or {}).items()
    }
    if not frozen_density_factors:
        raise ValueError(f"Run '{base_mass_run_name}' does not expose parameter_factors and cannot seed modal-fit-balanced-local.")

    modal_cfg = config["modal"]["balanced_local_fit"]
    reference_balanced_run_name = str(modal_cfg.get("reference_balanced_run_name") or "").strip()
    reference_balanced_run = _run_by_name(config, reference_balanced_run_name) if reference_balanced_run_name else None
    if reference_balanced_run is None or (reference_balanced_run.get("modal") or {}).get("status") != "completed":
        reference_balanced_run = _best_completed_modal_run(
            config,
            {"modal_fit", "modal_fit_local", "modal_baseline"},
            ranking="balanced",
        )
    if reference_balanced_run is None:
        raise ValueError("No completed balanced reference modal run is available to seed the balanced local refinement.")

    modal_baseline_run = _best_completed_modal_run(config, {"modal_baseline"}, ranking="strict")
    best_one_at_a_time_run = _best_completed_modal_run(config, {"modal_fit"}, ranking="strict")
    reference_strict_run = _best_completed_modal_run(config, {"modal_fit", "modal_fit_local", "modal_baseline"}, ranking="strict")

    created_runs: list[str] = []
    reused_runs: list[str] = []
    candidate_grid = _modal_balanced_candidate_grid_snapshot(config)
    signature_namespace = "modal_fit_balanced_local"
    method = str(modal_cfg.get("method", "modal_balanced_local_grid_refinement"))

    for candidate in _modal_balanced_local_candidates(config):
        existing = _find_existing_run_by_factors(config, frozen_density_factors, candidate["youngs_factors"])
        if existing is not None:
            reused_runs.append(str(existing.get("run_name")))
            continue

        run, was_created = ensure_run_for_factors(
            config=config,
            strategy="modal_fit_balanced_local",
            factors=frozen_density_factors,
            youngs_factors=candidate["youngs_factors"],
            signature_namespace=signature_namespace,
            notes=[
                "Balanced modal local refinement candidate.",
                f"Seed balanced modal run: {reference_balanced_run['run_name']}.",
                "Se mantienen congeladas las densidades de la run 005; solo varian E de PCB_MOD y PCB_FPGA alrededor del candidato equilibrado.",
            ],
            is_baseline_reference=False,
            allow_existing=True,
            launch_nastran=bool(modal_cfg.get("launch_nastran", True)),
            extra_payload={
                "modal_request": {
                    "base_mass_run_name": base_mass_run_name,
                    "reference_source": project_relative(config, config["resolved_paths"]["modal_reference_file"]),
                    "reference_source_type": str(config["modal"]["reference"]["source_type"]),
                    "num_modes": int(config["modal"]["num_modes"]),
                },
                "modal_balanced_local_refinement": {
                    "method": method,
                    "stage": str(candidate["stage"]),
                    "ranking_mode": "balanced",
                    "frozen_density_run_name": base_mass_run_name,
                    "reference_balanced_run_name": str(reference_balanced_run.get("run_name")),
                    "reference_strict_run_name": str(reference_strict_run.get("run_name")) if reference_strict_run else None,
                    "reference_modal_baseline_run_name": str(modal_baseline_run.get("run_name")) if modal_baseline_run else None,
                    "reference_best_one_at_a_time_run_name": str(best_one_at_a_time_run.get("run_name")) if best_one_at_a_time_run else None,
                    "candidate_values": candidate["candidate_values"],
                    "candidate_grid": candidate_grid,
                    "mass_feasibility_frozen": True,
                    "signature_namespace": signature_namespace,
                },
            },
        )
        if was_created:
            created_runs.append(str(run["run_name"]))
        else:
            reused_runs.append(str(run["run_name"]))

    rebuild_project_artifacts(config)
    candidate_runs = _candidate_runs_from_grid(config, frozen_density_factors, _modal_balanced_local_candidates(config))
    _update_modal_balanced_local_run_payloads(
        candidate_runs=candidate_runs,
        reference_balanced_run=reference_balanced_run,
        reference_strict_run=reference_strict_run,
        modal_baseline_run=modal_baseline_run,
        best_one_at_a_time_run=best_one_at_a_time_run,
        candidate_grid=candidate_grid,
        method=method,
    )
    artifacts = rebuild_project_artifacts(config)

    refreshed_candidates = _candidate_runs_from_grid(config, frozen_density_factors, _modal_balanced_local_candidates(config))
    ranked_candidates = sorted(refreshed_candidates, key=_modal_run_rank_balanced)
    best_candidate = ranked_candidates[0] if ranked_candidates else None

    candidate_ranking = []
    for rank, run in enumerate(ranked_candidates, start=1):
        summary = (run.get("modal") or {}).get("summary", {})
        balanced_block = run.get("modal_balanced_local_refinement") or {}
        candidate_ranking.append(
            {
                "rank": rank,
                "run_name": run.get("run_name"),
                "candidate_values": balanced_block.get("candidate_values", {}),
                "mean_mac": summary.get("mean_mac"),
                "worst_mac": summary.get("worst_mac"),
                "mean_frequency_error": summary.get("mean_frequency_error"),
                "worst_frequency_error": summary.get("worst_frequency_error"),
                "good_on_both_count": summary.get("good_on_both_count"),
                "comparison_vs_reference_balanced": balanced_block.get("comparison_vs_reference_balanced"),
                "comparison_vs_reference_strict": balanced_block.get("comparison_vs_reference_strict"),
                "comparison_vs_modal_baseline": balanced_block.get("comparison_vs_modal_baseline"),
                "comparison_vs_best_one_at_a_time": balanced_block.get("comparison_vs_best_one_at_a_time"),
            }
        )

    return {
        "strategy": method,
        "base_mass_run_name": base_mass_run_name,
        "reference_balanced_run_name": str(reference_balanced_run.get("run_name")),
        "reference_strict_run_name": str(reference_strict_run.get("run_name")) if reference_strict_run else None,
        "reference_modal_baseline_run_name": str(modal_baseline_run.get("run_name")) if modal_baseline_run else None,
        "reference_best_one_at_a_time_run_name": str(best_one_at_a_time_run.get("run_name")) if best_one_at_a_time_run else None,
        "new_runs_created": len(created_runs),
        "existing_runs_reused": len(reused_runs),
        "created_runs": created_runs,
        "reused_runs": reused_runs,
        "candidate_grid": candidate_grid,
        "candidate_count": len(ranked_candidates),
        "ranking_basis": [
            "1. maximize good_on_both",
            "2. minimize mean frequency error paired",
            "3. maximize mean MAC paired",
            "4. maximize worst MAC paired",
            "5. minimize worst frequency error paired",
        ],
        "best_local_run": None
        if best_candidate is None
        else {
            "run_name": best_candidate.get("run_name"),
            "candidate_values": (best_candidate.get("modal_balanced_local_refinement") or {}).get("candidate_values", {}),
            "mean_mac": (best_candidate.get("modal") or {}).get("summary", {}).get("mean_mac"),
            "worst_mac": (best_candidate.get("modal") or {}).get("summary", {}).get("worst_mac"),
            "mean_frequency_error": (best_candidate.get("modal") or {}).get("summary", {}).get("mean_frequency_error"),
            "worst_frequency_error": (best_candidate.get("modal") or {}).get("summary", {}).get("worst_frequency_error"),
            "good_on_both_count": (best_candidate.get("modal") or {}).get("summary", {}).get("good_on_both_count"),
            "comparison_vs_reference_balanced": (best_candidate.get("modal_balanced_local_refinement") or {}).get("comparison_vs_reference_balanced"),
            "comparison_vs_reference_strict": (best_candidate.get("modal_balanced_local_refinement") or {}).get("comparison_vs_reference_strict"),
            "comparison_vs_modal_baseline": (best_candidate.get("modal_balanced_local_refinement") or {}).get("comparison_vs_modal_baseline"),
            "comparison_vs_best_one_at_a_time": (best_candidate.get("modal_balanced_local_refinement") or {}).get("comparison_vs_best_one_at_a_time"),
        },
        "candidate_ranking": candidate_ranking,
        "artifacts": artifacts,
    }


def _targeted_modal_reference_strategies() -> set[str]:
    return {"modal_baseline", "modal_fit", "modal_fit_local", "modal_fit_balanced_local"}


def _recommended_modal_run_for_targeted(config: dict[str, Any]) -> dict[str, Any] | None:
    return _best_completed_modal_run(config, _targeted_modal_reference_strategies(), ranking="balanced")


def _paired_mode_diag_map(run: dict[str, Any]) -> dict[int, dict[str, Any]]:
    diagnostics = (run.get("modal") or {}).get("paired_mode_diagnostics") or run.get("paired_mode_diagnostics") or []
    return {int(item["reference_mode_number"]): item for item in diagnostics}


def _write_run_annotations(run: dict[str, Any], updates: dict[str, Any]) -> None:
    source_path = run.get("_source_path")
    if not source_path:
        return
    payload = {key: value for key, value in run.items() if key != "_source_path"}
    payload.update(updates)
    _write_json(Path(source_path), payload)


def _modal_local_sensitivity_candidates(
    config: dict[str, Any],
    base_youngs_factors: dict[str, float],
) -> list[dict[str, Any]]:
    sensitivity_cfg = config["modal"]["local_sensitivity"]
    perturbation_fraction = float(sensitivity_cfg.get("perturbation_fraction", 0.02))
    candidates: list[dict[str, Any]] = []
    for material_name in sensitivity_cfg.get("materials", []):
        current_factor = float(base_youngs_factors.get(str(material_name), 1.0))
        for direction, sign in [("decrease", -1.0), ("increase", 1.0)]:
            candidate_factor = round(current_factor * (1.0 + (sign * perturbation_fraction)), 6)
            if candidate_factor <= 0.0:
                continue
            youngs_factors = dict(base_youngs_factors)
            youngs_factors[str(material_name)] = candidate_factor
            candidates.append(
                {
                    "material_name": str(material_name),
                    "direction": direction,
                    "factor_before": round(current_factor, 6),
                    "factor_after": candidate_factor,
                    "factor_delta": round(candidate_factor - current_factor, 6),
                    "relative_delta_fraction": round((candidate_factor / current_factor) - 1.0, 8)
                    if abs(current_factor) > 1e-12
                    else None,
                    "youngs_factors": youngs_factors,
                }
            )
    return candidates


def _modal_sensitivity_row_rank(row: dict[str, Any]) -> tuple[float, float, float, float, float, float, float, float, float]:
    summary = row.get("modal_summary") or {}
    return (
        0.0 if str(row.get("modal_status")) == "completed" else 1.0,
        0.0 if bool(row.get("preserves_good_on_both")) else 1.0,
        -float(row.get("improved_problematic_modes_count", 0.0)),
        -float(row.get("problematic_frequency_error_improvement_total", 0.0)),
        -float(row.get("problematic_mac_improvement_total", 0.0)),
        float(summary.get("mean_frequency_error", 1e9)),
        -float(summary.get("mean_mac", 0.0)),
        -float(summary.get("worst_mac", 0.0)),
        float(summary.get("worst_frequency_error", 1e9)),
    )


def _best_effect_note(mode_effects: list[dict[str, Any]]) -> str:
    if not mode_effects:
        return "Sin efecto modal claro sobre los modos problematicos priorizados."
    ordered = sorted(
        mode_effects,
        key=lambda item: (
            max(float(item.get("frequency_error_improvement", 0.0)), 0.0)
            + max(float(item.get("delta_mac", 0.0)), 0.0),
            max(float(item.get("delta_mac", 0.0)), 0.0),
        ),
        reverse=True,
    )
    best = ordered[0]
    ref_mode = int(best["reference_mode_number"])
    freq_improvement = float(best.get("frequency_error_improvement", 0.0))
    mac_improvement = float(best.get("delta_mac", 0.0))
    if freq_improvement > 1e-9 and mac_improvement > 1e-9:
        return (
            f"Mejora simultaneamente MAC y frecuencia del modo problematico {ref_mode} "
            f"(dMAC {mac_improvement:+.4f}, dFreqErr {freq_improvement:+.4f})."
        )
    if freq_improvement > 1e-9:
        return f"Mejora sobre todo la frecuencia del modo problematico {ref_mode} (dFreqErr {freq_improvement:+.4f})."
    if mac_improvement > 1e-9:
        return f"Mejora sobre todo el MAC del modo problematico {ref_mode} (dMAC {mac_improvement:+.4f})."
    return f"No mejora claramente el modo problematico {ref_mode}; el efecto dominante es debil o neutro."


def _modal_local_sensitivity_summary(
    config: dict[str, Any],
    base_run: dict[str, Any],
    candidate_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    base_modal = base_run.get("modal") or {}
    base_summary = base_modal.get("summary") or {}
    base_diag_map = _paired_mode_diag_map(base_run)
    problematic_modes = (base_modal.get("problematic_modes") or {}).get("top_problematic", [])
    problematic_reference_modes = [int(item["reference_mode_number"]) for item in problematic_modes]
    base_good_on_both = int(base_summary.get("good_on_both_count", 0))

    rows: list[dict[str, Any]] = []
    for run in candidate_runs:
        run_modal = run.get("modal") or {}
        run_summary = run_modal.get("summary") or {}
        sensitivity_block = run.get("modal_local_sensitivity") or {}
        run_diag_map = _paired_mode_diag_map(run)
        mode_effects: list[dict[str, Any]] = []
        improved_problematic_modes_count = 0
        problematic_mac_improvement_total = 0.0
        problematic_frequency_error_improvement_total = 0.0
        for reference_mode_number in problematic_reference_modes:
            base_diag = base_diag_map.get(int(reference_mode_number))
            run_diag = run_diag_map.get(int(reference_mode_number))
            if not base_diag or not run_diag:
                continue
            delta_mac = round(float(run_diag.get("paired_mac", 0.0)) - float(base_diag.get("paired_mac", 0.0)), 8)
            frequency_error_improvement = round(
                float(base_diag.get("frequency_error_rel", 0.0)) - float(run_diag.get("frequency_error_rel", 0.0)),
                8,
            )
            delta_frequency_error = round(-frequency_error_improvement, 8)
            if delta_mac > 0.0 and frequency_error_improvement > 0.0:
                improved_problematic_modes_count += 1
            problematic_mac_improvement_total += max(delta_mac, 0.0)
            problematic_frequency_error_improvement_total += max(frequency_error_improvement, 0.0)
            mode_effects.append(
                {
                    "reference_mode_number": int(reference_mode_number),
                    "base_model_mode_number": base_diag.get("model_mode_number"),
                    "candidate_model_mode_number": run_diag.get("model_mode_number"),
                    "base_mac": base_diag.get("paired_mac"),
                    "candidate_mac": run_diag.get("paired_mac"),
                    "delta_mac": delta_mac,
                    "base_frequency_error_rel": base_diag.get("frequency_error_rel"),
                    "candidate_frequency_error_rel": run_diag.get("frequency_error_rel"),
                    "delta_frequency_error": delta_frequency_error,
                    "frequency_error_improvement": frequency_error_improvement,
                    "base_classification": base_diag.get("classification"),
                    "candidate_classification": run_diag.get("classification"),
                    "base_failure_driver": base_diag.get("failure_driver"),
                    "candidate_failure_driver": run_diag.get("failure_driver"),
                }
            )
        row = {
            "run_name": str(run.get("run_name")),
            "material_name": str(sensitivity_block.get("material_name") or "n/a"),
            "direction": str(sensitivity_block.get("direction") or "n/a"),
            "factor_before": sensitivity_block.get("factor_before"),
            "factor_after": sensitivity_block.get("factor_after"),
            "factor_delta": sensitivity_block.get("factor_delta"),
            "relative_delta_fraction": sensitivity_block.get("relative_delta_fraction"),
            "modal_status": str(run_modal.get("status") or run.get("status") or "unknown"),
            "modal_summary": {
                "mean_mac": run_summary.get("mean_mac"),
                "worst_mac": run_summary.get("worst_mac"),
                "mean_frequency_error": run_summary.get("mean_frequency_error"),
                "worst_frequency_error": run_summary.get("worst_frequency_error"),
                "good_on_both_count": run_summary.get("good_on_both_count"),
            },
            "delta_mean_mac": round(
                float(run_summary.get("mean_mac", 0.0)) - float(base_summary.get("mean_mac", 0.0)),
                8,
            ),
            "delta_worst_mac": round(
                float(run_summary.get("worst_mac", 0.0)) - float(base_summary.get("worst_mac", 0.0)),
                8,
            ),
            "delta_mean_frequency_error": round(
                float(run_summary.get("mean_frequency_error", 0.0)) - float(base_summary.get("mean_frequency_error", 0.0)),
                8,
            ),
            "delta_worst_frequency_error": round(
                float(run_summary.get("worst_frequency_error", 0.0)) - float(base_summary.get("worst_frequency_error", 0.0)),
                8,
            ),
            "frequency_error_direction": (
                "frequency error decreased"
                if float(run_summary.get("mean_frequency_error", 0.0)) < float(base_summary.get("mean_frequency_error", 0.0)) - 1e-12
                else "frequency error increased"
                if float(run_summary.get("mean_frequency_error", 0.0)) > float(base_summary.get("mean_frequency_error", 0.0)) + 1e-12
                else "frequency error unchanged"
            ),
            "delta_good_on_both": int(run_summary.get("good_on_both_count", 0)) - base_good_on_both,
            "preserves_good_on_both": int(run_summary.get("good_on_both_count", 0)) >= base_good_on_both,
            "problematic_mode_effects": mode_effects,
            "improved_problematic_modes_count": improved_problematic_modes_count,
            "problematic_mac_improvement_total": round(problematic_mac_improvement_total, 8),
            "problematic_frequency_error_improvement_total": round(problematic_frequency_error_improvement_total, 8),
        }
        row["effect_note"] = _best_effect_note(mode_effects)
        rows.append(row)

    rows.sort(key=_modal_sensitivity_row_rank)

    material_summaries: list[dict[str, Any]] = []
    for material_name in config["modal"]["local_sensitivity"].get("materials", []):
        material_rows = [row for row in rows if row["material_name"] == str(material_name)]
        if not material_rows:
            continue
        best_row = min(material_rows, key=_modal_sensitivity_row_rank)
        material_summaries.append(
            {
                "material_name": str(material_name),
                "recommended_direction": best_row["direction"],
                "factor_before": best_row["factor_before"],
                "recommended_factor": best_row["factor_after"],
                "run_name": best_row["run_name"],
                "effect_note": best_row["effect_note"],
                "preserves_good_on_both": best_row["preserves_good_on_both"],
                "improved_problematic_modes_count": best_row["improved_problematic_modes_count"],
                "problematic_frequency_error_improvement_total": best_row["problematic_frequency_error_improvement_total"],
                "problematic_mac_improvement_total": best_row["problematic_mac_improvement_total"],
            }
        )

    return {
        "strategy": str(config["modal"]["local_sensitivity"].get("method", "modal_local_sensitivity")),
        "base_run_name": str(base_run.get("run_name")),
        "perturbation_fraction": float(config["modal"]["local_sensitivity"].get("perturbation_fraction", 0.02)),
        "materials": [str(name) for name in config["modal"]["local_sensitivity"].get("materials", [])],
        "base_summary": {
            "mean_mac": base_summary.get("mean_mac"),
            "worst_mac": base_summary.get("worst_mac"),
            "mean_frequency_error": base_summary.get("mean_frequency_error"),
            "worst_frequency_error": base_summary.get("worst_frequency_error"),
            "good_on_both_count": base_summary.get("good_on_both_count"),
        },
        "problematic_modes_reference": problematic_modes,
        "rows": rows,
        "material_summaries": material_summaries,
    }


def _build_targeted_recommendation(
    config: dict[str, Any],
    base_run: dict[str, Any],
    sensitivity_summary: dict[str, Any],
) -> dict[str, Any]:
    base_youngs_factors = {
        str(name): float(value)
        for name, value in (base_run.get("youngs_modulus_factors") or {}).items()
    }
    ranked_material_rows = sorted(
        sensitivity_summary.get("material_summaries", []),
        key=lambda row: (
            0.0 if bool(row.get("preserves_good_on_both")) else 1.0,
            -float(row.get("improved_problematic_modes_count", 0.0)),
            -float(row.get("problematic_frequency_error_improvement_total", 0.0)),
            -float(row.get("problematic_mac_improvement_total", 0.0)),
        ),
    )

    selected_moves: list[dict[str, Any]] = []
    for row in ranked_material_rows:
        if bool(row.get("preserves_good_on_both")) and (
            float(row.get("problematic_frequency_error_improvement_total", 0.0)) > 1e-9
            or float(row.get("problematic_mac_improvement_total", 0.0)) > 1e-9
            or int(row.get("improved_problematic_modes_count", 0)) > 0
        ):
            selected_moves.append(
                {
                    "material_name": str(row["material_name"]),
                    "current_factor": round(float(row["factor_before"]), 6),
                    "recommended_factor": round(float(row["recommended_factor"]), 6),
                    "direction": str(row["recommended_direction"]),
                    "source_run_name": str(row["run_name"]),
                    "effect_note": str(row["effect_note"]),
                    "improved_problematic_modes_count": int(row.get("improved_problematic_modes_count", 0)),
                    "problematic_frequency_error_improvement_total": float(
                        row.get("problematic_frequency_error_improvement_total", 0.0)
                    ),
                    "problematic_mac_improvement_total": float(row.get("problematic_mac_improvement_total", 0.0)),
                }
            )

    if not selected_moves and ranked_material_rows:
        fallback = ranked_material_rows[0]
        selected_moves.append(
            {
                "material_name": str(fallback["material_name"]),
                "current_factor": round(float(fallback["factor_before"]), 6),
                "recommended_factor": round(float(fallback["recommended_factor"]), 6),
                "direction": str(fallback["recommended_direction"]),
                "source_run_name": str(fallback["run_name"]),
                "effect_note": str(fallback["effect_note"]),
                "improved_problematic_modes_count": int(fallback.get("improved_problematic_modes_count", 0)),
                "problematic_frequency_error_improvement_total": float(
                    fallback.get("problematic_frequency_error_improvement_total", 0.0)
                ),
                "problematic_mac_improvement_total": float(fallback.get("problematic_mac_improvement_total", 0.0)),
            }
        )

    max_moves = max(1, int(config["modal"]["targeted_fit"].get("max_parameter_moves_per_candidate", 2)))
    selected_moves = selected_moves[: max(1, min(3, max_moves + 1))]
    proposed_primary_candidate = dict(base_youngs_factors)
    for move in selected_moves[:max_moves]:
        proposed_primary_candidate[str(move["material_name"])] = float(move["recommended_factor"])

    notes = []
    if selected_moves:
        notes.append(
            "La recomendacion automatica se construye con la sensibilidad modal local alrededor de la run recomendada y prioriza preservar good_on_both."
        )
        for move in selected_moves[:max_moves]:
            notes.append(
                f"Probar {move['material_name']} en {move['recommended_factor']:.6f} ({move['direction']}) porque {move['effect_note']}"
            )
    else:
        notes.append("No se detecto un movimiento claramente beneficioso; conviene mantener la run recomendada actual.")

    return {
        "base_run_name": str(base_run.get("run_name")),
        "selected_moves": selected_moves,
        "max_parameter_moves_per_candidate": max_moves,
        "proposed_primary_candidate": proposed_primary_candidate,
        "notes": notes,
    }


def _modal_targeted_candidates(
    config: dict[str, Any],
    base_youngs_factors: dict[str, float],
    targeted_recommendation: dict[str, Any],
) -> list[dict[str, Any]]:
    max_candidates = max(1, int(config["modal"]["targeted_fit"].get("max_candidates", 5)))
    max_moves = max(1, int(config["modal"]["targeted_fit"].get("max_parameter_moves_per_candidate", 2)))
    selected_moves = list(targeted_recommendation.get("selected_moves", []))
    if not selected_moves:
        return []

    candidates: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()

    def add_candidate(move_indexes: list[int]) -> None:
        if len(move_indexes) > max_moves:
            return
        candidate_factors = dict(base_youngs_factors)
        candidate_values: dict[str, float] = {}
        moves: list[dict[str, Any]] = []
        for index in move_indexes:
            move = selected_moves[index]
            candidate_factors[str(move["material_name"])] = float(move["recommended_factor"])
            candidate_values[str(move["material_name"])] = float(move["recommended_factor"])
            moves.append(move)
        if _same_factor_map(candidate_factors, base_youngs_factors):
            return
        signature = youngs_signature(config, candidate_factors)
        if signature in seen_signatures:
            return
        seen_signatures.add(signature)
        candidates.append(
            {
                "youngs_factors": candidate_factors,
                "candidate_values": candidate_values,
                "moves": moves,
            }
        )

    move_count = len(selected_moves)
    if move_count >= 1:
        add_candidate([0])
    if move_count >= 2:
        add_candidate([1])
        add_candidate([0, 1])
    if move_count >= 3:
        add_candidate([2])
        add_candidate([0, 2])
    if move_count >= 3 and max_moves >= 3:
        add_candidate([0, 1, 2])

    return candidates[:max_candidates]


def _update_modal_targeted_run_payloads(
    candidate_runs: list[dict[str, Any]],
    base_recommended_run: dict[str, Any],
    targeted_recommendation: dict[str, Any],
) -> None:
    if not candidate_runs:
        return

    overall_rank = {
        str(run.get("run_name")): rank
        for rank, run in enumerate(sorted(candidate_runs, key=_modal_run_rank_balanced), start=1)
    }
    strict_rank = {
        str(run.get("run_name")): rank
        for rank, run in enumerate(sorted(candidate_runs, key=_modal_run_rank_strict), start=1)
    }

    for run in candidate_runs:
        source_path = run.get("_source_path")
        if not source_path:
            continue
        payload = {key: value for key, value in run.items() if key != "_source_path"}
        targeted_block = dict(payload.get("modal_targeted_fit") or {})
        targeted_block["rank_overall"] = overall_rank.get(str(run.get("run_name")))
        targeted_block["strict_rank_within_targeted"] = strict_rank.get(str(run.get("run_name")))
        targeted_block["comparison_vs_recommended"] = _modal_improvement_payload(run, base_recommended_run)
        targeted_block["targeted_recommendation"] = targeted_recommendation
        payload["modal_targeted_fit"] = targeted_block
        _write_json(Path(source_path), payload)


def run_modal_fit_targeted(config: dict[str, Any]) -> dict[str, Any]:
    base_mass_run_name = str(config["modal"]["base_mass_run_name"])
    base_mass_run = _run_by_name(config, base_mass_run_name)
    if base_mass_run is None:
        raise ValueError(f"Frozen mass baseline run '{base_mass_run_name}' was not found in runs/.")
    frozen_density_factors = {
        str(name): float(value) for name, value in (base_mass_run.get("parameter_factors") or {}).items()
    }
    if not frozen_density_factors:
        raise ValueError(f"Run '{base_mass_run_name}' does not expose parameter_factors and cannot seed modal-fit-targeted.")

    recommended_run = _recommended_modal_run_for_targeted(config)
    if recommended_run is None:
        raise ValueError("No completed balanced modal run is available to seed the targeted diagnostic phase.")

    base_youngs_factors = {
        str(name): float(value)
        for name, value in (recommended_run.get("youngs_modulus_factors") or {}).items()
    }
    if not base_youngs_factors:
        raise ValueError(f"Run '{recommended_run['run_name']}' does not expose youngs_modulus_factors.")

    local_sensitivity_cfg = config["modal"]["local_sensitivity"]
    sensitivity_created_runs: list[str] = []
    sensitivity_reused_runs: list[str] = []
    sensitivity_candidates = _modal_local_sensitivity_candidates(config, base_youngs_factors)
    for candidate in sensitivity_candidates:
        existing = _find_existing_run_by_factors(config, frozen_density_factors, candidate["youngs_factors"])
        if existing is not None and (existing.get("modal") or {}).get("status") == "completed":
            sensitivity_reused_runs.append(str(existing.get("run_name")))
            continue
        run, was_created = ensure_run_for_factors(
            config=config,
            strategy="modal_local_sensitivity",
            factors=frozen_density_factors,
            youngs_factors=candidate["youngs_factors"],
            notes=[
                f"Local modal sensitivity around recommended run {recommended_run['run_name']}.",
                f"Perturbacion {candidate['direction']} sobre {candidate['material_name']} con delta relativo {candidate['relative_delta_fraction']:+.2%}.",
                "Las densidades permanecen congeladas respecto a la run 005.",
            ],
            is_baseline_reference=False,
            allow_existing=False if existing is not None else True,
            launch_nastran=bool(local_sensitivity_cfg.get("launch_nastran", True)),
            extra_payload={
                "modal_request": {
                    "base_mass_run_name": base_mass_run_name,
                    "reference_source": project_relative(config, config["resolved_paths"]["modal_reference_file"]),
                    "reference_source_type": str(config["modal"]["reference"]["source_type"]),
                    "num_modes": int(config["modal"]["num_modes"]),
                },
                "modal_local_sensitivity": {
                    "method": str(local_sensitivity_cfg.get("method", "modal_local_sensitivity")),
                    "base_recommended_run_name": str(recommended_run["run_name"]),
                    "material_name": str(candidate["material_name"]),
                    "direction": str(candidate["direction"]),
                    "factor_before": float(candidate["factor_before"]),
                    "factor_after": float(candidate["factor_after"]),
                    "factor_delta": float(candidate["factor_delta"]),
                    "relative_delta_fraction": candidate["relative_delta_fraction"],
                    "perturbation_fraction": float(local_sensitivity_cfg.get("perturbation_fraction", 0.02)),
                    "mass_feasibility_frozen": True,
                },
            },
        )
        if was_created:
            sensitivity_created_runs.append(str(run["run_name"]))
        else:
            sensitivity_reused_runs.append(str(run["run_name"]))

    rebuild_project_artifacts(config)
    sensitivity_runs = []
    for candidate in sensitivity_candidates:
        run = _find_existing_run_by_factors(config, frozen_density_factors, candidate["youngs_factors"])
        if run is not None and (run.get("modal") or {}).get("status") == "completed":
            sensitivity_runs.append(run)

    sensitivity_summary = _modal_local_sensitivity_summary(config, recommended_run, sensitivity_runs)
    targeted_recommendation = _build_targeted_recommendation(config, recommended_run, sensitivity_summary)
    _write_run_annotations(
        recommended_run,
        {
            "modal_local_sensitivity": sensitivity_summary,
            "targeted_recommendation": targeted_recommendation,
        },
    )

    targeted_cfg = config["modal"]["targeted_fit"]
    targeted_candidates = _modal_targeted_candidates(config, base_youngs_factors, targeted_recommendation)
    created_runs: list[str] = []
    reused_runs: list[str] = []
    for candidate in targeted_candidates:
        existing = _find_existing_run_by_factors(config, frozen_density_factors, candidate["youngs_factors"])
        if existing is not None and (existing.get("modal") or {}).get("status") == "completed":
            _write_run_annotations(
                existing,
                {
                    "modal_targeted_fit": {
                        "method": str(targeted_cfg.get("method", "modal_targeted_micro_refinement")),
                        "base_recommended_run_name": str(recommended_run["run_name"]),
                        "candidate_values": candidate["candidate_values"],
                        "moves": candidate["moves"],
                        "ranking_mode": "balanced",
                        "mass_feasibility_frozen": True,
                    }
                },
            )
            reused_runs.append(str(existing.get("run_name")))
            continue
        run, was_created = ensure_run_for_factors(
            config=config,
            strategy="modal_fit_targeted",
            factors=frozen_density_factors,
            youngs_factors=candidate["youngs_factors"],
            notes=[
                f"Targeted modal candidate around recommended run {recommended_run['run_name']}.",
                "La combinacion se propone automaticamente a partir de la sensibilidad modal local de +/-2%.",
                "Las densidades permanecen congeladas respecto a la run 005.",
            ],
            is_baseline_reference=False,
            allow_existing=False if existing is not None else True,
            launch_nastran=bool(targeted_cfg.get("launch_nastran", True)),
            extra_payload={
                "modal_request": {
                    "base_mass_run_name": base_mass_run_name,
                    "reference_source": project_relative(config, config["resolved_paths"]["modal_reference_file"]),
                    "reference_source_type": str(config["modal"]["reference"]["source_type"]),
                    "num_modes": int(config["modal"]["num_modes"]),
                },
                "modal_targeted_fit": {
                    "method": str(targeted_cfg.get("method", "modal_targeted_micro_refinement")),
                    "base_recommended_run_name": str(recommended_run["run_name"]),
                    "candidate_values": candidate["candidate_values"],
                    "moves": candidate["moves"],
                    "ranking_mode": "balanced",
                    "mass_feasibility_frozen": True,
                },
            },
        )
        if was_created:
            created_runs.append(str(run["run_name"]))
        else:
            reused_runs.append(str(run["run_name"]))

    rebuild_project_artifacts(config)
    targeted_runs: list[dict[str, Any]] = []
    for candidate in targeted_candidates:
        run = _find_existing_run_by_factors(config, frozen_density_factors, candidate["youngs_factors"])
        if run is not None and (run.get("modal") or {}).get("status") == "completed":
            targeted_runs.append(run)

    _update_modal_targeted_run_payloads(targeted_runs, recommended_run, targeted_recommendation)
    artifacts = rebuild_project_artifacts(config)

    refreshed_targeted_runs: list[dict[str, Any]] = []
    for candidate in targeted_candidates:
        run = _find_existing_run_by_factors(config, frozen_density_factors, candidate["youngs_factors"])
        if run is not None and (run.get("modal") or {}).get("status") == "completed":
            refreshed_targeted_runs.append(run)
    refreshed_targeted_runs = sorted(refreshed_targeted_runs, key=_modal_run_rank_balanced)
    best_targeted_run = refreshed_targeted_runs[0] if refreshed_targeted_runs else None

    comparison_vs_recommended = _modal_improvement_payload(best_targeted_run, recommended_run) if best_targeted_run else None
    _write_run_annotations(
        recommended_run,
        {
            "modal_local_sensitivity": sensitivity_summary,
            "targeted_recommendation": targeted_recommendation,
            "best_targeted_modal_run": None if best_targeted_run is None else str(best_targeted_run.get("run_name")),
        },
    )
    artifacts = rebuild_project_artifacts(config)

    candidate_ranking = []
    for rank, run in enumerate(refreshed_targeted_runs, start=1):
        targeted_block = run.get("modal_targeted_fit") or {}
        summary = (run.get("modal") or {}).get("summary", {})
        candidate_ranking.append(
            {
                "rank": rank,
                "run_name": str(run.get("run_name")),
                "candidate_values": targeted_block.get("candidate_values", {}),
                "moves": targeted_block.get("moves", []),
                "mean_mac": summary.get("mean_mac"),
                "worst_mac": summary.get("worst_mac"),
                "mean_frequency_error": summary.get("mean_frequency_error"),
                "worst_frequency_error": summary.get("worst_frequency_error"),
                "good_on_both_count": summary.get("good_on_both_count"),
                "comparison_vs_recommended": targeted_block.get("comparison_vs_recommended"),
            }
        )

    return {
        "strategy": str(targeted_cfg.get("method", "modal_targeted_micro_refinement")),
        "base_mass_run_name": base_mass_run_name,
        "base_recommended_run_name": str(recommended_run.get("run_name")),
        "local_sensitivity": {
            "new_runs_created": len(sensitivity_created_runs),
            "existing_runs_reused": len(sensitivity_reused_runs),
            "created_runs": sensitivity_created_runs,
            "reused_runs": sensitivity_reused_runs,
            "summary": sensitivity_summary,
        },
        "targeted_recommendation": targeted_recommendation,
        "new_runs_created": len(created_runs),
        "existing_runs_reused": len(reused_runs),
        "created_runs": created_runs,
        "reused_runs": reused_runs,
        "candidate_count": len(refreshed_targeted_runs),
        "candidate_ranking": candidate_ranking,
        "best_targeted_modal_run": None
        if best_targeted_run is None
        else {
            "run_name": str(best_targeted_run.get("run_name")),
            "mean_mac": (best_targeted_run.get("modal") or {}).get("summary", {}).get("mean_mac"),
            "worst_mac": (best_targeted_run.get("modal") or {}).get("summary", {}).get("worst_mac"),
            "mean_frequency_error": (best_targeted_run.get("modal") or {}).get("summary", {}).get("mean_frequency_error"),
            "worst_frequency_error": (best_targeted_run.get("modal") or {}).get("summary", {}).get("worst_frequency_error"),
            "good_on_both_count": (best_targeted_run.get("modal") or {}).get("summary", {}).get("good_on_both_count"),
            "comparison_vs_recommended": comparison_vs_recommended,
        },
        "artifacts": artifacts,
    }


def _explicit_list_candidates(config: dict[str, Any]) -> list[dict[str, float]]:
    candidates: list[dict[str, float]] = []
    for item in config["search_strategy"]["explicit_list"]:
        if isinstance(item, dict) and "factors" in item:
            candidates.append(_normalize_factor_overrides(config, item["factors"]))
        elif isinstance(item, dict):
            candidates.append(_normalize_factor_overrides(config, item))
    return candidates


def _linspace_candidates(config: dict[str, Any]) -> list[dict[str, float]]:
    search = config["search_strategy"]["linspace_range"]
    default = search["default"]
    baseline = baseline_factors(config)
    candidates: list[dict[str, float]] = []
    for material in material_sequence(config):
        name = str(material["name"])
        material_cfg = search["per_material"].get(name, default)
        for factor in _linspace(
            float(material_cfg["start"]),
            float(material_cfg["stop"]),
            int(material_cfg["count"]),
        ):
            candidate = dict(baseline)
            candidate[name] = float(factor)
            candidates.append(candidate)
    return candidates


def _cartesian_candidates(config: dict[str, Any]) -> list[dict[str, float]]:
    search = config["search_strategy"]["cartesian_grid"]
    factor_lists: list[list[float]] = []
    for material in material_sequence(config):
        name = str(material["name"])
        values = search["per_material"].get(name, search["default_factors"])
        factor_lists.append([float(value) for value in values])

    combination_count = math.prod(len(values) for values in factor_lists)
    if combination_count > int(search["max_combinations"]):
        raise ValueError(
            f"Cartesian grid would generate {combination_count} combinations, above max_combinations={search['max_combinations']}."
        )

    candidates = []
    for combination in itertools.product(*factor_lists):
        candidate = {}
        for material, factor in zip(material_sequence(config), combination):
            candidate[str(material["name"])] = float(factor)
        candidates.append(candidate)
    return candidates


def _score_of(run: dict[str, Any]) -> float:
    score = run.get("score")
    return float(score) if score is not None else float("inf")


def _feasibility_rank_of_run(run: dict[str, Any]) -> tuple[float, float, float, float]:
    mass_fit = run.get("mass_fit") or {}
    ranking = mass_fit.get("ranking_key")
    if ranking:
        return (float(ranking[0]), float(ranking[1]), float(ranking[2]), float(ranking[3]))
    total_rel_error = float(run.get("mass_errors", {}).get("total_mass_rel_error", 1e9))
    return (1e9, 1e9, total_rel_error, _score_of(run))


def _evaluate_internal_candidate(
    config: dict[str, Any],
    base_text: str,
    factors: dict[str, float],
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    signature = factor_signature(config, factors)
    if signature in cache:
        return cache[signature]

    edited_text, updates = apply_mat1_factors(
        original_text=base_text,
        target_materials=config["materials"],
        density_factors=factors,
        youngs_modulus_factors=youngs_modulus_baseline_factors(config),
    )
    mass_summary = estimate_masses(edited_text, config)
    mass_fit_summary = build_mass_fit_summary(
        mass_summary,
        config,
        factors=factors,
        baseline_reference_factors=baseline_factors(config),
    )
    cache[signature] = {
        "factors": {name: round(float(value), 6) for name, value in factors.items()},
        "factor_signature": signature,
        "material_updates": updates,
        "mass_summary": mass_summary,
        "mass_fit": mass_fit_summary,
    }
    return cache[signature]


def _material_bounds(config: dict[str, Any], material_name: str) -> tuple[float, float]:
    feasibility_cfg = config["search_strategy"]["feasibility_with_slack"]
    default_bounds = feasibility_cfg["factor_bounds"]["default"]
    material_bounds = feasibility_cfg["factor_bounds"]["per_material"].get(material_name, {})
    return (
        float(material_bounds.get("min", default_bounds["min"])),
        float(material_bounds.get("max", default_bounds["max"])),
    )


def _control_rank_tuple(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        0.0 if row["within_band"] else 1.0,
        float(row["normalized_violation"]),
        float(row["normalized_target_error"]),
    )


def _control_attention_score(row: dict[str, Any]) -> float:
    if not row["within_band"]:
        return 1000.0 + float(row["normalized_violation"])
    return float(row["normalized_target_error"])


def _material_influenced_controls(config: dict[str, Any], material_name: str) -> list[str]:
    influenced: list[str] = []
    for control_key, control_cfg in config["mass_fit_controls"].items():
        if material_name in control_cfg["direct_materials"]:
            influenced.append(str(control_key))
        if any(material_name == str(item["name"]) for item in control_cfg["shared_materials"]):
            influenced.append(str(control_key))
    return sorted(set(influenced))


def calculate_proportional_factors(
    current_factors: dict[str, float],
    directional_limits: dict[str, dict[str, float]],
    progress: float,
) -> dict[str, float]:
    candidate = dict(current_factors)
    for material_name, limit in directional_limits.items():
        current_value = float(current_factors[material_name])
        limit_value = float(limit["factor_limit"])
        candidate[material_name] = round(current_value + (float(progress) * (limit_value - current_value)), 6)
    return candidate


def _partial_controls_within_band(candidate: dict[str, Any]) -> bool:
    return all(row["within_band"] for row in candidate["mass_fit"]["control_rows"] if row["control_key"] != "total")


def _solve_factor_for_control(
    config: dict[str, Any],
    base_text: str,
    current_candidate: dict[str, Any],
    material_name: str,
    control_key: str,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    steps = int(config["search_strategy"]["feasibility_with_slack"]["binary_search_steps"])
    lower_factor, upper_factor = _material_bounds(config, material_name)
    target_g = float(config["mass_fit_targets"][control_key]["target_g"])

    def evaluate_factor(value: float) -> dict[str, Any]:
        candidate_factors = dict(current_candidate["factors"])
        candidate_factors[material_name] = round(float(value), 6)
        return _evaluate_internal_candidate(config, base_text, candidate_factors, cache)

    low_candidate = evaluate_factor(lower_factor)
    high_candidate = evaluate_factor(upper_factor)
    low_actual = float(low_candidate["mass_fit"]["controls_by_key"][control_key]["actual_g"])
    high_actual = float(high_candidate["mass_fit"]["controls_by_key"][control_key]["actual_g"])

    if target_g <= low_actual:
        return low_candidate
    if target_g >= high_actual:
        return high_candidate

    left = lower_factor
    right = upper_factor
    best_candidate = current_candidate
    for _ in range(steps):
        middle = (left + right) / 2.0
        mid_candidate = evaluate_factor(middle)
        mid_actual = float(mid_candidate["mass_fit"]["controls_by_key"][control_key]["actual_g"])
        best_candidate = mid_candidate
        if mid_actual < target_g:
            left = middle
        else:
            right = middle
    return best_candidate


def _directional_limit_for_material(
    config: dict[str, Any],
    base_text: str,
    current_candidate: dict[str, Any],
    material_name: str,
    direction: str,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    steps = int(config["search_strategy"]["feasibility_with_slack"]["binary_search_steps"])
    current_factor = float(current_candidate["factors"][material_name])
    lower_factor, upper_factor = _material_bounds(config, material_name)
    bound_factor = upper_factor if direction == "up" else lower_factor
    if abs(bound_factor - current_factor) <= 1e-9:
        return {"material_name": material_name, "factor_limit": current_factor, "slack_total_mass_g": 0.0}

    influenced_controls = _material_influenced_controls(config, material_name)
    if not influenced_controls:
        candidate_factors = dict(current_candidate["factors"])
        candidate_factors[material_name] = bound_factor
        limit_candidate = _evaluate_internal_candidate(config, base_text, candidate_factors, cache)
        return {
            "material_name": material_name,
            "factor_limit": round(bound_factor, 6),
            "slack_total_mass_g": round(
                abs(float(limit_candidate["mass_summary"]["total_mass_g"]) - float(current_candidate["mass_summary"]["total_mass_g"])),
                6,
            ),
        }

    def evaluate_factor(value: float) -> dict[str, Any]:
        candidate_factors = dict(current_candidate["factors"])
        candidate_factors[material_name] = round(float(value), 6)
        return _evaluate_internal_candidate(config, base_text, candidate_factors, cache)

    def controls_still_within_band(candidate: dict[str, Any]) -> bool:
        return all(
            candidate["mass_fit"]["controls_by_key"][control_key]["within_band"]
            for control_key in influenced_controls
        )

    bound_candidate = evaluate_factor(bound_factor)
    if controls_still_within_band(bound_candidate):
        limit_candidate = bound_candidate
    else:
        low = current_factor
        high = bound_factor
        if direction == "down":
            low, high = bound_factor, current_factor
        feasible_factor = current_factor
        limit_candidate = current_candidate
        for _ in range(steps):
            middle = (low + high) / 2.0
            mid_candidate = evaluate_factor(middle)
            if controls_still_within_band(mid_candidate):
                feasible_factor = middle
                limit_candidate = mid_candidate
                if direction == "up":
                    low = middle
                else:
                    high = middle
            else:
                if direction == "up":
                    high = middle
                else:
                    low = middle
        bound_factor = feasible_factor

    return {
        "material_name": material_name,
        "factor_limit": round(bound_factor, 6),
        "slack_total_mass_g": round(
            abs(float(limit_candidate["mass_summary"]["total_mass_g"]) - float(current_candidate["mass_summary"]["total_mass_g"])),
            6,
        ),
    }


def _select_start_candidate(
    config: dict[str, Any],
    base_text: str,
    cache: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    baseline_candidate = _evaluate_internal_candidate(config, base_text, baseline_factors(config), cache)
    best_candidate = baseline_candidate
    source = "baseline"

    if not config["search_strategy"]["feasibility_with_slack"].get("start_from_best_existing", True):
        return best_candidate, source

    for run in load_run_records(config):
        factors = run.get("parameter_factors") or {}
        if not factors:
            continue
        candidate = _evaluate_internal_candidate(config, base_text, factors, cache)
        if is_better_feasibility(candidate["mass_fit"], best_candidate["mass_fit"]):
            best_candidate = candidate
            source = str(run.get("run_name"))

    return best_candidate, source


def _pick_primary_control(candidate: dict[str, Any], control_keys: list[str]) -> str | None:
    if not control_keys:
        return None
    best_row = max(
        (candidate["mass_fit"]["controls_by_key"][control_key] for control_key in control_keys),
        key=_control_attention_score,
    )
    return str(best_row["control_key"])


def _phase_a_adjust_variable(
    config: dict[str, Any],
    base_text: str,
    current_candidate: dict[str, Any],
    material_name: str,
    cache: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    influenced_controls = _material_influenced_controls(config, material_name)
    if not influenced_controls:
        return current_candidate, None

    primary_control_key = _pick_primary_control(current_candidate, influenced_controls)
    if primary_control_key is None:
        return current_candidate, None

    current_row = current_candidate["mass_fit"]["controls_by_key"][primary_control_key]
    solved_candidate = _solve_factor_for_control(
        config,
        base_text,
        current_candidate,
        material_name,
        primary_control_key,
        cache,
    )
    solved_row = solved_candidate["mass_fit"]["controls_by_key"][primary_control_key]

    current_rank = _control_rank_tuple(current_row)
    solved_rank = _control_rank_tuple(solved_row)
    if solved_rank >= current_rank and not is_better_feasibility(solved_candidate["mass_fit"], current_candidate["mass_fit"]):
        return current_candidate, None

    if abs(float(solved_candidate["factors"][material_name]) - float(current_candidate["factors"][material_name])) <= 1e-9:
        return current_candidate, None

    step = {
        "phase": "A",
        "material_name": material_name,
        "control_key": primary_control_key,
        "control_label": solved_row["label"],
        "factor_before": round(float(current_candidate["factors"][material_name]), 6),
        "factor_after": round(float(solved_candidate["factors"][material_name]), 6),
        "actual_before_g": round(float(current_row["actual_g"]), 6),
        "actual_after_g": round(float(solved_row["actual_g"]), 6),
        "target_g": round(float(solved_row["target_g"]), 6),
        "reason": f"Ajuste independiente de {material_name} para acercar {solved_row['label']} al centro de su banda.",
    }
    return solved_candidate, step


def _phase_b_total_correction(
    config: dict[str, Any],
    base_text: str,
    current_candidate: dict[str, Any],
    cache: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    total_row = current_candidate["mass_fit"]["controls_by_key"]["total"]
    if total_row["within_band"]:
        return current_candidate, None

    direction = "down" if float(total_row["actual_g"]) > float(total_row["upper_bound_g"]) else "up"
    directional_limits: dict[str, dict[str, float]] = {}
    for material in material_sequence(config):
        material_name = str(material["name"])
        limit = _directional_limit_for_material(
            config,
            base_text,
            current_candidate,
            material_name,
            direction,
            cache,
        )
        if abs(float(limit["factor_limit"]) - float(current_candidate["factors"][material_name])) > 1e-6:
            directional_limits[material_name] = limit

    if not directional_limits:
        return current_candidate, {
            "phase": "B",
            "direction": direction,
            "applied": False,
            "reason": "No hay slack disponible para corregir la masa total sin romper restricciones parciales.",
        }

    def evaluate_progress(progress: float) -> dict[str, Any]:
        candidate_factors = calculate_proportional_factors(current_candidate["factors"], directional_limits, progress)
        return _evaluate_internal_candidate(config, base_text, candidate_factors, cache)

    steps = int(config["search_strategy"]["feasibility_with_slack"]["binary_search_steps"])
    low = 0.0
    high = 1.0
    feasible_progress = 0.0
    feasible_candidate = current_candidate
    if _partial_controls_within_band(evaluate_progress(1.0)):
        feasible_progress = 1.0
        feasible_candidate = evaluate_progress(1.0)
    else:
        for _ in range(steps):
            middle = (low + high) / 2.0
            candidate = evaluate_progress(middle)
            if _partial_controls_within_band(candidate):
                feasible_progress = middle
                feasible_candidate = candidate
                low = middle
            else:
                high = middle

    total_target_g = float(config["mass_fit_targets"]["total"]["target_g"])
    chosen_progress = feasible_progress
    chosen_candidate = feasible_candidate
    if feasible_progress > 0.0:
        target_low = 0.0
        target_high = feasible_progress
        for _ in range(steps):
            middle = (target_low + target_high) / 2.0
            candidate = evaluate_progress(middle)
            actual_total = float(candidate["mass_fit"]["controls_by_key"]["total"]["actual_g"])
            chosen_progress = middle
            chosen_candidate = candidate
            if direction == "up":
                if actual_total < total_target_g:
                    target_low = middle
                else:
                    target_high = middle
            else:
                if actual_total > total_target_g:
                    target_low = middle
                else:
                    target_high = middle
        if not is_better_feasibility(chosen_candidate["mass_fit"], current_candidate["mass_fit"]):
            chosen_progress = feasible_progress
            chosen_candidate = feasible_candidate

    if abs(float(chosen_progress)) <= 1e-9:
        return current_candidate, {
            "phase": "B",
            "direction": direction,
            "applied": False,
            "reason": "La correccion proporcional no ha encontrado una mejora factible.",
            "eligible_variables": list(directional_limits),
        }

    step = {
        "phase": "B",
        "direction": direction,
        "applied": True,
        "progress": round(float(chosen_progress), 6),
        "mass_before_g": round(float(current_candidate["mass_summary"]["total_mass_g"]), 6),
        "mass_after_g": round(float(chosen_candidate["mass_summary"]["total_mass_g"]), 6),
        "eligible_variables": [
            {
                "material_name": material_name,
                "factor_before": round(float(current_candidate["factors"][material_name]), 6),
                "factor_limit": round(float(limit["factor_limit"]), 6),
                "slack_total_mass_g": round(float(limit["slack_total_mass_g"]), 6),
            }
            for material_name, limit in directional_limits.items()
        ],
        "reason": "Correccion proporcional de masa total repartida segun el slack disponible de cada variable.",
    }
    return chosen_candidate, step


def _persist_feasibility_candidate(
    config: dict[str, Any],
    candidate: dict[str, Any],
    pass_index: int,
    iteration_history: list[dict[str, Any]],
    notes: list[str],
) -> dict[str, Any]:
    run, _ = ensure_run_for_factors(
        config=config,
        strategy="feasibility_with_slack",
        factors=candidate["factors"],
        notes=notes,
        is_baseline_reference=False,
        allow_existing=True,
        launch_nastran=bool(config["search_strategy"]["feasibility_with_slack"].get("run_nastran_for_accepted_candidates", True)),
        extra_payload={
            "feasibility_fit": {
                "strategy": "feasibility_with_slack",
                "accepted_pass": pass_index,
                "accepted_from_internal_search": True,
                "iteration_history": iteration_history,
                "best_internal_summary": candidate["mass_fit"],
            }
        },
    )
    return run


def run_feasibility_fit(config: dict[str, Any]) -> dict[str, Any]:
    base_bdf_path = Path(config["resolved_paths"]["base_bdf"])
    base_text = base_bdf_path.read_text(encoding="utf-8", errors="ignore")
    cache: dict[str, dict[str, Any]] = {}
    feasibility_cfg = config["search_strategy"]["feasibility_with_slack"]
    start_candidate, start_source = _select_start_candidate(config, base_text, cache)
    current_candidate = start_candidate
    best_candidate = start_candidate
    best_feasible_candidate = start_candidate if start_candidate["mass_fit"]["feasible"] else None
    iteration_history: list[dict[str, Any]] = []
    accepted_run_names: list[str] = []
    max_passes = int(feasibility_cfg["max_passes"])
    max_persisted = int(feasibility_cfg["max_persisted_candidates"])
    persisted = 0
    variable_order = feasibility_cfg["variable_order"] or material_names(config)
    stagnation = 0

    for pass_index in range(1, max_passes + 1):
        pass_started = current_candidate
        steps: list[dict[str, Any]] = []

        for material_name in variable_order:
            current_candidate, step = _phase_a_adjust_variable(
                config,
                base_text,
                current_candidate,
                material_name,
                cache,
            )
            if step:
                steps.append(step)

        current_candidate, correction_step = _phase_b_total_correction(
            config,
            base_text,
            current_candidate,
            cache,
        )
        if correction_step:
            steps.append(correction_step)

        if is_better_feasibility(current_candidate["mass_fit"], best_candidate["mass_fit"]):
            best_candidate = current_candidate
        if current_candidate["mass_fit"]["feasible"] and (
            best_feasible_candidate is None or is_better_feasibility(current_candidate["mass_fit"], best_feasible_candidate["mass_fit"])
        ):
            best_feasible_candidate = current_candidate

        iteration_entry = {
            "pass_index": pass_index,
            "factors": current_candidate["factors"],
            "feasible": current_candidate["mass_fit"]["feasible"],
            "ranking_key": current_candidate["mass_fit"]["ranking_key"],
            "constraints_satisfied_count": current_candidate["mass_fit"]["constraints_satisfied_count"],
            "constraint_count": current_candidate["mass_fit"]["constraint_count"],
            "most_limiting_variable": current_candidate["mass_fit"]["most_limiting_variable"],
            "steps": steps,
        }
        iteration_history.append(iteration_entry)

        improved_this_pass = is_better_feasibility(current_candidate["mass_fit"], pass_started["mass_fit"])
        if improved_this_pass and feasibility_cfg.get("persist_intermediate_runs", True) and persisted < max_persisted:
            run = _persist_feasibility_candidate(
                config,
                current_candidate,
                pass_index,
                list(iteration_history),
                notes=[
                    f"Feasibility fit accepted candidate at pass {pass_index}.",
                    f"Search started from {start_source}.",
                ],
            )
            accepted_run_names.append(str(run["run_name"]))
            persisted += 1

        if current_candidate["mass_fit"]["feasible"]:
            break

        if not steps or not improved_this_pass:
            stagnation += 1
        else:
            stagnation = 0

        if stagnation >= 1:
            break

    final_candidate = best_feasible_candidate or best_candidate
    final_run = _persist_feasibility_candidate(
        config,
        final_candidate,
        len(iteration_history),
        list(iteration_history),
        notes=[
            "Final feasibility-fit candidate persisted.",
            f"Search started from {start_source}.",
            "Se ha usado el estimador interno para explorar y la infraestructura de runs para persistir los candidatos aceptados.",
        ],
    )
    if str(final_run["run_name"]) not in accepted_run_names:
        accepted_run_names.append(str(final_run["run_name"]))

    artifacts = rebuild_project_artifacts(config)
    return {
        "strategy": "feasibility_with_slack",
        "start_source": start_source,
        "passes_executed": len(iteration_history),
        "accepted_runs": accepted_run_names,
        "best_run_name": final_run["run_name"],
        "feasible_found": bool(final_candidate["mass_fit"]["feasible"]),
        "constraints_satisfied_count": int(final_candidate["mass_fit"]["constraints_satisfied_count"]),
        "constraint_count": int(final_candidate["mass_fit"]["constraint_count"]),
        "most_limiting_variable": final_candidate["mass_fit"]["most_limiting_variable"],
        "artifacts": artifacts,
    }


def run_sweep(config: dict[str, Any]) -> dict[str, Any]:
    method = str(config["search_strategy"]["method"])
    if method == "feasibility_with_slack":
        return run_feasibility_fit(config)

    max_new_runs = int(config["limits"]["max_runs"])
    created = 0
    reused = 0
    launched_runs: list[str] = []
    skipped_duplicates: list[str] = []

    def use_run(
        strategy: str,
        factors: dict[str, float],
        notes: list[str] | None = None,
        is_baseline_reference: bool = False,
    ) -> dict[str, Any]:
        nonlocal created, reused
        run, was_created = ensure_run_for_factors(
            config=config,
            strategy=strategy,
            factors=factors,
            notes=notes,
            is_baseline_reference=is_baseline_reference,
            allow_existing=True,
        )
        if was_created:
            created += 1
            launched_runs.append(str(run["run_name"]))
        else:
            reused += 1
            skipped_duplicates.append(str(run["run_name"]))
        return run

    if method == "coordinate_descent_simple":
        current = None
        baseline_run = use_run(
            strategy="baseline",
            factors=baseline_factors(config),
            notes=["Baseline reutilizada o creada como punto de arranque del sweep."],
            is_baseline_reference=True,
        )
        current = baseline_run
        best_existing = None
        if config["search_strategy"]["coordinate_descent_simple"].get("start_from_best_existing", True):
            best_existing = next(iter(sorted(load_run_records(config), key=_score_of)), None)
        if best_existing is not None and _score_of(best_existing) < _score_of(current):
            current = best_existing

        descent_cfg = config["search_strategy"]["coordinate_descent_simple"]
        variable_order = descent_cfg["variable_order"] or material_names(config)
        candidates = [float(value) for value in descent_cfg["candidate_factors"]]
        max_passes = int(descent_cfg["max_passes"])

        for pass_index in range(max_passes):
            improved = False
            for variable_name in variable_order:
                if created >= max_new_runs:
                    break
                local_runs = []
                for factor in candidates:
                    candidate_factors = dict(current["parameter_factors"])
                    candidate_factors[variable_name] = factor
                    local_runs.append(
                        use_run(
                            strategy="coordinate_descent_simple",
                            factors=candidate_factors,
                            notes=[f"Coordinate descent pass {pass_index + 1}, variable {variable_name}."],
                        )
                    )
                local_best = min(local_runs, key=_score_of)
                if _score_of(local_best) + 1e-12 < _score_of(current):
                    current = local_best
                    improved = True
            if not improved or created >= max_new_runs:
                break
    else:
        if method == "explicit_list":
            candidates = _explicit_list_candidates(config)
        elif method == "linspace_range":
            candidates = _linspace_candidates(config)
        elif method == "cartesian_grid":
            candidates = _cartesian_candidates(config)
        else:
            raise ValueError(f"Unknown sweep method: {method}")

        for candidate in candidates:
            if created >= max_new_runs:
                break
            use_run(
                strategy=method,
                factors=candidate,
                notes=[f"Candidate generated by {method}."],
            )

    artifacts = rebuild_project_artifacts(config)
    return {
        "method": method,
        "new_runs_created": created,
        "existing_runs_reused": reused,
        "launched_runs": launched_runs,
        "skipped_duplicates": skipped_duplicates,
        "artifacts": artifacts,
    }
