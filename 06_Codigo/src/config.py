from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "project.yaml"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _resolve_path(project_root: Path, raw_value: str) -> Path:
    path = Path(raw_value)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path


def project_relative(config: dict[str, Any], path: str | Path) -> str:
    raw_path = Path(path)
    try:
        return raw_path.resolve().relative_to(config["project_root"]).as_posix()
    except ValueError:
        return str(raw_path.resolve())


def _ensure_search_defaults(config: dict[str, Any]) -> None:
    search = config.setdefault("search_strategy", {})
    search.setdefault("method", "feasibility_with_slack")

    explicit_list = search.setdefault("explicit_list", [])
    if explicit_list is None:
        search["explicit_list"] = []

    linspace_range = search.setdefault("linspace_range", {})
    linspace_default = linspace_range.setdefault("default", {})
    linspace_default.setdefault("start", 0.85)
    linspace_default.setdefault("stop", 1.15)
    linspace_default.setdefault("count", 5)
    linspace_range.setdefault("per_material", {})

    cartesian = search.setdefault("cartesian_grid", {})
    cartesian.setdefault("default_factors", [0.95, 1.0, 1.05])
    cartesian.setdefault("per_material", {})
    cartesian.setdefault("max_combinations", 150)

    descent = search.setdefault("coordinate_descent_simple", {})
    descent.setdefault("candidate_factors", [0.85, 0.925, 1.0, 1.075, 1.15])
    descent.setdefault("max_passes", 2)
    descent.setdefault("variable_order", [])
    descent.setdefault("start_from_best_existing", True)

    feasibility = search.setdefault("feasibility_with_slack", {})
    feasibility.setdefault("max_passes", 6)
    feasibility.setdefault("binary_search_steps", 18)
    feasibility.setdefault("control_improvement_tolerance", 1e-4)
    feasibility.setdefault("max_persisted_candidates", 4)
    feasibility.setdefault("persist_intermediate_runs", True)
    feasibility.setdefault("run_nastran_for_accepted_candidates", True)
    feasibility.setdefault("start_from_best_existing", True)
    feasibility.setdefault("variable_order", [])
    factor_bounds = feasibility.setdefault("factor_bounds", {})
    default_bounds = factor_bounds.setdefault("default", {})
    default_bounds.setdefault("min", 0.5)
    default_bounds.setdefault("max", 3.5)
    factor_bounds.setdefault("per_material", {})


def _normalize_shared_materials(raw_items: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw_item in raw_items or []:
        if isinstance(raw_item, str):
            normalized.append({"name": str(raw_item), "weight": 1.0})
            continue
        if isinstance(raw_item, dict):
            normalized.append(
                {
                    "name": str(raw_item["name"]),
                    "weight": float(raw_item.get("weight", 1.0)),
                }
            )
    return normalized


def load_project_config(config_path: str | Path | None = None) -> dict[str, Any]:
    config_file = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not config_file.is_absolute():
        config_file = (PROJECT_ROOT / config_file).resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    config.setdefault("project", {})
    config.setdefault("paths", {})
    config["paths"].setdefault("job_name", "sfem_mass_fit")
    config["paths"].setdefault("base_bdf", "../02_SFEM-de-apoyo/SFEM_EUE.bdf")
    config["paths"].setdefault("runs_dir", "runs")
    config["paths"].setdefault("reports_dir", "reports/runs")
    config["paths"].setdefault("handoff_dir", "handoff")

    config.setdefault("runner", {})
    config["runner"].setdefault("command", 'nastran.exe "{bdf}"')
    config["runner"].setdefault("dry_run", False)
    config["runner"].setdefault("timeout_seconds", 600)

    config.setdefault("modal", {})
    modal = config["modal"]
    if not modal:
        raise ValueError("config/project.yaml must define a modal section for phase 2.")
    modal.setdefault("base_mass_run_name", "")
    modal.setdefault("num_modes", 10)
    modal.setdefault("mac_good_threshold", 0.8)
    modal.setdefault("frequency_error_good_threshold", 0.05)
    modal.setdefault("pairing_frequency_penalty_weight", 0.25)
    modal.setdefault("result_source_preference", [".f06"])
    modal.setdefault("measurement_points", [])
    modal.setdefault("modal_vector_order", [])
    modal.setdefault("reference", {})
    modal.setdefault("ranking_weights", {})
    modal["ranking_weights"].setdefault("mean_mac", 4.0)
    modal["ranking_weights"].setdefault("worst_mac", 2.0)
    modal["ranking_weights"].setdefault("mean_frequency_error", 2.0)
    modal["ranking_weights"].setdefault("worst_frequency_error", 1.0)
    modal["ranking_weights"].setdefault("threshold_hits", 1.0)
    modal.setdefault("fit", {})
    modal["fit"].setdefault("method", "youngs_modulus_one_at_a_time")
    modal["fit"].setdefault("candidate_factors", [0.95, 1.05])
    modal["fit"].setdefault("variable_order", [])
    modal["fit"].setdefault("launch_nastran", True)
    modal.setdefault("local_fit", {})
    modal["local_fit"].setdefault("method", "modal_local_grid_refinement")
    modal["local_fit"].setdefault("launch_nastran", True)
    modal["local_fit"].setdefault("reference_modal_fit_run_name", "")
    modal["local_fit"].setdefault("primary_grid", {})
    modal["local_fit"].setdefault("secondary_stage", {})
    modal["local_fit"]["secondary_stage"].setdefault("enabled", True)
    modal["local_fit"]["secondary_stage"].setdefault("trigger", "if_primary_completed")
    modal["local_fit"]["secondary_stage"].setdefault("shared_material", "Bandeja_FPGA")
    modal["local_fit"]["secondary_stage"].setdefault("factors", [0.95, 1.0, 1.05])
    modal.setdefault("balanced_local_fit", {})
    modal["balanced_local_fit"].setdefault("method", "modal_balanced_local_grid_refinement")
    modal["balanced_local_fit"].setdefault("launch_nastran", True)
    modal["balanced_local_fit"].setdefault("reference_balanced_run_name", "")
    modal["balanced_local_fit"].setdefault("primary_grid", {})
    modal.setdefault("diagnostics", {})
    modal["diagnostics"].setdefault("reference_run_name", "")
    modal["diagnostics"].setdefault("dominant_dofs_per_mode", 3)
    modal["diagnostics"].setdefault("problematic_top_n", 3)
    modal["diagnostics"].setdefault("good", {})
    modal["diagnostics"]["good"].setdefault("mac_min", modal["mac_good_threshold"])
    modal["diagnostics"]["good"].setdefault("frequency_error_max", modal["frequency_error_good_threshold"])
    modal["diagnostics"].setdefault("acceptable", {})
    modal["diagnostics"]["acceptable"].setdefault("mac_min", 0.6)
    modal["diagnostics"]["acceptable"].setdefault("frequency_error_max", 0.20)
    modal.setdefault("local_sensitivity", {})
    modal["local_sensitivity"].setdefault("method", "modal_local_sensitivity")
    modal["local_sensitivity"].setdefault("launch_nastran", True)
    modal["local_sensitivity"].setdefault("perturbation_fraction", 0.02)
    modal["local_sensitivity"].setdefault("materials", [])
    modal.setdefault("targeted_fit", {})
    modal["targeted_fit"].setdefault("method", "modal_targeted_micro_refinement")
    modal["targeted_fit"].setdefault("launch_nastran", True)
    modal["targeted_fit"].setdefault("max_candidates", 5)
    modal["targeted_fit"].setdefault("max_parameter_moves_per_candidate", 2)

    config.setdefault("naming", {})
    config["naming"].setdefault("decimals", 3)

    config.setdefault("limits", {})
    config["limits"].setdefault("max_runs", 40)

    config.setdefault("score", {})
    weights = config["score"].setdefault("weights", {})
    weights.setdefault("unsatisfied_constraints", 1000.0)
    weights.setdefault("max_normalized_violation", 100.0)
    weights.setdefault("total_mass_rel_error", 10.0)
    weights.setdefault("change_from_baseline", 1.0)
    weights.setdefault("failure_penalty", 10000.0)

    materials = config.setdefault("materials", [])
    if len(materials) != 6:
        raise ValueError("config/project.yaml must define exactly 6 variable MAT1 materials.")

    seen_material_ids: set[int] = set()
    seen_material_names: set[str] = set()
    for material in materials:
        material_id = int(material["id"])
        material_name = str(material["name"])
        if material_id in seen_material_ids:
            raise ValueError(f"Duplicate material id in config: {material_id}")
        if material_name in seen_material_names:
            raise ValueError(f"Duplicate material name in config: {material_name}")
        seen_material_ids.add(material_id)
        seen_material_names.add(material_name)
        material.setdefault("tag", material_name)
        material.setdefault("base_rho", None)
        material.setdefault("role", "independent_density")
        raw_targets = material.get("comparison_targets")
        if raw_targets is None:
            raw_targets = []
        elif isinstance(raw_targets, str):
            raw_targets = [raw_targets]
        material["comparison_targets"] = [str(item) for item in raw_targets if str(item).strip()]
        material.setdefault("interpretation", "")

    modal_fit_order = [str(item) for item in modal["fit"].get("variable_order", [])]
    if modal_fit_order:
        known_names = {str(item["name"]) for item in materials}
        unknown_modal_fit_names = [name for name in modal_fit_order if name not in known_names]
        if unknown_modal_fit_names:
            raise ValueError(
                "modal.fit.variable_order contains unknown materials: "
                + ", ".join(sorted(unknown_modal_fit_names))
            )

    local_fit_cfg = modal["local_fit"]
    primary_grid = {str(name): [float(value) for value in values] for name, values in local_fit_cfg.get("primary_grid", {}).items()}
    local_fit_cfg["primary_grid"] = primary_grid
    if primary_grid:
        known_names = {str(item["name"]) for item in materials}
        unknown_primary_grid_names = [name for name in primary_grid if name not in known_names]
        if unknown_primary_grid_names:
            raise ValueError(
                "modal.local_fit.primary_grid contains unknown materials: "
                + ", ".join(sorted(unknown_primary_grid_names))
            )
    secondary_shared_material = str(local_fit_cfg["secondary_stage"].get("shared_material", "Bandeja_FPGA"))
    if secondary_shared_material not in {str(item["name"]) for item in materials}:
        raise ValueError(
            "modal.local_fit.secondary_stage.shared_material contains an unknown material: "
            + secondary_shared_material
        )
    local_fit_cfg["secondary_stage"]["shared_material"] = secondary_shared_material
    local_fit_cfg["secondary_stage"]["factors"] = [
        float(value) for value in local_fit_cfg["secondary_stage"].get("factors", [0.95, 1.0, 1.05])
    ]

    balanced_local_fit_cfg = modal["balanced_local_fit"]
    balanced_primary_grid = {
        str(name): [float(value) for value in values]
        for name, values in balanced_local_fit_cfg.get("primary_grid", {}).items()
    }
    balanced_local_fit_cfg["primary_grid"] = balanced_primary_grid
    if balanced_primary_grid:
        known_names = {str(item["name"]) for item in materials}
        unknown_balanced_grid_names = [name for name in balanced_primary_grid if name not in known_names]
        if unknown_balanced_grid_names:
            raise ValueError(
                "modal.balanced_local_fit.primary_grid contains unknown materials: "
                + ", ".join(sorted(unknown_balanced_grid_names))
            )

    diagnostics_cfg = modal["diagnostics"]
    diagnostics_cfg["dominant_dofs_per_mode"] = int(diagnostics_cfg.get("dominant_dofs_per_mode", 3))
    diagnostics_cfg["problematic_top_n"] = int(diagnostics_cfg.get("problematic_top_n", 3))
    diagnostics_cfg["good"]["mac_min"] = float(diagnostics_cfg["good"].get("mac_min", modal["mac_good_threshold"]))
    diagnostics_cfg["good"]["frequency_error_max"] = float(
        diagnostics_cfg["good"].get("frequency_error_max", modal["frequency_error_good_threshold"])
    )
    diagnostics_cfg["acceptable"]["mac_min"] = float(diagnostics_cfg["acceptable"].get("mac_min", 0.6))
    diagnostics_cfg["acceptable"]["frequency_error_max"] = float(
        diagnostics_cfg["acceptable"].get("frequency_error_max", 0.20)
    )

    local_sensitivity_cfg = modal["local_sensitivity"]
    local_sensitivity_cfg["perturbation_fraction"] = float(local_sensitivity_cfg.get("perturbation_fraction", 0.02))
    local_sensitivity_cfg["materials"] = [str(name) for name in local_sensitivity_cfg.get("materials", [])]
    if local_sensitivity_cfg["materials"]:
        known_names = {str(item["name"]) for item in materials}
        unknown_sensitivity_names = [
            name for name in local_sensitivity_cfg["materials"] if name not in known_names
        ]
        if unknown_sensitivity_names:
            raise ValueError(
                "modal.local_sensitivity.materials contains unknown materials: "
                + ", ".join(sorted(unknown_sensitivity_names))
            )

    targeted_fit_cfg = modal["targeted_fit"]
    targeted_fit_cfg["max_candidates"] = int(targeted_fit_cfg.get("max_candidates", 5))
    targeted_fit_cfg["max_parameter_moves_per_candidate"] = int(
        targeted_fit_cfg.get("max_parameter_moves_per_candidate", 2)
    )

    targets = config.setdefault("targets", {})
    controls = config.setdefault("mass_fit_controls", {})
    targets_cfg = config.setdefault("mass_fit_targets", {})
    if not controls:
        raise ValueError("config/project.yaml must define mass_fit_controls.")
    if not targets_cfg:
        raise ValueError("config/project.yaml must define mass_fit_targets.")

    known_material_names = {str(item["name"]) for item in materials}
    normalized_controls: dict[str, Any] = {}
    for control_key, control_cfg in controls.items():
        direct_materials = [str(item) for item in control_cfg.get("direct_materials", [])]
        shared_materials = _normalize_shared_materials(control_cfg.get("shared_materials", []))
        for material_name in direct_materials:
            if material_name not in known_material_names:
                raise ValueError(f"Unknown material '{material_name}' in mass_fit_controls.{control_key}.direct_materials")
        for shared_item in shared_materials:
            if shared_item["name"] not in known_material_names:
                raise ValueError(f"Unknown material '{shared_item['name']}' in mass_fit_controls.{control_key}.shared_materials")
        normalized_controls[str(control_key)] = {
            "label": str(control_cfg.get("label", control_key)),
            "direct_materials": direct_materials,
            "shared_materials": shared_materials,
        }
    config["mass_fit_controls"] = normalized_controls

    normalized_targets: dict[str, Any] = {}
    for target_key, target_cfg in targets_cfg.items():
        normalized_targets[str(target_key)] = {
            "label": str(target_cfg.get("label", target_key)),
            "target_g": float(target_cfg["target_g"]),
            "tolerance_fraction": float(target_cfg.get("tolerance_fraction", 0.05)),
        }
    config["mass_fit_targets"] = normalized_targets

    control_keys = set(normalized_controls)
    target_keys = set(normalized_targets)
    partial_target_keys = target_keys - {"total"}
    if control_keys != partial_target_keys:
        raise ValueError("mass_fit_controls keys must match mass_fit_targets keys except for 'total'.")
    if "total" not in normalized_targets:
        raise ValueError("mass_fit_targets must include a 'total' entry.")

    targets["mass_budget_targets_g"] = {
        normalized_targets[key]["label"]: float(normalized_targets[key]["target_g"])
        for key in normalized_controls
    }
    targets["total_mass_g"] = float(normalized_targets["total"]["target_g"])
    targets["mass_budget_target_sum_g"] = float(sum(targets["mass_budget_targets_g"].values()))
    targets["mass_budget_target_total_delta_g"] = float(
        targets["mass_budget_target_sum_g"] - float(targets["total_mass_g"])
    )

    config.setdefault("sensitivity", {})
    config["sensitivity"].setdefault("local_variation_fraction", 0.05)

    _ensure_search_defaults(config)

    modal_reference = modal.setdefault("reference", {})
    reference_path = modal_reference.get("path")
    if not reference_path:
        raise ValueError("config/project.yaml modal.reference.path is required.")
    if not modal["base_mass_run_name"]:
        raise ValueError("config/project.yaml modal.base_mass_run_name is required.")
    if len(modal["measurement_points"]) != 4:
        raise ValueError("config/project.yaml modal.measurement_points must define exactly 4 fixed nodes.")
    if len(modal["modal_vector_order"]) != 12:
        raise ValueError("config/project.yaml modal.modal_vector_order must define exactly 12 vector components.")

    measurement_point_ids = {int(item["node_id"]) for item in modal["measurement_points"]}
    for point in modal["measurement_points"]:
        point["label"] = str(point["label"])
        point["node_id"] = int(point["node_id"])
        point["coordinates"] = [float(value) for value in point.get("coordinates", [])]

    reference_measurement_points = modal_reference.setdefault("measurement_points", [])
    if reference_measurement_points:
        normalized_reference_points: list[dict[str, Any]] = []
        seen_reference_ids: set[int] = set()
        mapped_target_ids: set[int] = set()
        for item in reference_measurement_points:
            source_node_id = int(item["source_node_id"])
            target_node_id = int(item["target_node_id"])
            if source_node_id in seen_reference_ids:
                raise ValueError(f"Duplicate source_node_id {source_node_id} in modal.reference.measurement_points.")
            if target_node_id not in measurement_point_ids:
                raise ValueError(
                    f"modal.reference.measurement_points maps to unknown target node {target_node_id}."
                )
            seen_reference_ids.add(source_node_id)
            mapped_target_ids.add(target_node_id)
            normalized_reference_points.append(
                {
                    "label": str(item.get("label", source_node_id)),
                    "source_node_id": source_node_id,
                    "target_node_id": target_node_id,
                }
            )
        if mapped_target_ids != measurement_point_ids:
            raise ValueError(
                "modal.reference.measurement_points must cover the same 4 canonical measurement nodes as modal.measurement_points."
            )
        modal_reference["measurement_points"] = normalized_reference_points

    for item in modal["modal_vector_order"]:
        item["label"] = str(item["label"])
        item["node_id"] = int(item["node_id"])
        item["component"] = str(item["component"])
        if item["node_id"] not in measurement_point_ids:
            raise ValueError(
                f"modal.modal_vector_order references node {item['node_id']} which is not present in modal.measurement_points."
            )
        if item["component"] not in {"T1", "T2", "T3"}:
            raise ValueError("modal.modal_vector_order components must be one of T1, T2 or T3.")

    resolved_paths = {
        "project_root": PROJECT_ROOT.resolve(),
        "config_file": config_file.resolve(),
        "base_bdf": _resolve_path(PROJECT_ROOT, str(config["paths"]["base_bdf"])),
        "runs_dir": _resolve_path(PROJECT_ROOT, str(config["paths"]["runs_dir"])),
        "reports_dir": _resolve_path(PROJECT_ROOT, str(config["paths"]["reports_dir"])),
        "handoff_dir": _resolve_path(PROJECT_ROOT, str(config["paths"]["handoff_dir"])),
        "modal_reference_file": _resolve_path(PROJECT_ROOT, str(reference_path)),
    }
    config["project_root"] = PROJECT_ROOT.resolve()
    config["resolved_paths"] = resolved_paths

    for key in ("runs_dir", "reports_dir", "handoff_dir"):
        resolved_paths[key].mkdir(parents=True, exist_ok=True)

    return config


def config_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    snapshot = deepcopy(config)
    snapshot["project_root"] = str(config["project_root"])
    snapshot["resolved_paths"] = {name: str(path) for name, path in config["resolved_paths"].items()}
    return snapshot


def material_sequence(config: dict[str, Any]) -> list[dict[str, Any]]:
    return list(config["materials"])


def material_names(config: dict[str, Any]) -> list[str]:
    return [str(item["name"]) for item in material_sequence(config)]


def baseline_factors(config: dict[str, Any]) -> dict[str, float]:
    return {str(item["name"]): 1.0 for item in material_sequence(config)}
