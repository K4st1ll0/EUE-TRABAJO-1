from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from .bdf_editor import apply_density_factors, iter_card_blocks, parse_nastran_float


def _float_field(fields: list[str], index: int) -> float | None:
    if index >= len(fields):
        return None
    return parse_nastran_float(fields[index])


def _int_field(fields: list[str], index: int) -> int | None:
    if index >= len(fields):
        return None
    value = fields[index].strip()
    if not value:
        return None
    return int(value)


def _subtract(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] ** 2) + (a[1] ** 2) + (a[2] ** 2))


def _triangle_area(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
) -> float:
    return 0.5 * _norm(_cross(_subtract(b, a), _subtract(c, a)))


def _quad_area(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
    d: tuple[float, float, float],
) -> float:
    return _triangle_area(a, b, c) + _triangle_area(a, c, d)


def _bar_area(bar_type: str, dimensions: list[float]) -> float:
    if bar_type != "BAR":
        raise ValueError(f"Unsupported PBARL type '{bar_type}' in internal mass estimator.")
    if len(dimensions) < 2:
        raise ValueError("PBARL BAR cards need at least two dimensions to estimate mass.")
    return dimensions[0] * dimensions[1]


def _to_grams(value_kg: float) -> float:
    return value_kg * 1000.0


def _property_region_label(lines: list[str], start_index: int) -> str | None:
    for index in range(start_index - 1, max(start_index - 8, -1), -1):
        stripped = lines[index].strip()
        if stripped.startswith("$ Elements and Element Properties for region :"):
            return stripped.split(":", 1)[1].strip()
        if stripped and not stripped.startswith("$"):
            break
    return None


def parse_model_data(text: str) -> dict[str, Any]:
    lines = text.splitlines(keepends=True)
    grids: dict[int, tuple[float, float, float]] = {}
    materials: dict[int, dict[str, Any]] = {}
    pshells: dict[int, dict[str, Any]] = {}
    pbarls: dict[int, dict[str, Any]] = {}
    shell_elements: list[dict[str, Any]] = []
    bar_elements: list[dict[str, Any]] = []
    unsupported_cards: set[str] = set()
    card_counts: dict[str, int] = defaultdict(int)

    ignored_cards = {
        "BEGIN",
        "BEGIN BU",
        "ENDDATA",
        "PARAM",
        "PBUSH",
        "CBUSH",
        "RBE2",
        "SPC1",
        "SPCADD",
        "EIGRL",
        "METHOD",
        "METHO",
        "CORD2R",
        "SOL 103",
        "CEND",
        "ECHO = N",
        "SUBCASE",
        "SUBTI",
        "SPC =",
        "SPCFO",
        "VECTO",
        "HDF5OUT",
    }

    for block in iter_card_blocks(text):
        fields = block.fields
        card_counts[block.name] += 1
        if block.name == "GRID":
            grid_id = _int_field(fields, 0)
            if grid_id is None:
                continue
            grids[grid_id] = (
                _float_field(fields, 2) or 0.0,
                _float_field(fields, 3) or 0.0,
                _float_field(fields, 4) or 0.0,
            )
        elif block.name == "MAT1":
            material_id = _int_field(fields, 0)
            if material_id is None:
                continue
            materials[material_id] = {
                "rho": _float_field(fields, 4) or 0.0,
                "e": _float_field(fields, 1),
                "nu": _float_field(fields, 3),
            }
        elif block.name == "PSHELL":
            property_id = _int_field(fields, 0)
            material_id = _int_field(fields, 1)
            if property_id is None or material_id is None:
                continue
            pshells[property_id] = {
                "property_type": "PSHELL",
                "material_id": material_id,
                "thickness": _float_field(fields, 2) or 0.0,
                "nsm": _float_field(fields, 7) or 0.0,
                "label": _property_region_label(lines, block.line_indices[0]),
            }
        elif block.name == "PBARL":
            property_id = _int_field(fields, 0)
            material_id = _int_field(fields, 1)
            if property_id is None or material_id is None:
                continue
            pbarls[property_id] = {
                "property_type": "PBARL",
                "material_id": material_id,
                "bar_type": fields[3].strip() if len(fields) > 3 else "",
                "dimensions": [
                    float(value)
                    for value in (parse_nastran_float(item) for item in fields[4:] if item.strip())
                    if value is not None
                ],
                "label": _property_region_label(lines, block.line_indices[0]),
            }
        elif block.name == "CQUAD4":
            shell_elements.append(
                {
                    "type": "CQUAD4",
                    "element_id": _int_field(fields, 0),
                    "property_id": _int_field(fields, 1),
                    "nodes": [_int_field(fields, index) for index in range(2, 6)],
                }
            )
        elif block.name == "CTRIA3":
            shell_elements.append(
                {
                    "type": "CTRIA3",
                    "element_id": _int_field(fields, 0),
                    "property_id": _int_field(fields, 1),
                    "nodes": [_int_field(fields, index) for index in range(2, 5)],
                }
            )
        elif block.name == "CBAR":
            bar_elements.append(
                {
                    "element_id": _int_field(fields, 0),
                    "property_id": _int_field(fields, 1),
                    "node_a": _int_field(fields, 2),
                    "node_b": _int_field(fields, 3),
                }
            )
        elif block.name not in ignored_cards:
            unsupported_cards.add(block.name)

    return {
        "grids": grids,
        "materials": materials,
        "pshells": pshells,
        "pbarls": pbarls,
        "shell_elements": shell_elements,
        "bar_elements": bar_elements,
        "unsupported_cards": sorted(unsupported_cards),
        "card_counts": dict(sorted(card_counts.items())),
    }


def calculate_target_band(target_g: float, tolerance_fraction: float) -> dict[str, float]:
    lower_bound_g = round(float(target_g) * (1.0 - float(tolerance_fraction)), 6)
    upper_bound_g = round(float(target_g) * (1.0 + float(tolerance_fraction)), 6)
    return {
        "target_g": round(float(target_g), 6),
        "tolerance_fraction": float(tolerance_fraction),
        "lower_bound_g": lower_bound_g,
        "upper_bound_g": upper_bound_g,
        "half_band_g": round(float(target_g) * float(tolerance_fraction), 6),
    }


def calculate_signed_error_percent(actual_g: float, target_g: float) -> float | None:
    if abs(float(target_g)) <= 1e-12:
        return None
    return round(100.0 * (float(actual_g) - float(target_g)) / float(target_g), 8)


def calculate_absolute_error_percent(signed_error_percent: float | None) -> float | None:
    if signed_error_percent is None:
        return None
    return round(abs(float(signed_error_percent)), 8)


def _assignment_kind(role: str) -> str:
    if role == "direct_mass_budget_reference":
        return "direct_control"
    if role == "shared_mass_budget_reference":
        return "shared_control"
    return "common_structural"


def _material_control_influence(config: dict[str, Any]) -> dict[str, list[str]]:
    influence: dict[str, list[str]] = defaultdict(list)
    for control_key, control_cfg in config["mass_fit_controls"].items():
        for material_name in control_cfg["direct_materials"]:
            influence[str(material_name)].append(str(control_key))
        for shared_item in control_cfg["shared_materials"]:
            influence[str(shared_item["name"])].append(str(control_key))
    return {key: sorted(set(value)) for key, value in influence.items()}


def _build_model_mass_contributions(
    config: dict[str, Any],
    masses_by_material_g: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, float], dict[str, float]]:
    contributions: list[dict[str, Any]] = []
    direct_control_materials_g: dict[str, float] = {}
    shared_control_materials_g: dict[str, float] = {}
    common_structural_materials_g: dict[str, float] = {}
    influence_map = _material_control_influence(config)

    for material in config["materials"]:
        material_name = str(material["name"])
        mass_g = round(float(masses_by_material_g.get(material_name, 0.0)), 6)
        role = str(material.get("role", "independent_density"))
        assignment_kind = _assignment_kind(role)
        influenced_controls = influence_map.get(material_name, [])

        if assignment_kind == "direct_control":
            direct_control_materials_g[material_name] = mass_g
        elif assignment_kind == "shared_control":
            shared_control_materials_g[material_name] = mass_g
        else:
            common_structural_materials_g[material_name] = mass_g

        contributions.append(
            {
                "material_id": int(material["id"]),
                "material_name": material_name,
                "role": role,
                "assignment_kind": assignment_kind,
                "controls": influenced_controls,
                "mass_g": mass_g,
                "interpretation": str(material.get("interpretation", "")).strip() or None,
            }
        )

    return contributions, direct_control_materials_g, shared_control_materials_g, common_structural_materials_g


def calculate_control_values(mass_summary: dict[str, Any], config: dict[str, Any]) -> dict[str, float]:
    masses_by_material_g = mass_summary["masses_by_material_g"]
    control_values_g: dict[str, float] = {}
    for control_key, control_cfg in config["mass_fit_controls"].items():
        actual_g = 0.0
        for material_name in control_cfg["direct_materials"]:
            actual_g += float(masses_by_material_g.get(str(material_name), 0.0))
        for shared_item in control_cfg["shared_materials"]:
            actual_g += float(shared_item["weight"]) * float(masses_by_material_g.get(str(shared_item["name"]), 0.0))
        control_values_g[str(control_key)] = round(actual_g, 6)
    control_values_g["total"] = round(float(mass_summary["total_mass_g"]), 6)
    return control_values_g


def build_mass_fit_control_rows(mass_summary: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    masses_by_material_g = mass_summary["masses_by_material_g"]
    control_rows: list[dict[str, Any]] = []

    for control_key, control_cfg in config["mass_fit_controls"].items():
        target_cfg = config["mass_fit_targets"][control_key]
        band = calculate_target_band(target_cfg["target_g"], target_cfg["tolerance_fraction"])

        direct_breakdown_g = {
            str(material_name): round(float(masses_by_material_g.get(str(material_name), 0.0)), 6)
            for material_name in control_cfg["direct_materials"]
        }
        shared_breakdown_g = {
            str(shared_item["name"]): round(
                float(shared_item["weight"]) * float(masses_by_material_g.get(str(shared_item["name"]), 0.0)),
                6,
            )
            for shared_item in control_cfg["shared_materials"]
        }

        actual_g = round(sum(direct_breakdown_g.values()) + sum(shared_breakdown_g.values()), 6)
        lower_gap_g = max(0.0, round(band["lower_bound_g"] - actual_g, 6))
        upper_gap_g = max(0.0, round(actual_g - band["upper_bound_g"], 6))
        violation_g = max(lower_gap_g, upper_gap_g)
        scale = band["half_band_g"] if band["half_band_g"] else max(abs(band["target_g"]), 1.0)

        control_rows.append(
            {
                "control_key": str(control_key),
                "label": str(target_cfg["label"]),
                "kind": "partial",
                "target_g": round(band["target_g"], 6),
                "tolerance_fraction": float(target_cfg["tolerance_fraction"]),
                "lower_bound_g": round(band["lower_bound_g"], 6),
                "upper_bound_g": round(band["upper_bound_g"], 6),
                "actual_g": actual_g,
                "direct_mass_g": round(sum(direct_breakdown_g.values()), 6),
                "shared_mass_g": round(sum(shared_breakdown_g.values()), 6),
                "direct_breakdown_g": direct_breakdown_g,
                "shared_breakdown_g": shared_breakdown_g,
                "direct_materials": [str(item) for item in control_cfg["direct_materials"]],
                "shared_materials": list(control_cfg["shared_materials"]),
                "within_band": violation_g <= 1e-9,
                "slack_down_g": round(max(actual_g - band["lower_bound_g"], 0.0), 6),
                "slack_up_g": round(max(band["upper_bound_g"] - actual_g, 0.0), 6),
                "lower_gap_g": lower_gap_g,
                "upper_gap_g": upper_gap_g,
                "violation_g": round(violation_g, 6),
                "normalized_violation": round(violation_g / scale if scale else 0.0, 8),
                "center_error_g": round(actual_g - band["target_g"], 6),
                "normalized_target_error": round(abs(actual_g - band["target_g"]) / band["target_g"], 8)
                if band["target_g"]
                else 0.0,
            }
        )

    total_target_cfg = config["mass_fit_targets"]["total"]
    total_band = calculate_target_band(total_target_cfg["target_g"], total_target_cfg["tolerance_fraction"])
    total_actual_g = round(float(mass_summary["total_mass_g"]), 6)
    total_lower_gap_g = max(0.0, round(total_band["lower_bound_g"] - total_actual_g, 6))
    total_upper_gap_g = max(0.0, round(total_actual_g - total_band["upper_bound_g"], 6))
    total_violation_g = max(total_lower_gap_g, total_upper_gap_g)
    total_scale = total_band["half_band_g"] if total_band["half_band_g"] else max(abs(total_band["target_g"]), 1.0)
    control_rows.append(
        {
            "control_key": "total",
            "label": str(total_target_cfg["label"]),
            "kind": "total",
            "target_g": round(total_band["target_g"], 6),
            "tolerance_fraction": float(total_target_cfg["tolerance_fraction"]),
            "lower_bound_g": round(total_band["lower_bound_g"], 6),
            "upper_bound_g": round(total_band["upper_bound_g"], 6),
            "actual_g": total_actual_g,
            "direct_mass_g": total_actual_g,
            "shared_mass_g": 0.0,
            "direct_breakdown_g": {},
            "shared_breakdown_g": {},
            "direct_materials": [],
            "shared_materials": [],
            "within_band": total_violation_g <= 1e-9,
            "slack_down_g": round(max(total_actual_g - total_band["lower_bound_g"], 0.0), 6),
            "slack_up_g": round(max(total_band["upper_bound_g"] - total_actual_g, 0.0), 6),
            "lower_gap_g": total_lower_gap_g,
            "upper_gap_g": total_upper_gap_g,
            "violation_g": round(total_violation_g, 6),
            "normalized_violation": round(total_violation_g / total_scale if total_scale else 0.0, 8),
            "center_error_g": round(total_actual_g - total_band["target_g"], 6),
            "normalized_target_error": round(abs(total_actual_g - total_band["target_g"]) / total_band["target_g"], 8)
            if total_band["target_g"]
            else 0.0,
        }
    )

    return control_rows


def _control_note(row: dict[str, Any]) -> str:
    if row["control_key"] == "total":
        return "Restriccion global de masa total."
    if row["shared_mass_g"] > 0.0:
        return "Control operativo con contribucion directa y masa compartida."
    return "Control operativo directo."


def _build_mass_budget_comparison(control_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparison_rows: list[dict[str, Any]] = []
    for row in control_rows:
        signed_error_percent = calculate_signed_error_percent(row["actual_g"], row["target_g"])
        absolute_error_percent = calculate_absolute_error_percent(signed_error_percent)
        if row["control_key"] == "total":
            actual_definition = "Total model mass"
        elif row["shared_mass_g"] > 0.0 and row["direct_mass_g"] > 0.0:
            actual_definition = "Direct contribution + shared contribution"
        elif row["shared_mass_g"] > 0.0:
            actual_definition = "Shared contribution"
        else:
            actual_definition = "Direct contribution"
        comparison_rows.append(
            {
                "control_key": row["control_key"],
                "target_item": row["label"],
                "target_g": row["target_g"],
                "lower_bound_g": row["lower_bound_g"],
                "upper_bound_g": row["upper_bound_g"],
                "direct_mass_g": row["direct_mass_g"],
                "shared_mass_g": row["shared_mass_g"],
                "actual_g": row["actual_g"],
                "delta_g": row["center_error_g"],
                "rel_error": row["normalized_target_error"],
                "signed_error_percent": signed_error_percent,
                "absolute_error_percent": absolute_error_percent,
                "within_band": row["within_band"],
                "within_tolerance_5pct": row["within_band"],
                "slack_down_g": row["slack_down_g"],
                "slack_up_g": row["slack_up_g"],
                "direct_materials": list(row["direct_materials"]),
                "shared_materials": list(row["shared_materials"]),
                "actual_definition": actual_definition,
                "comparison_type": "operational_control",
                "note": _control_note(row),
            }
        )
    return comparison_rows


def _material_association_row(
    material: dict[str, Any],
    masses_by_material_g: dict[str, float],
    factors: dict[str, float],
    control_rows_by_label: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    material_name = str(material["name"])
    final_mass_g = round(float(masses_by_material_g.get(material_name, 0.0)), 6)
    density_factor = round(float(factors.get(material_name, 1.0)), 6)
    base_density = material.get("base_rho")
    final_density = None if base_density is None else round(float(base_density) * density_factor, 6)
    comparison_targets = [str(item) for item in material.get("comparison_targets", [])]
    role = str(material.get("role", "independent_density"))

    association_type = "common_structural"
    association_note = "Common structural contribution used to help the total mass closure."
    associated_budget_item = "Total mass closure / common structural"
    associated_target_g = None
    signed_error_percent = None
    absolute_error_percent = None

    if role == "direct_mass_budget_reference" and comparison_targets:
        target_label = comparison_targets[0]
        target_row = control_rows_by_label.get(target_label)
        associated_budget_item = target_label
        associated_target_g = None if target_row is None else round(float(target_row["target_g"]), 6)
        signed_error_percent = (
            None
            if associated_target_g is None
            else calculate_signed_error_percent(final_mass_g, associated_target_g)
        )
        absolute_error_percent = calculate_absolute_error_percent(signed_error_percent)
        association_type = "direct_control"
        if material_name == "PCB_FPGA":
            association_note = (
                "Direct FPGA contribution; the full FPGA control also includes the shared contribution of Bandeja_FPGA."
            )
        else:
            association_note = f"Direct operational contribution to {target_label}."
    elif role == "shared_mass_budget_reference" and comparison_targets:
        association_type = "shared_control"
        associated_budget_item = " + ".join(comparison_targets) + " (shared contribution)"
        association_note = (
            "Shared contribution across multiple operational controls; no single signed error is forced on this row."
        )

    return {
        "material_id": int(material["id"]),
        "material": material_name,
        "density_factor": density_factor,
        "final_density": final_density,
        "final_mass_g": final_mass_g,
        "associated_budget_item": associated_budget_item,
        "associated_target_g": associated_target_g,
        "associated_target_items": [
            {
                "label": label,
                "target_g": round(float(control_rows_by_label[label]["target_g"]), 6),
            }
            for label in comparison_targets
            if label in control_rows_by_label
        ],
        "association_type": association_type,
        "association_note": association_note,
        "signed_error_percent": signed_error_percent,
        "absolute_error_percent": absolute_error_percent,
    }


def build_adjusted_material_mass_rows(
    config: dict[str, Any],
    mass_summary: dict[str, Any],
    factors: dict[str, float],
    control_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    control_rows_by_label = {str(row["label"]): row for row in control_rows}
    return [
        _material_association_row(material, mass_summary["masses_by_material_g"], factors, control_rows_by_label)
        for material in config["materials"]
    ]


def _most_limiting_variable(
    control_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    pressure_by_material: dict[str, float] = defaultdict(float)
    for row in control_rows:
        if row["within_band"]:
            continue
        if row["control_key"] == "total":
            pressure_by_material["Al_7075-T7351"] += float(row["normalized_violation"])
            continue
        for material_name in row["direct_materials"]:
            pressure_by_material[str(material_name)] += float(row["normalized_violation"])
        for shared_item in row["shared_materials"]:
            pressure_by_material[str(shared_item["name"])] += float(shared_item["weight"]) * float(row["normalized_violation"])
    if not pressure_by_material:
        return None
    material_name = max(pressure_by_material, key=pressure_by_material.get)
    return {
        "material_name": material_name,
        "pressure": round(float(pressure_by_material[material_name]), 8),
        "breakdown": {name: round(float(value), 8) for name, value in sorted(pressure_by_material.items())},
    }


def build_mass_fit_summary(
    mass_summary: dict[str, Any],
    config: dict[str, Any],
    factors: dict[str, float] | None = None,
    baseline_reference_factors: dict[str, float] | None = None,
) -> dict[str, Any]:
    reference_factors = baseline_reference_factors or {str(item["name"]): 1.0 for item in config["materials"]}
    current_factors = {
        str(item["name"]): float((factors or {}).get(str(item["name"]), reference_factors[str(item["name"])]))
        for item in config["materials"]
    }
    control_rows = build_mass_fit_control_rows(mass_summary, config)
    rows_by_key = {row["control_key"]: row for row in control_rows}
    mass_budget_comparison_rows = list(mass_summary.get("mass_budget_comparison_rows") or mass_summary.get("mass_budget_comparison") or [])
    adjusted_material_mass_rows = build_adjusted_material_mass_rows(config, mass_summary, current_factors, control_rows)
    control_values_g = {row["control_key"]: row["actual_g"] for row in control_rows}
    satisfied_count = sum(1 for row in control_rows if row["within_band"])
    constraint_count = len(control_rows)
    violated_rows = [row for row in control_rows if not row["within_band"]]
    max_normalized_violation = max((float(row["normalized_violation"]) for row in control_rows), default=0.0)
    total_row = rows_by_key["total"]
    total_signed_error_percent = calculate_signed_error_percent(total_row["actual_g"], total_row["target_g"])
    total_absolute_error_percent = calculate_absolute_error_percent(total_signed_error_percent)
    change_from_baseline_l1 = round(
        sum(abs(float(current_factors[name]) - float(reference_factors.get(name, 1.0))) for name in current_factors),
        8,
    )
    most_limiting_control = None
    if violated_rows:
        most_limiting_control = max(violated_rows, key=lambda item: float(item["normalized_violation"]))
    else:
        most_limiting_control = max(control_rows, key=lambda item: float(item["normalized_target_error"]))

    return {
        "strategy": "feasibility_with_slack",
        "feasible": satisfied_count == constraint_count,
        "constraint_count": constraint_count,
        "constraints_satisfied_count": satisfied_count,
        "unsatisfied_constraints_count": constraint_count - satisfied_count,
        "control_values_g": control_values_g,
        "control_rows": control_rows,
        "mass_budget_comparison_rows": mass_budget_comparison_rows,
        "adjusted_material_mass_rows": adjusted_material_mass_rows,
        "total_mass_signed_error_percent": total_signed_error_percent,
        "total_mass_absolute_error_percent": total_absolute_error_percent,
        "controls_by_key": rows_by_key,
        "violated_controls": [str(row["control_key"]) for row in violated_rows],
        "satisfied_controls": [str(row["control_key"]) for row in control_rows if row["within_band"]],
        "max_normalized_violation": round(max_normalized_violation, 8),
        "total_mass_rel_error": float(total_row["normalized_target_error"]),
        "total_mass_error_g": float(total_row["center_error_g"]),
        "change_from_baseline_l1": change_from_baseline_l1,
        "ranking_key": [
            constraint_count - satisfied_count,
            round(max_normalized_violation, 8),
            round(float(total_row["normalized_target_error"]), 8),
            change_from_baseline_l1,
        ],
        "most_limiting_control": {
            "control_key": most_limiting_control["control_key"],
            "label": most_limiting_control["label"],
            "normalized_violation": most_limiting_control["normalized_violation"],
            "center_error_g": most_limiting_control["center_error_g"],
        }
        if most_limiting_control
        else None,
        "most_limiting_variable": _most_limiting_variable(control_rows, config),
        "control_summary_table": [
            {
                "control_key": row["control_key"],
                "label": row["label"],
                "target_g": row["target_g"],
                "lower_bound_g": row["lower_bound_g"],
                "upper_bound_g": row["upper_bound_g"],
                "actual_g": row["actual_g"],
                "within_band": row["within_band"],
                "slack_down_g": row["slack_down_g"],
                "slack_up_g": row["slack_up_g"],
                "normalized_violation": row["normalized_violation"],
            }
            for row in control_rows
        ],
    }


def feasibility_ranking_key(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    ranking = summary.get("ranking_key", [float("inf"), float("inf"), float("inf"), float("inf")])
    return (
        float(ranking[0]),
        float(ranking[1]),
        float(ranking[2]),
        float(ranking[3]),
    )


def is_better_feasibility(candidate: dict[str, Any], reference: dict[str, Any] | None) -> bool:
    if reference is None:
        return True
    return feasibility_ranking_key(candidate) < feasibility_ranking_key(reference)


def estimate_masses(text: str, config: dict[str, Any]) -> dict[str, Any]:
    model = parse_model_data(text)
    grids = model["grids"]
    materials = model["materials"]
    pshells = model["pshells"]
    pbarls = model["pbarls"]

    config_materials = {int(item["id"]): item for item in config["materials"]}
    masses_kg_by_material_id: dict[int, float] = defaultdict(float)
    masses_kg_by_property: dict[tuple[str, int], float] = defaultdict(float)
    warnings: list[str] = []

    for element in model["shell_elements"]:
        property_id = element["property_id"]
        if property_id not in pshells:
            warnings.append(f"{element['type']} {element['element_id']} references missing PSHELL {property_id}.")
            continue
        nodes = [node for node in element["nodes"] if node is not None]
        if any(node not in grids for node in nodes):
            warnings.append(f"{element['type']} {element['element_id']} references missing GRID coordinates.")
            continue

        prop = pshells[property_id]
        material_id = prop["material_id"]
        if material_id not in materials:
            warnings.append(f"PSHELL {property_id} references missing MAT1 {material_id}.")
            continue

        coordinates = [grids[node] for node in nodes]
        if element["type"] == "CQUAD4" and len(coordinates) == 4:
            area = _quad_area(*coordinates)
        elif element["type"] == "CTRIA3" and len(coordinates) == 3:
            area = _triangle_area(*coordinates)
        else:
            warnings.append(f"{element['type']} {element['element_id']} has incomplete node data.")
            continue

        rho = float(materials[material_id]["rho"])
        surface_density = (prop["thickness"] * rho) + prop["nsm"]
        mass_value = area * surface_density
        masses_kg_by_material_id[material_id] += mass_value
        masses_kg_by_property[("PSHELL", int(property_id))] += mass_value

    for element in model["bar_elements"]:
        property_id = element["property_id"]
        if property_id not in pbarls:
            warnings.append(f"CBAR {element['element_id']} references missing PBARL {property_id}.")
            continue
        if element["node_a"] not in grids or element["node_b"] not in grids:
            warnings.append(f"CBAR {element['element_id']} references missing GRID coordinates.")
            continue

        prop = pbarls[property_id]
        material_id = prop["material_id"]
        if material_id not in materials:
            warnings.append(f"PBARL {property_id} references missing MAT1 {material_id}.")
            continue
        try:
            area = _bar_area(str(prop["bar_type"]), list(prop["dimensions"]))
        except ValueError as exc:
            warnings.append(str(exc))
            continue

        point_a = grids[element["node_a"]]
        point_b = grids[element["node_b"]]
        length = _norm(_subtract(point_b, point_a))
        rho = float(materials[material_id]["rho"])
        mass_value = length * area * rho
        masses_kg_by_material_id[material_id] += mass_value
        masses_kg_by_property[("PBARL", int(property_id))] += mass_value

    masses_by_material_g: dict[str, float] = {}
    for material_id, material_config in config_materials.items():
        mass_g = _to_grams(masses_kg_by_material_id.get(material_id, 0.0))
        masses_by_material_g[str(material_config["name"])] = round(mass_g, 6)

    masses_by_property_type_g: dict[str, float] = defaultdict(float)
    masses_by_property_g: list[dict[str, Any]] = []
    for (property_type, property_id), mass_kg in sorted(masses_kg_by_property.items()):
        property_data = pshells.get(property_id) if property_type == "PSHELL" else pbarls.get(property_id)
        material_id = int(property_data["material_id"]) if property_data else None
        material_name = (
            str(config_materials[material_id]["name"])
            if material_id is not None and material_id in config_materials
            else f"MAT1_{material_id}"
        )
        mass_g = round(_to_grams(mass_kg), 6)
        masses_by_property_type_g[property_type] += mass_g
        masses_by_property_g.append(
            {
                "property_type": property_type,
                "property_id": property_id,
                "label": property_data.get("label") if property_data else None,
                "material_id": material_id,
                "material_name": material_name,
                "mass_g": mass_g,
            }
        )

    model_mass_contributions, direct_control_materials_g, shared_control_materials_g, common_structural_materials_g = (
        _build_model_mass_contributions(config, masses_by_material_g)
    )
    supported_mass_cards_found = {
        key: model["card_counts"].get(key, 0)
        for key in ("MAT1", "PSHELL", "PBARL", "CQUAD4", "CTRIA3", "CBAR")
        if model["card_counts"].get(key, 0)
    }
    zero_mass_assumed_cards_found = {
        key: model["card_counts"].get(key, 0)
        for key in ("PBUSH", "CBUSH", "RBE2", "SPC1", "SPCADD")
        if model["card_counts"].get(key, 0)
    }
    limitations = [
        "El estimador no cubre elementos concentrados, solidos, compuestos o tarjetas NSM genericas si se anaden en futuras versiones del modelo.",
        "PBUSH/CBUSH, RBE2 y SPC se tratan como contribucion de masa nula para este modelo actual.",
    ]
    if not model["unsupported_cards"]:
        limitations.append("No se han detectado otras tarjetas de masa relevantes fuera del alcance actual en este BDF.")

    mass_summary = {
        "masses_by_material_g": masses_by_material_g,
        "model_mass_contributions": model_mass_contributions,
        "direct_control_materials_g": direct_control_materials_g,
        "shared_control_materials_g": shared_control_materials_g,
        "common_structural_materials_g": common_structural_materials_g,
        "direct_reference_mass_g": round(sum(float(value) for value in direct_control_materials_g.values()), 6),
        "shared_mass_g": round(sum(float(value) for value in shared_control_materials_g.values()), 6),
        "unassigned_mass_g": round(sum(float(value) for value in common_structural_materials_g.values()), 6),
        "unassigned_materials_g": common_structural_materials_g,
        "masses_by_property_type_g": {key: round(value, 6) for key, value in masses_by_property_type_g.items()},
        "masses_by_property_g": masses_by_property_g,
        "total_mass_g": round(sum(masses_by_material_g.values()), 6),
        "element_counts": {
            "shells": len(model["shell_elements"]),
            "bars": len(model["bar_elements"]),
        },
        "card_counts": model["card_counts"],
        "unsupported_cards": model["unsupported_cards"],
        "coverage_review": {
            "supported_mass_cards_found": supported_mass_cards_found,
            "zero_mass_assumed_cards_found": zero_mass_assumed_cards_found,
            "limitations": limitations,
        },
        "warnings": warnings,
    }
    control_rows = build_mass_fit_control_rows(mass_summary, config)
    mass_summary["mass_fit_controls"] = control_rows
    mass_summary["mass_budget_comparison"] = _build_mass_budget_comparison(control_rows)
    mass_summary["mass_budget_comparison_rows"] = list(mass_summary["mass_budget_comparison"])
    return mass_summary


def score_mass_fit(
    mass_summary: dict[str, Any],
    config: dict[str, Any],
    factors: dict[str, float] | None = None,
    baseline_reference_factors: dict[str, float] | None = None,
    mass_fit_summary: dict[str, Any] | None = None,
    nastran_failed: bool = False,
) -> dict[str, Any]:
    weights = config["score"]["weights"]
    fit_summary = mass_fit_summary or build_mass_fit_summary(
        mass_summary,
        config,
        factors=factors,
        baseline_reference_factors=baseline_reference_factors,
    )

    failure_penalty = float(weights["failure_penalty"]) if nastran_failed else 0.0
    score = (
        float(weights["unsatisfied_constraints"]) * float(fit_summary["unsatisfied_constraints_count"])
        + float(weights["max_normalized_violation"]) * float(fit_summary["max_normalized_violation"])
        + float(weights["total_mass_rel_error"]) * float(fit_summary["total_mass_rel_error"])
        + float(weights["change_from_baseline"]) * float(fit_summary["change_from_baseline_l1"])
        + failure_penalty
    )

    mass_budget_rel_errors = {
        str(item["target_item"]): round(float(item["rel_error"]), 8)
        for item in mass_summary.get("mass_budget_comparison", [])
    }
    mass_budget_error_g = {
        str(item["target_item"]): round(float(item["delta_g"]), 6)
        for item in mass_summary.get("mass_budget_comparison", [])
    }

    return {
        "score": round(score, 8),
        "components": {
            "unsatisfied_constraints": int(fit_summary["unsatisfied_constraints_count"]),
            "max_normalized_violation": round(float(fit_summary["max_normalized_violation"]), 8),
            "total_mass_rel_error": round(float(fit_summary["total_mass_rel_error"]), 8),
            "change_from_baseline_l1": round(float(fit_summary["change_from_baseline_l1"]), 8),
            "failure_penalty_applied": round(failure_penalty, 8),
        },
        "mass_budget_rel_errors": mass_budget_rel_errors,
        "mass_budget_error_g": mass_budget_error_g,
        "total_mass_error_g": round(float(fit_summary["total_mass_error_g"]), 6),
    }


def compute_local_sensitivity(reference_text: str, config: dict[str, Any]) -> dict[str, Any]:
    variation_fraction = float(config.get("sensitivity", {}).get("local_variation_fraction", 0.05))
    baseline_factors_map = {str(item["name"]): 1.0 for item in config["materials"]}
    baseline_summary = estimate_masses(reference_text, config)
    baseline_fit_summary = build_mass_fit_summary(
        baseline_summary,
        config,
        factors=baseline_factors_map,
        baseline_reference_factors=baseline_factors_map,
    )
    rows: list[dict[str, Any]] = []
    summary_by_material: list[dict[str, Any]] = []

    for material in config["materials"]:
        material_name = str(material["name"])
        minus_row: dict[str, Any] | None = None
        plus_row: dict[str, Any] | None = None
        for label, multiplier in (("-5%", 1.0 - variation_fraction), ("+5%", 1.0 + variation_fraction)):
            relative_factors = {str(item["name"]): 1.0 for item in config["materials"]}
            relative_factors[material_name] = multiplier
            perturbed_text, _ = apply_density_factors(reference_text, config["materials"], relative_factors)
            perturbed_summary = estimate_masses(perturbed_text, config)
            perturbed_fit_summary = build_mass_fit_summary(
                perturbed_summary,
                config,
                factors=relative_factors,
                baseline_reference_factors=baseline_factors_map,
            )
            row = {
                "material_name": material_name,
                "variation_label": label,
                "factor_multiplier": round(multiplier, 6),
                "total_mass_g": perturbed_summary["total_mass_g"],
                "delta_total_mass_g": round(
                    float(perturbed_summary["total_mass_g"]) - float(baseline_summary["total_mass_g"]),
                    6,
                ),
                "masses_by_material_g": perturbed_summary["masses_by_material_g"],
                "control_values_g": perturbed_fit_summary["control_values_g"],
                "control_rows": perturbed_fit_summary["control_rows"],
            }
            rows.append(row)
            if label == "-5%":
                minus_row = row
            else:
                plus_row = row

        summary_by_material.append(
            {
                "material_name": material_name,
                "minus_factor_multiplier": minus_row["factor_multiplier"] if minus_row else None,
                "minus_total_mass_g": minus_row["total_mass_g"] if minus_row else None,
                "minus_delta_total_mass_g": minus_row["delta_total_mass_g"] if minus_row else None,
                "plus_factor_multiplier": plus_row["factor_multiplier"] if plus_row else None,
                "plus_total_mass_g": plus_row["total_mass_g"] if plus_row else None,
                "plus_delta_total_mass_g": plus_row["delta_total_mass_g"] if plus_row else None,
            }
        )

    return {
        "variation_fraction": variation_fraction,
        "method": "internal_mass_estimator_one_at_a_time",
        "notes": [
            "La sensibilidad local se calcula alrededor del baseline variando una sola densidad cada vez en +/-5%.",
            "Es una estimacion interna basada en el BDF y no implica nuevas ejecuciones de Nastran.",
        ],
        "baseline_total_mass_g": baseline_summary["total_mass_g"],
        "baseline_masses_by_material_g": baseline_summary["masses_by_material_g"],
        "baseline_control_values_g": baseline_fit_summary["control_values_g"],
        "baseline_control_rows": baseline_fit_summary["control_rows"],
        "summary_by_material": summary_by_material,
        "rows": rows,
    }
