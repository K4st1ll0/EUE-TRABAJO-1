from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .bdf_editor import parse_nastran_float


EIGENVALUE_RE = re.compile(r"EIGENVALUE\s*=\s*([+\-0-9.DEde]+)")
MODE_HEADER_RE = re.compile(
    r"CYCLES\s*=\s*([+\-0-9.DEde]+)\s+R E A L\s+E I G E N V E C T O R\s+N O\s*\.\s*(\d+)",
    re.IGNORECASE,
)
GRID_ROW_RE = re.compile(
    r"^\s*(\d+)\s+G\s+([+\-0-9.DEde]+)\s+([+\-0-9.DEde]+)\s+([+\-0-9.DEde]+)\s+([+\-0-9.DEde]+)\s+([+\-0-9.DEde]+)\s+([+\-0-9.DEde]+)",
)


def modal_vector_labels(config: dict[str, Any]) -> list[str]:
    return [str(item["label"]) for item in config["modal"]["modal_vector_order"]]


def modal_measurement_points(config: dict[str, Any]) -> list[dict[str, Any]]:
    return list(config["modal"]["measurement_points"])


def _measurement_label_map(config: dict[str, Any]) -> dict[int, str]:
    return {
        int(item["node_id"]): str(item["label"])
        for item in config["modal"]["measurement_points"]
    }


def _vector_label_metadata(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    point_labels = _measurement_label_map(config)
    metadata: dict[str, dict[str, Any]] = {}
    for item in config["modal"]["modal_vector_order"]:
        metadata[str(item["label"])] = {
            "node_id": int(item["node_id"]),
            "component": str(item["component"]),
            "point_label": point_labels.get(int(item["node_id"]), str(item["node_id"])),
        }
    return metadata


def _reference_point_map(config: dict[str, Any]) -> dict[int, int]:
    mapping_rows = config["modal"]["reference"].get("measurement_points") or []
    return {int(item["source_node_id"]): int(item["target_node_id"]) for item in mapping_rows}


def _point_alias_map(config: dict[str, Any], source_name: str) -> dict[int, int]:
    reference_path = Path(str(config["resolved_paths"]["modal_reference_file"])).resolve()
    candidate_path = Path(str(source_name)).resolve()
    if candidate_path == reference_path:
        return _reference_point_map(config)
    return {int(item["node_id"]): int(item["node_id"]) for item in config["modal"]["measurement_points"]}


def build_modal_vector(mode: dict[str, Any], config: dict[str, Any]) -> list[float]:
    node_components = mode["node_components"]
    vector: list[float] = []
    for item in config["modal"]["modal_vector_order"]:
        node_id = int(item["node_id"])
        component = str(item["component"])
        if node_id not in node_components:
            raise ValueError(f"Mode {mode['mode_number']} is missing node {node_id} for modal vector construction.")
        if component not in node_components[node_id]:
            raise ValueError(f"Mode {mode['mode_number']} is missing component {component} at node {node_id}.")
        vector.append(float(node_components[node_id][component]))
    return vector


def calculate_mac(reference_vector: list[float], model_vector: list[float]) -> float:
    if len(reference_vector) != len(model_vector):
        raise ValueError("MAC vectors must have the same length.")
    numerator = sum(ref * model for ref, model in zip(reference_vector, model_vector)) ** 2
    reference_norm = sum(ref * ref for ref in reference_vector)
    model_norm = sum(model * model for model in model_vector)
    denominator = reference_norm * model_norm
    if denominator <= 0.0:
        return 0.0
    return round(numerator / denominator, 8)


def calculate_mac_matrix(
    reference_modes: list[dict[str, Any]],
    model_modes: list[dict[str, Any]],
) -> list[list[float]]:
    matrix: list[list[float]] = []
    for reference_mode in reference_modes:
        matrix.append(
            [
                calculate_mac(reference_mode["modal_vector"], model_mode["modal_vector"])
                for model_mode in model_modes
            ]
        )
    return matrix


def calculate_signed_frequency_error_percent(reference_frequency_hz: float, model_frequency_hz: float) -> float:
    if abs(reference_frequency_hz) <= 1e-12:
        return 0.0
    return round((float(model_frequency_hz) - float(reference_frequency_hz)) * 100.0 / abs(float(reference_frequency_hz)), 8)


def calculate_absolute_frequency_error_percent(signed_frequency_error_percent: float) -> float:
    return round(abs(float(signed_frequency_error_percent)), 8)


def _frequency_error(reference_frequency_hz: float, model_frequency_hz: float) -> float:
    if abs(reference_frequency_hz) <= 1e-12:
        return 0.0
    return round(abs(model_frequency_hz - reference_frequency_hz) / abs(reference_frequency_hz), 8)


def parse_modal_f06(
    text: str,
    config: dict[str, Any],
    source_name: str,
) -> dict[str, Any]:
    requested_nodes = {int(item["node_id"]) for item in config["modal"]["measurement_points"]}
    point_alias_map = _point_alias_map(config, source_name)
    requested_source_nodes = set(point_alias_map)
    requested_modes = int(config["modal"]["num_modes"])
    modes_by_number: dict[int, dict[str, Any]] = {}
    current_mode_number: int | None = None
    last_eigenvalue: float | None = None

    for raw_line in text.splitlines():
        eigenvalue_match = EIGENVALUE_RE.search(raw_line)
        if eigenvalue_match:
            last_eigenvalue = parse_nastran_float(eigenvalue_match.group(1))
            continue

        mode_header_match = MODE_HEADER_RE.search(raw_line)
        if mode_header_match:
            frequency_hz = parse_nastran_float(mode_header_match.group(1))
            mode_number = int(mode_header_match.group(2))
            current_mode_number = mode_number
            mode_entry = modes_by_number.setdefault(
                mode_number,
                {
                    "mode_number": mode_number,
                    "frequency_hz": None,
                    "eigenvalue": None,
                    "node_components": {},
                },
            )
            if frequency_hz is not None:
                mode_entry["frequency_hz"] = float(frequency_hz)
            if last_eigenvalue is not None:
                mode_entry["eigenvalue"] = float(last_eigenvalue)
            continue

        if current_mode_number is None:
            continue

        grid_match = GRID_ROW_RE.match(raw_line)
        if not grid_match:
            continue

        node_id = int(grid_match.group(1))
        if node_id not in requested_source_nodes:
            continue
        canonical_node_id = int(point_alias_map[node_id])

        components = {
            "T1": float(parse_nastran_float(grid_match.group(2)) or 0.0),
            "T2": float(parse_nastran_float(grid_match.group(3)) or 0.0),
            "T3": float(parse_nastran_float(grid_match.group(4)) or 0.0),
            "R1": float(parse_nastran_float(grid_match.group(5)) or 0.0),
            "R2": float(parse_nastran_float(grid_match.group(6)) or 0.0),
            "R3": float(parse_nastran_float(grid_match.group(7)) or 0.0),
        }
        modes_by_number[current_mode_number]["node_components"][canonical_node_id] = components

    selected_mode_numbers = sorted(modes_by_number)[:requested_modes]
    modes: list[dict[str, Any]] = []
    notes: list[str] = []

    for mode_number in selected_mode_numbers:
        mode_entry = modes_by_number[mode_number]
        missing_nodes = sorted(requested_nodes - set(mode_entry["node_components"]))
        if missing_nodes:
            raise ValueError(
                f"{source_name} is missing nodes {missing_nodes} in eigenvector mode {mode_number}."
            )
        if mode_entry["frequency_hz"] is None:
            raise ValueError(f"{source_name} is missing frequency information for mode {mode_number}.")

        modal_vector = build_modal_vector(mode_entry, config)
        mode_entry["modal_vector"] = [round(float(value), 12) for value in modal_vector]
        mode_entry["modal_vector_by_label"] = {
            label: round(float(value), 12)
            for label, value in zip(modal_vector_labels(config), modal_vector)
        }
        modes.append(mode_entry)

    if len(modes) < requested_modes:
        notes.append(
            f"{source_name} solo aporta {len(modes)} modos parseables frente a los {requested_modes} solicitados."
        )

    return {
        "source_name": source_name,
        "source_type": "nastran_f06_real_eigenvectors",
        "requested_nodes": sorted(requested_nodes),
        "requested_source_nodes": sorted(requested_source_nodes),
        "point_alias_map": {str(source): target for source, target in point_alias_map.items()},
        "requested_modes": requested_modes,
        "parsed_modes": len(modes),
        "modes": modes,
        "notes": notes,
    }


def _assignment_utility(mac_value: float, frequency_error: float, penalty_weight: float) -> float:
    return float(mac_value) - (float(penalty_weight) * float(frequency_error))


def pair_modal_modes(
    reference_modes: list[dict[str, Any]],
    model_modes: list[dict[str, Any]],
    mac_matrix: list[list[float]],
    pairing_frequency_penalty_weight: float,
) -> dict[str, Any]:
    if len(reference_modes) != len(model_modes):
        raise ValueError("Reference and model mode sets must have the same size for pairing.")
    size = len(reference_modes)
    frequency_errors = [
        [
            _frequency_error(reference_modes[row]["frequency_hz"], model_modes[column]["frequency_hz"])
            for column in range(size)
        ]
        for row in range(size)
    ]
    utilities = [
        [
            _assignment_utility(mac_matrix[row][column], frequency_errors[row][column], pairing_frequency_penalty_weight)
            for column in range(size)
        ]
        for row in range(size)
    ]

    @lru_cache(maxsize=None)
    def solve(reference_index: int, used_mask: int) -> tuple[float, tuple[int, ...]]:
        if reference_index >= size:
            return 0.0, ()

        best_utility = -math.inf
        best_assignment: tuple[int, ...] = ()
        for model_index in range(size):
            if used_mask & (1 << model_index):
                continue
            tail_utility, tail_assignment = solve(reference_index + 1, used_mask | (1 << model_index))
            total_utility = utilities[reference_index][model_index] + tail_utility
            if total_utility > best_utility + 1e-12:
                best_utility = total_utility
                best_assignment = (model_index,) + tail_assignment
                continue
            if abs(total_utility - best_utility) <= 1e-12 and best_assignment:
                current_mac = mac_matrix[reference_index][model_index]
                best_mac = mac_matrix[reference_index][best_assignment[0]]
                current_freq = frequency_errors[reference_index][model_index]
                best_freq = frequency_errors[reference_index][best_assignment[0]]
                if current_mac > best_mac + 1e-12 or (
                    abs(current_mac - best_mac) <= 1e-12 and current_freq < best_freq
                ):
                    best_assignment = (model_index,) + tail_assignment
        return best_utility, best_assignment

    total_utility, assignment = solve(0, 0)
    pairs: list[dict[str, Any]] = []
    diagonal_comparison: list[dict[str, Any]] = []

    for reference_index, reference_mode in enumerate(reference_modes):
        model_index = assignment[reference_index]
        model_mode = model_modes[model_index]
        pair_frequency_error = frequency_errors[reference_index][model_index]
        pair_signed_frequency_error_percent = calculate_signed_frequency_error_percent(
            reference_mode["frequency_hz"],
            model_mode["frequency_hz"],
        )
        pair_absolute_frequency_error_percent = calculate_absolute_frequency_error_percent(
            pair_signed_frequency_error_percent
        )
        diagonal_mac = mac_matrix[reference_index][reference_index]
        diagonal_frequency_error = frequency_errors[reference_index][reference_index]
        diagonal_signed_frequency_error_percent = calculate_signed_frequency_error_percent(
            reference_mode["frequency_hz"],
            model_modes[reference_index]["frequency_hz"],
        )
        diagonal_absolute_frequency_error_percent = calculate_absolute_frequency_error_percent(
            diagonal_signed_frequency_error_percent
        )
        pairs.append(
            {
                "reference_index": reference_index + 1,
                "model_index": model_index + 1,
                "reference_mode_number": reference_mode["mode_number"],
                "model_mode_number": model_mode["mode_number"],
                "reference_frequency_hz": round(float(reference_mode["frequency_hz"]), 6),
                "model_frequency_hz": round(float(model_mode["frequency_hz"]), 6),
                "frequency_error_rel": pair_frequency_error,
                "signed_frequency_error_percent": pair_signed_frequency_error_percent,
                "absolute_frequency_error_percent": pair_absolute_frequency_error_percent,
                "mac": mac_matrix[reference_index][model_index],
                "diagonal_mac": diagonal_mac,
                "diagonal_frequency_error_rel": diagonal_frequency_error,
                "diagonal_signed_frequency_error_percent": diagonal_signed_frequency_error_percent,
                "diagonal_absolute_frequency_error_percent": diagonal_absolute_frequency_error_percent,
                "pairing_utility": round(float(utilities[reference_index][model_index]), 8),
            }
        )
        diagonal_comparison.append(
            {
                "reference_index": reference_index + 1,
                "reference_mode_number": reference_mode["mode_number"],
                "model_mode_number": model_modes[reference_index]["mode_number"],
                "reference_frequency_hz": round(float(reference_mode["frequency_hz"]), 6),
                "model_frequency_hz": round(float(model_modes[reference_index]["frequency_hz"]), 6),
                "frequency_error_rel": diagonal_frequency_error,
                "signed_frequency_error_percent": diagonal_signed_frequency_error_percent,
                "absolute_frequency_error_percent": diagonal_absolute_frequency_error_percent,
                "mac": diagonal_mac,
            }
        )

    return {
        "objective": "maximize sum(MAC - pairing_frequency_penalty_weight * relative_frequency_error)",
        "pairing_frequency_penalty_weight": float(pairing_frequency_penalty_weight),
        "total_assignment_utility": round(float(total_utility), 8),
        "pairs": pairs,
        "diagonal_comparison": diagonal_comparison,
    }


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 8)


def _worst(values: list[float], reverse: bool = False) -> float:
    if not values:
        return 0.0
    return round(min(values) if reverse else max(values), 8)


def _modal_diagnostic_thresholds(config: dict[str, Any]) -> dict[str, dict[str, float]]:
    diagnostics = config["modal"].get("diagnostics", {})
    return {
        "good": {
            "mac_min": float(diagnostics.get("good", {}).get("mac_min", config["modal"]["mac_good_threshold"])),
            "frequency_error_max": float(
                diagnostics.get("good", {}).get("frequency_error_max", config["modal"]["frequency_error_good_threshold"])
            ),
        },
        "acceptable": {
            "mac_min": float(diagnostics.get("acceptable", {}).get("mac_min", 0.6)),
            "frequency_error_max": float(diagnostics.get("acceptable", {}).get("frequency_error_max", 0.20)),
        },
    }


def _classify_mode(mac_value: float, frequency_error_rel: float, config: dict[str, Any]) -> str:
    thresholds = _modal_diagnostic_thresholds(config)
    if float(mac_value) >= thresholds["good"]["mac_min"] and float(frequency_error_rel) <= thresholds["good"]["frequency_error_max"]:
        return "Good"
    if float(mac_value) >= thresholds["acceptable"]["mac_min"] and float(frequency_error_rel) <= thresholds["acceptable"]["frequency_error_max"]:
        return "Acceptable"
    return "Problematic"


def _failure_driver(mac_value: float, frequency_error_rel: float, config: dict[str, Any]) -> str:
    thresholds = _modal_diagnostic_thresholds(config)
    mac_gap = max(0.0, thresholds["good"]["mac_min"] - float(mac_value))
    freq_gap = max(0.0, float(frequency_error_rel) - thresholds["good"]["frequency_error_max"])
    if mac_gap <= 1e-12 and freq_gap <= 1e-12:
        return "none"
    if mac_gap > 1e-12 and freq_gap <= 1e-12:
        return "shape"
    if freq_gap > 1e-12 and mac_gap <= 1e-12:
        return "frequency"
    mac_norm = mac_gap / max(thresholds["good"]["mac_min"], 1e-12)
    freq_norm = freq_gap / max(thresholds["good"]["frequency_error_max"], 1e-12)
    if mac_norm >= freq_norm * 1.25:
        return "shape"
    if freq_norm >= mac_norm * 1.25:
        return "frequency"
    return "mixed"


def _component_contributions(mode: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    labels = modal_vector_labels(config)
    metadata = _vector_label_metadata(config)
    vector = [float(value) for value in mode.get("modal_vector", [])]
    energy_total = sum(value * value for value in vector)
    dominant_count = int(config["modal"].get("diagnostics", {}).get("dominant_dofs_per_mode", 3))

    component_rows: list[dict[str, Any]] = []
    node_energy: dict[int, dict[str, Any]] = {}
    for label, value in zip(labels, vector):
        meta = metadata[str(label)]
        energy_fraction = 0.0 if energy_total <= 1e-24 else (float(value) * float(value)) / energy_total
        component_rows.append(
            {
                "label": str(label),
                "node_id": int(meta["node_id"]),
                "point_label": str(meta["point_label"]),
                "component": str(meta["component"]),
                "value": round(float(value), 12),
                "energy_fraction": round(float(energy_fraction), 8),
            }
        )
        node_row = node_energy.setdefault(
            int(meta["node_id"]),
            {
                "node_id": int(meta["node_id"]),
                "point_label": str(meta["point_label"]),
                "energy_fraction": 0.0,
            },
        )
        node_row["energy_fraction"] += float(energy_fraction)

    component_rows.sort(key=lambda item: (-float(item["energy_fraction"]), -abs(float(item["value"])), str(item["label"])))
    node_rows = [
        {
            "node_id": int(item["node_id"]),
            "point_label": str(item["point_label"]),
            "energy_fraction": round(float(item["energy_fraction"]), 8),
        }
        for item in sorted(
            node_energy.values(),
            key=lambda row: (-float(row["energy_fraction"]), str(row["point_label"])),
        )
    ]

    return {
        "components": component_rows,
        "dominant_components": component_rows[:dominant_count],
        "nodes": node_rows,
        "dominant_nodes": node_rows[:dominant_count],
    }


def _dominant_summary(rows: list[dict[str, Any]], key: str = "label") -> str:
    if not rows:
        return "n/a"
    return ", ".join(
        f"{item[key]} ({float(item['energy_fraction']) * 100.0:.1f}%)"
        for item in rows
    )


def _diagnostic_note(
    reference_contrib: dict[str, Any],
    model_contrib: dict[str, Any],
    failure_driver: str,
) -> str:
    ref_dofs = reference_contrib.get("dominant_components", [])
    model_dofs = model_contrib.get("dominant_components", [])
    ref_nodes = reference_contrib.get("dominant_nodes", [])
    model_nodes = model_contrib.get("dominant_nodes", [])
    if not ref_dofs or not model_dofs:
        return "No se pudo identificar una dominancia modal clara."

    ref_main = ref_dofs[0]
    model_main = model_dofs[0]
    if ref_main["label"] == model_main["label"]:
        base_note = (
            f"Ambos vectores estan dominados por {ref_main['label']}; "
            f"referencia {_dominant_summary(ref_dofs[:2])}, modelo {_dominant_summary(model_dofs[:2])}."
        )
    elif ref_main["node_id"] == model_main["node_id"]:
        base_note = (
            f"La region dominante coincide en {ref_main['point_label']}, pero cambia el DOF principal de "
            f"{ref_main['label']} a {model_main['label']}."
        )
    elif ref_nodes and model_nodes:
        base_note = (
            f"La dominancia se desplaza de {ref_nodes[0]['point_label']} en la referencia a "
            f"{model_nodes[0]['point_label']} en el modelo."
        )
    else:
        base_note = (
            f"Referencia dominada por {_dominant_summary(ref_dofs[:2])}; "
            f"modelo dominado por {_dominant_summary(model_dofs[:2])}."
        )

    if failure_driver == "shape":
        return base_note + " El desacople principal parece venir de la forma modal."
    if failure_driver == "frequency":
        return base_note + " La forma modal es razonable, pero la frecuencia sigue siendo el principal problema."
    if failure_driver == "mixed":
        return base_note + " Hay una mezcla de problema de forma y de frecuencia."
    return base_note


def _problematic_priority_score(mac_value: float, frequency_error_rel: float, config: dict[str, Any]) -> float:
    thresholds = _modal_diagnostic_thresholds(config)
    mac_gap = max(0.0, thresholds["good"]["mac_min"] - float(mac_value))
    freq_gap = max(0.0, float(frequency_error_rel) - thresholds["good"]["frequency_error_max"])
    acceptable_mac_gap = max(0.0, thresholds["acceptable"]["mac_min"] - float(mac_value))
    acceptable_freq_gap = max(0.0, float(frequency_error_rel) - thresholds["acceptable"]["frequency_error_max"])
    mac_norm = mac_gap / max(thresholds["good"]["mac_min"], 1e-12)
    freq_norm = freq_gap / max(thresholds["good"]["frequency_error_max"], 1e-12)
    acceptable_mac_norm = acceptable_mac_gap / max(thresholds["acceptable"]["mac_min"], 1e-12)
    acceptable_freq_norm = acceptable_freq_gap / max(thresholds["acceptable"]["frequency_error_max"], 1e-12)
    return round((2.0 * mac_norm) + (2.0 * freq_norm) + acceptable_mac_norm + acceptable_freq_norm, 8)


def _build_paired_mode_diagnostics(
    reference_modes: list[dict[str, Any]],
    model_modes: list[dict[str, Any]],
    pairing: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    thresholds = _modal_diagnostic_thresholds(config)
    diagnostics: list[dict[str, Any]] = []
    counts = {"Good": 0, "Acceptable": 0, "Problematic": 0}

    for pair in pairing["pairs"]:
        reference_mode = reference_modes[int(pair["reference_index"]) - 1]
        model_mode = model_modes[int(pair["model_index"]) - 1]
        reference_contrib = _component_contributions(reference_mode, config)
        model_contrib = _component_contributions(model_mode, config)
        classification = _classify_mode(float(pair["mac"]), float(pair["frequency_error_rel"]), config)
        failure_driver = _failure_driver(float(pair["mac"]), float(pair["frequency_error_rel"]), config)
        priority_score = _problematic_priority_score(float(pair["mac"]), float(pair["frequency_error_rel"]), config)
        counts[classification] += 1
        diagnostics.append(
            {
                "reference_index": int(pair["reference_index"]),
                "model_index": int(pair["model_index"]),
                "reference_mode_number": int(pair["reference_mode_number"]),
                "model_mode_number": int(pair["model_mode_number"]),
                "reference_frequency_hz": round(float(pair["reference_frequency_hz"]), 6),
                "model_frequency_hz": round(float(pair["model_frequency_hz"]), 6),
                "paired_mac": round(float(pair["mac"]), 8),
                "frequency_error_rel": round(float(pair["frequency_error_rel"]), 8),
                "signed_frequency_error_percent": float(pair["signed_frequency_error_percent"]),
                "absolute_frequency_error_percent": float(pair["absolute_frequency_error_percent"]),
                "mac_good": bool(pair.get("mac_good")),
                "frequency_good": bool(pair.get("frequency_good")),
                "good_on_both": bool(pair.get("good_on_both")),
                "classification": classification,
                "failure_driver": failure_driver,
                "priority_score": priority_score,
                "reference_dominant_dofs": reference_contrib["dominant_components"],
                "model_dominant_dofs": model_contrib["dominant_components"],
                "reference_dominant_nodes": reference_contrib["dominant_nodes"],
                "model_dominant_nodes": model_contrib["dominant_nodes"],
                "reference_component_energy": reference_contrib["components"],
                "model_component_energy": model_contrib["components"],
                "reference_dominant_summary": _dominant_summary(reference_contrib["dominant_components"]),
                "model_dominant_summary": _dominant_summary(model_contrib["dominant_components"]),
                "reference_node_summary": _dominant_summary(reference_contrib["dominant_nodes"], key="point_label"),
                "model_node_summary": _dominant_summary(model_contrib["dominant_nodes"], key="point_label"),
                "diagnostic_note": _diagnostic_note(reference_contrib, model_contrib, failure_driver),
            }
        )

    problematic_top_n = int(config["modal"].get("diagnostics", {}).get("problematic_top_n", 3))
    problematic_modes = sorted(
        [item for item in diagnostics if item["classification"] == "Problematic"],
        key=lambda item: (
            -float(item["priority_score"]),
            -float(item["absolute_frequency_error_percent"]),
            float(item["paired_mac"]),
            int(item["reference_mode_number"]),
        ),
    )
    prioritized_problematic: list[dict[str, Any]] = []
    for index, item in enumerate(problematic_modes[:problematic_top_n], start=1):
        prioritized_item = dict(item)
        prioritized_item["priority_rank"] = index
        prioritized_problematic.append(prioritized_item)

    return diagnostics, {
        "criteria": thresholds,
        "counts": {
            "good": int(counts["Good"]),
            "acceptable": int(counts["Acceptable"]),
            "problematic": int(counts["Problematic"]),
        },
        "top_problematic": prioritized_problematic,
        "problematic_reference_modes": [int(item["reference_mode_number"]) for item in prioritized_problematic],
    }


def score_modal_summary(modal_summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    pairing = modal_summary["pairing"]
    weights = config["modal"]["ranking_weights"]
    num_modes = max(int(config["modal"]["num_modes"]), 1)
    good_both_fraction = float(pairing["good_on_both_count"]) / float(num_modes)

    score = (
        float(weights["mean_frequency_error"]) * float(pairing["mean_frequency_error"])
        + float(weights["worst_frequency_error"]) * float(pairing["worst_frequency_error"])
        - float(weights["mean_mac"]) * float(pairing["mean_mac"])
        - float(weights["worst_mac"]) * float(pairing["worst_mac"])
        - float(weights["threshold_hits"]) * good_both_fraction
    )
    return {
        "modal_score": round(score, 8),
        "components": {
            "mean_mac": round(float(pairing["mean_mac"]), 8),
            "worst_mac": round(float(pairing["worst_mac"]), 8),
            "mean_frequency_error": round(float(pairing["mean_frequency_error"]), 8),
            "worst_frequency_error": round(float(pairing["worst_frequency_error"]), 8),
            "good_on_both_fraction": round(good_both_fraction, 8),
        },
    }


def build_modal_correlation(
    reference_text: str,
    model_text: str,
    config: dict[str, Any],
    reference_source_name: str,
    model_source_name: str,
) -> dict[str, Any]:
    reference_summary = parse_modal_f06(reference_text, config, reference_source_name)
    model_summary = parse_modal_f06(model_text, config, model_source_name)

    reference_modes = reference_summary["modes"]
    model_modes = model_summary["modes"]
    if len(reference_modes) != len(model_modes):
        raise ValueError(
            f"Reference modes ({len(reference_modes)}) and model modes ({len(model_modes)}) do not match."
        )

    mac_matrix = calculate_mac_matrix(reference_modes, model_modes)
    pairing = pair_modal_modes(
        reference_modes,
        model_modes,
        mac_matrix,
        pairing_frequency_penalty_weight=float(config["modal"]["pairing_frequency_penalty_weight"]),
    )

    mac_threshold = float(config["modal"]["mac_good_threshold"])
    frequency_threshold = float(config["modal"]["frequency_error_good_threshold"])
    paired_mac_values = [float(item["mac"]) for item in pairing["pairs"]]
    paired_frequency_errors = [float(item["frequency_error_rel"]) for item in pairing["pairs"]]

    for item in pairing["pairs"]:
        item["mac_good"] = float(item["mac"]) >= mac_threshold
        item["frequency_good"] = float(item["frequency_error_rel"]) <= frequency_threshold
        item["good_on_both"] = bool(item["mac_good"] and item["frequency_good"])

    pairing["mean_mac"] = _mean(paired_mac_values)
    pairing["worst_mac"] = _worst(paired_mac_values, reverse=True)
    pairing["mean_frequency_error"] = _mean(paired_frequency_errors)
    pairing["worst_frequency_error"] = _worst(paired_frequency_errors)
    pairing["mac_good_count"] = sum(1 for item in pairing["pairs"] if item["mac_good"])
    pairing["frequency_good_count"] = sum(1 for item in pairing["pairs"] if item["frequency_good"])
    pairing["good_on_both_count"] = sum(1 for item in pairing["pairs"] if item["good_on_both"])

    paired_mode_diagnostics, problematic_modes = _build_paired_mode_diagnostics(
        reference_modes,
        model_modes,
        pairing,
        config,
    )

    modal_score = score_modal_summary({"pairing": pairing}, config)
    frequency_comparison = []
    for reference_mode, model_mode in zip(reference_modes, model_modes):
        signed_frequency_error_percent = calculate_signed_frequency_error_percent(
            reference_mode["frequency_hz"],
            model_mode["frequency_hz"],
        )
        frequency_comparison.append(
            {
                "reference_mode_number": reference_mode["mode_number"],
                "model_mode_number": model_mode["mode_number"],
                "reference_frequency_hz": round(float(reference_mode["frequency_hz"]), 6),
                "model_frequency_hz": round(float(model_mode["frequency_hz"]), 6),
                "frequency_error_rel": _frequency_error(reference_mode["frequency_hz"], model_mode["frequency_hz"]),
                "signed_frequency_error_percent": signed_frequency_error_percent,
                "absolute_frequency_error_percent": calculate_absolute_frequency_error_percent(
                    signed_frequency_error_percent
                ),
            }
        )

    return {
        "status": "completed",
        "reference": reference_summary,
        "model": model_summary,
        "measurement_points": modal_measurement_points(config),
        "vector_order_labels": modal_vector_labels(config),
        "frequency_comparison_diagonal": frequency_comparison,
        "mac_matrix": mac_matrix,
        "pairing": pairing,
        "paired_mode_diagnostics": paired_mode_diagnostics,
        "problematic_modes": problematic_modes,
        "summary": {
            "num_modes": len(reference_modes),
            "mean_mac": pairing["mean_mac"],
            "worst_mac": pairing["worst_mac"],
            "mean_frequency_error": pairing["mean_frequency_error"],
            "worst_frequency_error": pairing["worst_frequency_error"],
            "mac_good_count": pairing["mac_good_count"],
            "frequency_good_count": pairing["frequency_good_count"],
            "good_on_both_count": pairing["good_on_both_count"],
            "good_count": problematic_modes["counts"]["good"],
            "acceptable_count": problematic_modes["counts"]["acceptable"],
            "problematic_count": problematic_modes["counts"]["problematic"],
            "mac_good_threshold": mac_threshold,
            "frequency_error_good_threshold": frequency_threshold,
            "modal_score": modal_score["modal_score"],
            "modal_score_components": modal_score["components"],
        },
        "parser_notes": reference_summary["notes"] + model_summary["notes"] + [
            "El parser actual usa .f06 con bloques de REAL EIGENVECTOR y solo consume T1/T2/T3.",
            "La arquitectura queda preparada para soportar .pch u .op2 mas adelante sin meter esa complejidad todavia.",
        ],
    }
