from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import json

import numpy as np
from scipy.optimize import linear_sum_assignment


class ModalMetricError(ValueError):
    """Raised when modal comparison inputs are inconsistent."""


@dataclass(frozen=True)
class PairingEntry:
    reference_mode: int
    model_mode: int
    reference_frequency_hz: float
    model_frequency_hz: float
    relative_frequency_error: float
    mac: float


@dataclass(frozen=True)
class ModalComparisonResult:
    mac_matrix: list[list[float]]
    diagonal_mac: list[float]
    frequency_error_matrix: list[list[float]]
    pairing: tuple[PairingEntry, ...]
    mean_mac: float
    worst_mac: float
    mean_relative_frequency_error: float
    score: float
    score_mode: str
    ranking_key: tuple[float, ...]
    score_breakdown: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pairing"] = [asdict(item) for item in self.pairing]
        return payload


def build_phi_matrix(
    vectors_by_mode: dict[int, dict[int, tuple[float, float, float]]],
    point_order: list[int],
    num_modes: int,
) -> np.ndarray:
    rows: list[list[float]] = []
    for mode in range(1, num_modes + 1):
        if mode not in vectors_by_mode:
            raise ModalMetricError(f"Mode {mode} is missing from the vector data.")
        row: list[float] = []
        for point_id in point_order:
            if point_id not in vectors_by_mode[mode]:
                raise ModalMetricError(f"Point {point_id} is missing from mode {mode}.")
            row.extend(vectors_by_mode[mode][point_id])
        rows.append(row)
    return np.asarray(rows, dtype=float)


def mac(phi_ref: np.ndarray, phi_model: np.ndarray) -> float:
    numerator = float(phi_ref.T @ phi_model) ** 2
    denominator = float(phi_ref.T @ phi_ref) * float(phi_model.T @ phi_model)
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def compute_mac_matrix(reference_phi: np.ndarray, model_phi: np.ndarray) -> np.ndarray:
    matrix = np.zeros((reference_phi.shape[0], model_phi.shape[0]), dtype=float)
    for ref_index, ref_vector in enumerate(reference_phi):
        for model_index, model_vector in enumerate(model_phi):
            matrix[ref_index, model_index] = mac(ref_vector, model_vector)
    return matrix


def compute_frequency_error_matrix(
    reference_frequencies: list[float],
    model_frequencies: list[float],
) -> np.ndarray:
    matrix = np.zeros((len(reference_frequencies), len(model_frequencies)), dtype=float)
    for ref_index, ref_frequency in enumerate(reference_frequencies):
        for model_index, model_frequency in enumerate(model_frequencies):
            matrix[ref_index, model_index] = abs(model_frequency - ref_frequency) / ref_frequency
    return matrix


def _pair_modes(
    mac_matrix: np.ndarray,
    frequency_error_matrix: np.ndarray,
    reference_frequencies: list[float],
    model_frequencies: list[float],
    frequency_penalty_weight: float,
) -> tuple[PairingEntry, ...]:
    cost_matrix = (1.0 - mac_matrix) + frequency_penalty_weight * frequency_error_matrix
    ref_indices, model_indices = linear_sum_assignment(cost_matrix)

    pairing = []
    for ref_index, model_index in zip(ref_indices.tolist(), model_indices.tolist(), strict=True):
        pairing.append(
            PairingEntry(
                reference_mode=ref_index + 1,
                model_mode=model_index + 1,
                reference_frequency_hz=reference_frequencies[ref_index],
                model_frequency_hz=model_frequencies[model_index],
                relative_frequency_error=float(frequency_error_matrix[ref_index, model_index]),
                mac=float(mac_matrix[ref_index, model_index]),
            )
        )
    pairing.sort(key=lambda item: item.reference_mode)
    return tuple(pairing)


def _score_modal_comparison(
    mean_mac: float,
    mean_relative_frequency_error: float,
    relative_mass_error: float,
    scoring_mode: str,
    scoring_weights: dict[str, float],
) -> tuple[float, tuple[float, ...], dict[str, float]]:
    if scoring_mode == "weighted_error":
        score = (
            scoring_weights["mac"] * (1.0 - mean_mac)
            + scoring_weights["frequency"] * mean_relative_frequency_error
            + scoring_weights["mass"] * relative_mass_error
        )
        ranking_key = (score,)
        breakdown = {
            "mac_component": scoring_weights["mac"] * (1.0 - mean_mac),
            "frequency_component": scoring_weights["frequency"] * mean_relative_frequency_error,
            "mass_component": scoring_weights["mass"] * relative_mass_error,
        }
        return score, ranking_key, breakdown

    if scoring_mode == "mac_first":
        score = -mean_mac
        ranking_key = (-mean_mac, mean_relative_frequency_error, relative_mass_error)
        breakdown = {
            "primary_objective": mean_mac,
            "frequency_tiebreaker": mean_relative_frequency_error,
            "mass_tiebreaker": relative_mass_error,
        }
        return score, ranking_key, breakdown

    raise ModalMetricError(f"Unsupported scoring mode '{scoring_mode}'.")


def compare_modal_results(
    reference_frequencies: list[float],
    model_frequencies: list[float],
    reference_vectors: dict[int, dict[int, tuple[float, float, float]]],
    model_vectors: dict[int, dict[int, tuple[float, float, float]]],
    reference_point_order: list[int],
    model_point_order: list[int],
    num_modes: int,
    frequency_penalty_weight: float,
    scoring_mode: str,
    scoring_weights: dict[str, float],
    relative_mass_error: float,
) -> ModalComparisonResult:
    if len(reference_frequencies) < num_modes or len(model_frequencies) < num_modes:
        raise ModalMetricError("Not enough frequencies to compare the requested number of modes.")

    reference_phi = build_phi_matrix(reference_vectors, reference_point_order, num_modes)
    model_phi = build_phi_matrix(model_vectors, model_point_order, num_modes)

    truncated_reference_frequencies = reference_frequencies[:num_modes]
    truncated_model_frequencies = model_frequencies[:num_modes]
    mac_matrix = compute_mac_matrix(reference_phi, model_phi)
    frequency_error_matrix = compute_frequency_error_matrix(
        truncated_reference_frequencies,
        truncated_model_frequencies,
    )
    pairing = _pair_modes(
        mac_matrix=mac_matrix,
        frequency_error_matrix=frequency_error_matrix,
        reference_frequencies=truncated_reference_frequencies,
        model_frequencies=truncated_model_frequencies,
        frequency_penalty_weight=frequency_penalty_weight,
    )

    diagonal_mac = [float(mac_matrix[index, index]) for index in range(num_modes)]
    mean_mac = float(np.mean([item.mac for item in pairing]))
    worst_mac = float(np.min([item.mac for item in pairing]))
    mean_relative_frequency_error = float(
        np.mean([item.relative_frequency_error for item in pairing])
    )
    score, ranking_key, score_breakdown = _score_modal_comparison(
        mean_mac=mean_mac,
        mean_relative_frequency_error=mean_relative_frequency_error,
        relative_mass_error=relative_mass_error,
        scoring_mode=scoring_mode,
        scoring_weights=scoring_weights,
    )

    return ModalComparisonResult(
        mac_matrix=mac_matrix.tolist(),
        diagonal_mac=diagonal_mac,
        frequency_error_matrix=frequency_error_matrix.tolist(),
        pairing=pairing,
        mean_mac=mean_mac,
        worst_mac=worst_mac,
        mean_relative_frequency_error=mean_relative_frequency_error,
        score=score,
        score_mode=scoring_mode,
        ranking_key=ranking_key,
        score_breakdown=score_breakdown,
    )


def rank_runs(run_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(run_payloads, key=lambda item: tuple(item["modal_metrics"]["ranking_key"]))


def write_metrics_json(result: ModalComparisonResult, path: str) -> None:
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(result.to_dict(), stream, indent=2)

