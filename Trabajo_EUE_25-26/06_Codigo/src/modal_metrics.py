from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import json

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import (
    AcceptanceConfig,
    ModeFamilyLabelsConfig,
    PairingConfig,
    RankingConfig,
    ScoringConfig,
)
from .f06_parser import ParsedModalData


class ModalMetricError(ValueError):
    """Raised when modal comparison inputs are inconsistent."""


RANKING_SENTINEL = 1.0e30


@dataclass(frozen=True)
class PairingEntry:
    reference_mode: int
    model_mode: int
    reference_frequency_hz: float
    model_frequency_hz: float
    relative_frequency_error: float
    mac: float
    outside_gate: bool
    bad_pair: bool
    warning_pair: bool
    suspicious: bool
    suspicious_reason: str | None
    reference_family_label: str | None
    model_family_label: str | None


@dataclass(frozen=True)
class ModalComparisonResult:
    mac_matrix: list[list[float | None]]
    diagonal_mac: list[float | None]
    frequency_error_matrix: list[list[float | None]]
    allowed_pairs_matrix: list[list[bool]]
    pairing: tuple[PairingEntry, ...]
    paired_mac_by_mode: list[float | None]
    paired_frequency_error_by_mode: list[float | None]
    unpaired_reference_modes: tuple[int, ...]
    unpaired_model_modes: tuple[int, ...]
    valid_reference_mode_count: int
    valid_model_mode_count: int
    paired_valid_mode_count: int
    mean_mac: float | None
    worst_mac: float | None
    mean_relative_frequency_error: float | None
    max_relative_frequency_error: float | None
    bad_pair_count: int
    bad_pair_fraction: float
    warning_pair_count: int
    warning_pair_fraction: float
    stiffness_fit_indicator: float | None
    score: float | None
    score_mode: str
    score_legacy: float | None
    score_robust: float | None
    selected_ranking_mode: str
    ranking_key: tuple[float, ...]
    score_breakdown: dict[str, float | None]
    accepted: bool
    reject_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pairing"] = [asdict(item) for item in self.pairing]
        return payload


def build_phi_matrix(
    vectors_by_mode: dict[int, dict[int, tuple[float, float, float]]],
    point_order: list[int],
    modes: list[int],
) -> dict[int, np.ndarray]:
    rows: dict[int, np.ndarray] = {}
    for mode in modes:
        if mode not in vectors_by_mode:
            raise ModalMetricError(f"Mode {mode} is missing from the vector data.")
        row: list[float] = []
        for point_id in point_order:
            if point_id not in vectors_by_mode[mode]:
                raise ModalMetricError(f"Point {point_id} is missing from mode {mode}.")
            row.extend(vectors_by_mode[mode][point_id])
        rows[mode] = np.asarray(row, dtype=float)
    return rows


def mac(phi_ref: np.ndarray, phi_model: np.ndarray) -> float:
    numerator = float(phi_ref.T @ phi_model) ** 2
    denominator = float(phi_ref.T @ phi_ref) * float(phi_model.T @ phi_model)
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _compute_relative_frequency_error(reference_frequency_hz: float, model_frequency_hz: float) -> float:
    return abs(model_frequency_hz - reference_frequency_hz) / reference_frequency_hz


def _build_full_metric_matrices(
    reference_data: ParsedModalData,
    model_data: ParsedModalData,
    reference_point_order: list[int],
    model_point_order: list[int],
    num_modes: int,
    pairing_config: PairingConfig,
) -> tuple[list[list[float | None]], list[list[float | None]], list[list[bool]]]:
    reference_modes = list(reference_data.valid_modes)
    model_modes = list(model_data.valid_modes)
    reference_phi = build_phi_matrix(reference_data.vectors, reference_point_order, reference_modes)
    model_phi = build_phi_matrix(model_data.vectors, model_point_order, model_modes)

    mac_matrix: list[list[float | None]] = [[None for _ in range(num_modes)] for _ in range(num_modes)]
    frequency_error_matrix: list[list[float | None]] = [
        [None for _ in range(num_modes)] for _ in range(num_modes)
    ]
    allowed_pairs_matrix: list[list[bool]] = [[False for _ in range(num_modes)] for _ in range(num_modes)]

    for reference_mode in reference_modes:
        reference_vector = reference_phi[reference_mode]
        reference_frequency = reference_data.frequencies_hz[reference_mode]
        for model_mode in model_modes:
            model_vector = model_phi[model_mode]
            model_frequency = model_data.frequencies_hz[model_mode]
            mac_value = mac(reference_vector, model_vector)
            frequency_error = _compute_relative_frequency_error(reference_frequency, model_frequency)
            mac_matrix[reference_mode - 1][model_mode - 1] = float(mac_value)
            frequency_error_matrix[reference_mode - 1][model_mode - 1] = float(frequency_error)
            allowed_pairs_matrix[reference_mode - 1][model_mode - 1] = (
                frequency_error <= pairing_config.freq_gate
            )

    return mac_matrix, frequency_error_matrix, allowed_pairs_matrix


def _pair_valid_modes(
    mac_matrix: list[list[float | None]],
    frequency_error_matrix: list[list[float | None]],
    allowed_pairs_matrix: list[list[bool]],
    reference_data: ParsedModalData,
    model_data: ParsedModalData,
    pairing_config: PairingConfig,
    mode_family_labels: ModeFamilyLabelsConfig,
) -> tuple[PairingEntry, ...]:
    reference_modes = list(reference_data.valid_modes)
    model_modes = list(model_data.valid_modes)
    if not reference_modes or not model_modes:
        return ()

    cost_matrix = np.full((len(reference_modes), len(model_modes)), pairing_config.huge_penalty, dtype=float)
    for ref_index, reference_mode in enumerate(reference_modes):
        for model_index, model_mode in enumerate(model_modes):
            mac_value = mac_matrix[reference_mode - 1][model_mode - 1]
            frequency_error = frequency_error_matrix[reference_mode - 1][model_mode - 1]
            if mac_value is None or frequency_error is None:
                continue
            if allowed_pairs_matrix[reference_mode - 1][model_mode - 1]:
                cost_matrix[ref_index, model_index] = (1.0 - mac_value) + (
                    pairing_config.alpha_freq * frequency_error
                )

    ref_indices, model_indices = linear_sum_assignment(cost_matrix)
    pairing: list[PairingEntry] = []
    for ref_index, model_index in zip(ref_indices.tolist(), model_indices.tolist(), strict=True):
        reference_mode = reference_modes[ref_index]
        model_mode = model_modes[model_index]
        reference_frequency = reference_data.frequencies_hz[reference_mode]
        model_frequency = model_data.frequencies_hz[model_mode]
        mac_value = mac_matrix[reference_mode - 1][model_mode - 1]
        frequency_error = frequency_error_matrix[reference_mode - 1][model_mode - 1]
        if mac_value is None or frequency_error is None:
            continue

        outside_gate = not allowed_pairs_matrix[reference_mode - 1][model_mode - 1]
        bad_pair = mac_value < pairing_config.bad_pair_mac_threshold
        warning_pair = mac_value < pairing_config.warning_pair_mac_threshold
        reference_family_label = mode_family_labels.reference.get(reference_mode)
        model_family_label = mode_family_labels.model.get(model_mode)
        suspicious = False
        suspicious_reason = None
        if reference_family_label and model_family_label and reference_family_label != model_family_label:
            suspicious = True
            suspicious_reason = "mode_family_mismatch"

        pairing.append(
            PairingEntry(
                reference_mode=reference_mode,
                model_mode=model_mode,
                reference_frequency_hz=reference_frequency,
                model_frequency_hz=model_frequency,
                relative_frequency_error=frequency_error,
                mac=mac_value,
                outside_gate=outside_gate,
                bad_pair=bad_pair,
                warning_pair=warning_pair,
                suspicious=suspicious,
                suspicious_reason=suspicious_reason,
                reference_family_label=reference_family_label,
                model_family_label=model_family_label,
            )
        )

    pairing.sort(key=lambda item: item.reference_mode)
    return tuple(pairing)


def _legacy_score(
    *,
    scoring: ScoringConfig,
    mean_mac: float | None,
    mean_relative_frequency_error: float | None,
    relative_mass_error: float,
) -> tuple[float | None, dict[str, float | None]]:
    if mean_mac is None or mean_relative_frequency_error is None:
        return None, {
            "mac_component": None,
            "frequency_component": None,
            "mass_component": None,
        }

    if scoring.mode == "weighted_error":
        mac_component = scoring.weights.mac * (1.0 - mean_mac)
        frequency_component = scoring.weights.frequency * mean_relative_frequency_error
        mass_component = scoring.weights.mass * relative_mass_error
        return (
            mac_component + frequency_component + mass_component,
            {
                "mac_component": mac_component,
                "frequency_component": frequency_component,
                "mass_component": mass_component,
            },
        )

    if scoring.mode == "mac_first":
        return (
            -mean_mac,
            {
                "mac_component": mean_mac,
                "frequency_component": mean_relative_frequency_error,
                "mass_component": relative_mass_error,
            },
        )

    raise ModalMetricError(f"Unsupported legacy scoring mode '{scoring.mode}'.")


def _robust_score(
    *,
    mean_mac: float | None,
    mean_relative_frequency_error: float | None,
    bad_pair_fraction: float,
    ranking: RankingConfig,
) -> float | None:
    if mean_mac is None or mean_relative_frequency_error is None:
        return None
    return (
        mean_relative_frequency_error
        + ranking.robust_weights.mac * (1.0 - mean_mac)
        + ranking.robust_weights.bad_pair_fraction * bad_pair_fraction
    )


def _build_ranking_key(
    *,
    ranking: RankingConfig,
    bad_pair_count: int,
    mean_mac: float | None,
    mean_relative_frequency_error: float | None,
    score_robust: float | None,
) -> tuple[float, ...]:
    if ranking.mode == "hierarchical":
        return (
            float(bad_pair_count),
            mean_relative_frequency_error if mean_relative_frequency_error is not None else RANKING_SENTINEL,
            (1.0 - mean_mac) if mean_mac is not None else RANKING_SENTINEL,
        )
    if ranking.mode == "robust_scalar":
        return (score_robust if score_robust is not None else RANKING_SENTINEL,)
    raise ModalMetricError(f"Unsupported ranking mode '{ranking.mode}'.")


def _build_acceptance(
    *,
    acceptance: AcceptanceConfig,
    bad_pair_count: int,
    has_outside_gate_pair: bool,
    valid_model_mode_count: int,
    paired_valid_mode_count: int,
) -> tuple[bool, tuple[str, ...]]:
    reject_reasons: list[str] = []
    if paired_valid_mode_count == 0:
        reject_reasons.append("no_valid_pairs")
    if bad_pair_count > acceptance.max_bad_pairs_allowed:
        reject_reasons.append("too_many_bad_pairs")
    if acceptance.reject_if_any_pair_outside_gate and has_outside_gate_pair:
        reject_reasons.append("outside_frequency_gate")
    if valid_model_mode_count < acceptance.min_valid_modes:
        reject_reasons.append("valid_model_mode_count_below_min")
    if paired_valid_mode_count < acceptance.min_valid_modes:
        reject_reasons.append("paired_valid_mode_count_below_min")
    return (len(reject_reasons) == 0), tuple(reject_reasons)


def compare_modal_results(
    *,
    reference_data: ParsedModalData,
    model_data: ParsedModalData,
    reference_point_order: list[int],
    model_point_order: list[int],
    num_modes: int,
    pairing: PairingConfig,
    acceptance: AcceptanceConfig,
    scoring: ScoringConfig,
    ranking: RankingConfig,
    relative_mass_error: float,
    mode_family_labels: ModeFamilyLabelsConfig,
) -> ModalComparisonResult:
    if num_modes <= 0:
        raise ModalMetricError("num_modes must be greater than zero.")

    mac_matrix, frequency_error_matrix, allowed_pairs_matrix = _build_full_metric_matrices(
        reference_data=reference_data,
        model_data=model_data,
        reference_point_order=reference_point_order,
        model_point_order=model_point_order,
        num_modes=num_modes,
        pairing_config=pairing,
    )
    pairing_entries = _pair_valid_modes(
        mac_matrix=mac_matrix,
        frequency_error_matrix=frequency_error_matrix,
        allowed_pairs_matrix=allowed_pairs_matrix,
        reference_data=reference_data,
        model_data=model_data,
        pairing_config=pairing,
        mode_family_labels=mode_family_labels,
    )

    diagonal_mac = [mac_matrix[index][index] for index in range(num_modes)]
    paired_mac_by_mode: list[float | None] = [None for _ in range(num_modes)]
    paired_frequency_error_by_mode: list[float | None] = [None for _ in range(num_modes)]
    for entry in pairing_entries:
        paired_mac_by_mode[entry.reference_mode - 1] = entry.mac
        paired_frequency_error_by_mode[entry.reference_mode - 1] = entry.relative_frequency_error

    paired_reference_modes = {entry.reference_mode for entry in pairing_entries}
    paired_model_modes = {entry.model_mode for entry in pairing_entries}
    unpaired_reference_modes = tuple(
        mode for mode in range(1, num_modes + 1) if mode not in paired_reference_modes
    )
    unpaired_model_modes = tuple(
        mode for mode in range(1, num_modes + 1) if mode not in paired_model_modes
    )

    paired_valid_mode_count = len(pairing_entries)
    valid_reference_mode_count = reference_data.valid_mode_count
    valid_model_mode_count = model_data.valid_mode_count

    if pairing_entries:
        paired_macs = [entry.mac for entry in pairing_entries]
        paired_frequency_errors = [entry.relative_frequency_error for entry in pairing_entries]
        mean_mac = float(np.mean(paired_macs))
        worst_mac = float(np.min(paired_macs))
        mean_relative_frequency_error = float(np.mean(paired_frequency_errors))
        max_relative_frequency_error = float(np.max(paired_frequency_errors))
    else:
        mean_mac = None
        worst_mac = None
        mean_relative_frequency_error = None
        max_relative_frequency_error = None

    bad_pair_count = sum(1 for entry in pairing_entries if entry.bad_pair)
    warning_pair_count = sum(1 for entry in pairing_entries if entry.warning_pair)
    bad_pair_fraction = (
        bad_pair_count / paired_valid_mode_count if paired_valid_mode_count else 0.0
    )
    warning_pair_fraction = (
        warning_pair_count / paired_valid_mode_count if paired_valid_mode_count else 0.0
    )
    has_outside_gate_pair = any(entry.outside_gate for entry in pairing_entries)
    accepted, reject_reasons = _build_acceptance(
        acceptance=acceptance,
        bad_pair_count=bad_pair_count,
        has_outside_gate_pair=has_outside_gate_pair,
        valid_model_mode_count=valid_model_mode_count,
        paired_valid_mode_count=paired_valid_mode_count,
    )

    legacy_score, legacy_breakdown = _legacy_score(
        scoring=scoring,
        mean_mac=mean_mac,
        mean_relative_frequency_error=mean_relative_frequency_error,
        relative_mass_error=relative_mass_error,
    )
    score_robust = _robust_score(
        mean_mac=mean_mac,
        mean_relative_frequency_error=mean_relative_frequency_error,
        bad_pair_fraction=bad_pair_fraction,
        ranking=ranking,
    )
    ranking_key = _build_ranking_key(
        ranking=ranking,
        bad_pair_count=bad_pair_count,
        mean_mac=mean_mac,
        mean_relative_frequency_error=mean_relative_frequency_error,
        score_robust=score_robust,
    )
    stiffness_fit_indicator = (
        None
        if mean_mac is None or mean_relative_frequency_error is None
        else mean_relative_frequency_error + (0.10 * (1.0 - mean_mac))
    )

    return ModalComparisonResult(
        mac_matrix=mac_matrix,
        diagonal_mac=diagonal_mac,
        frequency_error_matrix=frequency_error_matrix,
        allowed_pairs_matrix=allowed_pairs_matrix,
        pairing=pairing_entries,
        paired_mac_by_mode=paired_mac_by_mode,
        paired_frequency_error_by_mode=paired_frequency_error_by_mode,
        unpaired_reference_modes=unpaired_reference_modes,
        unpaired_model_modes=unpaired_model_modes,
        valid_reference_mode_count=valid_reference_mode_count,
        valid_model_mode_count=valid_model_mode_count,
        paired_valid_mode_count=paired_valid_mode_count,
        mean_mac=mean_mac,
        worst_mac=worst_mac,
        mean_relative_frequency_error=mean_relative_frequency_error,
        max_relative_frequency_error=max_relative_frequency_error,
        bad_pair_count=bad_pair_count,
        bad_pair_fraction=bad_pair_fraction,
        warning_pair_count=warning_pair_count,
        warning_pair_fraction=warning_pair_fraction,
        stiffness_fit_indicator=stiffness_fit_indicator,
        score=legacy_score,
        score_mode=scoring.mode,
        score_legacy=legacy_score,
        score_robust=score_robust,
        selected_ranking_mode=ranking.mode,
        ranking_key=ranking_key,
        score_breakdown=legacy_breakdown,
        accepted=accepted,
        reject_reasons=reject_reasons,
    )


def rank_runs(run_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        run_payloads,
        key=lambda item: tuple(
            float(value)
            for value in (item.get("modal_metrics") or {}).get("ranking_key", [RANKING_SENTINEL])
        ),
    )


def write_metrics_json(result: ModalComparisonResult, path: str) -> None:
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(result.to_dict(), stream, indent=2)
