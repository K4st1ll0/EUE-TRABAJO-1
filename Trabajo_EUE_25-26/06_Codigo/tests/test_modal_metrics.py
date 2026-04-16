import numpy as np
import pytest

from src.config import (
    AcceptanceConfig,
    ModeFamilyLabelsConfig,
    PairingConfig,
    RankingConfig,
    RankingRobustWeights,
    ScoringConfig,
    ScoringWeights,
)
from src.f06_parser import ParsedModalData
from src.modal_metrics import compare_modal_results, mac


def _vector_to_mode_shape(vector: np.ndarray, point_order: list[int]) -> dict[int, tuple[float, float, float]]:
    mode_shape = {}
    for index, point_id in enumerate(point_order):
        start = index * 3
        mode_shape[point_id] = tuple(float(value) for value in vector[start : start + 3])
    return mode_shape


def _build_modal_data(
    *,
    point_order: list[int],
    num_modes: int,
    frequencies: dict[int, float],
    vectors: dict[int, dict[int, tuple[float, float, float]]],
    valid_modes: tuple[int, ...],
) -> ParsedModalData:
    requested_modes = tuple(range(1, num_modes + 1))
    invalid_modes = tuple(mode for mode in requested_modes if mode not in valid_modes)
    missing_frequency_modes = tuple(mode for mode in requested_modes if mode not in frequencies)
    missing_vector_modes = tuple(mode for mode in requested_modes if mode not in vectors)
    return ParsedModalData(
        frequencies_hz=frequencies,
        vectors=vectors,
        requested_modes=requested_modes,
        requested_point_ids=tuple(point_order),
        valid_modes=valid_modes,
        invalid_modes=invalid_modes,
        missing_frequency_modes=missing_frequency_modes,
        missing_vector_modes=missing_vector_modes,
        missing_points_by_mode={},
        valid_mode_count=len(valid_modes),
        vector_basis="T1_T2_T3",
        coordinate_system="global",
        source_format="f06",
        warnings=(),
    )


def _comparison_kwargs() -> dict[str, object]:
    return {
        "pairing": PairingConfig(
            freq_gate=0.20,
            alpha_freq=0.15,
            huge_penalty=1_000_000.0,
            bad_pair_mac_threshold=0.10,
            warning_pair_mac_threshold=0.30,
        ),
        "acceptance": AcceptanceConfig(
            max_bad_pairs_allowed=1,
            reject_if_any_pair_outside_gate=True,
            min_valid_modes=10,
        ),
        "scoring": ScoringConfig(
            mode="weighted_error",
            weights=ScoringWeights(mac=1.0, frequency=0.25, mass=0.10),
        ),
        "ranking": RankingConfig(
            mode="hierarchical",
            robust_weights=RankingRobustWeights(mac=0.05, bad_pair_fraction=0.20),
        ),
        "relative_mass_error": 0.0,
        "mode_family_labels": ModeFamilyLabelsConfig(reference={}, model={}),
    }


def test_mac_formula_matches_expected_limits() -> None:
    phi = np.array([1.0, 2.0, 3.0])
    orthogonal = np.array([0.0, 1.0, 0.0])

    assert mac(phi, phi) == 1.0
    assert mac(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 0.0
    assert mac(phi, orthogonal) == ((phi @ orthogonal) ** 2) / ((phi @ phi) * (orthogonal @ orthogonal))


def test_modal_pairing_uses_valid_submatrix_and_keeps_full_10x10_outputs() -> None:
    point_order = [1, 2, 3, 4]
    num_modes = 10
    reference_frequencies = {mode: 100.0 + 5.0 * (mode - 1) for mode in range(1, 11)}
    permutation = [2, 1, 3, 4, 5, 6, 7, 8, 9, 10]
    model_frequencies = {
        mode: reference_frequencies[permutation[mode - 1]] * 1.01 for mode in range(1, 11)
    }

    basis_vectors = np.eye(12)[:10]
    reference_vectors = {
        mode: _vector_to_mode_shape(basis_vectors[mode - 1], point_order)
        for mode in range(1, 11)
    }
    model_vectors = {
        mode: _vector_to_mode_shape(basis_vectors[permutation[mode - 1] - 1], point_order)
        for mode in range(1, 11)
    }

    reference_data = _build_modal_data(
        point_order=point_order,
        num_modes=num_modes,
        frequencies=reference_frequencies,
        vectors=reference_vectors,
        valid_modes=tuple(range(1, 11)),
    )
    model_data = _build_modal_data(
        point_order=point_order,
        num_modes=num_modes,
        frequencies=model_frequencies,
        vectors=model_vectors,
        valid_modes=tuple(range(1, 11)),
    )

    result = compare_modal_results(
        reference_data=reference_data,
        model_data=model_data,
        reference_point_order=point_order,
        model_point_order=point_order,
        num_modes=num_modes,
        **_comparison_kwargs(),
    )

    assert len(result.mac_matrix) == 10
    assert len(result.mac_matrix[0]) == 10
    assert result.pairing[0].model_mode == 2
    assert result.pairing[1].model_mode == 1
    assert result.mean_mac == pytest.approx(1.0)
    assert result.paired_valid_mode_count == 10
    assert result.accepted is True


def test_no_valid_pairs_sets_null_metrics_and_rejects_run() -> None:
    point_order = [1, 2, 3, 4]
    num_modes = 10
    reference_data = _build_modal_data(
        point_order=point_order,
        num_modes=num_modes,
        frequencies={},
        vectors={},
        valid_modes=(),
    )
    model_data = _build_modal_data(
        point_order=point_order,
        num_modes=num_modes,
        frequencies={},
        vectors={},
        valid_modes=(),
    )

    result = compare_modal_results(
        reference_data=reference_data,
        model_data=model_data,
        reference_point_order=point_order,
        model_point_order=point_order,
        num_modes=num_modes,
        **_comparison_kwargs(),
    )

    assert result.pairing == ()
    assert result.mean_mac is None
    assert result.worst_mac is None
    assert result.mean_relative_frequency_error is None
    assert result.max_relative_frequency_error is None
    assert result.accepted is False
    assert "no_valid_pairs" in result.reject_reasons
