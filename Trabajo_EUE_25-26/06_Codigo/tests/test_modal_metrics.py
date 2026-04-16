import numpy as np
import pytest

from src.modal_metrics import compare_modal_results, mac


def _vector_to_mode_shape(vector: np.ndarray, point_order: list[int]) -> dict[int, tuple[float, float, float]]:
    mode_shape = {}
    for index, point_id in enumerate(point_order):
        start = index * 3
        mode_shape[point_id] = tuple(float(value) for value in vector[start : start + 3])
    return mode_shape


def test_mac_formula_matches_expected_limits() -> None:
    phi = np.array([1.0, 2.0, 3.0])
    orthogonal = np.array([0.0, 1.0, 0.0])

    assert mac(phi, phi) == 1.0
    assert mac(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 0.0
    assert mac(phi, orthogonal) == ((phi @ orthogonal) ** 2) / ((phi @ phi) * (orthogonal @ orthogonal))


def test_modal_pairing_uses_mac_to_recover_permuted_modes() -> None:
    point_order = [1, 2, 3, 4]
    reference_frequencies = [100.0 + 5.0 * index for index in range(10)]
    permutation = [1, 0, 2, 3, 4, 5, 6, 7, 8, 9]
    model_frequencies = [reference_frequencies[index] * 1.01 for index in permutation]

    basis_vectors = np.eye(12)[:10]
    reference_vectors = {
        mode: _vector_to_mode_shape(basis_vectors[mode - 1], point_order)
        for mode in range(1, 11)
    }
    model_vectors = {
        mode: _vector_to_mode_shape(basis_vectors[permutation[mode - 1]], point_order)
        for mode in range(1, 11)
    }

    result = compare_modal_results(
        reference_frequencies=reference_frequencies,
        model_frequencies=model_frequencies,
        reference_vectors=reference_vectors,
        model_vectors=model_vectors,
        reference_point_order=point_order,
        model_point_order=point_order,
        num_modes=10,
        frequency_penalty_weight=0.05,
        scoring_mode="weighted_error",
        scoring_weights={"mac": 1.0, "frequency": 0.25, "mass": 0.10},
        relative_mass_error=0.0,
    )

    assert len(result.mac_matrix) == 10
    assert len(result.mac_matrix[0]) == 10
    assert result.pairing[0].model_mode == 2
    assert result.pairing[1].model_mode == 1
    assert result.mean_mac == 1.0
    assert result.stiffness_fit_indicator == pytest.approx(result.mean_relative_frequency_error)
