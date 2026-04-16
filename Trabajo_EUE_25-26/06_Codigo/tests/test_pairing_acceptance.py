import numpy as np

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
from src.modal_metrics import compare_modal_results


def _vector_to_mode_shape(vector: np.ndarray, point_order: list[int]) -> dict[int, tuple[float, float, float]]:
    mode_shape = {}
    for index, point_id in enumerate(point_order):
        start = index * 3
        mode_shape[point_id] = tuple(float(value) for value in vector[start : start + 3])
    return mode_shape


def _make_modal_data(
    point_order: list[int],
    num_modes: int,
    valid_modes: tuple[int, ...],
    frequencies: dict[int, float],
    vectors: dict[int, dict[int, tuple[float, float, float]]],
) -> ParsedModalData:
    requested_modes = tuple(range(1, num_modes + 1))
    return ParsedModalData(
        frequencies_hz=frequencies,
        vectors=vectors,
        requested_modes=requested_modes,
        requested_point_ids=tuple(point_order),
        valid_modes=valid_modes,
        invalid_modes=tuple(mode for mode in requested_modes if mode not in valid_modes),
        missing_frequency_modes=tuple(mode for mode in requested_modes if mode not in frequencies),
        missing_vector_modes=tuple(mode for mode in requested_modes if mode not in vectors),
        missing_points_by_mode={},
        valid_mode_count=len(valid_modes),
        vector_basis="T1_T2_T3",
        coordinate_system="global",
        source_format="f06",
        warnings=(),
    )


def _default_kwargs() -> dict[str, object]:
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
            mode="robust_scalar",
            robust_weights=RankingRobustWeights(mac=0.05, bad_pair_fraction=0.20),
        ),
        "relative_mass_error": 0.0,
        "mode_family_labels": ModeFamilyLabelsConfig(reference={}, model={}),
    }


def test_acceptance_rejects_when_paired_valid_mode_count_is_below_minimum() -> None:
    point_order = [1, 2, 3, 4]
    num_modes = 10
    basis_vectors = np.eye(12)[:10]
    reference_vectors = {
        mode: _vector_to_mode_shape(basis_vectors[mode - 1], point_order) for mode in range(1, 11)
    }
    model_vectors = {
        mode: _vector_to_mode_shape(basis_vectors[mode - 1], point_order) for mode in range(1, 10)
    }
    reference_data = _make_modal_data(
        point_order,
        num_modes,
        tuple(range(1, 11)),
        {mode: 100.0 + mode for mode in range(1, 11)},
        reference_vectors,
    )
    model_data = _make_modal_data(
        point_order,
        num_modes,
        tuple(range(1, 10)),
        {mode: 100.0 + mode for mode in range(1, 10)},
        model_vectors,
    )

    result = compare_modal_results(
        reference_data=reference_data,
        model_data=model_data,
        reference_point_order=point_order,
        model_point_order=point_order,
        num_modes=num_modes,
        **_default_kwargs(),
    )

    assert result.paired_valid_mode_count == 9
    assert result.accepted is False
    assert "paired_valid_mode_count_below_min" in result.reject_reasons
    assert result.paired_mac_by_mode[-1] is None


def test_pairing_marks_warning_and_bad_pairs_from_final_matches_only() -> None:
    point_order = [1, 2, 3, 4]
    num_modes = 10
    reference_frequencies = {mode: 100.0 + mode for mode in range(1, 11)}
    model_frequencies = {mode: 100.0 + mode for mode in range(1, 11)}
    basis_vectors = np.eye(12)[:10]
    reference_vectors = {
        mode: _vector_to_mode_shape(basis_vectors[mode - 1], point_order) for mode in range(1, 11)
    }
    model_vectors = {
        mode: _vector_to_mode_shape(basis_vectors[mode - 1], point_order) for mode in range(1, 11)
    }
    model_vectors[10] = _vector_to_mode_shape(np.roll(basis_vectors[9], 1), point_order)

    reference_data = _make_modal_data(
        point_order,
        num_modes,
        tuple(range(1, 11)),
        reference_frequencies,
        reference_vectors,
    )
    model_data = _make_modal_data(
        point_order,
        num_modes,
        tuple(range(1, 11)),
        model_frequencies,
        model_vectors,
    )

    result = compare_modal_results(
        reference_data=reference_data,
        model_data=model_data,
        reference_point_order=point_order,
        model_point_order=point_order,
        num_modes=num_modes,
        **_default_kwargs(),
    )

    assert result.warning_pair_count >= result.bad_pair_count
    assert result.bad_pair_count >= 1
    assert result.score_robust is not None
