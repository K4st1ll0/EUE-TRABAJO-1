from pathlib import Path

import pytest

from src.sweep_engine import _allocate_run_id, _build_run_id, _next_run_sequence


def test_build_run_id_uses_requested_parameter_pattern() -> None:
    run_id = _build_run_id(
        {
            "aluminum_density_factor": 0.8,
            "pcb_density_factor": 1.0,
            "aluminum_E_factor": 1.0,
            "pcb_E_factor": 0.95,
        }
    )

    assert run_id == "Al0.800_PCB1.000_ALE1.000_PCBE0.950"


def test_next_run_sequence_ignores_legacy_names_and_uses_highest_prefix(tmp_path: Path) -> None:
    (tmp_path / "legacy").mkdir()
    (tmp_path / "Al0.800_PCB1.000_ALE1.000_PCBE1.000").mkdir()
    (tmp_path / "001_Al0.800_PCB1.000_ALE1.000_PCBE1.000").mkdir()
    (tmp_path / "007_Al1.000_PCB1.000_ALE1.000_PCBE1.000").mkdir()

    assert _next_run_sequence(tmp_path) == 8


def test_allocate_run_id_prepends_sequence_prefix(tmp_path: Path) -> None:
    (tmp_path / "001_Al0.800_PCB1.000_ALE1.000_PCBE1.000").mkdir()
    (tmp_path / "002_Al0.800_PCB1.000_ALE1.000_PCBE1.000").mkdir()

    run_id = _allocate_run_id(
        tmp_path,
        {
            "aluminum_density_factor": 0.8,
            "pcb_density_factor": 1.0,
            "aluminum_E_factor": 1.0,
            "pcb_E_factor": 1.0,
        },
    )

    assert run_id == "003_Al0.800_PCB1.000_ALE1.000_PCBE1.000"


def test_next_run_sequence_stops_at_999(tmp_path: Path) -> None:
    for index in range(1, 1000):
        (tmp_path / f"{index:03d}_Al0.800_PCB1.000_ALE1.000_PCBE1.000").mkdir()

    with pytest.raises(RuntimeError, match="001 to 999"):
        _next_run_sequence(tmp_path)
