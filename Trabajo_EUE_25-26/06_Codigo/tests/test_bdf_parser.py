from __future__ import annotations

from pathlib import Path

from src.fem.bdf_parser import inspect_bdf
from src.fem.model_checks import inspect_bdf_structure


def test_bulk_only_detection(tmp_path: Path) -> None:
    bdf_path = tmp_path / "bulk_only.bdf"
    bdf_path.write_text(
        "\n".join(
            [
                "MAT1,1,7.1E10,2.6E10,0.33",
                "PSHELL,10,1,0.001",
                "PBARL,20,1,,BAR,0.01,0.02",
                "PBUSH,30,K,1.0,2.0,3.0,4.0,5.0,6.0",
                "GRID,1,,0.,0.,0.",
                "GRID,2,,1.,0.,0.",
                "CBAR,100,20,1,2,0.,0.,1.",
                "RBE2,200,1,123456,2",
                "SPC1,1,123456,1,THRU,2",
            ]
        ),
        encoding="utf-8",
    )

    structure = inspect_bdf_structure(bdf_path)
    assert structure.is_bulk_only is True
    assert structure.should_wrap is True

    inspection = inspect_bdf(bdf_path)
    assert inspection.card_counts["MAT1"] == 1
    assert inspection.card_counts["PSHELL"] == 1
    assert inspection.card_counts["PBARL"] == 1
    assert inspection.card_counts["PBUSH"] == 1


def test_executable_deck_detection(tmp_path: Path) -> None:
    bdf_path = tmp_path / "deck.bdf"
    bdf_path.write_text(
        "\n".join(
            [
                "SOL 103",
                "CEND",
                "SUBCASE 1",
                "BEGIN BULK",
                "GRID,1,,0.,0.,0.",
                "ENDDATA",
            ]
        ),
        encoding="utf-8",
    )

    structure = inspect_bdf_structure(bdf_path)
    assert structure.is_executable_deck is True
    assert structure.should_wrap is False
