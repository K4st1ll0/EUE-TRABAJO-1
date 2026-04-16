from pathlib import Path

import pytest

from src.config import (
    MassCompensationThresholds,
    MassConfig,
    MassFamilyConfig,
    MassImputationBandConfig,
    MassImputationRuleConfig,
    MassSubsetConfig,
)
from src.mass_calculator import calculate_total_mass


def _grid_star(node_id: int, x: float, y: float, z: float) -> str:
    first = f"{'GRID*':<8}{node_id:<16}{'':<16}{x:<16.8f}{y:<16.8f}"
    second = f"{'*':<8}{z:<16.8f}"
    return f"{first}\n{second}"


def test_mass_summary_reports_subset_mapping_unmapped_and_unsupported_mass(tmp_path: Path) -> None:
    bdf_text = "\n".join(
        [
            "$ Elements and Element Properties for region : 09_A_t-boards-PSU",
            "PSHELL 25 2 0.001 2 2",
            "CQUAD4 1 25 1 2 3 4 0. 0.",
            "$ Elements and Element Properties for region : 07_t-bandejas",
            "PSHELL 24 1 0.001 1 1",
            "CQUAD4 2 24 5 6 7 8 0. 0.",
            "$ Elements and Element Properties for region : 05_t-tapa",
            "PSHELL 26 1 0.001 1 1",
            "CQUAD4 3 26 1 2 3 4 0. 0.",
            "$ Elements and Element Properties for region : concentrated",
            "CONM2 10 9 0 0.500000",
            "PMASS 30 0.250000",
            "CMASS1 20 30 12 0 13 0",
            "NSM 100 PSHELL 24 0.100000",
            "MAT1,1,1.0,,0.3,1.0",
            "MAT1,2,1.0,,0.3,2.0",
            _grid_star(1, 0.0, 0.0, 0.0),
            _grid_star(2, 1.0, 0.0, 0.0),
            _grid_star(3, 1.0, 1.0, 0.0),
            _grid_star(4, 0.0, 1.0, 0.0),
            _grid_star(5, 0.0, 0.0, 1.0),
            _grid_star(6, 1.0, 0.0, 1.0),
            _grid_star(7, 1.0, 1.0, 1.0),
            _grid_star(8, 0.0, 1.0, 1.0),
            _grid_star(9, 0.0, 0.0, 2.0),
            _grid_star(12, 0.0, 0.0, 3.0),
            _grid_star(13, 1.0, 0.0, 3.0),
            "ENDDATA",
        ]
    )
    bdf_path = tmp_path / "sample.bdf"
    bdf_path.write_text(bdf_text, encoding="utf-8")

    mass_config = MassConfig(
        target_kg=0.754,
        subsets=(
            MassSubsetConfig(
                label="H-PSU module",
                target_kg=0.002,
                regions=("09_A_t-boards-PSU",),
            ),
            MassSubsetConfig(
                label="Backplane modules",
                target_kg=0.001,
                regions=("07_t-bandejas",),
            ),
            MassSubsetConfig(
                label="Concentrated",
                target_kg=0.750,
                node_ids=(9, 12, 13),
            ),
        ),
        compensation_warning_thresholds=MassCompensationThresholds(
            total_relative_delta_within=0.01,
            subset_relative_delta_exceeds=0.05,
        ),
    )

    summary = calculate_total_mass(
        bdf_path=bdf_path,
        mass_config=mass_config,
    )

    assert summary.total_mass == pytest.approx(0.754)
    assert summary.relative_error == pytest.approx(0.0)
    assert summary.target_basis == "nominal"
    assert summary.baseline_status == "active"
    rows = {row.label: row for row in summary.subset_rows}
    assert rows["H-PSU module"].computed_mass == pytest.approx(0.002)
    assert rows["Backplane modules"].computed_mass == pytest.approx(0.001)
    assert rows["Concentrated"].computed_mass == pytest.approx(0.75)
    assert any(row.contribution_id == "CQUAD4:3" for row in summary.unmapped_rows)
    assert summary.ledgers.unmapped_mass == pytest.approx(0.001)
    assert any(row.source == "NSM" for row in summary.unsupported_extra_mass_rows)
    assert "unmapped_mass_detected" in summary.warnings


def test_mass_subset_overlap_raises_hard_error(tmp_path: Path) -> None:
    bdf_text = "\n".join(
        [
            "$ Elements and Element Properties for region : 09_A_t-boards-PSU",
            "PSHELL 25 2 0.001 2 2",
            "CQUAD4 1 25 1 2 3 4 0. 0.",
            "MAT1,2,1.0,,0.3,2.0",
            _grid_star(1, 0.0, 0.0, 0.0),
            _grid_star(2, 1.0, 0.0, 0.0),
            _grid_star(3, 1.0, 1.0, 0.0),
            _grid_star(4, 0.0, 1.0, 0.0),
            "ENDDATA",
        ]
    )
    bdf_path = tmp_path / "overlap.bdf"
    bdf_path.write_text(bdf_text, encoding="utf-8")

    mass_config = MassConfig(
        target_kg=0.002,
        subsets=(
            MassSubsetConfig(label="A", target_kg=0.001, regions=("09_A_t-boards-PSU",)),
            MassSubsetConfig(label="B", target_kg=0.001, property_ids=(25,)),
        ),
    )

    with pytest.raises(ValueError, match="matches more than one subset"):
        calculate_total_mass(bdf_path=bdf_path, mass_config=mass_config)


def test_basic_budget_basis_is_reported_as_historical(tmp_path: Path) -> None:
    bdf_text = "\n".join(
        [
            "$ Elements and Element Properties for region : 09_A_t-boards-PSU",
            "PSHELL 25 2 0.001 2 2",
            "CQUAD4 1 25 1 2 3 4 0. 0.",
            "MAT1,2,1.0,,0.3,2.0",
            _grid_star(1, 0.0, 0.0, 0.0),
            _grid_star(2, 1.0, 0.0, 0.0),
            _grid_star(3, 1.0, 1.0, 0.0),
            _grid_star(4, 0.0, 1.0, 0.0),
            "ENDDATA",
        ]
    )
    bdf_path = tmp_path / "basic_basis.bdf"
    bdf_path.write_text(bdf_text, encoding="utf-8")

    summary = calculate_total_mass(
        bdf_path=bdf_path,
        mass_config=MassConfig(
            target_kg=0.002,
            budget_basis="basic",
            subsets=(MassSubsetConfig(label="A", target_kg=0.002, regions=("09_A_t-boards-PSU",)),),
        ),
    )

    assert summary.target_basis == "basic"
    assert summary.baseline_status == "historical"


def test_mass_imputation_builds_budget_rows_from_families_and_rules(tmp_path: Path) -> None:
    bdf_text = "\n".join(
        [
            "$ Elements and Element Properties for region : 09_A_t-boards-PSU",
            "PSHELL 25 2 0.001 2 2",
            "CQUAD4 1 25 1 2 3 4 0. 0.",
            "$ Elements and Element Properties for region : 07_t-bandejas",
            "PSHELL 24 1 0.001 1 1",
            "CQUAD4 2 24 5 6 7 8 0. 0.",
            "$ Elements and Element Properties for region : 05_t-tapa",
            "PSHELL 26 1 0.001 1 1",
            "CQUAD4 3 26 9 10 11 12 0. 0.",
            "MAT1,1,1.0,,0.3,1.0",
            "MAT1,2,1.0,,0.3,2.0",
            _grid_star(1, 0.0, 0.0, 0.0),
            _grid_star(2, 1.0, 0.0, 0.0),
            _grid_star(3, 1.0, 1.0, 0.0),
            _grid_star(4, 0.0, 1.0, 0.0),
            _grid_star(5, 0.0, 0.0, 0.6),
            _grid_star(6, 1.0, 0.0, 0.6),
            _grid_star(7, 1.0, 1.0, 0.6),
            _grid_star(8, 0.0, 1.0, 0.6),
            _grid_star(9, 0.0, 0.0, 1.0),
            _grid_star(10, 1.0, 0.0, 1.0),
            _grid_star(11, 1.0, 1.0, 1.0),
            _grid_star(12, 0.0, 1.0, 1.0),
            "ENDDATA",
        ]
    )
    bdf_path = tmp_path / "imputation.bdf"
    bdf_path.write_text(bdf_text, encoding="utf-8")

    summary = calculate_total_mass(
        bdf_path=bdf_path,
        mass_config=MassConfig(
            target_kg=0.005,
            subsets=(
                MassSubsetConfig(label="H-PSU module", target_kg=0.001, regions=("09_A_t-boards-PSU",)),
                MassSubsetConfig(label="Backplane modules", target_kg=0.001, regions=("07_t-bandejas",)),
            ),
            families=(
                MassFamilyConfig(label="PSU board", regions=("09_A_t-boards-PSU",)),
                MassFamilyConfig(label="common trays", regions=("07_t-bandejas",)),
                MassFamilyConfig(label="top cover", regions=("05_t-tapa",)),
            ),
            imputation_rules=(
                MassImputationRuleConfig(
                    family_label="PSU board",
                    mode="direct",
                    target_label="H-PSU module",
                ),
                MassImputationRuleConfig(
                    family_label="common trays",
                    mode="z_band_split",
                    z_bands=(
                        MassImputationBandConfig(target_label="H-PSU module", z_max=0.5),
                        MassImputationBandConfig(target_label="Backplane modules", z_min=0.5),
                    ),
                ),
                MassImputationRuleConfig(
                    family_label="top cover",
                    mode="proportional",
                    weights={
                        "H-PSU module": 3.0,
                        "Backplane modules": 1.0,
                    },
                ),
            ),
        ),
    )

    family_rows = {row.label: row for row in summary.family_rows}
    assert family_rows["PSU board"].computed_mass == pytest.approx(0.002)
    assert family_rows["common trays"].computed_mass == pytest.approx(0.001)
    assert family_rows["top cover"].computed_mass == pytest.approx(0.001)
    assert family_rows["unmapped residual"].computed_mass == pytest.approx(0.0)

    budget_rows = {row.label: row for row in summary.budget_imputation_rows}
    assert budget_rows["H-PSU module"].imputed_mass == pytest.approx(0.00275)
    assert budget_rows["Backplane modules"].imputed_mass == pytest.approx(0.00125)
    assert summary.imputation_residual_mass == pytest.approx(0.0)
    assert summary.budget_imputation_warnings == ()
    assert summary.budget_imputation_notes == (
        "z_band_split is an approximate geometric rule based on z-bands.",
        "rig base -> Backplane modules is a working functional hypothesis, not a geometrically demonstrated truth.",
    )


def test_rig_base_imputation_is_functional_only_and_preserves_subset_mapping(tmp_path: Path) -> None:
    bdf_text = "\n".join(
        [
            "$ Elements and Element Properties for region : 09_A_t-boards-PSU",
            "PSHELL 25 1 0.002 1 1",
            "CQUAD4 1 25 1 2 3 4 0. 0.",
            "$ Elements and Element Properties for region : 06_t-rig-base",
            "PSHELL 26 1 0.001 1 1",
            "CQUAD4 2 26 5 6 7 8 0. 0.",
            "MAT1,1,1.0,,0.3,1.0",
            _grid_star(1, 0.0, 0.0, 0.0),
            _grid_star(2, 1.0, 0.0, 0.0),
            _grid_star(3, 1.0, 1.0, 0.0),
            _grid_star(4, 0.0, 1.0, 0.0),
            _grid_star(5, 0.0, 0.0, 1.0),
            _grid_star(6, 1.0, 0.0, 1.0),
            _grid_star(7, 1.0, 1.0, 1.0),
            _grid_star(8, 0.0, 1.0, 1.0),
            "ENDDATA",
        ]
    )
    bdf_path = tmp_path / "rig_base_imputation.bdf"
    bdf_path.write_text(bdf_text, encoding="utf-8")

    base_config = MassConfig(
        target_kg=0.003,
        subsets=(
            MassSubsetConfig(
                label="H-PSU module",
                target_kg=0.003,
                regions=("09_A_t-boards-PSU", "06_t-rig-base"),
            ),
            MassSubsetConfig(
                label="Backplane modules",
                target_kg=0.001,
                regions=("07_t-bandejas",),
            ),
        ),
        families=(
            MassFamilyConfig(label="PSU board", regions=("09_A_t-boards-PSU",)),
        ),
        imputation_rules=(
            MassImputationRuleConfig(
                family_label="PSU board",
                mode="direct",
                target_label="H-PSU module",
            ),
        ),
    )

    before = calculate_total_mass(bdf_path=bdf_path, mass_config=base_config)
    after = calculate_total_mass(
        bdf_path=bdf_path,
        mass_config=MassConfig(
            target_kg=base_config.target_kg,
            subsets=base_config.subsets,
            families=base_config.families
            + (MassFamilyConfig(label="rig base", regions=("06_t-rig-base",)),),
            imputation_rules=base_config.imputation_rules
            + (
                MassImputationRuleConfig(
                    family_label="rig base",
                    mode="direct",
                    target_label="Backplane modules",
                ),
            ),
        ),
    )

    assert before.subset_rows == after.subset_rows
    assert before.warnings == ()
    assert after.warnings == ()
    assert before.budget_imputation_warnings == ("budget_imputation_residual_detected",)
    assert after.budget_imputation_warnings == ()
    assert after.imputation_residual_mass == pytest.approx(0.0)
    assert after.budget_imputation_notes == (
        "z_band_split is an approximate geometric rule based on z-bands.",
        "rig base -> Backplane modules is a working functional hypothesis, not a geometrically demonstrated truth.",
    )

    before_budget_rows = {row.label: row for row in before.budget_imputation_rows}
    after_budget_rows = {row.label: row for row in after.budget_imputation_rows}
    assert before_budget_rows["Backplane modules"].imputed_mass == pytest.approx(0.0)
    assert after_budget_rows["Backplane modules"].imputed_mass == pytest.approx(0.001)
