import json
from pathlib import Path

from src.report_generator import _manifest_to_index_entry, _prepare_mass_budget_rows, generate_run_report


def test_manifest_to_index_entry_marks_basic_runs_as_historical() -> None:
    entry = _manifest_to_index_entry(
        {
            "run_id": "001_basic",
            "mass": {
                "budget_comparison": {
                    "basis": "basic",
                }
            },
            "modal_metrics": {},
        }
    )

    assert entry["mass_target_basis"] == "basic"
    assert entry["mass_baseline_status"] == "historical"


def test_manifest_to_index_entry_keeps_nominal_runs_active() -> None:
    entry = _manifest_to_index_entry(
        {
            "run_id": "002_nominal",
            "mass": {
                "target_basis": "nominal",
                "baseline_status": "active",
            },
            "modal_metrics": {},
        }
    )

    assert entry["mass_target_basis"] == "nominal"
    assert entry["mass_baseline_status"] == "active"


def test_manifest_to_index_entry_can_infer_basis_from_config_snapshot(tmp_path) -> None:
    snapshot_dir = tmp_path / "config_snapshot"
    snapshot_dir.mkdir()
    (snapshot_dir / "resolved_snapshot.json").write_text(
        json.dumps({"project": {"mass": {"budget_basis": "nominal"}}}),
        encoding="utf-8",
    )

    entry = _manifest_to_index_entry(
        {
            "run_id": "003_snapshot",
            "config_snapshot_dir": str(snapshot_dir),
            "mass": {},
            "modal_metrics": {},
        }
    )

    assert entry["mass_target_basis"] == "nominal"
    assert entry["mass_baseline_status"] == "active"


def test_manifest_to_index_entry_falls_back_to_mass_relative_error_for_ranking() -> None:
    entry = _manifest_to_index_entry(
        {
            "run_id": "004_mass_fit",
            "mass": {
                "relative_error": 0.00123,
                "target_basis": "nominal",
            },
            "modal_metrics": {},
        }
    )

    assert entry["ranking_key"] == [0.00123]


def test_prepare_mass_budget_rows_derives_rows_from_subset_rows_and_total() -> None:
    rows, notes = _prepare_mass_budget_rows(
        {
            "mass": {
                "target_mass": 10.0,
                "total_mass": 9.5,
                "subset_rows": [
                    {
                        "label": "A",
                        "target_mass": 4.0,
                        "computed_mass": 3.0,
                        "delta_mass": -1.0,
                        "relative_delta": -0.25,
                    }
                ],
            }
        }
    )

    assert notes == []
    assert rows[0]["label"] == "A"
    assert rows[-1]["label"] == "Total mass"
    assert rows[-1]["delta_mass"] == -0.5


def test_prepare_mass_budget_rows_preserves_legacy_budget_comparison_rows() -> None:
    rows, notes = _prepare_mass_budget_rows(
        {
            "mass": {
                "budget_comparison": {
                    "rows": [
                        {
                            "category": "H-PSU module",
                            "budget_mass": 1.0,
                            "model_mass": 0.9,
                            "delta_mass": -0.1,
                            "relative_delta": -0.1,
                        }
                    ],
                    "notes": ["legacy note"],
                }
            }
        }
    )

    assert rows == [
        {
            "label": "H-PSU module",
            "target_mass": 1.0,
            "computed_mass": 0.9,
            "delta_mass": -0.1,
            "relative_delta": -0.1,
        }
    ]
    assert notes == ["legacy note"]


def test_generate_run_report_separates_legacy_and_functional_mass_layers(tmp_path: Path) -> None:
    output_path = tmp_path / "report.html"
    template_dir = Path(__file__).resolve().parents[1] / "src" / "templates"
    run_payload = {
        "project_name": "MSC Nastran FEM Automation",
        "run_id": "021_test",
        "command": "full-run",
        "sweep_preset": None,
        "status": "success",
        "created_at": "2026-04-16T15:00:00+02:00",
        "accepted": False,
        "score_robust_display": "n/a",
        "score_legacy_display": "n/a",
        "mean_mac_display": "n/a",
        "paired_valid_mode_count": 0,
        "valid_model_mode_count": 0,
        "valid_reference_mode_count": 0,
        "reject_reasons": [],
        "parameter_rows": [],
        "source_bdf": "source.bdf",
        "generated_bdf": "generated.bdf",
        "reference_frequencies_f06": "ref_freq.f06",
        "reference_vectors_f06": "ref_vec.f06",
        "config_snapshot_dir": "config_snapshot",
        "warnings": ["reference_parser_warning"],
        "errors": [],
        "modal_metrics": {
            "selected_ranking_mode": "hierarchical",
            "mean_mac": None,
            "worst_mac": None,
            "mean_relative_frequency_error": None,
            "max_relative_frequency_error": None,
            "bad_pair_count": 0,
            "warning_pair_count": 0,
            "mac_matrix": [],
            "frequency_error_matrix": [],
            "allowed_pairs_matrix": [],
        },
        "reference": {
            "requested_modes": [],
            "frequencies_hz": {},
            "valid_modes": [],
        },
        "model": {
            "frequencies_hz": {},
            "valid_modes": [],
        },
        "mass": {
            "total_mass": 3.0,
            "relative_error": 0.0,
            "target_mass": 3.0,
            "target_basis": "nominal",
            "baseline_status": "active",
            "ledgers": {
                "mapped_mass": 2.5,
                "unmapped_mass": 0.5,
                "unsupported_extra_mass": 0.0,
            },
            "warnings": ["unmapped_mass_detected"],
            "subset_rows": [
                {
                    "label": "H-PSU module",
                    "target_mass": 1.0,
                    "computed_mass": 1.0,
                    "delta_mass": 0.0,
                    "relative_delta": 0.0,
                }
            ],
            "budget_imputation_rows": [
                {
                    "label": "Backplane modules",
                    "target_mass": 1.0,
                    "imputed_mass": 0.9,
                    "delta_mass": -0.1,
                    "relative_delta": -0.1,
                }
            ],
            "budget_imputation_warnings": ["functional_warning"],
            "budget_imputation_notes": [
                "z_band_split is an approximate geometric rule based on z-bands.",
                "rig base -> Backplane modules is a working functional hypothesis, not a geometrically demonstrated truth.",
            ],
            "imputation_residual_mass": 0.0,
            "family_rows": [],
            "unmapped_rows": [],
            "unsupported_extra_mass_rows": [],
        },
    }

    generate_run_report(
        run_payload=run_payload,
        history=[],
        template_dir=template_dir,
        output_path=output_path,
    )

    html = output_path.read_text(encoding="utf-8")
    assert "Current mass assessment" in html
    assert "Residual after imputation" in html
    assert "Functional warnings" in html
    assert "Functional Mass Budget Imputation" in html
    assert "Legacy Direct Mapping Diagnostic" in html
    assert "Legacy direct FEM-&gt;subset mapping. Diagnostic only. Do not use for mass-fit decisions." in html
    assert "Open legacy direct FEM-&gt;subset mapping diagnostic" in html
    assert "Legacy mapping warnings" in html
    assert "Warnings funcionales" in html
    assert "Notas funcionales" in html
    assert "z_band_split is an approximate geometric rule based on z-bands." in html
    assert "rig base -> Backplane modules is a working functional hypothesis, not a geometrically demonstrated truth." in html
    assert "Relaci&oacute;n con el Mass Budget: Mapping Directo / Legacy" not in html
    assert html.index("Functional Mass Budget Imputation") < html.index("Legacy Direct Mapping Diagnostic")
    assert html.count("reference_parser_warning") == 1
    assert html.count("unmapped_mass_detected") == 1
    assert html.count("functional_warning") == 2
