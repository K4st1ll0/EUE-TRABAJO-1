import json

from src.report_generator import _manifest_to_index_entry, _prepare_mass_budget_rows


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
