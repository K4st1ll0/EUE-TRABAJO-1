from pathlib import Path
import json

import pytest

from src.config import load_config_bundle
from src.ia_report import IA_DETAILED_FILENAME, IA_OVERVIEW_FILENAME, IA_REPORT_FILENAME, refresh_ia_report
from src.main import main


def _grid_star(node_id: int, x: float, y: float, z: float) -> str:
    first = f"{'GRID*':<8}{node_id:<16}{'':<16}{x:<16.8f}{y:<16.8f}"
    second = f"{'*':<8}{z:<16.8f}"
    return f"{first}\n{second}"


def _write_project_files(project_root: Path) -> Path:
    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True)
    (project_root / "model").mkdir()
    (project_root / "reference").mkdir()
    (project_root / "tools").mkdir()
    (project_root / "runs").mkdir()
    (project_root / "reports").mkdir()

    (project_root / "README.md").write_text(
        "# IA Test Project\n\nProyecto de prueba para IA.\n\n## Instalacion\n\nNo aplica.\n",
        encoding="utf-8",
    )
    (project_root / "model" / "SFEM_EUE.bdf").write_text(
        "\n".join(
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
        ),
        encoding="utf-8",
    )
    (project_root / "reference" / "ref_freq.f06").write_text("freq", encoding="utf-8")
    (project_root / "reference" / "ref_vec.f06").write_text("vec", encoding="utf-8")
    (project_root / "tools" / "nastran.exe").write_text("", encoding="utf-8")
    (config_dir / "project_config.yaml").write_text(
        "\n".join(
            [
                "project:",
                "  name: IA Test",
                "paths:",
                "  bdf_input: model/SFEM_EUE.bdf",
                "  reference_frequencies_f06: reference/ref_freq.f06",
                "  reference_vectors_f06: reference/ref_vec.f06",
                "  runs_dir: runs",
                "  reports_dir: reports",
                "nastran:",
                "  executable: tools/nastran.exe",
                "  arguments: []",
                "analysis:",
                "  num_modes: 1",
                "  dry_run: false",
                "pairing:",
                "  freq_gate: 0.2",
                "  alpha_freq: 0.15",
                "  huge_penalty: 1000000.0",
                "  bad_pair_mac_threshold: 0.1",
                "  warning_pair_mac_threshold: 0.3",
                "acceptance:",
                "  max_bad_pairs_allowed: 1",
                "  reject_if_any_pair_outside_gate: true",
                "  min_valid_modes: 1",
                "scoring:",
                "  mode: weighted_error",
                "  weights:",
                "    mac: 1.0",
                "    frequency: 0.25",
                "    mass: 0.10",
                "ranking:",
                "  mode: hierarchical",
                "  robust_weights:",
                "    mac: 0.05",
                "    bad_pair_fraction: 0.20",
                "mass:",
                "  target_kg: 0.003",
                "  budget_basis: nominal",
                "  subsets:",
                "    - label: H-PSU module",
                "      target_kg: 0.003",
                "      regions: [09_A_t-boards-PSU]",
                "    - label: Backplane modules",
                "      target_kg: 0.001",
                "      regions: [07_t-bandejas]",
                "  families:",
                "    - label: rig base",
                "      regions: [06_t-rig-base]",
                "  imputation_rules:",
                "    - family: rig base",
                "      mode: direct",
                "      target: Backplane modules",
                "mode_family_labels:",
                "  reference: {}",
                "  model: {}",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "sensors.yaml").write_text(
        "reference_order_source: test\nsensors:\n  - name: s1\n    label: Sensor 1\n    model_node_id: 1\n    model_coordinates: [0.0, 0.0, 0.0]\n    reference_point_id: 1\n",
        encoding="utf-8",
    )
    (config_dir / "parameters.yaml").write_text(
        "material_groups:\n  aluminum: [1]\nparameters:\n  aluminum_density_factor:\n    group: aluminum\n    target: rho\n    mode: fixed\n    value: 1.0\n",
        encoding="utf-8",
    )
    return config_dir


def _write_run_artifacts(project_root: Path, run_id: str) -> Path:
    run_dir = project_root / "runs" / run_id
    run_dir.mkdir(parents=True)
    source_bdf = project_root / "model" / "SFEM_EUE.bdf"
    generated_bdf = run_dir / "SFEM_EUE.bdf"
    generated_bdf.write_text(source_bdf.read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "sfem_eue.f06").write_text("result f06", encoding="utf-8")
    (run_dir / "report.html").write_text("<html></html>", encoding="utf-8")
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")

    mass_payload = {
        "total_mass": 0.003,
        "relative_error": 0.0,
        "target_mass": 0.003,
        "target_basis": "nominal",
        "baseline_status": "active",
        "subset_rows": [{"label": "H-PSU module", "target_mass": 0.003, "computed_mass": 0.003, "delta_mass": 0.0, "relative_delta": 0.0}],
        "budget_imputation_rows": [{"label": "Backplane modules", "target_mass": 0.001, "imputed_mass": 0.001, "delta_mass": 0.0, "relative_delta": 0.0}],
        "budget_imputation_warnings": ["functional_warning"],
        "budget_imputation_notes": [
            "z_band_split is an approximate geometric rule based on z-bands.",
            "rig base -> Backplane modules is a working functional hypothesis, not a geometrically demonstrated truth.",
        ],
        "imputation_residual_mass": 0.0,
        "ledgers": {"mapped_mass": 0.003, "unmapped_mass": 0.0, "unsupported_extra_mass": 0.0},
        "warnings": ["unmapped_mass_detected"],
    }
    (run_dir / "mass_summary.json").write_text(json.dumps(mass_payload, indent=2), encoding="utf-8")

    manifest = {
        "run_id": run_id,
        "command": "full-run",
        "created_at": "2026-04-16T15:00:00+02:00",
        "status": "success",
        "source_bdf": str(source_bdf),
        "generated_bdf": str(generated_bdf),
        "reference_frequencies_f06": str(project_root / "reference" / "ref_freq.f06"),
        "reference_vectors_f06": str(project_root / "reference" / "ref_vec.f06"),
        "config_snapshot_dir": str(run_dir / "config_snapshot"),
        "parameter_values": {"aluminum_density_factor": 1.0},
        "acceptance_config": {"min_valid_modes": 1, "reject_if_any_pair_outside_gate": True},
        "mass": mass_payload,
        "reference": {"frequencies_hz": {1: 100.0}},
        "model": {"frequencies_hz": {2: 99.0}},
        "modal_metrics": {
            "selected_ranking_mode": "hierarchical",
            "pairing": [
                {
                    "reference_mode": 1,
                    "model_mode": 2,
                    "relative_frequency_error": 0.01,
                    "mac": 0.9,
                    "outside_gate": False,
                    "bad_pair": False,
                    "warning_pair": False,
                    "suspicious": False,
                }
            ],
            "worst_mac": 0.9,
            "max_relative_frequency_error": 0.01,
            "bad_pair_count": 0,
            "warning_pair_count": 0,
            "mean_mac": 0.9,
            "mean_relative_frequency_error": 0.01,
        },
        "accepted": False,
        "reject_reasons": ["paired_valid_mode_count_below_min"],
        "valid_reference_mode_count": 1,
        "valid_model_mode_count": 1,
        "paired_valid_mode_count": 1,
        "nastran": {"outputs": {"f06": str(run_dir / "sfem_eue.f06")}},
        "report_path": str(run_dir / "report.html"),
        "score_robust": 0.12,
        "score_legacy": 0.34,
    }
    (run_dir / "config_snapshot").mkdir()
    (run_dir / "config_snapshot" / "resolved_snapshot.json").write_text("{}", encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (project_root / "reports" / "runs_index.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-04-16T18:00:00+02:00",
                "active_mass_target_basis": "nominal",
                "runs": [
                    {
                        "run_id": run_id,
                        "created_at": "2026-04-16T15:00:00+02:00",
                        "command": "full-run",
                        "status": "success",
                        "selected_ranking_mode": "hierarchical",
                        "ranking_key": [0.12],
                        "score_robust": 0.12,
                        "score_legacy": 0.34,
                        "mean_mac": 0.9,
                        "mean_relative_frequency_error": 0.01,
                        "mass_target_basis": "nominal",
                        "mass_baseline_status": "active",
                        "report_path": str(run_dir / "report.html"),
                        "parameter_values": {"aluminum_density_factor": 1.0},
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (project_root / "reports" / "mass_budget_imputation_audit_021.json").write_text(
        json.dumps(
            {
                "audit_run_label": "021",
                "run_id": run_id,
                "generated_bdf": str(generated_bdf),
                "mass_source_of_truth": str(run_dir / "mass_summary.json"),
                "imputation_residual_mass": 0.0,
                "budget_imputation_warnings": ["functional_warning"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return run_dir


def test_refresh_ia_report_writes_overview_detailed_and_alias(tmp_path: Path) -> None:
    config_dir = _write_project_files(tmp_path)
    _write_run_artifacts(tmp_path, "021_case")
    bundle = load_config_bundle(config_dir)

    alias_path, overview = refresh_ia_report(
        bundle,
        trigger_command="mass-audit",
        affected_run_ids=["021_case"],
        artifacts_written=[str(tmp_path / "reports" / "mass_budget_imputation_audit_021.json")],
    )

    overview_path = tmp_path / "reports" / IA_OVERVIEW_FILENAME
    detailed_path = tmp_path / "reports" / IA_DETAILED_FILENAME
    assert alias_path == tmp_path / "reports" / IA_REPORT_FILENAME
    assert overview_path.exists()
    assert detailed_path.exists()
    assert alias_path.read_text(encoding="utf-8") == overview_path.read_text(encoding="utf-8")

    detailed = json.loads(detailed_path.read_text(encoding="utf-8"))
    assert overview["view"] == "overview"
    assert detailed["view"] == "detailed"
    assert overview["program_state"]["best_full_run_id"] == "021_case"
    assert overview["program_state"]["best_modal_run_id"] == "021_case"
    assert overview["program_state"]["best_mass_fit_run_id"] is None
    assert "current_focus" in overview
    assert isinstance(overview["next_actions"], list)
    assert "tests_summary" in overview
    assert "reduction_state" in overview
    assert isinstance(overview["equivalence_groups"], list)

    detailed_run = detailed["runs"][0]
    assert detailed_run["acceptance"]["accepted"] is False
    assert detailed_run["acceptance"]["acceptance_policy_snapshot"]["min_valid_modes"] == 1
    assert detailed_run["modal"]["paired_reference_modes"] == [1]
    assert detailed_run["modal"]["paired_model_modes"] == [2]
    assert detailed_run["modal"]["paired_mac_by_mode"]["1"] == pytest.approx(0.9)
    assert detailed_run["mass"]["legacy_direct_mapping"]["warnings"] == ["unmapped_mass_detected"]
    assert detailed_run["mass"]["functional_budget_imputation"]["warnings"] == ["functional_warning"]
    assert detailed_run["mass"]["functional_budget_imputation"]["imputation_residual_mass_kg"] == pytest.approx(0.0)
    assert detailed_run["provenance"]["generated_bdf_sha256"] is not None
    assert detailed_run["provenance"]["result_f06_sha256"] is not None
    assert overview["activity_log"][-1]["trigger_command"] == "mass-audit"


def test_main_mass_audit_refreshes_all_ia_outputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_dir = _write_project_files(tmp_path)
    _write_run_artifacts(tmp_path, "021_case")

    exit_code = main(["mass-audit", "--config-dir", str(config_dir), "--run-id", "021"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "IA report:" in output
    overview = json.loads((tmp_path / "reports" / IA_OVERVIEW_FILENAME).read_text(encoding="utf-8"))
    detailed = json.loads((tmp_path / "reports" / IA_DETAILED_FILENAME).read_text(encoding="utf-8"))
    mass_summary = json.loads((tmp_path / "runs" / "021_case" / "mass_summary.json").read_text(encoding="utf-8"))
    assert overview["runs"][0]["run_id"] == "021_case"
    assert detailed["runs"][0]["mass"]["functional_budget_imputation"]["imputation_residual_mass_kg"] == pytest.approx(
        mass_summary["imputation_residual_mass"]
    )
