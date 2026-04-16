from pathlib import Path
import json

import pytest

from src.config import load_config_bundle
from src.mass_audit import MassAuditError, _resolve_run_dir, run_mass_audit


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
    (project_root / "model" / "SFEM_EUE.bdf").write_text(bdf_text, encoding="utf-8")
    (project_root / "reference" / "ref_freq.f06").write_text("", encoding="utf-8")
    (project_root / "reference" / "ref_vec.f06").write_text("", encoding="utf-8")
    (project_root / "tools" / "nastran.exe").write_text("", encoding="utf-8")

    (config_dir / "project_config.yaml").write_text(
        "\n".join(
            [
                "project:",
                "  name: Audit Test",
                "",
                "paths:",
                "  bdf_input: model/SFEM_EUE.bdf",
                "  reference_frequencies_f06: reference/ref_freq.f06",
                "  reference_vectors_f06: reference/ref_vec.f06",
                "  runs_dir: runs",
                "  reports_dir: reports",
                "",
                "nastran:",
                "  executable: tools/nastran.exe",
                "  arguments: []",
                "",
                "analysis:",
                "  num_modes: 1",
                "  dry_run: false",
                "",
                "pairing:",
                "  freq_gate: 0.2",
                "  alpha_freq: 0.15",
                "  huge_penalty: 1000000.0",
                "  bad_pair_mac_threshold: 0.1",
                "  warning_pair_mac_threshold: 0.3",
                "",
                "acceptance:",
                "  max_bad_pairs_allowed: 1",
                "  reject_if_any_pair_outside_gate: true",
                "  min_valid_modes: 1",
                "",
                "scoring:",
                "  mode: weighted_error",
                "  weights:",
                "    mac: 1.0",
                "    frequency: 0.25",
                "    mass: 0.10",
                "",
                "ranking:",
                "  mode: hierarchical",
                "  robust_weights:",
                "    mac: 0.05",
                "    bad_pair_fraction: 0.20",
                "",
                "mass:",
                "  target_kg: 0.003",
                "  budget_basis: nominal",
                "  compensation_warning_thresholds:",
                "    total_relative_delta_within: 0.01",
                "    subset_relative_delta_exceeds: 0.05",
                "  subsets:",
                "    - label: H-PSU module",
                "      target_kg: 0.003",
                "      regions: [09_A_t-boards-PSU, 06_t-rig-base]",
                "    - label: Backplane modules",
                "      target_kg: 0.001",
                "      regions: [07_t-bandejas]",
                "  families:",
                "    - label: PSU board",
                "      regions: [09_A_t-boards-PSU]",
                "    - label: rig base",
                "      regions: [06_t-rig-base]",
                "  imputation_rules:",
                "    - family: PSU board",
                "      mode: direct",
                "      target: H-PSU module",
                "    - family: rig base",
                "      mode: direct",
                "      target: Backplane modules",
                "",
                "mode_family_labels:",
                "  reference: {}",
                "  model: {}",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "sensors.yaml").write_text(
        "\n".join(
            [
                "reference_order_source: test",
                "sensors:",
                "  - name: s1",
                "    label: Sensor 1",
                "    model_node_id: 1",
                "    model_coordinates: [0.0, 0.0, 0.0]",
                "    reference_point_id: 1",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "parameters.yaml").write_text(
        "\n".join(
            [
                "material_groups:",
                "  aluminum: [1]",
                "parameters:",
                "  aluminum_density_factor:",
                "    group: aluminum",
                "    target: rho",
                "    mode: fixed",
                "    value: 1.0",
            ]
        ),
        encoding="utf-8",
    )
    return config_dir


def _write_run_payload(run_dir: Path) -> None:
    payload = {
        "project_name": "Audit Test",
        "run_id": run_dir.name,
        "command": "full-run",
        "created_at": "2026-04-16T15:00:00+02:00",
        "status": "success",
        "dry_run": False,
        "sweep_preset": None,
        "source_bdf": "source_model.bdf",
        "generated_bdf": str(run_dir / "SFEM_EUE.bdf"),
        "reference_frequencies_f06": "reference/ref_freq.f06",
        "reference_vectors_f06": "reference/ref_vec.f06",
        "config_snapshot_dir": str(run_dir / "config_snapshot"),
        "parameter_values": {"aluminum_density_factor": 1.0},
        "parameter_rows": [],
        "pairing_config": {"freq_gate": 0.2},
        "acceptance_config": {},
        "ranking_config": {},
        "material_groups": {"aluminum": [1]},
        "modifications": [],
        "mass": {
            "total_mass": 0.003,
            "relative_error": 0.0,
            "target_mass": 0.003,
            "target_basis": "nominal",
            "baseline_status": "active",
            "method": "manual",
            "subset_rows": [
                {
                    "label": "H-PSU module",
                    "target_mass": 0.003,
                    "computed_mass": 0.003,
                    "delta_mass": 0.0,
                    "relative_delta": 0.0,
                },
                {
                    "label": "Backplane modules",
                    "target_mass": 0.001,
                    "computed_mass": 0.0,
                    "delta_mass": -0.001,
                    "relative_delta": -1.0,
                },
            ],
            "budget_imputation_rows": [
                {
                    "label": "H-PSU module",
                    "target_mass": 0.003,
                    "imputed_mass": 0.002,
                    "delta_mass": -0.001,
                    "relative_delta": -0.3333333333333333,
                },
                {
                    "label": "Backplane modules",
                    "target_mass": 0.001,
                    "imputed_mass": 0.0,
                    "delta_mass": -0.001,
                    "relative_delta": -1.0,
                },
            ],
            "family_budget_allocations": [],
            "imputation_residual_mass": 0.001,
            "unsupported_extra_mass_rows": [],
            "ledgers": {
                "mapped_mass": 0.003,
                "unmapped_mass": 0.0,
                "unsupported_extra_mass": 0.0,
            },
            "warnings": ["budget_imputation_residual_detected"],
        },
        "mass_target_basis": "nominal",
        "mass_baseline_status": "active",
        "nastran": {},
        "reference": {
            "requested_modes": [],
            "frequencies_hz": {},
            "valid_modes": [],
        },
        "model": {
            "frequencies_hz": {},
            "valid_modes": [],
        },
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
        "accepted": False,
        "reject_reasons": [],
        "valid_reference_mode_count": 0,
        "valid_model_mode_count": 0,
        "paired_valid_mode_count": 0,
        "warnings": ["budget_imputation_residual_detected", "reference_parser_warning"],
        "errors": [],
        "score_robust_display": "n/a",
        "score_legacy_display": "n/a",
        "mean_mac_display": "n/a",
        "report_path": str(run_dir / "report.html"),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_mass_audit_rebuilds_functional_audit_from_live_config(tmp_path: Path) -> None:
    config_dir = _write_project_files(tmp_path)
    run_dir = tmp_path / "runs" / "021_case"
    run_dir.mkdir(parents=True)
    run_bdf = (tmp_path / "model" / "SFEM_EUE.bdf").read_text(encoding="utf-8")
    (run_dir / "SFEM_EUE.bdf").write_text(run_bdf, encoding="utf-8")
    _write_run_payload(run_dir)

    (tmp_path / "reports" / "mass_budget_imputation_audit_021.json").write_text(
        json.dumps(
            {
                "subset_rows": [
                    {
                        "label": "H-PSU module",
                        "target_mass": 0.003,
                        "computed_mass": 0.003,
                        "delta_mass": 0.0,
                        "relative_delta": 0.0,
                    },
                    {
                        "label": "Backplane modules",
                        "target_mass": 0.001,
                        "computed_mass": 0.0,
                        "delta_mass": -0.001,
                        "relative_delta": -1.0,
                    },
                ],
                "budget_imputation_rows": [
                    {
                        "label": "H-PSU module",
                        "target_mass": 0.003,
                        "imputed_mass": 0.002,
                        "delta_mass": -0.001,
                        "relative_delta": -0.3333333333333333,
                    },
                    {
                        "label": "Backplane modules",
                        "target_mass": 0.001,
                        "imputed_mass": 0.0,
                        "delta_mass": -0.001,
                        "relative_delta": -1.0,
                    },
                ],
                "imputation_residual_mass": 0.001,
                "warnings": ["budget_imputation_residual_detected"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    bundle = load_config_bundle(config_dir)
    json_path, md_path, payload = run_mass_audit(bundle, "021")

    assert json_path.exists()
    assert md_path.exists()
    assert payload["run_id"] == "021_case"
    assert payload["config_source"] == "live project_config.yaml"
    assert payload["mass_source_of_truth"].endswith("mass_summary.json")
    assert payload["subset_rows_invariant_vs_before"] is True
    assert payload["imputation_residual_mass"] == pytest.approx(0.0)
    assert payload["budget_imputation_warnings"] == []
    assert payload["comparison"]["before"]["budget_imputation_warnings"] == [
        "budget_imputation_residual_detected"
    ]
    assert payload["comparison"]["delta"]["budget_imputation_warnings"]["removed"] == [
        "budget_imputation_residual_detected"
    ]
    budget_rows = {row["label"]: row for row in payload["budget_imputation_rows"]}
    assert budget_rows["Backplane modules"]["imputed_mass"] == pytest.approx(0.001)

    markdown = md_path.read_text(encoding="utf-8")
    assert "Config source: `live project_config.yaml`" in markdown
    assert "Mass source of truth: `" in markdown
    assert "subset_rows invariant vs previous functional audit: `True`" in markdown

    mass_summary = json.loads((run_dir / "mass_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    report_html = (run_dir / "report.html").read_text(encoding="utf-8")

    for payload_with_mass in (manifest, metrics):
        assert payload_with_mass["warnings"] == ["reference_parser_warning"]
        assert payload_with_mass["mass"]["imputation_residual_mass"] == pytest.approx(0.0)
        assert payload_with_mass["mass"]["budget_imputation_warnings"] == []
        assert payload_with_mass["mass"]["budget_imputation_notes"] == [
            "z_band_split is an approximate geometric rule based on z-bands.",
            "rig base -> Backplane modules is a working functional hypothesis, not a geometrically demonstrated truth.",
        ]
    assert mass_summary["imputation_residual_mass"] == pytest.approx(0.0)
    assert "Residual after imputation" in report_html
    assert "rig base -&gt; Backplane modules" not in report_html
    assert "rig base -> Backplane modules is a working functional hypothesis, not a geometrically demonstrated truth." in report_html


def test_resolve_run_dir_accepts_exact_and_prefixed_ids(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    exact = runs_dir / "021_case"
    exact.mkdir()

    assert _resolve_run_dir(runs_dir, "021_case") == exact
    assert _resolve_run_dir(runs_dir, "021") == exact


def test_resolve_run_dir_rejects_ambiguous_or_missing_ids(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    (runs_dir / "021_case_a").mkdir()
    (runs_dir / "021_case_b").mkdir()

    with pytest.raises(MassAuditError, match="ambiguous"):
        _resolve_run_dir(runs_dir, "021")
    with pytest.raises(MassAuditError, match="No run folder matches"):
        _resolve_run_dir(runs_dir, "099")
