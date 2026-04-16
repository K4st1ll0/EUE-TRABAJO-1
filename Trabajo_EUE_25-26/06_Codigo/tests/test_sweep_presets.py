from pathlib import Path
import json

from src.config import load_config_bundle
from src.sweep_engine import _expand_mass_fit_cases, _expand_sweep_cases


def _write_config_bundle(tmp_path: Path) -> Path:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    input_bdf = tmp_path / "model.bdf"
    input_bdf.write_text("BEGIN BULK\nENDDATA\n", encoding="utf-8")
    ref_freq = tmp_path / "ref_freq.f06"
    ref_freq.write_text("R E A L   E I G E N V A L U E S\n", encoding="utf-8")
    ref_vec = tmp_path / "ref_vec.f06"
    ref_vec.write_text("R E A L   E I G E N V E C T O R   N O .          1\n", encoding="utf-8")
    nastran = tmp_path / "nastran.exe"
    nastran.write_text("stub", encoding="utf-8")

    (configs_dir / "project_config.yaml").write_text(
        f"""
project:
  name: Test Project
paths:
  bdf_input: {input_bdf.name}
  reference_frequencies_f06: {ref_freq.name}
  reference_vectors_f06: {ref_vec.name}
  runs_dir: runs
  reports_dir: reports
nastran:
  executable: {nastran.name}
  arguments: []
analysis:
  num_modes: 10
  dry_run: true
mass:
  target_kg: 1.0
""".strip(),
        encoding="utf-8",
    )
    (configs_dir / "sensors.yaml").write_text(
        """
reference_order_source: unit-test
sensors:
  - name: s1
    label: S1
    model_node_id: 486
    model_coordinates: [0.0, 0.0, 0.1]
    reference_point_id: 10
  - name: s2
    label: S2
    model_node_id: 452
    model_coordinates: [0.0, 0.0, 0.1]
    reference_point_id: 11
  - name: s3
    label: S3
    model_node_id: 2040
    model_coordinates: [0.0, 0.0, 0.1]
    reference_point_id: 12
  - name: s4
    label: S4
    model_node_id: 402
    model_coordinates: [0.0, 0.0, 0.1]
    reference_point_id: 13
""".strip(),
        encoding="utf-8",
    )
    (configs_dir / "parameters.yaml").write_text(
        """
material_groups:
  aluminum: [1, 6]
  pcb: [2, 3]
parameters:
  aluminum_density_factor:
    group: aluminum
    target: rho
    mode: fixed
    value: 1.149256275
    range:
      min: 1.14
      max: 1.15
      num_steps: 2
  pcb_density_factor:
    group: pcb
    target: rho
    mode: fixed
    value: 0.9277072359650675
    range:
      min: 0.92
      max: 0.93
      num_steps: 2
  aluminum_E_factor:
    group: aluminum
    target: E
    mode: fixed
    value: 0.5336
  pcb_E_factor:
    group: pcb
    target: E
    mode: fixed
    value: 1.0
""".strip(),
        encoding="utf-8",
    )
    (configs_dir / "sweep_presets.yaml").write_text(
        """
presets:
  aluminum_e_fine:
    description: fine
    fixed_parameters:
      aluminum_density_factor: 1.149256275
      pcb_density_factor: 0.9277072359650675
      pcb_E_factor: 1.0
    varying_parameters:
      aluminum_E_factor:
        mode: list
        values: [0.5336]
  aluminum_e_refine_local:
    description: refine
    select_from_presets: [aluminum_e_fine]
    fixed_parameters:
      aluminum_density_factor: 1.149256275
      pcb_density_factor: 0.9277072359650675
      pcb_E_factor: 1.0
    varying_parameters:
      aluminum_E_factor:
        mode: centered_range
        half_window: 0.002
        step: 0.0005
  pcb_e_confirm:
    description: pcb
    select_from_presets: [aluminum_e_refine_local, aluminum_e_fine]
    fixed_parameters_from_best_run:
      - aluminum_density_factor
      - pcb_density_factor
      - aluminum_E_factor
    varying_parameters:
      pcb_E_factor:
        mode: list
        values: [0.97, 1.0, 1.03]
""".strip(),
        encoding="utf-8",
    )
    return configs_dir


def _write_index_and_manifest(tmp_path: Path, run_id: str, sweep_preset: str, parameter_values: dict[str, float]) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(exist_ok=True)
    (tmp_path / "runs" / run_id).mkdir(parents=True, exist_ok=True)
    index_payload = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "best_accepted_run_id": run_id,
        "best_global_run_id": run_id,
        "runs": [
            {
                "run_id": run_id,
                "sweep_preset": sweep_preset,
                "selected_ranking_mode": "hierarchical",
                "accepted": True,
                "ranking_key": [0.0, 0.01, 0.1],
                "score_robust": 0.1,
                "score_legacy": 0.2,
            }
        ],
    }
    (reports_dir / "runs_index.json").write_text(json.dumps(index_payload), encoding="utf-8")
    manifest_payload = {
        "run_id": run_id,
        "parameter_values": parameter_values,
    }
    (tmp_path / "runs" / run_id / "run_manifest.json").write_text(
        json.dumps(manifest_payload),
        encoding="utf-8",
    )


def _write_index_with_runs(tmp_path: Path, runs: list[dict[str, object]]) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(exist_ok=True)
    for run in runs:
        run_id = str(run["run_id"])
        (tmp_path / "runs" / run_id).mkdir(parents=True, exist_ok=True)
        (tmp_path / "runs" / run_id / "run_manifest.json").write_text(
            json.dumps({"run_id": run_id, "parameter_values": run["parameter_values"]}),
            encoding="utf-8",
        )
    index_payload = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "best_accepted_run_id": runs[0]["run_id"],
        "best_global_run_id": runs[0]["run_id"],
        "runs": [
            {
                "run_id": run["run_id"],
                "sweep_preset": run["sweep_preset"],
                "selected_ranking_mode": "hierarchical",
                "accepted": True,
                "ranking_key": run["ranking_key"],
                "score_robust": 0.1,
                "score_legacy": 0.2,
            }
            for run in runs
        ],
    }
    (reports_dir / "runs_index.json").write_text(json.dumps(index_payload), encoding="utf-8")


def test_expand_sweep_cases_uses_runs_index_for_deterministic_center(tmp_path: Path) -> None:
    configs_dir = _write_config_bundle(tmp_path)
    _write_index_and_manifest(
        tmp_path,
        "001_case",
        "aluminum_e_fine",
        {
          "aluminum_density_factor": 1.149256275,
          "pcb_density_factor": 0.9277072359650675,
          "aluminum_E_factor": 0.5336,
          "pcb_E_factor": 1.0,
        },
    )
    bundle = load_config_bundle(configs_dir)

    cases = _expand_sweep_cases(bundle, "aluminum_e_refine_local")

    aluminum_values = [case["aluminum_E_factor"] for case in cases]
    assert aluminum_values[0] == 0.5316
    assert aluminum_values[-1] == 0.5356
    assert len(aluminum_values) == 9


def test_expand_sweep_cases_can_freeze_parameters_from_best_run(tmp_path: Path) -> None:
    configs_dir = _write_config_bundle(tmp_path)
    _write_index_and_manifest(
        tmp_path,
        "001_case",
        "aluminum_e_fine",
        {
          "aluminum_density_factor": 1.149256275,
          "pcb_density_factor": 0.9277072359650675,
          "aluminum_E_factor": 0.534,
          "pcb_E_factor": 1.0,
        },
    )
    bundle = load_config_bundle(configs_dir)

    cases = _expand_sweep_cases(bundle, "pcb_e_confirm")

    assert [case["pcb_E_factor"] for case in cases] == [0.97, 1.0, 1.03]
    assert all(case["aluminum_E_factor"] == 0.534 for case in cases)


def test_expand_sweep_cases_prefers_first_available_source_preset(tmp_path: Path) -> None:
    configs_dir = _write_config_bundle(tmp_path)
    _write_index_with_runs(
        tmp_path,
        [
            {
                "run_id": "010_refine",
                "sweep_preset": "aluminum_e_refine_local",
                "ranking_key": [0.5, 0.2, 0.2],
                "parameter_values": {
                    "aluminum_density_factor": 1.149256275,
                    "pcb_density_factor": 0.9277072359650675,
                    "aluminum_E_factor": 0.5345,
                    "pcb_E_factor": 1.0,
                },
            },
            {
                "run_id": "001_fine",
                "sweep_preset": "aluminum_e_fine",
                "ranking_key": [0.0, 0.0, 0.0],
                "parameter_values": {
                    "aluminum_density_factor": 1.149256275,
                    "pcb_density_factor": 0.9277072359650675,
                    "aluminum_E_factor": 0.5336,
                    "pcb_E_factor": 1.0,
                },
            },
        ],
    )
    bundle = load_config_bundle(configs_dir)

    cases = _expand_sweep_cases(bundle, "pcb_e_confirm")

    assert all(case["aluminum_E_factor"] == 0.5345 for case in cases)


def test_expand_mass_fit_cases_uses_density_ranges_and_keeps_e_factors_fixed(tmp_path: Path) -> None:
    configs_dir = _write_config_bundle(tmp_path)
    bundle = load_config_bundle(configs_dir)

    cases = _expand_mass_fit_cases(bundle)

    assert len(cases) == 4
    assert {case["aluminum_density_factor"] for case in cases} == {1.14, 1.15}
    assert {case["pcb_density_factor"] for case in cases} == {0.92, 0.93}
    assert all(case["aluminum_E_factor"] == 0.5336 for case in cases)
    assert all(case["pcb_E_factor"] == 1.0 for case in cases)
