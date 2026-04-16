from pathlib import Path

from src.config import load_config_bundle


def test_config_loader_resolves_paths_and_parameter_modes(tmp_path: Path) -> None:
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
pairing:
  frequency_penalty_weight: 0.05
scoring:
  mode: weighted_error
  weights:
    mac: 1.0
    frequency: 0.2
    mass: 0.1
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
""".strip(),
        encoding="utf-8",
    )
    (configs_dir / "parameters.yaml").write_text(
        """
material_groups:
  aluminum: [1, 6]
parameters:
  aluminum_density_factor:
    group: aluminum
    target: rho
    mode: range
    range:
      min: 0.9
      max: 1.1
      num_steps: 3
""".strip(),
        encoding="utf-8",
    )

    bundle = load_config_bundle(configs_dir)

    assert bundle.project.name == "Test Project"
    assert bundle.project.paths.bdf_input == input_bdf.resolve()
    assert bundle.project.mass.budget_basis == "nominal"
    assert bundle.sensors.model_node_order == (486,)
    assert bundle.parameters.parameters["aluminum_density_factor"].expand_values() == [0.9, 1.0, 1.1]
    assert bundle.sweep_presets.presets == {}
