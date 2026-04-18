import copy
import json

from src.config import baseline_factors, load_project_config
from src.mass_model import (
    build_mass_fit_summary,
    calculate_absolute_error_percent,
    calculate_control_values,
    calculate_signed_error_percent,
    calculate_target_band,
    is_better_feasibility,
    score_mass_fit,
)
from src.reporting import (
    _mass_budget_comparison_rows_html,
    _normalize_run_record,
    _select_best_modal_run,
    _select_best_modal_run_balanced,
    _targeted_candidate_rows_html,
    build_ia_payload,
    write_ia_json,
)
from src.workflow import (
    _build_targeted_recommendation,
    _modal_balanced_local_candidates,
    _modal_local_primary_candidates,
    _modal_local_sensitivity_summary,
    build_run_name,
    calculate_proportional_factors,
)


def _synthetic_mass_summary(**overrides):
    masses = {
        "Al_7075-T7351": 224.0,
        "PCB_PSU": 913.0,
        "PCB_MOD": 823.0,
        "PCB_PROC": 979.0,
        "PCB_FPGA": 646.0,
        "Bandeja_FPGA": 225.0,
    }
    summary = {
        "masses_by_material_g": masses,
        "total_mass_g": sum(masses.values()),
    }
    summary.update(overrides)
    return summary


def test_config_detects_target_discrepancy():
    config = load_project_config()
    assert config["search_strategy"]["method"] == "feasibility_with_slack"
    assert config["targets"]["mass_budget_target_sum_g"] == 3811.0
    assert config["targets"]["mass_budget_target_total_delta_g"] == 1.0
    assert config["modal"]["local_fit"]["primary_grid"]["PCB_FPGA"] == [0.9, 0.95, 1.0]
    assert config["modal"]["balanced_local_fit"]["primary_grid"]["PCB_MOD"] == [0.9, 0.95, 1.0]
    assert config["modal"]["diagnostics"]["good"]["mac_min"] == 0.8
    assert config["modal"]["local_sensitivity"]["perturbation_fraction"] == 0.02
    assert config["modal"]["targeted_fit"]["max_candidates"] == 5


def test_run_name_format():
    config = load_project_config()
    run_name = build_run_name(config, 1, baseline_factors(config))
    assert run_name == "001_Al1.000_PSU1.000_MOD1.000_PROC1.000_FPGA1.000_BFPGA1.000"


def test_run_name_adds_e_suffix_when_youngs_changes():
    config = load_project_config()
    run_name = build_run_name(
        config,
        7,
        baseline_factors(config),
        {
            "Al_7075-T7351": 1.0,
            "PCB_PSU": 1.0,
            "PCB_MOD": 0.95,
            "PCB_PROC": 1.0,
            "PCB_FPGA": 1.0,
            "Bandeja_FPGA": 1.0,
        },
    )
    assert run_name.endswith("_EMOD0.950")


def test_modal_local_primary_grid_candidates():
    config = load_project_config()
    candidates = _modal_local_primary_candidates(config)
    assert len(candidates) == 9
    candidate_values = [item["candidate_values"] for item in candidates]
    assert {"PCB_FPGA": 0.9, "PCB_MOD": 0.95} in candidate_values
    assert {"PCB_FPGA": 1.0, "PCB_MOD": 1.05} in candidate_values


def test_modal_balanced_local_grid_candidates():
    config = load_project_config()
    candidates = _modal_balanced_local_candidates(config)
    assert len(candidates) == 9
    candidate_values = [item["candidate_values"] for item in candidates]
    assert {"PCB_FPGA": 0.9, "PCB_MOD": 0.9} in candidate_values
    assert {"PCB_FPGA": 1.0, "PCB_MOD": 1.0} in candidate_values


def test_target_band_calculation():
    band = calculate_target_band(913.0, 0.05)
    assert band["lower_bound_g"] == 867.35
    assert band["upper_bound_g"] == 958.65
    assert band["half_band_g"] == 45.65


def test_signed_and_absolute_mass_error_percent():
    signed = calculate_signed_error_percent(867.35, 913.0)
    absolute = calculate_absolute_error_percent(signed)
    assert round(signed, 6) == -5.0
    assert round(absolute, 6) == 5.0


def test_control_values_and_slack():
    config = load_project_config()
    mass_summary = _synthetic_mass_summary()
    control_values = calculate_control_values(mass_summary, config)
    assert control_values["psu"] == 913.0
    assert control_values["fpga"] == 871.0
    assert control_values["backplanes"] == 225.0
    fit = build_mass_fit_summary(
        mass_summary,
        config,
        factors=baseline_factors(config),
        baseline_reference_factors=baseline_factors(config),
    )
    psu = fit["controls_by_key"]["psu"]
    total = fit["controls_by_key"]["total"]
    assert psu["within_band"] is True
    assert round(psu["slack_down_g"], 3) == 45.65
    assert round(psu["slack_up_g"], 3) == 45.65
    assert total["within_band"] is True
    assert fit["feasible"] is True


def test_proportional_correction_interpolates_between_limits():
    factors = {"PCB_PSU": 1.0, "PCB_MOD": 1.0}
    candidate = calculate_proportional_factors(
        factors,
        {
            "PCB_PSU": {"factor_limit": 1.2},
            "PCB_MOD": {"factor_limit": 0.8},
        },
        0.5,
    )
    assert candidate["PCB_PSU"] == 1.1
    assert candidate["PCB_MOD"] == 0.9


def test_feasibility_selection_prefers_fewer_violations():
    config = load_project_config()
    feasible_summary = build_mass_fit_summary(
        _synthetic_mass_summary(),
        config,
        factors=baseline_factors(config),
        baseline_reference_factors=baseline_factors(config),
    )
    infeasible_summary = build_mass_fit_summary(
        _synthetic_mass_summary(
            masses_by_material_g={
                "Al_7075-T7351": 224.0,
                "PCB_PSU": 850.0,
                "PCB_MOD": 823.0,
                "PCB_PROC": 979.0,
                "PCB_FPGA": 646.0,
                "Bandeja_FPGA": 225.0,
            },
            total_mass_g=3747.0,
        ),
        config,
        factors=baseline_factors(config),
        baseline_reference_factors=baseline_factors(config),
    )
    assert infeasible_summary["unsatisfied_constraints_count"] > feasible_summary["unsatisfied_constraints_count"]
    assert is_better_feasibility(feasible_summary, infeasible_summary) is True
    assert is_better_feasibility(infeasible_summary, feasible_summary) is False


def test_score_payload_is_transparent():
    config = load_project_config()
    mass_summary = _synthetic_mass_summary()
    fit_summary = build_mass_fit_summary(
        mass_summary,
        config,
        factors=baseline_factors(config),
        baseline_reference_factors=baseline_factors(config),
    )
    score = score_mass_fit(
        {
            **mass_summary,
            "mass_budget_comparison": [
                {"target_item": "H-PSU module", "delta_g": 0.0, "rel_error": 0.0},
                {"target_item": "MOD module", "delta_g": 0.0, "rel_error": 0.0},
            ],
        },
        config,
        factors=baseline_factors(config),
        baseline_reference_factors=baseline_factors(config),
        mass_fit_summary=fit_summary,
        nastran_failed=False,
    )
    assert score["score"] == 0.0
    assert score["components"]["unsatisfied_constraints"] == 0
    assert "total_mass_rel_error" in score["components"]
    assert "H-PSU module" in score["mass_budget_rel_errors"]


def test_ia_json_generation(tmp_path):
    config = copy.deepcopy(load_project_config())
    config["project_root"] = tmp_path
    config["resolved_paths"]["config_file"] = tmp_path / "config" / "project.yaml"
    config["resolved_paths"]["runs_dir"] = tmp_path / "runs"
    config["resolved_paths"]["reports_dir"] = tmp_path / "reports" / "runs"
    config["resolved_paths"]["handoff_dir"] = tmp_path / "handoff"
    config["resolved_paths"]["handoff_dir"].mkdir(parents=True)
    fake_run = {
        "run_name": "001_Al1.000_PSU1.000_MOD1.000_PROC1.000_FPGA1.000_BFPGA1.000",
        "run_index": 1,
        "timestamp": "2026-04-17T20:00:00+02:00",
        "status": "completed",
        "strategy": "feasibility_with_slack",
        "is_baseline_reference": True,
        "score": 0.0,
        "total_mass_g": 3810.0,
        "unassigned_mass_g": 224.0,
        "shared_mass_g": 225.0,
        "direct_reference_mass_g": 3361.0,
        "report_paths": {"master": "reports/runs/001/report.html"},
        "run_json_path": "runs/001/run.json",
        "mass_errors": {"total_mass_rel_error": 0.0, "total_mass_error_g": 0.0},
        "total_mass_signed_error_percent": 0.0,
        "total_mass_absolute_error_percent": 0.0,
        "masses_by_material_g": {"Al_7075-T7351": 224.0},
        "mass_budget_comparison": [{"control_key": "psu", "target_item": "H-PSU module", "actual_g": 913.0, "target_g": 913.0, "delta_g": 0.0, "rel_error": 0.0, "signed_error_percent": 0.0, "absolute_error_percent": 0.0, "within_tolerance_5pct": True}],
        "mass_budget_comparison_rows": [{"control_key": "psu", "target_item": "H-PSU module", "actual_g": 913.0, "target_g": 913.0, "delta_g": 0.0, "rel_error": 0.0, "signed_error_percent": 0.0, "absolute_error_percent": 0.0, "within_tolerance_5pct": True}],
        "adjusted_material_mass_rows": [{"material": "Al_7075-T7351", "density_factor": 1.0, "final_density": 2800.0, "final_mass_g": 224.0, "associated_budget_item": "Total mass closure / common structural", "associated_target_g": None, "signed_error_percent": None, "absolute_error_percent": None}],
        "model_mass_contributions": [{"material_name": "Al_7075-T7351", "mass_g": 224.0}],
        "unassigned_materials_g": {"Al_7075-T7351": 224.0},
        "internal_mass": {"masses_by_property_type_g": {"PSHELL": 200.0}},
        "mass_fit": {
            "feasible": True,
            "constraints_satisfied_count": 6,
            "constraint_count": 6,
            "ranking_key": [0, 0.0, 0.0, 0.0],
            "most_limiting_variable": {"material_name": "Al_7075-T7351", "pressure": 0.0},
            "control_rows": [
                {
                    "control_key": "total",
                    "label": "Total mass",
                    "target_g": 3810.0,
                    "lower_bound_g": 3619.5,
                    "upper_bound_g": 4000.5,
                    "actual_g": 3810.0,
                    "within_band": True,
                    "slack_down_g": 190.5,
                    "slack_up_g": 190.5,
                }
            ],
        },
        "feasibility_fit": {
            "iteration_history": [{"pass_index": 1, "feasible": True}],
        },
        "nastran": {"mode": "live"},
        "notes": [],
        "exception": None,
        "youngs_modulus_factors": baseline_factors(config),
    }
    fake_modal_run = {
        **copy.deepcopy(fake_run),
        "run_name": "002_Al0.500_PSU1.586_MOD1.430_PROC1.701_FPGA1.102_BFPGA3.063_EFPGA1.050",
        "run_index": 2,
        "strategy": "modal_fit",
        "is_baseline_reference": False,
        "report_paths": {"master": "reports/runs/002/report.html"},
        "run_json_path": "runs/002/run.json",
        "youngs_modulus_factors": {
            "Al_7075-T7351": 1.0,
            "PCB_PSU": 1.0,
            "PCB_MOD": 1.0,
            "PCB_PROC": 1.0,
            "PCB_FPGA": 1.05,
            "Bandeja_FPGA": 1.0,
        },
        "modal_fit": {
            "method": "youngs_modulus_one_at_a_time",
            "frozen_density_run_name": config["modal"]["base_mass_run_name"],
            "candidate_material_name": "PCB_FPGA",
            "candidate_e_factor": 1.05,
            "mass_feasibility_frozen": True,
        },
        "modal": {
            "status": "completed",
            "base_mass_run_name": config["modal"]["base_mass_run_name"],
            "summary": {
                "mean_mac": 0.72,
                "worst_mac": 0.21,
                "mean_frequency_error": 0.12,
                "worst_frequency_error": 0.2,
                "good_on_both_count": 3,
            },
        },
    }
    fake_modal_baseline = {
        **copy.deepcopy(fake_run),
        "run_name": "003_Al0.500_PSU1.586_MOD1.430_PROC1.701_FPGA1.102_BFPGA3.063",
        "run_index": 3,
        "strategy": "modal_baseline",
        "is_baseline_reference": False,
        "report_paths": {"master": "reports/runs/003/report.html"},
        "run_json_path": "runs/003/run.json",
        "modal": {
            "status": "completed",
            "base_mass_run_name": config["modal"]["base_mass_run_name"],
            "summary": {
                "mean_mac": 0.68,
                "worst_mac": 0.1,
                "mean_frequency_error": 0.18,
                "worst_frequency_error": 0.25,
                "good_on_both_count": 1,
            },
        },
    }
    fake_modal_local = {
        **copy.deepcopy(fake_run),
        "run_name": "004_Al0.500_PSU1.586_MOD1.430_PROC1.701_FPGA1.102_BFPGA3.063_EMOD0.950_EFPGA0.900",
        "run_index": 4,
        "strategy": "modal_fit_local",
        "is_baseline_reference": False,
        "report_paths": {"master": "reports/runs/004/report.html"},
        "run_json_path": "runs/004/run.json",
        "youngs_modulus_factors": {
            "Al_7075-T7351": 1.0,
            "PCB_PSU": 1.0,
            "PCB_MOD": 0.95,
            "PCB_PROC": 1.0,
            "PCB_FPGA": 0.9,
            "Bandeja_FPGA": 1.0,
        },
        "modal_local_refinement": {
            "method": "modal_local_grid_refinement",
            "stage": "primary_grid",
            "rank_overall": 1,
            "rank_within_stage": 1,
            "candidate_values": {"PCB_FPGA": 0.9, "PCB_MOD": 0.95},
            "comparison_vs_modal_baseline": {
                "mean_mac": {"improvement_absolute": 0.03},
                "worst_mac": {"improvement_absolute": 0.04},
                "mean_frequency_error": {"improvement_absolute": 0.01},
                "worst_frequency_error": {"improvement_absolute": 0.02},
                "good_on_both_count": {"improvement_absolute": 1},
            },
            "comparison_vs_best_one_at_a_time": {
                "mean_mac": {"improvement_absolute": 0.01},
                "worst_mac": {"improvement_absolute": 0.02},
                "mean_frequency_error": {"improvement_absolute": -0.005},
                "worst_frequency_error": {"improvement_absolute": 0.01},
                "good_on_both_count": {"improvement_absolute": 0},
            },
        },
        "modal": {
            "status": "completed",
            "base_mass_run_name": config["modal"]["base_mass_run_name"],
            "summary": {
                "mean_mac": 0.73,
                "worst_mac": 0.24,
                "mean_frequency_error": 0.11,
                "worst_frequency_error": 0.19,
                "good_on_both_count": 4,
            },
        },
    }
    fake_modal_balanced = {
        **copy.deepcopy(fake_modal_local),
        "run_name": "005_Al0.500_PSU1.586_MOD1.430_PROC1.701_FPGA1.102_BFPGA3.063_EMOD0.950_EFPGA0.950",
        "run_index": 5,
        "strategy": "modal_fit_balanced_local",
        "report_paths": {"master": "reports/runs/005/report.html"},
        "run_json_path": "runs/005/run.json",
        "modal_balanced_local_refinement": {
            "method": "modal_balanced_local_grid_refinement",
            "rank_overall": 1,
            "candidate_values": {"PCB_FPGA": 0.95, "PCB_MOD": 0.95},
            "comparison_vs_reference_balanced": {
                "reference_run_name": fake_modal_local["run_name"],
                "mean_mac": {"improvement_absolute": 0.005},
                "worst_mac": {"improvement_absolute": 0.001},
                "mean_frequency_error": {"improvement_absolute": 0.003},
                "worst_frequency_error": {"improvement_absolute": 0.001},
                "good_on_both_count": {"improvement_absolute": 0},
            },
        },
        "modal": {
            "status": "completed",
            "base_mass_run_name": config["modal"]["base_mass_run_name"],
            "summary": {
                "mean_mac": 0.71,
                "worst_mac": 0.25,
                "mean_frequency_error": 0.105,
                "worst_frequency_error": 0.18,
                "good_on_both_count": 4,
            },
        },
    }
    payload = build_ia_payload(
        config,
        [fake_run, fake_modal_run, fake_modal_baseline, fake_modal_local, fake_modal_balanced],
        baseline_sensitivity={
            "variation_fraction": 0.05,
            "summary_by_material": [{"material_name": "Al_7075-T7351"}],
            "rows": [],
        },
    )
    assert payload["best_run"]["run_name"] == fake_modal_balanced["run_name"]
    assert payload["baseline"]["reference_run"]["run_name"] == fake_run["run_name"]
    assert payload["runner"]["dry_run"] is False
    assert payload["strategy_current"] == "feasibility_with_slack"
    assert payload["modal_strategy_current"] == "youngs_modulus_one_at_a_time"
    assert payload["strategy_current_modal_refinement"] == "modal_balanced_local_grid_refinement"
    assert payload["best_solution_feasible"] is True
    assert payload["physical_interpretation"]["bandeja_fpga_observation"]
    assert payload["baseline"]["control_rows"][0]["control_key"] == "total"
    assert payload["baseline"]["local_sensitivity"]["variation_fraction"] == 0.05
    assert payload["iteration_history_summary"][0]["pass_index"] == 1
    assert payload["modal_fit"]["candidate_ranking"][0]["run_name"] == fake_modal_run["run_name"]
    assert payload["modal_fit"]["best_candidate"]["changed_youngs_modulus_factors"]["PCB_FPGA"] == 1.05
    assert payload["best_modal_local_run"]["run_name"] == fake_modal_balanced["run_name"]
    assert payload["best_modal_run_strict"]["run_name"] == fake_modal_local["run_name"]
    assert payload["best_modal_run_balanced"]["run_name"] == fake_modal_balanced["run_name"]
    assert payload["recommended_modal_run"]["run_name"] == fake_modal_balanced["run_name"]
    assert payload["modal_local_refinement"]["ranking_local"][0]["run_name"] == fake_modal_local["run_name"]
    assert payload["balanced_local_refinement"]["ranking_local"][0]["run_name"] == fake_modal_balanced["run_name"]
    assert payload["modal_local_refinement"]["conclusions"]
    path = write_ia_json(
        config,
        [fake_run, fake_modal_run, fake_modal_baseline, fake_modal_local, fake_modal_balanced],
        baseline_sensitivity=payload["baseline"]["local_sensitivity"],
    )
    assert path.exists()
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["runs"][0]["run_name"] == fake_run["run_name"]


def test_balanced_modal_ranking_prefers_good_on_both():
    strict_run = {
        "modal": {"status": "completed", "summary": {"mean_mac": 0.75, "worst_mac": 0.20, "mean_frequency_error": 0.25, "worst_frequency_error": 0.4, "good_on_both_count": 0}},
        "run_name": "strict_run",
    }
    balanced_run = {
        "modal": {"status": "completed", "summary": {"mean_mac": 0.70, "worst_mac": 0.22, "mean_frequency_error": 0.12, "worst_frequency_error": 0.2, "good_on_both_count": 2}},
        "run_name": "balanced_run",
    }
    assert _select_best_modal_run([strict_run, balanced_run])["run_name"] == "strict_run"
    assert _select_best_modal_run_balanced([strict_run, balanced_run])["run_name"] == "balanced_run"


def test_mass_budget_reporting_rows_include_percent_fields():
    rows = _mass_budget_comparison_rows_html(
        [
            {
                "control_key": "psu",
                "target_item": "H-PSU module",
                "target_g": 913.0,
                "actual_g": 867.35,
                "signed_error_percent": -5.0,
                "absolute_error_percent": 5.0,
                "within_tolerance_5pct": True,
                "direct_materials": ["PCB_PSU"],
                "shared_materials": [],
            }
        ]
    )
    assert "H-PSU module" in rows[0][0]
    assert "-5.00%" in rows[0][3]
    assert "yes" in rows[0][5]


def test_normalize_run_record_maps_dry_run_status():
    config = load_project_config()
    normalized = _normalize_run_record(
        config,
        {
            "status": "completed",
            "total_mass_g": 3000.0,
            "target_total_mass_g": 3810.0,
            "unassigned_materials_g": {"Al_7075-T7351": 100.0},
            "nastran": {"mode": "dry-run"},
        },
    )
    assert normalized["status"] == "dry_run_completed"
    assert normalized["unassigned_mass_g"] == 100.0
    assert "total_mass_rel_error" in normalized["mass_errors"]


def test_modal_local_sensitivity_summary_and_recommendation():
    config = load_project_config()
    base_run = {
        "run_name": "020_base",
        "youngs_modulus_factors": {
            "Al_7075-T7351": 1.0,
            "PCB_PSU": 1.0,
            "PCB_MOD": 0.95,
            "PCB_PROC": 1.0,
            "PCB_FPGA": 0.9,
            "Bandeja_FPGA": 1.0,
        },
        "modal": {
            "summary": {
                "mean_mac": 0.68,
                "worst_mac": 0.06,
                "mean_frequency_error": 0.17,
                "worst_frequency_error": 0.31,
                "good_on_both_count": 1,
            },
            "paired_mode_diagnostics": [
                {
                    "reference_mode_number": 7,
                    "model_mode_number": 7,
                    "paired_mac": 0.20,
                    "frequency_error_rel": 0.24,
                    "classification": "Problematic",
                    "failure_driver": "mixed",
                },
                {
                    "reference_mode_number": 8,
                    "model_mode_number": 8,
                    "paired_mac": 0.55,
                    "frequency_error_rel": 0.18,
                    "classification": "Problematic",
                    "failure_driver": "frequency",
                },
            ],
            "problematic_modes": {
                "top_problematic": [
                    {"reference_mode_number": 7, "priority_rank": 1},
                    {"reference_mode_number": 8, "priority_rank": 2},
                ]
            },
        },
    }
    candidate_run = {
        "run_name": "021_candidate",
        "modal_local_sensitivity": {
            "material_name": "PCB_FPGA",
            "direction": "decrease",
            "factor_before": 0.9,
            "factor_after": 0.882,
            "factor_delta": -0.018,
            "relative_delta_fraction": -0.02,
        },
        "modal": {
            "status": "completed",
            "summary": {
                "mean_mac": 0.70,
                "worst_mac": 0.08,
                "mean_frequency_error": 0.16,
                "worst_frequency_error": 0.29,
                "good_on_both_count": 1,
            },
            "paired_mode_diagnostics": [
                {
                    "reference_mode_number": 7,
                    "model_mode_number": 7,
                    "paired_mac": 0.26,
                    "frequency_error_rel": 0.19,
                    "classification": "Acceptable",
                    "failure_driver": "frequency",
                },
                {
                    "reference_mode_number": 8,
                    "model_mode_number": 8,
                    "paired_mac": 0.57,
                    "frequency_error_rel": 0.17,
                    "classification": "Acceptable",
                    "failure_driver": "frequency",
                },
            ],
        },
    }
    summary = _modal_local_sensitivity_summary(config, base_run, [candidate_run])
    assert summary["rows"][0]["material_name"] == "PCB_FPGA"
    assert summary["rows"][0]["improved_problematic_modes_count"] == 2
    recommendation = _build_targeted_recommendation(config, base_run, summary)
    assert recommendation["selected_moves"][0]["material_name"] == "PCB_FPGA"
    assert recommendation["selected_moves"][0]["recommended_factor"] == 0.882


def test_targeted_candidate_rows_render_deltas():
    rows = _targeted_candidate_rows_html(
        [
            {
                "rank": 1,
                "run_name": "034_targeted",
                "candidate_values": {"PCB_FPGA": 0.882},
                "mean_mac": 0.71,
                "worst_mac": 0.09,
                "mean_frequency_error": 0.16,
                "good_on_both_count": 1,
                "comparison_vs_recommended": {
                    "mean_mac": {"improvement_absolute": 0.01},
                    "worst_mac": {"improvement_absolute": 0.02},
                    "mean_frequency_error": {"improvement_absolute": 0.005},
                },
            }
        ]
    )
    assert "034_targeted" in rows[0][1]
    assert "frequency error decreased" in rows[0][10]
