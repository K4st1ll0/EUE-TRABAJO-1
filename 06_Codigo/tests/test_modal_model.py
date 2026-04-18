import copy

from src.config import load_project_config
from src.modal_model import (
    build_modal_correlation,
    build_modal_vector,
    calculate_absolute_frequency_error_percent,
    calculate_mac,
    calculate_mac_matrix,
    calculate_signed_frequency_error_percent,
    pair_modal_modes,
    parse_modal_f06,
)
from src.reporting import write_master_report


def _modal_test_config(num_modes: int = 2):
    config = copy.deepcopy(load_project_config())
    config["modal"]["num_modes"] = num_modes
    return config


def _sample_f06(mode_rows):
    blocks = []
    for mode_number, frequency_hz, node_components in mode_rows:
        blocks.append(f"      EIGENVALUE =  {float(mode_number) * 1.0e6: .6E}")
        blocks.append(
            f"          CYCLES =  {float(frequency_hz): .6E}         R E A L   E I G E N V E C T O R   N O.          {mode_number}"
        )
        blocks.append("")
        blocks.append("      POINT ID.   TYPE          T1             T2             T3             R1             R2             R3")
        for node_id in (486, 452, 2040, 402):
            t1, t2, t3 = node_components[node_id]
            blocks.append(
                f"{node_id:>14}      G      {t1: .6E}   {t2: .6E}   {t3: .6E}   0.0            0.0            0.0"
            )
        blocks.append("")
    return "\n".join(blocks)


def test_parse_modal_f06_extracts_modes_and_requested_nodes():
    config = _modal_test_config()
    text = _sample_f06(
        [
            (1, 200.0, {486: (1.0, 0.0, 0.0), 452: (0.0, 1.0, 0.0), 2040: (0.0, 0.0, 1.0), 402: (1.0, 1.0, 1.0)}),
            (2, 250.0, {486: (2.0, 0.0, 0.0), 452: (0.0, 2.0, 0.0), 2040: (0.0, 0.0, 2.0), 402: (2.0, 2.0, 2.0)}),
        ]
    )
    parsed = parse_modal_f06(text, config, "sample_reference.f06")
    assert parsed["parsed_modes"] == 2
    assert parsed["modes"][0]["mode_number"] == 1
    assert parsed["modes"][0]["frequency_hz"] == 200.0
    assert parsed["modes"][0]["node_components"][486]["T1"] == 1.0
    assert parsed["modes"][1]["node_components"][402]["T3"] == 2.0


def test_parse_modal_f06_supports_reference_point_alias_mapping():
    config = _modal_test_config(num_modes=1)
    config["resolved_paths"]["modal_reference_file"] = "C:/tmp/reference_modal.f06"
    config["modal"]["reference"]["measurement_points"] = [
        {"label": "ref_1", "source_node_id": 52018382, "target_node_id": 486},
        {"label": "ref_2", "source_node_id": 52001152, "target_node_id": 452},
        {"label": "ref_3", "source_node_id": 52098820, "target_node_id": 2040},
        {"label": "ref_4", "source_node_id": 52763676, "target_node_id": 402},
    ]
    text = "\n".join(
        [
            "      EIGENVALUE =   1.000000E+06",
            "          CYCLES =   2.000000E+02         R E A L   E I G E N V E C T O R   N O.          1",
            "",
            "      POINT ID.   TYPE          T1             T2             T3             R1             R2             R3",
            "      52018382      G       1.000000E+00   0.000000E+00   0.000000E+00   0.0            0.0            0.0",
            "      52001152      G       0.000000E+00   1.000000E+00   0.000000E+00   0.0            0.0            0.0",
            "      52098820      G       0.000000E+00   0.000000E+00   1.000000E+00   0.0            0.0            0.0",
            "      52763676      G       1.000000E+00   1.000000E+00   1.000000E+00   0.0            0.0            0.0",
        ]
    )
    parsed = parse_modal_f06(text, config, "C:/tmp/reference_modal.f06")
    assert parsed["requested_source_nodes"] == [52001152, 52018382, 52098820, 52763676]
    assert parsed["modes"][0]["node_components"][486]["T1"] == 1.0
    assert parsed["modes"][0]["node_components"][402]["T3"] == 1.0


def test_build_modal_vector_respects_exact_order():
    config = _modal_test_config()
    mode = {
        "mode_number": 1,
        "node_components": {
            486: {"T1": 1.0, "T2": 2.0, "T3": 3.0},
            452: {"T1": 4.0, "T2": 5.0, "T3": 6.0},
            2040: {"T1": 7.0, "T2": 8.0, "T3": 9.0},
            402: {"T1": 10.0, "T2": 11.0, "T3": 12.0},
        },
    }
    assert build_modal_vector(mode, config) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]


def test_mac_formula_and_matrix():
    identical = calculate_mac([1.0, 2.0], [1.0, 2.0])
    orthogonal = calculate_mac([1.0, 0.0], [0.0, 1.0])
    assert identical == 1.0
    assert orthogonal == 0.0

    reference_modes = [{"modal_vector": [1.0, 0.0]}, {"modal_vector": [0.0, 1.0]}]
    model_modes = [{"modal_vector": [1.0, 0.0]}, {"modal_vector": [0.0, 1.0]}]
    matrix = calculate_mac_matrix(reference_modes, model_modes)
    assert matrix == [[1.0, 0.0], [0.0, 1.0]]


def test_pairing_prefers_high_mac_with_frequency_as_secondary_term():
    reference_modes = [
        {"mode_number": 1, "frequency_hz": 100.0, "modal_vector": [1.0, 0.0]},
        {"mode_number": 2, "frequency_hz": 200.0, "modal_vector": [0.0, 1.0]},
    ]
    model_modes = [
        {"mode_number": 8, "frequency_hz": 205.0, "modal_vector": [0.0, 1.0]},
        {"mode_number": 9, "frequency_hz": 102.0, "modal_vector": [1.0, 0.0]},
    ]
    matrix = calculate_mac_matrix(reference_modes, model_modes)
    pairing = pair_modal_modes(reference_modes, model_modes, matrix, pairing_frequency_penalty_weight=0.25)
    assert pairing["pairs"][0]["model_mode_number"] == 9
    assert pairing["pairs"][1]["model_mode_number"] == 8


def test_build_modal_correlation_returns_summary_metrics():
    config = _modal_test_config()
    reference_text = _sample_f06(
        [
            (1, 200.0, {486: (1.0, 0.0, 0.0), 452: (0.0, 1.0, 0.0), 2040: (0.0, 0.0, 1.0), 402: (1.0, 1.0, 1.0)}),
            (2, 250.0, {486: (0.0, 1.0, 0.0), 452: (1.0, 0.0, 0.0), 2040: (0.0, 0.0, 1.0), 402: (1.0, 0.0, 1.0)}),
        ]
    )
    model_text = _sample_f06(
        [
            (1, 202.0, {486: (1.0, 0.0, 0.0), 452: (0.0, 1.0, 0.0), 2040: (0.0, 0.0, 1.0), 402: (1.0, 1.0, 1.0)}),
            (2, 255.0, {486: (0.0, 1.0, 0.0), 452: (1.0, 0.0, 0.0), 2040: (0.0, 0.0, 1.0), 402: (1.0, 0.0, 1.0)}),
        ]
    )
    correlation = build_modal_correlation(
        reference_text,
        model_text,
        config,
        "reference.f06",
        "model.f06",
    )
    assert correlation["status"] == "completed"
    assert len(correlation["mac_matrix"]) == 2
    assert correlation["summary"]["mean_mac"] > 0.99
    assert correlation["summary"]["mean_frequency_error"] > 0.0


def test_signed_frequency_error_percent():
    signed = calculate_signed_frequency_error_percent(200.0, 210.0)
    absolute = calculate_absolute_frequency_error_percent(signed)
    assert signed == 5.0
    assert absolute == 5.0


def test_build_modal_correlation_produces_problematic_mode_diagnostics():
    config = _modal_test_config()
    reference_text = _sample_f06(
        [
            (1, 200.0, {486: (1.0, 0.0, 0.0), 452: (0.0, 1.0, 0.0), 2040: (0.0, 0.0, 1.0), 402: (1.0, 0.0, 0.0)}),
            (2, 250.0, {486: (0.0, 0.0, 1.0), 452: (0.0, 1.0, 0.0), 2040: (1.0, 0.0, 0.0), 402: (0.0, 1.0, 0.0)}),
        ]
    )
    model_text = _sample_f06(
        [
            (1, 202.0, {486: (1.0, 0.0, 0.0), 452: (0.0, 1.0, 0.0), 2040: (0.0, 0.0, 1.0), 402: (1.0, 0.0, 0.0)}),
            (2, 320.0, {486: (1.0, 0.0, 0.0), 452: (0.0, 0.0, 1.0), 2040: (0.0, 1.0, 0.0), 402: (1.0, 0.0, 0.0)}),
        ]
    )
    correlation = build_modal_correlation(reference_text, model_text, config, "reference.f06", "model.f06")
    diagnostics = correlation["paired_mode_diagnostics"]
    assert diagnostics[0]["classification"] == "Good"
    assert diagnostics[1]["classification"] == "Problematic"
    assert diagnostics[1]["failure_driver"] in {"shape", "mixed", "frequency"}
    assert correlation["problematic_modes"]["top_problematic"][0]["reference_mode_number"] == 2
    assert "domin" in diagnostics[1]["diagnostic_note"].lower()


def test_master_report_includes_modal_section(tmp_path):
    config = _modal_test_config()
    config["project_root"] = tmp_path
    config["resolved_paths"]["handoff_dir"] = tmp_path / "handoff"
    config["resolved_paths"]["handoff_dir"].mkdir(parents=True)
    modal_run = {
        "run_name": "006_modal",
        "run_index": 6,
        "strategy": "modal_baseline",
        "status": "completed",
        "report_paths": {"master": "reports/runs/006_modal/report.html"},
        "mass_fit": {"feasible": True, "constraints_satisfied_count": 6, "constraint_count": 6, "most_limiting_variable": None},
        "total_mass_g": 3876.6,
        "mass_errors": {"total_mass_rel_error": 0.01},
        "nastran": {"mode": "live"},
        "modal": {
            "status": "completed",
            "base_mass_run_name": config["modal"]["base_mass_run_name"],
            "summary": {
                "mean_mac": 0.92,
                "worst_mac": 0.81,
                "mean_frequency_error": 0.03,
                "good_on_both_count": 8,
            },
        },
        "modal_score": -1.0,
    }
    path = write_master_report(config, [modal_run], baseline_sensitivity=None)
    html = path.read_text(encoding="utf-8")
    assert "Phase 2 Modal Correlation" in html
    assert "006_modal" in html
    assert "0.920" in html
