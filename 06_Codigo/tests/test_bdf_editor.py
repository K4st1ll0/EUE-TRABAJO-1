from pathlib import Path

from src.bdf_editor import apply_density_factors, apply_mat1_factors, iter_card_blocks, read_variable_materials


def test_mat1_editing_updates_only_requested_rho():
    project_root = Path(__file__).resolve().parents[1]
    base_bdf = (project_root / "../02_SFEM-de-apoyo/SFEM_EUE.bdf").resolve()
    text = base_bdf.read_text(encoding="utf-8", errors="ignore")
    edited, updates = apply_density_factors(
        text,
        [
            {"id": 1, "name": "Al_7075-T7351"},
            {"id": 2, "name": "PCB_PSU"},
            {"id": 3, "name": "PCB_MOD"},
            {"id": 4, "name": "PCB_PROC"},
            {"id": 5, "name": "PCB_FPGA"},
            {"id": 6, "name": "Bandeja_FPGA"},
        ],
        {
            "Al_7075-T7351": 1.10,
            "PCB_PSU": 1.00,
            "PCB_MOD": 1.00,
            "PCB_PROC": 1.00,
            "PCB_FPGA": 1.00,
            "Bandeja_FPGA": 0.90,
        },
    )
    materials = {item.material_id: item for item in read_variable_materials(edited)}
    assert len(updates) == 6
    assert round(materials[1].rho, 6) == 3080.0
    assert round(materials[6].rho, 6) == 2520.0
    assert round(materials[2].rho, 6) == 9929.2266


def test_cbar_with_star_continuation_is_not_treated_as_large_field():
    snippet = (
        "CBAR     102     6       207     121     1.      0.      0.\n"
        "*                                        -1.6295-4       1.2114-4\n"
        "*        0.              -1.6295-4       1.2114-4        0.\n"
    )
    cards = list(iter_card_blocks(snippet))
    assert len(cards) == 1
    assert cards[0].name == "CBAR"
    assert cards[0].large_field is False
    assert cards[0].fields[0].strip() == "102"
    assert cards[0].fields[1].strip() == "6"


def test_mat1_editing_can_update_e_and_rho_independently():
    project_root = Path(__file__).resolve().parents[1]
    base_bdf = (project_root / "../05_FEM/SFEM_EUE.bdf").resolve()
    text = base_bdf.read_text(encoding="utf-8", errors="ignore")
    edited, updates = apply_mat1_factors(
        text,
        [
            {"id": 1, "name": "Al_7075-T7351"},
            {"id": 2, "name": "PCB_PSU"},
            {"id": 3, "name": "PCB_MOD"},
            {"id": 4, "name": "PCB_PROC"},
            {"id": 5, "name": "PCB_FPGA"},
            {"id": 6, "name": "Bandeja_FPGA"},
        ],
        {
            "Al_7075-T7351": 1.10,
            "PCB_PSU": 1.00,
            "PCB_MOD": 1.00,
            "PCB_PROC": 1.00,
            "PCB_FPGA": 1.00,
            "Bandeja_FPGA": 1.00,
        },
        {
            "Al_7075-T7351": 0.95,
            "PCB_PSU": 1.00,
            "PCB_MOD": 1.00,
            "PCB_PROC": 1.00,
            "PCB_FPGA": 1.00,
            "Bandeja_FPGA": 1.05,
        },
    )
    materials = {item.material_id: item for item in read_variable_materials(edited)}
    updates_by_name = {item["material_name"]: item for item in updates}
    assert round(materials[1].rho, 6) == 3080.0
    assert round(materials[1].e, 3) == round(7.0999998e10 * 0.95, 3)
    assert round(materials[6].e, 3) == round(7.0999998e10 * 1.05, 3)
    assert updates_by_name["Al_7075-T7351"]["density_factor"] == 1.10
    assert updates_by_name["Bandeja_FPGA"]["e_factor"] == 1.05
