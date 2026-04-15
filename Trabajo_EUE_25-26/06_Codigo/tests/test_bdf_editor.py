from pathlib import Path

from src.bdf_editor import MaterialFieldUpdate, copy_and_apply_mat1_updates


def test_mat1_editor_updates_only_requested_fields(tmp_path: Path) -> None:
    fixture = Path(__file__).resolve().parent / "fixtures" / "sample_mat1.bdf"
    source_bdf = tmp_path / "source.bdf"
    source_bdf.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
    generated_bdf = tmp_path / "generated.bdf"

    result = copy_and_apply_mat1_updates(
        source_bdf=source_bdf,
        destination_bdf=generated_bdf,
        updates=[
            MaterialFieldUpdate(
                mat1_id=1,
                field_name="E",
                factor=1.10,
                parameter_name="aluminum_E_factor",
                material_group="aluminum",
            ),
            MaterialFieldUpdate(
                mat1_id=1,
                field_name="rho",
                factor=0.90,
                parameter_name="aluminum_density_factor",
                material_group="aluminum",
            ),
        ],
    )

    original_text = source_bdf.read_text(encoding="utf-8")
    edited_text = generated_bdf.read_text(encoding="utf-8")

    assert "MAT1*    1" in original_text
    assert "MAT1,1," in edited_text
    assert "3.3000000000E-01" in edited_text
    assert "MAT1*    2" in edited_text
    assert len(result.modifications) == 2
    assert result.modifications[0].mat1_id == 1
    assert result.modifications[1].field_name == "rho"
