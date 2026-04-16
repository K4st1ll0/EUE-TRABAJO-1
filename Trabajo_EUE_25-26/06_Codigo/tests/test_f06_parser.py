from pathlib import Path

from src.f06_parser import parse_modal_data


def test_f06_parser_reads_t1_t2_t3_and_tracks_valid_modes(tmp_path: Path) -> None:
    point_ids = [486, 452, 2040, 402]
    f06_text = "\n".join(
        [
            "                                              R E A L   E I G E N V A L U E S",
            "   MODE    EXTRACTION      EIGENVALUE            RADIANS             CYCLES",
            "        1         1        1.000000E+06        1.000000E+03        1.591549E+02",
            "        2         2        2.000000E+06        1.414214E+03        2.250791E+02",
            "",
            "      EIGENVALUE =  1.000000E+06",
            "          CYCLES =  1.591549E+02         R E A L   E I G E N V E C T O R   N O .          1",
            "      POINT ID.   TYPE          T1             T2             T3             R1             R2             R3",
            "           486      G      1.0            2.0            3.0            10.0           20.0           30.0",
            "           452      G      4.0            5.0            6.0            40.0           50.0           60.0",
            "          2040      G      7.0            8.0            9.0            70.0           80.0           90.0",
            "           402      G      0.1            0.2            0.3            11.0           22.0           33.0",
            "      EIGENVALUE =  2.000000E+06",
            "          CYCLES =  2.250791E+02         R E A L   E I G E N V E C T O R   N O .          2",
            "      POINT ID.   TYPE          T1             T2             T3             R1             R2             R3",
            "           486      G      1.5            2.5            3.5            10.0           20.0           30.0",
            "           452      G      4.5            5.5            6.5            40.0           50.0           60.0",
        ]
    )
    f06_path = tmp_path / "sample.f06"
    f06_path.write_text(f06_text, encoding="utf-8")

    parsed = parse_modal_data(f06_path, point_ids=point_ids, num_modes=2)

    assert parsed.vector_basis == "T1_T2_T3"
    assert parsed.coordinate_system == "global"
    assert parsed.requested_point_ids == tuple(point_ids)
    assert parsed.valid_modes == (1,)
    assert parsed.invalid_modes == (2,)
    assert parsed.valid_mode_count == 1
    assert parsed.missing_points_by_mode[2] == (2040, 402)
    assert parsed.vectors[1][486] == (1.0, 2.0, 3.0)
