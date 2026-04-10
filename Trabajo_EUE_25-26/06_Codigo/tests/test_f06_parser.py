from __future__ import annotations

from pathlib import Path

from src.nastran.f06_parser import parse_real_eigenvalues


def test_parse_real_eigenvalues(tmp_path: Path) -> None:
    f06_path = tmp_path / "sample.f06"
    f06_path.write_text(
        "\n".join(
            [
                "                                              R E A L   E I G E N V A L U E S",
                "   MODE    EXTRACTION      EIGENVALUE            RADIANS             CYCLES            GENERALIZED         GENERALIZED",
                "    NO.       ORDER                                                                       MASS              STIFFNESS",
                "        1         1        1.677460E+06        1.295168E+03        2.061324E+02        1.000000E+00        1.677460E+06",
                "        2         2        2.429647E+06        1.558732E+03        2.480800E+02        1.000000E+00        2.429647E+06",
                "some trailer",
            ]
        ),
        encoding="utf-8",
    )

    result = parse_real_eigenvalues(f06_path)
    assert len(result.records) == 2
    assert result.records[0].mode == 1
    assert abs(result.records[1].cycles - 248.08) < 1e-6
    assert result.raw_block is not None
