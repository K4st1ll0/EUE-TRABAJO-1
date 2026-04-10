from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any


DATA_ROW_PATTERN = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+([\-+0-9.EeDd]+)\s+([\-+0-9.EeDd]+)\s+([\-+0-9.EeDd]+)\s+([\-+0-9.EeDd]+)\s+([\-+0-9.EeDd]+)\s*$"
)


def _to_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


@dataclass(slots=True)
class EigenvalueRecord:
    mode: int
    extraction_order: int
    eigenvalue: float
    radians: float
    cycles: float
    generalized_mass: float
    generalized_stiffness: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class F06ParseResult:
    source_path: Path
    records: list[EigenvalueRecord]
    raw_block: str | None
    warnings: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "records": [record.as_dict() for record in self.records],
            "raw_block": self.raw_block,
            "warnings": self.warnings,
        }


def parse_real_eigenvalues(path: Path) -> F06ParseResult:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start_index = None
    for index, line in enumerate(lines):
        if "R E A L   E I G E N V A L U E S" in line:
            start_index = index
            break

    if start_index is None:
        return F06ParseResult(
            source_path=path,
            records=[],
            raw_block=None,
            warnings=["No se encontro el bloque REAL EIGENVALUES en el F06."],
        )

    raw_lines: list[str] = []
    records: list[EigenvalueRecord] = []
    data_started = False

    for line in lines[start_index:]:
        raw_lines.append(line)
        match = DATA_ROW_PATTERN.match(line)
        if match:
            data_started = True
            records.append(
                EigenvalueRecord(
                    mode=int(match.group(1)),
                    extraction_order=int(match.group(2)),
                    eigenvalue=_to_float(match.group(3)),
                    radians=_to_float(match.group(4)),
                    cycles=_to_float(match.group(5)),
                    generalized_mass=_to_float(match.group(6)),
                    generalized_stiffness=_to_float(match.group(7)),
                )
            )
            continue

        if data_started and line.strip():
            break

    return F06ParseResult(
        source_path=path,
        records=records,
        raw_block="\n".join(raw_lines),
        warnings=[] if records else ["Se encontro el bloque pero no se parsearon filas numericas."],
    )
