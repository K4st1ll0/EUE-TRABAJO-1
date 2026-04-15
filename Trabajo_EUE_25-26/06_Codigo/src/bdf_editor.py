from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import re
import shutil


class BdfEditError(ValueError):
    """Raised when the BDF cannot be edited safely."""


_EXPONENTIAL_WITHOUT_E = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))([+-]\d+)$")


def parse_nastran_number(value: str) -> float:
    token = value.strip()
    if not token:
        raise ValueError("Cannot parse an empty numeric token.")
    token = token.replace("D", "E").replace("d", "E")
    if "E" not in token and "e" not in token:
        match = _EXPONENTIAL_WITHOUT_E.match(token)
        if match:
            token = f"{match.group(1)}E{match.group(2)}"
    return float(token)


def format_nastran_float(value: float) -> str:
    return f"{value:.10E}"


@dataclass(frozen=True)
class MaterialFieldUpdate:
    mat1_id: int
    field_name: str
    factor: float
    parameter_name: str
    material_group: str


@dataclass(frozen=True)
class MaterialModificationRecord:
    mat1_id: int
    field_name: str
    parameter_name: str
    material_group: str
    original_value: float
    new_value: float
    factor_applied: float


@dataclass(frozen=True)
class BdfEditResult:
    source_bdf: Path
    generated_bdf: Path
    modifications: tuple[MaterialModificationRecord, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_bdf": str(self.source_bdf),
            "generated_bdf": str(self.generated_bdf),
            "modifications": [asdict(item) for item in self.modifications],
        }


def _parse_mat1_star_card(lines: list[str], index: int) -> tuple[dict[str, float], int]:
    line = lines[index].strip()
    if index + 1 >= len(lines):
        raise BdfEditError("Incomplete MAT1* card: missing continuation line.")
    continuation = lines[index + 1].strip()
    if not continuation.startswith("*"):
        raise BdfEditError("Incomplete MAT1* card: continuation line does not start with '*'.")

    first_tokens = line.split()
    second_tokens = continuation.split()
    if len(first_tokens) < 5 or len(second_tokens) < 2:
        raise BdfEditError("MAT1* card does not contain the expected E, G, nu, and rho fields.")

    return (
        {
            "mid": int(first_tokens[1]),
            "E": parse_nastran_number(first_tokens[2]),
            "nu": parse_nastran_number(first_tokens[4]),
            "rho": parse_nastran_number(second_tokens[1]),
        },
        2,
    )


def _build_mat1_free_field_line(mid: int, young_modulus: float, nu: float, rho: float) -> str:
    return (
        f"MAT1,{mid},{format_nastran_float(young_modulus)},,"
        f"{format_nastran_float(nu)},{format_nastran_float(rho)}"
    )


def copy_and_apply_mat1_updates(
    source_bdf: Path,
    destination_bdf: Path,
    updates: list[MaterialFieldUpdate],
) -> BdfEditResult:
    if source_bdf.resolve() == destination_bdf.resolve():
        raise BdfEditError("The generated BDF must not overwrite the original BDF.")

    destination_bdf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_bdf, destination_bdf)

    lines = destination_bdf.read_text(encoding="utf-8").splitlines()
    updates_by_mat1: dict[int, dict[str, MaterialFieldUpdate]] = {}
    for update in updates:
        updates_by_mat1.setdefault(update.mat1_id, {})[update.field_name] = update

    output_lines: list[str] = []
    modifications: list[MaterialModificationRecord] = []
    visited_mat1_ids: set[int] = set()

    index = 0
    while index < len(lines):
        stripped = lines[index].lstrip()
        if stripped.startswith("MAT1*"):
            parsed, consumed = _parse_mat1_star_card(lines, index)
            mat1_id = int(parsed["mid"])
            if mat1_id in updates_by_mat1:
                visited_mat1_ids.add(mat1_id)
                effective_e = parsed["E"]
                effective_rho = parsed["rho"]
                for field_name, update in updates_by_mat1[mat1_id].items():
                    original_value = parsed[field_name]
                    new_value = original_value * update.factor
                    if field_name == "E":
                        effective_e = new_value
                    elif field_name == "rho":
                        effective_rho = new_value
                    else:
                        raise BdfEditError(f"Unsupported MAT1 field '{field_name}'.")
                    modifications.append(
                        MaterialModificationRecord(
                            mat1_id=mat1_id,
                            field_name=field_name,
                            parameter_name=update.parameter_name,
                            material_group=update.material_group,
                            original_value=original_value,
                            new_value=new_value,
                            factor_applied=update.factor,
                        )
                    )
                output_lines.append(
                    _build_mat1_free_field_line(
                        mid=mat1_id,
                        young_modulus=effective_e,
                        nu=parsed["nu"],
                        rho=effective_rho,
                    )
                )
            else:
                output_lines.extend(lines[index : index + consumed])
            index += consumed
            continue

        output_lines.append(lines[index])
        index += 1

    missing_ids = sorted(set(updates_by_mat1) - visited_mat1_ids)
    if missing_ids:
        raise BdfEditError(f"MAT1 IDs not found in the input BDF: {missing_ids}")

    destination_bdf.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    return BdfEditResult(
        source_bdf=source_bdf,
        generated_bdf=destination_bdf,
        modifications=tuple(modifications),
    )


def write_modifications_json(result: BdfEditResult, path: Path) -> None:
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

