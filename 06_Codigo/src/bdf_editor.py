from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


FIELD_WIDTH_SMALL = 8
FIELD_WIDTH_LARGE = 16
MATERIAL_COMMENT_PREFIX = "$ Material Record :"


def parse_nastran_float(raw_value: str | None) -> float | None:
    if raw_value is None:
        return None
    text = raw_value.strip()
    if not text:
        return None

    normalized = text.replace("D", "E").replace("d", "E")
    if "E" in normalized or "e" in normalized:
        return float(normalized)

    exponent_match = re.match(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))([+-]\d+)$", normalized)
    if exponent_match:
        return float(f"{exponent_match.group(1)}e{exponent_match.group(2)}")

    return float(normalized)


def format_large_field_float(value: float) -> str:
    candidates = [f"{value:.9E}", f"{value:.8E}", f"{value:.7E}"]
    for candidate in candidates:
        if len(candidate) <= FIELD_WIDTH_LARGE:
            return candidate.rjust(FIELD_WIDTH_LARGE)
    return candidates[-1][:FIELD_WIDTH_LARGE]


def _field_token(line: str) -> str:
    return line[:FIELD_WIDTH_SMALL].strip()


def _is_comment_or_blank(line: str) -> bool:
    stripped = line.strip()
    return not stripped or stripped.startswith("$")


def _is_continuation(line: str) -> bool:
    token = _field_token(line)
    return token == "" or token in {"*", "+"}


def _split_small_fields(line: str) -> list[str]:
    raw = line.rstrip("\n")
    return [raw[index : index + FIELD_WIDTH_SMALL] for index in range(0, len(raw), FIELD_WIDTH_SMALL)]


def _split_large_fields(line: str) -> list[str]:
    raw = line.rstrip("\n")
    chunks = [raw[:FIELD_WIDTH_SMALL]]
    for index in range(FIELD_WIDTH_SMALL, len(raw), FIELD_WIDTH_LARGE):
        chunks.append(raw[index : index + FIELD_WIDTH_LARGE])
    return chunks


@dataclass
class CardBlock:
    name: str
    raw_name: str
    fields: list[str]
    lines: list[str]
    line_indices: list[int]
    large_field: bool


@dataclass
class MaterialCard:
    material_id: int
    name: str
    e: float | None
    nu: float | None
    rho: float | None
    line_indices: list[int]
    large_field: bool
    fields: list[str]


def iter_card_blocks(text: str) -> Iterable[CardBlock]:
    lines = text.splitlines(keepends=True)
    index = 0
    while index < len(lines):
        line = lines[index]
        if _is_comment_or_blank(line) or _is_continuation(line):
            index += 1
            continue

        block_lines = [line]
        block_indices = [index]
        index += 1
        while index < len(lines):
            candidate = lines[index]
            if _is_comment_or_blank(candidate) or not _is_continuation(candidate):
                break
            block_lines.append(candidate)
            block_indices.append(index)
            index += 1

        raw_name = _field_token(block_lines[0])
        large_field = raw_name.endswith("*")
        if large_field:
            fields = _split_large_fields(block_lines[0])[1:]
            for item in block_lines[1:]:
                fields.extend(_split_large_fields(item)[1:])
        else:
            fields = _split_small_fields(block_lines[0])[1:]
            for item in block_lines[1:]:
                fields.extend(_split_small_fields(item)[1:])

        yield CardBlock(
            name=raw_name.rstrip("*"),
            raw_name=raw_name,
            fields=fields,
            lines=block_lines,
            line_indices=block_indices,
            large_field=large_field,
        )


def _find_material_comment(lines: list[str], start_index: int) -> str | None:
    for index in range(start_index - 1, max(start_index - 8, -1), -1):
        stripped = lines[index].strip()
        if stripped.startswith(MATERIAL_COMMENT_PREFIX):
            return stripped.split(":", 1)[1].strip()
        if stripped and not stripped.startswith("$"):
            break
    return None


def read_variable_materials(text: str) -> list[MaterialCard]:
    lines = text.splitlines(keepends=True)
    materials: list[MaterialCard] = []
    for block in iter_card_blocks(text):
        if block.name != "MAT1":
            continue
        material_id = int(block.fields[0].strip())
        material_name = _find_material_comment(lines, block.line_indices[0]) or f"MAT1_{material_id}"
        materials.append(
            MaterialCard(
                material_id=material_id,
                name=material_name,
                e=parse_nastran_float(block.fields[1] if len(block.fields) > 1 else None),
                nu=parse_nastran_float(block.fields[3] if len(block.fields) > 3 else None),
                rho=parse_nastran_float(block.fields[4] if len(block.fields) > 4 else None),
                line_indices=block.line_indices,
                large_field=block.large_field,
                fields=block.fields,
            )
        )
    return materials


def apply_density_factors(
    original_text: str,
    target_materials: list[dict[str, object]],
    factors: dict[str, float],
) -> tuple[str, list[dict[str, object]]]:
    baseline_youngs_factors = {str(item["name"]): 1.0 for item in target_materials}
    return apply_mat1_factors(
        original_text=original_text,
        target_materials=target_materials,
        density_factors=factors,
        youngs_modulus_factors=baseline_youngs_factors,
    )


def apply_mat1_factors(
    original_text: str,
    target_materials: list[dict[str, object]],
    density_factors: dict[str, float],
    youngs_modulus_factors: dict[str, float],
) -> tuple[str, list[dict[str, object]]]:
    lines = original_text.splitlines(keepends=True)
    materials_by_id = {item.material_id: item for item in read_variable_materials(original_text)}
    updates: list[dict[str, object]] = []

    for target in target_materials:
        material_id = int(target["id"])
        material_name = str(target["name"])
        density_factor = float(density_factors[material_name])
        youngs_factor = float(youngs_modulus_factors[material_name])
        card = materials_by_id.get(material_id)
        if card is None:
            raise ValueError(f"Expected MAT1 {material_id} for {material_name}, but it was not found.")
        if card.name != material_name:
            raise ValueError(
                f"MAT1 {material_id} mismatch. Expected '{material_name}', found '{card.name}'."
            )
        if not card.large_field or len(card.line_indices) < 2:
            raise ValueError(f"MAT1 {material_id} is not in the expected large-field format.")

        original_e = float(card.e or 0.0)
        original_rho = float(card.rho or 0.0)
        new_e = original_e * youngs_factor
        new_rho = original_rho * density_factor

        youngs_line_index = card.line_indices[0]
        raw_youngs_line = lines[youngs_line_index].rstrip("\n")
        youngs_fields = _split_large_fields(raw_youngs_line)
        while len(youngs_fields) < 3:
            youngs_fields.append("".rjust(FIELD_WIDTH_LARGE))
        youngs_fields[2] = format_large_field_float(new_e)
        lines[youngs_line_index] = "".join(youngs_fields).rstrip() + "\n"

        density_line_index = card.line_indices[1]
        raw_density_line = lines[density_line_index].rstrip("\n")
        density_fields = _split_large_fields(raw_density_line)
        while len(density_fields) < 2:
            density_fields.append("".rjust(FIELD_WIDTH_LARGE))
        density_fields[1] = format_large_field_float(new_rho)
        lines[density_line_index] = "".join(density_fields).rstrip() + "\n"

        updates.append(
            {
                "material_id": material_id,
                "material_name": material_name,
                "e_original": original_e,
                "e_new": new_e,
                "e_factor": youngs_factor,
                "rho_original": original_rho,
                "rho_new": new_rho,
                "factor": density_factor,
                "density_factor": density_factor,
                "line_numbers": [index + 1 for index in card.line_indices],
            }
        )

    return "".join(lines), updates
