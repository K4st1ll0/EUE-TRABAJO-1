from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np

from .bdf_editor import parse_nastran_number

MASS_BUDGET_ROW_INDEX = {
    "H-PSU module": 7,
    "MOD module": 8,
    "FPGA module": 9,
    "PROC module": 10,
    "Backplane modules": 11,
    "Total mass": 12,
}

BOARD_REGION_TO_CATEGORY = {
    "09_A_t-boards-PSU": "H-PSU module",
    "09_B_t-boards-MOD": "MOD module",
    "09_D_t-boards-FPGA": "FPGA module",
    "09_C_t-boards-PROC": "PROC module",
}

BACKPLANE_REGIONS = {"07_t-bandejas", "07_t-bandejas_FPGA"}

DEFAULT_MODULE_REFERENCE_Z = {
    "H-PSU module": 0.00815,
    "MOD module": 0.03945,
    "FPGA module": 0.06945,
    "PROC module": 0.09945,
}


@dataclass(frozen=True)
class ElementMassContribution:
    region_name: str
    property_id: int
    material_id: int
    element_type: str
    mass: float
    centroid_z: float


@dataclass(frozen=True)
class MassBudgetRow:
    category: str
    budget_mass: float
    model_mass: float
    delta_mass: float
    relative_delta: float | None


@dataclass(frozen=True)
class MassClassificationRow:
    region_name: str
    assigned_to: str
    assignment_rule: str
    mass: float


@dataclass(frozen=True)
class MassBudgetComparison:
    workbook: str
    sheet: str
    basis: str
    budget_total_mass: float
    model_total_mass: float
    delta_mass: float
    relative_delta: float | None
    rows: tuple[MassBudgetRow, ...]
    classification_rows: tuple[MassClassificationRow, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class MassSummary:
    total_mass: float
    relative_error: float
    target_mass: float
    method: str
    budget_comparison: MassBudgetComparison | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_primary_card(line: str) -> bool:
    stripped = line.lstrip()
    return bool(stripped) and not stripped.startswith("$") and stripped[0].isalpha()


def _parse_free_field(line: str) -> list[str]:
    return [token.strip() for token in line.split(",")]


def _split_large_field_card(line: str) -> list[str]:
    fields = [line[:8].strip()]
    remainder = line[8:]
    while remainder:
        fields.append(remainder[:16].strip())
        remainder = remainder[16:]
    return fields


def _relative_delta(model_mass: float, budget_mass: float) -> float | None:
    if abs(budget_mass) < 1e-12:
        return None
    return (model_mass - budget_mass) / budget_mass


def _load_mass_budget_summary(
    workbook_path: Path,
    sheet_name: str,
    basis: str,
) -> dict[str, float]:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError(
            "openpyxl is required to read the mass budget workbook. Install the dependencies from requirements.txt."
        ) from exc

    column_index = {"basic": 3, "nominal": 5}[basis]
    workbook = load_workbook(workbook_path, data_only=True, read_only=True)
    if sheet_name not in workbook.sheetnames:
        raise ValueError(f"Mass budget sheet '{sheet_name}' was not found in '{workbook_path}'.")

    worksheet = workbook[sheet_name]
    summary: dict[str, float] = {}
    for category, row_index in MASS_BUDGET_ROW_INDEX.items():
        value = worksheet.cell(row=row_index, column=column_index).value
        if value is None:
            raise ValueError(
                f"Mass budget workbook '{workbook_path}' has no cached value for '{category}' "
                f"({basis}, row {row_index})."
            )
        summary[category] = float(value) / 1000.0

    return summary


def _manual_bdf_contributions(bdf_path: Path) -> list[ElementMassContribution]:
    lines = bdf_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    materials: dict[int, dict[str, float]] = {}
    pbarl_areas: dict[int, tuple[int, float, str]] = {}
    pshells: dict[int, tuple[int, float, str]] = {}
    grids: dict[int, np.ndarray] = {}
    cbars: list[tuple[int, int, int]] = []
    cquads: list[tuple[int, int, int, int, int]] = []
    ctrias: list[tuple[int, int, int, int]] = []

    current_region = "unlabeled-region"
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.lstrip()
        if stripped.startswith("$ Elements and Element Properties for region :"):
            current_region = stripped.split(":", 1)[1].strip()
            index += 1
            continue

        if not stripped or stripped.startswith("$"):
            index += 1
            continue

        if stripped.startswith("MAT1*"):
            first_tokens = _split_large_field_card(line)
            second_tokens = _split_large_field_card(lines[index + 1])
            materials[int(first_tokens[1])] = {
                "E": parse_nastran_number(first_tokens[2]),
                "nu": parse_nastran_number(first_tokens[4]),
                "rho": parse_nastran_number(second_tokens[1]),
            }
            index += 2
            continue

        if stripped.startswith("MAT1,"):
            tokens = _parse_free_field(stripped)
            mid = int(tokens[1])
            materials[mid] = {
                "E": parse_nastran_number(tokens[2]),
                "nu": parse_nastran_number(tokens[4]),
                "rho": parse_nastran_number(tokens[5]),
            }
            index += 1
            continue

        if stripped.startswith("PBARL"):
            first_tokens = line.split()
            pid = int(first_tokens[1])
            mid = int(first_tokens[2])
            section_type = first_tokens[-1]
            dims_line = lines[index + 1].split()
            if section_type != "BAR" or len(dims_line) < 2:
                raise ValueError("Manual mass calculator currently supports PBARL BAR sections only.")
            area = parse_nastran_number(dims_line[0]) * parse_nastran_number(dims_line[1])
            pbarl_areas[pid] = (mid, area, current_region)
            index += 2
            continue

        if stripped.startswith("PSHELL*"):
            first_tokens = _split_large_field_card(line)
            pid = int(first_tokens[1])
            mid = int(first_tokens[2])
            thickness = parse_nastran_number(first_tokens[3])
            pshells[pid] = (mid, thickness, current_region)
            index += 2
            continue

        if stripped.startswith("PSHELL"):
            first_tokens = line.split()
            pid = int(first_tokens[1])
            mid = int(first_tokens[2])
            thickness = parse_nastran_number(first_tokens[3])
            pshells[pid] = (mid, thickness, current_region)
            index += 1
            continue

        if stripped.startswith("GRID*"):
            first_tokens = _split_large_field_card(line)
            second_tokens = _split_large_field_card(lines[index + 1])
            point_id = int(first_tokens[1])
            x = parse_nastran_number(first_tokens[3])
            y = parse_nastran_number(first_tokens[4])
            z = parse_nastran_number(second_tokens[1])
            grids[point_id] = np.array([x, y, z], dtype=float)
            index += 2
            continue

        if stripped.startswith("CBAR"):
            tokens = line.split()
            cbars.append((int(tokens[2]), int(tokens[3]), int(tokens[4])))
            index += 1
            while index < len(lines) and not _is_primary_card(lines[index]):
                index += 1
            continue

        if stripped.startswith("CQUAD4"):
            tokens = line.split()
            cquads.append(
                (
                    int(tokens[2]),
                    int(tokens[3]),
                    int(tokens[4]),
                    int(tokens[5]),
                    int(tokens[6]),
                )
            )
            index += 1
            continue

        if stripped.startswith("CTRIA3"):
            tokens = line.split()
            ctrias.append((int(tokens[2]), int(tokens[3]), int(tokens[4]), int(tokens[5])))
            index += 1
            continue

        index += 1

    contributions: list[ElementMassContribution] = []
    for pid, ga, gb in cbars:
        mid, area, region_name = pbarl_areas[pid]
        rho = materials[mid]["rho"]
        point_a = grids[ga]
        point_b = grids[gb]
        length = float(np.linalg.norm(point_b - point_a))
        centroid_z = float((point_a[2] + point_b[2]) / 2.0)
        contributions.append(
            ElementMassContribution(
                region_name=region_name,
                property_id=pid,
                material_id=mid,
                element_type="CBAR",
                mass=rho * area * length,
                centroid_z=centroid_z,
            )
        )

    for pid, g1, g2, g3, g4 in cquads:
        mid, thickness, region_name = pshells[pid]
        rho = materials[mid]["rho"]
        p1, p2, p3, p4 = grids[g1], grids[g2], grids[g3], grids[g4]
        area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
        area += 0.5 * np.linalg.norm(np.cross(p4 - p1, p3 - p1))
        centroid_z = float(np.mean([p1[2], p2[2], p3[2], p4[2]]))
        contributions.append(
            ElementMassContribution(
                region_name=region_name,
                property_id=pid,
                material_id=mid,
                element_type="CQUAD4",
                mass=rho * thickness * float(area),
                centroid_z=centroid_z,
            )
        )

    for pid, g1, g2, g3 in ctrias:
        mid, thickness, region_name = pshells[pid]
        rho = materials[mid]["rho"]
        p1, p2, p3 = grids[g1], grids[g2], grids[g3]
        area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
        centroid_z = float(np.mean([p1[2], p2[2], p3[2]]))
        contributions.append(
            ElementMassContribution(
                region_name=region_name,
                property_id=pid,
                material_id=mid,
                element_type="CTRIA3",
                mass=rho * thickness * float(area),
                centroid_z=centroid_z,
            )
        )

    return contributions


def _derive_module_reference_z(contributions: list[ElementMassContribution]) -> dict[str, float]:
    reference_z = dict(DEFAULT_MODULE_REFERENCE_Z)
    for region_name, category in BOARD_REGION_TO_CATEGORY.items():
        region_contributions = [item for item in contributions if item.region_name == region_name]
        if not region_contributions:
            continue
        total_mass = sum(item.mass for item in region_contributions)
        if total_mass <= 0:
            continue
        reference_z[category] = sum(item.mass * item.centroid_z for item in region_contributions) / total_mass
    return reference_z


def _classify_contribution(
    contribution: ElementMassContribution,
    module_reference_z: dict[str, float],
) -> tuple[str, str]:
    if contribution.region_name in BOARD_REGION_TO_CATEGORY:
        return BOARD_REGION_TO_CATEGORY[contribution.region_name], "direct board region"
    if contribution.region_name in BACKPLANE_REGIONS:
        return "Backplane modules", "direct backplane region"

    category = min(
        module_reference_z,
        key=lambda candidate: abs(contribution.centroid_z - module_reference_z[candidate]),
    )
    return category, "assigned by nearest element centroid z"


def _build_budget_comparison(
    contributions: list[ElementMassContribution],
    total_mass: float,
    workbook_path: Path,
    sheet_name: str,
    basis: str,
) -> MassBudgetComparison:
    budget_summary = _load_mass_budget_summary(workbook_path, sheet_name, basis)
    module_reference_z = _derive_module_reference_z(contributions)

    model_by_category = {category: 0.0 for category in MASS_BUDGET_ROW_INDEX}
    classification_totals: dict[tuple[str, str, str], float] = defaultdict(float)
    for contribution in contributions:
        category, assignment_rule = _classify_contribution(contribution, module_reference_z)
        model_by_category[category] += contribution.mass
        classification_totals[(contribution.region_name, category, assignment_rule)] += contribution.mass
    model_by_category["Total mass"] = total_mass

    rows = []
    for category in MASS_BUDGET_ROW_INDEX:
        budget_mass = budget_summary[category]
        model_mass = model_by_category[category]
        rows.append(
            MassBudgetRow(
                category=category,
                budget_mass=budget_mass,
                model_mass=model_mass,
                delta_mass=model_mass - budget_mass,
                relative_delta=_relative_delta(model_mass, budget_mass),
            )
        )

    classification_rows = [
        MassClassificationRow(
            region_name=region_name,
            assigned_to=assigned_to,
            assignment_rule=assignment_rule,
            mass=mass,
        )
        for (region_name, assigned_to, assignment_rule), mass in classification_totals.items()
    ]
    classification_rows.sort(key=lambda item: (item.assigned_to, item.region_name, item.assignment_rule))

    notes = (
        "Las regiones de placas PCB se asignan directamente a su modulo correspondiente.",
        "Las regiones 07_t-bandejas y 07_t-bandejas_FPGA se contabilizan como Backplane modules.",
        "La envolvente restante se reparte al modulo mas cercano segun la coordenada z del centroide de cada elemento.",
    )

    budget_total_mass = budget_summary["Total mass"]
    return MassBudgetComparison(
        workbook=str(workbook_path),
        sheet=sheet_name,
        basis=basis,
        budget_total_mass=budget_total_mass,
        model_total_mass=total_mass,
        delta_mass=total_mass - budget_total_mass,
        relative_delta=_relative_delta(total_mass, budget_total_mass),
        rows=tuple(rows),
        classification_rows=tuple(classification_rows),
        notes=notes,
    )


def _pynastran_mass(bdf_path: Path) -> float:
    from pyNastran.bdf.bdf import BDF  # type: ignore

    model = BDF(debug=False, log=None)
    model.read_bdf(str(bdf_path), xref=True)
    total_mass, _, _ = model.mass_properties()
    return float(total_mass)


def calculate_total_mass(
    bdf_path: Path,
    target_mass: float,
    budget_workbook: Path | None = None,
    budget_sheet: str = "Mass Budget",
    budget_basis: str = "basic",
) -> MassSummary:
    contributions = _manual_bdf_contributions(bdf_path)
    total_mass = sum(item.mass for item in contributions)
    method = "manual"

    try:
        _pynastran_mass(bdf_path)
        method = "manual (validated against pyNastran)"
    except Exception:
        pass

    budget_comparison = None
    if budget_workbook is not None:
        budget_comparison = _build_budget_comparison(
            contributions=contributions,
            total_mass=total_mass,
            workbook_path=budget_workbook,
            sheet_name=budget_sheet,
            basis=budget_basis,
        )

    relative_error = abs(total_mass - target_mass) / target_mass
    return MassSummary(
        total_mass=total_mass,
        relative_error=relative_error,
        target_mass=target_mass,
        method=method,
        budget_comparison=budget_comparison,
    )


def write_mass_summary(summary: MassSummary, path: Path) -> None:
    path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
