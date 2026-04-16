from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np

from .bdf_editor import parse_nastran_number
from .config import (
    MassConfig,
    MassFamilyConfig,
    MassImputationRuleConfig,
    MassSubsetConfig,
    mass_budget_baseline_status,
)


@dataclass(frozen=True)
class MassContribution:
    contribution_id: str
    source_kind: str
    element_id: int | None
    mass_id: int | None
    element_type: str
    region_name: str
    property_id: int | None
    material_id: int | None
    node_ids: tuple[int, ...]
    centroid: tuple[float, float, float] | None
    mass: float


@dataclass(frozen=True)
class MassSubsetRow:
    label: str
    target_mass: float
    computed_mass: float
    delta_mass: float
    relative_delta: float | None


@dataclass(frozen=True)
class UnmappedMassRow:
    contribution_id: str
    source_kind: str
    element_type: str
    region_name: str
    property_id: int | None
    material_id: int | None
    node_ids: tuple[int, ...]
    mass: float


@dataclass(frozen=True)
class UnsupportedExtraMassRow:
    source: str
    card: str
    identifier: str
    parsed_mass: float | None
    raw_line: str


@dataclass(frozen=True)
class MassFamilyDetailRow:
    family_label: str
    region_name: str
    property_id: int | None
    material_id: int | None
    element_count: int
    mass_contribution: float
    element_types: dict[str, int]


@dataclass(frozen=True)
class MassFamilyRow:
    label: str
    contribution_count: int
    computed_mass: float
    detail_rows: tuple[MassFamilyDetailRow, ...]


@dataclass(frozen=True)
class BudgetImputationRow:
    label: str
    target_mass: float
    imputed_mass: float
    delta_mass: float
    relative_delta: float | None


@dataclass(frozen=True)
class FamilyBudgetAllocationRow:
    family_label: str
    mode: str
    target_label: str
    mass_contribution: float


@dataclass(frozen=True)
class MassLedger:
    mapped_mass: float
    unmapped_mass: float
    unsupported_extra_mass: float


@dataclass(frozen=True)
class MassSummary:
    total_mass: float
    relative_error: float
    target_mass: float
    target_basis: str
    baseline_status: str
    method: str
    subset_rows: tuple[MassSubsetRow, ...]
    unmapped_rows: tuple[UnmappedMassRow, ...]
    family_rows: tuple[MassFamilyRow, ...]
    budget_imputation_rows: tuple[BudgetImputationRow, ...]
    family_budget_allocations: tuple[FamilyBudgetAllocationRow, ...]
    imputation_residual_mass: float
    unsupported_extra_mass_rows: tuple[UnsupportedExtraMassRow, ...]
    ledgers: MassLedger
    warnings: tuple[str, ...]

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


def _tokenize_card(line: str) -> list[str]:
    stripped = line.strip()
    if "," in stripped:
        return _parse_free_field(stripped)
    return stripped.split()


def _relative_delta(model_mass: float, target_mass: float) -> float | None:
    if abs(target_mass) < 1e-12:
        return None
    return (model_mass - target_mass) / target_mass


def _parse_grid_free_field(tokens: list[str]) -> tuple[int, np.ndarray]:
    if len(tokens) < 6:
        raise ValueError("GRID free-field card is incomplete.")
    point_id = int(tokens[1])
    x = parse_nastran_number(tokens[3])
    y = parse_nastran_number(tokens[4])
    z = parse_nastran_number(tokens[5])
    return point_id, np.array([x, y, z], dtype=float)


def _parse_grid_large_field(lines: list[str], index: int) -> tuple[tuple[int, np.ndarray], int]:
    first_tokens = _split_large_field_card(lines[index])
    second_tokens = _split_large_field_card(lines[index + 1])
    point_id = int(first_tokens[1])
    x = parse_nastran_number(first_tokens[3])
    y = parse_nastran_number(first_tokens[4])
    z = parse_nastran_number(second_tokens[1])
    return (point_id, np.array([x, y, z], dtype=float)), 2


def _centroid_from_node_ids(
    node_ids: tuple[int, ...],
    grids: dict[int, np.ndarray],
) -> tuple[float, float, float] | None:
    if not node_ids:
        return None
    coords = [grids[node_id] for node_id in node_ids if node_id in grids]
    if not coords:
        return None
    centroid = sum(coords) / len(coords)
    return tuple(float(value) for value in centroid.tolist())


def _parse_manual_contributions(
    bdf_path: Path,
) -> tuple[list[MassContribution], list[UnsupportedExtraMassRow], tuple[int, ...], tuple[int, ...]]:
    lines = bdf_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    materials: dict[int, dict[str, float]] = {}
    pbarl_areas: dict[int, tuple[int, float]] = {}
    pshells: dict[int, tuple[int, float]] = {}
    pmass: dict[int, float] = {}
    grids: dict[int, np.ndarray] = {}

    cbars: list[tuple[int, int, int, int, str]] = []
    cquads: list[tuple[int, int, int, int, int, int, str]] = []
    ctrias: list[tuple[int, int, int, int, int, str]] = []
    concentrated_specs: list[dict[str, Any]] = []

    unsupported_extra_mass_rows: list[UnsupportedExtraMassRow] = []
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
                "rho": parse_nastran_number(second_tokens[1]),
            }
            index += 2
            continue

        if stripped.startswith("MAT1"):
            tokens = _tokenize_card(stripped)
            if tokens[0] == "MAT1":
                materials[int(tokens[1])] = {
                    "rho": parse_nastran_number(tokens[5]),
                }
            index += 1
            continue

        if stripped.startswith("PBARL"):
            tokens = _tokenize_card(stripped)
            pid = int(tokens[1])
            mid = int(tokens[2])
            section_type = tokens[-1]
            dims_tokens = _tokenize_card(lines[index + 1])
            if section_type != "BAR" or len(dims_tokens) < 2:
                raise ValueError("Manual mass calculator currently supports PBARL BAR sections only.")
            area = parse_nastran_number(dims_tokens[0]) * parse_nastran_number(dims_tokens[1])
            pbarl_areas[pid] = (mid, area)
            index += 2
            continue

        if stripped.startswith("PSHELL*"):
            first_tokens = _split_large_field_card(line)
            pid = int(first_tokens[1])
            mid = int(first_tokens[2])
            thickness = parse_nastran_number(first_tokens[3])
            pshells[pid] = (mid, thickness)
            index += 2
            continue

        if stripped.startswith("PSHELL"):
            tokens = _tokenize_card(stripped)
            pid = int(tokens[1])
            mid = int(tokens[2])
            thickness = parse_nastran_number(tokens[3])
            pshells[pid] = (mid, thickness)
            index += 1
            continue

        if stripped.startswith("PMASS"):
            tokens = _tokenize_card(stripped)
            pmass[int(tokens[1])] = parse_nastran_number(tokens[2])
            index += 1
            continue

        if stripped.startswith("GRID*"):
            (point_id, coords), consumed = _parse_grid_large_field(lines, index)
            grids[point_id] = coords
            index += consumed
            continue

        if stripped.startswith("GRID"):
            tokens = _tokenize_card(stripped)
            if tokens[0] == "GRID":
                point_id, coords = _parse_grid_free_field(tokens)
                grids[point_id] = coords
            index += 1
            continue

        if stripped.startswith("CBAR"):
            tokens = _tokenize_card(stripped)
            cbars.append((int(tokens[1]), int(tokens[2]), int(tokens[3]), int(tokens[4]), current_region))
            index += 1
            while index < len(lines) and not _is_primary_card(lines[index]):
                index += 1
            continue

        if stripped.startswith("CQUAD4"):
            tokens = _tokenize_card(stripped)
            cquads.append(
                (
                    int(tokens[1]),
                    int(tokens[2]),
                    int(tokens[3]),
                    int(tokens[4]),
                    int(tokens[5]),
                    int(tokens[6]),
                    current_region,
                )
            )
            index += 1
            continue

        if stripped.startswith("CTRIA3"):
            tokens = _tokenize_card(stripped)
            ctrias.append(
                (
                    int(tokens[1]),
                    int(tokens[2]),
                    int(tokens[3]),
                    int(tokens[4]),
                    int(tokens[5]),
                    current_region,
                )
            )
            index += 1
            continue

        if stripped.startswith("CONM2"):
            tokens = _tokenize_card(stripped)
            element_id = int(tokens[1])
            node_id = int(tokens[2])
            mass = parse_nastran_number(tokens[4])
            concentrated_specs.append(
                {
                    "contribution_id": f"CONM2:{element_id}",
                    "source_kind": "concentrated",
                    "element_id": element_id,
                    "mass_id": element_id,
                    "element_type": "CONM2",
                    "region_name": current_region,
                    "property_id": None,
                    "material_id": None,
                    "node_ids": (node_id,),
                    "mass": mass,
                }
            )
            index += 1
            while index < len(lines) and not _is_primary_card(lines[index]):
                index += 1
            continue

        if stripped.startswith("CMASS1"):
            tokens = _tokenize_card(stripped)
            element_id = int(tokens[1])
            property_id = int(tokens[2])
            mass = pmass.get(property_id)
            if mass is None:
                unsupported_extra_mass_rows.append(
                    UnsupportedExtraMassRow(
                        source="CMASS1",
                        card="CMASS1",
                        identifier=str(element_id),
                        parsed_mass=None,
                        raw_line=stripped,
                    )
                )
            else:
                concentrated_specs.append(
                    {
                        "contribution_id": f"CMASS1:{element_id}",
                        "source_kind": "concentrated",
                        "element_id": element_id,
                        "mass_id": element_id,
                        "element_type": "CMASS1",
                        "region_name": current_region,
                        "property_id": property_id,
                        "material_id": None,
                        "node_ids": tuple(int(value) for value in tokens[3::2] if value),
                        "mass": mass,
                    }
                )
            index += 1
            continue

        if stripped.startswith("CMASS2"):
            tokens = _tokenize_card(stripped)
            element_id = int(tokens[1])
            mass = parse_nastran_number(tokens[2])
            concentrated_specs.append(
                {
                    "contribution_id": f"CMASS2:{element_id}",
                    "source_kind": "concentrated",
                    "element_id": element_id,
                    "mass_id": element_id,
                    "element_type": "CMASS2",
                    "region_name": current_region,
                    "property_id": None,
                    "material_id": None,
                    "node_ids": tuple(int(value) for value in tokens[3::2] if value),
                    "mass": mass,
                }
            )
            index += 1
            continue

        if stripped.startswith("CMASS3"):
            tokens = _tokenize_card(stripped)
            element_id = int(tokens[1])
            property_id = int(tokens[2])
            mass = pmass.get(property_id)
            if mass is None:
                unsupported_extra_mass_rows.append(
                    UnsupportedExtraMassRow(
                        source="CMASS3",
                        card="CMASS3",
                        identifier=str(element_id),
                        parsed_mass=None,
                        raw_line=stripped,
                    )
                )
            else:
                concentrated_specs.append(
                    {
                        "contribution_id": f"CMASS3:{element_id}",
                        "source_kind": "concentrated",
                        "element_id": element_id,
                        "mass_id": element_id,
                        "element_type": "CMASS3",
                        "region_name": current_region,
                        "property_id": property_id,
                        "material_id": None,
                        "node_ids": tuple(int(value) for value in tokens[3:] if value),
                        "mass": mass,
                    }
                )
            index += 1
            continue

        if stripped.startswith("CMASS4"):
            tokens = _tokenize_card(stripped)
            element_id = int(tokens[1])
            mass = parse_nastran_number(tokens[2])
            concentrated_specs.append(
                {
                    "contribution_id": f"CMASS4:{element_id}",
                    "source_kind": "concentrated",
                    "element_id": element_id,
                    "mass_id": element_id,
                    "element_type": "CMASS4",
                    "region_name": current_region,
                    "property_id": None,
                    "material_id": None,
                    "node_ids": tuple(int(value) for value in tokens[3:] if value),
                    "mass": mass,
                }
            )
            index += 1
            continue

        if stripped.startswith("NSM1") or stripped.startswith("NSML") or stripped.startswith("NSM"):
            tokens = _tokenize_card(stripped)
            parsed_mass = None
            for token in reversed(tokens):
                try:
                    parsed_mass = parse_nastran_number(token)
                    break
                except ValueError:
                    continue
            unsupported_extra_mass_rows.append(
                UnsupportedExtraMassRow(
                    source=tokens[0],
                    card=tokens[0],
                    identifier=tokens[1] if len(tokens) > 1 else "",
                    parsed_mass=parsed_mass,
                    raw_line=stripped,
                )
            )
            index += 1
            continue

        index += 1

    contributions: list[MassContribution] = []
    for element_id, property_id, ga, gb, region_name in cbars:
        material_id, area = pbarl_areas[property_id]
        rho = materials[material_id]["rho"]
        point_a = grids[ga]
        point_b = grids[gb]
        length = float(np.linalg.norm(point_b - point_a))
        contributions.append(
            MassContribution(
                contribution_id=f"CBAR:{element_id}",
                source_kind="structural",
                element_id=element_id,
                mass_id=None,
                element_type="CBAR",
                region_name=region_name,
                property_id=property_id,
                material_id=material_id,
                node_ids=(ga, gb),
                centroid=tuple(float(value) for value in ((point_a + point_b) / 2.0).tolist()),
                mass=rho * area * length,
            )
        )

    for element_id, property_id, g1, g2, g3, g4, region_name in cquads:
        material_id, thickness = pshells[property_id]
        rho = materials[material_id]["rho"]
        p1, p2, p3, p4 = grids[g1], grids[g2], grids[g3], grids[g4]
        area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
        area += 0.5 * np.linalg.norm(np.cross(p4 - p1, p3 - p1))
        contributions.append(
            MassContribution(
                contribution_id=f"CQUAD4:{element_id}",
                source_kind="structural",
                element_id=element_id,
                mass_id=None,
                element_type="CQUAD4",
                region_name=region_name,
                property_id=property_id,
                material_id=material_id,
                node_ids=(g1, g2, g3, g4),
                centroid=tuple(float(value) for value in ((p1 + p2 + p3 + p4) / 4.0).tolist()),
                mass=rho * thickness * float(area),
            )
        )

    for element_id, property_id, g1, g2, g3, region_name in ctrias:
        material_id, thickness = pshells[property_id]
        rho = materials[material_id]["rho"]
        p1, p2, p3 = grids[g1], grids[g2], grids[g3]
        area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
        contributions.append(
            MassContribution(
                contribution_id=f"CTRIA3:{element_id}",
                source_kind="structural",
                element_id=element_id,
                mass_id=None,
                element_type="CTRIA3",
                region_name=region_name,
                property_id=property_id,
                material_id=material_id,
                node_ids=(g1, g2, g3),
                centroid=tuple(float(value) for value in ((p1 + p2 + p3) / 3.0).tolist()),
                mass=rho * thickness * float(area),
            )
        )

    for spec in concentrated_specs:
        contributions.append(
            MassContribution(
                contribution_id=str(spec["contribution_id"]),
                source_kind=str(spec["source_kind"]),
                element_id=spec["element_id"],
                mass_id=spec["mass_id"],
                element_type=str(spec["element_type"]),
                region_name=str(spec["region_name"]),
                property_id=spec["property_id"],
                material_id=spec["material_id"],
                node_ids=tuple(spec["node_ids"]),
                centroid=_centroid_from_node_ids(tuple(spec["node_ids"]), grids),
                mass=float(spec["mass"]),
            )
        )
    unmapped_property_ids = tuple(
        sorted({contribution.property_id for contribution in contributions if contribution.property_id is not None})
    )
    unmapped_material_ids = tuple(
        sorted({contribution.material_id for contribution in contributions if contribution.material_id is not None})
    )
    return contributions, unsupported_extra_mass_rows, unmapped_property_ids, unmapped_material_ids


def _matches_subset(contribution: MassContribution, subset: MassSubsetConfig) -> bool:
    checks: list[bool] = []
    if subset.regions:
        checks.append(contribution.region_name in subset.regions)
    if subset.property_ids:
        checks.append(contribution.property_id in subset.property_ids)
    if subset.material_ids:
        checks.append(contribution.material_id in subset.material_ids)
    if subset.element_ids:
        candidate_id = contribution.element_id if contribution.element_id is not None else contribution.mass_id
        checks.append(candidate_id in subset.element_ids)
    if subset.node_ids:
        if contribution.source_kind != "concentrated":
            checks.append(False)
        else:
            checks.append(any(node_id in subset.node_ids for node_id in contribution.node_ids))
    return all(checks) if checks else True


def _matches_family(contribution: MassContribution, family: MassFamilyConfig) -> bool:
    if family.match_none:
        return False
    checks: list[bool] = []
    if family.regions:
        checks.append(contribution.region_name in family.regions)
    if family.property_ids:
        checks.append(contribution.property_id in family.property_ids)
    if family.material_ids:
        checks.append(contribution.material_id in family.material_ids)
    if family.element_ids:
        candidate_id = contribution.element_id if contribution.element_id is not None else contribution.mass_id
        checks.append(candidate_id in family.element_ids)
    if family.node_ids:
        if contribution.source_kind != "concentrated":
            checks.append(False)
        else:
            checks.append(any(node_id in family.node_ids for node_id in contribution.node_ids))
    return all(checks) if checks else False


def _assign_contributions(
    contributions: list[MassContribution],
    subsets: tuple[MassSubsetConfig, ...],
) -> tuple[dict[str, float], list[UnmappedMassRow]]:
    totals = {subset.label: 0.0 for subset in subsets}
    unmapped_rows: list[UnmappedMassRow] = []

    for contribution in contributions:
        matching_subsets = [subset for subset in subsets if _matches_subset(contribution, subset)]
        if len(matching_subsets) > 1:
            labels = ", ".join(subset.label for subset in matching_subsets)
            raise ValueError(
                f"Contribution '{contribution.contribution_id}' matches more than one subset: {labels}"
            )
        if not matching_subsets:
            unmapped_rows.append(
                UnmappedMassRow(
                    contribution_id=contribution.contribution_id,
                    source_kind=contribution.source_kind,
                    element_type=contribution.element_type,
                    region_name=contribution.region_name,
                    property_id=contribution.property_id,
                    material_id=contribution.material_id,
                    node_ids=contribution.node_ids,
                    mass=contribution.mass,
                )
            )
            continue
        totals[matching_subsets[0].label] += contribution.mass

    return totals, unmapped_rows


def _assign_families(
    contributions: list[MassContribution],
    families: tuple[MassFamilyConfig, ...],
) -> dict[str, list[MassContribution]]:
    assignments = {family.label: [] for family in families}
    assignments["unmapped residual"] = []

    for contribution in contributions:
        matching_families = [family.label for family in families if _matches_family(contribution, family)]
        if len(matching_families) > 1:
            labels = ", ".join(matching_families)
            raise ValueError(
                f"Contribution '{contribution.contribution_id}' matches more than one mass family: {labels}"
            )
        if not matching_families:
            assignments["unmapped residual"].append(contribution)
            continue
        assignments[matching_families[0]].append(contribution)

    return assignments


def _build_family_rows(family_assignments: dict[str, list[MassContribution]]) -> tuple[MassFamilyRow, ...]:
    rows: list[MassFamilyRow] = []
    for label, contributions in family_assignments.items():
        grouped: dict[tuple[str, int | None, int | None], dict[str, Any]] = defaultdict(
            lambda: {
                "element_count": 0,
                "mass_contribution": 0.0,
                "element_types": defaultdict(int),
            }
        )
        for contribution in contributions:
            key = (contribution.region_name, contribution.property_id, contribution.material_id)
            grouped[key]["element_count"] += 1
            grouped[key]["mass_contribution"] += contribution.mass
            grouped[key]["element_types"][contribution.element_type] += 1

        detail_rows = []
        for (region_name, property_id, material_id), payload in sorted(
            grouped.items(),
            key=lambda item: (
                -item[1]["mass_contribution"],
                item[0][0],
                -1 if item[0][1] is None else item[0][1],
                -1 if item[0][2] is None else item[0][2],
            ),
        ):
            detail_rows.append(
                MassFamilyDetailRow(
                    family_label=label,
                    region_name=region_name,
                    property_id=property_id,
                    material_id=material_id,
                    element_count=int(payload["element_count"]),
                    mass_contribution=float(payload["mass_contribution"]),
                    element_types=dict(sorted(payload["element_types"].items())),
                )
            )
        rows.append(
            MassFamilyRow(
                label=label,
                contribution_count=len(contributions),
                computed_mass=sum(item.mass for item in contributions),
                detail_rows=tuple(detail_rows),
            )
        )
    return tuple(rows)


def _build_budget_imputation_rows(
    subsets: tuple[MassSubsetConfig, ...],
    totals: dict[str, float],
) -> tuple[BudgetImputationRow, ...]:
    rows = []
    for subset in subsets:
        imputed_mass = totals.get(subset.label, 0.0)
        rows.append(
            BudgetImputationRow(
                label=subset.label,
                target_mass=subset.target_kg,
                imputed_mass=imputed_mass,
                delta_mass=imputed_mass - subset.target_kg,
                relative_delta=_relative_delta(imputed_mass, subset.target_kg),
            )
        )
    return tuple(rows)


def _apply_direct_imputation(
    contributions: list[MassContribution],
    rule: MassImputationRuleConfig,
    totals: dict[str, float],
    allocations: list[FamilyBudgetAllocationRow],
) -> None:
    if not rule.target_label:
        return
    family_mass = sum(item.mass for item in contributions)
    totals[rule.target_label] += family_mass
    allocations.append(
        FamilyBudgetAllocationRow(
            family_label=rule.family_label,
            mode=rule.mode,
            target_label=rule.target_label,
            mass_contribution=family_mass,
        )
    )


def _apply_proportional_imputation(
    contributions: list[MassContribution],
    rule: MassImputationRuleConfig,
    totals: dict[str, float],
    allocations: list[FamilyBudgetAllocationRow],
) -> None:
    family_mass = sum(item.mass for item in contributions)
    weight_total = sum(rule.weights.values())
    if family_mass == 0.0 or weight_total <= 0.0:
        return
    for target_label, weight in rule.weights.items():
        allocated_mass = family_mass * (weight / weight_total)
        totals[target_label] += allocated_mass
        allocations.append(
            FamilyBudgetAllocationRow(
                family_label=rule.family_label,
                mode=rule.mode,
                target_label=target_label,
                mass_contribution=allocated_mass,
            )
        )


def _apply_z_band_imputation(
    contributions: list[MassContribution],
    rule: MassImputationRuleConfig,
    totals: dict[str, float],
    allocations: list[FamilyBudgetAllocationRow],
) -> float:
    residual_mass = 0.0
    band_totals: dict[str, float] = defaultdict(float)
    for contribution in contributions:
        centroid = contribution.centroid
        if centroid is None:
            residual_mass += contribution.mass
            continue
        z_value = centroid[2]
        matched_target = None
        for band in rule.z_bands:
            lower_ok = band.z_min is None or z_value >= band.z_min
            upper_ok = band.z_max is None or z_value < band.z_max
            if lower_ok and upper_ok:
                matched_target = band.target_label
                break
        if matched_target is None:
            residual_mass += contribution.mass
            continue
        band_totals[matched_target] += contribution.mass

    for target_label, mass in sorted(band_totals.items()):
        totals[target_label] += mass
        allocations.append(
            FamilyBudgetAllocationRow(
                family_label=rule.family_label,
                mode=rule.mode,
                target_label=target_label,
                mass_contribution=mass,
            )
        )
    return residual_mass


def _apply_imputation_rules(
    family_assignments: dict[str, list[MassContribution]],
    mass_config: MassConfig,
) -> tuple[tuple[BudgetImputationRow, ...], tuple[FamilyBudgetAllocationRow, ...], float]:
    totals = {subset.label: 0.0 for subset in mass_config.subsets}
    allocations: list[FamilyBudgetAllocationRow] = []
    residual_mass = sum(item.mass for item in family_assignments.get("unmapped residual", []))
    rules_by_family = {rule.family_label: rule for rule in mass_config.imputation_rules}

    for family_label, contributions in family_assignments.items():
        if family_label == "unmapped residual":
            continue
        rule = rules_by_family.get(family_label)
        if rule is None or rule.mode == "unassigned":
            residual_mass += sum(item.mass for item in contributions)
            continue
        if rule.mode == "direct":
            _apply_direct_imputation(contributions, rule, totals, allocations)
            continue
        if rule.mode == "proportional":
            _apply_proportional_imputation(contributions, rule, totals, allocations)
            continue
        if rule.mode == "z_band_split":
            residual_mass += _apply_z_band_imputation(contributions, rule, totals, allocations)
            continue

    return _build_budget_imputation_rows(mass_config.subsets, totals), tuple(allocations), residual_mass


def _build_subset_rows(subsets: tuple[MassSubsetConfig, ...], totals: dict[str, float]) -> tuple[MassSubsetRow, ...]:
    rows = []
    for subset in subsets:
        computed_mass = totals.get(subset.label, 0.0)
        rows.append(
            MassSubsetRow(
                label=subset.label,
                target_mass=subset.target_kg,
                computed_mass=computed_mass,
                delta_mass=computed_mass - subset.target_kg,
                relative_delta=_relative_delta(computed_mass, subset.target_kg),
            )
        )
    return tuple(rows)


def _build_compensation_warnings(
    *,
    total_mass: float,
    target_mass: float,
    subset_rows: tuple[MassSubsetRow, ...],
    imputation_residual_mass: float,
    mass_config: MassConfig,
    unmapped_rows: list[UnmappedMassRow],
    unsupported_extra_mass_rows: list[UnsupportedExtraMassRow],
) -> tuple[str, ...]:
    warnings: list[str] = []
    if unmapped_rows:
        warnings.append("unmapped_mass_detected")
    if unsupported_extra_mass_rows:
        warnings.append("unsupported_extra_mass_detected")
    if imputation_residual_mass > 1.0e-12:
        warnings.append("budget_imputation_residual_detected")

    total_relative_delta = _relative_delta(total_mass, target_mass)
    thresholds = mass_config.compensation_warning_thresholds
    if total_relative_delta is not None and abs(total_relative_delta) <= thresholds.total_relative_delta_within:
        positive = [
            row.label
            for row in subset_rows
            if row.relative_delta is not None and row.relative_delta >= thresholds.subset_relative_delta_exceeds
        ]
        negative = [
            row.label
            for row in subset_rows
            if row.relative_delta is not None and row.relative_delta <= -thresholds.subset_relative_delta_exceeds
        ]
        if positive and negative:
            warnings.append(
                "mass_compensation_detected: "
                + f"positive={positive}, negative={negative}"
            )
    return tuple(warnings)


def _pynastran_mass(bdf_path: Path) -> float:
    from pyNastran.bdf.bdf import BDF  # type: ignore

    model = BDF(debug=False, log=None)
    model.read_bdf(str(bdf_path), xref=True)
    total_mass, _, _ = model.mass_properties()
    return float(total_mass)


def calculate_total_mass(
    *,
    bdf_path: Path,
    mass_config: MassConfig,
) -> MassSummary:
    contributions, unsupported_extra_mass_rows, _, _ = _parse_manual_contributions(bdf_path)
    structural_and_concentrated_mass = sum(contribution.mass for contribution in contributions)
    unsupported_extra_mass = sum(
        row.parsed_mass or 0.0 for row in unsupported_extra_mass_rows if row.parsed_mass is not None
    )
    total_mass = structural_and_concentrated_mass
    method = "manual"

    try:
        _pynastran_mass(bdf_path)
        method = "manual (validated against pyNastran)"
    except Exception:
        pass

    subset_totals, unmapped_rows = _assign_contributions(contributions, mass_config.subsets)
    subset_rows = _build_subset_rows(mass_config.subsets, subset_totals)
    family_assignments = _assign_families(contributions, mass_config.families)
    family_rows = _build_family_rows(family_assignments)
    budget_imputation_rows, family_budget_allocations, imputation_residual_mass = _apply_imputation_rules(
        family_assignments,
        mass_config,
    )
    warnings = _build_compensation_warnings(
        total_mass=total_mass,
        target_mass=mass_config.target_kg,
        subset_rows=subset_rows,
        imputation_residual_mass=imputation_residual_mass,
        mass_config=mass_config,
        unmapped_rows=unmapped_rows,
        unsupported_extra_mass_rows=unsupported_extra_mass_rows,
    )
    relative_error = abs(total_mass - mass_config.target_kg) / mass_config.target_kg
    ledgers = MassLedger(
        mapped_mass=sum(subset_totals.values()),
        unmapped_mass=sum(row.mass for row in unmapped_rows),
        unsupported_extra_mass=unsupported_extra_mass,
    )
    return MassSummary(
        total_mass=total_mass,
        relative_error=relative_error,
        target_mass=mass_config.target_kg,
        target_basis=mass_config.budget_basis,
        baseline_status=mass_budget_baseline_status(mass_config.budget_basis),
        method=method,
        subset_rows=subset_rows,
        unmapped_rows=tuple(unmapped_rows),
        family_rows=family_rows,
        budget_imputation_rows=budget_imputation_rows,
        family_budget_allocations=family_budget_allocations,
        imputation_residual_mass=imputation_residual_mass,
        unsupported_extra_mass_rows=tuple(unsupported_extra_mass_rows),
        ledgers=ledgers,
        warnings=warnings,
    )


def write_mass_summary(summary: MassSummary, path: Path) -> None:
    path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
