from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .model_checks import DeckCheckResult, inspect_bdf_structure
from .parameter_registry import build_parameter_registry

try:
    from pyNastran.bdf.bdf import BDF
except ImportError:  # pragma: no cover
    BDF = None


ELEMENT_CARD_TYPES = {
    "CBAR",
    "CBEAM",
    "CBUSH",
    "CDAMP1",
    "CDAMP2",
    "CELAS1",
    "CELAS2",
    "CHEXA",
    "CONM2",
    "CPENTA",
    "CQUAD4",
    "CROD",
    "CTETRA",
    "CTRIA3",
    "RBE2",
}


@dataclass(slots=True)
class BdfInspectionResult:
    source_path: Path
    parser_backend: str
    structure: DeckCheckResult
    card_counts: dict[str, int]
    constraint_counts: dict[str, int]
    node_count: int
    element_count: int
    element_breakdown: dict[str, int]
    editable_parameters: dict[str, list[dict[str, Any]]]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_path"] = str(self.source_path)
        payload["structure"] = self.structure.as_dict()
        return payload

    def summary_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for key, value in self.card_counts.items():
            rows.append({"section": "card_counts", "item": key, "value": value})
        for key, value in self.constraint_counts.items():
            rows.append({"section": "constraint_counts", "item": key, "value": value})
        rows.append({"section": "totals", "item": "GRID", "value": self.node_count})
        rows.append({"section": "totals", "item": "elements_total", "value": self.element_count})
        for key, values in self.editable_parameters.items():
            rows.append({"section": "editable_parameters", "item": key, "value": len(values)})
        return rows


def _extract_card_name(line: str) -> str:
    if "," in line:
        return line.split(",", 1)[0].strip().strip("*+").upper()
    return line[:8].strip().strip("*+").upper()


def _iter_card_names(text: str) -> list[str]:
    current_name = None
    names: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("$"):
            continue
        first = line[:8].strip()
        is_continuation = first in {"", "*", "+"} or line.lstrip().startswith(("*", "+"))
        if is_continuation and current_name is not None:
            continue
        current_name = _extract_card_name(line)
        if current_name:
            names.append(current_name)
    return names


def _manual_inspection(path: Path, structure: DeckCheckResult) -> BdfInspectionResult:
    text = path.read_text(encoding="utf-8", errors="ignore")
    names = _iter_card_names(text)
    counts = Counter(names)
    constraint_counts = {
        "SPC": counts.get("SPC", 0),
        "SPC1": counts.get("SPC1", 0),
        "SPCADD": counts.get("SPCADD", 0),
    }
    element_breakdown = {name: count for name, count in counts.items() if name in ELEMENT_CARD_TYPES}
    warnings = [
        "pyNastran no esta disponible; se usa un parser textual simplificado.",
        "El registro editable puede ser parcial sin pyNastran.",
    ]
    return BdfInspectionResult(
        source_path=path,
        parser_backend="text_fallback",
        structure=structure,
        card_counts={
            "MAT1": counts.get("MAT1", 0),
            "PSHELL": counts.get("PSHELL", 0),
            "PBARL": counts.get("PBARL", 0),
            "PBUSH": counts.get("PBUSH", 0),
            "RBE2": counts.get("RBE2", 0),
        },
        constraint_counts=constraint_counts,
        node_count=counts.get("GRID", 0),
        element_count=sum(element_breakdown.values()),
        element_breakdown=element_breakdown,
        editable_parameters={
            "materials_mat1": [],
            "shell_thickness_pshell": [],
            "beam_sections_pbarl": [],
            "bushing_stiffness_pbush": [],
        },
        warnings=warnings,
    )


def _load_pynastran_model(path: Path, structure: DeckCheckResult) -> Any:
    model = BDF(debug=False, log=None)
    model.read_bdf(str(path), xref=False, validate=False, punch=structure.should_wrap)
    return model


def inspect_bdf(path: Path) -> BdfInspectionResult:
    structure = inspect_bdf_structure(path)
    if BDF is None:
        return _manual_inspection(path, structure)

    try:
        model = _load_pynastran_model(path, structure)
    except Exception as exc:
        fallback = _manual_inspection(path, structure)
        fallback.warnings.append(f"pyNastran fallo y se uso fallback textual: {exc}")
        return fallback

    properties = list(getattr(model, "properties", {}).values())
    rigid_elements = list(getattr(model, "rigid_elements", {}).values())
    standard_elements = list(getattr(model, "elements", {}).values())

    card_counts = {
        "MAT1": sum(1 for item in getattr(model, "materials", {}).values() if item.type == "MAT1"),
        "PSHELL": sum(1 for item in properties if item.type == "PSHELL"),
        "PBARL": sum(1 for item in properties if item.type == "PBARL"),
        "PBUSH": sum(1 for item in properties if item.type == "PBUSH"),
        "RBE2": sum(1 for item in rigid_elements if item.type == "RBE2"),
    }

    constraint_counts_counter: Counter[str] = Counter()
    for entries in getattr(model, "spcs", {}).values():
        for entry in entries:
            constraint_counts_counter[getattr(entry, "type", "SPC")] += 1
    for entry in getattr(model, "spcadds", {}).values():
        constraint_counts_counter[getattr(entry, "type", "SPCADD")] += 1

    element_breakdown_counter: Counter[str] = Counter()
    for item in standard_elements:
        element_breakdown_counter[item.type] += 1
    for item in rigid_elements:
        element_breakdown_counter[item.type] += 1

    return BdfInspectionResult(
        source_path=path,
        parser_backend="pyNastran",
        structure=structure,
        card_counts=card_counts,
        constraint_counts=dict(constraint_counts_counter),
        node_count=len(getattr(model, "nodes", {})),
        element_count=len(standard_elements) + len(rigid_elements),
        element_breakdown=dict(sorted(element_breakdown_counter.items())),
        editable_parameters=build_parameter_registry(model),
        warnings=list(structure.messages),
    )
