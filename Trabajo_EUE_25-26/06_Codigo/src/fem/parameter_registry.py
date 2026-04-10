from __future__ import annotations

from collections import defaultdict
from typing import Any

from .card_extractors import extract_mat1, extract_pbarl, extract_pbush, extract_pshell


def build_parameter_registry(model: Any) -> dict[str, list[dict[str, Any]]]:
    registry: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for material in getattr(model, "materials", {}).values():
        if getattr(material, "type", "").upper() == "MAT1":
            registry["materials_mat1"].append(extract_mat1(material))

    for prop in getattr(model, "properties", {}).values():
        prop_type = getattr(prop, "type", "").upper()
        if prop_type == "PSHELL":
            registry["shell_thickness_pshell"].append(extract_pshell(prop))
        elif prop_type == "PBARL":
            registry["beam_sections_pbarl"].append(extract_pbarl(prop))
        elif prop_type == "PBUSH":
            registry["bushing_stiffness_pbush"].append(extract_pbush(prop))

    for values in registry.values():
        values.sort(key=lambda item: (item.get("id") is None, item.get("id")))

    return dict(registry)
