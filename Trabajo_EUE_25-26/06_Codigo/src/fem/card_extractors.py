from __future__ import annotations

from typing import Any


def _to_number(value: Any) -> Any:
    if value in ("", None):
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("D", "E").replace("d", "e")
    try:
        if any(marker in text for marker in (".", "E", "e")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _raw_fields(card: Any) -> list[Any]:
    if hasattr(card, "raw_fields"):
        return list(card.raw_fields())
    if isinstance(card, dict):
        return list(card.get("raw_fields", []))
    return []


def extract_mat1(card: Any) -> dict[str, Any]:
    fields = _raw_fields(card)
    return {
        "family": "materials_mat1",
        "card_type": "MAT1",
        "id": _to_number(fields[1]) if len(fields) > 1 else None,
        "fields": {
            "E": _to_number(fields[2]) if len(fields) > 2 else None,
            "G": _to_number(fields[3]) if len(fields) > 3 else None,
            "nu": _to_number(fields[4]) if len(fields) > 4 else None,
            "rho": _to_number(fields[5]) if len(fields) > 5 else None,
        },
    }


def extract_pshell(card: Any) -> dict[str, Any]:
    fields = _raw_fields(card)
    return {
        "family": "shell_thickness_pshell",
        "card_type": "PSHELL",
        "id": _to_number(fields[1]) if len(fields) > 1 else None,
        "fields": {
            "mid1": _to_number(fields[2]) if len(fields) > 2 else None,
            "thickness": _to_number(fields[3]) if len(fields) > 3 else None,
            "mid2": _to_number(fields[4]) if len(fields) > 4 else None,
            "mid3": _to_number(fields[6]) if len(fields) > 6 else None,
            "nsm": _to_number(fields[8]) if len(fields) > 8 else None,
        },
    }


def extract_pbarl(card: Any) -> dict[str, Any]:
    fields = _raw_fields(card)
    dims = [_to_number(value) for value in fields[5:] if value not in (None, "")]
    return {
        "family": "beam_sections_pbarl",
        "card_type": "PBARL",
        "id": _to_number(fields[1]) if len(fields) > 1 else None,
        "fields": {
            "mid": _to_number(fields[2]) if len(fields) > 2 else None,
            "group": fields[3] if len(fields) > 3 else None,
            "section_type": fields[4] if len(fields) > 4 else None,
            "dimensions": dims,
        },
    }


def extract_pbush(card: Any) -> dict[str, Any]:
    fields = _raw_fields(card)
    section = None
    stiffness: list[Any] = []
    for value in fields[2:]:
        if isinstance(value, str) and value.strip():
            section = value.strip().upper()
            continue
        if section == "K":
            stiffness.append(_to_number(value))
    return {
        "family": "bushing_stiffness_pbush",
        "card_type": "PBUSH",
        "id": _to_number(fields[1]) if len(fields) > 1 else None,
        "fields": {
            "stiffness": stiffness,
        },
    }
