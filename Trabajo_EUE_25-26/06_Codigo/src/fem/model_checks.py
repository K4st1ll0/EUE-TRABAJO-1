from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re


TOKEN_PATTERNS = {
    "sol": re.compile(r"^\s*SOL\b", re.MULTILINE),
    "cend": re.compile(r"^\s*CEND\b", re.MULTILINE),
    "begin_bulk": re.compile(r"^\s*BEGIN\s+BULK\b", re.MULTILINE),
    "enddata": re.compile(r"^\s*ENDDATA\b", re.MULTILINE),
}


@dataclass(slots=True)
class DeckCheckResult:
    source_path: Path
    has_sol: bool
    has_cend: bool
    has_begin_bulk: bool
    has_enddata: bool
    is_bulk_only: bool
    is_executable_deck: bool
    should_wrap: bool
    decision: str
    messages: list[str]

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["source_path"] = str(self.source_path)
        return payload


def inspect_bdf_structure(path: Path) -> DeckCheckResult:
    text = path.read_text(encoding="utf-8", errors="ignore")
    has_sol = bool(TOKEN_PATTERNS["sol"].search(text))
    has_cend = bool(TOKEN_PATTERNS["cend"].search(text))
    has_begin_bulk = bool(TOKEN_PATTERNS["begin_bulk"].search(text))
    has_enddata = bool(TOKEN_PATTERNS["enddata"].search(text))
    has_any_control_token = any([has_sol, has_cend, has_begin_bulk, has_enddata])
    is_bulk_only = not has_any_control_token
    is_executable_deck = has_sol and has_cend and has_begin_bulk
    should_wrap = is_bulk_only

    messages: list[str] = []
    if is_bulk_only:
        messages.append("El BDF base parece bulk-only; se generara un wrapper SOL 103.")
    elif is_executable_deck:
        messages.append("El BDF base ya contiene control ejecutivo/case control; se copiara tal cual.")
    else:
        messages.append(
            "El BDF contiene marcas parciales de deck; por prudencia se copiara sin envolver."
        )

    decision = "generated_modal_wrapper" if should_wrap else "copied_executable_deck"
    return DeckCheckResult(
        source_path=path,
        has_sol=has_sol,
        has_cend=has_cend,
        has_begin_bulk=has_begin_bulk,
        has_enddata=has_enddata,
        is_bulk_only=is_bulk_only,
        is_executable_deck=is_executable_deck,
        should_wrap=should_wrap,
        decision=decision,
        messages=messages,
    )
