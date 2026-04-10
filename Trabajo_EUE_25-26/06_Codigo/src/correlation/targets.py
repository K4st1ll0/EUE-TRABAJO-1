from __future__ import annotations

from dataclasses import dataclass

from src.nastran.f06_parser import F06ParseResult


@dataclass(slots=True)
class ModalTarget:
    mode: int
    cycles: float


def build_modal_targets(parse_result: F06ParseResult, limit: int) -> list[ModalTarget]:
    sorted_records = sorted(parse_result.records, key=lambda item: item.mode)
    return [ModalTarget(mode=record.mode, cycles=record.cycles) for record in sorted_records[:limit]]
