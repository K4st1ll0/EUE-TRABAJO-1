from __future__ import annotations

from typing import Any

from src.nastran.f06_parser import F06ParseResult

from .metrics import relative_error, relative_error_percent, summarize_relative_errors


def compare_modal_frequencies(
    baseline: F06ParseResult, reference: F06ParseResult, limit: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline_records = sorted(baseline.records, key=lambda item: item.mode)
    reference_records = sorted(reference.records, key=lambda item: item.mode)
    compared = min(limit, len(baseline_records), len(reference_records))

    rows: list[dict[str, Any]] = []
    for index in range(compared):
        baseline_row = baseline_records[index]
        reference_row = reference_records[index]
        rel = relative_error(baseline_row.cycles, reference_row.cycles)
        rows.append(
            {
                "mode": reference_row.mode,
                "baseline_mode": baseline_row.mode,
                "reference_cycles": reference_row.cycles,
                "baseline_cycles": baseline_row.cycles,
                "reference_eigenvalue": reference_row.eigenvalue,
                "baseline_eigenvalue": baseline_row.eigenvalue,
                "relative_error": rel,
                "relative_error_percent": relative_error_percent(
                    baseline_row.cycles, reference_row.cycles
                ),
            }
        )

    summary = summarize_relative_errors(row["relative_error"] for row in rows)
    summary["compared_modes"] = compared
    return rows, summary
