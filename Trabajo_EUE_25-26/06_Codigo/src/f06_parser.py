from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json
import re

from .bdf_editor import parse_nastran_number


class F06ParseError(ValueError):
    """Raised when the required modal data cannot be extracted from an F06 file."""


MODE_SHAPE_HEADER_RE = re.compile(r"R E A L   E I G E N V E C T O R   N O \.\s+(\d+)")


@dataclass(frozen=True)
class ParsedModalData:
    frequencies_hz: dict[int, float]
    vectors: dict[int, dict[int, tuple[float, float, float]]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_eigenvalues_table(lines: list[str], num_modes: int) -> dict[int, float]:
    frequencies: dict[int, float] = {}
    inside_table = False
    for line in lines:
        if "R E A L   E I G E N V A L U E S" in line:
            inside_table = True
            continue
        if not inside_table:
            continue

        tokens = line.split()
        if len(tokens) >= 5 and tokens[0].isdigit() and tokens[1].isdigit():
            mode = int(tokens[0])
            frequency_hz = parse_nastran_number(tokens[4])
            frequencies[mode] = frequency_hz
            if len(frequencies) >= num_modes:
                break
        elif frequencies and not line.strip():
            break

    if len(frequencies) < num_modes:
        raise F06ParseError(
            f"Expected {num_modes} frequencies but only found {len(frequencies)} in the F06 file."
        )
    return frequencies


def _parse_mode_vectors(
    lines: list[str],
    point_ids: set[int],
    num_modes: int,
) -> dict[int, dict[int, tuple[float, float, float]]]:
    vectors: dict[int, dict[int, tuple[float, float, float]]] = {}
    current_mode: int | None = None

    for line in lines:
        header_match = MODE_SHAPE_HEADER_RE.search(line)
        if header_match:
            current_mode = int(header_match.group(1))
            continue

        if current_mode is None or current_mode > num_modes:
            continue

        tokens = line.split()
        if len(tokens) < 5 or not tokens[0].isdigit():
            continue

        point_id = int(tokens[0])
        if point_id not in point_ids:
            continue

        vectors.setdefault(current_mode, {})[point_id] = (
            parse_nastran_number(tokens[2]),
            parse_nastran_number(tokens[3]),
            parse_nastran_number(tokens[4]),
        )

    missing_modes = [mode for mode in range(1, num_modes + 1) if mode not in vectors]
    if missing_modes:
        raise F06ParseError(f"Missing mode vectors for modes: {missing_modes}")

    missing_points: list[str] = []
    for mode, mode_vectors in vectors.items():
        absent = sorted(point_ids - set(mode_vectors))
        if absent:
            missing_points.append(f"mode {mode}: {absent}")
    if missing_points:
        raise F06ParseError("Missing requested points in F06 vectors: " + "; ".join(missing_points))

    return vectors


def parse_modal_data(f06_path: Path, point_ids: list[int], num_modes: int) -> ParsedModalData:
    lines = f06_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    frequencies = _parse_eigenvalues_table(lines, num_modes)
    vectors = _parse_mode_vectors(lines, set(point_ids), num_modes)
    return ParsedModalData(frequencies_hz=frequencies, vectors=vectors)


def parse_reference_modal_data(
    frequencies_f06: Path,
    vectors_f06: Path,
    reference_point_ids: list[int],
    num_modes: int,
) -> ParsedModalData:
    frequency_lines = frequencies_f06.read_text(encoding="utf-8", errors="ignore").splitlines()
    vector_lines = vectors_f06.read_text(encoding="utf-8", errors="ignore").splitlines()
    return ParsedModalData(
        frequencies_hz=_parse_eigenvalues_table(frequency_lines, num_modes),
        vectors=_parse_mode_vectors(vector_lines, set(reference_point_ids), num_modes),
    )


def write_parsed_results(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
