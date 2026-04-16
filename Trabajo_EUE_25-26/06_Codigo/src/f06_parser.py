from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json
import re

from .bdf_editor import parse_nastran_number


class F06ParseError(ValueError):
    """Raised when an F06 file cannot be parsed at all."""


MODE_SHAPE_HEADER_RE = re.compile(r"R E A L   E I G E N V E C T O R   N O \.\s+(\d+)")


@dataclass(frozen=True)
class ParsedModalData:
    frequencies_hz: dict[int, float]
    vectors: dict[int, dict[int, tuple[float, float, float]]]
    requested_modes: tuple[int, ...]
    requested_point_ids: tuple[int, ...]
    valid_modes: tuple[int, ...]
    invalid_modes: tuple[int, ...]
    missing_frequency_modes: tuple[int, ...]
    missing_vector_modes: tuple[int, ...]
    missing_points_by_mode: dict[int, tuple[int, ...]]
    valid_mode_count: int
    vector_basis: str
    coordinate_system: str
    source_format: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_f06_lines(path: Path) -> list[str]:
    if not path.exists():
        raise F06ParseError(f"F06 file not found: {path}")
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def read_f06_frequencies(lines: list[str], num_modes: int) -> tuple[dict[int, float], list[str]]:
    frequencies: dict[int, float] = {}
    warnings: list[str] = []
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
            if mode > num_modes:
                continue
            try:
                frequency_hz = parse_nastran_number(tokens[4])
            except ValueError:
                continue
            frequencies[mode] = frequency_hz
        elif frequencies and not line.strip():
            break

    if not frequencies:
        warnings.append("No frequency rows were parsed from the F06 eigenvalue table.")
    missing_modes = [mode for mode in range(1, num_modes + 1) if mode not in frequencies]
    if missing_modes:
        warnings.append(
            "Missing frequencies for modes: " + ", ".join(str(mode) for mode in missing_modes)
        )
    return frequencies, warnings


def read_f06_mode_vectors(
    lines: list[str],
    point_ids: tuple[int, ...],
    num_modes: int,
) -> tuple[dict[int, dict[int, tuple[float, float, float]]], list[str]]:
    requested_points = set(point_ids)
    vectors: dict[int, dict[int, tuple[float, float, float]]] = {}
    warnings: list[str] = []
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

        try:
            point_id = int(tokens[0])
            t1 = parse_nastran_number(tokens[2])
            t2 = parse_nastran_number(tokens[3])
            t3 = parse_nastran_number(tokens[4])
        except ValueError:
            continue

        if point_id not in requested_points:
            continue

        vectors.setdefault(current_mode, {})[point_id] = (t1, t2, t3)

    if not vectors:
        warnings.append("No eigenvector rows were parsed from the F06 file.")

    return vectors, warnings


def validate_modal_data(
    frequencies_hz: dict[int, float],
    vectors: dict[int, dict[int, tuple[float, float, float]]],
    point_ids: tuple[int, ...],
    num_modes: int,
) -> dict[str, Any]:
    requested_modes = tuple(range(1, num_modes + 1))
    missing_frequency_modes: list[int] = []
    missing_vector_modes: list[int] = []
    missing_points_by_mode: dict[int, tuple[int, ...]] = {}
    valid_modes: list[int] = []
    warnings: list[str] = []

    for mode in requested_modes:
        if mode not in frequencies_hz:
            missing_frequency_modes.append(mode)
            continue
        if mode not in vectors:
            missing_vector_modes.append(mode)
            continue
        missing_points = tuple(point_id for point_id in point_ids if point_id not in vectors[mode])
        if missing_points:
            missing_points_by_mode[mode] = missing_points
            continue
        valid_modes.append(mode)

    invalid_modes = tuple(mode for mode in requested_modes if mode not in set(valid_modes))
    if missing_vector_modes:
        warnings.append(
            "Missing eigenvector blocks for modes: "
            + ", ".join(str(mode) for mode in missing_vector_modes)
        )
    if missing_points_by_mode:
        warnings.append(
            "Missing requested sensor points for modes: "
            + "; ".join(
                f"{mode} -> {list(missing_points_by_mode[mode])}"
                for mode in sorted(missing_points_by_mode)
            )
        )

    return {
        "requested_modes": requested_modes,
        "valid_modes": tuple(valid_modes),
        "invalid_modes": invalid_modes,
        "missing_frequency_modes": tuple(missing_frequency_modes),
        "missing_vector_modes": tuple(missing_vector_modes),
        "missing_points_by_mode": missing_points_by_mode,
        "valid_mode_count": len(valid_modes),
        "warnings": tuple(warnings),
    }


def parse_modal_data(f06_path: Path, point_ids: list[int], num_modes: int) -> ParsedModalData:
    point_order = tuple(point_ids)
    lines = _read_f06_lines(f06_path)
    frequencies_hz, frequency_warnings = read_f06_frequencies(lines, num_modes)
    vectors, vector_warnings = read_f06_mode_vectors(lines, point_order, num_modes)
    validation = validate_modal_data(frequencies_hz, vectors, point_order, num_modes)
    return ParsedModalData(
        frequencies_hz=frequencies_hz,
        vectors=vectors,
        requested_modes=validation["requested_modes"],
        requested_point_ids=point_order,
        valid_modes=validation["valid_modes"],
        invalid_modes=validation["invalid_modes"],
        missing_frequency_modes=validation["missing_frequency_modes"],
        missing_vector_modes=validation["missing_vector_modes"],
        missing_points_by_mode=validation["missing_points_by_mode"],
        valid_mode_count=validation["valid_mode_count"],
        vector_basis="T1_T2_T3",
        coordinate_system="global",
        source_format="f06",
        warnings=tuple(frequency_warnings + vector_warnings + list(validation["warnings"])),
    )


def parse_reference_modal_data(
    frequencies_f06: Path,
    vectors_f06: Path,
    reference_point_ids: list[int],
    num_modes: int,
) -> ParsedModalData:
    point_order = tuple(reference_point_ids)
    frequency_lines = _read_f06_lines(frequencies_f06)
    vector_lines = _read_f06_lines(vectors_f06)
    frequencies_hz, frequency_warnings = read_f06_frequencies(frequency_lines, num_modes)
    vectors, vector_warnings = read_f06_mode_vectors(vector_lines, point_order, num_modes)
    validation = validate_modal_data(frequencies_hz, vectors, point_order, num_modes)
    return ParsedModalData(
        frequencies_hz=frequencies_hz,
        vectors=vectors,
        requested_modes=validation["requested_modes"],
        requested_point_ids=point_order,
        valid_modes=validation["valid_modes"],
        invalid_modes=validation["invalid_modes"],
        missing_frequency_modes=validation["missing_frequency_modes"],
        missing_vector_modes=validation["missing_vector_modes"],
        missing_points_by_mode=validation["missing_points_by_mode"],
        valid_mode_count=validation["valid_mode_count"],
        vector_basis="T1_T2_T3",
        coordinate_system="global",
        source_format="f06",
        warnings=tuple(frequency_warnings + vector_warnings + list(validation["warnings"])),
    )


def write_parsed_results(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
