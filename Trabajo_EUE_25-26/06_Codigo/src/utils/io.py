from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

from .paths import ensure_directory


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"No JSON serializer for {type(value)!r}")


def write_json(path: Path, data: Any) -> Path:
    ensure_directory(path.parent)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=True, default=_json_default),
        encoding="utf-8",
    )
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, content: str) -> Path:
    ensure_directory(path.parent)
    path.write_text(content, encoding="utf-8")
    return path


def write_csv_rows(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    ensure_directory(path.parent)
    rows_list = list(rows)
    if pd is not None:
        pd.DataFrame(rows_list).to_csv(path, index=False)
        return path

    if not rows_list:
        path.write_text("", encoding="utf-8")
        return path

    fieldnames = list(rows_list[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_list)
    return path
