from __future__ import annotations

from typing import Iterable

import numpy as np


def relative_error(baseline: float, reference: float) -> float:
    if reference == 0:
        return float("nan")
    return float((baseline - reference) / reference)


def relative_error_percent(baseline: float, reference: float) -> float:
    return float(abs(relative_error(baseline, reference)) * 100.0)


def summarize_relative_errors(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.array(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"count": 0, "mean_abs_percent": None, "max_abs_percent": None}
    return {
        "count": int(finite.size),
        "mean_abs_percent": float(np.mean(np.abs(finite) * 100.0)),
        "max_abs_percent": float(np.max(np.abs(finite) * 100.0)),
    }
