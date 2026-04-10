from __future__ import annotations

from src.correlation.metrics import relative_error, relative_error_percent


def test_relative_error() -> None:
    assert abs(relative_error(110.0, 100.0) - 0.1) < 1e-12
    assert abs(relative_error_percent(110.0, 100.0) - 10.0) < 1e-12
