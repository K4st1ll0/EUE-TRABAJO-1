from __future__ import annotations

from pathlib import Path


def code_root() -> Path:
    return Path(__file__).resolve().parents[2]


def project_root() -> Path:
    return code_root().parent


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_from(base: Path, value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (base / candidate).resolve()


def safe_relative(target: Path, start: Path) -> str:
    try:
        return target.resolve().relative_to(start.resolve()).as_posix()
    except ValueError:
        return target.resolve().as_posix()
