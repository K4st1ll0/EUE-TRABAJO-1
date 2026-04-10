from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .paths import code_root, resolve_from


@dataclass(slots=True)
class PathsConfig:
    base_bdf: Path
    reference_modes_f06: Path
    reference_accelerometers_f06: Path
    mass_budget_xlsx: Path
    docs_root: Path
    runs_root: Path
    baseline_f06_fallback: Path

    def as_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


@dataclass(slots=True)
class SolverConfig:
    nastran_executable: str
    requested_modes: int
    timeout_seconds: int


@dataclass(slots=True)
class ComparisonConfig:
    compare_first_modes: int


@dataclass(slots=True)
class ReportingConfig:
    project_status_label: str


@dataclass(slots=True)
class ProjectConfig:
    config_path: Path
    code_root: Path
    project_root: Path
    paths: PathsConfig
    solver: SolverConfig
    comparison: ComparisonConfig
    reporting: ReportingConfig
    validation_messages: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_path": str(self.config_path),
            "code_root": str(self.code_root),
            "project_root": str(self.project_root),
            "paths": self.paths.as_dict(),
            "solver": asdict(self.solver),
            "comparison": asdict(self.comparison),
            "reporting": asdict(self.reporting),
            "validation_messages": self.validation_messages,
        }


def _require(data: dict[str, Any], key: str) -> Any:
    if key not in data:
        raise KeyError(f"Missing required config key: {key}")
    return data[key]


def load_project_config(config_path: Path | None = None) -> ProjectConfig:
    config_path = config_path or (code_root() / "config" / "project.yaml")
    config_path = config_path.resolve()
    code_dir = config_path.parents[1]
    project_dir = code_dir.parent

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    raw_paths = _require(data, "paths")
    raw_solver = _require(data, "solver")
    raw_comparison = _require(data, "comparison")
    raw_reporting = _require(data, "reporting")

    paths = PathsConfig(
        base_bdf=resolve_from(project_dir, raw_paths["base_bdf"]),
        reference_modes_f06=resolve_from(project_dir, raw_paths["reference_modes_f06"]),
        reference_accelerometers_f06=resolve_from(
            project_dir, raw_paths["reference_accelerometers_f06"]
        ),
        mass_budget_xlsx=resolve_from(project_dir, raw_paths["mass_budget_xlsx"]),
        docs_root=resolve_from(project_dir, raw_paths["docs_root"]),
        runs_root=resolve_from(project_dir, raw_paths["runs_root"]),
        baseline_f06_fallback=resolve_from(project_dir, raw_paths["baseline_f06_fallback"]),
    )

    solver = SolverConfig(
        nastran_executable=str(raw_solver.get("nastran_executable", "")).strip(),
        requested_modes=int(raw_solver.get("requested_modes", 20)),
        timeout_seconds=int(raw_solver.get("timeout_seconds", 900)),
    )
    comparison = ComparisonConfig(
        compare_first_modes=int(raw_comparison.get("compare_first_modes", 10))
    )
    reporting = ReportingConfig(
        project_status_label=str(raw_reporting.get("project_status_label", "Phase 1"))
    )

    validation_messages: list[str] = []
    for label, path in paths.as_dict().items():
        if label in {"docs_root", "runs_root"}:
            continue
        if not Path(path).exists():
            validation_messages.append(f"Ruta no encontrada: {label} -> {path}")

    return ProjectConfig(
        config_path=config_path,
        code_root=code_dir,
        project_root=project_dir,
        paths=paths,
        solver=solver,
        comparison=comparison,
        reporting=reporting,
        validation_messages=validation_messages,
    )
