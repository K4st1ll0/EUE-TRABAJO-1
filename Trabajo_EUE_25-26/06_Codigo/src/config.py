from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
import json
import shutil

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is invalid."""


def _ensure_mapping(data: Any, context: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ConfigError(f"{context} must be a mapping.")
    return data


def _resolve_project_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _stringify_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _stringify_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stringify_paths(item) for item in value]
    return value


@dataclass(frozen=True)
class PathsConfig:
    bdf_input: Path
    reference_frequencies_f06: Path
    reference_vectors_f06: Path
    runs_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class NastranConfig:
    executable: Path
    arguments: tuple[str, ...]


@dataclass(frozen=True)
class AnalysisConfig:
    num_modes: int
    dry_run: bool


@dataclass(frozen=True)
class PairingConfig:
    frequency_penalty_weight: float


@dataclass(frozen=True)
class ScoringWeights:
    mac: float
    frequency: float
    mass: float


@dataclass(frozen=True)
class ScoringConfig:
    mode: str
    weights: ScoringWeights


@dataclass(frozen=True)
class MassConfig:
    target_kg: float
    budget_workbook: Path | None = None
    budget_sheet: str = "Mass Budget"
    budget_basis: str = "basic"


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    paths: PathsConfig
    nastran: NastranConfig
    analysis: AnalysisConfig
    pairing: PairingConfig
    scoring: ScoringConfig
    mass: MassConfig


@dataclass(frozen=True)
class SensorDefinition:
    name: str
    label: str
    model_node_id: int
    model_coordinates: tuple[float, float, float]
    reference_point_id: int


@dataclass(frozen=True)
class SensorsConfig:
    reference_order_source: str
    sensors: tuple[SensorDefinition, ...]

    @property
    def model_node_order(self) -> tuple[int, ...]:
        return tuple(sensor.model_node_id for sensor in self.sensors)

    @property
    def reference_point_order(self) -> tuple[int, ...]:
        return tuple(sensor.reference_point_id for sensor in self.sensors)


@dataclass(frozen=True)
class ParameterRange:
    min: float
    max: float
    num_steps: int

    def expand(self) -> list[float]:
        if self.num_steps < 2:
            return [self.min]
        if self.max < self.min:
            raise ConfigError("Parameter range max must be greater than or equal to min.")
        step = (self.max - self.min) / (self.num_steps - 1)
        return [self.min + step * index for index in range(self.num_steps)]


@dataclass(frozen=True)
class ParameterDefinition:
    name: str
    group: str
    target: str
    mode: str
    value: float | None = None
    values: tuple[float, ...] = ()
    range: ParameterRange | None = None

    def expand_values(self) -> list[float]:
        if self.mode == "fixed":
            if self.value is None:
                raise ConfigError(f"Parameter '{self.name}' is fixed but has no 'value'.")
            return [self.value]
        if self.mode == "list":
            if not self.values:
                raise ConfigError(f"Parameter '{self.name}' is in list mode but has no 'values'.")
            return list(self.values)
        if self.mode == "range":
            if self.range is None:
                raise ConfigError(f"Parameter '{self.name}' is in range mode but has no 'range'.")
            return self.range.expand()
        raise ConfigError(f"Unsupported parameter mode '{self.mode}' for '{self.name}'.")

    def require_single_value(self) -> float:
        values = self.expand_values()
        if len(values) != 1:
            raise ConfigError(
                f"Parameter '{self.name}' expands to {len(values)} values. "
                "This command only supports a single fixed value right now."
            )
        return values[0]


@dataclass(frozen=True)
class ParametersConfig:
    material_groups: dict[str, tuple[int, ...]]
    parameters: dict[str, ParameterDefinition]


@dataclass(frozen=True)
class LoadedConfigBundle:
    project_root: Path
    config_dir: Path
    source_files: dict[str, Path]
    project: ProjectConfig
    sensors: SensorsConfig
    parameters: ParametersConfig

    def to_dict(self) -> dict[str, Any]:
        return _stringify_paths(
            {
                "project_root": self.project_root,
                "config_dir": self.config_dir,
                "source_files": self.source_files,
                "project": asdict(self.project),
                "sensors": asdict(self.sensors),
                "parameters": {
                    "material_groups": self.parameters.material_groups,
                    "parameters": {
                        name: asdict(definition) for name, definition in self.parameters.parameters.items()
                    },
                },
            }
        )


def _load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    return _ensure_mapping(data, f"Configuration file '{path.name}'")


def _build_project_config(project_root: Path, data: dict[str, Any]) -> ProjectConfig:
    project_meta = _ensure_mapping(data.get("project", {}), "project")
    paths_data = _ensure_mapping(data.get("paths", {}), "paths")
    nastran_data = _ensure_mapping(data.get("nastran", {}), "nastran")
    analysis_data = _ensure_mapping(data.get("analysis", {}), "analysis")
    pairing_data = _ensure_mapping(data.get("pairing", {}), "pairing")
    scoring_data = _ensure_mapping(data.get("scoring", {}), "scoring")
    weights_data = _ensure_mapping(scoring_data.get("weights", {}), "scoring.weights")
    mass_data = _ensure_mapping(data.get("mass", {}), "mass")

    try:
        paths = PathsConfig(
            bdf_input=_resolve_project_path(project_root, str(paths_data["bdf_input"])),
            reference_frequencies_f06=_resolve_project_path(
                project_root, str(paths_data["reference_frequencies_f06"])
            ),
            reference_vectors_f06=_resolve_project_path(
                project_root, str(paths_data["reference_vectors_f06"])
            ),
            runs_dir=_resolve_project_path(project_root, str(paths_data.get("runs_dir", "runs"))),
            reports_dir=_resolve_project_path(project_root, str(paths_data.get("reports_dir", "reports"))),
        )
        nastran = NastranConfig(
            executable=_resolve_project_path(project_root, str(nastran_data["executable"])),
            arguments=tuple(str(item) for item in nastran_data.get("arguments", [])),
        )
        analysis = AnalysisConfig(
            num_modes=int(analysis_data.get("num_modes", 10)),
            dry_run=bool(analysis_data.get("dry_run", False)),
        )
        pairing = PairingConfig(
            frequency_penalty_weight=float(pairing_data.get("frequency_penalty_weight", 0.05))
        )
        scoring = ScoringConfig(
            mode=str(scoring_data.get("mode", "weighted_error")),
            weights=ScoringWeights(
                mac=float(weights_data.get("mac", 1.0)),
                frequency=float(weights_data.get("frequency", 0.25)),
                mass=float(weights_data.get("mass", 0.10)),
            ),
        )
        budget_workbook = mass_data.get("budget_workbook")
        mass = MassConfig(
            target_kg=float(mass_data["target_kg"]),
            budget_workbook=None
            if not budget_workbook
            else _resolve_project_path(project_root, str(budget_workbook)),
            budget_sheet=str(mass_data.get("budget_sheet", "Mass Budget")),
            budget_basis=str(mass_data.get("budget_basis", "basic")).lower(),
        )
    except KeyError as exc:
        raise ConfigError(f"Missing required project configuration key: {exc}") from exc

    project = ProjectConfig(
        name=str(project_meta.get("name", "MSC Nastran FEM Automation")),
        paths=paths,
        nastran=nastran,
        analysis=analysis,
        pairing=pairing,
        scoring=scoring,
        mass=mass,
    )
    _validate_project_config(project)
    return project


def _build_sensors_config(data: dict[str, Any]) -> SensorsConfig:
    sensor_entries = data.get("sensors", [])
    if not isinstance(sensor_entries, list) or not sensor_entries:
        raise ConfigError("sensors.yaml must define a non-empty 'sensors' list.")

    sensors: list[SensorDefinition] = []
    for entry in sensor_entries:
        mapping = _ensure_mapping(entry, "sensors entry")
        coordinates = mapping.get("model_coordinates")
        if not isinstance(coordinates, list) or len(coordinates) != 3:
            raise ConfigError(f"Sensor '{mapping.get('name', '<unknown>')}' needs 3 coordinates.")
        sensors.append(
            SensorDefinition(
                name=str(mapping["name"]),
                label=str(mapping["label"]),
                model_node_id=int(mapping["model_node_id"]),
                model_coordinates=tuple(float(value) for value in coordinates),
                reference_point_id=int(mapping["reference_point_id"]),
            )
        )

    return SensorsConfig(
        reference_order_source=str(data.get("reference_order_source", "manual")),
        sensors=tuple(sensors),
    )


def _build_parameters_config(data: dict[str, Any]) -> ParametersConfig:
    raw_groups = _ensure_mapping(data.get("material_groups", {}), "material_groups")
    raw_parameters = _ensure_mapping(data.get("parameters", {}), "parameters")

    material_groups = {
        group_name: tuple(int(item) for item in values)
        for group_name, values in raw_groups.items()
    }

    parameters: dict[str, ParameterDefinition] = {}
    for name, definition_data in raw_parameters.items():
        mapping = _ensure_mapping(definition_data, f"parameter '{name}'")
        range_data = mapping.get("range")
        parameter_range = None
        if range_data is not None:
            range_mapping = _ensure_mapping(range_data, f"parameter '{name}'.range")
            parameter_range = ParameterRange(
                min=float(range_mapping["min"]),
                max=float(range_mapping["max"]),
                num_steps=int(range_mapping["num_steps"]),
            )
        values = mapping.get("values", [])
        if values is None:
            values = []

        parameter = ParameterDefinition(
            name=name,
            group=str(mapping["group"]),
            target=str(mapping["target"]),
            mode=str(mapping.get("mode", "fixed")),
            value=None if mapping.get("value") is None else float(mapping["value"]),
            values=tuple(float(item) for item in values),
            range=parameter_range,
        )
        if parameter.group not in material_groups:
            raise ConfigError(
                f"Parameter '{name}' references unknown material group '{parameter.group}'."
            )
        if parameter.target not in {"E", "rho"}:
            raise ConfigError(f"Parameter '{name}' target must be 'E' or 'rho'.")
        parameter.expand_values()
        parameters[name] = parameter

    return ParametersConfig(material_groups=material_groups, parameters=parameters)


def _validate_project_config(project: ProjectConfig) -> None:
    required_paths = [
        project.paths.bdf_input,
        project.paths.reference_frequencies_f06,
        project.paths.reference_vectors_f06,
        project.nastran.executable,
    ]
    for path in required_paths:
        if not path.exists():
            raise ConfigError(f"Configured path does not exist: {path}")

    if project.analysis.num_modes <= 0:
        raise ConfigError("analysis.num_modes must be greater than zero.")
    if project.pairing.frequency_penalty_weight < 0:
        raise ConfigError("pairing.frequency_penalty_weight must be non-negative.")
    if project.scoring.mode not in {"weighted_error", "mac_first"}:
        raise ConfigError("scoring.mode must be 'weighted_error' or 'mac_first'.")
    if project.mass.target_kg <= 0:
        raise ConfigError("mass.target_kg must be greater than zero.")
    if project.mass.budget_basis not in {"basic", "nominal"}:
        raise ConfigError("mass.budget_basis must be 'basic' or 'nominal'.")
    if project.mass.budget_workbook is not None and not project.mass.budget_workbook.exists():
        raise ConfigError(f"Configured path does not exist: {project.mass.budget_workbook}")


def load_config_bundle(config_dir: str | Path = "configs") -> LoadedConfigBundle:
    config_dir_path = Path(config_dir)
    if not config_dir_path.is_absolute():
        config_dir_path = (Path.cwd() / config_dir_path).resolve()
    project_root = config_dir_path.parent.resolve()

    source_files = {
        "project_config": config_dir_path / "project_config.yaml",
        "sensors": config_dir_path / "sensors.yaml",
        "parameters": config_dir_path / "parameters.yaml",
    }

    project = _build_project_config(project_root, _load_yaml_file(source_files["project_config"]))
    sensors = _build_sensors_config(_load_yaml_file(source_files["sensors"]))
    parameters = _build_parameters_config(_load_yaml_file(source_files["parameters"]))

    return LoadedConfigBundle(
        project_root=project_root,
        config_dir=config_dir_path,
        source_files=source_files,
        project=project,
        sensors=sensors,
        parameters=parameters,
    )


def ensure_runtime_directories(bundle: LoadedConfigBundle) -> None:
    bundle.project.paths.runs_dir.mkdir(parents=True, exist_ok=True)
    bundle.project.paths.reports_dir.mkdir(parents=True, exist_ok=True)


def override_dry_run(bundle: LoadedConfigBundle, dry_run: bool) -> LoadedConfigBundle:
    if bundle.project.analysis.dry_run == dry_run:
        return bundle
    analysis = replace(bundle.project.analysis, dry_run=dry_run)
    project = replace(bundle.project, analysis=analysis)
    return replace(bundle, project=project)


def write_config_snapshot(bundle: LoadedConfigBundle, destination_dir: Path) -> Path:
    snapshot_dir = destination_dir / "config_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    for source_path in bundle.source_files.values():
        shutil.copy2(source_path, snapshot_dir / source_path.name)

    resolved_snapshot = snapshot_dir / "resolved_snapshot.json"
    resolved_snapshot.write_text(
        json.dumps(bundle.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return snapshot_dir
