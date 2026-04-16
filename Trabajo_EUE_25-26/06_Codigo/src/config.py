from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any
import json
import shutil

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is invalid."""


ACTIVE_MASS_BUDGET_BASIS = "nominal"


def _ensure_mapping(data: Any, context: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ConfigError(f"{context} must be a mapping.")
    return data


def _ensure_list(data: Any, context: str) -> list[Any]:
    if not isinstance(data, list):
        raise ConfigError(f"{context} must be a list.")
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


def _coerce_float_tuple(values: Any, context: str) -> tuple[float, ...]:
    if values is None:
        return ()
    if not isinstance(values, list):
        raise ConfigError(f"{context} must be a list of numbers.")
    return tuple(float(item) for item in values)


def _coerce_int_tuple(values: Any, context: str) -> tuple[int, ...]:
    if values is None:
        return ()
    if not isinstance(values, list):
        raise ConfigError(f"{context} must be a list of integers.")
    return tuple(int(item) for item in values)


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
    freq_gate: float
    alpha_freq: float
    huge_penalty: float
    bad_pair_mac_threshold: float
    warning_pair_mac_threshold: float


@dataclass(frozen=True)
class AcceptanceConfig:
    max_bad_pairs_allowed: int
    reject_if_any_pair_outside_gate: bool
    min_valid_modes: int


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
class RankingRobustWeights:
    mac: float
    bad_pair_fraction: float


@dataclass(frozen=True)
class RankingConfig:
    mode: str
    robust_weights: RankingRobustWeights


@dataclass(frozen=True)
class MassCompensationThresholds:
    total_relative_delta_within: float
    subset_relative_delta_exceeds: float


@dataclass(frozen=True)
class MassSubsetConfig:
    label: str
    target_kg: float
    regions: tuple[str, ...] = ()
    property_ids: tuple[int, ...] = ()
    material_ids: tuple[int, ...] = ()
    element_ids: tuple[int, ...] = ()
    node_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class MassFamilyConfig:
    label: str
    match_none: bool = False
    regions: tuple[str, ...] = ()
    property_ids: tuple[int, ...] = ()
    material_ids: tuple[int, ...] = ()
    element_ids: tuple[int, ...] = ()
    node_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class MassImputationBandConfig:
    target_label: str
    z_min: float | None = None
    z_max: float | None = None


@dataclass(frozen=True)
class MassImputationRuleConfig:
    family_label: str
    mode: str
    target_label: str | None = None
    weights: dict[str, float] = field(default_factory=dict)
    z_bands: tuple[MassImputationBandConfig, ...] = ()


@dataclass(frozen=True)
class MassConfig:
    target_kg: float
    budget_workbook: Path | None = None
    budget_sheet: str = "Mass Budget"
    budget_basis: str = ACTIVE_MASS_BUDGET_BASIS
    subsets: tuple[MassSubsetConfig, ...] = ()
    families: tuple[MassFamilyConfig, ...] = ()
    imputation_rules: tuple[MassImputationRuleConfig, ...] = ()
    compensation_warning_thresholds: MassCompensationThresholds = field(
        default_factory=lambda: MassCompensationThresholds(
            total_relative_delta_within=0.01,
            subset_relative_delta_exceeds=0.05,
        )
    )


@dataclass(frozen=True)
class ModeFamilyLabelsConfig:
    reference: dict[int, str]
    model: dict[int, str]


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    paths: PathsConfig
    nastran: NastranConfig
    analysis: AnalysisConfig
    pairing: PairingConfig
    acceptance: AcceptanceConfig
    scoring: ScoringConfig
    ranking: RankingConfig
    mass: MassConfig
    mode_family_labels: ModeFamilyLabelsConfig


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
class SweepParameterSpec:
    mode: str
    value: float | None = None
    values: tuple[float, ...] = ()
    half_window: float | None = None
    step: float | None = None

    def expand(self, *, center_value: float | None = None) -> list[float]:
        if self.mode == "fixed":
            if self.value is None:
                raise ConfigError("A sweep parameter in fixed mode requires 'value'.")
            return [self.value]

        if self.mode == "list":
            if not self.values:
                raise ConfigError("A sweep parameter in list mode requires 'values'.")
            return list(self.values)

        if self.mode == "centered_range":
            if center_value is None:
                raise ConfigError("A centered_range sweep parameter requires a center value.")
            if self.half_window is None or self.step is None:
                raise ConfigError("A centered_range sweep parameter requires 'half_window' and 'step'.")
            if self.step <= 0:
                raise ConfigError("A centered_range sweep parameter requires a positive step.")
            span = 2.0 * self.half_window
            steps = int(round(span / self.step))
            values = [
                round(center_value - self.half_window + (self.step * index), 10)
                for index in range(steps + 1)
            ]
            return values

        raise ConfigError(f"Unsupported sweep parameter mode '{self.mode}'.")


@dataclass(frozen=True)
class SweepPresetConfig:
    name: str
    description: str
    select_from_presets: tuple[str, ...]
    fixed_parameters: dict[str, float]
    fixed_parameters_from_best_run: tuple[str, ...]
    varying_parameters: dict[str, SweepParameterSpec]


@dataclass(frozen=True)
class SweepPresetsConfig:
    presets: dict[str, SweepPresetConfig]


@dataclass(frozen=True)
class LoadedConfigBundle:
    project_root: Path
    config_dir: Path
    source_files: dict[str, Path]
    project: ProjectConfig
    sensors: SensorsConfig
    parameters: ParametersConfig
    sweep_presets: SweepPresetsConfig

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
                "sweep_presets": {
                    name: asdict(preset) for name, preset in self.sweep_presets.presets.items()
                },
            }
        )


def mass_budget_baseline_status(budget_basis: str | None) -> str:
    return "active" if (budget_basis or "").lower() == ACTIVE_MASS_BUDGET_BASIS else "historical"


def _load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    return _ensure_mapping(data, f"Configuration file '{path.name}'")


def _build_mass_subset_config(entries: Any) -> tuple[MassSubsetConfig, ...]:
    if entries is None:
        return ()

    subsets: list[MassSubsetConfig] = []
    seen_labels: set[str] = set()
    for entry in _ensure_list(entries, "mass.subsets"):
        mapping = _ensure_mapping(entry, "mass.subsets entry")
        label = str(mapping["label"])
        if label in seen_labels:
            raise ConfigError(f"Duplicate mass subset label '{label}'.")
        seen_labels.add(label)
        subsets.append(
            MassSubsetConfig(
                label=label,
                target_kg=float(mapping["target_kg"]),
                regions=tuple(str(item) for item in mapping.get("regions", [])),
                property_ids=_coerce_int_tuple(mapping.get("property_ids"), f"{label}.property_ids"),
                material_ids=_coerce_int_tuple(mapping.get("material_ids"), f"{label}.material_ids"),
                element_ids=_coerce_int_tuple(mapping.get("element_ids"), f"{label}.element_ids"),
                node_ids=_coerce_int_tuple(mapping.get("node_ids"), f"{label}.node_ids"),
            )
        )
    return tuple(subsets)


def _build_mass_family_config(entries: Any) -> tuple[MassFamilyConfig, ...]:
    if entries is None:
        return ()

    families: list[MassFamilyConfig] = []
    seen_labels: set[str] = set()
    for entry in _ensure_list(entries, "mass.families"):
        mapping = _ensure_mapping(entry, "mass.families entry")
        label = str(mapping["label"])
        if label in seen_labels:
            raise ConfigError(f"Duplicate mass family label '{label}'.")
        seen_labels.add(label)
        families.append(
            MassFamilyConfig(
                label=label,
                match_none=bool(mapping.get("match_none", False)),
                regions=tuple(str(item) for item in mapping.get("regions", [])),
                property_ids=_coerce_int_tuple(mapping.get("property_ids"), f"{label}.property_ids"),
                material_ids=_coerce_int_tuple(mapping.get("material_ids"), f"{label}.material_ids"),
                element_ids=_coerce_int_tuple(mapping.get("element_ids"), f"{label}.element_ids"),
                node_ids=_coerce_int_tuple(mapping.get("node_ids"), f"{label}.node_ids"),
            )
        )
    return tuple(families)


def _build_mass_imputation_rules(entries: Any) -> tuple[MassImputationRuleConfig, ...]:
    if entries is None:
        return ()

    rules: list[MassImputationRuleConfig] = []
    for entry in _ensure_list(entries, "mass.imputation_rules"):
        mapping = _ensure_mapping(entry, "mass.imputation_rules entry")
        family_label = str(mapping["family"])
        weights = {
            str(key): float(value)
            for key, value in _ensure_mapping(
                mapping.get("weights", {}),
                f"mass.imputation_rules[{family_label}].weights",
            ).items()
        }
        z_bands: list[MassImputationBandConfig] = []
        for band_entry in _ensure_list(
            mapping.get("z_bands", []),
            f"mass.imputation_rules[{family_label}].z_bands",
        ):
            band_mapping = _ensure_mapping(
                band_entry,
                f"mass.imputation_rules[{family_label}].z_bands entry",
            )
            z_bands.append(
                MassImputationBandConfig(
                    target_label=str(band_mapping["target"]),
                    z_min=None if band_mapping.get("z_min") is None else float(band_mapping["z_min"]),
                    z_max=None if band_mapping.get("z_max") is None else float(band_mapping["z_max"]),
                )
            )
        rules.append(
            MassImputationRuleConfig(
                family_label=family_label,
                mode=str(mapping.get("mode", "direct")).lower(),
                target_label=None if mapping.get("target") is None else str(mapping["target"]),
                weights=weights,
                z_bands=tuple(z_bands),
            )
        )
    return tuple(rules)


def _build_mode_family_labels(data: Any) -> ModeFamilyLabelsConfig:
    mapping = _ensure_mapping(data or {}, "mode_family_labels")
    reference = {
        int(key): str(value)
        for key, value in _ensure_mapping(mapping.get("reference", {}), "mode_family_labels.reference").items()
    }
    model = {
        int(key): str(value)
        for key, value in _ensure_mapping(mapping.get("model", {}), "mode_family_labels.model").items()
    }
    return ModeFamilyLabelsConfig(reference=reference, model=model)


def _build_project_config(project_root: Path, data: dict[str, Any]) -> ProjectConfig:
    project_meta = _ensure_mapping(data.get("project", {}), "project")
    paths_data = _ensure_mapping(data.get("paths", {}), "paths")
    nastran_data = _ensure_mapping(data.get("nastran", {}), "nastran")
    analysis_data = _ensure_mapping(data.get("analysis", {}), "analysis")
    pairing_data = _ensure_mapping(data.get("pairing", {}), "pairing")
    acceptance_data = _ensure_mapping(data.get("acceptance", {}), "acceptance")
    scoring_data = _ensure_mapping(data.get("scoring", {}), "scoring")
    scoring_weights_data = _ensure_mapping(scoring_data.get("weights", {}), "scoring.weights")
    ranking_data = _ensure_mapping(data.get("ranking", {}), "ranking")
    robust_weights_data = _ensure_mapping(
        ranking_data.get("robust_weights", {}),
        "ranking.robust_weights",
    )
    mass_data = _ensure_mapping(data.get("mass", {}), "mass")
    compensation_data = _ensure_mapping(
        mass_data.get("compensation_warning_thresholds", {}),
        "mass.compensation_warning_thresholds",
    )

    try:
        paths = PathsConfig(
            bdf_input=_resolve_project_path(project_root, str(paths_data["bdf_input"])),
            reference_frequencies_f06=_resolve_project_path(
                project_root,
                str(paths_data["reference_frequencies_f06"]),
            ),
            reference_vectors_f06=_resolve_project_path(
                project_root,
                str(paths_data["reference_vectors_f06"]),
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
            freq_gate=float(pairing_data.get("freq_gate", 0.20)),
            alpha_freq=float(
                pairing_data.get("alpha_freq", pairing_data.get("frequency_penalty_weight", 0.15))
            ),
            huge_penalty=float(pairing_data.get("huge_penalty", 1_000_000.0)),
            bad_pair_mac_threshold=float(pairing_data.get("bad_pair_mac_threshold", 0.10)),
            warning_pair_mac_threshold=float(pairing_data.get("warning_pair_mac_threshold", 0.30)),
        )
        acceptance = AcceptanceConfig(
            max_bad_pairs_allowed=int(acceptance_data.get("max_bad_pairs_allowed", 1)),
            reject_if_any_pair_outside_gate=bool(
                acceptance_data.get("reject_if_any_pair_outside_gate", True)
            ),
            min_valid_modes=int(acceptance_data.get("min_valid_modes", analysis.num_modes)),
        )
        scoring = ScoringConfig(
            mode=str(scoring_data.get("mode", "weighted_error")),
            weights=ScoringWeights(
                mac=float(scoring_weights_data.get("mac", 1.0)),
                frequency=float(scoring_weights_data.get("frequency", 0.25)),
                mass=float(scoring_weights_data.get("mass", 0.10)),
            ),
        )
        ranking = RankingConfig(
            mode=str(ranking_data.get("mode", "hierarchical")).lower(),
            robust_weights=RankingRobustWeights(
                mac=float(robust_weights_data.get("mac", 0.05)),
                bad_pair_fraction=float(robust_weights_data.get("bad_pair_fraction", 0.20)),
            ),
        )
        budget_workbook = mass_data.get("budget_workbook")
        mass = MassConfig(
            target_kg=float(mass_data["target_kg"]),
            budget_workbook=None
            if not budget_workbook
            else _resolve_project_path(project_root, str(budget_workbook)),
            budget_sheet=str(mass_data.get("budget_sheet", "Mass Budget")),
            budget_basis=str(mass_data.get("budget_basis", ACTIVE_MASS_BUDGET_BASIS)).lower(),
            subsets=_build_mass_subset_config(mass_data.get("subsets")),
            families=_build_mass_family_config(mass_data.get("families")),
            imputation_rules=_build_mass_imputation_rules(mass_data.get("imputation_rules")),
            compensation_warning_thresholds=MassCompensationThresholds(
                total_relative_delta_within=float(
                    compensation_data.get("total_relative_delta_within", 0.01)
                ),
                subset_relative_delta_exceeds=float(
                    compensation_data.get("subset_relative_delta_exceeds", 0.05)
                ),
            ),
        )
        mode_family_labels = _build_mode_family_labels(data.get("mode_family_labels"))
    except KeyError as exc:
        raise ConfigError(f"Missing required project configuration key: {exc}") from exc

    project = ProjectConfig(
        name=str(project_meta.get("name", "MSC Nastran FEM Automation")),
        paths=paths,
        nastran=nastran,
        analysis=analysis,
        pairing=pairing,
        acceptance=acceptance,
        scoring=scoring,
        ranking=ranking,
        mass=mass,
        mode_family_labels=mode_family_labels,
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


def _build_sweep_presets_config(data: dict[str, Any]) -> SweepPresetsConfig:
    raw_presets = _ensure_mapping(data.get("presets", {}), "presets")
    presets: dict[str, SweepPresetConfig] = {}
    for preset_name, preset_data in raw_presets.items():
        mapping = _ensure_mapping(preset_data, f"sweep preset '{preset_name}'")
        fixed_parameters = {
            str(name): float(value)
            for name, value in _ensure_mapping(
                mapping.get("fixed_parameters", {}),
                f"sweep preset '{preset_name}'.fixed_parameters",
            ).items()
        }
        varying_parameters: dict[str, SweepParameterSpec] = {}
        for parameter_name, parameter_data in _ensure_mapping(
            mapping.get("varying_parameters", {}),
            f"sweep preset '{preset_name}'.varying_parameters",
        ).items():
            parameter_mapping = _ensure_mapping(
                parameter_data,
                f"sweep preset '{preset_name}'.varying_parameters.{parameter_name}",
            )
            varying_parameters[str(parameter_name)] = SweepParameterSpec(
                mode=str(parameter_mapping.get("mode", "list")),
                value=None
                if parameter_mapping.get("value") is None
                else float(parameter_mapping["value"]),
                values=_coerce_float_tuple(
                    parameter_mapping.get("values"),
                    f"sweep preset '{preset_name}'.varying_parameters.{parameter_name}.values",
                ),
                half_window=None
                if parameter_mapping.get("half_window") is None
                else float(parameter_mapping["half_window"]),
                step=None
                if parameter_mapping.get("step") is None
                else float(parameter_mapping["step"]),
            )
        presets[preset_name] = SweepPresetConfig(
            name=preset_name,
            description=str(mapping.get("description", preset_name)),
            select_from_presets=tuple(str(item) for item in mapping.get("select_from_presets", [])),
            fixed_parameters=fixed_parameters,
            fixed_parameters_from_best_run=tuple(
                str(item) for item in mapping.get("fixed_parameters_from_best_run", [])
            ),
            varying_parameters=varying_parameters,
        )
    return SweepPresetsConfig(presets=presets)


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
    if project.pairing.freq_gate < 0:
        raise ConfigError("pairing.freq_gate must be non-negative.")
    if project.pairing.alpha_freq < 0:
        raise ConfigError("pairing.alpha_freq must be non-negative.")
    if project.pairing.huge_penalty <= 0:
        raise ConfigError("pairing.huge_penalty must be greater than zero.")
    if not 0.0 <= project.pairing.bad_pair_mac_threshold <= 1.0:
        raise ConfigError("pairing.bad_pair_mac_threshold must be within [0, 1].")
    if not 0.0 <= project.pairing.warning_pair_mac_threshold <= 1.0:
        raise ConfigError("pairing.warning_pair_mac_threshold must be within [0, 1].")
    if project.pairing.warning_pair_mac_threshold < project.pairing.bad_pair_mac_threshold:
        raise ConfigError(
            "pairing.warning_pair_mac_threshold must be greater than or equal to "
            "pairing.bad_pair_mac_threshold."
        )
    if project.acceptance.max_bad_pairs_allowed < 0:
        raise ConfigError("acceptance.max_bad_pairs_allowed must be non-negative.")
    if project.acceptance.min_valid_modes <= 0:
        raise ConfigError("acceptance.min_valid_modes must be greater than zero.")
    if project.scoring.mode not in {"weighted_error", "mac_first"}:
        raise ConfigError("scoring.mode must be 'weighted_error' or 'mac_first'.")
    if project.ranking.mode not in {"hierarchical", "robust_scalar"}:
        raise ConfigError("ranking.mode must be 'hierarchical' or 'robust_scalar'.")
    if project.mass.target_kg <= 0:
        raise ConfigError("mass.target_kg must be greater than zero.")
    if project.mass.budget_basis not in {"basic", "nominal"}:
        raise ConfigError("mass.budget_basis must be 'basic' or 'nominal'.")
    if project.mass.budget_workbook is not None and not project.mass.budget_workbook.exists():
        raise ConfigError(f"Configured path does not exist: {project.mass.budget_workbook}")
    if project.mass.compensation_warning_thresholds.total_relative_delta_within < 0:
        raise ConfigError("mass.compensation_warning_thresholds.total_relative_delta_within must be non-negative.")
    if project.mass.compensation_warning_thresholds.subset_relative_delta_exceeds < 0:
        raise ConfigError("mass.compensation_warning_thresholds.subset_relative_delta_exceeds must be non-negative.")
    subset_labels = {subset.label for subset in project.mass.subsets}
    for subset in project.mass.subsets:
        if subset.target_kg < 0:
            raise ConfigError(f"mass.subsets '{subset.label}' target_kg must be non-negative.")
    family_labels = {family.label for family in project.mass.families}
    for rule in project.mass.imputation_rules:
        if rule.family_label not in family_labels:
            raise ConfigError(
                f"mass.imputation_rules references unknown family '{rule.family_label}'."
            )
        if rule.mode not in {"direct", "proportional", "z_band_split", "unassigned"}:
            raise ConfigError(
                f"mass.imputation_rules for family '{rule.family_label}' uses unsupported mode '{rule.mode}'."
            )
        if rule.mode == "direct":
            if not rule.target_label or rule.target_label not in subset_labels:
                raise ConfigError(
                    f"mass.imputation_rules for family '{rule.family_label}' needs a valid target."
                )
        if rule.mode == "proportional":
            if not rule.weights:
                raise ConfigError(
                    f"mass.imputation_rules for family '{rule.family_label}' needs non-empty weights."
                )
            if any(weight < 0 for weight in rule.weights.values()):
                raise ConfigError(
                    f"mass.imputation_rules for family '{rule.family_label}' cannot use negative weights."
                )
            if sum(rule.weights.values()) <= 0:
                raise ConfigError(
                    f"mass.imputation_rules for family '{rule.family_label}' needs positive total weight."
                )
            unknown_targets = sorted(set(rule.weights) - subset_labels)
            if unknown_targets:
                raise ConfigError(
                    f"mass.imputation_rules for family '{rule.family_label}' references unknown targets {unknown_targets}."
                )
        if rule.mode == "z_band_split":
            if not rule.z_bands:
                raise ConfigError(
                    f"mass.imputation_rules for family '{rule.family_label}' needs at least one z band."
                )
            for band in rule.z_bands:
                if band.target_label not in subset_labels:
                    raise ConfigError(
                        f"mass.imputation_rules for family '{rule.family_label}' references unknown target '{band.target_label}'."
                    )
                if band.z_min is not None and band.z_max is not None and band.z_max <= band.z_min:
                    raise ConfigError(
                        f"mass.imputation_rules for family '{rule.family_label}' has an invalid z band."
                    )


def _validate_sweep_presets(
    presets: SweepPresetsConfig,
    parameters: ParametersConfig,
    project: ProjectConfig,
) -> None:
    known_parameter_names = set(parameters.parameters)
    for preset_name, preset in presets.presets.items():
        for parameter_name in preset.fixed_parameters:
            if parameter_name not in known_parameter_names:
                raise ConfigError(
                    f"Sweep preset '{preset_name}' references unknown parameter '{parameter_name}'."
                )
        for parameter_name in preset.fixed_parameters_from_best_run:
            if parameter_name not in known_parameter_names:
                raise ConfigError(
                    f"Sweep preset '{preset_name}' references unknown parameter '{parameter_name}'."
                )
        for parameter_name, spec in preset.varying_parameters.items():
            if parameter_name not in known_parameter_names:
                raise ConfigError(
                    f"Sweep preset '{preset_name}' references unknown parameter '{parameter_name}'."
                )
            spec.expand(center_value=0.0 if spec.mode == "centered_range" else None)
        for source_preset in preset.select_from_presets:
            if source_preset not in presets.presets:
                raise ConfigError(
                    f"Sweep preset '{preset_name}' references unknown source preset '{source_preset}'."
                )
        if not preset.varying_parameters:
            raise ConfigError(f"Sweep preset '{preset_name}' must define at least one varying parameter.")
        if project.ranking.mode not in {"hierarchical", "robust_scalar"}:
            raise ConfigError("Unsupported ranking mode configured for sweep selection.")


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
    optional_sweep_file = config_dir_path / "sweep_presets.yaml"
    if optional_sweep_file.exists():
        source_files["sweep_presets"] = optional_sweep_file

    project = _build_project_config(project_root, _load_yaml_file(source_files["project_config"]))
    sensors = _build_sensors_config(_load_yaml_file(source_files["sensors"]))
    parameters = _build_parameters_config(_load_yaml_file(source_files["parameters"]))
    sweep_presets = _build_sweep_presets_config(
        _load_yaml_file(optional_sweep_file) if optional_sweep_file.exists() else {}
    )
    _validate_sweep_presets(sweep_presets, parameters, project)

    return LoadedConfigBundle(
        project_root=project_root,
        config_dir=config_dir_path,
        source_files=source_files,
        project=project,
        sensors=sensors,
        parameters=parameters,
        sweep_presets=sweep_presets,
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
