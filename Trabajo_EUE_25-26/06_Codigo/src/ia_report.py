from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
import json
import re
import subprocess

from .config import LoadedConfigBundle


IA_SCHEMA_VERSION = 2
IA_REPORT_FILENAME = "IA.json"
IA_OVERVIEW_FILENAME = "IA_overview.json"
IA_DETAILED_FILENAME = "IA_detailed.json"
SUPPORTED_COMMANDS = ["full-run", "modal-fit", "mass-fit", "sweep", "mass-audit"]
MASS_SOURCE_OF_TRUTH_POLICY = (
    "For existing runs, the functional mass source of truth is runs/<run_id>/mass_summary.json. "
    "Reports and functional audits must summarize that file instead of reconstructing a parallel mass state."
)
MODAL_COMMANDS = {"full-run", "modal-fit", "sweep"}


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).astimezone().isoformat()


def _to_posix(value: str | Path) -> str:
    return str(value).replace("\\", "/")


def _project_relative_path(path: str | Path, project_root: Path) -> str:
    absolute = Path(path)
    if not absolute.is_absolute():
        absolute = (project_root / absolute).resolve()
    else:
        absolute = absolute.resolve()
    try:
        return _to_posix(absolute.relative_to(project_root.resolve()))
    except ValueError:
        return _to_posix(absolute)


def _normalize_optional_path(path: Any, project_root: Path) -> str | None:
    if isinstance(path, Path):
        return _project_relative_path(path, project_root)
    if not isinstance(path, str) or not path.strip():
        return None
    return _project_relative_path(path, project_root)


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_mapping(path: Path) -> dict[str, Any] | None:
    payload = _load_json(path)
    return payload if isinstance(payload, dict) else None


def _load_json_sequence(path: Path) -> list[Any] | None:
    payload = _load_json(path)
    return payload if isinstance(payload, list) else None


def _read_readme_summary(readme_path: Path) -> str:
    if not readme_path.exists():
        return ""
    lines = readme_path.read_text(encoding="utf-8").splitlines()
    collected: list[str] = []
    inside_code_block = False
    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            inside_code_block = not inside_code_block
            continue
        if inside_code_block:
            continue
        if stripped == "## Instalacion":
            break
        if stripped:
            collected.append(stripped)
    return "\n".join(collected)


def _build_structure_entries() -> list[dict[str, str]]:
    return [
        {"path": "src", "purpose": "Python automation code for configuration, execution, parsing, audits, and reporting."},
        {"path": "configs", "purpose": "Project, sensors, parameters, and sweep preset configuration files."},
        {"path": "runs", "purpose": "One folder per run with BDF, metrics, manifests, mass summaries, and HTML report."},
        {"path": "reports", "purpose": "Aggregated indexes, standalone audits, and project-level reporting artifacts."},
        {"path": "tests", "purpose": "Pytest coverage for configuration, modal logic, mass logic, audits, and reports."},
    ]


def _build_current_config(bundle: LoadedConfigBundle) -> dict[str, Any]:
    mass_config = bundle.project.mass
    return {
        "config_dir": _to_posix(bundle.config_dir),
        "source_files": {name: _to_posix(path) for name, path in bundle.source_files.items()},
        "paths": {
            "bdf_input": _to_posix(bundle.project.paths.bdf_input),
            "reference_frequencies_f06": _to_posix(bundle.project.paths.reference_frequencies_f06),
            "reference_vectors_f06": _to_posix(bundle.project.paths.reference_vectors_f06),
            "runs_dir": _to_posix(bundle.project.paths.runs_dir),
            "reports_dir": _to_posix(bundle.project.paths.reports_dir),
        },
        "mass_budget_basis": mass_config.budget_basis,
        "mass_target_kg": mass_config.target_kg,
        "functional_families": [
            {
                "label": family.label,
                "match_none": family.match_none,
                "regions": list(family.regions),
                "property_ids": list(family.property_ids),
                "material_ids": list(family.material_ids),
                "element_ids": list(family.element_ids),
                "node_ids": list(family.node_ids),
            }
            for family in mass_config.families
        ],
        "functional_imputation_rules": [
            {
                "family": rule.family_label,
                "mode": rule.mode,
                "target": rule.target_label,
                "weights": dict(rule.weights),
                "z_bands": [
                    {
                        "target": band.target_label,
                        "z_min": band.z_min,
                        "z_max": band.z_max,
                    }
                    for band in rule.z_bands
                ],
            }
            for rule in mass_config.imputation_rules
        ],
    }


def _run_git_command(project_root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(project_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _parse_git_repo_root(project_root: Path) -> Path | None:
    result = _run_git_command(project_root, ["rev-parse", "--show-toplevel"])
    if result.returncode != 0:
        return None
    output = result.stdout.strip()
    return Path(output).resolve() if output else None


def _git_path_to_project_relative(raw_path: str, repo_root: Path, project_root: Path) -> str | None:
    candidate = raw_path.strip()
    if not candidate:
        return None
    if " -> " in candidate:
        candidate = candidate.split(" -> ", 1)[1].strip()
    absolute = (repo_root / candidate).resolve()
    project_resolved = project_root.resolve()
    if not absolute.is_relative_to(project_resolved):
        return None
    relative = absolute.relative_to(project_resolved)
    normalized = _to_posix(relative)
    if normalized in {
        f"reports/{IA_REPORT_FILENAME}",
        f"reports/{IA_OVERVIEW_FILENAME}",
        f"reports/{IA_DETAILED_FILENAME}",
    }:
        return None
    return normalized


def _collect_git_status(project_root: Path) -> dict[str, Any]:
    repo_root = _parse_git_repo_root(project_root)
    if repo_root is None:
        return {
            "available": False,
            "branch": None,
            "last_commit": None,
            "is_dirty": None,
            "modified_files": [],
            "untracked_files": [],
        }

    branch_result = _run_git_command(project_root, ["branch", "--show-current"])
    log_result = _run_git_command(project_root, ["log", "-1", "--format=%H%n%aI%n%s"])
    status_result = _run_git_command(project_root, ["status", "--short"])

    branch = branch_result.stdout.strip() if branch_result.returncode == 0 else None
    last_commit: dict[str, Any] | None = None
    if log_result.returncode == 0:
        lines = [line for line in log_result.stdout.splitlines() if line.strip()]
        if lines:
            last_commit = {
                "hash": lines[0],
                "date": lines[1] if len(lines) > 1 else None,
                "subject": lines[2] if len(lines) > 2 else None,
            }
    modified_files: list[str] = []
    untracked_files: list[str] = []
    if status_result.returncode == 0:
        for line in status_result.stdout.splitlines():
            if len(line) < 4:
                continue
            status_code = line[:2]
            raw_path = line[3:]
            normalized_path = _git_path_to_project_relative(raw_path, repo_root, project_root)
            if normalized_path is None:
                continue
            if status_code == "??":
                untracked_files.append(normalized_path)
            else:
                modified_files.append(normalized_path)
    return {
        "available": True,
        "branch": branch or None,
        "last_commit": last_commit,
        "is_dirty": bool(modified_files or untracked_files),
        "modified_files": sorted(set(modified_files)),
        "untracked_files": sorted(set(untracked_files)),
    }


def _sha256_file(path: Path, cache: dict[str, str | None]) -> str | None:
    key = str(path.resolve())
    if key in cache:
        return cache[key]
    if not path.exists() or not path.is_file():
        cache[key] = None
        return None
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    cache[key] = digest.hexdigest()
    return cache[key]


def _sha256_directory(path: Path, cache: dict[str, str | None]) -> str | None:
    key = f"dir::{path.resolve()}"
    if key in cache:
        return cache[key]
    if not path.exists() or not path.is_dir():
        cache[key] = None
        return None
    digest = sha256()
    for file_path in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(_to_posix(file_path.relative_to(path)).encode("utf-8"))
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    cache[key] = digest.hexdigest()
    return cache[key]


def _normalize_mode_mapping(payload: Any) -> dict[str, float | None]:
    if not isinstance(payload, dict):
        return {}

    def sort_key(raw_key: str) -> tuple[int, str]:
        return (0, f"{int(raw_key):08d}") if str(raw_key).isdigit() else (1, str(raw_key))

    normalized: dict[str, float | None] = {}
    for key in sorted((str(item) for item in payload.keys()), key=sort_key):
        value = payload.get(key)
        normalized[key] = None if value is None else float(value)
    return normalized


def _scan_manifests(runs_dir: Path) -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    for manifest_path in sorted(runs_dir.glob("*/run_manifest.json")):
        payload = _load_json_mapping(manifest_path)
        if payload is None:
            continue
        run_id = payload.get("run_id")
        if isinstance(run_id, str):
            manifests[run_id] = payload
    return manifests


def _build_acceptance_block(manifest: dict[str, Any]) -> dict[str, Any]:
    reject_reasons = manifest.get("reject_reasons")
    acceptance_policy = manifest.get("acceptance_config")
    return {
        "accepted": bool(manifest.get("accepted", False)),
        "reject_reasons": list(reject_reasons) if isinstance(reject_reasons, list) else [],
        "acceptance_policy_snapshot": acceptance_policy if isinstance(acceptance_policy, dict) else {},
    }


def _build_modal_block(manifest: dict[str, Any]) -> dict[str, Any]:
    modal_metrics = manifest.get("modal_metrics")
    reference = manifest.get("reference")
    model = manifest.get("model")
    modal_metrics = modal_metrics if isinstance(modal_metrics, dict) else {}
    reference = reference if isinstance(reference, dict) else {}
    model = model if isinstance(model, dict) else {}
    pairing_entries = modal_metrics.get("pairing")
    pairing_entries = pairing_entries if isinstance(pairing_entries, list) else []

    paired_reference_modes: list[int] = []
    paired_model_modes: list[int] = []
    paired_mac_by_mode: dict[str, float | None] = {}
    paired_relative_frequency_error_by_mode: dict[str, float | None] = {}
    outside_gate_count = 0
    suspicious_pair_count = 0
    for entry in pairing_entries:
        if not isinstance(entry, dict):
            continue
        reference_mode = entry.get("reference_mode")
        model_mode = entry.get("model_mode")
        if isinstance(reference_mode, int):
            paired_reference_modes.append(reference_mode)
            paired_mac_by_mode[str(reference_mode)] = None if entry.get("mac") is None else float(entry["mac"])
            paired_relative_frequency_error_by_mode[str(reference_mode)] = (
                None
                if entry.get("relative_frequency_error") is None
                else float(entry["relative_frequency_error"])
            )
        if isinstance(model_mode, int):
            paired_model_modes.append(model_mode)
        if bool(entry.get("outside_gate", False)):
            outside_gate_count += 1
        if bool(entry.get("suspicious", False)):
            suspicious_pair_count += 1

    return {
        "valid_reference_mode_count": int(manifest.get("valid_reference_mode_count", 0) or 0),
        "valid_model_mode_count": int(manifest.get("valid_model_mode_count", 0) or 0),
        "paired_valid_mode_count": int(manifest.get("paired_valid_mode_count", 0) or 0),
        "frequencies_reference_hz": _normalize_mode_mapping(reference.get("frequencies_hz")),
        "frequencies_model_hz": _normalize_mode_mapping(model.get("frequencies_hz")),
        "paired_reference_modes": paired_reference_modes,
        "paired_model_modes": paired_model_modes,
        "paired_mac_by_mode": paired_mac_by_mode,
        "paired_relative_frequency_error_by_mode": paired_relative_frequency_error_by_mode,
        "worst_mac": None if modal_metrics.get("worst_mac") is None else float(modal_metrics["worst_mac"]),
        "max_relative_frequency_error": None
        if modal_metrics.get("max_relative_frequency_error") is None
        else float(modal_metrics["max_relative_frequency_error"]),
        "bad_pair_count": int(modal_metrics.get("bad_pair_count", 0) or 0),
        "warning_pair_count": int(modal_metrics.get("warning_pair_count", 0) or 0),
        "outside_gate_count": outside_gate_count,
        "suspicious_pair_count": suspicious_pair_count,
    }


def _build_mass_block(mass_payload: dict[str, Any]) -> dict[str, Any]:
    ledgers = mass_payload.get("ledgers")
    ledgers = ledgers if isinstance(ledgers, dict) else {}
    return {
        "legacy_direct_mapping": {
            "rows": list(mass_payload.get("subset_rows", []))
            if isinstance(mass_payload.get("subset_rows"), list)
            else [],
            "warnings": list(mass_payload.get("warnings", []))
            if isinstance(mass_payload.get("warnings"), list)
            else [],
        },
        "functional_budget_imputation": {
            "rows": list(mass_payload.get("budget_imputation_rows", []))
            if isinstance(mass_payload.get("budget_imputation_rows"), list)
            else [],
            "warnings": list(mass_payload.get("budget_imputation_warnings", []))
            if isinstance(mass_payload.get("budget_imputation_warnings"), list)
            else [],
            "notes": list(mass_payload.get("budget_imputation_notes", []))
            if isinstance(mass_payload.get("budget_imputation_notes"), list)
            else [],
            "imputation_residual_mass_kg": mass_payload.get("imputation_residual_mass"),
            "unmapped_mass_kg": ledgers.get("unmapped_mass"),
            "unsupported_extra_mass_kg": ledgers.get("unsupported_extra_mass"),
        },
    }


def _find_result_f06_path(manifest: dict[str, Any], run_dir: Path) -> Path | None:
    nastran_payload = manifest.get("nastran")
    nastran_payload = nastran_payload if isinstance(nastran_payload, dict) else {}
    outputs = nastran_payload.get("outputs")
    outputs = outputs if isinstance(outputs, dict) else {}
    f06_path = outputs.get("f06")
    if isinstance(f06_path, str) and f06_path.strip():
        return Path(f06_path)
    candidates = sorted(run_dir.glob("*.f06"))
    return candidates[0] if candidates else None


def _build_provenance_block(
    manifest: dict[str, Any],
    run_dir: Path,
    git_status: dict[str, Any],
    hash_cache: dict[str, str | None],
) -> dict[str, Any]:
    config_snapshot_dir = manifest.get("config_snapshot_dir")
    snapshot_hash = None
    if isinstance(config_snapshot_dir, str) and config_snapshot_dir.strip():
        snapshot_hash = _sha256_directory(Path(config_snapshot_dir), hash_cache)
    result_f06_path = _find_result_f06_path(manifest, run_dir)
    last_commit = git_status.get("last_commit")
    return {
        "config_snapshot_hash": snapshot_hash,
        "input_bdf_sha256": _sha256_file(Path(manifest["source_bdf"]), hash_cache)
        if isinstance(manifest.get("source_bdf"), str)
        else None,
        "generated_bdf_sha256": _sha256_file(Path(manifest["generated_bdf"]), hash_cache)
        if isinstance(manifest.get("generated_bdf"), str)
        else None,
        "reference_freq_f06_sha256": _sha256_file(Path(manifest["reference_frequencies_f06"]), hash_cache)
        if isinstance(manifest.get("reference_frequencies_f06"), str)
        else None,
        "reference_vec_f06_sha256": _sha256_file(Path(manifest["reference_vectors_f06"]), hash_cache)
        if isinstance(manifest.get("reference_vectors_f06"), str)
        else None,
        "result_f06_sha256": _sha256_file(result_f06_path, hash_cache) if result_f06_path is not None else None,
        "git_commit": None if not isinstance(last_commit, dict) else last_commit.get("hash"),
    }


def _build_detailed_run_entry(
    project_root: Path,
    run_id: str,
    source_entry: dict[str, Any],
    manifest: dict[str, Any] | None,
    mass_summary: dict[str, Any] | None,
    git_status: dict[str, Any],
    hash_cache: dict[str, str | None],
) -> dict[str, Any]:
    manifest = manifest or {}
    mass_payload = mass_summary if isinstance(mass_summary, dict) else manifest.get("mass")
    if not isinstance(mass_payload, dict):
        mass_payload = {}
    run_dir = project_root / "runs" / run_id
    report_path = source_entry.get("report_path") or manifest.get("report_path") or str(run_dir / "report.html")
    return {
        "run_id": run_id,
        "created_at": source_entry.get("created_at") or manifest.get("created_at"),
        "command": source_entry.get("command") or manifest.get("command"),
        "status": source_entry.get("status") or manifest.get("status"),
        "parameter_values": source_entry.get("parameter_values") or manifest.get("parameter_values") or {},
        "selected_ranking_mode": source_entry.get("selected_ranking_mode")
        or (manifest.get("modal_metrics") or {}).get("selected_ranking_mode"),
        "ranking_key": source_entry.get("ranking_key") or manifest.get("ranking_key"),
        "score_robust": source_entry.get("score_robust", manifest.get("score_robust")),
        "score_legacy": source_entry.get("score_legacy", manifest.get("score_legacy")),
        "mean_mac": source_entry.get("mean_mac", (manifest.get("modal_metrics") or {}).get("mean_mac")),
        "mean_relative_frequency_error": source_entry.get(
            "mean_relative_frequency_error",
            (manifest.get("modal_metrics") or {}).get("mean_relative_frequency_error"),
        ),
        "mass_target_basis": mass_payload.get("target_basis", source_entry.get("mass_target_basis")),
        "mass_baseline_status": mass_payload.get("baseline_status", source_entry.get("mass_baseline_status")),
        "paths": {
            "report_path": _normalize_optional_path(report_path, project_root),
            "run_manifest_path": _normalize_optional_path(run_dir / "run_manifest.json", project_root),
            "mass_summary_path": _normalize_optional_path(run_dir / "mass_summary.json", project_root),
            "metrics_path": _normalize_optional_path(run_dir / "metrics.json", project_root),
            "generated_bdf_path": _normalize_optional_path(manifest.get("generated_bdf"), project_root),
            "result_f06_path": _normalize_optional_path(_find_result_f06_path(manifest, run_dir), project_root),
        },
        "acceptance": _build_acceptance_block(manifest),
        "modal": _build_modal_block(manifest),
        "mass": _build_mass_block(mass_payload),
        "provenance": _build_provenance_block(manifest, run_dir, git_status, hash_cache),
    }


def _build_overview_run_entry(detailed_entry: dict[str, Any]) -> dict[str, Any]:
    acceptance = detailed_entry["acceptance"]
    modal = detailed_entry["modal"]
    mass = detailed_entry["mass"]
    return {
        "run_id": detailed_entry["run_id"],
        "created_at": detailed_entry["created_at"],
        "command": detailed_entry["command"],
        "status": detailed_entry["status"],
        "parameter_values": detailed_entry["parameter_values"],
        "selected_ranking_mode": detailed_entry["selected_ranking_mode"],
        "ranking_key": detailed_entry["ranking_key"],
        "score_robust": detailed_entry["score_robust"],
        "score_legacy": detailed_entry["score_legacy"],
        "mean_mac": detailed_entry["mean_mac"],
        "mean_relative_frequency_error": detailed_entry["mean_relative_frequency_error"],
        "mass_target_basis": detailed_entry["mass_target_basis"],
        "mass_baseline_status": detailed_entry["mass_baseline_status"],
        "report_path": detailed_entry["paths"]["report_path"],
        "run_manifest_path": detailed_entry["paths"]["run_manifest_path"],
        "mass_summary_path": detailed_entry["paths"]["mass_summary_path"],
        "acceptance": {
            "accepted": acceptance["accepted"],
            "reject_reasons": acceptance["reject_reasons"],
        },
        "modal": {
            "paired_valid_mode_count": modal["paired_valid_mode_count"],
            "worst_mac": modal["worst_mac"],
            "max_relative_frequency_error": modal["max_relative_frequency_error"],
            "bad_pair_count": modal["bad_pair_count"],
            "warning_pair_count": modal["warning_pair_count"],
            "outside_gate_count": modal["outside_gate_count"],
            "suspicious_pair_count": modal["suspicious_pair_count"],
        },
        "mass": {
            "legacy_direct_mapping": {
                "warnings": mass["legacy_direct_mapping"]["warnings"],
            },
            "functional_budget_imputation": {
                "warnings": mass["functional_budget_imputation"]["warnings"],
                "notes": mass["functional_budget_imputation"]["notes"],
                "imputation_residual_mass_kg": mass["functional_budget_imputation"]["imputation_residual_mass_kg"],
                "unmapped_mass_kg": mass["functional_budget_imputation"]["unmapped_mass_kg"],
                "unsupported_extra_mass_kg": mass["functional_budget_imputation"]["unsupported_extra_mass_kg"],
            },
        },
        "provenance": {
            "git_commit": detailed_entry["provenance"]["git_commit"],
            "generated_bdf_sha256": detailed_entry["provenance"]["generated_bdf_sha256"],
            "result_f06_sha256": detailed_entry["provenance"]["result_f06_sha256"],
        },
    }


def _collect_runs(
    project_root: Path,
    runs_dir: Path,
    runs_index_payload: dict[str, Any],
    git_status: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifests_by_id = _scan_manifests(runs_dir)
    indexed_runs = runs_index_payload.get("runs")
    indexed_runs = indexed_runs if isinstance(indexed_runs, list) else []
    source_entries: list[dict[str, Any]] = [entry for entry in indexed_runs if isinstance(entry, dict)]
    known_run_ids = {
        entry["run_id"]
        for entry in source_entries
        if isinstance(entry.get("run_id"), str)
    }
    for run_id, manifest in manifests_by_id.items():
        if run_id in known_run_ids:
            continue
        source_entries.append(
            {
                "run_id": run_id,
                "created_at": manifest.get("created_at"),
                "command": manifest.get("command"),
                "status": manifest.get("status"),
                "report_path": manifest.get("report_path"),
                "parameter_values": manifest.get("parameter_values"),
            }
        )
    source_entries.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)

    hash_cache: dict[str, str | None] = {}
    detailed_runs: list[dict[str, Any]] = []
    for entry in source_entries:
        run_id = entry.get("run_id")
        if not isinstance(run_id, str):
            continue
        detailed_runs.append(
            _build_detailed_run_entry(
                project_root=project_root,
                run_id=run_id,
                source_entry=entry,
                manifest=manifests_by_id.get(run_id),
                mass_summary=_load_json_mapping(runs_dir / run_id / "mass_summary.json"),
                git_status=git_status,
                hash_cache=hash_cache,
            )
        )
    return [_build_overview_run_entry(entry) for entry in detailed_runs], detailed_runs


def _collect_functional_mass_audits(project_root: Path, reports_dir: Path) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for path in sorted(
        reports_dir.glob("mass_budget_imputation_audit_*.json"),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    ):
        payload = _load_json_mapping(path)
        if payload is None:
            continue
        warnings = payload.get("budget_imputation_warnings", [])
        audits.append(
            {
                "audit_run_label": payload.get("audit_run_label"),
                "run_id": payload.get("run_id"),
                "updated_at": _mtime_iso(path),
                "audit_path": _project_relative_path(path, project_root),
                "generated_bdf": _normalize_optional_path(payload.get("generated_bdf"), project_root),
                "mass_source_of_truth": _normalize_optional_path(payload.get("mass_source_of_truth"), project_root),
                "imputation_residual_mass": payload.get("imputation_residual_mass"),
                "budget_imputation_warnings": warnings if isinstance(warnings, list) else [],
            }
        )
    return audits


def _collect_recent_reports(project_root: Path, runs_dir: Path, limit: int = 10) -> list[dict[str, Any]]:
    report_paths = sorted(
        runs_dir.glob("*/report.html"),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )
    return [
        {
            "run_id": path.parent.name,
            "updated_at": _mtime_iso(path),
            "report_path": _project_relative_path(path, project_root),
        }
        for path in report_paths[:limit]
    ]


def _build_recent_artifacts(
    project_root: Path,
    runs_dir: Path,
    reports_dir: Path,
    runs_index_payload: dict[str, Any],
    functional_mass_audits: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "runs_index_generated_at": runs_index_payload.get("generated_at"),
        "latest_functional_mass_audits": functional_mass_audits[:10],
        "latest_reports": _collect_recent_reports(project_root, runs_dir),
        "reports_index_path": _project_relative_path(reports_dir / "index.html", project_root),
        "runs_index_path": _project_relative_path(reports_dir / "runs_index.json", project_root),
    }


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _normalize_activity_event(
    project_root: Path,
    trigger_command: str,
    status: str,
    affected_run_ids: list[str],
    artifacts_written: list[str],
) -> dict[str, Any]:
    normalized_artifacts = [
        _project_relative_path(path, project_root)
        for path in artifacts_written
        if isinstance(path, str) and path.strip()
    ]
    normalized_artifacts.extend(
        [
            _to_posix(Path("reports") / IA_OVERVIEW_FILENAME),
            _to_posix(Path("reports") / IA_DETAILED_FILENAME),
            _to_posix(Path("reports") / IA_REPORT_FILENAME),
        ]
    )
    return {
        "timestamp": _now_iso(),
        "trigger_command": trigger_command,
        "status": status,
        "affected_run_ids": _dedupe_preserve_order([run_id for run_id in affected_run_ids if run_id]),
        "artifacts_written": _dedupe_preserve_order(normalized_artifacts),
    }


def _load_existing_activity_log(path: Path) -> list[dict[str, Any]]:
    payload = _load_json_mapping(path)
    if payload is None:
        return []
    activity_log = payload.get("activity_log")
    if not isinstance(activity_log, list):
        return []
    return [entry for entry in activity_log if isinstance(entry, dict)]


def _ranking_tuple(entry: dict[str, Any]) -> tuple[Any, ...]:
    ranking_key = entry.get("ranking_key")
    if isinstance(ranking_key, list) and ranking_key:
        return tuple(ranking_key)
    if ranking_key is None:
        return (float("inf"),)
    return (ranking_key,)


def _is_modal_run(entry: dict[str, Any]) -> bool:
    return bool(entry.get("command") in MODAL_COMMANDS or isinstance(entry.get("modal"), dict))


def _best_run_id(runs: list[dict[str, Any]], predicate: Any) -> str | None:
    candidates = [entry for entry in runs if predicate(entry)]
    if not candidates:
        return None
    return min(candidates, key=_ranking_tuple).get("run_id")


def _build_equivalence_groups(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for entry in runs:
        parameter_values = entry.get("parameter_values") or {}
        signature = json.dumps(parameter_values, sort_keys=True)
        group = groups.setdefault(
            signature,
            {
                "parameter_signature": signature,
                "parameter_values": parameter_values,
                "run_ids": [],
                "commands": [],
                "statuses": [],
                "latest_run_id": None,
                "latest_created_at": None,
            },
        )
        group["run_ids"].append(entry.get("run_id"))
        group["commands"].append(entry.get("command"))
        group["statuses"].append(entry.get("status"))
        created_at = entry.get("created_at")
        if group["latest_created_at"] is None or str(created_at) > str(group["latest_created_at"]):
            group["latest_created_at"] = created_at
            group["latest_run_id"] = entry.get("run_id")
    normalized: list[dict[str, Any]] = []
    for group in groups.values():
        normalized.append(
            {
                "parameter_signature": group["parameter_signature"],
                "parameter_values": group["parameter_values"],
                "run_ids": group["run_ids"],
                "run_count": len(group["run_ids"]),
                "commands": _dedupe_preserve_order([item for item in group["commands"] if item]),
                "statuses": _dedupe_preserve_order([item for item in group["statuses"] if item]),
                "latest_run_id": group["latest_run_id"],
            }
        )
    normalized.sort(key=lambda item: (-item["run_count"], item["parameter_signature"]))
    return normalized


def _build_tests_summary(project_root: Path) -> dict[str, Any]:
    tests_dir = project_root / "tests"
    nodeids_path = project_root / ".pytest_cache" / "v" / "cache" / "nodeids"
    lastfailed_path = project_root / ".pytest_cache" / "v" / "cache" / "lastfailed"
    nodeids = _load_json_sequence(nodeids_path) or []
    lastfailed = _load_json_mapping(lastfailed_path) or {}
    test_files = sorted({_to_posix(path.relative_to(project_root)) for path in tests_dir.glob("test_*.py")})
    if nodeids:
        extracted_files = sorted(
            {
                nodeid.split("::", 1)[0]
                for nodeid in nodeids
                if isinstance(nodeid, str) and "::" in nodeid
            }
        )
        if extracted_files:
            test_files = extracted_files
    last_run_at = None
    cache_paths = [path for path in (nodeids_path, lastfailed_path) if path.exists()]
    if cache_paths:
        last_run_at = max(_mtime_iso(path) for path in cache_paths)
    failed_count = len(lastfailed) if last_run_at is not None else None
    passed_count = (len(nodeids) - len(lastfailed)) if nodeids else None
    return {
        "passed": passed_count,
        "failed": failed_count,
        "last_run_at": last_run_at,
        "test_files": test_files,
    }


def _search_text(pattern: str, root: Path) -> bool:
    regex = re.compile(pattern, flags=re.IGNORECASE)
    if not root.exists():
        return False
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".yaml", ".yml", ".json", ".txt", ".md", ".bdf", ".dat"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if regex.search(text):
            return True
    return False


def _build_reduction_state(project_root: Path) -> dict[str, Any]:
    interface_nodes_defined = _search_text(r"interface[_ -]?nodes?", project_root / "configs")
    reduced_model_generated = any(
        re.search(r"(^|[_-])(reduced|craig|cb)([_-]|$)", path.stem, flags=re.IGNORECASE)
        for path in project_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".bdf", ".dat", ".json", ".op2"}
    )
    pch_generated = any(path.is_file() for path in (project_root / "runs").rglob("*.pch"))
    asm_generated = any(path.is_file() for path in project_root.rglob("*.asm"))
    return {
        "cb_ready": bool(interface_nodes_defined and reduced_model_generated and pch_generated and asm_generated),
        "interface_nodes_defined": interface_nodes_defined,
        "reduced_model_generated": reduced_model_generated,
        "pch_generated": pch_generated,
        "asm_generated": asm_generated,
    }


def _build_current_focus_and_next_actions(
    runs: list[dict[str, Any]],
    reduction_state: dict[str, Any],
) -> tuple[str, list[str]]:
    has_accepted_modal = any(_is_modal_run(run) and run["acceptance"]["accepted"] for run in runs)
    residual_runs = [
        run["run_id"]
        for run in runs
        if (run["mass"]["functional_budget_imputation"]["imputation_residual_mass_kg"] or 0.0) > 1.0e-12
    ]
    if not has_accepted_modal:
        focus = "No accepted modal run exists yet; the project focus remains modal-fit convergence with functional mass already traceable."
    elif residual_runs:
        focus = "Some runs still keep non-zero functional mass residuals; mass closure is not homogeneous across the run history."
    elif not reduction_state["cb_ready"]:
        focus = "Mass and modal traces are in place; the next structural milestone is Craig-Bampton preparation."
    else:
        focus = "The project has mass, modal, and reduction traces aligned; the focus can move to integration and downstream use."
    next_actions: list[str] = []
    if not has_accepted_modal:
        next_actions.append("Resume modal-fit sweeps and aim for the first accepted modal run.")
    if residual_runs:
        next_actions.append("Use mass-audit on runs with non-zero functional residuals to align functional mass summaries and reports.")
    if not reduction_state["interface_nodes_defined"]:
        next_actions.append("Define interface nodes explicitly to prepare the Craig-Bampton reduction workflow.")
    if not reduction_state["pch_generated"]:
        next_actions.append("Generate a .pch artifact for the reduction chain once the interface definition is fixed.")
    if not next_actions:
        next_actions.append("Keep IA_overview and IA_detailed as the primary project snapshots for future diagnostics.")
    return focus, next_actions


def _build_program_state(runs: list[dict[str, Any]], runs_index_payload: dict[str, Any], current_basis: str) -> dict[str, Any]:
    latest_run_id = runs[0]["run_id"] if runs else None
    latest_success_run_id = next((run["run_id"] for run in runs if run.get("status") == "success"), None)
    return {
        "run_count": len(runs),
        "latest_run_id": latest_run_id,
        "latest_success_run_id": latest_success_run_id,
        "best_mass_fit_run_id": _best_run_id(runs, lambda item: item.get("command") == "mass-fit"),
        "best_modal_run_id": _best_run_id(runs, _is_modal_run),
        "best_full_run_id": _best_run_id(runs, lambda item: item.get("command") == "full-run"),
        "best_accepted_modal_run_id": _best_run_id(runs, lambda item: _is_modal_run(item) and item["acceptance"]["accepted"]),
        "active_mass_target_basis": runs_index_payload.get("active_mass_target_basis", current_basis),
        "mass_source_of_truth_policy": MASS_SOURCE_OF_TRUTH_POLICY,
    }


def _build_shared_payload(
    bundle: LoadedConfigBundle,
    *,
    trigger_command: str,
    status: str,
    affected_run_ids: list[str],
    artifacts_written: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    project_root = bundle.project_root.resolve()
    runs_dir = bundle.project.paths.runs_dir
    reports_dir = bundle.project.paths.reports_dir
    runs_index_payload = _load_json_mapping(reports_dir / "runs_index.json") or {}
    git_status = _collect_git_status(project_root)
    overview_runs, detailed_runs = _collect_runs(project_root, runs_dir, runs_index_payload, git_status)
    functional_mass_audits = _collect_functional_mass_audits(project_root, reports_dir)
    reduction_state = _build_reduction_state(project_root)
    tests_summary = _build_tests_summary(project_root)
    current_focus, next_actions = _build_current_focus_and_next_actions(detailed_runs, reduction_state)
    activity_log = _load_existing_activity_log(reports_dir / IA_REPORT_FILENAME)
    activity_log.append(
        _normalize_activity_event(
            project_root=project_root,
            trigger_command=trigger_command,
            status=status,
            affected_run_ids=affected_run_ids,
            artifacts_written=artifacts_written,
        )
    )
    shared = {
        "schema_version": IA_SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "project": {
            "name": bundle.project.name,
            "project_root": _to_posix(project_root),
            "reports_dir": _to_posix(reports_dir),
            "runs_dir": _to_posix(runs_dir),
            "readme_summary": _read_readme_summary(project_root / "README.md"),
            "supported_commands": list(SUPPORTED_COMMANDS),
        },
        "program_state": _build_program_state(detailed_runs, runs_index_payload, bundle.project.mass.budget_basis),
        "structure": _build_structure_entries(),
        "current_config": _build_current_config(bundle),
        "git_status": git_status,
        "tests_summary": tests_summary,
        "current_focus": current_focus,
        "next_actions": next_actions,
        "reduction_state": reduction_state,
        "equivalence_groups": _build_equivalence_groups(detailed_runs),
        "recent_artifacts": _build_recent_artifacts(
            project_root=project_root,
            runs_dir=runs_dir,
            reports_dir=reports_dir,
            runs_index_payload=runs_index_payload,
            functional_mass_audits=functional_mass_audits,
        ),
        "functional_mass_audits": functional_mass_audits,
        "activity_log": activity_log,
    }
    return shared, overview_runs, detailed_runs


def build_ia_report_payload(
    bundle: LoadedConfigBundle,
    *,
    trigger_command: str,
    status: str,
    affected_run_ids: list[str],
    artifacts_written: list[str],
) -> dict[str, Any]:
    shared, overview_runs, _ = _build_shared_payload(
        bundle,
        trigger_command=trigger_command,
        status=status,
        affected_run_ids=affected_run_ids,
        artifacts_written=artifacts_written,
    )
    return {**shared, "view": "overview", "runs": overview_runs}


def build_ia_detailed_payload(
    bundle: LoadedConfigBundle,
    *,
    trigger_command: str,
    status: str,
    affected_run_ids: list[str],
    artifacts_written: list[str],
) -> dict[str, Any]:
    shared, _, detailed_runs = _build_shared_payload(
        bundle,
        trigger_command=trigger_command,
        status=status,
        affected_run_ids=affected_run_ids,
        artifacts_written=artifacts_written,
    )
    return {**shared, "view": "detailed", "runs": detailed_runs}


def refresh_ia_report(
    bundle: LoadedConfigBundle,
    *,
    trigger_command: str,
    status: str = "success",
    affected_run_ids: list[str] | None = None,
    artifacts_written: list[str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    affected_run_ids = [] if affected_run_ids is None else list(affected_run_ids)
    artifacts_written = [] if artifacts_written is None else list(artifacts_written)
    overview_payload = build_ia_report_payload(
        bundle,
        trigger_command=trigger_command,
        status=status,
        affected_run_ids=affected_run_ids,
        artifacts_written=artifacts_written,
    )
    detailed_payload = build_ia_detailed_payload(
        bundle,
        trigger_command=trigger_command,
        status=status,
        affected_run_ids=affected_run_ids,
        artifacts_written=artifacts_written,
    )
    reports_dir = bundle.project.paths.reports_dir
    overview_path = reports_dir / IA_OVERVIEW_FILENAME
    detailed_path = reports_dir / IA_DETAILED_FILENAME
    alias_path = reports_dir / IA_REPORT_FILENAME
    overview_text = json.dumps(overview_payload, indent=2)
    overview_path.write_text(overview_text, encoding="utf-8")
    alias_path.write_text(overview_text, encoding="utf-8")
    detailed_path.write_text(json.dumps(detailed_payload, indent=2), encoding="utf-8")
    return alias_path, overview_payload
