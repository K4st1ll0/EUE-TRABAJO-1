from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ensure_runtime_directories, load_config_bundle, override_dry_run
from .ia_report import refresh_ia_report
from .mass_audit import run_mass_audit
from .sweep_engine import Phase2PendingError, run_mass_fit, run_single_case, run_sweep


def _collect_run_artifacts(run_payload: dict[str, object]) -> list[str]:
    report_path = run_payload.get("report_path")
    run_id = run_payload.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        return [str(report_path)] if isinstance(report_path, str) else []

    run_dir = Path(str(report_path)).parent if isinstance(report_path, str) else None
    if run_dir is None:
        return []

    artifacts = [
        run_dir / "run_manifest.json",
        run_dir / "metrics.json",
        run_dir / "mass_summary.json",
        run_dir / "report.html",
    ]
    return [str(path) for path in artifacts]


def _refresh_ia_and_print(
    bundle,
    *,
    trigger_command: str,
    affected_run_ids: list[str],
    artifacts_written: list[str],
) -> None:
    ia_path, _ = refresh_ia_report(
        bundle,
        trigger_command=trigger_command,
        status="success",
        affected_run_ids=affected_run_ids,
        artifacts_written=artifacts_written,
    )
    print(f"IA report: {ia_path}")


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config-dir",
        default="configs",
        help="Directory containing project_config.yaml, sensors.yaml, parameters.yaml, and optional sweep_presets.yaml.",
    )
    common.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the actual Nastran execution and generate a trace-only run.",
    )

    parser = argparse.ArgumentParser(
        description="Automate MSC Nastran FEM runs, modal parsing, MAC computation, and reporting.",
        parents=[common],
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("full-run", "modal-fit", "mass-fit"):
        subparsers.add_parser(command, parents=[common])
    sweep_parser = subparsers.add_parser("sweep", parents=[common])
    sweep_parser.add_argument(
        "--preset",
        required=True,
        help="Name of the sweep preset defined in sweep_presets.yaml.",
    )
    mass_audit_parser = subparsers.add_parser("mass-audit", parents=[common])
    mass_audit_parser.add_argument(
        "--run-id",
        required=True,
        help="Existing run identifier or sequence prefix to audit, for example '021' or the full run id.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        bundle = load_config_bundle(args.config_dir)
        if args.dry_run:
            bundle = override_dry_run(bundle, True)
        ensure_runtime_directories(bundle)

        if args.command in {"full-run", "modal-fit"}:
            run_payload = run_single_case(bundle, args.command)
            print(f"Run ID: {run_payload['run_id']}")
            print(f"Status: {run_payload['status']}")
            print(f"Report: {run_payload['report_path']}")
            _refresh_ia_and_print(
                bundle,
                trigger_command=args.command,
                affected_run_ids=[str(run_payload["run_id"])],
                artifacts_written=_collect_run_artifacts(run_payload),
            )
            return 0

        if args.command == "mass-fit":
            run_payloads = run_mass_fit(bundle)
            print(f"Completed {len(run_payloads)} runs for command: mass-fit")
            if run_payloads:
                best = run_payloads[0]
                print(f"Best mass-fit run: {best['run_id']}")
                print(f"Report: {best['report_path']}")
            _refresh_ia_and_print(
                bundle,
                trigger_command=args.command,
                affected_run_ids=[str(payload["run_id"]) for payload in run_payloads],
                artifacts_written=[
                    artifact
                    for payload in run_payloads
                    for artifact in _collect_run_artifacts(payload)
                ],
            )
            return 0

        if args.command == "sweep":
            run_payloads = run_sweep(bundle, args.preset)
            print(f"Completed {len(run_payloads)} runs for preset: {args.preset}")
            _refresh_ia_and_print(
                bundle,
                trigger_command=args.command,
                affected_run_ids=[str(payload["run_id"]) for payload in run_payloads],
                artifacts_written=[
                    artifact
                    for payload in run_payloads
                    for artifact in _collect_run_artifacts(payload)
                ],
            )
            return 0

        if args.command == "mass-audit":
            json_path, md_path, audit_payload = run_mass_audit(bundle, args.run_id)
            run_id = str(audit_payload["run_id"])
            run_dir = bundle.project.paths.runs_dir / run_id
            print(f"Run ID: {audit_payload['run_id']}")
            print(f"Audit JSON: {json_path}")
            print(f"Audit Markdown: {md_path}")
            _refresh_ia_and_print(
                bundle,
                trigger_command=args.command,
                affected_run_ids=[run_id],
                artifacts_written=[
                    str(json_path),
                    str(md_path),
                    str(run_dir / "mass_summary.json"),
                    str(run_dir / "run_manifest.json"),
                    str(run_dir / "metrics.json"),
                    str(run_dir / "report.html"),
                ],
            )
            return 0

        parser.error(f"Unknown command: {args.command}")
        return 2
    except Phase2PendingError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Execution failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
