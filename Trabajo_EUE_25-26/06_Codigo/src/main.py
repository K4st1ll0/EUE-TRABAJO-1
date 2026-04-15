from __future__ import annotations

from pathlib import Path
import argparse
import sys

from .config import ensure_runtime_directories, load_config_bundle, override_dry_run
from .sweep_engine import Phase2PendingError, run_mass_fit, run_single_case, run_sweep


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config-dir",
        default="configs",
        help="Directory containing project_config.yaml, sensors.yaml, and parameters.yaml.",
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
    for command in ("full-run", "modal-fit", "mass-fit", "sweep"):
        subparsers.add_parser(command, parents=[common])
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
            return 0

        if args.command == "mass-fit":
            run_mass_fit()
            return 0

        if args.command == "sweep":
            run_sweep()
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
