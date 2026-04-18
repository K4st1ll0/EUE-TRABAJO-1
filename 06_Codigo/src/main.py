from __future__ import annotations

import argparse
import json

from .config import config_snapshot, load_project_config
from .reporting import rebuild_project_artifacts
from .workflow import (
    run_baseline,
    run_feasibility_fit,
    run_modal_baseline,
    run_modal_fit,
    run_modal_fit_balanced_local,
    run_modal_fit_local,
    run_modal_fit_targeted,
    run_sweep,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mass-fit workflow for the EUE FEM model.")
    parser.add_argument("--config", default="config/project.yaml", help="Path to the project YAML config.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("baseline", help="Generate a baseline run with all factors = 1.0.")
    subparsers.add_parser("sweep", help="Launch the sweep configured in config/project.yaml.")
    subparsers.add_parser("feasibility-fit", help="Run the feasibility-with-slack mass-fit strategy.")
    subparsers.add_parser("modal-baseline", help="Run phase 2 modal correlation from the frozen mass baseline.")
    subparsers.add_parser("modal-fit", help="Run a small E-only modal fit with densities frozen to the accepted mass baseline.")
    subparsers.add_parser("modal-fit-local", help="Run a small local modal refinement around the best one-at-a-time candidates.")
    subparsers.add_parser("modal-fit-balanced-local", help="Run a balanced local modal refinement around the current balanced candidate.")
    subparsers.add_parser("modal-fit-targeted", help="Run a targeted modal micro-refinement guided by local modal diagnostics around the recommended run.")
    subparsers.add_parser("rebuild-report", help="Rebuild HTML reports and handoff/IA.json from runs/.")
    subparsers.add_parser("show-config", help="Print the resolved project config.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_project_config(args.config)

    if args.command == "baseline":
        run = run_baseline(config)
        print(json.dumps({"run_name": run["run_name"], "status": run["status"], "score": run["score"]}, indent=2))
        return

    if args.command == "sweep":
        result = run_sweep(config)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.command == "feasibility-fit":
        result = run_feasibility_fit(config)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.command == "modal-baseline":
        run = run_modal_baseline(config)
        print(json.dumps({"run_name": run["run_name"], "status": run["status"]}, indent=2, ensure_ascii=False))
        return

    if args.command == "modal-fit":
        result = run_modal_fit(config)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.command == "modal-fit-local":
        result = run_modal_fit_local(config)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.command == "modal-fit-balanced-local":
        result = run_modal_fit_balanced_local(config)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.command == "modal-fit-targeted":
        result = run_modal_fit_targeted(config)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.command == "rebuild-report":
        result = rebuild_project_artifacts(config)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.command == "show-config":
        print(json.dumps(config_snapshot(config), indent=2, ensure_ascii=False))
        return


if __name__ == "__main__":
    main()
