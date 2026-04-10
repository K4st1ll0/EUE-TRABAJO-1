from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.io import read_json, write_text
from src.utils.paths import ensure_directory

from .html_builder import card_grid, data_table, key_value_table, paragraph, render_html_page, section, unordered_list


def _is_actionable_message(message: str) -> bool:
    normalized = message.strip().lower()
    if normalized.startswith("el bdf base parece bulk-only"):
        return False
    if normalized.startswith("el bdf base ya contiene control"):
        return False
    if normalized.startswith("el bdf contiene marcas parciales"):
        return False
    return True


def _timestamp_to_text(value: str | None) -> str:
    if not value:
        return "No disponible"
    return value


def write_bdf_summary_html(summary: dict[str, Any], path: Path) -> Path:
    cards_rows = [{"card_type": key, "count": value} for key, value in summary["card_counts"].items()]
    constraint_rows = [
        {"constraint_type": key, "count": value}
        for key, value in summary["constraint_counts"].items()
    ]

    editable_sections = []
    for family, records in summary["editable_parameters"].items():
        editable_sections.append(section(f"Editable: {family}", data_table(records)))

    body = (
        f"<h1>Resumen del BDF base</h1>"
        + paragraph(f"Generado: {summary['generated_at']}")
        + section(
            "Resumen",
            card_grid(
                [
                    {"title": "Nodos GRID", "value": summary["node_count"], "caption": "Total detectado"},
                    {
                        "title": "Elementos",
                        "value": summary["element_count"],
                        "caption": "Elementos estandar + rigidos",
                    },
                    {
                        "title": "Backend parser",
                        "value": summary["parser_backend"],
                        "caption": "Motor de lectura utilizado",
                    },
                ]
            ),
        )
        + section("Rutas usadas", key_value_table(summary["paths"]))
        + section("Deck check", key_value_table(summary["structure"]))
        + section("Conteos de cards", data_table(cards_rows))
        + section("Restricciones SPC", data_table(constraint_rows))
        + section("Breakdown de elementos", data_table(summary["element_breakdown_rows"]))
        + section("Warnings", unordered_list(summary["warnings"]))
        + "".join(editable_sections)
    )
    return write_text(path, render_html_page("Resumen BDF", body, stylesheet_href="../assets/style.css"))


def _baseline_cards(summary: dict[str, Any]) -> list[dict[str, Any]]:
    search = summary["nastran_search"]
    execution = summary["execution"]
    modal = summary["modal_summary"]
    pending = len(summary.get("pending_items", []))
    status_name = str(summary["status"]).lower()
    status_class = "ok" if status_name == "completed" else "warn"
    if status_name == "failed":
        status_class = "error"
    return [
        {"title": "Run", "value": summary["run_id"], "caption": "Identificador del baseline"},
        {
            "title": "Estado",
            "value": summary["status"],
            "caption": execution.get("status", "sin estado"),
            "status": status_class,
        },
        {
            "title": "Deck strategy",
            "value": summary["deck_strategy"],
            "caption": summary["deck_decision"],
        },
        {
            "title": "Nastran",
            "value": str(search.get("executable") or "No detectado"),
            "caption": search.get("selected_reason") or "sin seleccion",
            "status": "ok" if search.get("executable") else "warn",
        },
        {
            "title": "Modos comparados",
            "value": modal.get("compared_modes", 0),
            "caption": f"Pendientes: {pending}",
        },
    ]


def build_baseline_body(summary: dict[str, Any]) -> str:
    comparison_rows = summary.get("comparison_rows", [])
    search_rows = [{"checked_location": item} for item in summary["nastran_search"]["checked_locations"]]
    candidate_rows = [{"candidate": item} for item in summary["nastran_search"]["candidates"]]
    body = (
        f"<h1>Baseline {summary['run_id']}</h1>"
        + paragraph(f"Generado: {summary['generated_at']}")
        + section("Resumen", card_grid(_baseline_cards(summary)))
        + section("Rutas usadas", key_value_table(summary["paths"]))
        + section("Decision del deck", unordered_list(summary["deck_messages"]))
        + section("Busqueda de Nastran", unordered_list(summary["nastran_search"]["messages"]))
        + section("Ubicaciones revisadas", data_table(search_rows))
        + section("Candidatos detectados", data_table(candidate_rows))
        + section("Ejecucion", key_value_table(summary["execution"]))
        + section("Warnings", unordered_list(summary.get("warnings", [])))
        + section("Errores", unordered_list(summary.get("errors", [])))
        + section("Pendientes", unordered_list(summary.get("pending_items", [])))
        + section("Resumen modal", key_value_table(summary["modal_summary"]))
        + section("Comparacion de frecuencias", data_table(comparison_rows))
    )
    return body


def write_baseline_report(summary: dict[str, Any], path: Path, *, docs_copy: bool) -> Path:
    if docs_copy:
        html = render_html_page(
            f"Baseline {summary['run_id']}",
            build_baseline_body(summary),
            stylesheet_href="../assets/style.css",
        )
    else:
        html = render_html_page(
            f"Baseline {summary['run_id']}",
            build_baseline_body(summary),
            stylesheet_href=None,
        )
    return write_text(path, html)


def write_project_index(
    docs_root: Path,
    inspect_summary_path: Path | None,
    baseline_summaries: list[dict[str, Any]],
    status_label: str,
) -> Path:
    ensure_directory(docs_root)
    latest_inspect = read_json(inspect_summary_path) if inspect_summary_path and inspect_summary_path.exists() else None
    latest_baseline = max(
        baseline_summaries,
        key=lambda item: item.get("generated_at", ""),
        default=None,
    )

    pending: list[str] = []
    if latest_inspect:
        pending.extend(
            message for message in latest_inspect.get("warnings", []) if _is_actionable_message(message)
        )
    if latest_baseline:
        pending.extend(latest_baseline.get("errors", []))
        pending.extend(latest_baseline.get("pending_items", []))

    cards = [
        {
            "title": "Estado del proyecto",
            "value": status_label,
            "caption": "Ultima actualizacion: " + _timestamp_to_text(
                (latest_baseline or latest_inspect or {}).get("generated_at")
            ),
        },
        {
            "title": "Ultimo inspect",
            "value": latest_inspect.get("generated_at", "No ejecutado") if latest_inspect else "No ejecutado",
            "caption": "Resumen BDF disponible" if latest_inspect else "Sin resumen",
        },
        {
            "title": "Ultimo baseline",
            "value": latest_baseline.get("run_id", "No ejecutado") if latest_baseline else "No ejecutado",
            "caption": latest_baseline.get("generated_at", "Sin fecha") if latest_baseline else "Sin run",
        },
        {
            "title": "Errores o pendientes",
            "value": len(pending),
            "caption": "Elementos a revisar",
            "status": "warn" if pending else "ok",
        },
    ]

    links = [
        '<li><a href="reports/bdf_summary.html">Resumen del BDF</a></li>',
    ]
    for summary in sorted(baseline_summaries, key=lambda item: item.get("run_id", "")):
        links.append(
            f'<li><a href="runs/baseline_{summary["run_id"]}.html">Run {summary["run_id"]}</a></li>'
        )

    body = (
        "<h1>Correlacion FEM E-Box</h1>"
        + paragraph(f"Generado: {datetime.now().isoformat(timespec='seconds')}")
        + section("Estado general", card_grid(cards))
        + section(
            "Ultimo inspect",
            key_value_table(
                {
                    "archivo": str(inspect_summary_path) if inspect_summary_path and inspect_summary_path.exists() else "No disponible",
                    "timestamp": latest_inspect.get("generated_at", "No ejecutado") if latest_inspect else "No ejecutado",
                    "backend": latest_inspect.get("parser_backend", "-") if latest_inspect else "-",
                }
            ),
        )
        + section(
            "Ultimo baseline",
            key_value_table(
                {
                    "run_id": latest_baseline.get("run_id", "No ejecutado") if latest_baseline else "No ejecutado",
                    "estado": latest_baseline.get("status", "-") if latest_baseline else "-",
                    "f06_source": latest_baseline.get("f06_source", "-") if latest_baseline else "-",
                }
            ),
        )
        + section("Errores o pendientes", unordered_list(pending))
        + section("Enlaces", "<ul>" + "".join(links) + "</ul>")
    )

    return write_text(docs_root / "index.html", render_html_page("Indice del proyecto", body, stylesheet_href="assets/style.css"))
