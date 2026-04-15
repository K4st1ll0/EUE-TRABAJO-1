from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any
import json
import math


def _build_heatmap_cell(value: float) -> dict[str, str]:
    value = max(0.0, min(1.0, value))
    low = (247, 239, 228)
    high = (34, 113, 91)
    rgb = tuple(int(low[index] + (high[index] - low[index]) * value) for index in range(3))
    foreground = "#f8fafc" if value > 0.55 else "#13232f"
    return {
        "value": f"{value:.4f}",
        "style": f"background: rgb{rgb}; color: {foreground};",
    }


def _build_score_history_svg(history: list[dict[str, Any]]) -> str:
    if not history:
        return ""

    scores = [float(item.get("score", 0.0)) for item in history]
    width = 460
    height = 140
    padding = 20
    inner_width = width - 2 * padding
    inner_height = height - 2 * padding
    minimum = min(scores)
    maximum = max(scores)
    span = maximum - minimum if not math.isclose(maximum, minimum) else 1.0

    points = []
    for index, score in enumerate(scores):
        x = padding if len(scores) == 1 else padding + inner_width * index / (len(scores) - 1)
        normalized = (score - minimum) / span
        y = height - padding - normalized * inner_height
        points.append((x, y, score))

    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y, _ in points)
    circles = "\n".join(
        f"<circle cx='{x:.2f}' cy='{y:.2f}' r='4' fill='#0f766e' />"
        for x, y, _ in points
    )

    return f"""
<svg viewBox="0 0 {width} {height}" role="img" aria-label="Score history">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc" rx="16" />
  <line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#cbd5e1" />
  <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#cbd5e1" />
  <polyline fill="none" stroke="#0f766e" stroke-width="3" points="{polyline}" />
  {circles}
</svg>
""".strip()


def _prepare_frequency_rows(run_payload: dict[str, Any]) -> list[dict[str, Any]]:
    modal_metrics = run_payload.get("modal_metrics")
    if not modal_metrics:
        return []

    pairing = modal_metrics["pairing"]
    diagonal = modal_metrics["diagonal_mac"]
    rows = []
    for entry in pairing:
        reference_mode = int(entry["reference_mode"])
        rows.append(
            {
                "reference_mode": reference_mode,
                "model_mode": entry["model_mode"],
                "reference_frequency_hz": f"{entry['reference_frequency_hz']:.4f}",
                "model_frequency_hz": f"{entry['model_frequency_hz']:.4f}",
                "relative_frequency_error_pct": f"{entry['relative_frequency_error'] * 100:.3f}",
                "paired_mac": f"{entry['mac']:.4f}",
                "diagonal_mac": f"{diagonal[reference_mode - 1]:.4f}",
            }
        )
    return rows


def _prepare_heatmap_rows(run_payload: dict[str, Any]) -> list[dict[str, Any]]:
    modal_metrics = run_payload.get("modal_metrics")
    if not modal_metrics:
        return []

    rows = []
    for index, row in enumerate(modal_metrics["mac_matrix"], start=1):
        rows.append(
            {
                "reference_mode": index,
                "cells": [_build_heatmap_cell(float(value)) for value in row],
            }
        )
    return rows


def _load_template(template_dir: Path, template_name: str) -> Any:
    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except ImportError as exc:
        raise RuntimeError(
            "Jinja2 is required to render the HTML report. Install the dependencies from requirements.txt."
        ) from exc

    environment = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return environment.get_template(template_name)


def generate_run_report(
    run_payload: dict[str, Any],
    history: list[dict[str, Any]],
    template_dir: Path,
    output_path: Path,
) -> None:
    template = _load_template(template_dir, "report.html.j2")
    context = {
        "run": run_payload,
        "frequency_rows": _prepare_frequency_rows(run_payload),
        "heatmap_rows": _prepare_heatmap_rows(run_payload),
        "score_history_svg": _build_score_history_svg(history),
        "history": history,
        "report_generated_at": run_payload["created_at"],
    }
    output_path.write_text(template.render(**context), encoding="utf-8")


def rebuild_reports_index(runs_dir: Path, reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    for manifest_path in sorted(runs_dir.glob("*/run_manifest.json")):
        manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))

    manifests.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    rows = []
    for item in manifests:
        relative_report = Path(item["report_path"]).relative_to(runs_dir.parent).as_posix()
        rows.append(
            "<tr>"
            f"<td>{escape(item['run_id'])}</td>"
            f"<td>{escape(item['command'])}</td>"
            f"<td>{escape(item['status'])}</td>"
            f"<td>{item.get('score_display', 'n/a')}</td>"
            f"<td>{item.get('mean_mac_display', 'n/a')}</td>"
            f"<td><a href='../{escape(relative_report)}'>Abrir reporte</a></td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html lang="es">
  <head>
    <meta charset="utf-8" />
    <title>Run Reports</title>
    <style>
      :root {{
        color-scheme: light;
        --bg: #f3f6f8;
        --card: #ffffff;
        --ink: #12202d;
        --muted: #5e7285;
        --line: #dbe4ea;
        --accent: #0f766e;
      }}
      body {{
        margin: 0;
        padding: 32px;
        background: radial-gradient(circle at top left, #dbeef0, transparent 38%), var(--bg);
        color: var(--ink);
        font: 15px/1.5 "Aptos", "Trebuchet MS", sans-serif;
      }}
      .shell {{
        max-width: 1080px;
        margin: 0 auto;
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 32px;
        box-shadow: 0 24px 64px rgba(15, 23, 42, 0.08);
      }}
      h1 {{
        margin-top: 0;
        font-family: "Georgia", "Times New Roman", serif;
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
      }}
      th, td {{
        padding: 12px 14px;
        border-bottom: 1px solid var(--line);
        text-align: left;
      }}
      th {{
        color: var(--muted);
        font-size: 12px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }}
      a {{
        color: var(--accent);
        text-decoration: none;
      }}
    </style>
  </head>
  <body>
    <main class="shell">
      <h1>Histórico de Corridas</h1>
      <table>
        <thead>
          <tr>
            <th>Run ID</th>
            <th>Comando</th>
            <th>Estado</th>
            <th>Score</th>
            <th>Mean MAC</th>
            <th>Reporte</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows) if rows else '<tr><td colspan="6">Todavía no hay corridas registradas.</td></tr>'}
        </tbody>
      </table>
    </main>
  </body>
</html>
"""
    index_path = reports_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
    return index_path

